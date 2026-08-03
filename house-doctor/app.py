"""D89/Idea-E: House Doctor — continuous differential diagnosis for Kai's system health.

Receives observation lists from the proactive observer, classifies symptoms,
applies a differential diagnosis rule table, and files medical_report memories
to memu-core.  Calls notify-service for WARNING/CRITICAL cases.

POST /diagnose   — classify + diagnose a set of observations
GET  /rules      — list the active diagnosis rule set
GET  /health     — standard health check
"""
from __future__ import annotations

import logging
import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional

import httpx

from common.http_hygiene import pooled_client
from fastapi import FastAPI, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("house-doctor")

app = FastAPI(title="House Doctor", version="0.1.0")

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
NOTIFY_URL = os.getenv("NOTIFY_URL", "http://notify-service:8031")

# In-memory ring buffer: last 20 diagnoses available without a memu-core round-trip.
# Agentic reads this at the start of each proactive cycle to carry forward recent health history.
_recent_diagnoses: Deque[Dict[str, Any]] = deque(maxlen=20)


class DiagnosisRequest(BaseModel):
    observations: List[str]
    world_state: Optional[Dict[str, Any]] = None


class DiagnosisRule:
    def __init__(
        self,
        rule_id: str,
        pattern: List[str],
        severity: str,
        diagnosis: str,
        treatment: str,
        differential: str = "",
    ) -> None:
        self.rule_id = rule_id
        self.pattern = pattern
        self.severity = severity
        self.diagnosis = diagnosis
        self.treatment = treatment
        self.differential = differential

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.rule_id,
            "pattern": self.pattern,
            "severity": self.severity,
            "diagnosis": self.diagnosis,
            "treatment": self.treatment,
            "differential": self.differential,
        }


_RULES: List[DiagnosisRule] = [
    DiagnosisRule(
        "D001",
        ["cpu_high", "docker_unhealthy"],
        "WARNING",
        "Resource pressure causing service failures — high CPU is starving containers",
        "Inspect container logs (`docker logs <name>`); restart unhealthy services; identify the CPU-heavy process with `top`",
        "If restarting services does not reduce CPU, the cause is external — look for a runaway host process",
    ),
    DiagnosisRule(
        "D002",
        ["ram_high", "docker_unhealthy"],
        "WARNING",
        "Memory leak in a failing service — RAM pressure is killing containers",
        "Run `docker stats` to find the high-memory container; restart it; consider adding a memory limit in compose",
        "If RAM stays high after restart, the leak may be in a non-containerised process",
    ),
    DiagnosisRule(
        "D003",
        ["cpu_high", "ram_high"],
        "WARNING",
        "Possible runaway process or resource contention — both CPU and RAM elevated simultaneously",
        "Run `ps aux --sort=-%cpu` and `ps aux --sort=-%mem`; kill or restart the offending process",
        "Could be a legitimate heavy workload — check if operator initiated a build or batch job",
    ),
    DiagnosisRule(
        "D004",
        ["sensor_anomaly", "docker_unhealthy"],
        "WARNING",
        "Sensor anomaly coinciding with service failure — possible cascade where unhealthy service is causing the anomaly",
        "Cross-check which service is unhealthy and whether it provides the anomalous sensor — if so, restart it first",
        "Anomaly may precede the failure — in that case the root cause is upstream of the service",
    ),
    DiagnosisRule(
        "D005",
        ["aq_degraded", "calendar_soon"],
        "INFO",
        "Poor air quality before an upcoming meeting — cognitive performance risk",
        "Notify operator; suggest opening windows or moving to a better-ventilated space before the meeting",
        "If AQ source is external (outdoor pollution), indoor mitigation is limited — prioritise air purifier",
    ),
    DiagnosisRule(
        "D006",
        ["docker_unhealthy"],
        "INFO",
        "One or more services are unhealthy",
        "Check `docker ps` for unhealthy containers; run `docker logs <name>` to inspect the failure reason",
        "Healthcheck may be too aggressive — confirm the service is genuinely failing before restarting",
    ),
    DiagnosisRule(
        "D007",
        ["cpu_high"],
        "INFO",
        "Elevated CPU load — sustained spike may indicate a background process",
        "Monitor with `top`; check recent cron jobs or builds; if persists >15 min, investigate",
        "May be benign — index rebuild, compile, or backup in progress",
    ),
    DiagnosisRule(
        "D008",
        ["ram_high"],
        "INFO",
        "Elevated memory usage — may be cache growth or a slow leak",
        "Check `free -h`; look for processes with growing RSS over time; consider clearing page cache (`sync; echo 3 > /proc/sys/vm/drop_caches`)",
        "Some RAM pressure is normal under load; only act if swap is also active",
    ),
    DiagnosisRule(
        "D009",
        ["sensor_anomaly", "cpu_high", "ram_high"],
        "CRITICAL",
        "System-wide anomaly — CPU, RAM, and sensor deviations firing together indicate a serious event",
        "Immediate investigation required; check for active intrusion, runaway ML job, or infrastructure failure; consider alerting operator directly",
        "Could be a planned heavy workload — verify with operator before drastic action",
    ),
]

_SEVERITY_ORDER = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}


def _classify_observations(observations: List[str]) -> List[str]:
    tags: List[str] = []
    for o in observations:
        lo = o.lower()
        if ("cpu" in lo or "cpu" in o) and ("%" in o or "high" in lo or "anomaly" in lo):
            tags.append("cpu_high")
        if ("ram" in lo or "memory" in lo) and ("%" in o or "high" in lo or "anomaly" in lo or "pressure" in lo):
            tags.append("ram_high")
        if "docker:" in lo and "unhealthy" in lo:
            tags.append("docker_unhealthy")
        if "air quality:" in lo or "aqi" in lo or "pm2.5" in lo:
            tags.append("aq_degraded")
        if "anomaly" in lo:
            tags.append("sensor_anomaly")
        if "calendar" in lo or "meeting" in lo or "event in" in lo or "schedule" in lo:
            tags.append("calendar_soon")
    return list(dict.fromkeys(tags))  # preserve order, deduplicate


def _apply_rules(tags: List[str]) -> List[DiagnosisRule]:
    matched = [r for r in _RULES if all(p in tags for p in r.pattern)]
    matched.sort(key=lambda r: _SEVERITY_ORDER.get(r.severity, 9))
    return matched


async def _write_medical_report(
    diagnoses: List[DiagnosisRule],
    tags: List[str],
    observations: List[str],
) -> None:
    if not diagnoses:
        return
    primary = diagnoses[0]
    ts = datetime.now(timezone.utc).isoformat()
    report = {
        "timestamp": ts,
        "severity": primary.severity,
        "active_tags": tags,
        "primary_diagnosis": primary.diagnosis,
        "primary_treatment": primary.treatment,
        "all_diagnoses": [d.to_dict() for d in diagnoses],
        "source_observations": observations[:10],
    }
    summary = f"[House Doctor/{primary.severity}] {primary.diagnosis}"
    _recent_diagnoses.append(report)  # local ring buffer — zero-latency reads

    async with pooled_client(timeout=5.0) as client:
        try:
            await client.post(
                f"{MEMU_URL}/memory/memorize",
                json={"content": summary, "metadata": report, "category": "medical_report", "user_id": "keeper"},
            )
        except Exception as exc:
            logger.warning("Could not write medical report to memu-core: %s", exc)

        if primary.severity in ("WARNING", "CRITICAL"):
            try:
                await client.post(
                    f"{NOTIFY_URL}/notify",
                    json={"message": f"House Doctor [{primary.severity}]: {primary.diagnosis}", "channel": "system"},
                )
            except Exception:
                pass


@app.post("/diagnose")
async def diagnose(req: DiagnosisRequest) -> Dict[str, Any]:
    """Classify observations and apply differential diagnosis rules."""
    tags = _classify_observations(req.observations)
    diagnoses = _apply_rules(tags)
    if diagnoses:
        await _write_medical_report(diagnoses, tags, req.observations)
    return {
        "tags_detected": tags,
        "diagnoses": [d.to_dict() for d in diagnoses],
        "primary_severity": diagnoses[0].severity if diagnoses else "OK",
        "diagnosis_count": len(diagnoses),
        "report_written": len(diagnoses) > 0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/diagnoses/recent")
async def recent_diagnoses(limit: int = Query(default=10, ge=1, le=20)) -> Dict[str, Any]:
    """Return the most recent diagnoses from the in-memory ring buffer.

    Agentic reads this at the start of each proactive cycle to carry forward
    recent health history without re-sending observations.
    """
    entries = list(_recent_diagnoses)[-limit:]
    return {
        "diagnoses": entries,
        "count": len(entries),
        "buffer_size": len(_recent_diagnoses),
    }


@app.get("/rules")
async def list_rules() -> Dict[str, Any]:
    return {"rules": [r.to_dict() for r in _RULES], "count": len(_RULES)}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "service": "house-doctor", "rules": len(_RULES), "recent_diagnoses": len(_recent_diagnoses)}
