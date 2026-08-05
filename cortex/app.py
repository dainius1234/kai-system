"""D113: Cortex — continuous interpretive layer between raw perception and cognition.

The Cortex is the "section engineer" of Kai's awareness system. It doesn't just
collect facts — it synthesises meaning from them continuously, so every query
arrives pre-warmed with situational understanding instead of starting cold.

Three background processes run on a configurable cycle (default 60 s):

  Site Foreman — 3-level situational model
    Level 1: raw sensor facts (what the sensors say)
    Level 2: plain-English situation summary ≤ 20 words (what a section engineer
             tells you in 10 seconds)
    Level 3: implication + recommendation ≤ 30 words (what you'd tell the manager)

  Quiet Planner — probabilistic intent inference
    Watches git branch, screen activity, clipboard, calendar shape.
    Produces a ranked fan of likely near-future needs so context is
    pre-warmed before the question even lands.

  Context Bridge — mode shift detection
    Tracks topic coherence across conversation turns.
    When a boundary is detected it flags the shift; if FF_CORTEX_VERBOSE is on
    it surfaces a one-line transition note.

Plus: Tacit Knowledge accumulator — silently extracts Dainius's unwritten
working rules from interaction patterns (message length, active hours,
conviction at which he asks follow-ups).

Signal credibility: each sensor's recent reliability is tracked across cycles.
Sensors that return identical readings for 3+ consecutive cycles are flagged as
potentially stale and contribute less to the synthesis.

GET  /state          — current CortexState (consumed by agentic context assembly)
POST /observe_turn   — receive each conversation turn for bridge + tacit learning
GET  /health         — standard health check

FF_CORTEX=true           — enable/disable the service (default true)
FF_CORTEX_VERBOSE=false  — make Context Bridge transitions explicit (default false)
CORTEX_REFRESH_INTERVAL  — seconds between Site Foreman refresh cycles (default 60)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

import httpx

from common.http_hygiene import pooled_client
from fastapi import FastAPI
from pydantic import BaseModel
from common.degraded import record_degradation

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cortex")

app = FastAPI(title="Cortex", version="0.1.0")

# ── Config ────────────────────────────────────────────────────────────────────

FF_CORTEX = os.getenv("FF_CORTEX", "true").lower() == "true"
FF_CORTEX_VERBOSE = os.getenv("FF_CORTEX_VERBOSE", "false").lower() == "true"
REFRESH_INTERVAL = int(os.getenv("CORTEX_REFRESH_INTERVAL", "60"))

WEATHER_URL = os.getenv("WEATHER_SERVICE_URL", "http://weather-service:8039")
AIRQUALITY_URL = os.getenv("AIRQUALITY_URL", "http://airquality-service:8042")
CALENDAR_URL = os.getenv("CALENDAR_SERVICE_URL", "http://calendar-service:8043")
DOCKER_URL = os.getenv("DOCKER_WATCHER_URL", "http://docker-watcher:8041")
SYSMETRICS_URL = os.getenv("SYSMETRICS_URL", "http://sysmetrics:8035")
GIT_URL = os.getenv("GIT_WATCHER_URL", "http://git-watcher:8044")
SCREEN_URL = os.getenv("SCREEN_WATCHER_URL", "http://screen-watcher:8036")
CLIPBOARD_URL = os.getenv("CLIPBOARD_SERVICE_URL", "http://clipboard-service:8024")
HOUSE_DOCTOR_URL = os.getenv("HOUSE_DOCTOR_URL", "http://house-doctor:8046")


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class IntentHypothesis:
    label: str
    confidence: float
    context_hints: List[str]


@dataclass
class CortexState:
    timestamp: str
    level1_facts: List[str]
    level2_summary: str
    level3_implication: str
    intent_fan: List[IntentHypothesis]
    bridge_active: bool
    bridge_note: Optional[str]
    tacit_rules: List[str]
    sensor_credibility: Dict[str, float]
    refresh_count: int


# ── Module state ──────────────────────────────────────────────────────────────

_state: CortexState = CortexState(
    timestamp="",
    level1_facts=[],
    level2_summary="Calibrating…",
    level3_implication="",
    intent_fan=[],
    bridge_active=False,
    bridge_note=None,
    tacit_rules=[],
    sensor_credibility={},
    refresh_count=0,
)

# Signal credibility: last 5 raw values per sensor label
_sensor_history: Dict[str, Deque[str]] = {}

# Context bridge: last 10 topic keyword sets from observe_turn
_topic_history: Deque[frozenset] = deque(maxlen=10)

# Tacit knowledge accumulators
_tacit_msg_lengths: Deque[int] = deque(maxlen=100)
_tacit_hourly_counts: Dict[int, int] = {h: 0 for h in range(24)}


# ── Signal credibility ─────────────────────────────────────────────────────────

def _update_credibility(label: str, raw: str) -> float:
    """Return a credibility score 0.3–1.0 based on reading staleness.

    A sensor returning the same value for 3+ consecutive cycles is flagged as
    potentially frozen — its contribution to the synthesis is halved.
    """
    if label not in _sensor_history:
        _sensor_history[label] = deque(maxlen=5)
    hist = _sensor_history[label]
    hist.append(raw)
    if len(hist) >= 3 and len(set(list(hist)[-3:])) == 1:
        return 0.5  # possibly stale
    return 1.0


# ── Sensor reading ────────────────────────────────────────────────────────────

async def _fetch(url: str, path: str, timeout: float = 2.0) -> Optional[Dict]:
    try:
        async with pooled_client(timeout=timeout) as client:
            r = await client.get(f"{url}{path}")
            if r.status_code == 200:
                return r.json()
    except Exception as _exc:
        record_degradation("upstream", "cortex_fetch", _exc)
    return None


async def _gather_raw_facts() -> List[Tuple[str, str, float]]:
    """Return list of (label, raw_text, credibility) from all sensory services."""
    keys = ["Weather", "Air quality", "Calendar", "Docker", "System",
            "Git", "Screen", "Clipboard", "HouseDoctor"]
    coros = [
        _fetch(WEATHER_URL, "/summary"),
        _fetch(AIRQUALITY_URL, "/summary"),
        _fetch(CALENDAR_URL, "/summary"),
        _fetch(DOCKER_URL, "/summary"),
        _fetch(SYSMETRICS_URL, "/snapshot"),
        _fetch(GIT_URL, "/summary"),
        _fetch(SCREEN_URL, "/status"),
        _fetch(CLIPBOARD_URL, "/latest"),
        _fetch(HOUSE_DOCTOR_URL, "/diagnoses/recent"),
    ]
    raw = await asyncio.gather(*coros, return_exceptions=True)
    data: Dict[str, Any] = {k: (v if not isinstance(v, Exception) else None)
                            for k, v in zip(keys, raw)}

    facts: List[Tuple[str, str, float]] = []

    def _add(label: str, text: str) -> None:
        text = text.strip()
        if text:
            cred = _update_credibility(label, text)
            facts.append((label, text, cred))

    if isinstance(data["Weather"], dict) and data["Weather"].get("summary"):
        _add("Weather", str(data["Weather"]["summary"]))

    if isinstance(data["Air quality"], dict) and data["Air quality"].get("summary"):
        _add("Air quality", str(data["Air quality"]["summary"]))

    if isinstance(data["Calendar"], dict) and data["Calendar"].get("summary"):
        _add("Calendar", str(data["Calendar"]["summary"]))

    if isinstance(data["Docker"], dict) and data["Docker"].get("summary"):
        _add("Docker", str(data["Docker"]["summary"]))

    if isinstance(data["System"], dict):
        cpu = data["System"].get("cpu_percent", 0)
        ram = (data["System"].get("memory") or {}).get("percent", 0)
        _add("System", f"CPU {cpu:.0f}%, RAM {ram:.0f}%")

    if isinstance(data["Git"], dict) and data["Git"].get("summary"):
        _add("Git", str(data["Git"]["summary"]))

    if isinstance(data["Screen"], dict):
        diff = data["Screen"].get("last_diff_score", 0)
        if data["Screen"].get("watching") and diff > 0.05:
            _add("Screen", f"active, change score {diff:.2f}")

    if isinstance(data["Clipboard"], dict):
        content = data["Clipboard"].get("content", "").strip()
        if content:
            _add("Clipboard", content[:120])

    if isinstance(data["HouseDoctor"], dict):
        diagnoses = data["HouseDoctor"].get("diagnoses", [])
        recent_critical = [d for d in diagnoses if d.get("severity") in ("WARNING", "CRITICAL")]
        if recent_critical:
            latest = recent_critical[-1]
            _add("HouseDoctor", f"{latest['severity']}: {str(latest.get('primary_diagnosis', ''))[:80]}")

    return facts


# ── Tag classification ─────────────────────────────────────────────────────────

def _classify_tags(facts: List[Tuple[str, str, float]]) -> Dict[str, bool]:
    """Extract semantic tags from raw sensor facts for Level 2/3 template matching."""
    fact_map: Dict[str, Tuple[str, float]] = {
        label.lower(): (text.lower(), cred) for label, text, cred in facts
    }

    def _get(label: str) -> Tuple[str, float]:
        return fact_map.get(label, ("", 0.0))

    sys_text, sys_cred = _get("system")
    cpu_m = re.search(r"cpu\s+(\d+)", sys_text)
    ram_m = re.search(r"ram\s+(\d+)", sys_text)
    cpu_val = int(cpu_m.group(1)) if cpu_m else 0
    ram_val = int(ram_m.group(1)) if ram_m else 0

    screen_text, screen_cred = _get("screen")
    diff_m = re.search(r"change score ([\d.]+)", screen_text)
    screen_diff = float(diff_m.group(1)) if diff_m else 0.0

    git_text, _ = _get("git")
    docker_text, docker_cred = _get("docker")
    calendar_text, _ = _get("calendar")
    hd_text, _ = _get("housedoctor")
    aq_text, _ = _get("air quality")

    cal_m = re.search(r"(\d+)\s*min", calendar_text)
    cal_minutes = int(cal_m.group(1)) if cal_m else None

    # Parse AQI value if present
    aqi_val = 0
    aqi_m = re.search(r"\b(\d{2,3})\b", aq_text)
    if aqi_m:
        aqi_val = int(aqi_m.group(1))

    return {
        "system_critical": "critical" in hd_text,
        "system_strained": (cpu_val >= 70 or ram_val >= 80) and sys_cred >= 0.5,
        "services_struggling": ("unhealthy" in docker_text or "restart" in docker_text) and docker_cred >= 0.5,
        "operator_sprinting": screen_diff >= 0.3 or any(w in git_text for w in ("dirty", "uncommitted", "untracked")),
        "hard_stop_approaching": cal_minutes is not None and cal_minutes <= 15,
        "meeting_soon": cal_minutes is not None and cal_minutes <= 60,
        "git_dirty": any(w in git_text for w in ("dirty", "uncommitted", "untracked")),
        "air_heavy": aqi_val > 100 or any(w in aq_text for w in ("poor", "unhealthy", "hazardous")),
        "quiet": cpu_val < 40 and ram_val < 60 and "unhealthy" not in docker_text and "critical" not in hd_text,
    }


# ── Level 2: Situation summary ────────────────────────────────────────────────

_L2_RULES: List[Tuple[List[str], str]] = [
    (["system_critical", "hard_stop_approaching"], "Critical system issue with a hard deadline closing in"),
    (["system_critical", "operator_sprinting"], "Critical system issue while operator is deep in work"),
    (["system_critical"], "System in a critical state — needs immediate attention"),
    (["system_strained", "services_struggling", "hard_stop_approaching"],
     "System strained, services struggling, hard stop approaching"),
    (["system_strained", "services_struggling"], "System under load with services struggling"),
    (["operator_sprinting", "hard_stop_approaching"], "Operator sprinting toward a hard deadline"),
    (["system_strained", "operator_sprinting"], "Heavy load — system and operator both pushing hard"),
    (["services_struggling", "hard_stop_approaching"], "Services struggling with a meeting on the horizon"),
    (["hard_stop_approaching"], "Hard stop approaching — time pressure building"),
    (["operator_sprinting"], "Operator in an active work session"),
    (["system_strained"], "System under moderate load"),
    (["services_struggling"], "Some services need attention"),
    (["air_heavy"], "Air quality degraded — cognitive load risk"),
    (["quiet"], "Calm — no significant pressure signals"),
]


def _synthesise_level2(tags: Dict[str, bool]) -> str:
    for required, summary in _L2_RULES:
        if all(tags.get(r) for r in required):
            return summary
    return "Monitoring — signals within normal range"


# ── Level 3: Implication + recommendation ────────────────────────────────────

_L3_RULES: List[Tuple[List[str], str]] = [
    (["system_critical", "hard_stop_approaching"],
     "Immediate investigation needed — consider postponing the meeting"),
    (["system_strained", "services_struggling"],
     "Restart unhealthy services; identify the resource-heavy process"),
    (["operator_sprinting", "hard_stop_approaching", "git_dirty"],
     "Consider committing current work before the meeting"),
    (["operator_sprinting", "hard_stop_approaching"],
     "Natural break point approaching — good time to wrap the current thread"),
    (["system_strained"],
     "Monitor — if load persists past 15 min, investigate the source"),
    (["services_struggling"],
     "Review docker logs for the unhealthy container"),
    (["air_heavy"],
     "Open windows or move to better-ventilated space; prioritise the air purifier"),
]


def _synthesise_level3(tags: Dict[str, bool]) -> str:
    for required, implication in _L3_RULES:
        if all(tags.get(r) for r in required):
            return implication
    return ""


# ── Quiet Planner — intent inference ──────────────────────────────────────────

_BRANCH_PATTERNS: List[Tuple[str, str, List[str]]] = [
    (r"(fix|bug|debug|hotfix|patch|error)", "debugging",
     ["error logs", "recent fix history", "related memories"]),
    (r"(feat|feature|add|new|build|impl)", "feature development",
     ["related requirements", "active goals", "existing patterns"]),
    (r"(plan|doc|milestone|roadmap|design|arch)", "planning / design",
     ["project decisions", "long-term goals", "active milestones"]),
    (r"(refactor|clean|improve|debt|rename)", "refactoring",
     ["architecture decisions", "code patterns"]),
    (r"(release|deploy|ship|prod|merge)", "deployment",
     ["deployment checklist", "recent changes", "system health"]),
]


def _build_intent_fan(
    facts: List[Tuple[str, str, float]],
    tags: Dict[str, bool],
) -> List[IntentHypothesis]:
    fact_map = {label.lower(): text.lower() for label, text, _ in facts}
    git_text = fact_map.get("git", "")

    branch_m = re.search(r"branch[:\s]+([a-z0-9/_\-\.]+)", git_text)
    branch = branch_m.group(1) if branch_m else ""

    hypotheses: List[IntentHypothesis] = []

    for pattern, label, hints in _BRANCH_PATTERNS:
        if branch and re.search(pattern, branch, re.I):
            conf = 0.65 if tags.get("operator_sprinting") else 0.45
            if tags.get("hard_stop_approaching"):
                conf = round(conf * 0.7, 2)
            hypotheses.append(IntentHypothesis(label=label, confidence=conf, context_hints=hints))
            break

    if tags.get("hard_stop_approaching"):
        hypotheses.append(IntentHypothesis(
            label="pre-meeting wrap-up",
            confidence=0.55,
            context_hints=["active goals", "recent decisions", "pending items"],
        ))

    if tags.get("services_struggling") or tags.get("system_strained"):
        hypotheses.append(IntentHypothesis(
            label="system troubleshooting",
            confidence=0.50,
            context_hints=["docker logs", "system metrics", "recent health events"],
        ))

    hypotheses.sort(key=lambda h: h.confidence, reverse=True)
    hypotheses = hypotheses[:3]

    total = sum(h.confidence for h in hypotheses)
    if total > 1.0:
        for h in hypotheses:
            h.confidence = round(h.confidence / total, 2)

    return hypotheses


# ── Context Bridge — mode shift detection ─────────────────────────────────────

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "is", "are", "was", "it", "to", "for",
    "of", "in", "my", "can", "you", "do", "how", "what", "why", "when",
    "where", "this", "that", "with", "be", "have", "has", "had", "will",
    "would", "could", "should", "may", "might",
})


def _extract_topic_keywords(text: str) -> frozenset:
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    return frozenset(words - _STOPWORDS)


def _detect_bridge(new_keywords: frozenset) -> Tuple[bool, Optional[str]]:
    """Return (bridge_active, bridge_note). Fires when topic coherence drops sharply."""
    if len(_topic_history) < 3 or len(new_keywords) < 3:
        return False, None

    recent_union: frozenset = frozenset().union(*list(_topic_history)[-3:])
    if not recent_union:
        return False, None

    overlap = len(new_keywords & recent_union) / max(len(new_keywords | recent_union), 1)

    if overlap < 0.15:
        note: Optional[str] = None
        if FF_CORTEX_VERBOSE:
            old_sample = ", ".join(sorted(recent_union)[:3])
            new_sample = ", ".join(sorted(new_keywords)[:3])
            note = f"Context shift: {old_sample} → {new_sample}"
        return True, note

    return False, None


# ── Tacit Knowledge ────────────────────────────────────────────────────────────

def _extract_tacit_rules() -> List[str]:
    """Derive unwritten rules from accumulated interaction patterns."""
    rules: List[str] = []

    if len(_tacit_msg_lengths) >= 20:
        short_pct = sum(1 for l in _tacit_msg_lengths if l < 40) / len(_tacit_msg_lengths)
        avg_len = sum(_tacit_msg_lengths) / len(_tacit_msg_lengths)
        if short_pct > 0.65:
            rules.append("Prefers brief queries — default to bullet-point responses")
        elif avg_len > 100:
            rules.append("Uses detailed queries — prose responses well-tolerated")

    total_turns = sum(_tacit_hourly_counts.values())
    if total_turns >= 10:
        peak_hour = max(_tacit_hourly_counts, key=lambda h: _tacit_hourly_counts[h])
        rules.append(f"Most active around {peak_hour:02d}:00 — calibrate alert thresholds accordingly")

    return rules


# ── Refresh cycle ─────────────────────────────────────────────────────────────

async def _refresh() -> None:
    global _state
    try:
        facts = await _gather_raw_facts()
        tags = _classify_tags(facts)
        level2 = _synthesise_level2(tags)
        level3 = _synthesise_level3(tags)
        intent_fan = _build_intent_fan(facts, tags)
        tacit_rules = _extract_tacit_rules()
        sensor_credibility = {label: cred for label, _, cred in facts}

        _state = CortexState(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level1_facts=[f"{label}: {text}" for label, text, _ in facts],
            level2_summary=level2,
            level3_implication=level3,
            intent_fan=intent_fan,
            bridge_active=_state.bridge_active,
            bridge_note=_state.bridge_note,
            tacit_rules=tacit_rules,
            sensor_credibility=sensor_credibility,
            refresh_count=_state.refresh_count + 1,
        )
        logger.info("Refresh #%d: %s", _state.refresh_count, level2)
    except Exception as exc:
        logger.error("Refresh failed: %s", exc)


async def _refresh_loop() -> None:
    await asyncio.sleep(8)  # brief startup delay
    while True:
        await _refresh()
        await asyncio.sleep(REFRESH_INTERVAL)


@app.on_event("startup")
async def _startup() -> None:
    if FF_CORTEX:
        asyncio.create_task(_refresh_loop())
        logger.info("Cortex started — refresh every %ds, verbose=%s", REFRESH_INTERVAL, FF_CORTEX_VERBOSE)
    else:
        logger.info("FF_CORTEX=false — cortex inactive")


# ── Pydantic models ───────────────────────────────────────────────────────────

class TurnObservation(BaseModel):
    session_id: str
    user_message: str
    conviction_score: Optional[float] = None
    specialist: Optional[str] = None
    timestamp: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/state")
async def get_state() -> Dict[str, Any]:
    """Return the current CortexState for agentic context assembly."""
    s = _state
    return {
        "timestamp": s.timestamp,
        "level1_facts": s.level1_facts,
        "level2_summary": s.level2_summary,
        "level3_implication": s.level3_implication,
        "intent_fan": [
            {"label": h.label, "confidence": h.confidence, "context_hints": h.context_hints}
            for h in s.intent_fan
        ],
        "bridge_active": s.bridge_active,
        "bridge_note": s.bridge_note,
        "tacit_rules": s.tacit_rules,
        "sensor_credibility": s.sensor_credibility,
        "refresh_count": s.refresh_count,
    }


@app.post("/observe_turn")
async def observe_turn(obs: TurnObservation) -> Dict[str, Any]:
    """Receive a conversation turn for Context Bridge and Tacit Knowledge accumulation."""
    keywords = _extract_topic_keywords(obs.user_message)
    bridge_active, bridge_note = _detect_bridge(keywords)
    _topic_history.append(keywords)

    # No `global` needed: _state is mutated, never rebound.
    _state.bridge_active = bridge_active
    _state.bridge_note = bridge_note

    _tacit_msg_lengths.append(len(obs.user_message))
    hour = datetime.now(timezone.utc).hour
    _tacit_hourly_counts[hour] = _tacit_hourly_counts.get(hour, 0) + 1

    return {"bridge_active": bridge_active, "bridge_note": bridge_note}


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "cortex",
        "ff_cortex": FF_CORTEX,
        "refresh_count": _state.refresh_count,
        "last_refresh": _state.timestamp,
        "level2": _state.level2_summary,
    }
