"""Kai Introspection Service.

Owns the cold-path self-improvement endpoints that used to live inside
agentic/app.py: dream consolidation, evolver failure analysis, and the
security self-audit. None of these are on the chat/run hot path, so they
run as a separate process — a bug or hang here (e.g. a slow dream-cycle
clustering pass, a crash in evolver pattern analysis) can no longer stall
or take down live chat.

Checkpoint create/restore stay in agentic/app.py because they read and
mutate that process's live circuit-breaker/budget state directly; this
service has no access to that state. The one place this service still
needs a checkpoint written (post-dream) it asks the core service to do it
over HTTP, best-effort, same as every other non-critical side effect here.
"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict

import httpx
from fastapi import FastAPI

from common.runtime import AuditStream, INJECTION_RE, detect_device, sanitize_string, setup_json_logger
from kai_config import build_saver, run_dream_cycle, analyze_failures, load_evolver_reports
from security_audit import run_security_audit

logger = setup_json_logger("agentic-introspect", os.getenv("LOG_PATH", "/tmp/agentic_introspect.json.log"))
DEVICE = detect_device()

app = FastAPI(title="Kai Introspection Service", version="0.1.0")

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
AGENTIC_CORE_URL = os.getenv("AGENTIC_CORE_URL", "http://agentic:8007")
audit = AuditStream("agentic-introspect", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")
saver = build_saver()


@app.middleware("http")
async def audit_middleware(request, call_next):
    try:
        response = await call_next(request)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        raise


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "device": DEVICE}


@app.post("/dream")
async def trigger_dream() -> Dict[str, Any]:
    """Trigger a dream consolidation cycle.

    Can be called manually by the operator or automatically by heartbeat
    when the system detects extended idle time (> 30 min).
    """
    episodes = saver.recall(user_id="keeper", days=30)
    if len(episodes) < 5:
        return {"status": "insufficient_data", "message": "Need at least 5 episodes to dream."}

    cycle = run_dream_cycle(episodes)

    actionable = [i for i in cycle.insights if i.actionable]
    stored = 0
    for insight in actionable[:5]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{MEMU_URL}/memory/memorize",
                    json={
                        "timestamp": datetime.utcnow().isoformat(),
                        "event_type": "dream_insight",
                        "result_raw": insight.description,
                        "metrics": {"insight_type": insight.insight_type, "confidence": insight.confidence},
                        "relevance": insight.confidence,
                        "importance": 0.85,
                        "user_id": "kai",
                    },
                )
                stored += 1
        except Exception:
            logger.debug("Dream insight memorize failed")

    # H3b: post-dream checkpoint — core owns live breaker/budget state,
    # so ask it to snapshot itself rather than capturing state here.
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{AGENTIC_CORE_URL}/checkpoint",
                json={"label": f"post-dream-{cycle.cycle_id[:8]}"},
            )
    except Exception:
        logger.debug("Post-dream checkpoint request failed (non-critical)")

    return {
        "status": "ok",
        "cycle_id": cycle.cycle_id,
        "episodes_analysed": cycle.episodes_analysed,
        "insights_count": len(cycle.insights),
        "insights_stored": stored,
        "merged_rules": cycle.merged_rules,
        "failure_clusters": cycle.failure_clusters,
        "boundary_gaps": len(cycle.boundary_shifts),
        "duration_ms": cycle.duration_ms,
        "insights": [i.to_dict() for i in cycle.insights],
    }


@app.post("/evolve/analyze")
async def evolve_analyze() -> Dict[str, Any]:
    """Analyze recent failures and generate evolution suggestions.

    Returns concrete fix recommendations based on recurring failure patterns.
    """
    episodes = saver.recall(user_id="keeper", days=30)
    report = analyze_failures(episodes)

    stored = 0
    for s in report.suggestions:
        if s.priority in ("critical", "high"):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{MEMU_URL}/memory/memorize",
                        json={
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_type": "evolution_suggestion",
                            "result_raw": f"[{s.priority}] {s.fix}",
                            "metrics": {
                                "failure_class": s.failure_class,
                                "frequency": s.frequency,
                                "confidence": s.confidence,
                            },
                            "relevance": s.confidence,
                            "importance": 0.9 if s.priority == "critical" else 0.8,
                            "user_id": "kai",
                        },
                    )
                    stored += 1
            except Exception:
                pass

    return {
        "status": "ok",
        "report": report.to_dict(),
        "suggestions_stored": stored,
    }


@app.get("/evolve/suggestions")
async def evolve_suggestions() -> Dict[str, Any]:
    """Get all stored evolution reports and their suggestions."""
    reports = load_evolver_reports()
    return {
        "status": "ok",
        "report_count": len(reports),
        "reports": reports[-5:],
    }


@app.get("/security/audit")
async def security_audit_endpoint() -> Dict[str, Any]:
    """Run the security self-hacking audit against live defences."""
    audit_result = run_security_audit(
        injection_re=INJECTION_RE,
        sanitize_fn=sanitize_string,
    )
    return audit_result.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8023")))
