"""Supervisor — watchdog and circuit-breaker control loop.

Periodically health-checks every core service, maintains per-service
circuit breakers, and fires alerts through the local NOTIFY_URL gateway
when something trips.  The dashboard can poll /status or /breakers
to render fleet health.

This service does NOT make any decisions — it only observes and reports.
Memu-core remains the orchestrator; supervisor is the ops-level safety net.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

from common.runtime import (
    CircuitBreaker,
    ErrorBudget,
    detect_device,
    setup_json_logger,
)

logger = setup_json_logger("supervisor", os.getenv("LOG_PATH", "/tmp/supervisor.json.log"))
DEVICE = detect_device()

app = FastAPI(title="Supervisor", version="0.2.0")

# ── service registry ────────────────────────────────────────────────
# Each entry is (name, health URL).  URLs use Docker-internal hostnames.
# The supervisor discovers the correct list from env config with sensible
# defaults matching docker-compose.minimal.yml.

SERVICES: List[Dict[str, str]] = [
    {"name": "tool-gate", "url": os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")},
    {"name": "memu-core", "url": os.getenv("MEMU_URL", "http://memu-core:8001")},
    {"name": "heartbeat", "url": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")},
    {"name": "dashboard", "url": os.getenv("DASHBOARD_URL", "http://dashboard:8080")},
    {"name": "verifier", "url": os.getenv("VERIFIER_URL", "http://verifier:8052")},
    {"name": "executor", "url": os.getenv("EXECUTOR_URL", "http://executor:8002")},
    {"name": "fusion-engine", "url": os.getenv("FUSION_URL", "http://fusion-engine:8053")},
    {"name": "memory-compressor", "url": os.getenv("MEMORY_COMPRESSOR_URL", "http://memory-compressor:8057")},
    {"name": "ledger-worker", "url": os.getenv("LEDGER_WORKER_URL", "http://ledger-worker:8056")},
    {"name": "metrics-gateway", "url": os.getenv("METRICS_GATEWAY_URL", "http://metrics-gateway:8058")},
]

# Extra services added when running the full stack.
_extra_names = os.getenv("SUPERVISOR_EXTRA_SERVICES", "")  # comma-separated name=url
for _pair in _extra_names.split(","):
    _pair = _pair.strip()
    if "=" in _pair:
        _n, _u = _pair.split("=", 1)
        SERVICES.append({"name": _n.strip(), "url": _u.strip()})

# ── circuit breakers ────────────────────────────────────────────────
FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "3"))
RECOVERY_SECONDS = int(os.getenv("CB_RECOVERY_SECONDS", "30"))
CHECK_INTERVAL = int(os.getenv("SUPERVISOR_CHECK_INTERVAL", "15"))

breakers: Dict[str, CircuitBreaker] = {
    svc["name"]: CircuitBreaker(
        failure_threshold=FAILURE_THRESHOLD,
        recovery_seconds=RECOVERY_SECONDS,
    )
    for svc in SERVICES
}

# Per-service last-known status so /status is always instant
_last_status: Dict[str, Dict[str, Any]] = {}

# ── notification gateway ────────────────────────────────────────────
NOTIFY_URL = os.getenv("NOTIFY_URL", "")

budget = ErrorBudget(window_seconds=300)

# ── proactive nudge config ──────────────────────────────────────────
TELEGRAM_ALERT_URL = os.getenv("TELEGRAM_ALERT_URL", "http://telegram-bot:8025/alert")
PROACTIVE_INTERVAL = int(os.getenv("PROACTIVE_INTERVAL", "900"))  # 15 min
_last_proactive_check = 0.0
_nudges_sent: Dict[str, float] = {}  # memory_id → timestamp (dedup)
NUDGE_COOLDOWN = int(os.getenv("NUDGE_COOLDOWN", "7200"))  # 2h between same nudge


def _send_notification(message: str) -> None:
    """Route alert through local gateway.  Skips silently if unconfigured."""
    if not NOTIFY_URL:
        logger.debug("Notification skipped (NOTIFY_URL not set)")
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(NOTIFY_URL, json={"text": message})
    except Exception:
        logger.warning("Notification delivery failed")


# ── health-check loop ──────────────────────────────────────────────
async def _check_service(client: httpx.AsyncClient, svc: Dict[str, str]) -> Dict[str, Any]:
    """Hit /health on a single service and update its circuit breaker."""
    name = svc["name"]
    url = f"{svc['url']}/health"
    cb = breakers[name]
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            cb.record_success()
            return {"name": name, "healthy": True, "status_code": 200, "breaker": cb.state}
        cb.record_failure()
        return {"name": name, "healthy": False, "status_code": resp.status_code, "breaker": cb.state}
    except Exception as exc:
        cb.record_failure()
        return {"name": name, "healthy": False, "error": str(exc)[:120], "breaker": cb.state}


async def _sweep() -> List[Dict[str, Any]]:
    """Check every registered service in parallel."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = await asyncio.gather(*[_check_service(client, svc) for svc in SERVICES])

    for r in results:
        name = r["name"]
        _last_status[name] = {**r, "checked_at": time.time()}
        cb = breakers[name]
        if cb.state == "open":
            _send_notification(f"[supervisor] circuit OPEN for {name}")
            logger.warning("Circuit breaker OPEN: %s", name)
    return list(results)


# ── proactive nudge engine ──────────────────────────────────────────
async def _proactive_check() -> None:
    """Poll memu-core for time-sensitive memories and push nudges via Telegram."""
    global _last_proactive_check
    now = time.time()
    if now - _last_proactive_check < PROACTIVE_INTERVAL:
        return
    _last_proactive_check = now

    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{memu_url}/memory/proactive")
            if resp.status_code != 200:
                return
            data = resp.json()
    except Exception:
        logger.debug("proactive check: memu-core unreachable")
        return

    nudges = data.get("nudges", [])
    if not nudges:
        return

    # deduplicate — don't send the same nudge within NUDGE_COOLDOWN
    to_send = []
    for nudge in nudges:
        mid = nudge.get("memory_id", "")
        last_sent = _nudges_sent.get(mid, 0.0)
        if now - last_sent >= NUDGE_COOLDOWN:
            to_send.append(nudge)

    if not to_send:
        return

    # build a single message with all nudges
    lines = ["🧠 *Kai — Proactive Nudges*\n"]
    for nudge in to_send[:5]:
        cat = nudge.get("category", "general")
        msg = nudge.get("nudge_message", "")
        signals = ", ".join(nudge.get("time_signals", []))
        lines.append(f"• [{cat}] {msg}")
        if signals:
            lines.append(f"  _Triggers: {signals}_")

    message = "\n".join(lines)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                TELEGRAM_ALERT_URL,
                json={"text": message},
            )
            if resp.status_code == 200:
                for nudge in to_send:
                    _nudges_sent[nudge.get("memory_id", "")] = now
                logger.info("Proactive nudge sent: %d items", len(to_send))
    except Exception:
        logger.debug("proactive nudge: telegram-bot unreachable")


async def _background_loop() -> None:
    """Runs forever in the background, sweeping at CHECK_INTERVAL."""
    while True:
        try:
            await _sweep()
        except Exception:
            logger.exception("Sweep failed")
        try:
            await _proactive_check()
        except Exception:
            logger.exception("Proactive check failed")
        await asyncio.sleep(CHECK_INTERVAL)


@app.on_event("startup")
async def start_background_tasks() -> None:
    asyncio.create_task(_background_loop())
    logger.info(
        "Supervisor started — monitoring %d services every %ds",
        len(SERVICES),
        CHECK_INTERVAL,
    )


# ── HTTP endpoints ──────────────────────────────────────────────────
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Fleet health overview.  Returns cached results from last sweep
    so this endpoint is always fast and side-effect-free."""
    open_count = sum(1 for cb in breakers.values() if cb.state == "open")
    fleet = "degraded" if open_count else "healthy"

    # v7: pull quarantine count + verifier verdict stats for fleet view
    quarantine_count = 0
    verifier_verdicts: Dict[str, Any] = {}
    try:
        memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
        async with httpx.AsyncClient(timeout=2.0) as client:
            q_resp = await client.get(f"{memu_url}/memory/quarantine/list")
            if q_resp.status_code == 200:
                quarantine_count = q_resp.json().get("count", 0)
    except Exception:
        pass
    try:
        verifier_url = os.getenv("VERIFIER_URL", "http://verifier:8052")
        async with httpx.AsyncClient(timeout=2.0) as client:
            v_resp = await client.get(f"{verifier_url}/metrics")
            if v_resp.status_code == 200:
                verifier_verdicts = v_resp.json().get("verdicts", {})
    except Exception:
        pass

    return {
        "fleet": fleet,
        "open_breakers": open_count,
        "quarantine_count": quarantine_count,
        "verifier_verdicts": verifier_verdicts,
        "services": _last_status,
    }


@app.get("/breakers")
async def get_breakers() -> Dict[str, Any]:
    """Per-service circuit breaker snapshots for the dashboard."""
    return {name: cb.snapshot() for name, cb in breakers.items()}


@app.post("/sweep")
async def trigger_sweep() -> Dict[str, Any]:
    """On-demand sweep — useful for the dashboard 'refresh' button."""
    results = await _sweep()
    return {"results": results}


@app.middleware("http")
async def metrics_middleware(request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8051")))
