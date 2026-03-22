"""Supervisor — dual-layer watchdog, circuit-breaker, and self-heal control loop.

Layer 2 (System-Level) of the resilience architecture:
  - Calls *deep* /health on every service (checks real dependencies)
  - Enforces circuit breakers — prevents cascade calls to dead services
  - Maintains recovery action registry (can restart/heal services)
  - Dead-man's switch on its own background loop
  - Provides fleet diagnostics to the dashboard

Memu-core remains the cognitive orchestrator; supervisor is the ops-level
safety net that ENFORCES resilience, not just observes it.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI

from common.resilience import TaskWatchdog
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

# ── Layer 2: task watchdog (detects frozen background loops) ────────
_watchdog = TaskWatchdog(stale_seconds=CHECK_INTERVAL * 3)

# ── Layer 2: recovery actions registry ──────────────────────────────
# Maps service → URL to POST when that service needs recovery.
# Services can expose a /recover endpoint that flushes caches, reconnects
# DB pools, etc.  This is the "self-heal" hook.
RECOVERY_ACTIONS: Dict[str, str] = {}
for svc in SERVICES:
    base = svc["url"]
    RECOVERY_ACTIONS[svc["name"]] = f"{base}/recover"

# Track recovery attempts to avoid infinite loops
_recovery_attempts: Dict[str, float] = {}
RECOVERY_COOLDOWN = int(os.getenv("RECOVERY_COOLDOWN", "120"))  # 2min between retries

# ── fleet health history (rolling window for trend detection) ───────
_fleet_history: List[Dict[str, Any]] = []  # last N sweep summaries
FLEET_HISTORY_MAX = 60  # ~15min at 15s interval

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
    """Hit /health on a single service and update its circuit breaker.

    Layer 2 upgrades:
      - Checks 'status' field in response (supports deep health)
      - Marks 'degraded' services as partial failures
      - Triggers recovery action when circuit opens
    """
    name = svc["name"]
    url = f"{svc['url']}/health"
    cb = breakers[name]
    try:
        resp = await client.get(url)
        if resp.status_code == 200:
            data = resp.json()
            svc_status = data.get("status", "ok")
            if svc_status == "degraded":
                # Service is up but unhealthy internally
                cb.record_failure()
                return {"name": name, "healthy": False, "degraded": True,
                        "checks": data.get("checks", {}),
                        "status_code": 200, "breaker": cb.state}
            cb.record_success()
            return {"name": name, "healthy": True, "status_code": 200,
                    "breaker": cb.state, "checks": data.get("checks", {})}
        cb.record_failure()
        return {"name": name, "healthy": False, "status_code": resp.status_code, "breaker": cb.state}
    except Exception as exc:
        cb.record_failure()
        return {"name": name, "healthy": False, "error": str(exc)[:120], "breaker": cb.state}


async def _attempt_recovery(client: httpx.AsyncClient, name: str) -> bool:
    """Layer 2: attempt to self-heal a service via its /recover endpoint."""
    now = time.time()
    last_attempt = _recovery_attempts.get(name, 0.0)
    if now - last_attempt < RECOVERY_COOLDOWN:
        return False  # cooldown active

    recover_url = RECOVERY_ACTIONS.get(name)
    if not recover_url:
        return False

    _recovery_attempts[name] = now
    try:
        resp = await client.post(recover_url, timeout=10.0)
        if resp.status_code == 200:
            logger.info("Recovery succeeded for %s", name)
            _send_notification(f"[supervisor] ✅ self-heal SUCCESS for {name}")
            return True
        logger.warning("Recovery returned %d for %s", resp.status_code, name)
    except Exception:
        logger.warning("Recovery endpoint unreachable for %s", name)
    _send_notification(f"[supervisor] ❌ self-heal FAILED for {name} — manual intervention needed")
    return False


async def _sweep() -> List[Dict[str, Any]]:
    """Check every registered service in parallel.

    Layer 2 upgrades:
      - Attempts recovery on services with open circuit breakers
      - Records fleet health history for trend detection
      - Reports frozen background tasks via TaskWatchdog
    """
    async with httpx.AsyncClient(timeout=5.0) as client:
        results = await asyncio.gather(*[_check_service(client, svc) for svc in SERVICES])

    unhealthy_count = 0
    recovered = []

    for r in results:
        name = r["name"]
        _last_status[name] = {**r, "checked_at": time.time()}
        cb = breakers[name]
        if not r.get("healthy"):
            unhealthy_count += 1
        if cb.state == "open":
            _send_notification(f"[supervisor] circuit OPEN for {name}")
            logger.warning("Circuit breaker OPEN: %s", name)
            # Layer 2: attempt self-heal
            async with httpx.AsyncClient(timeout=10.0) as rclient:
                ok = await _attempt_recovery(rclient, name)
                if ok:
                    recovered.append(name)

    # Record fleet snapshot for trend detection
    snap = {
        "ts": time.time(),
        "total": len(results),
        "healthy": len(results) - unhealthy_count,
        "unhealthy": unhealthy_count,
        "recovered": recovered,
        "frozen_tasks": _watchdog.frozen(),
    }
    _fleet_history.append(snap)
    if len(_fleet_history) > FLEET_HISTORY_MAX:
        _fleet_history.pop(0)

    return list(results)


# ── proactive nudge engine ──────────────────────────────────────────

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")


async def _get_current_mode() -> str:
    """Fetch effective mode from tool-gate (schedule-aware)."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{TOOL_GATE_URL}/gate/mode")
            if resp.status_code == 200:
                return str(resp.json().get("mode", "PUB")).upper()
    except Exception:
        pass
    return "PUB"


async def _proactive_check() -> None:
    """Poll memu-core for time-sensitive memories and push nudges via Telegram.

    P4d: Uses mode-filtered proactive endpoint with anti-annoyance.
    Falls back to full scan if filtered endpoint is unavailable.
    """
    global _last_proactive_check
    now = time.time()
    if now - _last_proactive_check < PROACTIVE_INTERVAL:
        return
    _last_proactive_check = now

    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    mode = await _get_current_mode()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # P4d: use mode-filtered proactive (includes anti-annoyance)
            resp = await client.get(f"{memu_url}/memory/proactive/filtered", params={"mode": mode})
            if resp.status_code != 200:
                # fallback to full scan
                resp = await client.get(f"{memu_url}/memory/proactive/full")
                if resp.status_code != 200:
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
        mid = nudge.get("memory_id", "") or nudge.get("message", "")[:50]
        last_sent = _nudges_sent.get(mid, 0.0)
        if now - last_sent >= NUDGE_COOLDOWN:
            to_send.append(nudge)

    if not to_send:
        return

    # build a single message with all nudges
    lines = ["🧠 *Kai — Proactive*\n"]
    type_icons = {
        "reminder": "⏰",
        "silence": "🤫",
        "goal_deadline": "🎯",
        "drift": "🧭",
        "fading_memory": "💭",
        "greeting": "👋",
        "check_in": "💚",
        "scheduled_task": "📅",
        "briefing": "📰",
        "escalation": "📢",
        "echo": "🪞",
    }
    for nudge in to_send[:5]:
        ntype = nudge.get("type", "reminder")
        icon = type_icons.get(ntype, "•")
        cat = nudge.get("category", "general")
        msg = nudge.get("message", "")
        lines.append(f"{icon} [{cat}] {msg}")

    message = "\n".join(lines)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                TELEGRAM_ALERT_URL,
                json={"text": message},
            )
            if resp.status_code == 200:
                for nudge in to_send:
                    key = nudge.get("memory_id", "") or nudge.get("message", "")[:50]
                    _nudges_sent[key] = now
                logger.info("Proactive nudge sent: %d items (mode=%s, types: %s)",
                            len(to_send), mode,
                            ", ".join(set(n.get("type", "?") for n in to_send)))
    except Exception:
        logger.debug("proactive nudge: telegram-bot unreachable")


async def _greeting_check() -> None:
    """P4f: Check if Kai should send a proactive greeting or check-in."""
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # try greeting first
            resp = await client.get(f"{memu_url}/memory/greeting")
            if resp.status_code == 200:
                data = resp.json()
                greeting = data.get("greeting")
                if greeting:
                    await client.post(
                        TELEGRAM_ALERT_URL,
                        json={"text": f"👋 {greeting}"},
                    )
                    logger.info("Greeting sent: %s", greeting[:60])
                    return

            # if no greeting, try check-in
            resp = await client.get(f"{memu_url}/memory/check-in")
            if resp.status_code == 200:
                data = resp.json()
                check_in = data.get("check_in")
                if check_in:
                    await client.post(
                        TELEGRAM_ALERT_URL,
                        json={"text": f"💚 {check_in}"},
                    )
                    logger.info("Check-in sent: %s", check_in[:60])
    except Exception:
        logger.debug("greeting/check-in: unreachable")


async def _fire_due_items() -> None:
    """P21: Fire due reminders and scheduled tasks via Telegram."""
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # fire due reminders
            resp = await client.get(f"{memu_url}/memory/reminders/due")
            if resp.status_code == 200:
                due_rems = resp.json().get("reminders", [])
                for r in due_rems[:5]:
                    rid = r.get("reminder_id", "")
                    text = r.get("text", "Reminder")[:300]
                    repeat = r.get("repeat", "once")
                    icon = "🔁" if repeat != "once" else "⏰"
                    try:
                        await client.post(
                            TELEGRAM_ALERT_URL,
                            json={"text": f"{icon} *Reminder:* {text}"},
                        )
                        await client.post(f"{memu_url}/memory/reminders/{rid}/fire")
                        logger.info("Reminder fired: %s", text[:50])
                    except Exception:
                        pass

            # fire due scheduled tasks
            resp = await client.get(f"{memu_url}/memory/schedule/due")
            if resp.status_code == 200:
                due_tasks = resp.json().get("tasks", [])
                for t in due_tasks[:5]:
                    tid = t.get("task_id", "")
                    title = t.get("title", "Task")[:300]
                    try:
                        await client.post(
                            TELEGRAM_ALERT_URL,
                            json={"text": f"📅 *Scheduled:* {title}"},
                        )
                        await client.post(f"{memu_url}/memory/schedule/task/{tid}/fire")
                        logger.info("Scheduled task fired: %s", title[:50])
                    except Exception:
                        pass
    except Exception:
        logger.debug("fire_due_items: memu-core unreachable")


async def _check_escalations() -> None:
    """P22: Check nudge escalation ladder and send escalated nudges via Telegram."""
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{memu_url}/memory/nudge/ladder")
            if resp.status_code == 200:
                targets = resp.json().get("targets", [])
                for t in targets[:3]:
                    if t.get("level", 1) >= 3:  # tough love or intervention
                        icon = "🚨" if t["level"] >= 4 else "📢"
                        try:
                            await client.post(
                                TELEGRAM_ALERT_URL,
                                json={"text": f"{icon} *Escalated nudge:* {t['target']} "
                                              f"({t.get('name', '?')}, {t.get('dismissals', 0)}x ignored)"},
                            )
                            logger.info("Escalated nudge sent: %s level=%d", t["target"], t["level"])
                        except Exception:
                            pass
    except Exception:
        logger.debug("check_escalations: memu-core unreachable")


async def _background_loop() -> None:
    """Runs forever in the background, sweeping at CHECK_INTERVAL.

    Layer 2: beats the TaskWatchdog each iteration so the /health
    endpoint can detect if THIS loop freezes.
    """
    while True:
        _watchdog.heartbeat("main_loop")
        try:
            await _sweep()
        except Exception:
            logger.exception("Sweep failed")
        _watchdog.heartbeat("proactive")
        try:
            await _proactive_check()
        except Exception:
            logger.exception("Proactive check failed")
        try:
            await _fire_due_items()
        except Exception:
            logger.exception("Fire due items failed")
        try:
            await _check_escalations()
        except Exception:
            logger.exception("Escalation check failed")
        try:
            await _greeting_check()
        except Exception:
            logger.exception("Greeting check failed")
        _watchdog.heartbeat("main_loop")
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
async def health() -> Dict[str, Any]:
    """Deep health — checks if background loop is alive, not just the process."""
    frozen = _watchdog.frozen()
    if frozen:
        return {"status": "degraded", "device": DEVICE,
                "frozen_tasks": frozen, "watchdog": _watchdog.snapshot()}
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/status")
async def status() -> Dict[str, Any]:
    """Fleet health overview with trend data and recovery history."""
    open_count = sum(1 for cb in breakers.values() if cb.state == "open")
    fleet = "degraded" if open_count else "healthy"

    # fleet trend — last 5 snapshots for mini-sparkline
    trend = []
    for snap in _fleet_history[-5:]:
        trend.append({
            "ts": snap["ts"],
            "healthy": snap["healthy"],
            "unhealthy": snap["unhealthy"],
        })

    # frozen task warning
    frozen = _watchdog.frozen()

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
        "trend": trend,
        "frozen_tasks": frozen,
        "recovery_attempts": {k: v for k, v in _recovery_attempts.items()},
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


@app.get("/fleet/history")
async def fleet_history() -> Dict[str, Any]:
    """Rolling fleet health history for trend analysis."""
    return {"history": _fleet_history, "count": len(_fleet_history)}


@app.get("/watchdog")
async def watchdog_status() -> Dict[str, Any]:
    """Background task watchdog status — shows which loops are alive/frozen."""
    return {"watchdog": _watchdog.snapshot(), "frozen": _watchdog.frozen()}


@app.post("/recover/{service_name}")
async def manual_recover(service_name: str) -> Dict[str, Any]:
    """Manually trigger recovery for a named service."""
    if service_name not in RECOVERY_ACTIONS:
        return {"ok": False, "error": f"unknown service: {service_name}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        ok = await _attempt_recovery(client, service_name)
    return {"ok": ok, "service": service_name}


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
