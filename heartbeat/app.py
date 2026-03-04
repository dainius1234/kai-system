from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("heartbeat", os.getenv("LOG_PATH", "/tmp/heartbeat.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Heartbeat Monitor", version="0.4.0")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
ALERT_WINDOW = int(os.getenv("ALERT_WINDOW", "300"))
AUTO_SLEEP_SECONDS = int(os.getenv("AUTO_SLEEP_SECONDS", "1800"))
SLEEP_COOLDOWN_SECONDS = int(os.getenv("SLEEP_COOLDOWN_SECONDS", "600"))
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
EXECUTOR_LOG_PATH = Path(os.getenv("EXECUTOR_LOG_PATH", "/var/log/sovereign/executor.log"))
CPU_USAGE_LIMIT = float(os.getenv("CPU_USAGE_LIMIT", "0.95"))
CPU_TEMP_LIMIT_C = float(os.getenv("CPU_TEMP_LIMIT_C", "90"))
GPU_TEMP_LIMIT_C = float(os.getenv("GPU_TEMP_LIMIT_C", "90"))

last_tick = time.time()
last_activity = time.time()
last_sleep_action = 0.0
budget = ErrorBudget(window_seconds=300)


class EventPayload(BaseModel):
    status: str
    reason: str


# Notification gateway URL — route alerts through a local service (e.g.
# perception-telegram) to preserve air-gap. Never call api.telegram.org
# directly from sovereign core services.
NOTIFY_URL = os.getenv("NOTIFY_URL", "")


def _send_notification(message: str) -> None:
    """Send alert via local notification gateway. Skips silently if unconfigured."""
    if not NOTIFY_URL:
        logger.debug("Notification skipped (NOTIFY_URL not configured)")
        return
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(NOTIFY_URL, json={"text": message})
    except Exception:
        logger.warning("Notification send failed")


def _scan_executor_log() -> int:
    if not EXECUTOR_LOG_PATH.exists():
        return 0
    data = EXECUTOR_LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    patterns = ["timeout", "blocked", "injection"]
    hits = sum(data.lower().count(p) for p in patterns)
    if hits:
        _send_notification(f"[sovereign-heartbeat] executor alerts detected: {hits} hit(s)")
    return hits


def _cpu_usage_ratio() -> float:
    try:
        load1, _, _ = os.getloadavg()
        cpus = max(os.cpu_count() or 1, 1)
        return min(load1 / cpus, 2.0)
    except Exception:
        return 0.0


def _cpu_temp_c() -> float:
    thermal = Path("/sys/class/thermal")
    if not thermal.exists():
        return 0.0
    temps: list[float] = []
    for sensor in thermal.glob("thermal_zone*/temp"):
        try:
            raw = float(sensor.read_text(encoding="utf-8").strip())
            temps.append(raw / 1000.0 if raw > 1000 else raw)
        except Exception:
            continue
    return max(temps) if temps else 0.0


def _gpu_temp_c() -> float:
    if not shutil.which("nvidia-smi"):
        return 0.0
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True, timeout=3)
        values = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return max(values) if values else 0.0
    except Exception:
        return 0.0


async def _watchdog_check() -> Dict[str, float]:
    usage = _cpu_usage_ratio()
    cpu_t = _cpu_temp_c()
    gpu_t = _gpu_temp_c()
    if usage >= CPU_USAGE_LIMIT or cpu_t >= CPU_TEMP_LIMIT_C or gpu_t >= GPU_TEMP_LIMIT_C:
        _send_notification(
            f"[watchdog] resource pressure: cpu_usage={usage:.2f}, cpu_temp={cpu_t:.1f}C, gpu_temp={gpu_t:.1f}C. Triggering auto-sleep"
        )
        await _auto_sleep_check()
    return {"cpu_usage": usage, "cpu_temp_c": cpu_t, "gpu_temp_c": gpu_t}


async def _auto_sleep_check() -> None:
    global last_sleep_action
    now = time.time()
    if now - last_activity <= AUTO_SLEEP_SECONDS:
        return
    if now - last_sleep_action <= SLEEP_COOLDOWN_SECONDS:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{MEMU_URL}/memory/compress")
        logger.info("System sleeping")
        last_sleep_action = now
    except Exception:
        logger.warning("System sleeping trigger failed")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global last_activity
    last_activity = time.time()
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/event")
async def event(payload: EventPayload) -> Dict[str, str]:
    msg = f"executor event: {payload.status} ({payload.reason})"
    logger.info(msg)
    _send_notification(msg)
    return {"status": "ok"}


@app.post("/tick")
async def tick() -> Dict[str, str]:
    global last_tick
    last_tick = time.time()
    return {"status": "ok", "message": "heartbeat received"}


@app.get("/status")
async def status() -> Dict[str, Any]:
    await _auto_sleep_check()
    watchdog = await _watchdog_check()
    elapsed = time.time() - last_tick
    state = "healthy" if elapsed <= ALERT_WINDOW else "stale"
    return {"status": state, "elapsed_seconds": f"{elapsed:.1f}", "check_interval": str(CHECK_INTERVAL), "alert_window": str(ALERT_WINDOW), "intrusion_hits": str(_scan_executor_log()), "watchdog": watchdog}


# ═══════════════════════════════════════════════════════════════════════
#  P14: TEMPORAL SELF-MODEL — "/self-assessment"
#
#  Weekly self-assessment: compare this week's metrics against the
#  previous week to detect improvement, decline, or stability.
#  "Systems that cannot measure themselves cannot improve themselves."
# ═══════════════════════════════════════════════════════════════════════

ASSESSMENT_WINDOW_DAYS = int(os.getenv("ASSESSMENT_WINDOW_DAYS", "7"))


def _trend(current: float, previous: float) -> str:
    """Label the direction of change between two numeric values."""
    if previous == 0:
        return "new" if current > 0 else "stable"
    delta = (current - previous) / max(abs(previous), 1e-9)
    if delta > 0.10:
        return "improving"
    if delta < -0.10:
        return "declining"
    return "stable"


async def _fetch_memu_stats(days: int, offset_days: int = 0) -> Dict[str, Any]:
    """Fetch memory stats from memu-core for a given window."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{MEMU_URL}/memory/stats")
            if r.status_code == 200:
                return r.json()
    except Exception as exc:
        logger.warning("Failed to fetch memu stats: %s", exc)
    return {}


async def _fetch_recent_episodes(days: int) -> Dict[str, Any]:
    """Fetch recent episode data for self-assessment metrics."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{MEMU_URL}/memory/diagnostics")
            if r.status_code == 200:
                return r.json()
    except Exception as exc:
        logger.warning("Failed to fetch diagnostics: %s", exc)
    return {}


@app.get("/self-assessment")
async def self_assessment() -> Dict[str, Any]:
    """Compute a temporal self-model comparing current vs previous period.

    Metrics tracked:
      - total_memories: growth in knowledge base
      - error_budget_usage: how often the system hit errors
      - uptime_ratio: heartbeat freshness
      - cpu_peak / gpu_peak: resource pressure trends

    Each metric gets a trend label: improving | declining | stable | new
    """
    stats = await _fetch_memu_stats(ASSESSMENT_WINDOW_DAYS)
    diagnostics = await _fetch_recent_episodes(ASSESSMENT_WINDOW_DAYS)

    # current snapshot
    total_memories = stats.get("total_memories", stats.get("total", 0))
    categories = stats.get("category_count", stats.get("categories", 0))
    budget_snap = budget.snapshot()
    error_rate = budget_snap.get("error_rate", 0.0)
    budget_ok = budget_snap.get("budget_ok", True)
    elapsed = time.time() - last_tick
    uptime_ratio = round(1.0 - min(elapsed / max(ALERT_WINDOW, 1), 1.0), 3)

    # resource snapshot
    cpu_pct = _cpu_usage_ratio()
    gpu_temp = _gpu_temp_c()
    cpu_temp = _cpu_temp_c()

    current = {
        "total_memories": total_memories,
        "categories": categories,
        "error_rate": round(error_rate, 4),
        "budget_ok": budget_ok,
        "uptime_ratio": uptime_ratio,
        "cpu_usage": round(cpu_pct, 3),
        "cpu_temp_c": cpu_temp,
        "gpu_temp_c": gpu_temp,
    }

    # We store previous assessment in memory so we can compare.
    # On first run, there's no previous — everything is "new".
    previous = _load_previous_assessment()

    trends = {}
    for key in ["total_memories", "error_rate", "uptime_ratio", "cpu_usage"]:
        curr_val = current.get(key, 0)
        prev_val = previous.get(key, 0) if previous else 0
        raw_trend = _trend(float(curr_val), float(prev_val))
        # for error_rate and cpu_usage, improving means going DOWN
        if key in ("error_rate", "cpu_usage") and raw_trend == "improving":
            raw_trend = "declining"
        elif key in ("error_rate", "cpu_usage") and raw_trend == "declining":
            raw_trend = "improving"
        trends[key] = raw_trend

    # overall health
    declining_count = sum(1 for v in trends.values() if v == "declining")
    improving_count = sum(1 for v in trends.values() if v == "improving")
    if declining_count >= 2:
        overall = "needs_attention"
    elif improving_count >= 2:
        overall = "improving"
    else:
        overall = "stable"

    assessment = {
        "status": "ok",
        "window_days": ASSESSMENT_WINDOW_DAYS,
        "current": current,
        "trends": trends,
        "overall": overall,
        "has_previous": previous is not None,
    }

    # save current as previous for next assessment
    _save_assessment(current)

    return assessment


# simple file-based storage for previous assessment
_ASSESSMENT_FILE = Path(os.getenv("ASSESSMENT_FILE", "/tmp/heartbeat_assessment.json"))


def _load_previous_assessment() -> Dict[str, Any] | None:
    import json
    if _ASSESSMENT_FILE.exists():
        try:
            return json.loads(_ASSESSMENT_FILE.read_text())
        except Exception:
            return None
    return None


def _save_assessment(current: Dict[str, Any]) -> None:
    import json
    try:
        _ASSESSMENT_FILE.write_text(json.dumps(current))
    except Exception as exc:
        logger.warning("Failed to save assessment: %s", exc)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
