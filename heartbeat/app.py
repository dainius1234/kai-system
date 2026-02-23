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
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
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


def _send_telegram_alert(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        with httpx.Client(timeout=5.0) as client:
            client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message})
    except Exception:
        logger.warning("Telegram notification failed")


def _scan_executor_log() -> int:
    if not EXECUTOR_LOG_PATH.exists():
        return 0
    data = EXECUTOR_LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    patterns = ["timeout", "blocked", "injection"]
    hits = sum(data.lower().count(p) for p in patterns)
    if hits:
        _send_telegram_alert(f"[sovereign-heartbeat] executor alerts detected: {hits} hit(s)")
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
        _send_telegram_alert(
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
    await _auto_sleep_check()
    await _watchdog_check()
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/event")
async def event(payload: EventPayload) -> Dict[str, str]:
    msg = f"executor event: {payload.status} ({payload.reason})"
    logger.info(msg)
    _send_telegram_alert(msg)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
