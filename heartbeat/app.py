from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict
import logging
import os
import shutil
import subprocess
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, Tuple

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

LOG_PATH = os.getenv("LOG_PATH", "/tmp/heartbeat.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("heartbeat")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Heartbeat Monitor", version="0.3.0")

CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
ALERT_WINDOW = int(os.getenv("ALERT_WINDOW", "300"))
AUTO_SLEEP_SECONDS = int(os.getenv("AUTO_SLEEP_SECONDS", "1800"))
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
EXECUTOR_LOG_PATH = Path(os.getenv("EXECUTOR_LOG_PATH", "/var/log/sovereign/executor.log"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

last_tick = time.time()
last_activity = time.time()
last_sleep_action = 0.0
budget = ErrorBudget(window_seconds=300)
SERVICE_CONTAINER = os.getenv("SERVICE_CONTAINER", "heartbeat")
ERROR_WINDOW_SECONDS = 300

last_tick = time.time()
last_activity = time.time()
_metrics: Deque[Tuple[float, int]] = deque()


class EventPayload(BaseModel):
    status: str
    reason: str


def _prune_metrics(now: float) -> None:
    while _metrics and now - _metrics[0][0] > ERROR_WINDOW_SECONDS:
        _metrics.popleft()


def _record_status(code: int) -> None:
    now = time.time()
    _metrics.append((now, code))
    _prune_metrics(now)


def _error_budget() -> Dict[str, float]:
    now = time.time()
    _prune_metrics(now)
    total = len(_metrics)
    if total == 0:
        return {"error_ratio": 0.0, "total": 0}
    errors = sum(1 for _, code in _metrics if code in {429, 500, 408})
    return {"error_ratio": errors / total, "total": total}


def _maybe_restart() -> None:
    budget = _error_budget()
    if budget["total"] < 10:
        return
    if budget["error_ratio"] > 0.03 and shutil.which("docker"):
        logger.error("Error budget breached (%.2f), restarting %s", budget["error_ratio"], SERVICE_CONTAINER)
        subprocess.run(["docker", "restart", SERVICE_CONTAINER], check=False)


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
    global last_activity
    if time.time() - last_activity > AUTO_SLEEP_SECONDS:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(f"{MEMU_URL}/memory/compress")
            logger.info("System sleeping")
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
        _record_status(response.status_code)
        _maybe_restart()
        return response
    except Exception:
        _record_status(500)
        _maybe_restart()
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    await _auto_sleep_check()
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()
async def metrics() -> Dict[str, Any]:
    return _error_budget()


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
    elapsed = time.time() - last_tick
    state = "healthy" if elapsed <= ALERT_WINDOW else "stale"
    return {"status": state, "elapsed_seconds": f"{elapsed:.1f}", "check_interval": str(CHECK_INTERVAL), "alert_window": str(ALERT_WINDOW), "intrusion_hits": str(_scan_executor_log())}
async def status() -> Dict[str, str]:
    await _auto_sleep_check()
    elapsed = time.time() - last_tick
    state = "healthy" if elapsed <= ALERT_WINDOW else "stale"
    return {
        "status": state,
        "elapsed_seconds": f"{elapsed:.1f}",
        "check_interval": str(CHECK_INTERVAL),
        "alert_window": str(ALERT_WINDOW),
        "intrusion_hits": str(_scan_executor_log()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
