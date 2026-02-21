from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from common.runtime import AuditStream, ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("heartbeat", os.getenv("LOG_PATH", "/tmp/heartbeat.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Heartbeat Monitor", version="0.5.0")
CHECK_INTERVAL = int(os.getenv("CHECK_INTERVAL", "60"))
ALERT_WINDOW = int(os.getenv("ALERT_WINDOW", "300"))
AUTO_SLEEP_SECONDS = int(os.getenv("AUTO_SLEEP_SECONDS", "1800"))
SLEEP_COOLDOWN_SECONDS = int(os.getenv("SLEEP_COOLDOWN_SECONDS", "600"))
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://dashboard:8080")
EXECUTOR_LOG_PATH = Path(os.getenv("EXECUTOR_LOG_PATH", "/var/log/sovereign/executor.log"))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
CONTINGENCY_PATH = Path(os.getenv("CONTINGENCY_PATH", "/app/contingencies.json"))

audit = AuditStream("heartbeat", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")
last_tick = time.time()
last_activity = time.time()
last_sleep_action = 0.0
budget = ErrorBudget(window_seconds=300)


class EventPayload(BaseModel):
    status: str
    reason: str


class ContingencyResult(BaseModel):
    event: str
    method_statement: str
    checklist: List[str]
    executed_actions: List[Dict[str, str]]


def _load_contingencies() -> Dict[str, Dict[str, Any]]:
    if CONTINGENCY_PATH.exists():
        try:
            return json.loads(CONTINGENCY_PATH.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Failed to parse contingency library file: %s", CONTINGENCY_PATH)
    return {
        "intrusion_detected": {
            "method_statement": "Contain suspicious execution and preserve state for keeper review.",
            "checklist": ["Notify keeper", "Restart executor", "Checkpoint memory", "Audit log"],
            "actions": ["notify_keeper", "restart_executor", "memu_compress", "audit_log"],
        }
    }


CONTINGENCIES = _load_contingencies()


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


async def _run_action(action: str, event: str) -> Dict[str, str]:
    try:
        if action == "notify_keeper":
            _send_telegram_alert(f"Contingency triggered: {event}")
            return {"action": action, "status": "ok"}
        if action == "restart_executor":
            proc = subprocess.run(
                ["docker", "compose", "-f", "docker-compose.sovereign.yml", "restart", "executor"],
                check=False,
                capture_output=True,
                text=True,
            )
            return {"action": action, "status": "ok" if proc.returncode == 0 else "failed"}
        if action == "memu_compress":
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(f"{MEMU_URL}/memory/compress")
            return {"action": action, "status": "ok" if resp.status_code < 400 else "failed"}
        if action == "go_no_go_check":
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{DASHBOARD_URL}/go-no-go")
            return {"action": action, "status": "ok" if resp.status_code < 400 else "failed"}
        if action == "audit_log":
            audit.log("info", f"contingency executed: {event}")
            return {"action": action, "status": "ok"}
        return {"action": action, "status": "skipped"}
    except Exception as exc:
        return {"action": action, "status": f"failed:{exc.__class__.__name__}"}


async def run_contingency(event: str) -> ContingencyResult:
    scenario = CONTINGENCIES.get(event)
    if scenario is None:
        raise HTTPException(status_code=404, detail=f"Unknown contingency event: {event}")

    results: List[Dict[str, str]] = []
    for action in scenario.get("actions", []):
        results.append(await _run_action(action, event))

    return ContingencyResult(
        event=event,
        method_statement=scenario.get("method_statement", ""),
        checklist=scenario.get("checklist", []),
        executed_actions=results,
    )


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global last_activity
    last_activity = time.time()
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    await _auto_sleep_check()
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/contingency/library")
async def contingency_library() -> Dict[str, Any]:
    return {"status": "ok", "events": CONTINGENCIES}


@app.post("/contingency/run", response_model=ContingencyResult)
async def contingency_run(event: str = Query(..., description="Contingency event name")) -> ContingencyResult:
    result = await run_contingency(event)
    return result


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
    hits = _scan_executor_log()
    if hits > 0:
        await run_contingency("intrusion_detected")
    return {
        "status": state,
        "elapsed_seconds": f"{elapsed:.1f}",
        "check_interval": str(CHECK_INTERVAL),
        "alert_window": str(ALERT_WINDOW),
        "intrusion_hits": str(hits),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8010")))
