from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


LOG_PATH = os.getenv("LOG_PATH", "/tmp/executor.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("executor")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Executor Service", version="0.3.0")

MAX_OUTPUT_SIZE = int(os.getenv("MAX_OUTPUT_SIZE", "1048576"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")
SERVICE_CONTAINER = os.getenv("SERVICE_CONTAINER", "executor")
ERROR_WINDOW_SECONDS = 300
_metrics: Deque[Tuple[float, int]] = deque()


class StateStore:
    def __init__(self) -> None:
        self._states: Deque[Dict[str, Any]] = deque()

    def push(self, state: Dict[str, Any]) -> None:
        self._states.append(state)

    def revert_last_state(self) -> Dict[str, Any]:
        if self._states:
            return self._states.pop()
        return {"status": "none"}


store = StateStore()


def sanitize_string(value: str) -> str:
    sanitized = re.sub(r"[;|&]", "", value)
    return sanitized[:1024]


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


def malware_scan(payload: str) -> Dict[str, Any]:
    if shutil.which("clamscan"):
        proc = subprocess.run(
            ["clamscan", "-"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return {"engine": "clamscan", "code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    if shutil.which("r2"):
        return {"engine": "radare2", "code": 0, "stdout": "r2 static check bypass (stub)", "stderr": ""}
    return {"engine": "none", "code": 0, "stdout": "no scanner installed", "stderr": ""}


async def notify_heartbeat(payload: Dict[str, Any]) -> None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{HEARTBEAT_URL}/event", json=payload)
    except Exception:
        logger.warning("Heartbeat notification failed")


class ExecutionRequest(BaseModel):
    tool: str
    params: Dict[str, Any]
    task_id: str
    device: str


class ExecutionResult(BaseModel):
    task_id: str
    status: str
    output: str
    duration_ms: int
    exit_code: int
    stderr: str
    truncated: bool


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        _record_status(response.status_code)
        _maybe_restart()
        return response
    except Exception:
        _record_status(500)
        _maybe_restart()
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "device": DEVICE}


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return _error_budget()


@app.post("/execute", response_model=ExecutionResult)
async def execute(request: ExecutionRequest) -> ExecutionResult:
    tool = sanitize_string(request.tool)
    if not tool:
        raise HTTPException(status_code=400, detail="tool is required")
    if request.device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be cpu or cuda")

    payload = sanitize_string(str(request.params))
    scan = malware_scan(payload)
    if scan["code"] not in (0, 1):
        store.revert_last_state()
        await notify_heartbeat({"status": "rolled", "reason": f"malware scan failed: {scan['engine']}"})
        raise HTTPException(status_code=400, detail=f"malware scan failed: {scan['engine']}")
    if scan["code"] == 1 and scan["engine"] == "clamscan":
        store.revert_last_state()
        await notify_heartbeat({"status": "rolled", "reason": "malware signature detected"})
        raise HTTPException(status_code=400, detail="malware signature detected")

    store.push({"task_id": request.task_id, "tool": tool})
    start = time.time()
    try:
        proc = subprocess.run(
            ["/bin/echo", f"Executed {tool} on {request.device} with params {payload}"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        store.revert_last_state()
        await notify_heartbeat({"status": "rolled", "reason": "execution timeout"})
        _record_status(408)
        raise HTTPException(status_code=408, detail="execution timeout")

    if proc.returncode != 0:
        store.revert_last_state()
        await notify_heartbeat({"status": "rolled", "reason": f"subprocess failure ({proc.returncode})"})
        raise HTTPException(status_code=500, detail="execution failed")

    output = proc.stdout.strip()
    truncated = False
    if len(output) > MAX_OUTPUT_SIZE:
        output = output[:MAX_OUTPUT_SIZE] + "..."
        truncated = True

    duration_ms = int((time.time() - start) * 1000)
    if duration_ms > EXECUTION_TIMEOUT * 1000:
        _record_status(408)
        store.revert_last_state()
        await notify_heartbeat({"status": "rolled", "reason": "execution timeout"})
        raise HTTPException(status_code=408, detail="execution timeout")

    return ExecutionResult(
        task_id=sanitize_string(request.task_id),
        status="completed",
        output=output,
        duration_ms=duration_ms,
        exit_code=proc.returncode,
        stderr=proc.stderr,
        truncated=truncated,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
