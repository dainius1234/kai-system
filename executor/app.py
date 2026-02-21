from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, init_audit_or_exit, sanitize_string, setup_json_logger

logger = setup_json_logger("executor", os.getenv("LOG_PATH", "/tmp/executor.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)
audit = init_audit_or_exit("executor", logger)

app = FastAPI(title="Executor Service", version="0.5.0")
MAX_OUTPUT_SIZE = int(os.getenv("MAX_OUTPUT_SIZE", "1048576"))
EXECUTION_TIMEOUT = int(os.getenv("EXECUTION_TIMEOUT", "30"))
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "http://heartbeat:8010")
TOKENS_CAPACITY = int(os.getenv("TOKENS_CAPACITY", "1000"))
budget = ErrorBudget(window_seconds=300)


class StateStore:
    def __init__(self) -> None:
        self._states: list[Dict[str, Any]] = []

    def push(self, state: Dict[str, Any]) -> None:
        self._states.append(state)

    def revert_last_state(self) -> Dict[str, Any]:
        return self._states.pop() if self._states else {"status": "none"}


store = StateStore()
last_tokens_per_sec = 0.0


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
    tokens_per_sec: float


def malware_scan(payload: str) -> Dict[str, Any]:
    if shutil.which("clamscan"):
        proc = subprocess.run(["clamscan", "-"], input=payload, capture_output=True, text=True, timeout=10, check=False)
        return {"engine": "clamscan", "code": proc.returncode, "stderr": proc.stderr}
    return {"engine": "none", "code": 0, "stderr": ""}


async def notify_heartbeat(reason: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(f"{HEARTBEAT_URL}/event", json={"status": "rolled", "reason": reason})
    except Exception:
        logger.warning("Heartbeat notification failed")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
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


@app.get("/alive")
async def alive() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    snap = budget.snapshot()
    snap["tokens_per_sec"] = last_tokens_per_sec
    snap["load_ratio"] = min(last_tokens_per_sec / max(TOKENS_CAPACITY, 1), 1.0)
    return snap


@app.post("/execute", response_model=ExecutionResult)
async def execute(request: ExecutionRequest) -> ExecutionResult:
    global last_tokens_per_sec

    tool = sanitize_string(request.tool)
    if not tool:
        raise HTTPException(status_code=400, detail="tool is required")
    if request.device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be cpu or cuda")

    payload = sanitize_string(str(request.params))
    scan = malware_scan(payload)
    if scan["code"] == 1:
        store.revert_last_state()
        await notify_heartbeat("malware signature detected")
        audit.write("ERROR", "execution blocked by malware scan")
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
        await notify_heartbeat("execution timeout")
        audit.write("ERROR", "execution timeout")
        raise HTTPException(status_code=408, detail="execution timeout")

    if proc.returncode != 0:
        store.revert_last_state()
        await notify_heartbeat(f"subprocess failure ({proc.returncode})")
        audit.write("ERROR", f"subprocess failure code={proc.returncode}")
        raise HTTPException(status_code=500, detail="execution failed")

    output = proc.stdout.strip()
    truncated = len(output) > MAX_OUTPUT_SIZE
    if truncated:
        output = output[:MAX_OUTPUT_SIZE] + "..."

    duration_ms = int((time.time() - start) * 1000)
    if duration_ms > EXECUTION_TIMEOUT * 1000:
        store.revert_last_state()
        await notify_heartbeat("execution timeout")
        audit.write("ERROR", "execution timeout after process finished")
        raise HTTPException(status_code=408, detail="execution timeout")

    token_estimate = max(len(output.split()), 1)
    last_tokens_per_sec = token_estimate / max(duration_ms / 1000, 0.001)
    if last_tokens_per_sec / max(TOKENS_CAPACITY, 1) > 0.8:
        audit.write("WARN", f"high load ratio detected tps={last_tokens_per_sec:.2f}")

    return ExecutionResult(
        task_id=sanitize_string(request.task_id),
        status="completed",
        output=output,
        duration_ms=duration_ms,
        exit_code=proc.returncode,
        stderr=proc.stderr,
        truncated=truncated,
        tokens_per_sec=last_tokens_per_sec,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8002")))
