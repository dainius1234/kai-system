from __future__ import annotations

import os
from typing import Any, Dict, List
import logging
import os
import shutil
import subprocess
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, Tuple

import httpx
from fastapi import FastAPI, Request

from common.runtime import AuditStream, ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("dashboard", os.getenv("LOG_PATH", "/tmp/dashboard.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Dashboard", version="0.4.0")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
LEDGER_URL = os.getenv("LEDGER_URL", "postgresql://keeper:***@postgres:5432/sovereign")
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("dashboard", required=os.getenv("AUDIT_REQUIRED", "false").lower()=="true")

LOG_PATH = os.getenv("LOG_PATH", "/tmp/dashboard.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("dashboard")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Dashboard", version="0.2.0")

TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
LEDGER_URL = os.getenv("LEDGER_URL", "postgresql://keeper:***@postgres:5432/sovereign")
ERROR_WINDOW_SECONDS = 300
SERVICE_CONTAINER = os.getenv("SERVICE_CONTAINER", "sovereign-dashboard")
_metrics: Deque[Tuple[float, int]] = deque()

NODES = {
    "tool-gate": f"{TOOL_GATE_URL}/health",
    "memu-core": os.getenv("MEMU_URL", "http://memu-core:8001") + "/health",
    "langgraph": os.getenv("LANGGRAPH_URL", "http://langgraph:8007") + "/health",
    "executor": os.getenv("EXECUTOR_URL", "http://executor:8002") + "/health",
    "heartbeat": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010") + "/status",
}

NO_GO_GRACE_REQUESTS = int(os.getenv("NO_GO_GRACE_REQUESTS", "20"))
MAX_ERROR_RATIO = float(os.getenv("MAX_ERROR_RATIO", "0.05"))


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        raise

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
        subprocess.run(["docker", "restart", SERVICE_CONTAINER], check=False)


async def fetch_status() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient() as client:
        for name, url in NODES.items():
            try:
                resp = await client.get(url, timeout=2.0)
                resp.raise_for_status()
                results[name] = {"status": "ok", "details": resp.json()}
            except Exception as exc:  # noqa: BLE001
                results[name] = {"status": "down", "error": str(exc)}
    return results


async def build_go_no_go_report() -> Dict[str, Any]:
    reasons: List[str] = []
    statuses = await fetch_status()
    down_nodes = [name for name, payload in statuses.items() if payload.get("status") != "ok"]
    if down_nodes:
        reasons.append(f"Critical services are down: {', '.join(down_nodes)}")

    async with httpx.AsyncClient() as client:
        tool_health_resp = await client.get(f"{TOOL_GATE_URL}/health", timeout=2.0)
        tool_health_resp.raise_for_status()
        tool_health = tool_health_resp.json()

        ledger_stats_resp = await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)
        ledger_stats_resp.raise_for_status()
        ledger_stats = ledger_stats_resp.json()

        metrics = budget.snapshot()

    mode = str(tool_health.get("mode", "PUB")).upper()
    if mode != "WORK":
        reasons.append("Tool Gate is not in WORK mode.")

    ledger_count = int(ledger_stats.get("count", 0))
    if ledger_count < NO_GO_GRACE_REQUESTS:
        reasons.append(
            f"Not enough proof yet ({ledger_count}/{NO_GO_GRACE_REQUESTS} gate decisions observed)."
        )

    error_ratio = float(metrics.get("error_ratio", 0.0))
    if error_ratio > MAX_ERROR_RATIO:
        reasons.append(
            f"Recent API error ratio is too high ({error_ratio:.1%} > {MAX_ERROR_RATIO:.1%})."
        )

    go = len(reasons) == 0
    return {
        "decision": "GO" if go else "NO_GO",
        "trust_status": "trusted" if go else "prove-first",
        "summary": "System looks stable enough to proceed." if go else "Hold execution until blockers are fixed.",
        "checks": {
            "required_mode": "WORK",
            "current_mode": mode,
            "minimum_gate_decisions": NO_GO_GRACE_REQUESTS,
            "current_gate_decisions": ledger_count,
            "max_error_ratio": MAX_ERROR_RATIO,
            "current_error_ratio": error_ratio,
            "down_nodes": down_nodes,
        },
        "reasons": reasons,
    }
async def fetch_ledger_size() -> int:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)
        resp.raise_for_status()
        payload = resp.json()
    return int(payload.get("count", 0))


async def fetch_memory_count() -> int:
    memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{memu_url}/memory/stats", timeout=2.0)
        resp.raise_for_status()
        payload = resp.json()
    return int(payload.get("records", 0))


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
    return {"status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)", "tool_gate_url": TOOL_GATE_URL}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()
    return {
        "status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "tool_gate_url": TOOL_GATE_URL,
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return _error_budget()


@app.get("/")
async def index() -> Dict[str, object]:
    statuses = await fetch_status()
    alive_nodes = [name for name, payload in statuses.items() if payload.get("status") == "ok"]
    async with httpx.AsyncClient() as client:
        ledger_size = int((await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)).json().get("count", 0))
        memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
        memory_count = int((await client.get(f"{memu_url}/memory/stats", timeout=2.0)).json().get("records", 0))

    go_no_go = await build_go_no_go_report()
    return {
        "service": "dashboard",
        "status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "tool_gate_url": TOOL_GATE_URL,
        "ledger_url": LEDGER_URL,
        "alive_nodes": alive_nodes,
        "node_status": statuses,
        "ledger_size": ledger_size,
        "memory_count": memory_count,
        "go_no_go": go_no_go,
    }


@app.get("/go-no-go")
async def go_no_go() -> Dict[str, Any]:
    return await build_go_no_go_report()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
