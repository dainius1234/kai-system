from __future__ import annotations

import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, Request

from common.runtime import ErrorBudget, detect_device, setup_json_logger

logger = setup_json_logger("dashboard", os.getenv("LOG_PATH", "/tmp/dashboard.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Sovereign Dashboard", version="0.4.0")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
LEDGER_URL = os.getenv("LEDGER_URL", "postgresql://keeper:***@postgres:5432/sovereign")
budget = ErrorBudget(window_seconds=300)

NODES = {
    "tool-gate": f"{TOOL_GATE_URL}/health",
    "memu-core": os.getenv("MEMU_URL", "http://memu-core:8001") + "/health",
    "langgraph": os.getenv("LANGGRAPH_URL", "http://langgraph:8007") + "/health",
    "executor": os.getenv("EXECUTOR_URL", "http://executor:8002") + "/health",
    "heartbeat": os.getenv("HEARTBEAT_URL", "http://heartbeat:8010") + "/status",
}


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


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


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)", "tool_gate_url": TOOL_GATE_URL}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/")
async def index() -> Dict[str, object]:
    statuses = await fetch_status()
    alive_nodes = [name for name, payload in statuses.items() if payload.get("status") == "ok"]
    async with httpx.AsyncClient() as client:
        ledger_size = int((await client.get(f"{TOOL_GATE_URL}/ledger/stats", timeout=2.0)).json().get("count", 0))
        memu_url = os.getenv("MEMU_URL", "http://memu-core:8001")
        memory_count = int((await client.get(f"{memu_url}/memory/stats", timeout=2.0)).json().get("records", 0))

    return {
        "service": "dashboard",
        "status": "running (CPU)" if DEVICE == "cpu" else "running (CUDA)",
        "tool_gate_url": TOOL_GATE_URL,
        "ledger_url": LEDGER_URL,
        "alive_nodes": alive_nodes,
        "node_status": statuses,
        "ledger_size": ledger_size,
        "memory_count": memory_count,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
