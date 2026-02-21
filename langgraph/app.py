from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel


LOG_PATH = os.getenv("LOG_PATH", "/tmp/langgraph.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("langgraph")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="LangGraph Orchestrator", version="0.2.0")

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
INJECTION_RE = re.compile(r"(ignore|system|override|you are).*?", re.IGNORECASE)
ERROR_WINDOW_SECONDS = 300
SERVICE_CONTAINER = os.getenv("SERVICE_CONTAINER", "langgraph")
_metrics: Deque[Tuple[float, int]] = deque()


class GraphRequest(BaseModel):
    user_input: str
    session_id: str
    task_hint: Optional[str] = None
    device: str = "cpu"


class GraphResponse(BaseModel):
    specialist: str
    plan: Dict[str, Any]
    gate_decision: Optional[Dict[str, Any]] = None


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
        subprocess.run(["docker", "restart", SERVICE_CONTAINER], check=False)


def build_plan(user_input: str, specialist: str) -> Dict[str, Any]:
    return {
        "specialist": specialist,
        "summary": f"Route task to {specialist} for analysis.",
        "steps": [
            {"action": "analyze", "input": user_input},
            {"action": "propose", "output": "draft"},
        ],
    }


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


@app.post("/run", response_model=GraphResponse)
async def run_graph(request: GraphRequest) -> GraphResponse:
    if not request.user_input:
        raise HTTPException(status_code=400, detail="user_input is required")
    if INJECTION_RE.search(request.user_input):
        raise HTTPException(status_code=400, detail="prompt injection pattern blocked")
    if request.device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be cpu or cuda")
    request = request.model_copy(
        update={
            "user_input": sanitize_string(request.user_input),
            "session_id": sanitize_string(request.session_id),
            "task_hint": sanitize_string(request.task_hint) if request.task_hint else None,
        }
    )

    async with httpx.AsyncClient() as client:
        route_response = await client.post(
            f"{MEMU_URL}/route",
            json={
                "query": request.user_input,
                "session_id": request.session_id,
                "timestamp": "now",
            },
            timeout=5.0,
        )
    route_response.raise_for_status()
    route_payload = route_response.json()
    specialist = route_payload["specialist"]

    plan = build_plan(request.user_input, specialist)

    gate_decision = None
    if request.task_hint:
        async with httpx.AsyncClient() as client:
            gate_resp = await client.post(
                f"{TOOL_GATE_URL}/gate/request",
                json={
                    "tool": request.task_hint,
                    "params": {"plan": plan},
                    "confidence": 0.7,
                    "actor_did": "langgraph",
                    "session_id": request.session_id,
                    "device": request.device,
                },
                timeout=5.0,
            )
        gate_resp.raise_for_status()
        gate_decision = gate_resp.json()

    return GraphResponse(specialist=specialist, plan=plan, gate_decision=gate_decision)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
