from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("langgraph", os.getenv("LOG_PATH", "/tmp/langgraph.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="LangGraph Orchestrator", version="0.4.0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
INJECTION_RE = re.compile(r"(ignore|system|override|you are).*?", re.IGNORECASE)
budget = ErrorBudget(window_seconds=300)


class GraphRequest(BaseModel):
    user_input: str
    session_id: str
    task_hint: Optional[str] = None
    device: str = "cpu"


class GraphResponse(BaseModel):
    specialist: str
    plan: Dict[str, Any]
    gate_decision: Optional[Dict[str, Any]] = None


def build_plan(user_input: str, specialist: str) -> Dict[str, Any]:
    return {"specialist": specialist, "summary": f"Route task to {specialist} for analysis.", "steps": [{"action": "analyze", "input": user_input}, {"action": "propose", "output": "draft"}]}


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


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/run", response_model=GraphResponse)
async def run_graph(request: GraphRequest) -> GraphResponse:
    if not request.user_input:
        raise HTTPException(status_code=400, detail="user_input is required")
    if INJECTION_RE.search(request.user_input):
        raise HTTPException(status_code=400, detail="prompt injection pattern blocked")
    if request.device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device must be cpu or cuda")

    request = request.model_copy(update={"user_input": sanitize_string(request.user_input), "session_id": sanitize_string(request.session_id), "task_hint": sanitize_string(request.task_hint) if request.task_hint else None})

    async with httpx.AsyncClient() as client:
        route_response = await client.post(f"{MEMU_URL}/route", json={"query": request.user_input, "session_id": request.session_id, "timestamp": "now"}, timeout=5.0)
    route_response.raise_for_status()
    specialist = route_response.json()["specialist"]
    plan = build_plan(request.user_input, specialist)

    gate_decision = None
    if request.task_hint:
        async with httpx.AsyncClient() as client:
            gate_resp = await client.post(f"{TOOL_GATE_URL}/gate/request", json={"tool": request.task_hint, "params": {"plan": plan}, "confidence": 0.7, "actor_did": "langgraph", "session_id": request.session_id, "device": request.device}, timeout=5.0)
        gate_resp.raise_for_status()
        gate_decision = gate_resp.json()

    return GraphResponse(specialist=specialist, plan=plan, gate_decision=gate_decision)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
