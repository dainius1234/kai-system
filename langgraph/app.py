from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


app = FastAPI(title="LangGraph Orchestrator", version="0.2.0")

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")


class GraphRequest(BaseModel):
    user_input: str
    session_id: str
    task_hint: Optional[str] = None


class GraphResponse(BaseModel):
    specialist: str
    plan: Dict[str, Any]
    gate_decision: Optional[Dict[str, Any]] = None


def build_plan(user_input: str, specialist: str, device: str) -> Dict[str, Any]:
    return {
        "specialist": specialist,
        "device": device,
        "summary": f"Route task to {specialist} for analysis.",
        "steps": [
            {"action": "analyze", "input": user_input},
            {"action": "propose", "output": "draft"},
        ],
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/run", response_model=GraphResponse)
async def run_graph(request: GraphRequest) -> GraphResponse:
    if not request.user_input:
        raise HTTPException(status_code=400, detail="user_input is required")

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
    device = route_payload.get("context_payload", {}).get("device", "cpu")

    plan = build_plan(request.user_input, specialist, device)

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
                    "device": device,
                    "request_source": "langgraph",
                },
                timeout=5.0,
            )
        gate_resp.raise_for_status()
        gate_decision = gate_resp.json()

    return GraphResponse(specialist=specialist, plan=plan, gate_decision=gate_decision)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
