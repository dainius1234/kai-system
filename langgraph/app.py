from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from common.runtime import ErrorBudget, detect_device, init_audit_or_exit, sanitize_string, setup_json_logger

logger = setup_json_logger("langgraph", os.getenv("LOG_PATH", "/tmp/langgraph.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)
audit = init_audit_or_exit("langgraph", logger)

app = FastAPI(title="LangGraph Orchestrator", version="0.5.0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
EPISODE_PREFIX = os.getenv("EPISODE_PREFIX", "episodes")
INJECTION_RE = re.compile(r"(ignore|system|override|you are).*?", re.IGNORECASE)
budget = ErrorBudget(window_seconds=300)

try:
    import redis

    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
except Exception:
    redis_client = None


class GraphRequest(BaseModel):
    user_input: str
    session_id: str
    task_hint: Optional[str] = None
    device: str = "cpu"
    user_id: str = "keeper"


class GraphResponse(BaseModel):
    specialist: str
    plan: Dict[str, Any]
    gate_decision: Optional[Dict[str, Any]] = None


def build_plan(user_input: str, specialist: str) -> Dict[str, Any]:
    return {
        "specialist": specialist,
        "summary": f"Route task to {specialist} for analysis.",
        "steps": [{"action": "analyze", "input": user_input}, {"action": "propose", "output": "draft"}],
    }


def _episode_key(user_id: str) -> str:
    return f"{EPISODE_PREFIX}:{sanitize_string(user_id)}"


def save_episode(user_id: str, payload: Dict[str, Any]) -> None:
    if not redis_client:
        return
    data = dict(payload)
    data["ts"] = time.time()
    redis_client.lpush(_episode_key(user_id), json.dumps(data))
    redis_client.ltrim(_episode_key(user_id), 0, 500)


def recall_last_episode(user_id: str, days: int = 7) -> List[Dict[str, Any]]:
    if not redis_client:
        return []
    cutoff = time.time() - max(days, 1) * 86400
    records = redis_client.lrange(_episode_key(user_id), 0, 200)
    output: List[Dict[str, Any]] = []
    for raw in records:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if float(item.get("ts", 0.0)) >= cutoff:
            output.append(item)
    return output


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
    return {"status": "ok", "device": DEVICE, "episodic_backend": "redis" if redis_client else "disabled"}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/episodes/recall")
async def recall_endpoint(user_id: str = "keeper", days: int = 7) -> Dict[str, Any]:
    episodes = recall_last_episode(user_id=user_id, days=days)
    return {"status": "ok", "count": len(episodes), "episodes": episodes}


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
            "user_id": sanitize_string(request.user_id),
        }
    )

    async with httpx.AsyncClient() as client:
        route_response = await client.post(
            f"{MEMU_URL}/route",
            json={"query": request.user_input, "session_id": request.session_id, "timestamp": "now"},
            timeout=5.0,
        )
    route_response.raise_for_status()
    specialist = route_response.json()["specialist"]
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

    episode = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "user_input": request.user_input,
        "specialist": specialist,
        "plan": plan,
        "gate_decision": gate_decision,
    }
    save_episode(request.user_id, episode)
    audit.write("INFO", f"episode checkpointed user={request.user_id} session={request.session_id}")

    return GraphResponse(specialist=specialist, plan=plan, gate_decision=gate_decision)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
