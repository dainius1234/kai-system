from __future__ import annotations

import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional
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

from common.runtime import AuditStream, CircuitBreaker, ErrorBudget, detect_device, sanitize_string, setup_json_logger
from common.self_emp_advisor import advise, load_expenses, load_income_total, thresholds
from config import build_saver
from conviction import build_plan, low_conviction_feedback, score_conviction

logger = setup_json_logger("langgraph", os.getenv("LOG_PATH", "/tmp/langgraph.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="LangGraph Orchestrator", version="0.5.0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
TELEGRAM_ALERT_URL = os.getenv("TELEGRAM_ALERT_URL", "http://perception-telegram:9000/alert")
INJECTION_RE = re.compile(r"(ignore|system|override|you are).*?", re.IGNORECASE)
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("langgraph", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")
saver = build_saver()
MIN_CONVICTION = 8.0
MAX_RETHINKS = 3
last_low_conviction_alert = 0.0
SELF_EMP_ROOT = os.getenv("SELF_EMP_ROOT", "/data/self-emp")
INCOME_CSV = os.getenv("INCOME_CSV", f"{SELF_EMP_ROOT}/Accounting/income.csv")
EXPENSES_LOG = os.getenv("EXPENSES_LOG", f"{SELF_EMP_ROOT}/Accounting/expenses.log")
MEMU_BREAKER = CircuitBreaker(failure_threshold=int(os.getenv("MEMU_BREAKER_THRESHOLD", "3")), recovery_seconds=int(os.getenv("MEMU_BREAKER_RECOVERY", "30")))
TOOL_GATE_BREAKER = CircuitBreaker(failure_threshold=int(os.getenv("TOOL_BREAKER_THRESHOLD", "3")), recovery_seconds=int(os.getenv("TOOL_BREAKER_RECOVERY", "30")))


def infer_specialist_fallback(user_input: str, task_hint: Optional[str]) -> str:
    combined = f"{user_input} {task_hint or ''}".lower()
    if any(token in combined for token in ("image", "vision", "camera", "diagram")):
        return "Qwen-VL"
    if any(token in combined for token in ("plan", "reason", "policy", "risk")):
        return "DeepSeek-V4"
    return "Kimi-2.5"

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


class EpisodeRequest(BaseModel):
    user_id: str = "keeper"
    days: int = 7



async def fetch_offline_chunks(query: str, user_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{MEMU_URL}/memory/retrieve",
                params={"query": query, "user_id": user_id, "top_k": top_k},
                timeout=5.0,
            )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []
    except Exception:
        return []




def strategy_node(user_input: str) -> Dict[str, object]:
    income_total = load_income_total(INCOME_CSV)
    expenses_lines = load_expenses(EXPENSES_LOG)
    suggestions = advise(income_total=income_total, expenses_lines=expenses_lines)
    return {
        "advisor_mode": True,
        "input": user_input,
        "income_total": income_total,
        "suggestions": suggestions,
        "thresholds": thresholds(),
    }



async def maybe_alert_mtd_proximity(strategy: Dict[str, object]) -> None:
    th = strategy.get("thresholds", {})
    income = float(strategy.get("income_total", 0.0))
    mtd = float((th or {}).get("mtd_start", 50000))
    left = mtd - income
    if 0 <= left <= 2000:
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                await client.post(TELEGRAM_ALERT_URL, json={"text": f"Alert: £{max(left,0):.0f} left till MTD — prep GnuCash"})
        except Exception:
            logger.warning("Failed to deliver MTD proximity alert")

async def maybe_alert_low_conviction_average() -> None:
    global last_low_conviction_alert
    episodes = saver.recall(user_id="keeper", days=7)
    scores = [float(e.get("final_conviction", e.get("conviction_score", 0))) for e in episodes if e.get("final_conviction") or e.get("conviction_score")]
    if not scores:
        return
    avg_score = sum(scores) / len(scores)
    now = time.time()
    if avg_score < 7.0 and (now - last_low_conviction_alert) > 24 * 3600:
        last_low_conviction_alert = now
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                await client.post(TELEGRAM_ALERT_URL, json={"text": f"Kai needs tuning: 7-day conviction average={avg_score:.2f}/10"})
        except Exception:
            logger.warning("Failed to deliver low-conviction alert")
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
        budget.record(response.status_code)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        _record_status(response.status_code)
        _maybe_restart()
        return response
    except Exception:
        _record_status(500)
        _maybe_restart()
        raise


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "device": DEVICE, "dependencies": {"memu": MEMU_BREAKER.snapshot(), "tool_gate": TOOL_GATE_BREAKER.snapshot()}}


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

    specialist = infer_specialist_fallback(request.user_input, request.task_hint)
    if MEMU_BREAKER.allow():
        try:
            async with httpx.AsyncClient() as client:
                route_response = await client.post(
                    f"{MEMU_URL}/route",
                    json={"query": request.user_input, "session_id": request.session_id, "timestamp": "now"},
                    timeout=5.0,
                )
            route_response.raise_for_status()
            payload = route_response.json()
            specialist = payload.get("specialist", specialist)
            MEMU_BREAKER.record_success()
        except httpx.HTTPError:
            MEMU_BREAKER.record_failure()
            logger.warning("MEMU route unavailable; using local specialist fallback")
    else:
        logger.warning("MEMU circuit open; using local specialist fallback")

    rethink_count = 0
    chunks = await fetch_offline_chunks(request.user_input, user_id="keeper", top_k=5)
    plan = build_plan(request.user_input, specialist, chunks)
    conviction = score_conviction(request.user_input, plan, chunks, rethink_count)

    while conviction < MIN_CONVICTION and rethink_count < MAX_RETHINKS:
        rethink_count += 1
        feedback = low_conviction_feedback(conviction, chunks)
        prompt = f"{request.user_input}\n\nReflection: {feedback}"
        chunks = await fetch_offline_chunks(prompt, user_id="keeper", top_k=5)
        plan = build_plan(prompt, specialist, chunks)
        plan["reflection_feedback"] = feedback
        plan["rethink_count"] = rethink_count
        conviction = score_conviction(prompt, plan, chunks, rethink_count)

    if conviction < MIN_CONVICTION:
        plan["summary"] = f"Conviction too low ({conviction}/10). Need more data — suggest file or clarify?"

    strategy = strategy_node(request.user_input)
    plan["strategy"] = strategy

    gate_decision = None
    if request.task_hint:
        if TOOL_GATE_BREAKER.allow():
            try:
                async with httpx.AsyncClient() as client:
                    gate_resp = await client.post(
                        f"{TOOL_GATE_URL}/gate/request",
                        json={
                            "tool": request.task_hint,
                            "params": {"plan": plan},
                            "confidence": min(max(conviction / 10.0, 0.0), 1.0),
                            "actor_did": "langgraph",
                            "session_id": request.session_id,
                            "device": request.device,
                            "nonce": str(uuid.uuid4()),
                            "ts": time.time(),
                        },
                        timeout=5.0,
                    )
                gate_resp.raise_for_status()
                gate_decision = gate_resp.json()
                TOOL_GATE_BREAKER.record_success()
            except httpx.HTTPStatusError as exc:
                TOOL_GATE_BREAKER.record_failure()
                gate_decision = {"approved": False, "status": "blocked", "reason": f"tool-gate rejected request ({exc.response.status_code})"}
            except httpx.HTTPError:
                TOOL_GATE_BREAKER.record_failure()
                gate_decision = {"approved": False, "status": "unavailable", "reason": "tool-gate unavailable"}
        else:
            gate_decision = {"approved": False, "status": "blocked", "reason": "tool-gate circuit open"}

    episode_id = str(uuid.uuid4())
    saver.save_episode(
        {
            "episode_id": episode_id,
            "user_id": "keeper",
            "ts": time.time(),
            "input": request.user_input,
            "output": plan.get("summary", ""),
            "outcome_score": 1.0 if gate_decision else 0.7,
            "conviction_score": conviction,
            "rethink_count": rethink_count,
            "final_conviction": conviction,
        }
    )
    saver.decay("keeper", days=30, score_threshold=0.2)
    await maybe_alert_low_conviction_average()
    await maybe_alert_mtd_proximity(strategy)
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


@app.post("/episodes/recall")
async def recall_last_episode(req: EpisodeRequest) -> Dict[str, Any]:
    user_id = sanitize_string(req.user_id)
    episodes = saver.recall(user_id=user_id, days=req.days)
    raw_context = "\n".join(f"[{e.get('ts')}] IN={e.get('input')} OUT={e.get('output')} C={e.get('final_conviction')}" for e in episodes)
    return {"status": "ok", "count": len(episodes), "context": raw_context, "episodes": episodes}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
