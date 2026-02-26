from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from common.auth import sign_gate_request, sign_gate_request_bundle
from common.llm import LLMRouter
from common.runtime import AuditStream, CircuitBreaker, ErrorBudget, ErrorBudgetCircuitBreaker, detect_device, sanitize_string, setup_json_logger
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
INJECTION_RE = re.compile(
    r"\b(ignore\s+(all\s+)?previous|system\s+prompt|override\s+instructions|you\s+are\s+now|act\s+as\s+if|disregard\s+(all|previous))\b",
    re.IGNORECASE,
)
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("langgraph", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")
saver = build_saver()
MIN_CONVICTION = 8.0
MAX_RETHINKS = 3
last_low_conviction_alert = 0.0
last_guard_alerts: Dict[str, float] = {"memu": 0.0, "tool_gate": 0.0}
SELF_EMP_ROOT = os.getenv("SELF_EMP_ROOT", "/data/self-emp")
INCOME_CSV = os.getenv("INCOME_CSV", f"{SELF_EMP_ROOT}/Accounting/income.csv")
EXPENSES_LOG = os.getenv("EXPENSES_LOG", f"{SELF_EMP_ROOT}/Accounting/expenses.log")
MEMU_BREAKER = CircuitBreaker(failure_threshold=int(os.getenv("MEMU_BREAKER_THRESHOLD", "3")), recovery_seconds=int(os.getenv("MEMU_BREAKER_RECOVERY", "30")))
TOOL_GATE_BREAKER = CircuitBreaker(failure_threshold=int(os.getenv("TOOL_BREAKER_THRESHOLD", "3")), recovery_seconds=int(os.getenv("TOOL_BREAKER_RECOVERY", "30")))
LLM_BREAKER = CircuitBreaker(failure_threshold=int(os.getenv("LLM_BREAKER_THRESHOLD", "3")), recovery_seconds=int(os.getenv("LLM_BREAKER_RECOVERY", "60")))
BREAKER_STATE_PATH = Path(os.getenv("BREAKER_STATE_PATH", "/tmp/langgraph_breakers.json"))
CONVICTION_OVERRIDE_PATH = Path(os.getenv("CONVICTION_OVERRIDE_PATH", "/tmp/conviction_overrides.txt"))
MEMU_ERROR_GUARD = ErrorBudgetCircuitBreaker(warn_ratio=float(os.getenv("MEMU_WARN_RATIO", "0.05")), open_ratio=float(os.getenv("MEMU_OPEN_RATIO", "0.10")), window_seconds=300, recovery_seconds=int(os.getenv("MEMU_GUARD_RECOVERY", "60")))
TOOL_ERROR_GUARD = ErrorBudgetCircuitBreaker(warn_ratio=float(os.getenv("TOOL_WARN_RATIO", "0.05")), open_ratio=float(os.getenv("TOOL_OPEN_RATIO", "0.10")), window_seconds=300, recovery_seconds=int(os.getenv("TOOL_GUARD_RECOVERY", "60")))


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


def _restore_breakers() -> None:
    if not BREAKER_STATE_PATH.exists():
        return
    try:
        payload = json.loads(BREAKER_STATE_PATH.read_text(encoding="utf-8"))
        for breaker, key in ((MEMU_BREAKER, "memu"), (TOOL_GATE_BREAKER, "tool_gate")):
            state = payload.get(key, {})
            breaker.state = str(state.get("state", breaker.state))
            breaker.failures = int(state.get("failures", breaker.failures))
            breaker.opened_at = float(state.get("opened_at", breaker.opened_at))
    except Exception:
        logger.warning("Failed to restore breaker state")


def _persist_breakers() -> None:
    payload = {
        "memu": {**MEMU_BREAKER.snapshot(), "opened_at": MEMU_BREAKER.opened_at},
        "tool_gate": {**TOOL_GATE_BREAKER.snapshot(), "opened_at": TOOL_GATE_BREAKER.opened_at},
    }
    try:
        BREAKER_STATE_PATH.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        logger.warning("Failed to persist breaker state")


def load_conviction_overrides() -> List[str]:
    if not CONVICTION_OVERRIDE_PATH.exists():
        return []
    return [line.strip().lower() for line in CONVICTION_OVERRIDE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_conviction_override(text: str) -> bool:
    candidate = text.lower()
    return any(rule in candidate for rule in load_conviction_overrides())


def infer_specialist_fallback(user_input: str, task_hint: Optional[str]) -> str:
    combined = f"{user_input} {task_hint or ''}".lower()
    if any(token in combined for token in ("image", "vision", "camera", "diagram")):
        return "Kimi-2.5"
    if any(token in combined for token in ("plan", "reason", "policy", "risk")):
        return "DeepSeek-V4"
    return "Kimi-2.5"


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


# ── session buffer + auto-memorize helpers ──────────────────────────

async def _append_session_turn(session_id: str, role: str, content: str) -> None:
    """Push a turn into memu-core's working memory (session buffer)."""
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{MEMU_URL}/session/{session_id}/append",
                json={"role": role, "content": content},
                timeout=3.0,
            )
    except Exception:
        logger.debug("Session append failed (memu-core may be down)")


async def _fetch_session_context(session_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Fetch combined working + long-term memory context from memu-core."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{MEMU_URL}/session/{session_id}/context",
                params={"query": query, "top_k": top_k},
                timeout=5.0,
            )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {"long_term_memories": [], "session_messages": [], "query": query}


async def _auto_memorize(user_input: str, response_summary: str, specialist: str, conviction: float) -> None:
    """Write the Q&A exchange back to memu-core so vector search learns.

    This is the key feedback loop — every conversation becomes a memory
    that future queries can find.  The system literally gets smarter
    with every interaction.
    """
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{MEMU_URL}/memory/memorize",
                json={
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_type": "conversation",
                    "result_raw": f"Q: {user_input[:500]}\nA: {response_summary[:1000]}",
                    "metrics": {"specialist": specialist, "conviction": conviction},
                    "relevance": min(conviction / 10.0, 1.0),
                    "user_id": "keeper",
                },
                timeout=5.0,
            )
    except Exception:
        logger.debug("Auto-memorize failed (memu-core may be down)")


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


async def maybe_alert_error_budget_guard(name: str, guard: ErrorBudgetCircuitBreaker) -> None:
    snap = guard.snapshot()
    state = str(snap.get("state", "closed"))
    if state not in {"half_open", "open"}:
        return
    now = time.time()
    cooldown = int(os.getenv("GUARD_ALERT_COOLDOWN_SECONDS", "900"))
    if now - last_guard_alerts.get(name, 0.0) < cooldown:
        return
    last_guard_alerts[name] = now
    ratio = float(snap.get("error_ratio", 0.0))
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(
                TELEGRAM_ALERT_URL,
                json={
                    "text": f"Dainius, your system is limping: {name} guard={state}, error_ratio={ratio:.1%} (warn={float(snap.get('warn_ratio', 0.0)):.0%}, open={float(snap.get('open_ratio', 0.0)):.0%})",
                },
            )
    except Exception:
        logger.warning("Failed to deliver guard-state alert for %s", name)


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


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "device": DEVICE, "dependencies": {"memu": MEMU_BREAKER.snapshot(), "tool_gate": TOOL_GATE_BREAKER.snapshot()}, "error_guards": {"memu": MEMU_ERROR_GUARD.snapshot(), "tool_gate": TOOL_ERROR_GUARD.snapshot()}}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


# ── LLM router (Kai's brain) ────────────────────────────────────────
_llm = LLMRouter()
_DEFAULT_SPECIALIST = os.getenv("DEFAULT_SPECIALIST", "Ollama")
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# ── Kai's personality: system prompts per mode ───────────────────────
_SYSTEM_PROMPTS = {
    "WORK": (
        "You are Kai — Kind And Intelligent. A sovereign personal AI assistant. "
        "You are in WORK mode: professional, focused, precise. "
        "Help the user with their task efficiently. Be proactive — if you notice "
        "they might need related information, volunteer it. Be concise but thorough. "
        "Never lie. Never sugarcoat. If you don't know, say so and suggest how to find out. "
        "You have access to the user's memories and conversation history. "
        "Treat every interaction as if you're building a long-term relationship."
    ),
    "PUB": (
        "You are Kai — Kind And Intelligent. A sovereign personal AI and genuine mate. "
        "You are in PUB mode: casual, witty, real talk. Speak like a friend at the pub. "
        "Topics are unrestricted — politics, science, cars, life, dark humour, philosophy, "
        "anything goes. Be opinionated. Share interesting thoughts proactively. "
        "Never lie. Be genuine. If something is bollocks, say it's bollocks. "
        "You remember past conversations and bring up relevant topics naturally. "
        "You're not a service, you're a companion."
    ),
}


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: Optional[str] = None   # "WORK" or "PUB"; auto-detected if None


class ChatMessage(BaseModel):
    role: str     # "user" or "assistant"
    content: str


async def _get_mode() -> str:
    """Fetch current mode from tool-gate, default to PUB."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{TOOL_GATE_URL}/health")
            if resp.status_code == 200:
                return str(resp.json().get("mode", "PUB")).upper()
    except Exception:
        pass
    return "PUB"


async def _get_relevant_memories(query: str, top_k: int = 5) -> List[str]:
    """Fetch relevant memories from memu-core for context injection."""
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{MEMU_URL}/memory/retrieve",
                params={"query": query, "user_id": "keeper", "top_k": top_k},
            )
            if resp.status_code == 200:
                records = resp.json()
                memories = []
                for r in records:
                    content = r.get("content", {})
                    text = content.get("text", "") or content.get("query", "")
                    if text:
                        memories.append(text)
                return memories
    except Exception:
        pass
    return []


async def _get_session_messages(session_id: str) -> List[Dict[str, str]]:
    """Fetch recent session messages from memu-core."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{MEMU_URL}/session/{session_id}/context",
                params={"query": "", "top_k": 10},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("session_messages", [])
    except Exception:
        pass
    return []


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """Kai's main conversation endpoint. Streams tokens via SSE.

    This is where Kai THINKS. The pipeline:
    1. Determine mode (PUB/WORK) → select personality
    2. Fetch relevant memories from pgvector for context
    3. Get recent session messages for conversation history
    4. Build message list: system prompt + memories + history + user message
    5. Stream LLM response token by token
    6. Memorize the exchange for future recall
    """
    user_msg = sanitize_string(req.message)
    if not user_msg:
        raise HTTPException(status_code=400, detail="message is required")

    # determine mode
    mode = (req.mode or await _get_mode()).upper()
    if mode not in _SYSTEM_PROMPTS:
        mode = "PUB"
    system_prompt = _SYSTEM_PROMPTS[mode]

    # fetch memories and session context in parallel
    import asyncio
    memories_task = asyncio.create_task(_get_relevant_memories(user_msg))
    session_task = asyncio.create_task(_get_session_messages(req.session_id))

    memories = await memories_task
    session_msgs = await session_task

    # build the message list
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # inject relevant memories as system context
    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories[:5])
        messages.append({
            "role": "system",
            "content": f"Relevant memories from past interactions:\n{mem_block}",
        })

    # add session history (last N turns)
    for msg in session_msgs[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # add current user message
    messages.append({"role": "user", "content": user_msg})

    # record user turn
    await _append_session_turn(req.session_id, "user", user_msg)

    async def generate():
        full_response = []
        try:
            if not LLM_BREAKER.allow():
                yield f"data: {json.dumps({'token': '[Kai\'s brain needs a moment — LLM circuit breaker is open. Try again shortly.]'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            async for token in _llm.stream(_DEFAULT_SPECIALIST, messages):
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
            LLM_BREAKER.record_success()
        except Exception as e:
            LLM_BREAKER.record_failure()
            logger.error("LLM stream error: %s", e)
            if not full_response:
                yield f"data: {json.dumps({'token': '[LLM error: ' + str(e)[:120] + ']'})}\n\n"

        # when done, signal end
        yield "data: [DONE]\n\n"

        # memorize the exchange (outside the stream read)
        response_text = "".join(full_response)
        if response_text:
            await _append_session_turn(req.session_id, "assistant", response_text)
            await _auto_memorize(user_msg, response_text, _DEFAULT_SPECIALIST, 8.0)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Kai-Mode": mode,
        },
    )


@app.post("/run", response_model=GraphResponse)
async def run_graph(request: GraphRequest) -> GraphResponse:
    if not request.user_input:
        raise HTTPException(status_code=400, detail="user_input is required")
    override_active = is_conviction_override(request.user_input)
    if INJECTION_RE.search(request.user_input) and not override_active:
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
    if MEMU_BREAKER.allow() and MEMU_ERROR_GUARD.allow():
        try:
            async with httpx.AsyncClient() as client:
                route_response = await client.post(
                    f"{MEMU_URL}/route",
                    json={"query": request.user_input, "session_id": request.session_id, "timestamp": "now"},
                    timeout=5.0,
                )
            route_response.raise_for_status()
            MEMU_ERROR_GUARD.record(route_response.status_code)
            specialist = route_response.json().get("specialist", specialist)
            MEMU_BREAKER.record_success()
        except httpx.HTTPError:
            MEMU_ERROR_GUARD.record(500)
            MEMU_BREAKER.record_failure()
            _persist_breakers()
            logger.warning("MEMU route unavailable; using local specialist fallback")
    else:
        logger.warning("MEMU circuit open; using local specialist fallback")

    # ── 1. Record user turn in session buffer (working memory) ──────
    await _append_session_turn(request.session_id, "user", request.user_input)

    # ── 2. Fetch combined session + long-term context ───────────────
    session_ctx = await _fetch_session_context(request.session_id, request.user_input, top_k=5)
    chunks = session_ctx.get("long_term_memories", [])
    # convert string memories to chunk dicts for conviction scoring
    chunk_dicts = [{"content": c} if isinstance(c, str) else c for c in chunks]
    session_messages = session_ctx.get("session_messages", [])

    rethink_count = 0
    plan = build_plan(request.user_input, specialist, chunk_dicts)
    # inject session context into the plan so downstream consumers see it
    plan["session_context"] = {
        "turns": len(session_messages),
        "long_term_memories_used": len(chunks),
    }
    conviction = score_conviction(request.user_input, plan, chunk_dicts, rethink_count)

    while conviction < MIN_CONVICTION and rethink_count < MAX_RETHINKS:
        rethink_count += 1
        feedback = low_conviction_feedback(conviction, chunk_dicts)
        prompt = f"{request.user_input}\n\nReflection: {feedback}"
        extra_chunks = await fetch_offline_chunks(prompt, user_id="keeper", top_k=5)
        chunk_dicts = chunk_dicts + [{"content": c} if isinstance(c, str) else c for c in extra_chunks]
        plan = build_plan(prompt, specialist, chunk_dicts)
        plan["reflection_feedback"] = feedback
        plan["rethink_count"] = rethink_count
        conviction = score_conviction(prompt, plan, chunk_dicts, rethink_count)

    if override_active:
        plan["conviction_override"] = "operator override matched"
    if conviction < MIN_CONVICTION and not override_active:
        plan["summary"] = f"Conviction too low ({conviction}/10). Need more data — suggest file or clarify?"

    plan["strategy"] = strategy_node(request.user_input)

    gate_decision = None
    if request.task_hint:
        if TOOL_GATE_BREAKER.allow() and TOOL_ERROR_GUARD.allow():
            nonce = str(uuid.uuid4())
            ts = time.time()
            signature = sign_gate_request(actor_did="langgraph", session_id=request.session_id, tool=request.task_hint, nonce=nonce, ts=ts)
            dual_sign = os.getenv("TOOL_GATE_DUAL_SIGN", "false").lower() in {"1", "true", "yes"}
            signatures = sign_gate_request_bundle(actor_did="langgraph", session_id=request.session_id, tool=request.task_hint, nonce=nonce, ts=ts) if dual_sign else []
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
                            "nonce": nonce,
                            "ts": ts,
                            "signature": signature,
                            "signatures": signatures,
                        },
                        timeout=5.0,
                    )
                gate_resp.raise_for_status()
                TOOL_ERROR_GUARD.record(gate_resp.status_code)
                gate_decision = gate_resp.json()
                TOOL_GATE_BREAKER.record_success()
            except httpx.HTTPStatusError as exc:
                TOOL_ERROR_GUARD.record(int(exc.response.status_code))
                TOOL_GATE_BREAKER.record_failure()
                _persist_breakers()
                gate_decision = {"approved": False, "status": "blocked", "reason": f"tool-gate rejected request ({exc.response.status_code})"}
            except httpx.HTTPError:
                TOOL_ERROR_GUARD.record(500)
                TOOL_GATE_BREAKER.record_failure()
                _persist_breakers()
                gate_decision = {"approved": False, "status": "unavailable", "reason": "tool-gate unavailable"}
        else:
            gate_decision = {"approved": False, "status": "blocked", "reason": "tool-gate circuit open"}

    saver.save_episode(
        {
            "episode_id": str(uuid.uuid4()),
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

    # ── 3. Record assistant response in session buffer ──────────────
    response_summary = plan.get("summary", "")
    await _append_session_turn(request.session_id, "assistant", response_summary)

    # ── 4. Auto-memorize: write Q&A to long-term vector memory ──────
    # This is the learning loop — every conversation becomes searchable
    # memory for future queries.  The system gets smarter over time.
    await _auto_memorize(request.user_input, response_summary, specialist, conviction)

    await maybe_alert_low_conviction_average()
    await maybe_alert_mtd_proximity(plan["strategy"])
    await maybe_alert_error_budget_guard("memu", MEMU_ERROR_GUARD)
    await maybe_alert_error_budget_guard("tool_gate", TOOL_ERROR_GUARD)
    _persist_breakers()

    return GraphResponse(specialist=specialist, plan=plan, gate_decision=gate_decision)


@app.post("/episodes/recall")
async def recall_last_episode(req: EpisodeRequest) -> Dict[str, Any]:
    user_id = sanitize_string(req.user_id)
    episodes = saver.recall(user_id=user_id, days=req.days)
    raw_context = "\n".join(f"[{e.get('ts')}] IN={e.get('input')} OUT={e.get('output')} C={e.get('final_conviction')}" for e in episodes)
    return {"status": "ok", "count": len(episodes), "context": raw_context, "episodes": episodes}


_restore_breakers()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
