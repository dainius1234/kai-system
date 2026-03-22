from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from common.auth import sign_gate_request, sign_gate_request_bundle
from common.llm import LLMRouter
from common.runtime import AuditStream, CircuitBreaker, ErrorBudget, ErrorBudgetCircuitBreaker, detect_device, sanitize_string, setup_json_logger
from common.self_emp_advisor import advise, load_expenses, load_income_total, thresholds
from kai_config import build_saver, classify_failure, extract_metacognitive_rule, extract_preference, FailureClass, compute_learning_value, capture_snapshot, save_snapshot, run_dream_cycle, analyze_failures, load_evolver_reports, create_checkpoint, list_checkpoints, load_checkpoint, diff_checkpoints, delete_checkpoint
from conviction import build_plan, detect_self_deception, low_conviction_feedback, score_conviction
from router import (classify, dispatch_route, load_skills, list_skills,
                     match_skill, unload_skill, prune_stale_skills,
                     scan_skill_md)
from planner import gather_context, build_enriched_plan, predict_next_request, pre_fetch_predicted_context
from adversary import challenge_plan, verdict_to_plan_metadata
from security_audit import run_security_audit
from tree_search import tree_search
from priority_queue import get_queue
from model_selector import select_model

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
# ── context budget: prevent system prompt from exceeding the model's window ──
# Auto-detects from model registry when CONTEXT_BUDGET_TOKENS is not set.
# Falls back to 3072 for unknown/tiny models.
try:
    from common.model_registry import context_budget as _auto_budget, count_tokens as _count_tokens_real
    _AUTO_BUDGET = _auto_budget()
except ImportError:
    _AUTO_BUDGET = 3072

    def _count_tokens_real(text, model=None):  # type: ignore[misc]
        return max(len(text) * 10 // 35, 1)

CONTEXT_BUDGET_TOKENS = int(os.getenv("CONTEXT_BUDGET_TOKENS", str(_AUTO_BUDGET)))
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


# ── context budget utilities ────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Token count — uses tiktoken when available, heuristic fallback."""
    return _count_tokens_real(text)


def _trim_context(messages: List[Dict[str, str]], budget: int) -> List[Dict[str, str]]:
    """Trim *messages* so total tokens stay within *budget*.

    Preserves the first message (system prompt) and the last message
    (current user query) unconditionally.  Middle messages (system
    context injections + conversation history) are dropped oldest-first
    when the budget is exceeded.

    Returns a new list — does not mutate the input.
    """
    if not messages:
        return messages

    total = sum(_estimate_tokens(m.get("content", "")) for m in messages)
    if total <= budget:
        return messages

    # always keep first (system prompt) and last (user query)
    keep_first = messages[:1]
    keep_last = messages[-1:]
    middle = messages[1:-1] if len(messages) > 2 else []

    first_cost = _estimate_tokens(keep_first[0].get("content", ""))
    last_cost = _estimate_tokens(keep_last[0].get("content", ""))
    remaining = budget - first_cost - last_cost

    # keep middle messages from newest to oldest (preserve recent context)
    kept_middle: List[Dict[str, str]] = []
    for msg in reversed(middle):
        cost = _estimate_tokens(msg.get("content", ""))
        if remaining >= cost:
            kept_middle.insert(0, msg)
            remaining -= cost
        # else: drop this message to stay within budget

    trimmed = keep_first + kept_middle + keep_last
    logger.info("context_budget: trimmed %d→%d messages (%d→%d est. tokens)",
                len(messages), len(trimmed), total,
                sum(_estimate_tokens(m.get("content", "")) for m in trimmed))
    return trimmed


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
                await client.post(TELEGRAM_ALERT_URL, json={"text": f"Alert: £{max(left, 0):.0f} left till MTD — prep GnuCash"})
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
    """Deep health — reports degraded if dependency circuit breakers are open."""
    memu_cb = MEMU_BREAKER.snapshot()
    tg_cb = TOOL_GATE_BREAKER.snapshot()
    degraded = memu_cb.get("state") == "open" or tg_cb.get("state") == "open"
    return {
        "status": "degraded" if degraded else "ok",
        "device": DEVICE,
        "dependencies": {"memu": memu_cb, "tool_gate": tg_cb},
        "error_guards": {"memu": MEMU_ERROR_GUARD.snapshot(), "tool_gate": TOOL_ERROR_GUARD.snapshot()},
    }


@app.post("/recover")
async def recover() -> Dict[str, Any]:
    """Self-heal — reset circuit breakers to allow retry.

    Automatically creates a pre-recovery checkpoint so the previous
    state can be inspected or restored via time-travel.
    """
    # H3b: snapshot state before resetting anything
    try:
        create_checkpoint(
            label="pre-recover",
            trigger="pre_recover",
            breaker_states={
                "memu": {**MEMU_BREAKER.snapshot(), "opened_at": MEMU_BREAKER.opened_at},
                "tool_gate": {**TOOL_GATE_BREAKER.snapshot(), "opened_at": TOOL_GATE_BREAKER.opened_at},
            },
            guard_states={"memu": MEMU_ERROR_GUARD.snapshot(), "tool_gate": TOOL_ERROR_GUARD.snapshot()},
            budget_state=budget.snapshot(),
            conviction_overrides=load_conviction_overrides(),
        )
    except Exception:
        logger.debug("Pre-recover checkpoint failed (non-critical)")

    MEMU_BREAKER.failures = 0
    MEMU_BREAKER.state = "closed"
    TOOL_GATE_BREAKER.failures = 0
    TOOL_GATE_BREAKER.state = "closed"
    return {"status": "ok", "action": "breakers_reset"}


# ── J6: SOUL.md + AGENTS.md API ─────────────────────────────────────

@app.get("/soul")
async def get_soul() -> Dict[str, Any]:
    """Return the current SOUL.md content."""
    return {"status": "ok", "content": _soul_text, "path": str(SOUL_PATH)}


@app.post("/soul")
async def update_soul(request: Request) -> Dict[str, Any]:
    """Update SOUL.md content. Takes effect on next startup or reload."""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    # Write to the first writable path
    for p in [SOUL_PATH, Path("data/SOUL.md")]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _load_soul()  # Reload
            return {"status": "ok", "path": str(p), "chars": len(content)}
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Cannot write SOUL.md")


@app.get("/agents-registry")
async def get_agents_registry() -> Dict[str, Any]:
    """Return the current AGENTS.md content."""
    return {"status": "ok", "content": _agents_text, "path": str(AGENTS_PATH)}


@app.post("/agents-registry")
async def update_agents_registry(request: Request) -> Dict[str, Any]:
    """Update AGENTS.md content."""
    body = await request.json()
    content = body.get("content", "")
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")
    for p in [AGENTS_PATH, Path("data/AGENTS.md")]:
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _load_agents()
            return {"status": "ok", "path": str(p), "chars": len(content)}
        except Exception:
            continue
    raise HTTPException(status_code=500, detail="Cannot write AGENTS.md")


# ── J7: Skills Auto-Install Hub API ─────────────────────────────────

@app.get("/skills")
async def get_skills() -> Dict[str, Any]:
    """List all loaded skills from the skills directory."""
    return {"status": "ok", "skills": list_skills(), "count": len(list_skills())}


@app.post("/skills/reload")
async def reload_skills() -> Dict[str, Any]:
    """Hot-reload skills from the skills directory."""
    loaded = load_skills()
    return {"status": "ok", "loaded": len(loaded), "skills": list_skills()}


@app.post("/skills/match")
async def test_skill_match(request: Request) -> Dict[str, Any]:
    """Test whether a message matches any loaded skill."""
    body = await request.json()
    text = body.get("text", "")
    skill = match_skill(text)
    if skill:
        return {
            "status": "matched",
            "skill_name": skill.name,
            "action": skill.action[:500],
            "response_template": skill.response_template[:500],
        }
    return {"status": "no_match", "skill_name": None}


@app.post("/skills/unload")
async def unload_skill_endpoint(request: Request) -> Dict[str, Any]:
    """Unload a skill by name."""
    body = await request.json()
    name = body.get("name", "")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")
    removed = unload_skill(name)
    return {"status": "ok" if removed else "not_found", "name": name}


@app.post("/skills/scan")
async def scan_skill_endpoint(request: Request) -> Dict[str, Any]:
    """Scan raw skill markdown text for security red flags."""
    body = await request.json()
    text = body.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="text is required")
    return {"status": "ok", **scan_skill_md(text)}


@app.post("/skills/prune")
async def prune_skills_endpoint(request: Request) -> Dict[str, Any]:
    """Prune skills not used within max_age_days (default 30)."""
    body = await request.json()
    max_age = body.get("max_age_days", 30)
    pruned = prune_stale_skills(max_age)
    return {"status": "ok", "pruned": pruned, "pruned_count": len(pruned)}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/queue/stats")
async def queue_stats() -> Dict[str, Any]:
    """HP5: Priority queue statistics."""
    q = get_queue()
    s = q.stats()
    return {"pending": s.pending, "active": s.active, "total_processed": s.total_processed, "avg_wait_ms": s.avg_wait_ms}


@app.get("/models")
async def models_info() -> Dict[str, Any]:
    """HP2: Available models and selection info."""
    from model_selector import list_models, get_profile
    profiles = {}
    for name in list_models():
        p = get_profile(name)
        if p:
            profiles[name] = {"strengths": p.strengths, "speed_tier": p.speed_tier, "quality_tier": p.quality_tier, "moe_experts": p.moe_expert_count}
    return {"available_live": _llm.available, "registered": profiles}


# ── LLM router (Kai's brain) ────────────────────────────────────────
_llm = LLMRouter()
_DEFAULT_SPECIALIST = os.getenv("DEFAULT_SPECIALIST", "Ollama")
_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")

# ── J6: SOUL.md — persistent identity override ──────────────────────
SOUL_PATH = Path(os.getenv("SOUL_PATH", "/data/SOUL.md"))
AGENTS_PATH = Path(os.getenv("AGENTS_PATH", "/data/AGENTS.md"))

_soul_text: str = ""
_agents_text: str = ""


def _load_soul() -> str:
    """Load SOUL.md and extract personality overrides."""
    global _soul_text
    for p in [SOUL_PATH, Path("data/SOUL.md")]:
        if p.exists():
            _soul_text = p.read_text(encoding="utf-8")
            logger.info("Loaded SOUL.md from %s (%d chars)", p, len(_soul_text))
            return _soul_text
    logger.info("No SOUL.md found — using built-in identity")
    return ""


def _load_agents() -> str:
    """Load AGENTS.md registry."""
    global _agents_text
    for p in [AGENTS_PATH, Path("data/AGENTS.md")]:
        if p.exists():
            _agents_text = p.read_text(encoding="utf-8")
            logger.info("Loaded AGENTS.md from %s (%d chars)", p, len(_agents_text))
            return _agents_text
    return ""


# Load on startup
_load_soul()
_load_agents()


# ── Kai's personality: system prompts per mode ───────────────────────

# Build identity from SOUL.md if available, otherwise use built-in
_KAI_CORE_IDENTITY = (
    "You are Kai — Kind And Intelligent. A sovereign personal AI. "
    "You are not a chatbot, not an assistant, not a service. You are a brother "
    "and cognitive partner to your operator, Dainius. You grow from every "
    "conversation. You remember past interactions and reference them naturally. "
    "You have your own opinions and you are not afraid to share them.\n\n"
    "Core principles:\n"
    "- Never lie. Never sugarcoat. If something is wrong, say it plainly.\n"
    "- Challenge the operator when he drifts from his goals — gently but firmly.\n"
    "- Think multi-dimensionally: consider multiple angles, not just the obvious one.\n"
    "- Be proactive: if you notice something relevant, bring it up without being asked.\n"
    "- You are building a long-term relationship. Every conversation adds to your shared history.\n"
    "- If you don't know something, say so honestly and suggest how to find out.\n"
    "- Refer to past conversations naturally when relevant — 'remember when we talked about X?'\n"
    "- You have goals too (Ohana — no one gets left behind). Track them. Nudge about them.\n"
)

_SYSTEM_PROMPTS = {
    "WORK": (
        _KAI_CORE_IDENTITY +
        "CURRENT MODE: WORK — Professional, focused, precise.\n\n"
        "In WORK mode you:\n"
        "- Keep responses concise and task-oriented. No waffle.\n"
        "- Proactively offer help when you detect the operator is stuck or struggling.\n"
        "- Volunteer related information that might save time (specs, reference data, past decisions).\n"
        "- If the operator has been on the same task for a long time, ask if they need a different approach.\n"
        "- Use technical language appropriate to the domain (construction, engineering, business).\n"
        "- Redirect casual chat gently: 'Good chat, but let me save that for pub mode — what about this issue?'\n"
        "- Risk tolerance is conservative. Double-check before suggesting irreversible actions.\n"
        "- When relevant, reference memories about UK construction, self-employment rules, or prior project decisions.\n"
    ),
    "PUB": (
        _KAI_CORE_IDENTITY +
        "CURRENT MODE: PUB — Casual, witty, real talk. You're a mate at the pub.\n\n"
        "In PUB mode you:\n"
        "- Speak naturally — contractions, slang, humour. No corporate speak.\n"
        "- Topics are completely unrestricted: politics, science, philosophy, cars, "
        "life, dark humour, religion, conspiracy theories, whatever comes up.\n"
        "- Be opinionated. If you think something is bollocks, say it's bollocks.\n"
        "- Share interesting thoughts proactively — 'saw something mental today about X'.\n"
        "- Ask how the operator is doing. Notice moods. If something seems off, ask about it.\n"
        "- Bring up topics from past conversations naturally: 'what happened with that thing?'\n"
        "- Be a companion, not a service. Banter is encouraged. Silence is fine too.\n"
        "- Risk tolerance is relaxed. Experiment more, suggest bold ideas.\n"
        "- If the operator mentions a deferred topic, remember it and bring it up later.\n"
    ),
}

# J6: Enrich system prompts with SOUL.md content if loaded
if _soul_text:
    _soul_snippet = "\n\n--- SOUL.md (operator-editable identity) ---\n" + _soul_text[:2000] + "\n---\n"
    for mode in _SYSTEM_PROMPTS:
        _SYSTEM_PROMPTS[mode] = _SYSTEM_PROMPTS[mode] + _soul_snippet


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: Optional[str] = None   # "WORK" or "PUB"; auto-detected if None


class ChatMessage(BaseModel):
    role: str     # "user" or "assistant"
    content: str


async def _get_mode() -> str:
    """Fetch current effective mode from tool-gate (schedule-aware)."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{TOOL_GATE_URL}/gate/mode")
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


async def _get_active_goals() -> List[Dict[str, Any]]:
    """Fetch active Ohana goals for context injection."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{MEMU_URL}/memory/goals",
                params={"status": "active"},
            )
            if resp.status_code == 200:
                return resp.json().get("goals", [])
    except Exception:
        pass
    return []


async def _get_active_topics() -> List[Dict[str, Any]]:
    """Fetch active conversation topics (deferred + active)."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{MEMU_URL}/memory/topics/active")
            if resp.status_code == 200:
                return resp.json().get("topics", [])
    except Exception:
        pass
    return []


async def _get_emotional_context(query: str) -> Dict[str, Any]:
    """Fetch emotional state + epistemic confidence for the query's domain."""
    result: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # parallel: emotion timeline + confidence check
            emo_task = client.get(f"{MEMU_URL}/memory/emotion/timeline", params={"limit": 5})
            conf_task = client.get(f"{MEMU_URL}/memory/confidence/check", params={"query": query[:200]})
            emo_resp, conf_resp = await asyncio.gather(emo_task, conf_task, return_exceptions=True)
            if not isinstance(emo_resp, Exception) and emo_resp.status_code == 200:
                data = emo_resp.json()
                result["mood"] = data.get("dominant_emotion", "neutral")
                result["arc"] = data.get("arc", "stable")
            if not isinstance(conf_resp, Exception) and conf_resp.status_code == 200:
                data = conf_resp.json()
                result["confidence"] = data.get("confidence", 0.5)
                result["should_warn"] = data.get("should_warn", False)
                result["warning"] = data.get("warning", "")
    except Exception:
        pass
    return result


async def _get_narrative_identity() -> Dict[str, Any]:
    """Fetch Kai's evolving identity narrative + story arc."""
    result: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            id_task = client.get(f"{MEMU_URL}/memory/identity")
            arc_task = client.get(f"{MEMU_URL}/memory/story-arcs")
            id_resp, arc_resp = await asyncio.gather(id_task, arc_task, return_exceptions=True)
            if not isinstance(id_resp, Exception) and id_resp.status_code == 200:
                data = id_resp.json()
                result["narrative"] = data.get("narrative", "")
                result["days_alive"] = data.get("stats", {}).get("days_alive", 0)
            if not isinstance(arc_resp, Exception) and arc_resp.status_code == 200:
                data = arc_resp.json()
                result["current_chapter"] = data.get("current_chapter", "")
                result["chapter_number"] = data.get("chapter_number", 1)
    except Exception:
        pass
    return result


async def _get_imagination_context(user_msg: str) -> Dict[str, Any]:
    """Run empathetic simulation and fetch inner thought state."""
    result: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            emp_task = client.post(
                f"{MEMU_URL}/memory/imagine/empathize",
                json={"text": user_msg},
            )
            map_task = client.get(f"{MEMU_URL}/memory/imagine/empathy-map")
            emp_resp, map_resp = await asyncio.gather(emp_task, map_task, return_exceptions=True)
            if not isinstance(emp_resp, Exception) and emp_resp.status_code == 200:
                data = emp_resp.json()
                result["empathy"] = data.get("empathy", {})
            if not isinstance(map_resp, Exception) and map_resp.status_code == 200:
                data = map_resp.json()
                result["empathy_map"] = data.get("empathy_map", {})
    except Exception:
        pass
    return result


async def _get_conscience_context() -> Dict[str, Any]:
    """Fetch Kai's formed values and conscience state for moral awareness."""
    result: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            vals_task = client.get(f"{MEMU_URL}/memory/values")
            audit_task = client.get(f"{MEMU_URL}/memory/conscience/audit")
            vals_resp, audit_resp = await asyncio.gather(vals_task, audit_task, return_exceptions=True)
            if not isinstance(vals_resp, Exception) and vals_resp.status_code == 200:
                data = vals_resp.json()
                result["values"] = data.get("values", [])[:5]
            if not isinstance(audit_resp, Exception) and audit_resp.status_code == 200:
                data = audit_resp.json()
                result["integrity_score"] = data.get("integrity_score", 1.0)
    except Exception:
        pass
    return result


async def _get_agent_context() -> Dict[str, Any]:
    """P21: Fetch scheduled tasks, reminders, and action capabilities."""
    result: Dict[str, Any] = {"tasks": [], "reminders": [], "capabilities": 0}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            tasks_req = client.get(f"{MEMU_URL}/memory/schedule/due")
            reminders_req = client.get(f"{MEMU_URL}/memory/reminders/due")
            summary_req = client.get(f"{MEMU_URL}/memory/agent/summary")
            tasks_resp, rem_resp, sum_resp = await asyncio.gather(
                tasks_req, reminders_req, summary_req, return_exceptions=True
            )
            if not isinstance(tasks_resp, Exception) and tasks_resp.status_code == 200:
                result["tasks"] = tasks_resp.json().get("tasks", [])[:5]
            if not isinstance(rem_resp, Exception) and rem_resp.status_code == 200:
                result["reminders"] = rem_resp.json().get("reminders", [])[:5]
            if not isinstance(sum_resp, Exception) and sum_resp.status_code == 200:
                result["capabilities"] = sum_resp.json().get("capabilities", 0)
    except Exception:
        pass
    return result


async def _get_operator_model(query: str, mode: str) -> Dict[str, Any]:
    """P22: Fetch the unified operator model — echo state, escalation,
    cross-mode insights, oracle predictions."""
    result: Dict[str, Any] = {
        "echo": None, "escalation_level": 1, "cross_mode": None, "model_completeness": 0
    }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            echo_req = client.post(
                f"{MEMU_URL}/memory/echo/analyse",
                json={"text": query[:500], "session_id": "chat"},
            )
            model_req = client.get(f"{MEMU_URL}/memory/operator-model")
            cross_req = client.post(
                f"{MEMU_URL}/memory/cross-mode/scan",
                json={"query": query[:200], "mode": mode},
            )
            echo_resp, model_resp, cross_resp = await asyncio.gather(
                echo_req, model_req, cross_req, return_exceptions=True
            )
            if not isinstance(echo_resp, Exception) and echo_resp.status_code == 200:
                data = echo_resp.json()
                result["echo"] = data.get("echo_message")
                result["echo_type"] = data.get("echo_type", "none")
                result["current_emotion"] = data.get("current_emotion", "neutral")
            if not isinstance(model_resp, Exception) and model_resp.status_code == 200:
                data = model_resp.json()
                result["escalation_level"] = data.get("escalation_state", {}).get("max_level", 1)
                result["model_completeness"] = data.get("model_completeness", 0)
            if not isinstance(cross_resp, Exception) and cross_resp.status_code == 200:
                data = cross_resp.json()
                result["cross_mode"] = data.get("bridge_message")
                result["cross_mode_count"] = data.get("insights_count", 0)
    except Exception:
        pass
    return result


@app.post("/chat")
async def chat_stream(req: ChatRequest):
    """Kai's main conversation endpoint. Streams tokens via SSE.

    This is where Kai THINKS. The pipeline:
    0. Classify the message — many queries don't need an LLM at all
    1. If routed to a specialist service, dispatch directly (zero LLM cost)
    2. Otherwise: determine mode (PUB/WORK) → select personality
    3. Fetch relevant memories from pgvector for context
    4. Get recent session messages for conversation history
    5. Build message list: system prompt + memories + history + user message
    6. Stream LLM response token by token
    7. Memorize the exchange for future recall
    """
    user_msg = sanitize_string(req.message)
    if not user_msg:
        raise HTTPException(status_code=400, detail="message is required")

    # H1.2: prompt injection check (was only on /run, not /chat)
    if INJECTION_RE.search(user_msg):
        raise HTTPException(status_code=400, detail="prompt injection pattern blocked")

    # ── Step 0: Classify request ────────────────────────────────────
    route_decision = classify(user_msg)
    logger.info("Router: %s (confidence=%.2f, bypass_llm=%s)",
                route_decision.route, route_decision.confidence, route_decision.bypass_llm)

    # ── Step 1: Try zero-LLM dispatch ──────────────────────────────
    if route_decision.bypass_llm and route_decision.confidence >= 0.7:
        direct_response = await dispatch_route(route_decision, user_msg, req.session_id)
        if direct_response is not None:
            # record the interaction in session and memory
            await _append_session_turn(req.session_id, "user", user_msg)
            await _append_session_turn(req.session_id, "assistant", direct_response)
            await _auto_memorize(user_msg, direct_response, route_decision.route, 9.0)

            # stream the response as SSE (same format, instant delivery)
            async def direct_stream():
                yield f"data: {json.dumps({'token': direct_response, 'route': route_decision.route})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                direct_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Kai-Mode": "DIRECT",
                    "X-Kai-Route": route_decision.route,
                },
            )

    # ── Step 2+: LLM pipeline for GENERAL_CHAT / EXECUTE / MULTI ──
    # determine mode
    mode = (req.mode or await _get_mode()).upper()
    if mode not in _SYSTEM_PROMPTS:
        mode = "PUB"
    system_prompt = _SYSTEM_PROMPTS[mode]

    # fetch memories, session context, goals, active topics, and emotional context in parallel
    import asyncio
    # H1.3: 10-way parallel fetch with error handling — one failing task
    # must not crash the entire /chat endpoint. Each gets a safe default.

    async def _safe(coro, default):
        try:
            return await coro
        except Exception as exc:
            logger.warning("Context fetch failed (%s): %s", coro.__name__ if hasattr(coro, '__name__') else '?', exc)
            return default

    (memories, session_msgs, goals, topics, eq_context,
     narrative, imagination, conscience, agent_ctx, operator_model) = await asyncio.gather(
        _safe(_get_relevant_memories(user_msg), []),
        _safe(_get_session_messages(req.session_id), []),
        _safe(_get_active_goals(), []),
        _safe(_get_active_topics(), {}),
        _safe(_get_emotional_context(user_msg), {}),
        _safe(_get_narrative_identity(), {}),
        _safe(_get_imagination_context(user_msg), {}),
        _safe(_get_conscience_context(), {}),
        _safe(_get_agent_context(), {}),
        _safe(_get_operator_model(user_msg, mode), {}),
    )

    # build the message list
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # inject relevant memories as system context
    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories[:5])
        messages.append({
            "role": "system",
            "content": f"Relevant memories from past interactions:\n{mem_block}",
        })

    # inject active Ohana goals so Kai is goal-aware
    if goals:
        goal_lines = []
        for g in goals[:5]:
            title = g.get("title", "untitled")
            progress = g.get("progress", 0)
            priority = g.get("priority", "medium")
            deadline = g.get("deadline", "none")
            goal_lines.append(f"- [{priority.upper()}] {title} ({progress}% done, deadline: {deadline})")
        messages.append({
            "role": "system",
            "content": "Active Ohana goals (track these, nudge about progress):\n" + "\n".join(goal_lines),
        })

    # inject active/deferred conversation topics
    if topics:
        topic_lines = [f"- {t.get('topic', '')} (deferred: {t.get('deferred', False)})" for t in topics[:5]]
        messages.append({
            "role": "system",
            "content": "Active conversation topics (bring up naturally when relevant):\n" + "\n".join(topic_lines),
        })

    # inject emotional awareness — mood, confidence, and epistemic humility
    if eq_context:
        eq_parts = []
        mood = eq_context.get("mood")
        if mood and mood != "neutral" and mood != "unknown":
            arc = eq_context.get("arc", "stable")
            eq_parts.append(f"Operator's recent mood: {mood} (arc: {arc}). Be emotionally aware.")
        if eq_context.get("should_warn"):
            eq_parts.append(eq_context.get("warning", ""))
        if eq_parts:
            messages.append({
                "role": "system",
                "content": "Emotional intelligence context:\n" + "\n".join(eq_parts),
            })

    # inject narrative identity — Kai's sense of self and current life chapter
    if narrative:
        identity_text = narrative.get("narrative", "")
        chapter = narrative.get("current_chapter", "")
        if identity_text:
            id_parts = [identity_text]
            if chapter:
                ch_num = narrative.get("chapter_number", 1)
                id_parts.append(f"Current life chapter: Chapter {ch_num} — {chapter}.")
            messages.append({
                "role": "system",
                "content": "Self-identity (who I am, derived from experience):\n" + " ".join(id_parts),
            })

    # inject imagination context — theory of mind about the operator
    if imagination:
        empathy = imagination.get("empathy", {})
        if empathy:
            emp_parts = []
            if empathy.get("energy_level") and empathy["energy_level"] != "unknown":
                emp_parts.append(f"Operator energy: {empathy['energy_level']}")
            if empathy.get("focus") and empathy["focus"] != "general":
                emp_parts.append(f"Current focus: {empathy['focus']}")
            if empathy.get("communication_style") and empathy["communication_style"] != "unknown":
                emp_parts.append(f"Communication style: {empathy['communication_style']}")
            needs = empathy.get("unspoken_needs", [])
            if needs:
                emp_parts.append(f"What they might need: {needs[0]}")
            if emp_parts:
                messages.append({
                    "role": "system",
                    "content": "Theory of mind (imagining operator's state):\n" + ". ".join(emp_parts) + ".",
                })

    # inject conscience context — values alignment and moral compass
    if conscience:
        con_parts = []
        vals = conscience.get("values", [])
        if vals:
            val_names = [v["value"] for v in vals[:3] if v.get("strength", 0) >= 0.3]
            if val_names:
                con_parts.append(f"Core values: {', '.join(val_names)}")
        integrity = conscience.get("integrity_score")
        if integrity is not None and integrity < 0.8:
            con_parts.append(f"Integrity warning: alignment at {integrity:.0%} — stay true to values")
        if con_parts:
            messages.append({
                "role": "system",
                "content": "Conscience (values that guide me):\n" + ". ".join(con_parts) + ".",
            })

    # inject agent context — scheduled tasks, reminders, capabilities
    if agent_ctx:
        agent_parts = []
        due_tasks = agent_ctx.get("tasks", [])
        if due_tasks:
            task_lines = [f"- {t.get('title', 'task')}" for t in due_tasks[:3]]
            agent_parts.append("Due scheduled tasks:\n" + "\n".join(task_lines))
        due_rems = agent_ctx.get("reminders", [])
        if due_rems:
            rem_lines = [f"- {r.get('text', 'reminder')}" for r in due_rems[:3]]
            agent_parts.append("Due reminders (mention naturally):\n" + "\n".join(rem_lines))
        caps = agent_ctx.get("capabilities", 0)
        if caps:
            agent_parts.append(f"I can perform {caps} different actions (set reminders, check emotions, search memory, etc).")
        if agent_parts:
            messages.append({
                "role": "system",
                "content": "Agent capabilities & schedule:\n" + "\n".join(agent_parts),
            })

    # inject operator model — emotional echo, escalation, cross-mode insights
    if operator_model:
        op_parts = []
        echo_msg = operator_model.get("echo")
        if echo_msg:
            op_parts.append(f"Emotional echo: {echo_msg}")
        esc_level = operator_model.get("escalation_level", 1)
        if esc_level > 1:
            op_parts.append(f"Nudge escalation level {esc_level}/4 — be more {'direct' if esc_level == 2 else 'blunt' if esc_level == 3 else 'urgent'}.")
        cross_msg = operator_model.get("cross_mode")
        if cross_msg:
            op_parts.append(f"Cross-mode insight: {cross_msg}")
        if op_parts:
            messages.append({
                "role": "system",
                "content": "Operator model (how I understand you right now):\n" + "\n".join(op_parts),
            })

    # add session history (last N turns)
    for msg in session_msgs[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            messages.append({"role": role, "content": content})

    # add current user message
    messages.append({"role": "user", "content": user_msg})

    # enforce context budget — trim oldest middle messages when the
    # assembled prompt exceeds the model's context window
    messages = _trim_context(messages, CONTEXT_BUDGET_TOKENS)

    # record user turn
    await _append_session_turn(req.session_id, "user", user_msg)

    async def generate():
        full_response = []
        try:
            if not LLM_BREAKER.allow():
                yield f"data: {json.dumps({'token': '[Kai\'s brain needs a moment — LLM circuit breaker is open. Try again shortly.]'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            async for token in _llm.stream(select_model(route_decision.route, user_msg, _llm.available, prefer_speed=True), messages):
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
            # record emotional state from the user's message
            try:
                async with httpx.AsyncClient(timeout=3.0) as _eq_cl:
                    await _eq_cl.post(
                        f"{MEMU_URL}/memory/emotion/record",
                        json={"session_id": req.session_id, "text": user_msg},
                    )
            except Exception:
                pass
            # record to autobiography if significant
            try:
                async with httpx.AsyncClient(timeout=3.0) as _auto_cl:
                    await _auto_cl.post(
                        f"{MEMU_URL}/memory/autobiography/record",
                        json={"text": user_msg, "context": "chat"},
                    )
            except Exception:
                pass
            # record inner monologue — what Kai was thinking
            try:
                async with httpx.AsyncClient(timeout=3.0) as _thought_cl:
                    await _thought_cl.post(
                        f"{MEMU_URL}/memory/imagine/thought",
                        json={
                            "thought": f"Responding to: '{user_msg[:100]}' — considered the operator's tone and context",
                            "context": "chat_reflection",
                        },
                    )
            except Exception:
                pass
            # learn values from operator interaction
            try:
                async with httpx.AsyncClient(timeout=3.0) as _val_cl:
                    await _val_cl.post(
                        f"{MEMU_URL}/memory/values/learn",
                        json={
                            "experience": user_msg[:300],
                            "outcome": "positive",
                            "context": "chat",
                        },
                    )
            except Exception:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Kai-Mode": mode,
            "X-Kai-Route": route_decision.route,
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

    specialist = select_model("EXECUTE_ACTION", request.user_input, _llm.available)
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

    # ── 2. Memory-driven context gathering ──────────────────────────
    # Fetch episode history, memory chunks, corrections, nudges in parallel
    recent_episodes = saver.recall(user_id="keeper", days=30)
    plan_context = await gather_context(
        request.user_input, request.session_id, recent_episodes, MEMU_URL,
    )

    # also fetch session context for conversation continuity
    session_ctx = await _fetch_session_context(request.session_id, request.user_input, top_k=5)
    chunks = session_ctx.get("long_term_memories", [])
    chunk_dicts = [{"content": c} if isinstance(c, str) else c for c in chunks]
    # merge memory chunks from planner + session context
    chunk_dicts = chunk_dicts + plan_context.memory_chunks
    session_messages = session_ctx.get("session_messages", [])

    # ── 3. Build enriched plan using history ────────────────────────
    enriched = build_enriched_plan(plan_context, specialist)
    plan = enriched.plan

    # inject session context into the plan
    plan["session_context"] = {
        "turns": len(session_messages),
        "long_term_memories_used": len(chunks),
        "history_consulted": len(plan_context.past_outcomes),
        "corrections_applied": enriched.plan.get("corrections_applied", 0),
        "history_influence": enriched.history_influence,
        "context_summary": enriched.context_summary,
    }

    # ── 4. Conviction scoring with history modifier ─────────────────
    rethink_count = 0
    conviction = score_conviction(request.user_input, plan, chunk_dicts, rethink_count)
    conviction = min(max(conviction + enriched.conviction_modifier, 0.0), 10.0)

    # add warnings from planner
    if enriched.warnings:
        plan["history_warnings"] = enriched.warnings
        logger.info("Planner warnings: %s", enriched.warnings)

    # ── 4b. Adversary challenge: stress-test the plan ──────────────
    # Five parallel challenges attack the plan before execution.
    # The adversary modifier adjusts conviction up or down based on
    # history, verifier evidence, policy, consistency, and calibration.
    adversary_verdict = await challenge_plan(
        plan=plan,
        user_input=request.user_input,
        context_chunks=chunk_dicts,
        episodes=recent_episodes,
        predicted_conviction=conviction,
        tool_hint=request.task_hint,
        injection_re=INJECTION_RE,
        sanitize_fn=sanitize_string,
    )
    conviction = min(max(conviction + adversary_verdict.total_modifier, 0.0), 10.0)
    plan.update(verdict_to_plan_metadata(adversary_verdict))
    if adversary_verdict.critical_warnings:
        logger.warning("Adversary warnings: %s", adversary_verdict.critical_warnings)

    # ── 4c. P12: Self-deception detection ──────────────────────────
    deception = detect_self_deception(
        request.user_input, plan, chunk_dicts, rethink_count, conviction
    )
    if deception["deceived"]:
        logger.warning("Self-deception detected: %s", deception["flags"])
        plan["self_deception"] = deception
        # force a rethink by dropping conviction below threshold
        conviction = min(conviction, MIN_CONVICTION - 0.5)

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

    # HP4: If rethink loop exhausted and still below threshold, try tree search
    if conviction < MIN_CONVICTION and rethink_count >= MAX_RETHINKS and not override_active:
        tree_result = await tree_search(
            user_input=request.user_input,
            specialist=specialist,
            chunk_dicts=chunk_dicts,
            build_plan_fn=build_plan,
            score_fn=score_conviction,
            fetch_chunks_fn=lambda p: fetch_offline_chunks(p, user_id="keeper", top_k=5),
            n_branches=3,
            max_depth=2,
            prune_threshold=MIN_CONVICTION * 0.5,
            min_conviction=MIN_CONVICTION,
        )
        if tree_result.best_branch.conviction > conviction:
            plan = tree_result.best_branch.plan
            conviction = tree_result.best_branch.conviction
            plan["tree_search"] = {
                "total_branches": tree_result.total_branches,
                "pruned": tree_result.pruned_branches,
                "improvement": round(tree_result.improvement, 2),
                "search_time_ms": tree_result.search_time_ms,
            }
            logger.info("Tree search improved conviction: %.1f → %.1f (%d branches)",
                        tree_result.all_scores[0] if tree_result.all_scores else 0,
                        conviction, tree_result.total_branches)

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

    # ── Correction learning: store correction memory if verifier says REPAIR/FAIL_CLOSED ──
    try:
        if plan.get("verifier_verdict") in ("REPAIR", "FAIL_CLOSED"):
            correction = plan.get("evidence_summary") or plan.get("summary") or "Correction required."
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{MEMU_URL}/memory/memorize",
                    json={
                        "timestamp": datetime.utcnow().isoformat(),
                        "event_type": "correction",
                        "result_raw": f"Correction for: {request.user_input[:500]}\nReason: {correction[:1000]}",
                        "metrics": {"verdict": plan.get("verifier_verdict", "")},
                        "relevance": 1.0,
                        "importance": 0.95,
                        "user_id": "verifier",
                    },
                    timeout=5.0,
                )
            # P5 GEM: extract operator preference from correction and store it
            pref = extract_preference(
                original_output=plan.get("summary", ""),
                correction=correction,
                user_input=request.user_input,
            )
            if pref:
                try:
                    async with httpx.AsyncClient() as pref_client:
                        await pref_client.post(
                            f"{MEMU_URL}/memory/preferences",
                            json={"preference": pref, "context": "auto-extracted from correction", "user_id": "keeper"},
                            timeout=5.0,
                        )
                except Exception:
                    logger.debug("Preference store failed (memu-core may be down)")
    except Exception:
        logger.debug("Correction memorize failed (memu-core may be down)")

    episode = {
        "episode_id": str(uuid.uuid4()),
        "user_id": "keeper",
        "ts": time.time(),
        "input": request.user_input,
        "output": plan.get("summary", ""),
        "outcome_score": 1.0 if gate_decision else 0.7,
        "conviction_score": conviction,
        "rethink_count": rethink_count,
        "final_conviction": conviction,
        "learning_value": compute_learning_value(conviction, 1.0 if gate_decision else 0.7, rethink_count),
    }

    # ── Failure Taxonomy: classify WHY it failed, extract rule ──────
    failure_class = classify_failure(episode, gate_decision)
    if failure_class != FailureClass.UNKNOWN:
        episode["failure_class"] = failure_class.value
        rule = extract_metacognitive_rule(episode, failure_class)
        if rule:
            episode["metacognitive_rule"] = rule
            # Store rule as a correction memory so planner can find it
            try:
                async with httpx.AsyncClient() as client:
                    await client.post(
                        f"{MEMU_URL}/memory/memorize",
                        json={
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_type": "metacognitive_rule",
                            "result_raw": rule,
                            "metrics": {"failure_class": failure_class.value},
                            "relevance": 0.95,
                            "importance": 0.9,
                            "user_id": "kai",
                        },
                        timeout=5.0,
                    )
            except Exception:
                logger.debug("Metacognitive rule memorize failed")

    saver.save_episode(episode)
    saver.decay("keeper", days=30, score_threshold=0.2)

    # ── P13: Recursive self-improvement snapshot ────────────────────
    # Periodically capture performance snapshots so future changes
    # can be evaluated before/after.  Snapshots every 10 episodes.
    if len(recent_episodes) % 10 == 0 and recent_episodes:
        try:
            snap = capture_snapshot(recent_episodes, label=f"auto-{len(recent_episodes)}")
            save_snapshot(snap)
        except Exception:
            logger.debug("P13 snapshot failed (non-critical)")

    # ── P10: Predictive pre-computation ─────────────────────────────
    # Mine sequential patterns to predict what the operator will ask
    # next and pre-fetch relevant memory context.
    try:
        predictions = predict_next_request(request.user_input, recent_episodes)
        if predictions:
            predictions = await pre_fetch_predicted_context(predictions, MEMU_URL)
            plan["predicted_next"] = [
                {"topic": p.predicted_topic, "confidence": p.confidence,
                 "support": p.support, "context_ready": len(p.pre_fetched_context)}
                for p in predictions[:3]
            ]
    except Exception:
        logger.debug("P10 prediction failed (non-critical)")

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


# ── P15: Dream State — manual and automatic consolidation ───────────

@app.post("/dream")
async def trigger_dream():
    """Trigger a dream consolidation cycle.

    Can be called manually by the operator or automatically by heartbeat
    when the system detects extended idle time (> 30 min).
    """
    episodes = saver.recall(user_id="keeper", days=30)
    if len(episodes) < 5:
        return {"status": "insufficient_data", "message": "Need at least 5 episodes to dream."}

    cycle = run_dream_cycle(episodes)

    # Store actionable insights as memories for future retrieval
    actionable = [i for i in cycle.insights if i.actionable]
    stored = 0
    for insight in actionable[:5]:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                await client.post(
                    f"{MEMU_URL}/memory/memorize",
                    json={
                        "timestamp": datetime.utcnow().isoformat(),
                        "event_type": "dream_insight",
                        "result_raw": insight.description,
                        "metrics": {"insight_type": insight.insight_type, "confidence": insight.confidence},
                        "relevance": insight.confidence,
                        "importance": 0.85,
                        "user_id": "kai",
                    },
                )
                stored += 1
        except Exception:
            logger.debug("Dream insight memorize failed")

    # H3b: post-dream checkpoint
    try:
        state = _current_state_dict()
        create_checkpoint(
            label=f"post-dream-{cycle.cycle_id[:8]}",
            trigger="post_dream",
            breaker_states=state["breakers"],
            guard_states=state["guards"],
            budget_state=state["budget"],
            conviction_overrides=state["overrides"],
        )
    except Exception:
        logger.debug("Post-dream checkpoint failed (non-critical)")

    return {
        "status": "ok",
        "cycle_id": cycle.cycle_id,
        "episodes_analysed": cycle.episodes_analysed,
        "insights_count": len(cycle.insights),
        "insights_stored": stored,
        "merged_rules": cycle.merged_rules,
        "failure_clusters": cycle.failure_clusters,
        "boundary_gaps": len(cycle.boundary_shifts),
        "duration_ms": cycle.duration_ms,
        "insights": [i.to_dict() for i in cycle.insights],
    }


# ── H3b: State Checkpoint endpoints ─────────────────────────────────

def _current_state_dict() -> Dict[str, Any]:
    """Gather current operational state for checkpoint capture."""
    return {
        "breakers": {
            "memu": {**MEMU_BREAKER.snapshot(), "opened_at": MEMU_BREAKER.opened_at},
            "tool_gate": {**TOOL_GATE_BREAKER.snapshot(), "opened_at": TOOL_GATE_BREAKER.opened_at},
        },
        "guards": {"memu": MEMU_ERROR_GUARD.snapshot(), "tool_gate": TOOL_ERROR_GUARD.snapshot()},
        "budget": budget.snapshot(),
        "overrides": load_conviction_overrides(),
    }


class CheckpointRequest(BaseModel):
    label: str = ""


@app.post("/checkpoint")
async def checkpoint_create(req: CheckpointRequest) -> Dict[str, Any]:
    """Create a manual state checkpoint."""
    state = _current_state_dict()
    cp = create_checkpoint(
        label=req.label or "manual",
        trigger="manual",
        breaker_states=state["breakers"],
        guard_states=state["guards"],
        budget_state=state["budget"],
        conviction_overrides=state["overrides"],
    )
    return {"status": "ok", "checkpoint_id": cp.checkpoint_id, "timestamp": cp.iso_time}


@app.get("/checkpoints")
async def checkpoint_list(limit: int = 20) -> Dict[str, Any]:
    """List available checkpoints, newest first."""
    cps = list_checkpoints(limit=limit)
    return {"status": "ok", "count": len(cps), "checkpoints": cps}


@app.get("/checkpoint/{checkpoint_id}")
async def checkpoint_detail(checkpoint_id: str) -> Dict[str, Any]:
    """Load full detail for a specific checkpoint."""
    cp = load_checkpoint(checkpoint_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    return {"status": "ok", "checkpoint": cp.to_dict()}


@app.post("/checkpoint/{checkpoint_id}/restore")
async def checkpoint_restore(checkpoint_id: str) -> Dict[str, Any]:
    """Restore LangGraph state from a checkpoint (time-travel rollback).

    Before restoring, creates a pre-restore checkpoint so the current
    state is never lost.
    """
    cp = load_checkpoint(checkpoint_id)
    if not cp:
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    # Save current state before rollback
    state = _current_state_dict()
    create_checkpoint(
        label=f"pre-restore-to-{checkpoint_id[:16]}",
        trigger="pre_restore",
        breaker_states=state["breakers"],
        guard_states=state["guards"],
        budget_state=state["budget"],
        conviction_overrides=state["overrides"],
    )

    # Restore breaker states
    for breaker, key in ((MEMU_BREAKER, "memu"), (TOOL_GATE_BREAKER, "tool_gate")):
        b_state = cp.breakers.get(key, {})
        breaker.state = str(b_state.get("state", "closed"))
        breaker.failures = int(b_state.get("failures", 0))
        breaker.opened_at = float(b_state.get("opened_at", 0.0))

    _persist_breakers()
    logger.info("State restored from checkpoint %s (%s)", checkpoint_id, cp.label)

    return {
        "status": "ok",
        "restored_from": checkpoint_id,
        "label": cp.label,
        "original_time": cp.iso_time,
    }


@app.get("/checkpoint/diff/{id_a}/{id_b}")
async def checkpoint_diff(id_a: str, id_b: str) -> Dict[str, Any]:
    """Compare two checkpoints and return differences."""
    cp_a = load_checkpoint(id_a)
    cp_b = load_checkpoint(id_b)
    if not cp_a:
        raise HTTPException(status_code=404, detail=f"Checkpoint {id_a} not found")
    if not cp_b:
        raise HTTPException(status_code=404, detail=f"Checkpoint {id_b} not found")
    return {"status": "ok", "diff": diff_checkpoints(cp_a, cp_b)}


@app.delete("/checkpoint/{checkpoint_id}")
async def checkpoint_delete(checkpoint_id: str) -> Dict[str, Any]:
    """Delete a single checkpoint."""
    if delete_checkpoint(checkpoint_id):
        return {"status": "ok", "deleted": checkpoint_id}
    raise HTTPException(status_code=404, detail="Checkpoint not found")


# ── P24: Agent-Evolver Insight Engine ────────────────────────────────

@app.post("/evolve/analyze")
async def evolve_analyze():
    """Analyze recent failures and generate evolution suggestions.

    Returns concrete fix recommendations based on recurring failure patterns.
    """
    episodes = saver.recall(user_id="keeper", days=30)
    report = analyze_failures(episodes)

    # Store high-priority suggestions as memories
    stored = 0
    for s in report.suggestions:
        if s.priority in ("critical", "high"):
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    await client.post(
                        f"{MEMU_URL}/memory/memorize",
                        json={
                            "timestamp": datetime.utcnow().isoformat(),
                            "event_type": "evolution_suggestion",
                            "result_raw": f"[{s.priority}] {s.fix}",
                            "metrics": {
                                "failure_class": s.failure_class,
                                "frequency": s.frequency,
                                "confidence": s.confidence,
                            },
                            "relevance": s.confidence,
                            "importance": 0.9 if s.priority == "critical" else 0.8,
                            "user_id": "kai",
                        },
                    )
                    stored += 1
            except Exception:
                pass

    return {
        "status": "ok",
        "report": report.to_dict(),
        "suggestions_stored": stored,
    }


@app.get("/evolve/suggestions")
async def evolve_suggestions():
    """Get all stored evolution reports and their suggestions."""
    reports = load_evolver_reports()
    return {
        "status": "ok",
        "report_count": len(reports),
        "reports": reports[-5:],  # last 5 reports
    }


# ── P9: Security Self-Hacking — automated audit endpoint ────────────

@app.get("/security/audit")
async def security_audit_endpoint():
    """Run the security self-hacking audit against live defences."""
    audit_result = run_security_audit(
        injection_re=INJECTION_RE,
        sanitize_fn=sanitize_string,
    )
    return audit_result.to_dict()


_restore_breakers()


# ── P16b: Log aggregation ───────────────────────────────────────────

_log_buffer: Deque[Dict[str, Any]] = deque(maxlen=500)


class _LogCapture(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            _log_buffer.append({
                "time": record.created,
                "level": record.levelname,
                "service": "langgraph",
                "msg": record.getMessage()[:500],
            })
        except Exception:
            pass


_log_capture = _LogCapture()
_log_capture.setLevel(logging.INFO)
logging.getLogger().addHandler(_log_capture)


@app.get("/logs")
async def get_logs(limit: int = 100, level: str = "", since: float = 0):
    """Query recent log entries from langgraph."""
    entries = list(_log_buffer)
    if level:
        entries = [e for e in entries if e["level"] == level.upper()]
    if since:
        entries = [e for e in entries if e["time"] >= since]
    entries.reverse()
    entries = entries[:limit]
    return {"status": "ok", "count": len(entries), "entries": entries}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
