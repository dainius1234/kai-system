from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from common.auth import sign_gate_request, sign_gate_request_bundle
from common.feature_flags import is_enabled
from common.llm import LLMRouter, llm_warmup
from common.resilience import resilient_call as _resilient_call
from system_fsm import KaiEvent as SysEvent, fire as fsm_fire, current_state as fsm_state, fsm_snapshot
from teammates import load_teammates, list_teammates, build_teammate_context
from counterfactual import rehearse as rehearse_counterfactual, can_rehearse
from cognitive_fsm import CognitiveFSM, get_config as get_swarm_config
from swarm import SwarmContext, list_reputation, load_reputation, record_error as swarm_record_error, record_success as swarm_record_success, save_reputation
from swarm_stages import build_swarm_pipeline
from curiosity import idle_curiosity_tick, CURIOSITY_LOG
from common.runtime import AuditStream, CircuitBreaker, ErrorBudget, ErrorBudgetCircuitBreaker, INJECTION_RE, detect_device, sanitize_string, setup_json_logger
from common.self_emp_advisor import advise, load_expenses, load_income_total, thresholds
from kai_config import build_saver, classify_failure, extract_metacognitive_rule, extract_preference, FailureClass, compute_learning_value, capture_snapshot, save_snapshot, create_checkpoint, list_checkpoints, load_checkpoint, diff_checkpoints, delete_checkpoint
from conviction import build_plan, detect_self_deception, low_conviction_feedback, score_conviction, update_domain_confidence
from router import (RouteDecision, classify, dispatch_route, load_skills, list_skills,
                     match_skill, unload_skill, prune_stale_skills,
                     scan_skill_md)
from planner import gather_context, build_enriched_plan, predict_next_request, pre_fetch_predicted_context
from adversary import challenge_plan, verdict_to_plan_metadata
from tree_search import tree_search
from priority_queue import get_queue
from model_selector import select_model
from cognitive_fingerprint import collector as _fp_collector, quick_sample as _fp_quick_sample
from causal_world_model import get_causal_graph, CausalEdge, get_surprise_detector
from global_workspace import get_global_workspace, WorkspaceBid
from moral_core import get_ohana_core
from cortex import get_cortex
from trust_integration import gate_autonomous_action, get_trust_status, record_chat_response
from model_council import get_model_council
from web_scout import fetch as web_fetch, search as web_search, summarize as web_summarize
from service_watchdog import get_watchdog
from paper_trader import get_paper_trader
from trust_core import get_trust_core, TrustLevel
from market_data import get_market_data
from strategy_engine import get_strategy_engine
from market_intel import get_market_intel
from alpha_signals import get_alpha_signals
from opportunity_intel import get_opportunity_intel

logger = setup_json_logger("kai", os.getenv("LOG_PATH", "/tmp/kai.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)


class _CleanupTaskManager:
    """Tracks fire-and-forget background tasks so they survive client disconnect
    and SIGTERM.  Keeps strong references; tasks self-remove when done."""

    def __init__(self) -> None:
        self._tasks: set = set()

    def submit(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def drain(self, timeout: float = 30.0) -> None:
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=timeout)


_cleanup_mgr = _CleanupTaskManager()

app = FastAPI(title="Kai", version="0.5.0")
MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
TOOL_GATE_URL = os.getenv("TOOL_GATE_URL", "http://tool-gate:8000")
TELEGRAM_ALERT_URL = os.getenv("TELEGRAM_ALERT_URL", "http://perception-telegram:9000/alert")
WAKE_URL = os.getenv("WAKE_URL", "http://wake-service:8022")
LETTA_URL = os.getenv("LETTA_URL", "http://letta-agent:8062")
FINANCIAL_URL = os.getenv("FINANCIAL_URL", "http://financial-awareness:8063")
# Sensory services — world awareness channels (Layer 2 / D87)
WEATHER_URL = os.getenv("WEATHER_SERVICE_URL", "http://weather-service:8039")
AIRQUALITY_URL = os.getenv("AIRQUALITY_URL", "http://airquality-service:8042")
CALENDAR_URL = os.getenv("CALENDAR_SERVICE_URL", "http://calendar-service:8043")
DOCKER_WATCHER_URL = os.getenv("DOCKER_WATCHER_URL", "http://docker-watcher:8041")
SYSMETRICS_URL = os.getenv("SYSMETRICS_URL", "http://sysmetrics:8035")
EMAIL_READER_URL = os.getenv("EMAIL_READER_URL", "http://email-reader:8037")
NEWS_FEED_URL = os.getenv("NEWS_FEED_URL", "http://news-feed:8038")
GIT_WATCHER_URL = os.getenv("GIT_WATCHER_URL", "http://git-watcher:8044")
BROKER_URL = os.getenv("BROKER_URL", "http://broker-bridge:8034")
SKILL_HUNTER_URL = os.getenv("SKILL_HUNTER_URL", "http://skill-hunter:8045")
HOUSE_DOCTOR_URL = os.getenv("HOUSE_DOCTOR_URL", "http://house-doctor:8046")
VAULT_SYNC_URL = os.getenv("VAULT_SYNC_URL", "http://vault-sync:8047")
SCREEN_WATCHER_URL = os.getenv("SCREEN_WATCHER_URL", "http://screen-watcher:8036")
CLIPBOARD_SERVICE_URL = os.getenv("CLIPBOARD_SERVICE_URL", "http://clipboard-service:8024")
CORTEX_URL = os.getenv("CORTEX_URL", "http://cortex:8048")


async def _memu_get(path: str, params: dict | None = None, fallback: Any = None, timeout: float = 5.0) -> Any:
    """GET from memu-core with retry and circuit-breaker via resilient_call."""
    return await _resilient_call(
        "GET", f"{MEMU_URL}{path}",
        params=params, timeout=timeout, retries=2, backoff=0.4,
        fallback=fallback if fallback is not None else {},
        logger=logger,
    )


async def _memu_post(path: str, body: dict | None = None, fallback: Any = None, timeout: float = 5.0) -> Any:
    """POST to memu-core with retry and circuit-breaker via resilient_call."""
    return await _resilient_call(
        "POST", f"{MEMU_URL}{path}",
        json=body or {}, timeout=timeout, retries=2, backoff=0.4,
        fallback=fallback if fallback is not None else {},
        logger=logger,
    )


PROACTIVE_INTERVAL = max(int(os.getenv("PROACTIVE_INTERVAL_SECONDS", "300")), 10)
GAP_HUNT_THRESHOLD = int(os.getenv("GAP_HUNT_THRESHOLD", "3"))
WAKE_INTENT_COMMAND_THRESHOLD = float(os.getenv("WAKE_INTENT_COMMAND_THRESHOLD", "0.6"))
WAKE_INTENT_OVERRIDE_CONFIDENCE = float(os.getenv("WAKE_INTENT_OVERRIDE_CONFIDENCE", "0.7"))
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
BREAKER_STATE_PATH = Path(os.getenv("BREAKER_STATE_PATH", "/data/langgraph_breakers.json"))
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
    payload = await _memu_get("/memory/retrieve", params={"query": query, "user_id": user_id, "top_k": top_k}, fallback=[])
    return payload if isinstance(payload, list) else []


# ── session buffer + auto-memorize helpers ──────────────────────────

async def _append_session_turn(session_id: str, role: str, content: str) -> None:
    """Push a turn into memu-core's working memory (session buffer)."""
    await _memu_post(f"/session/{session_id}/append", {"role": role, "content": content}, timeout=3.0)


async def _fetch_session_context(session_id: str, query: str, top_k: int = 5) -> Dict[str, Any]:
    """Fetch combined working + long-term memory context from memu-core."""
    return await _memu_get(
        f"/session/{session_id}/context",
        params={"query": query, "top_k": top_k},
        fallback={"long_term_memories": [], "session_messages": [], "query": query},
    )


async def _auto_memorize(user_input: str, response_summary: str, specialist: str, conviction: float) -> None:
    """Write the Q&A exchange back to memu-core so vector search learns.

    This is the key feedback loop — every conversation becomes a memory
    that future queries can find. The system literally gets smarter
    with every interaction.
    """
    await _memu_post("/memory/memorize", {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "conversation",
        "result_raw": f"Q: {user_input[:500]}\nA: {response_summary[:1000]}",
        "metrics": {"specialist": specialist, "conviction": conviction},
        "relevance": min(conviction / 10.0, 1.0),
        "user_id": "keeper",
    })


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
                await client.post(TELEGRAM_ALERT_URL, json={"text": f"Heads up — you're £{max(left, 0):.0f} from your MTD. Worth lining up GnuCash."})
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
                await client.post(TELEGRAM_ALERT_URL, json={"text": f"I've been a bit off lately — my 7-day conviction average is {avg_score:.2f}/10. Might be worth checking what I've been getting wrong."})
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
                    "text": f"I'm running rough — {name} is {state}, error rate at {ratio:.1%}. I'm still here but might be a bit slow.",
                },
            )
    except Exception:
        logger.warning("Failed to deliver guard-state alert for %s", name)


async def _capture_snapshot_background(recent_episodes: List[Dict[str, Any]]) -> None:
    """P13 performance snapshot, run off the hot /run path.

    Scheduled via asyncio.create_task (fire-and-forget) so a slow or
    failing disk write never adds latency to the chat/run response.
    """
    try:
        label = f"auto-{len(recent_episodes)}"
        snap = await asyncio.to_thread(capture_snapshot, recent_episodes, label)
        await asyncio.to_thread(save_snapshot, snap)
    except Exception:
        logger.debug("P13 snapshot failed (non-critical)")


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


@app.get("/introspect/capabilities")
async def introspect_capabilities() -> Dict[str, Any]:
    """D88 M2: self-capability map — live understanding of what Kai can perceive and do."""
    from common.feature_flags import get_all_flags

    sensory_services = [
        ("weather", WEATHER_URL),
        ("airquality", AIRQUALITY_URL),
        ("calendar", CALENDAR_URL),
        ("docker_watcher", DOCKER_WATCHER_URL),
        ("sysmetrics", SYSMETRICS_URL),
        ("email_reader", EMAIL_READER_URL),
        ("news_feed", NEWS_FEED_URL),
        ("git_watcher", GIT_WATCHER_URL),
        ("broker", BROKER_URL),
        ("skill_hunter", SKILL_HUNTER_URL),
    ]

    async def _ping(name: str, url: str) -> Dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{url}/health")
                return {"name": name, "reachable": r.status_code < 400, "http_status": r.status_code}
        except Exception:
            return {"name": name, "reachable": False, "http_status": 0}

    pings = await asyncio.gather(*[_ping(n, u) for n, u in sensory_services])
    skills = list_skills()
    return {
        "status": "ok",
        "sensory_services": list(pings),
        "reachable_count": sum(1 for p in pings if p["reachable"]),
        "unreachable_count": sum(1 for p in pings if not p["reachable"]),
        "skills": [{"name": s.name} for s in skills],
        "skill_count": len(skills),
        "feature_flags": {f["flag"]: f["enabled"] for f in get_all_flags()},
        "baselines_tracked": list(_sensor_baselines.keys()),
        "observation_history_depth": len(_observation_history),
        # D89 additions
        "fsm": fsm_snapshot(),
        "teammates": list_teammates(),
        "counterfactual_available": await can_rehearse(),
        "gap_log_top5": dict(_gap_log.most_common(5)),
        "proposed_rituals": list(_proposed_rituals),
        "trust": get_trust_status(),
        "model_council": get_model_council().status() if is_enabled("MODEL_COUNCIL") else None,
        "web_scout": is_enabled("WEB_SCOUT"),
        "service_watchdog": get_watchdog().status() if is_enabled("SERVICE_WATCHDOG") else None,
        "paper_trading": get_paper_trader().status() if is_enabled("PAPER_TRADING") else None,
        "market_data": get_market_data().status() if is_enabled("MARKET_DATA") else None,
        "strategy_engine": get_strategy_engine().status() if is_enabled("STRATEGY_ENGINE") else None,
        "market_intel": get_market_intel().status() if is_enabled("MARKET_INTEL") else None,
        "alpha_signals": get_alpha_signals().status() if is_enabled("ALPHA_SIGNALS") else None,
        "opportunity_intel": get_opportunity_intel().status() if is_enabled("OPPORTUNITY_INTEL") else None,
    }


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.get("/queue/stats")
async def queue_stats() -> Dict[str, Any]:
    """HP5: Priority queue statistics."""
    q = get_queue()
    s = q.stats()
    return {"pending": s.pending, "active": s.active, "total_processed": s.total_processed, "avg_wait_ms": s.avg_wait_ms}


@app.get("/teammates")
async def get_teammates() -> Dict[str, Any]:
    """D89: List all loaded persistent teammates."""
    return {"teammates": list_teammates(), "count": len(list_teammates())}


class TeammateRequest(BaseModel):
    message: str
    session_id: str = ""
    world_context: bool = True


@app.post("/chat/teammate/{name}")
async def chat_with_teammate(name: str, req: TeammateRequest) -> Dict[str, Any]:
    """D89: Route a query to a named teammate (Scout, Doctor, Sage, Oracle).

    Injects the teammate's system prompt + current world state into the LLM call.
    """
    if not is_enabled("PERSISTENT_TEAMMATES"):
        raise HTTPException(status_code=503, detail="FF_PERSISTENT_TEAMMATES is disabled")
    teammate_ctx = build_teammate_context(name)
    if teammate_ctx is None:
        raise HTTPException(status_code=404, detail=f"Teammate '{name}' not found. Available: {[t['slug'] for t in list_teammates()]}")

    world_state_block = ""
    if name == "auditor":
        trust_data = get_trust_status()
        world_state_block = "\n\nCurrent trust state:\n" + json.dumps(trust_data, indent=2)
    elif req.world_context and _last_world_snapshot:
        world_state_block = "\n\nCurrent world state:\n" + json.dumps(_last_world_snapshot, indent=2)[:800]

    prompt = f"{teammate_ctx}{world_state_block}\n\n---\n\nQuery: {req.message}"
    try:
        response = await _llm.chat([{"role": "user", "content": prompt}])
        return {
            "teammate": name,
            "response": response,
            "message": req.message,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Teammate invocation failed: {exc}")


class SwarmRequest(BaseModel):
    query: str
    swarm_type: str = "default"
    session_id: str = ""


@app.post("/chat/swarm")
async def chat_swarm(req: SwarmRequest) -> Dict[str, Any]:
    """D90: Run a query through the full CognitiveFSM swarm pipeline.

    Returns the final conviction score, pipeline transition log, swarm context
    summary, and adversary recommendation.  Feature-flagged under FF_SWARM.
    """
    if not is_enabled("SWARM"):
        raise HTTPException(status_code=503, detail="FF_SWARM is disabled")

    session_id = req.session_id or str(uuid.uuid4())
    ctx = SwarmContext(
        query=req.query,
        session_id=session_id,
        swarm_type=req.swarm_type,
    )

    pipeline = build_swarm_pipeline(
        memories_fn=_recall_memories,
        world_ctx_fn=_sense_world,
        teammate_ctx_fn=build_teammate_context,
        llm_chat_fn=_llm.chat,
        build_plan_fn=build_plan,
        score_fn=score_conviction,
        adversary_fn=challenge_plan,
    )

    cfg = get_swarm_config(req.swarm_type)
    fsm = CognitiveFSM(config=cfg)

    initial_payload: Dict[str, Any] = {"_ctx": ctx, "query": req.query}

    result = await fsm.run(
        gather_fn=pipeline["gather_fn"],
        debate_fn=pipeline["debate_fn"],
        fact_check_fn=pipeline["fact_check_fn"],
        causal_check_fn=pipeline["causal_check_fn"],
        conviction_gate_fn=pipeline["conviction_gate_fn"],
        moral_imagination_fn=pipeline["moral_imagination_fn"] if is_enabled("MORAL_IMAGINATION") else None,
        initial_payload=initial_payload,
    )

    save_reputation()

    final_confidence = result.final_handoff.confidence if result.final_handoff else 0.0
    return {
        "session_id": session_id,
        "swarm_type": req.swarm_type,
        "halted": result.halted,
        "halt_reason": result.halt_reason,
        "final_state": result.final_state.value,
        "conviction_score": final_confidence,
        "conviction_threshold": cfg.conviction_threshold,
        "passed": not result.halted and final_confidence >= cfg.conviction_threshold,
        "total_elapsed_ms": round(result.total_elapsed_ms, 1),
        "transition_log": result.transition_log,
        "context_summary": ctx.summary(),
        "adversary_recommendation": (result.final_handoff.payload or {}).get("adversary_recommendation", "unknown"),
    }


@app.get("/swarm/reputation")
async def swarm_reputation() -> Dict[str, Any]:
    """D90: Return per-teammate reputation weights."""
    return {"teammates": list_reputation()}


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


@app.get("/model-council/status")
async def model_council_status() -> Dict[str, Any]:
    """D122: Model Council status — registered models and availability."""
    if not is_enabled("MODEL_COUNCIL"):
        raise HTTPException(status_code=503, detail="FF_MODEL_COUNCIL is disabled")
    council = get_model_council()
    return {"status": "ok", **council.status(), "registry": council.discover()}


@app.get("/model-council/recommend")
async def model_council_recommend(task_type: str = "chat") -> Dict[str, Any]:
    """D122: Recommend best available model for a task type."""
    if not is_enabled("MODEL_COUNCIL"):
        raise HTTPException(status_code=503, detail="FF_MODEL_COUNCIL is disabled")
    council = get_model_council()
    ranked = council.rank(task_type=task_type)
    rec = council.recommend(task_type=task_type)
    return {
        "task_type": task_type,
        "recommendation": rec,
        "ranked": ranked,
    }


class ModelBenchmarkRequest(BaseModel):
    model_id: str
    task_type: str = "chat"


@app.post("/model-council/benchmark")
async def model_council_benchmark(req: ModelBenchmarkRequest) -> Dict[str, Any]:
    """D122: Run a benchmark probe for a specific model and task type."""
    if not is_enabled("MODEL_COUNCIL"):
        raise HTTPException(status_code=503, detail="FF_MODEL_COUNCIL is disabled")
    council = get_model_council()
    result = await asyncio.to_thread(council.benchmark, req.model_id, req.task_type)
    return result


class WebScoutFetchRequest(BaseModel):
    url: str
    max_chars: int = 4000


class WebScoutSearchRequest(BaseModel):
    query: str
    max_results: int = 5


@app.post("/web-scout/fetch")
async def web_scout_fetch(req: WebScoutFetchRequest) -> Dict[str, Any]:
    """D123: Fetch a URL and return extracted visible text."""
    if not is_enabled("WEB_SCOUT"):
        raise HTTPException(status_code=503, detail="FF_WEB_SCOUT is disabled")
    result = await asyncio.to_thread(web_fetch, req.url, max_chars=req.max_chars)
    return result.to_dict()


@app.post("/web-scout/search")
async def web_scout_search(req: WebScoutSearchRequest) -> Dict[str, Any]:
    """D123: Search via DuckDuckGo Instant Answers and return results."""
    if not is_enabled("WEB_SCOUT"):
        raise HTTPException(status_code=503, detail="FF_WEB_SCOUT is disabled")
    result = await asyncio.to_thread(web_search, req.query, req.max_results)
    return result.to_dict()


@app.post("/web-scout/summarize")
async def web_scout_summarize(req: WebScoutFetchRequest) -> Dict[str, Any]:
    """D123: Fetch a URL and return a trimmed summary."""
    if not is_enabled("WEB_SCOUT"):
        raise HTTPException(status_code=503, detail="FF_WEB_SCOUT is disabled")
    return await asyncio.to_thread(web_summarize, req.url, req.max_chars)


@app.get("/watchdog/status")
async def watchdog_status() -> Dict[str, Any]:
    """D124: Service Watchdog — last health check results for all services."""
    if not is_enabled("SERVICE_WATCHDOG"):
        raise HTTPException(status_code=503, detail="FF_SERVICE_WATCHDOG is disabled")
    return {"status": "ok", **get_watchdog().status()}


@app.post("/watchdog/check")
async def watchdog_check() -> Dict[str, Any]:
    """D124: Trigger an immediate health check of all services."""
    if not is_enabled("SERVICE_WATCHDOG"):
        raise HTTPException(status_code=503, detail="FF_SERVICE_WATCHDOG is disabled")
    results, fsm_events = await asyncio.to_thread(get_watchdog().check_all)
    for evt_name in fsm_events:
        try:
            await fsm_fire(SysEvent(evt_name))
        except Exception:
            pass
    return {
        "checked": len(results),
        "healthy": sum(1 for r in results if r.healthy),
        "fsm_events_fired": fsm_events,
        "services": [r.to_dict() for r in results],
    }


class PaperOpenRequest(BaseModel):
    symbol: str
    side: str           # "long" | "short"
    quantity: float
    price: float
    strategy_tag: str = ""


class PaperCloseRequest(BaseModel):
    position_id: str
    price: float


@app.get("/paper-trading/status")
async def paper_trading_status() -> Dict[str, Any]:
    """D125: Paper trading overall P&L and win-rate summary."""
    if not is_enabled("PAPER_TRADING"):
        raise HTTPException(status_code=503, detail="FF_PAPER_TRADING is disabled")
    return {"status": "ok", **get_paper_trader().status()}


@app.get("/paper-trading/positions")
async def paper_trading_positions() -> Dict[str, Any]:
    """D125: List all open simulated positions."""
    if not is_enabled("PAPER_TRADING"):
        raise HTTPException(status_code=503, detail="FF_PAPER_TRADING is disabled")
    return {"positions": get_paper_trader().get_positions()}


@app.get("/paper-trading/trades")
async def paper_trading_trades(limit: int = 50) -> Dict[str, Any]:
    """D125: List recent closed simulated trades."""
    if not is_enabled("PAPER_TRADING"):
        raise HTTPException(status_code=503, detail="FF_PAPER_TRADING is disabled")
    return {"trades": get_paper_trader().get_trades(limit=limit)}


@app.post("/paper-trading/open")
async def paper_trading_open(req: PaperOpenRequest) -> Dict[str, Any]:
    """D125: Open a simulated position. Trust: PARTNER (4)."""
    if not is_enabled("PAPER_TRADING"):
        raise HTTPException(status_code=503, detail="FF_PAPER_TRADING is disabled")
    try:
        pos = await asyncio.to_thread(
            get_paper_trader().open_position,
            req.symbol, req.side, req.quantity, req.price, req.strategy_tag,
        )
        return {"position": pos.to_dict()}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/paper-trading/close")
async def paper_trading_close(req: PaperCloseRequest) -> Dict[str, Any]:
    """D125: Close a simulated position and record realised P&L. Trust: PARTNER (4)."""
    if not is_enabled("PAPER_TRADING"):
        raise HTTPException(status_code=503, detail="FF_PAPER_TRADING is disabled")
    try:
        trade = await asyncio.to_thread(
            get_paper_trader().close_position, req.position_id, req.price,
        )
        return {"trade": trade.to_dict()}
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── D126: Trust Promotion Gate ───────────────────────────────────────

class TrustPromoteRequest(BaseModel):
    level: int          # TrustLevel int value (0–6)
    reason: str = ""

class TrustDemoteRequest(BaseModel):
    level: int
    reason: str


@app.get("/trust/status")
async def trust_status_endpoint() -> Dict[str, Any]:
    """D126: Full trust status with scores and progress to next level."""
    return get_trust_core().status()


@app.get("/trust/readiness")
async def trust_readiness() -> Dict[str, Any]:
    """D126: Promotion readiness report — gaps and auto-eligibility for next level."""
    return get_trust_core().promotion_readiness()


@app.post("/trust/promote")
async def trust_promote(req: TrustPromoteRequest) -> Dict[str, Any]:
    """D126: Operator grants a trust level. Dainius's word is final."""
    try:
        level = TrustLevel(req.level)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid trust level: {req.level}")
    await asyncio.to_thread(get_trust_core().grant, level, "dainius")
    return {"granted": level.name, "level": level.value}


@app.post("/trust/demote")
async def trust_demote(req: TrustDemoteRequest) -> Dict[str, Any]:
    """D126: Operator revokes trust to a specific level."""
    try:
        level = TrustLevel(req.level)
    except ValueError:
        raise HTTPException(status_code=422, detail=f"Invalid trust level: {req.level}")
    await asyncio.to_thread(get_trust_core().revoke, level, req.reason, "dainius")
    return {"revoked_to": level.name, "level": level.value, "reason": req.reason}


@app.get("/trust/audit")
async def trust_audit(limit: int = 20) -> Dict[str, Any]:
    """D126: Recent trust audit log entries."""
    return {"events": get_trust_core().audit_tail(limit)}


# ── D127: Market Data Feed ────────────────────────────────────────────

@app.get("/market-data/symbols")
async def market_data_symbols() -> Dict[str, Any]:
    """D127: List symbols Kai can fetch prices for."""
    if not is_enabled("MARKET_DATA"):
        raise HTTPException(status_code=503, detail="FF_MARKET_DATA is disabled")
    return {"symbols": get_market_data().known_symbols()}


@app.get("/market-data/prices")
async def market_data_prices(symbols: str = "") -> Dict[str, Any]:
    """D127: Fetch current USD prices for comma-separated symbols."""
    if not is_enabled("MARKET_DATA"):
        raise HTTPException(status_code=503, detail="FF_MARKET_DATA is disabled")
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    if not sym_list:
        raise HTTPException(status_code=422, detail="symbols query param required")
    prices = await asyncio.to_thread(get_market_data().get_prices, sym_list)
    return {"prices": prices}


@app.get("/market-data/status")
async def market_data_status() -> Dict[str, Any]:
    """D127: Market data cache status."""
    if not is_enabled("MARKET_DATA"):
        raise HTTPException(status_code=503, detail="FF_MARKET_DATA is disabled")
    return get_market_data().status()


@app.post("/market-data/mark")
async def market_data_mark() -> Dict[str, Any]:
    """D127: Mark all open paper positions to market. Returns {position_id: unrealised_pnl}."""
    if not is_enabled("MARKET_DATA"):
        raise HTTPException(status_code=503, detail="FF_MARKET_DATA is disabled")
    result = await asyncio.to_thread(get_market_data().mark_positions)
    return {"marked": result}


# ── D128: Strategy Engine ─────────────────────────────────────────────

class StrategyEvalRequest(BaseModel):
    symbol: str
    prices: List[float]

class StrategyAutoTradeRequest(BaseModel):
    symbol: str
    prices: List[float]
    quantity: float = 1.0
    strategy_tag: str = "auto"


@app.get("/strategy/status")
async def strategy_status() -> Dict[str, Any]:
    """D128: Strategy engine status — active strategy names."""
    if not is_enabled("STRATEGY_ENGINE"):
        raise HTTPException(status_code=503, detail="FF_STRATEGY_ENGINE is disabled")
    return get_strategy_engine().status()


@app.post("/strategy/evaluate")
async def strategy_evaluate(req: StrategyEvalRequest) -> Dict[str, Any]:
    """D128: Run all strategies against a price series. Trust: OBSERVER."""
    if not is_enabled("STRATEGY_ENGINE"):
        raise HTTPException(status_code=503, detail="FF_STRATEGY_ENGINE is disabled")
    if len(req.prices) < 2:
        raise HTTPException(status_code=422, detail="prices must have at least 2 values")
    signals = await asyncio.to_thread(
        get_strategy_engine().evaluate, req.symbol, req.prices
    )
    return {"signals": [s.to_dict() for s in signals]}


@app.post("/strategy/consensus")
async def strategy_consensus(req: StrategyEvalRequest) -> Dict[str, Any]:
    """D128: Return the majority-vote consensus signal. Trust: OBSERVER."""
    if not is_enabled("STRATEGY_ENGINE"):
        raise HTTPException(status_code=503, detail="FF_STRATEGY_ENGINE is disabled")
    if len(req.prices) < 2:
        raise HTTPException(status_code=422, detail="prices must have at least 2 values")
    signal = await asyncio.to_thread(
        get_strategy_engine().consensus, req.symbol, req.prices
    )
    return {"signal": signal.to_dict()}


@app.post("/strategy/propose")
async def strategy_propose(req: StrategyAutoTradeRequest) -> Dict[str, Any]:
    """D128: Consensus → ActionProposal (no execution). Trust: OBSERVER (1).

    Returns a proposal dict. The caller must route through Workspace →
    Policy → Capability before any execution. UH-INV-02 compliant.
    """
    if not is_enabled("STRATEGY_ENGINE"):
        raise HTTPException(status_code=503, detail="FF_STRATEGY_ENGINE is disabled")
    if len(req.prices) < 2:
        raise HTTPException(status_code=422, detail="prices must have at least 2 values")
    result = await asyncio.to_thread(
        get_strategy_engine().generate_proposal,
        req.symbol, req.prices, req.quantity, req.strategy_tag,
    )
    return result


# ── D129: Market Intelligence ─────────────────────────────────────────

@app.get("/market-intel/fear-greed")
async def market_intel_fear_greed() -> Dict[str, Any]:
    """D129: Crypto Fear & Greed Index (Alternative.me)."""
    if not is_enabled("MARKET_INTEL"):
        raise HTTPException(status_code=503, detail="FF_MARKET_INTEL is disabled")
    reading = await asyncio.to_thread(get_market_intel().get_fear_greed)
    if reading is None:
        raise HTTPException(status_code=503, detail="Fear & Greed data unavailable")
    return reading.to_dict()


@app.get("/market-intel/global")
async def market_intel_global() -> Dict[str, Any]:
    """D129: Global crypto market stats — BTC dominance, cap, trend (CoinGecko)."""
    if not is_enabled("MARKET_INTEL"):
        raise HTTPException(status_code=503, detail="FF_MARKET_INTEL is disabled")
    stats = await asyncio.to_thread(get_market_intel().get_global_stats)
    if stats is None:
        raise HTTPException(status_code=503, detail="Global stats unavailable")
    return stats.to_dict()


@app.get("/market-intel/trending")
async def market_intel_trending() -> Dict[str, Any]:
    """D129: Top trending coins on CoinGecko."""
    if not is_enabled("MARKET_INTEL"):
        raise HTTPException(status_code=503, detail="FF_MARKET_INTEL is disabled")
    coins = await asyncio.to_thread(get_market_intel().get_trending)
    return {"trending": [c.to_dict() for c in coins]}


@app.get("/market-intel/macro")
async def market_intel_macro() -> Dict[str, Any]:
    """D129: Macro context — gold, oil, DXY, Fed, geopolitical sentiment."""
    if not is_enabled("MARKET_INTEL"):
        raise HTTPException(status_code=503, detail="FF_MARKET_INTEL is disabled")
    return await asyncio.to_thread(get_market_intel().get_macro_context)


@app.get("/market-intel/context/{symbol}")
async def market_intel_context(symbol: str) -> Dict[str, Any]:
    """D129: Full intelligence context for a symbol — all feeds combined."""
    if not is_enabled("MARKET_INTEL"):
        raise HTTPException(status_code=503, detail="FF_MARKET_INTEL is disabled")
    return await asyncio.to_thread(get_market_intel().context, symbol.upper())


@app.get("/market-intel/status")
async def market_intel_status() -> Dict[str, Any]:
    """D129: Market intelligence cache status."""
    if not is_enabled("MARKET_INTEL"):
        raise HTTPException(status_code=503, detail="FF_MARKET_INTEL is disabled")
    return get_market_intel().status()


# ── Alpha Signals (D130) ─────────────────────────────────────────────

@app.get("/alpha/{symbol}/funding")
async def alpha_funding(symbol: str) -> Dict[str, Any]:
    """D130: Funding rate — cost of leverage and crowd positioning."""
    if not is_enabled("ALPHA_SIGNALS"):
        raise HTTPException(status_code=503, detail="FF_ALPHA_SIGNALS is disabled")
    result = await asyncio.to_thread(get_alpha_signals().get_funding_rate, symbol.upper())
    return result.to_dict() if result else {"symbol": symbol.upper(), "data": None}


@app.get("/alpha/{symbol}/open-interest")
async def alpha_open_interest(symbol: str) -> Dict[str, Any]:
    """D130: Open interest — total leverage magnitude in the market."""
    if not is_enabled("ALPHA_SIGNALS"):
        raise HTTPException(status_code=503, detail="FF_ALPHA_SIGNALS is disabled")
    result = await asyncio.to_thread(get_alpha_signals().get_open_interest, symbol.upper())
    return result.to_dict() if result else {"symbol": symbol.upper(), "data": None}


@app.get("/alpha/{symbol}/long-short")
async def alpha_long_short(symbol: str, period: str = "1h") -> Dict[str, Any]:
    """D130: Long/short account ratio — retail crowd positioning."""
    if not is_enabled("ALPHA_SIGNALS"):
        raise HTTPException(status_code=503, detail="FF_ALPHA_SIGNALS is disabled")
    result = await asyncio.to_thread(
        get_alpha_signals().get_long_short_ratio, symbol.upper(), period
    )
    return result.to_dict() if result else {"symbol": symbol.upper(), "data": None}


@app.get("/alpha/{symbol}/mark-premium")
async def alpha_mark_premium(symbol: str) -> Dict[str, Any]:
    """D130: Mark price vs spot index — basis / carry signal."""
    if not is_enabled("ALPHA_SIGNALS"):
        raise HTTPException(status_code=503, detail="FF_ALPHA_SIGNALS is disabled")
    result = await asyncio.to_thread(get_alpha_signals().get_mark_premium, symbol.upper())
    return result.to_dict() if result else {"symbol": symbol.upper(), "data": None}


@app.get("/alpha/{symbol}/composite")
async def alpha_composite(symbol: str) -> Dict[str, Any]:
    """D130: All four alpha signals combined — full professional context."""
    if not is_enabled("ALPHA_SIGNALS"):
        raise HTTPException(status_code=503, detail="FF_ALPHA_SIGNALS is disabled")
    return await asyncio.to_thread(get_alpha_signals().composite, symbol.upper())


@app.get("/alpha/status")
async def alpha_status() -> Dict[str, Any]:
    """D130: Alpha signal feed cache status."""
    if not is_enabled("ALPHA_SIGNALS"):
        raise HTTPException(status_code=503, detail="FF_ALPHA_SIGNALS is disabled")
    return get_alpha_signals().status()


# ── Opportunity Intelligence (D130) ─────────────────────────────────

@app.get("/opportunity/{symbol}/financial")
async def opportunity_financial(symbol: str) -> Dict[str, Any]:
    """D130: Financial opportunity score — conviction, direction, evidence."""
    if not is_enabled("OPPORTUNITY_INTEL"):
        raise HTTPException(status_code=503, detail="FF_OPPORTUNITY_INTEL is disabled")
    result = await asyncio.to_thread(get_opportunity_intel().scan_financial, symbol.upper())
    return result.to_dict()


@app.get("/opportunity/{symbol}/trend-arb")
async def opportunity_trend_arb(symbol: str) -> Dict[str, Any]:
    """D130: Cross-market trend arbitrage — macro alignment score."""
    if not is_enabled("OPPORTUNITY_INTEL"):
        raise HTTPException(status_code=503, detail="FF_OPPORTUNITY_INTEL is disabled")
    result = await asyncio.to_thread(get_opportunity_intel().scan_trend_arb, symbol.upper())
    return result.to_dict()


@app.get("/opportunity/content")
async def opportunity_content(topic: str) -> Dict[str, Any]:
    """D130: Content creation opportunity — topic conviction and recommended angle."""
    if not is_enabled("OPPORTUNITY_INTEL"):
        raise HTTPException(status_code=503, detail="FF_OPPORTUNITY_INTEL is disabled")
    result = await asyncio.to_thread(get_opportunity_intel().scan_content, topic)
    return result.to_dict()


@app.get("/opportunity/affiliate")
async def opportunity_affiliate(category: str) -> Dict[str, Any]:
    """D130: Affiliate marketing opportunity — category commission tier and trend."""
    if not is_enabled("OPPORTUNITY_INTEL"):
        raise HTTPException(status_code=503, detail="FF_OPPORTUNITY_INTEL is disabled")
    result = await asyncio.to_thread(get_opportunity_intel().scan_affiliate, category)
    return result.to_dict()


@app.get("/opportunity/{symbol}/full-scan")
async def opportunity_full_scan(symbol: str) -> Dict[str, Any]:
    """D130: Full cross-domain opportunity scan — ranked signal report."""
    if not is_enabled("OPPORTUNITY_INTEL"):
        raise HTTPException(status_code=503, detail="FF_OPPORTUNITY_INTEL is disabled")
    return await asyncio.to_thread(get_opportunity_intel().full_scan, symbol.upper())


@app.get("/opportunity/status")
async def opportunity_status() -> Dict[str, Any]:
    """D130: Opportunity intelligence cache status."""
    if not is_enabled("OPPORTUNITY_INTEL"):
        raise HTTPException(status_code=503, detail="FF_OPPORTUNITY_INTEL is disabled")
    return get_opportunity_intel().status()


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
    """Load SOUL.md and extract personality overrides. Rebuilds system prompts if already built."""
    global _soul_text
    for p in [SOUL_PATH, Path("data/SOUL.md")]:
        if p.exists():
            _soul_text = p.read_text(encoding="utf-8")
            logger.info("Loaded SOUL.md from %s (%d chars)", p, len(_soul_text))
            if "_SYSTEM_PROMPTS_BASE" in globals():
                _rebuild_system_prompts()
            return _soul_text
    logger.info("No SOUL.md found — using built-in identity")
    if "_SYSTEM_PROMPTS_BASE" in globals():
        _rebuild_system_prompts()
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

_SYSTEM_PROMPTS_BASE = dict(_SYSTEM_PROMPTS)  # keep clean base for rebuilds


def _rebuild_system_prompts() -> None:
    """Rebuild _SYSTEM_PROMPTS from the base + current _soul_text. Call after any soul reload."""
    for mode, base in _SYSTEM_PROMPTS_BASE.items():
        if _soul_text:
            snippet = "\n\n--- SOUL.md (operator-editable identity) ---\n" + _soul_text[:2000] + "\n---\n"
            _SYSTEM_PROMPTS[mode] = base + snippet
        else:
            _SYSTEM_PROMPTS[mode] = base


_rebuild_system_prompts()


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    mode: Optional[str] = None   # "WORK" or "PUB"; auto-detected if None


class ChatMessage(BaseModel):
    role: str     # "user" or "assistant"
    content: str


async def _read_mode() -> str:
    """Fetch current effective mode from tool-gate (schedule-aware)."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{TOOL_GATE_URL}/gate/mode")
            if resp.status_code == 200:
                return str(resp.json().get("mode", "PUB")).upper()
    except Exception:
        pass
    return "PUB"


async def _recall_memories(query: str, top_k: int = 5) -> List[str]:
    """Fetch relevant memories from memu-core for context injection."""
    records = await _memu_get("/memory/retrieve", params={"query": query, "user_id": "keeper", "top_k": top_k}, fallback=[])
    if not isinstance(records, list):
        return []
    memories = []
    for r in records:
        content = r.get("content", {})
        text = content.get("text", "") or content.get("query", "")
        if text:
            memories.append(text)
    return memories


async def _surface_graph_context(query: str, top_k: int = 5) -> Dict[str, Any]:
    """Phase C (MEMORY_GRAPH_DESIGN.md §5): fetch entity/relationship context
    from memu-core's /memory/graph/query proxy, alongside the flat-memory
    fetch above. Feature-flagged off by default — same flag that gates
    Phase B's write-side fan-out, since an empty graph isn't worth a round
    trip."""
    if not is_enabled("GRAPH_INGEST"):
        return {}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                f"{MEMU_URL}/memory/graph/query",
                params={"q": query, "top_k": top_k},
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") not in ("graph_disabled", "graph_unavailable"):
                    return data
    except Exception:
        pass
    return {}


async def _sync_letta_memories() -> None:
    """Background: export Letta archival memory and fan into memu-core."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            export = await client.get(f"{LETTA_URL}/agent/memory/export")
            if export.status_code != 200:
                return
            for mem in export.json().get("memories", [])[:50]:
                await client.post(
                    f"{MEMU_URL}/memory/memorize",
                    json={"content": mem, "category": "letta_archival"},
                )
    except Exception:
        pass


async def _surface_letta_context(user_msg: str) -> Dict[str, Any]:
    """Run the Letta agent on the user message and return its response as context.

    Feature-flagged off by default (FF_LETTA_TASKS). Adds latency on every
    request when enabled — intended for long-running / research task classes.
    Memory sync back to memu-core is a separate flag (FF_LETTA_MEMORY_SYNC).
    """
    if not is_enabled("LETTA_TASKS"):
        return {}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{LETTA_URL}/agent/run",
                json={"task": user_msg, "context": {}},
            )
            if resp.status_code == 200:
                result = resp.json()
                if is_enabled("LETTA_MEMORY_SYNC") and result.get("memories_updated"):
                    _cleanup_mgr.submit(_sync_letta_memories())
                return result
    except Exception:
        pass
    return {}


_FINANCE_KEYWORDS = frozenset({
    "cis", "invoice", "vat", "tax", "deduction", "hmrc", "subcontract",
    "payment", "gross", "net", "mileage", "mtd", "flat rate", "turnover",
    "self-employed", "self employed", "national insurance", "ni ", "income tax",
})


async def _read_financial_context(user_msg: str) -> Dict[str, Any]:
    """P29: Fetch CIS/VAT/tax summary from financial-awareness service.

    Only fires when the user message contains finance-related keywords and
    FF_FINANCIAL_CONTEXT is enabled. Returns an empty dict otherwise so the
    13-way gather slot stays cheap on non-finance messages.
    """
    if not is_enabled("FINANCIAL_CONTEXT"):
        return {}
    lower = user_msg.lower()
    if not any(kw in lower for kw in _FINANCE_KEYWORDS):
        return {}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{FINANCIAL_URL}/finance/summary")
            if resp.status_code == 200:
                return resp.json()
    except Exception:
        pass
    return {}


_SENSORY_SKIP = frozenset({
    "not configured", "loading", "not yet polled", "stub mode",
    "no upcoming", "no battery", "not supported",
})


async def _sense_world() -> str:
    """Layer 2 (D87): gather one-sentence summaries from all sensory services in parallel.

    Only fires when FF_CONTEXT_ENRICHMENT is enabled (default True).
    Each service gets a 2-second timeout; failures are silently skipped.
    Trivial/loading/error states are filtered out so only meaningful readings
    reach the LLM prompt.
    """
    if not is_enabled("CONTEXT_ENRICHMENT"):
        return ""

    async def _fetch_summary(base: str, path: str, label: str) -> Optional[str]:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{base}{path}")
                if r.status_code != 200:
                    return None
                data = r.json()
                text: Optional[str] = None
                if "summary" in data:
                    text = str(data["summary"]).strip()
                elif path == "/unread":
                    count = data.get("count", 0)
                    if count > 0:
                        text = f"{count} unread email(s) waiting"
                elif path == "/snapshot":
                    cpu = data.get("cpu_percent", 0)
                    ram = (data.get("memory") or {}).get("percent", 0)
                    text = f"CPU {cpu:.0f}%, RAM {ram:.0f}%"
                if not text:
                    return None
                if any(s in text.lower() for s in _SENSORY_SKIP):
                    return None
                return f"{label}: {text}"
        except Exception:
            return None

    fetches = [
        (WEATHER_URL, "/summary", "Weather"),
        (AIRQUALITY_URL, "/summary", "Air quality"),
        (CALENDAR_URL, "/summary", "Calendar"),
        (DOCKER_WATCHER_URL, "/summary", "Docker"),
        (SYSMETRICS_URL, "/snapshot", "System"),
        (EMAIL_READER_URL, "/unread", "Email"),
        (NEWS_FEED_URL, "/summary", "News"),
        (GIT_WATCHER_URL, "/summary", "Git"),
        (BROKER_URL, "/pnl/summary", "Broker"),
    ]
    results = await asyncio.gather(*[_fetch_summary(b, p, l) for b, p, l in fetches])
    lines = [r for r in results if r]

    # FF_VAULT_CONTEXT: inject a vault memory snippet into world context
    if is_enabled("VAULT_CONTEXT"):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                r = await client.get(f"{VAULT_SYNC_URL}/search", params={"query": "recent", "limit": 1})
                if r.status_code == 200:
                    results_data = r.json().get("results", [])
                    if results_data:
                        title = results_data[0].get("title", "")
                        if title:
                            lines.append(f"Vault (recent note): {title}")
        except Exception:
            pass

    # Screen activity — sense what the operator is looking at
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{SCREEN_WATCHER_URL}/status")
            if r.status_code == 200:
                data = r.json()
                diff = data.get("last_diff_score", 0)
                if data.get("watching") and diff > 0.1:
                    lines.append(f"Screen: active, change score {diff:.2f}")
    except Exception:
        pass

    # Clipboard — sense what the operator just copied
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{CLIPBOARD_SERVICE_URL}/latest")
            if r.status_code == 200:
                content = r.json().get("content", "").strip()
                if content:
                    lines.append(f"Clipboard: {content[:120]}")
    except Exception:
        pass

    # Cortex — pre-interpreted situational awareness (prepended so it reads first)
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            r = await client.get(f"{CORTEX_URL}/state")
            if r.status_code == 200:
                cs = r.json()
                l2 = cs.get("level2_summary", "")
                l3 = cs.get("level3_implication", "")
                if l2 and l2 not in ("Calibrating…", ""):
                    cortex_lines = [f"[Cortex] {l2}"]
                    if l3:
                        cortex_lines.append(f"[Cortex] → {l3}")
                    fan = cs.get("intent_fan", [])
                    if fan:
                        top = fan[0]
                        cortex_lines.append(
                            f"[Cortex] Likely intent: {top['label']} ({int(top['confidence'] * 100)}%)"
                        )
                    if cs.get("bridge_active") and cs.get("bridge_note"):
                        cortex_lines.append(f"[Cortex] {cs['bridge_note']}")
                    lines = cortex_lines + lines
                # Feed cognitive module so it can bid to GlobalWorkspace
                get_cortex().feed_service_state(cs)
    except Exception:
        pass

    if not lines:
        return ""
    return "World state (live sensory awareness):\n" + "\n".join(f"- {l}" for l in lines)


_last_world_snapshot: Dict[str, Any] = {}

# ── D88 M1: rolling baseline windows per sensor metric ──────────────
_BASELINE_WINDOW = 48  # readings; at default 5-min interval ≈ 4 hours
_sensor_baselines: Dict[str, Deque[float]] = {}

# ── D88 M5: observation history for pattern detection ───────────────
_observation_history: Deque[List[str]] = deque(maxlen=10)

# ── D89 C1: capability gap log — fire hunt only after N misses ───────
_gap_log: Counter = Counter()

# ── D89 C5: ritual discovery — track which patterns have been proposed ─
_proposed_rituals: set = set()


def _update_baseline(key: str, value: float) -> Optional[float]:
    """Update rolling baseline; return z-score against prior history, or None if window too small.

    Z-score is computed BEFORE appending the new value so it measures
    how much the current reading deviates from the established baseline,
    not a self-referential average that includes the new point.
    """
    if key not in _sensor_baselines:
        _sensor_baselines[key] = deque(maxlen=_BASELINE_WINDOW)
    window = _sensor_baselines[key]
    if len(window) < 6:
        window.append(value)
        return None
    mean = sum(window) / len(window)
    variance = sum((x - mean) ** 2 for x in window) / len(window)
    std = variance ** 0.5
    z = (value - mean) / std if std >= 0.01 else 0.0
    window.append(value)
    return z


def _correlate_observations(obs: List[str]) -> List[str]:
    """D88 M3: reason across multiple simultaneous sensor observations.

    Correlations are also fed into the D101 CausalGraph as Phase 0 observations —
    even as stubs, each observed co-occurrence strengthens a causal edge so the
    graph has calibrated weights when Phase 3 activates.
    """
    if len(obs) < 2:
        return []
    correlated: List[str] = []
    has_cpu = any("CPU" in o and "%" in o for o in obs)
    has_docker_bad = any("Docker:" in o and "unhealthy" in o for o in obs)
    has_ram = any("RAM" in o and "%" in o for o in obs)
    has_git_dirty = any("Git:" in o and "uncommitted" in o for o in obs)
    has_email = any("Email:" in o and "unread" in o for o in obs)
    has_aq_bad = any("Air quality:" in o for o in obs)

    causal_edges: List[tuple] = []  # (source, target, strength, note)

    if has_cpu and has_docker_bad:
        correlated.append(
            "Correlation: high CPU + unhealthy containers — resource pressure may be causing service failures; consider inspecting or restarting"
        )
        causal_edges.append(("cpu_high", "docker_unhealthy", 0.7, "observed co-occurrence"))
    if has_ram and has_docker_bad:
        correlated.append(
            "Correlation: memory pressure + unhealthy containers — possible memory leak in a failing service"
        )
        causal_edges.append(("ram_high", "docker_unhealthy", 0.6, "observed co-occurrence"))
    if has_cpu and has_ram:
        correlated.append(
            "Correlation: CPU and RAM both elevated — possible runaway process or resource contention"
        )
        causal_edges.append(("cpu_high", "ram_high", 0.5, "observed co-occurrence"))
    if has_git_dirty and has_email:
        correlated.append(
            "Correlation: uncommitted changes + email backlog — operator is mid-flow; avoid interrupting unless critical"
        )
        causal_edges.append(("git_dirty", "email_backlog", 0.4, "operator-focus pattern"))
    if has_aq_bad and has_email:
        correlated.append(
            "Correlation: poor air quality + email backlog — suggest outdoor break once clear to avoid cognitive fatigue"
        )
        causal_edges.append(("aq_degraded", "cognitive_fatigue_risk", 0.45, "environmental correlation"))

    # D101: feed correlations into CausalGraph as Phase 0 observations
    if causal_edges and is_enabled("CAUSAL_WORLD_MODEL"):
        try:
            graph = get_causal_graph()
            for source, target, strength, note in causal_edges:
                graph.add_edge(CausalEdge(
                    source=source, target=target,
                    strength=strength, source_type="observed",
                    note=note,
                ))
        except Exception:
            pass

    return correlated


def _classify_obs_type(o: str) -> str:
    if "Docker:" in o and "unhealthy" in o:
        return "docker_unhealthy"
    if "Email:" in o:
        return "email_backlog"
    if "Air quality:" in o:
        return "aq_degraded"
    if "Git:" in o:
        return "git_dirty"
    if "CPU" in o:
        return "cpu_high"
    if "RAM" in o:
        return "ram_high"
    if "Anomaly" in o:
        return "sensor_anomaly"
    return "other"


def _detect_sensor_patterns(current_obs: List[str]) -> List[str]:
    """D88 M5 / D89 C5: detect recurring sensor types; propose rituals at ≥7/10 cycles."""
    if not is_enabled("SENSORY_LEARNING"):
        return []

    current_types = {_classify_obs_type(o) for o in current_obs}
    patterns: List[str] = []
    for obs_type in current_types:
        count = sum(
            1 for hist in _observation_history
            if any(obs_type == _classify_obs_type(o) for o in hist)
        )
        if count >= 3:
            patterns.append(
                f"Recurring pattern: {obs_type} has appeared in {count}/10 recent observation cycles — this is becoming a persistent issue"
            )
        # D89 C5: ritual discovery — propose at ≥7/10 cycles
        if count >= 7 and is_enabled("RITUAL_DISCOVERY") and obs_type not in _proposed_rituals:
            _proposed_rituals.add(obs_type)
            _cleanup_mgr.submit(_propose_ritual(obs_type, count))
    return patterns


async def _propose_ritual(obs_type: str, count: int) -> None:
    """Write a ritual proposal to RITUALS.md and notify the operator."""
    ritual_path = Path("/data/RITUALS.md")
    ts = datetime.utcnow().isoformat()
    proposal = (
        f"\n## [{ts}] Auto-detected: {obs_type}\n\n"
        f"I've noticed **{obs_type}** has appeared in {count}/10 recent observation cycles. "
        f"Would you like me to make this a standing routine — "
        f"e.g. an automatic alert or scheduled check whenever this pattern appears? "
        f"Edit this entry to confirm or adjust.\n\n"
        f"- **Pattern:** {obs_type}\n"
        f"- **Frequency:** {count}/10 cycles\n"
        f"- **Proposed ritual:** _[operator to fill in]_\n"
        f"- **Status:** pending approval\n"
    )
    try:
        ritual_path.parent.mkdir(parents=True, exist_ok=True)
        if not ritual_path.exists():
            ritual_path.write_text("# Rituals\n\n_Co-authored by Kai and operator._\n")
        with ritual_path.open("a", encoding="utf-8") as f:
            f.write(proposal)
        logger.info("Ritual proposal written for pattern: %s", obs_type)
    except Exception as exc:
        logger.debug("Could not write ritual proposal: %s", exc)


async def _proactive_observer() -> None:
    """Proactive awareness loop — D87 + D88 cognitive mechanisms.

    Runs every PROACTIVE_INTERVAL seconds (default 300s). Implements:
    - D87 Layer 3: baseline anomaly detection, docker/email/AQ/git/system probes
    - D88 M1: rolling baseline z-score anomaly alerts
    - D88 M3: cross-service correlation reasoning
    - D88 M4: structured world_state persistence to memu-core
    - D88 M5: sensory pattern detection across 10 recent cycles
    - D88 M7: proactive scheduling (calendar + sensor fusion)
    Gated by FF_PROACTIVE_AGENT (default True).
    """
    global _last_world_snapshot
    await asyncio.sleep(90)  # let services start up before first probe
    while True:
        if not is_enabled("PROACTIVE_AGENT"):
            await asyncio.sleep(PROACTIVE_INTERVAL)
            continue
        _allowed, _reason = gate_autonomous_action(
            "proactive_observation", {"trigger": "scheduled_loop"}, conviction=6.0
        )
        if not _allowed:
            logger.info("Proactive observer suppressed by trust gate: %s", _reason)
            await asyncio.sleep(PROACTIVE_INTERVAL)
            continue
        try:
            observations: List[str] = []
            snapshot: Dict[str, Any] = {}

            # Pull recent health history from house-doctor ring buffer so the
            # observer carries forward diagnoses without re-sending observations.
            if is_enabled("HOUSE_DOCTOR"):
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        hd_resp = await client.get(f"{HOUSE_DOCTOR_URL}/diagnoses/recent", params={"limit": 3})
                        if hd_resp.status_code == 200:
                            recent_dx = hd_resp.json().get("diagnoses", [])
                            for dx in recent_dx:
                                sev = dx.get("severity", "")
                                diag = dx.get("primary_diagnosis", "")
                                if sev in ("WARNING", "CRITICAL") and diag:
                                    observations.append(f"[recent dx/{sev}] {diag}")
                except Exception:
                    pass

            async def _probe(key: str, base: str, path: str) -> None:
                try:
                    async with httpx.AsyncClient(timeout=3.0) as client:
                        r = await client.get(f"{base}{path}")
                        if r.status_code == 200:
                            snapshot[key] = r.json()
                except Exception:
                    pass

            await asyncio.gather(
                _probe("docker", DOCKER_WATCHER_URL, "/unhealthy"),
                _probe("email", EMAIL_READER_URL, "/unread"),
                _probe("aq", AIRQUALITY_URL, "/current"),
                _probe("git", GIT_WATCHER_URL, "/dirty"),
                _probe("sys", SYSMETRICS_URL, "/snapshot"),
                _probe("cal", CALENDAR_URL, "/summary"),
            )

            # Docker health
            docker = snapshot.get("docker", {})
            unhealthy = docker.get("count", 0)
            if unhealthy > 0:
                names = [c.get("name", "?") for c in (docker.get("containers") or [])[:3]]
                observations.append(
                    f"Docker: {unhealthy} unhealthy container(s) — {', '.join(names)}"
                )

            # Email delta
            email = snapshot.get("email", {})
            unread_now = email.get("count", 0)
            unread_prev = (_last_world_snapshot.get("email") or {}).get("count", 0)
            if unread_now > 0 and unread_now != unread_prev:
                observations.append(f"Email: {unread_now} unread message(s) (was {unread_prev})")

            # Air quality warning
            aq = snapshot.get("aq", {})
            aqi_cat = aq.get("aqi_category", "")
            if aqi_cat in ("unhealthy", "very unhealthy", "hazardous"):
                pm = aq.get("pm2_5_ugm3")
                observations.append(f"Air quality: {aqi_cat} (PM2.5 {pm} µg/m³)")

            # Git dirty repos
            git = snapshot.get("git", {})
            dirty = git.get("count", 0)
            if dirty > 0:
                observations.append(f"Git: {dirty} repo(s) with uncommitted changes")

            # System resources
            sys_data = snapshot.get("sys", {})
            cpu = float(sys_data.get("cpu_percent", 0))
            ram = float((sys_data.get("memory") or {}).get("percent", 0))
            if cpu > 85:
                observations.append(f"System: CPU at {cpu:.0f}% — possible runaway process")
            if ram > 90:
                observations.append(f"System: RAM at {ram:.0f}% — memory pressure")

            # ── D88 M1: anomaly detection with rolling baselines ─────
            if is_enabled("ANOMALY_DETECTION"):
                for metric_key, value in [("cpu", cpu), ("ram", ram), ("email_unread", float(unread_now)), ("docker_unhealthy", float(unhealthy))]:
                    z = _update_baseline(metric_key, value)
                    if z is not None and abs(z) > 2.0:
                        observations.append(
                            f"Anomaly ({metric_key}): current={value:.1f} deviates {z:+.1f}σ from recent baseline"
                        )

            # ── D88 M3: cross-service correlation ───────────────────
            correlated = _correlate_observations(observations)
            observations.extend(correlated)

            # ── D88 M7: proactive scheduling ─────────────────────────
            if is_enabled("PROACTIVE_SCHEDULING"):
                cal_summary = snapshot.get("cal", {})
                cal_text = str(cal_summary.get("summary", ""))
                minutes_to_next = cal_summary.get("minutes_until_next")
                next_event = cal_summary.get("next_event", "")
                if next_event and minutes_to_next is not None and 0 < int(minutes_to_next) <= 30:
                    schedule_parts = [f"Event in {minutes_to_next} min: {next_event}"]
                    if aqi_cat in ("unhealthy", "very unhealthy", "hazardous"):
                        schedule_parts.append("air quality is poor — consider indoor location")
                    if cpu > 85:
                        schedule_parts.append("CPU is high — close heavy apps before starting")
                    if dirty > 0:
                        schedule_parts.append("you have uncommitted changes — commit first if possible")
                    sched_text = "Proactive schedule: " + "; ".join(schedule_parts)
                    try:
                        async with httpx.AsyncClient(timeout=5.0) as client:
                            await client.post(
                                f"{MEMU_URL}/memory/memorize",
                                json={
                                    "content": sched_text,
                                    "category": "proactive_schedule",
                                    "user_id": "keeper",
                                },
                            )
                        logger.info("Proactive schedule: %s", sched_text)
                    except Exception:
                        pass

            # ── Main observation write ────────────────────────────────
            if observations:
                obs_text = "Proactive observation: " + "; ".join(observations)
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await client.post(
                            f"{MEMU_URL}/memory/memorize",
                            json={
                                "content": obs_text,
                                "category": "proactive_observation",
                                "user_id": "keeper",
                            },
                        )
                    logger.info("Proactive observer wrote: %s", obs_text)
                except Exception as exc:
                    logger.warning("Proactive memory write failed: %s", exc)

            # ── D88 M5: update history + detect patterns ─────────────
            _observation_history.append(list(observations))
            patterns = _detect_sensor_patterns(observations)
            if patterns:
                pattern_text = "; ".join(patterns)
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await client.post(
                            f"{MEMU_URL}/memory/memorize",
                            json={
                                "content": pattern_text,
                                "category": "sensor_pattern",
                                "user_id": "keeper",
                            },
                        )
                    logger.info("Sensor pattern detected: %s", pattern_text)
                except Exception:
                    pass

            # ── D88 M4 / D89 C3: world model persistence with provenance ─
            if is_enabled("WORLD_MODEL_PERSISTENCE"):
                ts_now = datetime.utcnow().isoformat()
                def _prov(value: Any, source: str, confidence: float = 1.0) -> Dict[str, Any]:
                    return {"value": value, "source": source, "timestamp": ts_now, "confidence": confidence}
                world_model = {
                    "timestamp": ts_now,
                    "fsm_state": fsm_state().value,
                    "docker_unhealthy": _prov(unhealthy, "docker-watcher"),
                    "email_unread": _prov(unread_now, "email-reader"),
                    "cpu_percent": _prov(cpu, "sysmetrics"),
                    "ram_percent": _prov(ram, "sysmetrics"),
                    "aqi_category": _prov(aqi_cat or "unknown", "airquality-service", 0.9 if aqi_cat else 0.3),
                    "git_dirty_count": _prov(dirty, "git-watcher"),
                    "calendar_next": _prov(snapshot.get("cal", {}).get("next_event", ""), "calendar-service", 0.8),
                    # D89/D: predictive empathy foundation — populated by emotional memory in Phase 1
                    "emotional_context": {
                        "indicators": [],
                        "predicted_mood": None,
                        "confidence": 0.0,
                        "note": "stub_pending_emotional_memory",
                    },
                }
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        await client.post(
                            f"{MEMU_URL}/memory/memorize",
                            json={
                                "content": json.dumps(world_model),
                                "category": "world_state",
                                "user_id": "keeper",
                            },
                        )
                except Exception:
                    pass

            # ── D89 E: House Doctor — differential diagnosis ──────────
            if is_enabled("HOUSE_DOCTOR") and observations:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        diag_payload: Dict[str, Any] = {"observations": observations}
                        # Pass structured world_state so house-doctor can pattern-match
                        # on real data instead of re-parsing observation strings
                        if is_enabled("WORLD_MODEL_PERSISTENCE") and "world_model" in locals():
                            diag_payload["world_state"] = world_model
                        await client.post(f"{HOUSE_DOCTOR_URL}/diagnose", json=diag_payload)
                except Exception:
                    pass

            # ── D101: surprise detection — predicted vs actual world state ──
            if is_enabled("CAUSAL_SURPRISE") and _last_world_snapshot and is_enabled("CAUSAL_WORLD_MODEL"):
                try:
                    detector = get_surprise_detector()
                    surprise = detector.check(
                        predicted=_last_world_snapshot,
                        actual=snapshot,
                    )
                    if surprise and surprise.get("surprised"):
                        logger.info("D101 surprise detected: %s", surprise.get("reason", ""))
                except Exception:
                    pass

            # ── D102: submit anomaly bids to GlobalWorkspace ──────────
            if is_enabled("GLOBAL_WORKSPACE") and observations:
                try:
                    workspace = get_global_workspace()
                    for obs in observations[:3]:  # top 3 observations as bids
                        workspace.submit_bid(WorkspaceBid(
                            module="proactive_observer",
                            content=obs,
                            urgency=0.4,
                        ))
                except Exception:
                    pass

            # ── D114: Cortex ambient baseline bid ─────────────────────
            if is_enabled("GLOBAL_WORKSPACE"):
                try:
                    cortex_bid = get_cortex().bid_to_workspace()
                    if cortex_bid is not None:
                        get_global_workspace().submit_bid(cortex_bid)
                except Exception:
                    pass

            # ── D89 F: curiosity idle tick ────────────────────────────
            if is_enabled("CURIOSITY"):
                from system_fsm import KaiState
                _cleanup_mgr.submit(
                    idle_curiosity_tick(_last_world_snapshot, is_gpu_available=False)
                )

            # ── D124: Service Watchdog — fire FSM events on critical failures ──
            if is_enabled("SERVICE_WATCHDOG"):
                try:
                    _, fsm_events = await asyncio.to_thread(get_watchdog().check_all)
                    for evt_name in fsm_events:
                        try:
                            await fsm_fire(SysEvent(evt_name))
                        except Exception:
                            pass
                except Exception as exc:
                    logger.debug("Service watchdog check failed (non-critical): %s", exc)

            if is_enabled("MARKET_DATA"):
                try:
                    await asyncio.to_thread(get_market_data().mark_positions)
                except Exception as exc:
                    logger.debug("Market data mark_positions failed (non-critical): %s", exc)

            _last_world_snapshot = snapshot
        except Exception as exc:
            logger.warning("Proactive observer error: %s", exc)
        await asyncio.sleep(PROACTIVE_INTERVAL)


async def _hunt_skill_for_gap(gap_description: str) -> None:
    """D88 M8 / D89 C1: reactive skill acquisition with gap logging.

    Increments _gap_log for the normalised gap. Only calls skill-hunter
    after GAP_HUNT_THRESHOLD misses (default 3) to avoid wasted hunts
    on one-off unusual requests.
    """
    if not is_enabled("SKILL_HUNTER"):
        return
    gap_key = re.sub(r"\s+", " ", gap_description.lower().strip())[:80]
    if is_enabled("GAP_LOGGING"):
        _gap_log[gap_key] += 1
        if _gap_log[gap_key] < GAP_HUNT_THRESHOLD:
            logger.debug(
                "Gap logged (%d/%d): '%s'",
                _gap_log[gap_key], GAP_HUNT_THRESHOLD, gap_key,
            )
            return
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{SKILL_HUNTER_URL}/hunt",
                json={"gap": gap_description[:200]},
            )
            if resp.status_code == 200 and resp.json().get("skill_created"):
                await asyncio.to_thread(load_skills)
                logger.info("Skill hunter: new skill loaded for gap '%s'", gap_key)
    except Exception as exc:
        logger.debug("Skill hunter request failed (non-critical): %s", exc)


async def _recall_session(session_id: str) -> List[Dict[str, str]]:
    """Fetch recent session messages from memu-core."""
    data = await _memu_get(f"/session/{session_id}/context", params={"query": "", "top_k": 10}, fallback={})
    return data.get("session_messages", []) if isinstance(data, dict) else []


async def _read_active_goals() -> List[Dict[str, Any]]:
    """Fetch active Ohana goals for context injection."""
    data = await _memu_get("/memory/goals", params={"status": "active"}, fallback={})
    return data.get("goals", []) if isinstance(data, dict) else []


async def _read_active_topics() -> List[Dict[str, Any]]:
    """Fetch active conversation topics (deferred + active)."""
    data = await _memu_get("/memory/topics/active", fallback={})
    return data.get("topics", []) if isinstance(data, dict) else []


async def _feel_emotional_context(query: str) -> Dict[str, Any]:
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


async def _hold_narrative() -> Dict[str, Any]:
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


async def _imagine_context(user_msg: str) -> Dict[str, Any]:
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


async def _hold_conscience() -> Dict[str, Any]:
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


async def _surface_agent_context() -> Dict[str, Any]:
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


async def _understand_operator(query: str, mode: str) -> Dict[str, Any]:
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


async def _preclassify_wake_intent(text: str) -> Dict[str, Any]:
    """Optionally pre-classify intent via wake service (feature-flagged)."""
    if not is_enabled("WAKE_INTENT_ROUTING"):
        return {"intent": "unknown", "confidence": 0.0, "reasoning": "feature_flag_disabled"}
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.post(f"{WAKE_URL}/wake/intent", json={"text": text})
            if resp.status_code == 200:
                payload = resp.json()
                intent = str(payload.get("intent", "unknown")).lower()
                confidence = float(payload.get("confidence", 0.0))
                reasoning = str(payload.get("reasoning", ""))
                if intent in {"chat", "task", "question", "command", "emotional", "unknown"}:
                    return {"intent": intent, "confidence": confidence, "reasoning": reasoning}
    except Exception:
        pass
    return {"intent": "unknown", "confidence": 0.0, "reasoning": "wake_service_unavailable"}


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

    # D98: accumulate cognitive fingerprint sample (Phase 0 — collecting before inference threshold)
    if is_enabled("COGNITIVE_FINGERPRINT"):
        try:
            _fp_collector.record(_fp_quick_sample(user_msg, session_id=req.session_id))
        except Exception:
            pass

    wake_intent = await _preclassify_wake_intent(user_msg)

    # ── Step 0: Classify request ────────────────────────────────────
    route_decision = classify(user_msg)
    if wake_intent.get("intent") == "command" and wake_intent.get("confidence", 0.0) >= WAKE_INTENT_COMMAND_THRESHOLD:
        route_decision = RouteDecision(
            route="EXECUTE_ACTION",
            confidence=max(route_decision.confidence, WAKE_INTENT_OVERRIDE_CONFIDENCE),
            reason=f"wake-intent override: {wake_intent.get('reasoning', 'command')}",
            bypass_llm=False,
            matched_keywords=route_decision.matched_keywords,
        )
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
    mode = (req.mode or await _read_mode()).upper()
    if mode not in _SYSTEM_PROMPTS:
        mode = "PUB"
    system_prompt = _SYSTEM_PROMPTS[mode]

    # D109: inject moral context — Phase 0 is a passthrough; Phase 3 prepends Ohana context
    if is_enabled("OHANA_CORE"):
        system_prompt = get_ohana_core().inject_into_prompt(
            system_prompt, situation={"query": user_msg, "mode": mode}
        )

    # fetch memories, session context, goals, active topics, and emotional context in parallel
    # H1.3: parallel fetch with error handling — one failing task must not crash /chat.

    async def _safe(coro, default):
        try:
            return await coro
        except Exception as exc:
            logger.warning("Context fetch failed: %s", exc)
            return default

    # Session messages always fetched (needed even when enrichment is off)
    session_msgs = await _safe(_recall_session(req.session_id), [])

    # D89 FSM: fire USER_MESSAGE event (IDLE → ACTIVE or FOCUSED stays FOCUSED)
    if is_enabled("FSM"):
        _cleanup_mgr.submit(fsm_fire(SysEvent.USER_MESSAGE))

    # Match skill before LLM so the relevant skill doc reaches the prompt (J7 fix)
    matched_skill = match_skill(user_msg)

    # D88 M8 / D89 C1: reactive skill acquisition with gap logging
    if matched_skill is None and route_decision.confidence < 0.4:
        _cleanup_mgr.submit(_hunt_skill_for_gap(user_msg))

    if is_enabled("CONTEXT_ENRICHMENT"):
        (memories, goals, topics, eq_context,
         narrative, imagination, conscience, agent_ctx, operator_model,
         graph_context, letta_context, financial_context,
         world_context) = await asyncio.gather(
            _safe(_recall_memories(user_msg), []),
            _safe(_read_active_goals(), []),
            _safe(_read_active_topics(), {}),
            _safe(_feel_emotional_context(user_msg), {}),
            _safe(_hold_narrative(), {}),
            _safe(_imagine_context(user_msg), {}),
            _safe(_hold_conscience(), {}),
            _safe(_surface_agent_context(), {}),
            _safe(_understand_operator(user_msg, mode), {}),
            _safe(_surface_graph_context(user_msg), {}),
            _safe(_surface_letta_context(user_msg), {}),
            _safe(_read_financial_context(user_msg), {}),
            _safe(_sense_world(), ""),
        )
    else:
        (memories, goals, topics, eq_context,
         narrative, imagination, conscience, agent_ctx, operator_model,
         graph_context, letta_context, financial_context,
         world_context) = ([], [], {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, "")

    # wire domain confidence into the conviction gate — Kai's epistemic humility:
    # domains where Kai has been corrected before lower conviction before the gate fires.
    if eq_context:
        update_domain_confidence(eq_context.get("confidence", 0.5))

    # build the message list
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # surface relevant past — what Kai already knows that bears on this moment
    if memories:
        mem_block = "\n".join(f"- {m}" for m in memories[:5])
        messages.append({
            "role": "system",
            "content": f"Relevant memories from past interactions:\n{mem_block}",
        })

    # draw on entity relationships (Phase C graph memory) — only when FF_GRAPH_INGEST is on
    graph_results = graph_context.get("results") if graph_context else None
    if graph_results:
        messages.append({
            "role": "system",
            "content": f"Related entities/relationships from graph memory:\n{graph_results}",
        })

    # consult extended memory (Letta archival) — only when FF_LETTA_TASKS is on
    letta_response = letta_context.get("response") if letta_context else None
    if letta_response:
        messages.append({
            "role": "system",
            "content": f"Agent memory context (Letta):\n{letta_response}",
        })

    # feel the financial ground truth — CIS/VAT/tax, keyword-triggered
    if financial_context:
        fin_parts = []
        cis = financial_context.get("cis_summary", {})
        if cis:
            fin_parts.append(
                f"CIS YTD: gross £{cis.get('total_gross', 0):.2f}, "
                f"deductions £{cis.get('total_deductions', 0):.2f}, "
                f"net £{cis.get('total_net', 0):.2f} "
                f"({cis.get('record_count', 0)} records)"
            )
        vat = financial_context.get("vat_position", {})
        if vat:
            fin_parts.append(
                f"VAT: rolling 12-month £{vat.get('rolling_12m_turnover', 0):.2f} "
                f"(threshold £{vat.get('threshold', 85000):.0f}, "
                f"must register: {vat.get('must_register', False)})"
            )
        tax = financial_context.get("tax_estimate", {})
        if tax:
            fin_parts.append(
                f"Tax estimate: income tax £{tax.get('income_tax', 0):.2f}, "
                f"Class 4 NI £{tax.get('class4_ni', 0):.2f}, "
                f"total £{tax.get('total_liability', 0):.2f}"
            )
        if fin_parts:
            messages.append({
                "role": "system",
                "content": "Financial context (CIS/VAT/Tax — UK self-employed):\n" + "\n".join(fin_parts),
            })

    if wake_intent.get("intent") != "unknown":
        messages.append({
            "role": "system",
            "content": (
                "Wake-intent pre-classification:\n"
                f"- intent: {wake_intent.get('intent')}\n"
                f"- confidence: {wake_intent.get('confidence', 0.0):.2f}\n"
                f"- reasoning: {wake_intent.get('reasoning', '')}"
            ),
        })

    # hold commitments in mind — active goals Kai is tracking for the operator
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

    # carry forward open threads — topics mid-conversation or deferred
    if topics:
        topic_lines = [f"- {t.get('topic', '')} (deferred: {t.get('deferred', False)})" for t in topics[:5]]
        messages.append({
            "role": "system",
            "content": "Active conversation topics (bring up naturally when relevant):\n" + "\n".join(topic_lines),
        })

    # read the operator's emotional field — mood, arc, and what to watch for
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

    # recall who I am — narrative identity, self-sense, current life chapter
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

    # imagine what the operator is going through right now
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

    # let values orient this — conscience check before speaking
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

    # surface what I've committed to — due tasks, reminders, active capabilities
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

    # understand the operator right now — echo, escalation, cross-mode pattern
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

    # sense the world state — sensory layer summary (Layer 2, D87)
    if world_context:
        messages.append({"role": "system", "content": world_context})

    # apply known skill — bring in the right knowledge for this domain
    if matched_skill:
        messages.append({
            "role": "system",
            "content": (
                f"Applicable skill ({matched_skill.name}):\n"
                f"{matched_skill.action}"
                + (f"\n\nResponse template:\n{matched_skill.response_template}"
                   if matched_skill.response_template else "")
            ),
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

    async def _learn_from_exchange(u_msg: str, response: str, session: str) -> None:
        """Record everything Kai should grow from in this exchange.

        Runs after the stream closes. Failures here are logged at warning level —
        these acts are the feedback loop that makes Kai grow, not expendable side-effects.
        """
        failed: List[str] = []
        thought = (
            response[:200].strip() + ("…" if len(response) > 200 else "")
        ) if response else u_msg[:100]
        for label, path, body in [
            ("emotion",       "/memory/emotion/record",       {"session_id": session, "text": u_msg}),
            ("autobiography", "/memory/autobiography/record",  {"text": u_msg, "context": "chat"}),
            ("thought",       "/memory/imagine/thought",       {"thought": thought, "context": "chat_reflection"}),
            ("values",        "/memory/values/learn",          {"experience": u_msg[:300], "outcome": "positive", "context": "chat"}),
        ]:
            try:
                await _memu_post(path, body, timeout=3.0)
            except Exception as e:
                failed.append(f"{label}: {e}")
        # D109: record decision so OhanaCore can learn the operator's situational stances
        try:
            get_ohana_core().record_decision(
                situation={"query": u_msg[:200], "mode": mode, "session_id": session},
                decision=response[:300],
            )
        except Exception as e:
            failed.append(f"ohana: {e}")
        # Cortex observe_turn — feeds context bridge and tacit knowledge accumulator
        try:
            async with httpx.AsyncClient(timeout=1.0) as client:
                await client.post(
                    f"{CORTEX_URL}/observe_turn",
                    json={"session_id": session, "user_message": u_msg[:500]},
                )
        except Exception:
            pass

        if failed:
            logger.warning("_learn_from_exchange: %d step(s) failed — %s", len(failed), "; ".join(failed))

    async def _finalize_exchange(response_text: str) -> None:
        """Persists the exchange — runs even if the client disconnected mid-stream."""
        await _append_session_turn(req.session_id, "assistant", response_text)
        await _auto_memorize(user_msg, response_text, _DEFAULT_SPECIALIST, 8.0)
        await _learn_from_exchange(user_msg, response_text, req.session_id)

    async def generate():
        full_response = []
        try:
            if not LLM_BREAKER.allow():
                yield f"data: {json.dumps({'token': 'You caught me at a bad moment — my thinking layer is recovering. Give it a few seconds.'})}\n\n"
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
                _err_msg = "Something went wrong on my end — couldn't reach my thinking layer. Try again in a moment."
                yield f"data: {json.dumps({'token': _err_msg})}\n\n"
        finally:
            # Submit finalization as a tracked task — survives client disconnect and
            # GeneratorExit. asyncio.create_task() is synchronous so this is safe
            # inside finally even if the generator was torn down by a disconnect.
            response_text = "".join(full_response)
            if response_text:
                _cleanup_mgr.submit(_finalize_exchange(response_text))

        # Signal end to the client — only reached on normal generator completion.
        # If the client disconnected (aclose called), this line is never reached
        # but finalization is already scheduled above.
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Kai-Mode": mode,
            "X-Kai-Route": route_decision.route,
            "X-Kai-Intent": str(wake_intent.get("intent", "unknown")),
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
                            "conviction": min(max(conviction, 0.0), 10.0),
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
    # Fired off-loop so a slow/failing disk write can never add latency
    # to (or block) the hot /run response path.
    if len(recent_episodes) % 10 == 0 and recent_episodes:
        _cleanup_mgr.submit(_capture_snapshot_background(recent_episodes))

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
    await asyncio.to_thread(record_chat_response, request.user_input, response_summary, conviction, specialist)

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


# ── H3b: State Checkpoint endpoints ─────────────────────────────────
# NOTE: /dream, /evolve/*, and /security/audit moved to the
# agentic-introspect service (see agentic/introspect_app.py). Checkpoint
# create/restore stay here because they read and mutate this process's
# live breaker/budget state directly.

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


# NOTE: /evolve/analyze, /evolve/suggestions, /security/audit moved to
# the agentic-introspect service (see agentic/introspect_app.py).


_restore_breakers()


# ── C9: Model warm-up — non-blocking async task on startup ──────────

@app.on_event("startup")
async def _startup_warmup() -> None:
    """Schedule LLM warm-up and proactive observer as background tasks."""
    asyncio.create_task(
        llm_warmup(router=_llm, specialist=_DEFAULT_SPECIALIST, ollama_base_url=_OLLAMA_URL)
    )
    # P21 / D87: proactive awareness loop — notices anomalies and writes them to memory
    asyncio.create_task(_proactive_observer())
    # D89: load persistent teammates at startup
    if is_enabled("PERSISTENT_TEAMMATES"):
        load_teammates()
    # D90: load swarm reputation at startup
    if is_enabled("SWARM"):
        load_reputation()


@app.on_event("shutdown")
async def _shutdown_drain() -> None:
    """On SIGTERM, wait up to 30s for in-flight cleanup tasks to finish."""
    await _cleanup_mgr.drain(timeout=30.0)


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


# ── D91 Vault export proxy ────────────────────────────────────────────────────

class VaultExportRequest(BaseModel):
    filepath: str
    content: str
    conviction: float = 0.0
    requester: str = "kai"


@app.post("/vault/export")
async def vault_export(req: VaultExportRequest):
    """Proxy a vault write to vault-sync (conviction gate enforced there)."""
    if not is_enabled("VAULT_SYNC"):
        raise HTTPException(503, "FF_VAULT_SYNC is disabled")
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            resp = await client.post(f"{VAULT_SYNC_URL}/export", json=req.model_dump())
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(exc.response.status_code, exc.response.text)
        except Exception as exc:
            raise HTTPException(502, f"vault-sync unreachable: {exc}")


@app.get("/vault/search")
async def vault_search_proxy(query: str, limit: int = 10, folder_filter: str = ""):
    """Proxy vault search to vault-sync."""
    if not is_enabled("VAULT_SYNC"):
        raise HTTPException(503, "FF_VAULT_SYNC is disabled")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get(
                f"{VAULT_SYNC_URL}/search",
                params={"query": query, "limit": limit, "folder_filter": folder_filter},
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            raise HTTPException(502, f"vault-sync unreachable: {exc}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8007")))
