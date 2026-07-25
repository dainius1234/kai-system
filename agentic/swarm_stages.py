"""D90: Swarm stage function factories for the CognitiveFSM pipeline.

Each factory returns a StageFunc = Callable[[AgentHandoff, SwarmConfig], Coroutine[AgentHandoff]].
Dependencies are injected so stages are testable without live services.

Teammate assignments:
  GATHER         → Scout   (evidence collection)
  DEBATE         → Sage    (conviction + counterargument)
  FACT_CHECK     → Doctor  (claim verification)
  CAUSAL_CHECK   → Oracle  (consequence tracing)
  CONVICTION_GATE → Sage + adversary (conflict resolution)

SwarmContext travels through handoff.payload["_ctx"] across every stage.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from cognitive_fsm import AgentHandoff, HandoffStatus, SwarmConfig
from questioner import SocraticQuestioner
from swarm import (
    SwarmContext,
    get_rep,
    record_error,
    record_success,
    resolve_conflict,
)

# ── Type aliases for injected dependencies ──────────────────────────

MemoriesFn  = Callable[[str], Awaitable[List[str]]]
WorldCtxFn  = Callable[[], Awaitable[str]]
TeammateCtxFn = Callable[[str], Optional[str]]          # slug → system-prompt block or None
LLMChatFn   = Callable[[List[Dict[str, str]]], Awaitable[str]]
BuildPlanFn = Callable[[str, str, List[Dict[str, Any]]], Dict[str, Any]]
ScoreFn     = Callable[[str, Dict[str, Any], List[Dict[str, Any]], int], float]
AdversaryFn = Callable[..., Awaitable[Any]]             # challenge_plan signature


# ── Helper: extract context from payload ────────────────────────────

def _ctx_from(handoff: AgentHandoff) -> SwarmContext:
    """Pull SwarmContext from handoff payload; raise if missing."""
    ctx = handoff.payload.get("_ctx")
    if not isinstance(ctx, SwarmContext):
        raise ValueError("handoff.payload['_ctx'] must be a SwarmContext")
    return ctx


# ── QUESTIONER stage (D92: Socratic pre-GATHER) ─────────────────────

def make_questioner_stage(
    questioner: SocraticQuestioner,
) -> Callable[[AgentHandoff, SwarmConfig], Awaitable[AgentHandoff]]:
    """Run SocraticQuestioner before Scout; injects enriched_query into SwarmContext."""

    async def questioner_stage(handoff: AgentHandoff, cfg: SwarmConfig) -> AgentHandoff:
        t0 = time.monotonic()
        ctx = _ctx_from(handoff)

        try:
            result = await questioner.decompose(ctx.query)
            ctx.decomposition_questions = result.questions
            ctx.enriched_query = result.enriched_query
            ctx.log_stage(
                "questioner", "socratic", "complete",
                (time.monotonic() - t0) * 1000, 8.0,
            )
        except Exception as exc:
            ctx.log_stage(
                "questioner", "socratic", "failed",
                (time.monotonic() - t0) * 1000, 0.0,
            )
            # non-fatal: GATHER proceeds with the original query
            import logging
            logging.getLogger("kai.swarm_stages").debug("Questioner failed, continuing: %s", exc)

        return AgentHandoff(
            from_stage="questioner",
            to_stage="gather",
            status=HandoffStatus.COMPLETE,
            confidence=handoff.confidence,
            payload={**handoff.payload, "_ctx": ctx},
            claims=handoff.claims,
        )

    return questioner_stage


# ── GATHER stage (Scout) ─────────────────────────────────────────────

def make_gather_stage(
    memories_fn: MemoriesFn,
    world_ctx_fn: WorldCtxFn,
    teammate_ctx_fn: TeammateCtxFn,
    llm_chat_fn: LLMChatFn,
) -> Callable[[AgentHandoff, SwarmConfig], Awaitable[AgentHandoff]]:
    """Scout leads: parallel memory + world fetch, LLM extracts structured claims."""

    async def gather(handoff: AgentHandoff, cfg: SwarmConfig) -> AgentHandoff:
        t0 = time.monotonic()
        ctx = _ctx_from(handoff)
        slug = "scout"

        try:
            memories, world = await asyncio.gather(
                memories_fn(ctx.query),
                world_ctx_fn(),
            )

            evidence_items = [{"source": "memory", "content": m} for m in memories]
            if world:
                evidence_items.append({"source": "world_state", "content": world})
            ctx.evidence.extend(evidence_items)

            scout_sys = teammate_ctx_fn(slug) or "You are Scout, a diligent evidence collector."
            evidence_text = "\n".join(
                f"- [{e['source']}] {e['content']}" for e in evidence_items
            ) or "(no evidence available)"

            messages = [
                {"role": "system", "content": scout_sys},
                {
                    "role": "user",
                    "content": (
                        f"Query: {ctx.query}\n\n"
                        f"Evidence gathered:\n{evidence_text}\n\n"
                        "Extract up to 5 specific, falsifiable claims from this evidence. "
                        "Return ONLY a JSON array of claim strings."
                    ),
                },
            ]
            raw = await llm_chat_fn(messages)

            try:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                claims: List[str] = json.loads(raw[start:end]) if start >= 0 else []
                claims = [c for c in claims if isinstance(c, str)][:5]
            except Exception:
                claims = []

            ctx.claims.extend(claims)
            confidence = min(10.0, len(evidence_items) * 1.5 + len(claims) * 0.5)
            record_success(slug, confidence)
            ctx.log_stage("gather", slug, "complete", (time.monotonic() - t0) * 1000, confidence)

            return AgentHandoff(
                from_stage="gather",
                to_stage="debate",
                status=HandoffStatus.COMPLETE,
                confidence=round(confidence, 2),
                payload={**handoff.payload, "_ctx": ctx},
                claims=[{"claim": c} for c in claims],
            )

        except Exception as exc:
            record_error(slug)
            ctx.log_stage("gather", slug, "failed", (time.monotonic() - t0) * 1000, 0.0)
            return AgentHandoff(
                from_stage="gather",
                to_stage="debate",
                status=HandoffStatus.FAILED,
                confidence=0.0,
                payload={**handoff.payload, "_ctx": ctx},
                halt_reason=f"GATHER failed: {exc}",
            )

    return gather


# ── DEBATE stage (Sage) ──────────────────────────────────────────────

def make_debate_stage(
    build_plan_fn: BuildPlanFn,
    score_fn: ScoreFn,
    teammate_ctx_fn: TeammateCtxFn,
    llm_chat_fn: LLMChatFn,
) -> Callable[[AgentHandoff, SwarmConfig], Awaitable[AgentHandoff]]:
    """Sage leads: build a plan, score conviction, generate a structured counterargument."""

    async def debate(handoff: AgentHandoff, cfg: SwarmConfig) -> AgentHandoff:
        t0 = time.monotonic()
        ctx = _ctx_from(handoff)
        slug = "sage"
        rethink_count = handoff.loop_count

        try:
            chunks = [{"content": e["content"]} for e in ctx.evidence]
            plan = build_plan_fn(ctx.query, "DeepSeek-V4", chunks)
            conviction = score_fn(ctx.query, plan, chunks, rethink_count)
            ctx.teammate_votes[slug] = conviction

            sage_sys = teammate_ctx_fn(slug) or "You are Sage, a rigorous critical thinker."
            claims_text = "\n".join(f"- {c}" for c in ctx.claims) or "(no claims yet)"

            messages = [
                {"role": "system", "content": sage_sys},
                {
                    "role": "user",
                    "content": (
                        f"Query: {ctx.query}\n\n"
                        f"Current claims:\n{claims_text}\n\n"
                        f"Plan conviction score: {conviction}/10\n\n"
                        "Challenge these claims: identify the weakest point, propose a "
                        "stronger alternative framing, and state whether claims are "
                        "CONSENSUS or CONTESTED. Reply in ≤3 sentences."
                    ),
                },
            ]
            counter = await llm_chat_fn(messages)
            ctx.challenges.append(counter)

            status = HandoffStatus.CONSENSUS if conviction >= 6.0 else HandoffStatus.NO_CONSENSUS
            record_success(slug, conviction)
            ctx.log_stage("debate", slug, status.value, (time.monotonic() - t0) * 1000, conviction)

            return AgentHandoff(
                from_stage="debate",
                to_stage="fact_check",
                status=status,
                confidence=round(conviction, 2),
                payload={**handoff.payload, "_ctx": ctx},
                claims=handoff.claims,
            )

        except Exception as exc:
            record_error(slug)
            ctx.log_stage("debate", slug, "failed", (time.monotonic() - t0) * 1000, 0.0)
            return AgentHandoff(
                from_stage="debate",
                to_stage="fact_check",
                status=HandoffStatus.FAILED,
                confidence=0.0,
                payload={**handoff.payload, "_ctx": ctx},
                halt_reason=f"DEBATE failed: {exc}",
            )

    return debate


# ── FACT_CHECK stage (Doctor) ────────────────────────────────────────

def make_fact_check_stage(
    memories_fn: MemoriesFn,
    teammate_ctx_fn: TeammateCtxFn,
    llm_chat_fn: LLMChatFn,
) -> Callable[[AgentHandoff, SwarmConfig], Awaitable[AgentHandoff]]:
    """Doctor leads: verify each claim against retrieved evidence; write verdicts dict."""

    async def fact_check(handoff: AgentHandoff, cfg: SwarmConfig) -> AgentHandoff:
        t0 = time.monotonic()
        ctx = _ctx_from(handoff)
        slug = "doctor"

        try:
            support_memories = await memories_fn(ctx.query)
            evidence_text = "\n".join(f"- {m}" for m in support_memories) or "(no supporting evidence)"

            doctor_sys = teammate_ctx_fn(slug) or "You are Doctor, a meticulous fact-checker."
            claims_to_check = ctx.claims[:5] or [ctx.query]
            claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims_to_check))

            messages = [
                {"role": "system", "content": doctor_sys},
                {
                    "role": "user",
                    "content": (
                        f"Evidence base:\n{evidence_text}\n\n"
                        f"Claims to verify:\n{claims_text}\n\n"
                        "For each numbered claim reply with exactly one of: "
                        "supported | unsupported | uncertain. "
                        "Return a JSON object mapping claim text to verdict. "
                        "Example: {\"claim text\": \"supported\"}. Only JSON."
                    ),
                },
            ]
            raw = await llm_chat_fn(messages)

            try:
                start = raw.find("{")
                end = raw.rfind("}") + 1
                verdicts: Dict[str, str] = json.loads(raw[start:end]) if start >= 0 else {}
                valid_vals = {"supported", "unsupported", "uncertain"}
                verdicts = {k: v for k, v in verdicts.items() if v in valid_vals}
            except Exception:
                verdicts = {}

            if not verdicts:
                for c in claims_to_check:
                    verdicts[c] = "uncertain"

            ctx.verdicts.update(verdicts)
            supported_count = sum(1 for v in verdicts.values() if v == "supported")
            total = len(verdicts)
            confidence = round((supported_count / total) * 10.0, 2) if total else 5.0
            status = HandoffStatus.PASS if confidence >= 4.0 else HandoffStatus.FAIL

            record_success(slug, confidence)
            ctx.log_stage("fact_check", slug, status.value, (time.monotonic() - t0) * 1000, confidence)

            return AgentHandoff(
                from_stage="fact_check",
                to_stage="causal_check",
                status=status,
                confidence=confidence,
                payload={**handoff.payload, "_ctx": ctx},
                claims=handoff.claims,
            )

        except Exception as exc:
            record_error(slug)
            ctx.log_stage("fact_check", slug, "failed", (time.monotonic() - t0) * 1000, 0.0)
            return AgentHandoff(
                from_stage="fact_check",
                to_stage="causal_check",
                status=HandoffStatus.FAIL,
                confidence=0.0,
                payload={**handoff.payload, "_ctx": ctx},
                halt_reason=f"FACT_CHECK failed: {exc}",
            )

    return fact_check


# ── CAUSAL_CHECK stage (Oracle) ──────────────────────────────────────

def make_causal_check_stage(
    teammate_ctx_fn: TeammateCtxFn,
    llm_chat_fn: LLMChatFn,
) -> Callable[[AgentHandoff, SwarmConfig], Awaitable[AgentHandoff]]:
    """Oracle leads: trace consequence chains for each supported claim."""

    async def causal_check(handoff: AgentHandoff, cfg: SwarmConfig) -> AgentHandoff:
        t0 = time.monotonic()
        ctx = _ctx_from(handoff)
        slug = "oracle"

        try:
            supported_claims = [
                claim for claim, verdict in ctx.verdicts.items()
                if verdict == "supported"
            ] or ctx.claims[:3] or [ctx.query]

            oracle_sys = teammate_ctx_fn(slug) or "You are Oracle, a systems-thinking causal analyst."
            claims_text = "\n".join(f"- {c}" for c in supported_claims[:3])

            messages = [
                {"role": "system", "content": oracle_sys},
                {
                    "role": "user",
                    "content": (
                        f"Query: {ctx.query}\n\n"
                        f"Supported claims:\n{claims_text}\n\n"
                        "For each claim, trace the most likely causal chain: "
                        "what does this imply → what does that imply → final consequence. "
                        "Return a JSON array of causal chain strings (one per claim). Only JSON."
                    ),
                },
            ]
            raw = await llm_chat_fn(messages)

            try:
                start = raw.find("[")
                end = raw.rfind("]") + 1
                chains: List[str] = json.loads(raw[start:end]) if start >= 0 else []
                chains = [c for c in chains if isinstance(c, str)][:3]
            except Exception:
                chains = []

            ctx.causal_chains.extend(chains)
            confidence = min(10.0, 5.0 + len(chains) * 1.5)
            record_success(slug, confidence)
            ctx.log_stage("causal_check", slug, "complete", (time.monotonic() - t0) * 1000, confidence)

            return AgentHandoff(
                from_stage="causal_check",
                to_stage="conviction_gate",
                status=HandoffStatus.COMPLETE,
                confidence=round(confidence, 2),
                payload={**handoff.payload, "_ctx": ctx},
                claims=handoff.claims,
            )

        except Exception as exc:
            record_error(slug)
            ctx.log_stage("causal_check", slug, "failed", (time.monotonic() - t0) * 1000, 0.0)
            return AgentHandoff(
                from_stage="causal_check",
                to_stage="conviction_gate",
                status=HandoffStatus.COMPLETE,
                confidence=5.0,
                payload={**handoff.payload, "_ctx": ctx},
            )

    return causal_check


# ── CONVICTION_GATE stage (Sage + adversary) ─────────────────────────

def make_conviction_gate_stage(
    adversary_fn: AdversaryFn,
    teammate_ctx_fn: TeammateCtxFn,
) -> Callable[[AgentHandoff, SwarmConfig], Awaitable[AgentHandoff]]:
    """Run adversary challenge_plan then resolve_conflict to produce final score."""

    async def conviction_gate(handoff: AgentHandoff, cfg: SwarmConfig) -> AgentHandoff:
        t0 = time.monotonic()
        ctx = _ctx_from(handoff)

        try:
            chunks = [{"content": e["content"]} for e in ctx.evidence]
            plan = {
                "specialist": "DeepSeek-V4",
                "summary": f"Swarm analysis for: {ctx.query}",
                "steps": [{"action": "analyze", "input": ctx.query}],
                "claims": ctx.claims,
            }

            verdict = await adversary_fn(
                plan=plan,
                user_input=ctx.query,
                context_chunks=chunks,
                episodes=[],
                predicted_conviction=handoff.confidence,
            )

            final_score = resolve_conflict(ctx, cfg, adversary_modifier=verdict.total_modifier)

            ctx.log_stage(
                "conviction_gate", "sage", "complete",
                (time.monotonic() - t0) * 1000, final_score,
            )

            return AgentHandoff(
                from_stage="conviction_gate",
                to_stage="present",
                status=HandoffStatus.COMPLETE,
                confidence=final_score,
                payload={
                    **handoff.payload,
                    "_ctx": ctx,
                    "adversary_summary": verdict.summary,
                    "adversary_recommendation": verdict.recommendation,
                    "conviction_score": final_score,
                },
                claims=handoff.claims,
            )

        except Exception as exc:
            ctx.log_stage("conviction_gate", "sage", "failed", (time.monotonic() - t0) * 1000, 0.0)
            return AgentHandoff(
                from_stage="conviction_gate",
                to_stage="present",
                status=HandoffStatus.COMPLETE,
                confidence=handoff.confidence,
                payload={**handoff.payload, "_ctx": ctx},
                claims=handoff.claims,
            )

    return conviction_gate


# ── Convenience: build all five stages at once ───────────────────────

def build_swarm_pipeline(
    memories_fn: MemoriesFn,
    world_ctx_fn: WorldCtxFn,
    teammate_ctx_fn: TeammateCtxFn,
    llm_chat_fn: LLMChatFn,
    build_plan_fn: BuildPlanFn,
    score_fn: ScoreFn,
    adversary_fn: AdversaryFn,
    questioner: Optional[SocraticQuestioner] = None,
) -> Dict[str, Any]:
    """Return all stage functions ready to pass to CognitiveFSM.run()."""
    pipeline: Dict[str, Any] = {
        "gather_fn": make_gather_stage(memories_fn, world_ctx_fn, teammate_ctx_fn, llm_chat_fn),
        "debate_fn": make_debate_stage(build_plan_fn, score_fn, teammate_ctx_fn, llm_chat_fn),
        "fact_check_fn": make_fact_check_stage(memories_fn, teammate_ctx_fn, llm_chat_fn),
        "causal_check_fn": make_causal_check_stage(teammate_ctx_fn, llm_chat_fn),
        "conviction_gate_fn": make_conviction_gate_stage(adversary_fn, teammate_ctx_fn),
    }
    if questioner is not None:
        pipeline["questioner_fn"] = make_questioner_stage(questioner)
    return pipeline
