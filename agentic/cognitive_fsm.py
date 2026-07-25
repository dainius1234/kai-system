"""D89: Cognitive Reasoning Pipeline FSM — the deterministic spine of Kai's multi-agent reasoning.

This is the orchestrator that takes a query through the full reasoning pipeline:
    GATHER → DEBATE → FACT_CHECK → CAUSAL_CHECK → CONVICTION_GATE → PRESENT

Safety guarantees (baked in, never configurable):
    - Every loop has a hard cap (MAX_RETRIES, default 3)
    - Every state has a timeout
    - Exhausted retries → HALT → return to caller for operator escalation
    - All transitions are logged; no silent infinite loops

Schema-validated handoffs between states use AgentHandoff — agents never
communicate through free text, only through typed structured outputs.

Per-swarm configs tune the numbers (timeouts, thresholds) but not the rules.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("kai.cognitive_fsm")


# ── Cognitive states ──────────────────────────────────────────────────

class CogState(str, Enum):
    IDLE = "idle"
    GATHER = "gather"
    DEBATE = "debate"
    FACT_CHECK = "fact_check"
    CAUSAL_CHECK = "causal_check"
    MORAL_IMAGINATION = "moral_imagination"
    CONVICTION_GATE = "conviction_gate"
    RETHINK = "rethink"
    ESCALATE_LOOP = "escalate_loop"
    PRESENT = "present"
    HALT = "halt"  # max retries exhausted — escalate to operator


class HandoffStatus(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"
    DEGRADED = "degraded"  # oracle/reasoning step failed — epistemic absence, not hard failure
    NEEDS_INPUT = "needs_input"
    CONSENSUS = "consensus"
    NO_CONSENSUS = "no_consensus"
    PASS = "pass"
    FAIL = "fail"


# ── Schema-validated agent handoff ────────────────────────────────────

@dataclass
class AgentHandoff:
    """Typed, machine-readable output from each pipeline stage.

    The receiving stage reads status + confidence to determine the
    next transition — never parses free text.
    """
    from_stage: str
    to_stage: str
    status: HandoffStatus
    confidence: float                    # 0.0–10.0
    payload: Dict[str, Any] = field(default_factory=dict)
    claims: List[Dict[str, str]] = field(default_factory=list)
    loop_count: int = 0
    elapsed_ms: float = 0.0
    halt_reason: Optional[str] = None


# ── Per-swarm configuration ───────────────────────────────────────────

@dataclass
class SwarmConfig:
    """Tunable parameters per swarm type. Hard safety rules are not tunable."""
    name: str
    gather_timeout_s: float = 15.0
    debate_timeout_s: float = 20.0
    fact_check_timeout_s: float = 10.0
    causal_check_timeout_s: float = 8.0
    moral_imagination_timeout_s: float = 3.0
    conviction_gate_timeout_s: float = 5.0
    conviction_threshold: float = 7.0     # out of 10
    max_debate_retries: int = 3
    max_rethink_retries: int = 3
    max_escalate_retries: int = 3


# Preset configs for different swarm types
SWARM_CONFIGS: Dict[str, SwarmConfig] = {
    "trading": SwarmConfig(
        name="Trading War Room",
        gather_timeout_s=10.0,
        debate_timeout_s=15.0,
        fact_check_timeout_s=8.0,
        causal_check_timeout_s=6.0,
        conviction_gate_timeout_s=4.0,
        conviction_threshold=8.0,    # higher bar for financial decisions
        max_debate_retries=2,        # faster — market windows close
        max_rethink_retries=2,
        max_escalate_retries=2,
    ),
    "research": SwarmConfig(
        name="Research Swarm",
        gather_timeout_s=30.0,
        debate_timeout_s=45.0,
        fact_check_timeout_s=20.0,
        causal_check_timeout_s=15.0,
        conviction_gate_timeout_s=10.0,
        conviction_threshold=6.5,    # lower bar — depth over certainty
        max_debate_retries=5,        # more iterations for complex topics
        max_rethink_retries=5,
        max_escalate_retries=3,
    ),
    "skill_forge": SwarmConfig(
        name="Skill Forge",
        gather_timeout_s=20.0,
        debate_timeout_s=30.0,
        fact_check_timeout_s=15.0,
        causal_check_timeout_s=10.0,
        conviction_gate_timeout_s=8.0,
        conviction_threshold=6.0,    # iterative — lower threshold
        max_debate_retries=5,
        max_rethink_retries=5,
        max_escalate_retries=3,
    ),
    "default": SwarmConfig(name="Default"),
}


# ── Stage function type ───────────────────────────────────────────────

StageFunc = Callable[[AgentHandoff, SwarmConfig], Coroutine[Any, Any, AgentHandoff]]


# ── Pipeline run result ───────────────────────────────────────────────

@dataclass
class PipelineResult:
    final_state: CogState
    final_handoff: Optional[AgentHandoff]
    halted: bool
    halt_reason: Optional[str]
    total_elapsed_ms: float
    transition_log: List[Dict[str, Any]]


# ── Cognitive FSM pipeline ─────────────────────────────────────────────

class CognitiveFSM:
    """Runs a query through the full cognitive pipeline with timeout + retry guards."""

    def __init__(self, config: Optional[SwarmConfig] = None) -> None:
        self.config = config or SWARM_CONFIGS["default"]
        self._transition_log: List[Dict[str, Any]] = []

    def _log_transition(
        self,
        from_state: CogState,
        to_state: CogState,
        handoff: AgentHandoff,
    ) -> None:
        entry = {
            "from": from_state.value,
            "to": to_state.value,
            "status": handoff.status.value,
            "confidence": handoff.confidence,
            "loop_count": handoff.loop_count,
            "elapsed_ms": handoff.elapsed_ms,
        }
        self._transition_log.append(entry)
        logger.info(
            "CognitiveFSM [%s] %s --[%s/%.1f]--> %s",
            self.config.name,
            from_state.value,
            handoff.status.value,
            handoff.confidence,
            to_state.value,
        )

    async def _run_stage(
        self,
        state: CogState,
        fn: StageFunc,
        handoff: AgentHandoff,
        timeout_s: float,
    ) -> AgentHandoff:
        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(fn(handoff, self.config), timeout=timeout_s)
        except asyncio.TimeoutError:
            result = AgentHandoff(
                from_stage=state.value,
                to_stage="timeout",
                status=HandoffStatus.FAILED,
                confidence=0.0,
                halt_reason=f"{state.value} exceeded timeout of {timeout_s}s",
            )
        result.elapsed_ms = (time.monotonic() - t0) * 1000
        return result

    async def run(
        self,
        gather_fn: StageFunc,
        debate_fn: StageFunc,
        fact_check_fn: StageFunc,
        causal_check_fn: StageFunc,
        conviction_gate_fn: StageFunc,
        moral_imagination_fn: Optional[StageFunc] = None,
        initial_payload: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """Run the full cognitive pipeline. Returns PipelineResult with final state."""
        cfg = self.config
        t_start = time.monotonic()
        self._transition_log = []

        handoff = AgentHandoff(
            from_stage="start",
            to_stage=CogState.GATHER.value,
            status=HandoffStatus.COMPLETE,
            confidence=0.0,
            payload=initial_payload or {},
        )

        # ── GATHER ───────────────────────────────────────────────────
        state = CogState.GATHER
        handoff = await self._run_stage(state, gather_fn, handoff, cfg.gather_timeout_s)
        if handoff.status == HandoffStatus.FAILED:
            return self._halt(state, handoff, t_start)

        # ── DEBATE (with ESCALATE_LOOP retry) ────────────────────────
        debate_retries = 0
        while True:
            state = CogState.DEBATE
            handoff = await self._run_stage(state, debate_fn, handoff, cfg.debate_timeout_s)
            if handoff.status in (HandoffStatus.CONSENSUS, HandoffStatus.COMPLETE):
                self._log_transition(CogState.DEBATE, CogState.FACT_CHECK, handoff)
                break
            state = CogState.ESCALATE_LOOP
            debate_retries += 1
            handoff.loop_count = debate_retries
            if debate_retries >= cfg.max_debate_retries:
                handoff.halt_reason = f"Debate failed to reach consensus after {debate_retries} attempts"
                return self._halt(CogState.ESCALATE_LOOP, handoff, t_start)
            self._log_transition(CogState.DEBATE, CogState.ESCALATE_LOOP, handoff)
            logger.warning("Debate escalation loop %d/%d", debate_retries, cfg.max_debate_retries)

        # ── FACT_CHECK ───────────────────────────────────────────────
        state = CogState.FACT_CHECK
        handoff = await self._run_stage(state, fact_check_fn, handoff, cfg.fact_check_timeout_s)
        if handoff.status == HandoffStatus.FAIL:
            self._log_transition(CogState.FACT_CHECK, CogState.GATHER, handoff)
            handoff = await self._run_stage(CogState.GATHER, gather_fn, handoff, cfg.gather_timeout_s)
            if handoff.status == HandoffStatus.FAILED:
                return self._halt(CogState.GATHER, handoff, t_start)
        else:
            self._log_transition(CogState.FACT_CHECK, CogState.CAUSAL_CHECK, handoff)

        # ── CAUSAL_CHECK ─────────────────────────────────────────────
        state = CogState.CAUSAL_CHECK
        handoff = await self._run_stage(state, causal_check_fn, handoff, cfg.causal_check_timeout_s)

        # ── MORAL_IMAGINATION (optional, FF-gated) ────────────────────
        if moral_imagination_fn is not None:
            self._log_transition(CogState.CAUSAL_CHECK, CogState.MORAL_IMAGINATION, handoff)
            state = CogState.MORAL_IMAGINATION
            handoff = await self._run_stage(
                state, moral_imagination_fn, handoff, cfg.moral_imagination_timeout_s
            )
            self._log_transition(CogState.MORAL_IMAGINATION, CogState.CONVICTION_GATE, handoff)
        else:
            self._log_transition(CogState.CAUSAL_CHECK, CogState.CONVICTION_GATE, handoff)

        # ── CONVICTION_GATE (with RETHINK retry) ─────────────────────
        rethink_retries = 0
        while True:
            state = CogState.CONVICTION_GATE
            handoff = await self._run_stage(state, conviction_gate_fn, handoff, cfg.conviction_gate_timeout_s)
            if handoff.confidence >= cfg.conviction_threshold:
                self._log_transition(CogState.CONVICTION_GATE, CogState.PRESENT, handoff)
                break
            state = CogState.RETHINK
            rethink_retries += 1
            handoff.loop_count = rethink_retries
            if rethink_retries >= cfg.max_rethink_retries:
                handoff.halt_reason = (
                    f"Conviction {handoff.confidence:.1f} < threshold {cfg.conviction_threshold} "
                    f"after {rethink_retries} rethink cycles"
                )
                return self._halt(CogState.RETHINK, handoff, t_start)
            self._log_transition(CogState.CONVICTION_GATE, CogState.RETHINK, handoff)
            logger.warning(
                "Rethink %d/%d (conviction %.1f < %.1f)",
                rethink_retries, cfg.max_rethink_retries,
                handoff.confidence, cfg.conviction_threshold,
            )

        total_ms = (time.monotonic() - t_start) * 1000
        return PipelineResult(
            final_state=CogState.PRESENT,
            final_handoff=handoff,
            halted=False,
            halt_reason=None,
            total_elapsed_ms=total_ms,
            transition_log=list(self._transition_log),
        )

    def _halt(
        self,
        state: CogState,
        handoff: AgentHandoff,
        t_start: float,
    ) -> PipelineResult:
        self._log_transition(state, CogState.HALT, handoff)
        logger.error("CognitiveFSM HALT from %s: %s", state.value, handoff.halt_reason)
        return PipelineResult(
            final_state=CogState.HALT,
            final_handoff=handoff,
            halted=True,
            halt_reason=handoff.halt_reason,
            total_elapsed_ms=(time.monotonic() - t_start) * 1000,
            transition_log=list(self._transition_log),
        )


def get_config(swarm_type: str) -> SwarmConfig:
    """Return the SwarmConfig for the named swarm type (falls back to 'default')."""
    return SWARM_CONFIGS.get(swarm_type, SWARM_CONFIGS["default"])
