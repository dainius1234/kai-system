"""D90 Swarm Assembly — test suite.

Tests cover:
    - SwarmContext accumulation (evidence, claims, verdicts, causal chains)
    - TeammateRep reputation math (weight, reliability, avg_confidence)
    - load/save/get_rep/record_success/record_error/list_reputation
    - resolve_conflict() priority hierarchy
    - make_gather_stage / make_debate_stage / make_fact_check_stage
    - make_causal_check_stage / make_conviction_gate_stage
    - build_swarm_pipeline (end-to-end via CognitiveFSM)
    - Feature flag FF_SWARM
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "agentic"))
sys.path.insert(0, str(_ROOT / "common"))


# ══════════════════════════════════════════════════════════════════════
# SwarmContext
# ══════════════════════════════════════════════════════════════════════

class TestSwarmContext:
    def _make(self):
        from swarm import SwarmContext
        return SwarmContext(query="test query", session_id="s1", swarm_type="default")

    def test_initial_empty(self):
        ctx = self._make()
        assert ctx.evidence == []
        assert ctx.claims == []
        assert ctx.verdicts == {}
        assert ctx.causal_chains == []

    def test_log_stage(self):
        ctx = self._make()
        ctx.log_stage("gather", "scout", "complete", 123.4, 7.5)
        assert len(ctx.stage_log) == 1
        entry = ctx.stage_log[0]
        assert entry["stage"] == "gather"
        assert entry["teammate"] == "scout"
        assert entry["elapsed_ms"] == 123.4
        assert entry["confidence"] == 7.5

    def test_summary_counts(self):
        ctx = self._make()
        ctx.evidence.append({"source": "memory", "content": "fact1"})
        ctx.claims.extend(["claim A", "claim B"])
        ctx.verdicts["claim A"] = "supported"
        ctx.causal_chains.append("A → B → C")
        ctx.teammate_votes["scout"] = 7.0
        s = ctx.summary()
        assert s["evidence_count"] == 1
        assert s["claim_count"] == 2
        assert s["verdict_count"] == 1
        assert s["causal_chain_count"] == 1


# ══════════════════════════════════════════════════════════════════════
# TeammateRep
# ══════════════════════════════════════════════════════════════════════

class TestTeammateRep:
    def test_initial_weight_is_zero(self):
        from swarm import TeammateRep
        rep = TeammateRep(slug="scout")
        assert rep.weight() == 0.0

    def test_avg_confidence_no_calls(self):
        from swarm import TeammateRep
        rep = TeammateRep(slug="scout")
        assert rep.avg_confidence == 0.0

    def test_reliability_after_calls(self):
        from swarm import TeammateRep
        rep = TeammateRep(slug="scout", total_calls=4, successful_handoffs=3, total_confidence=21.0)
        assert rep.reliability == 0.75
        assert rep.avg_confidence == 7.0

    def test_weight_formula(self):
        from swarm import TeammateRep
        rep = TeammateRep(slug="scout", total_calls=2, successful_handoffs=2, total_confidence=16.0)
        # reliability = 1.0, avg_confidence = 8.0 → weight = 1.0 * 0.8 = 0.8
        assert abs(rep.weight() - 0.8) < 0.001

    def test_error_count_lowers_reliability(self):
        from swarm import TeammateRep
        rep = TeammateRep(slug="scout", total_calls=4, successful_handoffs=2, total_confidence=10.0, error_count=2)
        assert rep.reliability == 0.5


# ══════════════════════════════════════════════════════════════════════
# Reputation management
# ══════════════════════════════════════════════════════════════════════

class TestReputation:
    def test_record_success_increments(self):
        import swarm as sw
        sw._REPUTATION = {}
        sw.record_success("scout", 8.0)
        rep = sw._REPUTATION["scout"]
        assert rep.total_calls == 1
        assert rep.successful_handoffs == 1
        assert rep.total_confidence == 8.0

    def test_record_error_increments(self):
        import swarm as sw
        sw._REPUTATION = {}
        sw.record_error("doctor")
        rep = sw._REPUTATION["doctor"]
        assert rep.total_calls == 1
        assert rep.error_count == 1
        assert rep.successful_handoffs == 0

    def test_get_rep_creates_on_miss(self):
        import swarm as sw
        sw._REPUTATION = {}
        rep = sw.get_rep("oracle")
        assert rep.slug == "oracle"
        assert rep.total_calls == 0

    def test_list_reputation_returns_all(self):
        import swarm as sw
        sw._REPUTATION = {}
        sw.record_success("scout", 7.0)
        sw.record_error("doctor")
        listing = sw.list_reputation()
        slugs = {r["slug"] for r in listing}
        assert "scout" in slugs
        assert "doctor" in slugs

    def test_save_and_load_round_trip(self, tmp_path):
        import swarm as sw
        sw._REPUTATION = {}
        sw.record_success("sage", 9.0)
        with patch.object(sw, "REPUTATION_PATH", tmp_path / "rep.json"):
            sw.save_reputation()
            sw._REPUTATION = {}
            sw.load_reputation()
        assert "sage" in sw._REPUTATION
        assert sw._REPUTATION["sage"].total_calls == 1

    def test_load_handles_missing_file(self, tmp_path):
        import swarm as sw
        with patch.object(sw, "REPUTATION_PATH", tmp_path / "nonexistent.json"):
            sw.load_reputation()
        assert isinstance(sw._REPUTATION, dict)


# ══════════════════════════════════════════════════════════════════════
# resolve_conflict
# ══════════════════════════════════════════════════════════════════════

class TestResolveConflict:
    def _make_ctx(self):
        from swarm import SwarmContext
        return SwarmContext(query="q", session_id="s", swarm_type="default")

    def test_empty_context_gives_mid_score(self):
        from swarm import resolve_conflict, SwarmContext
        from cognitive_fsm import SWARM_CONFIGS
        ctx = self._make_ctx()
        score = resolve_conflict(ctx, SWARM_CONFIGS["default"])
        # evidence=0 → 0, causal=0 → 0, verdict=5.0 neutral, vote=5.0, skeptic=5.0
        # final = 0*0.30 + 0*0.25 + 5.0*0.20 + 5.0*0.15 + 5.0*0.10 = 2.25
        assert 0.0 <= score <= 10.0

    def test_rich_evidence_raises_score(self):
        from swarm import resolve_conflict, SwarmContext
        from cognitive_fsm import SWARM_CONFIGS
        import swarm as sw
        sw._REPUTATION = {}
        ctx = self._make_ctx()
        for i in range(5):
            ctx.evidence.append({"source": "mem", "content": f"fact{i}"})
        ctx.causal_chains.extend(["A→B→C", "X→Y→Z"])
        ctx.verdicts = {"c1": "supported", "c2": "supported"}
        score = resolve_conflict(ctx, SWARM_CONFIGS["default"])
        assert score > 5.0

    def test_adversary_negative_modifier_lowers_score(self):
        from swarm import resolve_conflict
        from cognitive_fsm import SWARM_CONFIGS
        ctx = self._make_ctx()
        score_pos = resolve_conflict(ctx, SWARM_CONFIGS["default"], adversary_modifier=1.0)
        score_neg = resolve_conflict(ctx, SWARM_CONFIGS["default"], adversary_modifier=-3.0)
        assert score_pos > score_neg

    def test_score_clamped_0_to_10(self):
        from swarm import resolve_conflict
        from cognitive_fsm import SWARM_CONFIGS
        ctx = self._make_ctx()
        for _ in range(20):
            ctx.evidence.append({"source": "m", "content": "x"})
            ctx.causal_chains.append("a→b")
        ctx.verdicts = {f"c{i}": "supported" for i in range(10)}
        score = resolve_conflict(ctx, SWARM_CONFIGS["default"], adversary_modifier=1.0)
        assert 0.0 <= score <= 10.0

    def test_reputation_weight_used(self):
        import swarm as sw
        from cognitive_fsm import SWARM_CONFIGS
        sw._REPUTATION = {}
        sw.record_success("scout", 9.0)
        ctx = sw.SwarmContext(query="q", session_id="s", swarm_type="default")
        ctx.teammate_votes["scout"] = 9.0
        score = sw.resolve_conflict(ctx, SWARM_CONFIGS["default"])
        assert score >= 0.0  # smoke: reputation path exercised


# ══════════════════════════════════════════════════════════════════════
# Stage functions
# ══════════════════════════════════════════════════════════════════════

def _make_handoff(query: str = "test", swarm_type: str = "default"):
    from swarm import SwarmContext
    from cognitive_fsm import AgentHandoff, HandoffStatus
    ctx = SwarmContext(query=query, session_id="s1", swarm_type=swarm_type)
    return AgentHandoff(
        from_stage="start",
        to_stage="gather",
        status=HandoffStatus.COMPLETE,
        confidence=5.0,
        payload={"_ctx": ctx},
    )


class TestGatherStage:
    @pytest.mark.anyio
    async def test_gather_populates_evidence_and_claims(self):
        from swarm_stages import make_gather_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        memories_fn = AsyncMock(return_value=["mem1", "mem2"])
        world_fn = AsyncMock(return_value="Weather: sunny")
        teammate_fn = MagicMock(return_value="[Scout system prompt]")
        llm_fn = AsyncMock(return_value='["claim A", "claim B"]')

        stage = make_gather_stage(memories_fn, world_fn, teammate_fn, llm_fn)
        handoff = _make_handoff()
        result = await stage(handoff, SWARM_CONFIGS["default"])

        assert result.status == HandoffStatus.COMPLETE
        ctx = result.payload["_ctx"]
        assert len(ctx.evidence) >= 2
        assert len(ctx.claims) == 2

    @pytest.mark.anyio
    async def test_gather_handles_llm_parse_failure(self):
        from swarm_stages import make_gather_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        stage = make_gather_stage(
            AsyncMock(return_value=[]),
            AsyncMock(return_value=""),
            MagicMock(return_value=None),
            AsyncMock(return_value="not json"),
        )
        result = await stage(_make_handoff(), SWARM_CONFIGS["default"])
        assert result.status == HandoffStatus.COMPLETE
        ctx = result.payload["_ctx"]
        assert ctx.claims == []

    @pytest.mark.anyio
    async def test_gather_exception_returns_failed(self):
        from swarm_stages import make_gather_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus
        import swarm as sw
        sw._REPUTATION = {}

        stage = make_gather_stage(
            AsyncMock(side_effect=RuntimeError("boom")),
            AsyncMock(return_value=""),
            MagicMock(return_value=None),
            AsyncMock(return_value="[]"),
        )
        result = await stage(_make_handoff(), SWARM_CONFIGS["default"])
        assert result.status == HandoffStatus.FAILED


class TestDebateStage:
    @pytest.mark.anyio
    async def test_debate_high_conviction_returns_consensus(self):
        from swarm_stages import make_debate_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        build_plan_fn = MagicMock(return_value={"specialist": "X", "summary": "s", "steps": [{"action": "a"}]})
        score_fn = MagicMock(return_value=8.0)
        teammate_fn = MagicMock(return_value="[Sage]")
        llm_fn = AsyncMock(return_value="Claims look CONSENSUS.")

        stage = make_debate_stage(build_plan_fn, score_fn, teammate_fn, llm_fn)
        handoff = _make_handoff()
        handoff.payload["_ctx"].claims = ["claim A"]
        result = await stage(handoff, SWARM_CONFIGS["default"])

        assert result.status == HandoffStatus.CONSENSUS
        assert result.confidence == 8.0

    @pytest.mark.anyio
    async def test_debate_low_conviction_returns_no_consensus(self):
        from swarm_stages import make_debate_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        stage = make_debate_stage(
            MagicMock(return_value={"specialist": "X", "summary": "s", "steps": []}),
            MagicMock(return_value=3.0),
            MagicMock(return_value=None),
            AsyncMock(return_value="CONTESTED"),
        )
        result = await stage(_make_handoff(), SWARM_CONFIGS["default"])
        assert result.status == HandoffStatus.NO_CONSENSUS

    @pytest.mark.anyio
    async def test_debate_records_challenge(self):
        from swarm_stages import make_debate_stage
        from cognitive_fsm import SWARM_CONFIGS

        stage = make_debate_stage(
            MagicMock(return_value={"specialist": "X", "summary": "s", "steps": []}),
            MagicMock(return_value=7.0),
            MagicMock(return_value=None),
            AsyncMock(return_value="Some counter"),
        )
        handoff = _make_handoff()
        await stage(handoff, SWARM_CONFIGS["default"])
        ctx = handoff.payload["_ctx"]
        assert len(ctx.challenges) == 1


class TestFactCheckStage:
    @pytest.mark.anyio
    async def test_fact_check_verdicts_written(self):
        from swarm_stages import make_fact_check_stage
        from cognitive_fsm import SWARM_CONFIGS

        llm_resp = json.dumps({"claim A": "supported", "claim B": "unsupported"})
        stage = make_fact_check_stage(
            AsyncMock(return_value=["evidence1"]),
            MagicMock(return_value="[Doctor]"),
            AsyncMock(return_value=llm_resp),
        )
        handoff = _make_handoff()
        handoff.payload["_ctx"].claims = ["claim A", "claim B"]
        result = await stage(handoff, SWARM_CONFIGS["default"])

        ctx = result.payload["_ctx"]
        assert "claim A" in ctx.verdicts
        assert ctx.verdicts["claim A"] == "supported"

    @pytest.mark.anyio
    async def test_fact_check_pass_when_mostly_supported(self):
        from swarm_stages import make_fact_check_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        verdicts = {"c1": "supported", "c2": "supported", "c3": "supported"}
        stage = make_fact_check_stage(
            AsyncMock(return_value=[]),
            MagicMock(return_value=None),
            AsyncMock(return_value=json.dumps(verdicts)),
        )
        handoff = _make_handoff()
        handoff.payload["_ctx"].claims = ["c1", "c2", "c3"]
        result = await stage(handoff, SWARM_CONFIGS["default"])
        assert result.status == HandoffStatus.PASS

    @pytest.mark.anyio
    async def test_fact_check_fill_uncertain_on_bad_json(self):
        from swarm_stages import make_fact_check_stage
        from cognitive_fsm import SWARM_CONFIGS

        stage = make_fact_check_stage(
            AsyncMock(return_value=[]),
            MagicMock(return_value=None),
            AsyncMock(return_value="not json at all"),
        )
        handoff = _make_handoff()
        handoff.payload["_ctx"].claims = ["claim X"]
        result = await stage(handoff, SWARM_CONFIGS["default"])
        ctx = result.payload["_ctx"]
        assert "claim X" in ctx.verdicts
        assert ctx.verdicts["claim X"] == "uncertain"


class TestCausalCheckStage:
    @pytest.mark.anyio
    async def test_causal_chains_populated(self):
        from swarm_stages import make_causal_check_stage
        from cognitive_fsm import SWARM_CONFIGS

        chains = ["A → B → C", "X → Y → Z"]
        stage = make_causal_check_stage(
            MagicMock(return_value="[Oracle]"),
            AsyncMock(return_value=json.dumps(chains)),
        )
        handoff = _make_handoff()
        handoff.payload["_ctx"].verdicts = {"c1": "supported"}
        result = await stage(handoff, SWARM_CONFIGS["default"])

        ctx = result.payload["_ctx"]
        assert len(ctx.causal_chains) == 2

    @pytest.mark.anyio
    async def test_causal_check_survives_exception(self):
        from swarm_stages import make_causal_check_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        stage = make_causal_check_stage(
            MagicMock(return_value=None),
            AsyncMock(side_effect=RuntimeError("network")),
        )
        result = await stage(_make_handoff(), SWARM_CONFIGS["default"])
        assert result.status == HandoffStatus.DEGRADED
        assert result.confidence == 3.5  # degraded fallback

    @pytest.mark.anyio
    async def test_causal_confidence_scales_with_chain_count(self):
        from swarm_stages import make_causal_check_stage
        from cognitive_fsm import SWARM_CONFIGS

        stage = make_causal_check_stage(
            MagicMock(return_value=None),
            AsyncMock(return_value=json.dumps(["chain1", "chain2", "chain3"])),
        )
        result = await stage(_make_handoff(), SWARM_CONFIGS["default"])
        assert result.confidence > 5.0


class TestConvictionGateStage:
    @pytest.mark.anyio
    async def test_conviction_gate_score_in_payload(self):
        from swarm_stages import make_conviction_gate_stage
        from cognitive_fsm import SWARM_CONFIGS

        mock_verdict = MagicMock()
        mock_verdict.total_modifier = 0.5
        mock_verdict.summary = "5/6 passed"
        mock_verdict.recommendation = "proceed"

        adversary_fn = AsyncMock(return_value=mock_verdict)
        stage = make_conviction_gate_stage(adversary_fn, MagicMock(return_value=None))

        result = await stage(_make_handoff(), SWARM_CONFIGS["default"])
        assert "conviction_score" in result.payload
        assert 0.0 <= result.payload["conviction_score"] <= 10.0

    @pytest.mark.anyio
    async def test_conviction_gate_survives_adversary_failure(self):
        from swarm_stages import make_conviction_gate_stage
        from cognitive_fsm import SWARM_CONFIGS, HandoffStatus

        stage = make_conviction_gate_stage(
            AsyncMock(side_effect=RuntimeError("adversary down")),
            MagicMock(return_value=None),
        )
        handoff = _make_handoff()
        handoff.confidence = 6.0
        result = await stage(handoff, SWARM_CONFIGS["default"])
        assert result.status == HandoffStatus.COMPLETE
        assert result.confidence == 6.0  # fallback to prior handoff confidence


# ══════════════════════════════════════════════════════════════════════
# End-to-end pipeline
# ══════════════════════════════════════════════════════════════════════

def _smart_llm():
    """LLM mock that returns stage-appropriate JSON based on message content."""
    async def _fn(messages):
        last = messages[-1]["content"]
        if "falsifiable claims" in last:
            return '["claim A", "claim B"]'
        if "JSON object mapping claim" in last:
            return '{"claim A": "supported", "claim B": "supported"}'
        if "JSON array of causal" in last:
            return '["A causes B", "B causes C", "C leads to D"]'
        return "CONSENSUS"
    return _fn


class TestSwarmPipelineE2E:
    @pytest.mark.anyio
    async def test_full_pipeline_reaches_present(self):
        from swarm import SwarmContext
        from swarm_stages import build_swarm_pipeline
        from cognitive_fsm import CognitiveFSM, SWARM_CONFIGS, CogState
        import swarm as sw
        sw._REPUTATION = {}

        mock_verdict = MagicMock()
        mock_verdict.total_modifier = 1.0
        mock_verdict.summary = "ok"
        mock_verdict.recommendation = "proceed"

        pipeline = build_swarm_pipeline(
            memories_fn=AsyncMock(return_value=["fact1", "fact2", "fact3", "fact4", "fact5"]),
            world_ctx_fn=AsyncMock(return_value="sunny"),
            teammate_ctx_fn=MagicMock(return_value="[teammate]"),
            llm_chat_fn=_smart_llm(),
            build_plan_fn=MagicMock(return_value={"specialist": "X", "summary": "s", "steps": [{"action": "a"}]}),
            score_fn=MagicMock(return_value=9.0),
            adversary_fn=AsyncMock(return_value=mock_verdict),
        )

        ctx = SwarmContext(query="Is the market going up?", session_id="e2e", swarm_type="default")
        fsm = CognitiveFSM(config=SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=pipeline["gather_fn"],
            debate_fn=pipeline["debate_fn"],
            fact_check_fn=pipeline["fact_check_fn"],
            causal_check_fn=pipeline["causal_check_fn"],
            conviction_gate_fn=pipeline["conviction_gate_fn"],
            initial_payload={"_ctx": ctx},
        )

        assert result.final_state == CogState.PRESENT
        assert not result.halted

    @pytest.mark.anyio
    async def test_pipeline_halts_on_gather_failure(self):
        from swarm import SwarmContext
        from swarm_stages import build_swarm_pipeline
        from cognitive_fsm import CognitiveFSM, SWARM_CONFIGS, CogState
        import swarm as sw
        sw._REPUTATION = {}

        pipeline = build_swarm_pipeline(
            memories_fn=AsyncMock(side_effect=RuntimeError("down")),
            world_ctx_fn=AsyncMock(return_value=""),
            teammate_ctx_fn=MagicMock(return_value=None),
            llm_chat_fn=_smart_llm(),
            build_plan_fn=MagicMock(return_value={}),
            score_fn=MagicMock(return_value=5.0),
            adversary_fn=AsyncMock(return_value=MagicMock(total_modifier=0.0, summary="", recommendation="proceed")),
        )

        ctx = SwarmContext(query="q", session_id="s", swarm_type="default")
        fsm = CognitiveFSM(config=SWARM_CONFIGS["default"])
        result = await fsm.run(
            gather_fn=pipeline["gather_fn"],
            debate_fn=pipeline["debate_fn"],
            fact_check_fn=pipeline["fact_check_fn"],
            causal_check_fn=pipeline["causal_check_fn"],
            conviction_gate_fn=pipeline["conviction_gate_fn"],
            initial_payload={"_ctx": ctx},
        )
        assert result.halted
        assert result.final_state == CogState.HALT


# ══════════════════════════════════════════════════════════════════════
# Feature flag
# ══════════════════════════════════════════════════════════════════════

class TestSwarmFlag:
    def test_swarm_flag_registered(self):
        from feature_flags import get_all_flags
        flags = {f["flag"] for f in get_all_flags()}
        assert "SWARM" in flags

    def test_swarm_flag_default_on(self):
        import os
        os.environ.pop("FF_SWARM", None)
        from feature_flags import is_enabled
        assert is_enabled("SWARM") is True

    def test_swarm_flag_can_be_disabled(self):
        import os
        os.environ["FF_SWARM"] = "false"
        from feature_flags import is_enabled
        result = is_enabled("SWARM")
        os.environ.pop("FF_SWARM", None)
        assert result is False
