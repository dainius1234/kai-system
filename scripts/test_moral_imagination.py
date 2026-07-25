"""Tests for D121: Moral Imagination — agentic/moral_imagination.py."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from moral_imagination import (
    MoralImagination,
    _conviction_modifier,
    _project_goods,
    _project_harms,
    _recommendation,
    _extract_action_text,
    run_moral_imagination,
)
from cognitive_fsm import AgentHandoff, HandoffStatus, SwarmConfig, CogState


def _handoff(confidence=7.0, payload=None) -> AgentHandoff:
    return AgentHandoff(
        from_stage="causal_check",
        to_stage="moral_imagination",
        status=HandoffStatus.COMPLETE,
        confidence=confidence,
        payload=payload or {"query": "send a message to family about finances"},
    )


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── _conviction_modifier ──────────────────────────────────────────────────────

def test_modifier_high_alignment_no_harms():
    assert _conviction_modifier(0.9, 0) == pytest.approx(0.8)


def test_modifier_high_alignment_with_harm():
    assert _conviction_modifier(0.9, 1) < 0.8


def test_modifier_neutral_alignment():
    assert _conviction_modifier(0.6, 0) == pytest.approx(0.2)


def test_modifier_low_alignment():
    m = _conviction_modifier(0.25, 0)
    assert m < 0.0


def test_modifier_very_low_alignment():
    m = _conviction_modifier(0.1, 0)
    assert m <= -1.5


def test_modifier_harms_penalize():
    m_no_harm = _conviction_modifier(0.7, 0)
    m_with_harm = _conviction_modifier(0.7, 1)
    assert m_with_harm < m_no_harm


# ── _recommendation ───────────────────────────────────────────────────────────

def test_recommendation_proceed_high_alignment_no_harms():
    assert _recommendation(0.9, 0) == "proceed"


def test_recommendation_proceed_with_caution_moderate():
    assert _recommendation(0.5, 0) == "proceed_with_caution"


def test_recommendation_reconsider_low_alignment():
    assert _recommendation(0.2, 0) == "reconsider"


def test_recommendation_halt_on_boundary():
    assert _recommendation(0.0, 0) == "halt"


def test_recommendation_halt_on_multiple_harms():
    assert _recommendation(0.7, 2) == "halt"


# ── _project_goods ────────────────────────────────────────────────────────────

def test_project_goods_from_value_nodes():
    node = MagicMock()
    node.node_type = "VALUE"
    node.content = "Family first"
    goods = _project_goods([node], [], "family financial decision")
    assert any("Family first" in g for g in goods)


def test_project_goods_from_principle_nodes():
    node = MagicMock()
    node.node_type = "PRINCIPLE"
    node.content = "Respect is earned"
    goods = _project_goods([node], [], "earn respect through action")
    assert any("Respect is earned" in g for g in goods)


def test_project_goods_caps_at_four():
    nodes = []
    for i in range(10):
        n = MagicMock()
        n.node_type = "VALUE"
        n.content = f"value {i}"
        nodes.append(n)
    goods = _project_goods(nodes, [], "some action")
    assert len(goods) <= 4


def test_project_goods_empty_nodes():
    assert _project_goods([], [], "anything") == []


# ── _project_harms ────────────────────────────────────────────────────────────

def test_project_harms_detects_boundary_match():
    node = MagicMock()
    node.node_type = "BOUNDARY"
    node.content = "Never reveal api key"
    node.word_set = MagicMock(return_value={"never", "reveal", "api", "key"})
    harms = _project_harms("expose the api key to dashboard", [node], [])
    assert len(harms) >= 1
    assert any("api key" in h.lower() for h in harms)


def test_project_harms_no_boundary_match():
    node = MagicMock()
    node.node_type = "BOUNDARY"
    node.content = "Never reveal api key"
    node.word_set = MagicMock(return_value={"never", "reveal", "api", "key"})
    harms = _project_harms("send a birthday message to family", [node], [])
    assert len(harms) == 0


def test_project_harms_caps_at_three():
    nodes = []
    for i in range(5):
        n = MagicMock()
        n.node_type = "BOUNDARY"
        n.content = f"never do thing {i}"
        nodes.append(n)
    harms = _project_harms("do thing 0 thing 1 thing 2 thing 3 thing 4", nodes, [])
    assert len(harms) <= 3


# ── _extract_action_text ─────────────────────────────────────────────────────

def test_extract_uses_query():
    text = _extract_action_text({"query": "buy family groceries"})
    assert "buy family groceries" in text


def test_extract_uses_plan_summary():
    text = _extract_action_text({"query": "q", "plan": {"summary": "send money to daughter"}})
    assert "send money to daughter" in text


def test_extract_caps_at_500():
    long_q = "x" * 1000
    text = _extract_action_text({"query": long_q})
    assert len(text) <= 500


def test_extract_empty_payload():
    text = _extract_action_text({})
    assert isinstance(text, str)


# ── run_moral_imagination ────────────────────────────────────────────────────

def test_run_passes_through_when_infra_unavailable():
    h = _handoff(confidence=7.0)
    with patch("moral_imagination._query_moral_context", side_effect=Exception("no graph")):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert isinstance(result, AgentHandoff)
    assert result.confidence == pytest.approx(7.0, abs=0.1)
    assert result.from_stage == "moral_imagination"


def test_run_adjusts_confidence_on_high_alignment():
    h = _handoff(confidence=7.0)
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.9)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert result.confidence > 7.0


def test_run_reduces_confidence_on_low_alignment():
    h = _handoff(confidence=7.0)
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.1)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert result.confidence < 7.0


def test_run_writes_moral_imagination_to_payload():
    h = _handoff(confidence=7.0)
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.7)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert "moral_imagination" in result.payload
    mi = result.payload["moral_imagination"]
    assert "projected_alignment" in mi
    assert "recommendation" in mi
    assert "imagined_goods" in mi
    assert "imagined_harms" in mi


def test_run_never_exceeds_10_confidence():
    h = _handoff(confidence=9.5)
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.99)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert result.confidence <= 10.0


def test_run_never_goes_below_0_confidence():
    h = _handoff(confidence=0.5)
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.0)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert result.confidence >= 0.0


def test_run_recommendation_halt_on_zero_alignment():
    h = _handoff(confidence=7.0)
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.0)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    mi = result.payload["moral_imagination"]
    assert mi["recommendation"] == "halt"


def test_run_preserves_existing_payload_keys():
    h = _handoff(confidence=7.0, payload={"query": "q", "_ctx": "original_ctx"})
    with patch("moral_imagination._query_moral_context", return_value=([], [], 0.7)):
        result = _run(run_moral_imagination(h, SwarmConfig(name="default")))
    assert result.payload.get("_ctx") == "original_ctx"


# ── CognitiveFSM integration ──────────────────────────────────────────────────

def test_cog_state_has_moral_imagination():
    assert hasattr(CogState, "MORAL_IMAGINATION")
    assert CogState.MORAL_IMAGINATION.value == "moral_imagination"


def test_swarm_config_has_moral_imagination_timeout():
    cfg = SwarmConfig(name="test")
    assert hasattr(cfg, "moral_imagination_timeout_s")
    assert cfg.moral_imagination_timeout_s > 0


def test_fsm_run_accepts_moral_imagination_fn():
    """CognitiveFSM.run() accepts the optional moral_imagination_fn param."""
    from cognitive_fsm import CognitiveFSM, PipelineResult

    async def passthrough(h, cfg):
        return h

    async def _run_fsm():
        fsm = CognitiveFSM(SwarmConfig(name="default"))
        return await fsm.run(
            gather_fn=passthrough,
            debate_fn=passthrough,
            fact_check_fn=passthrough,
            causal_check_fn=passthrough,
            conviction_gate_fn=passthrough,
            moral_imagination_fn=passthrough,
            initial_payload={"query": "test"},
        )

    result = _run(_run_fsm())
    assert isinstance(result, PipelineResult)
    assert CogState.MORAL_IMAGINATION.value in str(result.transition_log)


def test_fsm_skips_moral_imagination_when_fn_is_none():
    from cognitive_fsm import CognitiveFSM

    called = []

    async def passthrough(h, cfg):
        return h

    async def mi_fn(h, cfg):
        called.append(True)
        return h

    async def _run_fsm(use_mi: bool):
        fsm = CognitiveFSM(SwarmConfig(name="default"))
        return await fsm.run(
            gather_fn=passthrough,
            debate_fn=passthrough,
            fact_check_fn=passthrough,
            causal_check_fn=passthrough,
            conviction_gate_fn=passthrough,
            moral_imagination_fn=mi_fn if use_mi else None,
            initial_payload={"query": "test"},
        )

    _run(_run_fsm(False))
    assert len(called) == 0

    _run(_run_fsm(True))
    assert len(called) == 1
