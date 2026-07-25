"""Tests for D93: Autonomous Hypothesis Engine (hypothesis.py)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from hypothesis import Hypothesis, HypothesisEngine, _append_to_log


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Hypothesis dataclass ──────────────────────────────────────────────

def test_hypothesis_defaults():
    h = Hypothesis(
        statement="If X is true, Y should follow.",
        basis_memory="memory about X",
        test_predicate="Y should follow",
    )
    assert h.result == "untested"
    assert h.confidence == 0.0
    assert h.formed_at > 0


# ── HypothesisEngine — no LLM, no memories ───────────────────────────

def test_run_cycle_empty_seeds():
    engine = HypothesisEngine()
    result = run(engine.run_cycle([]))
    assert result == []


def test_run_cycle_no_llm_generates_fallback_hypothesis():
    engine = HypothesisEngine(llm_chat_fn=None)
    result = run(engine.run_cycle(["memory about sensor anomaly"]))
    assert len(result) == 1
    h = result[0]
    assert isinstance(h, Hypothesis)
    assert "sensor anomaly" in h.basis_memory
    assert h.result in ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "untested")


def test_run_cycle_caps_at_max_hypotheses():
    engine = HypothesisEngine(llm_chat_fn=None)
    seeds = [f"memory topic {i}" for i in range(10)]
    result = run(engine.run_cycle(seeds))
    assert len(result) <= engine.MAX_HYPOTHESES_PER_CYCLE


# ── HypothesisEngine — with LLM ──────────────────────────────────────

def test_run_cycle_llm_forms_hypothesis():
    llm_response = "If the anomaly pattern holds, then future readings should show the same deviation."
    mock_llm = AsyncMock(return_value=llm_response)
    engine = HypothesisEngine(llm_chat_fn=mock_llm)
    result = run(engine.run_cycle(["temperature sensor behaving oddly"]))
    assert len(result) == 1
    assert "deviation" in result[0].statement or "anomaly" in result[0].statement.lower()


def test_run_cycle_llm_falls_back_on_empty_response():
    mock_llm = AsyncMock(return_value="")
    engine = HypothesisEngine(llm_chat_fn=mock_llm)
    result = run(engine.run_cycle(["some memory"]))
    # empty LLM response → None from _form_hypothesis → no hypothesis in result
    assert len(result) == 0


def test_run_cycle_llm_fails_gracefully():
    mock_llm = AsyncMock(side_effect=ConnectionError("LLM down"))
    engine = HypothesisEngine(llm_chat_fn=mock_llm)
    result = run(engine.run_cycle(["memory X"]))
    # LLM fails → fallback hypothesis via None-LLM path → still tested
    assert isinstance(result, list)


# ── _test_hypothesis — with memories ─────────────────────────────────

def test_test_hypothesis_inconclusive_no_evidence():
    engine = HypothesisEngine(memories_fn=AsyncMock(return_value=[]))
    h = Hypothesis(
        statement="If X holds, Y follows.",
        basis_memory="X observation",
        test_predicate="Y follows",
    )
    tested = run(engine._test_hypothesis(h))
    assert tested.result == "INCONCLUSIVE"
    assert tested.confidence > 0


def test_test_hypothesis_with_evidence_no_llm():
    memories = AsyncMock(return_value=["evidence A", "evidence B"])
    engine = HypothesisEngine(llm_chat_fn=None, memories_fn=memories)
    h = Hypothesis(
        statement="If heat rises, pressure increases.",
        basis_memory="heat observation",
        test_predicate="pressure increases",
    )
    tested = run(engine._test_hypothesis(h))
    assert tested.result == "INCONCLUSIVE"  # no LLM to adjudicate
    assert tested.confidence == 5.0


def test_test_hypothesis_llm_verdict_supported():
    memories = AsyncMock(return_value=["supporting evidence here"])
    mock_llm = AsyncMock(return_value="SUPPORTED — two memory entries confirm the link.")
    engine = HypothesisEngine(llm_chat_fn=mock_llm, memories_fn=memories)
    h = Hypothesis(
        statement="If X holds, Y follows.",
        basis_memory="X",
        test_predicate="Y",
    )
    tested = run(engine._test_hypothesis(h))
    assert tested.result == "SUPPORTED"
    assert tested.confidence == 8.0


def test_test_hypothesis_llm_verdict_refuted():
    memories = AsyncMock(return_value=["contradicting evidence"])
    mock_llm = AsyncMock(return_value="REFUTED — evidence directly contradicts the claim.")
    engine = HypothesisEngine(llm_chat_fn=mock_llm, memories_fn=memories)
    h = Hypothesis(statement="X", basis_memory="X", test_predicate="X")
    tested = run(engine._test_hypothesis(h))
    assert tested.result == "REFUTED"
    assert tested.confidence == 7.0


# ── _append_to_log ────────────────────────────────────────────────────

def test_append_to_log_does_not_raise(tmp_path, monkeypatch):
    import hypothesis as hyp_mod
    monkeypatch.setattr(hyp_mod, "CURIOSITY_LOG", tmp_path / "CURIOSITY.md")
    h = Hypothesis(
        statement="If A, then B.",
        basis_memory="memory A",
        test_predicate="B",
        result="SUPPORTED",
        rationale="evidence confirms",
        confidence=8.0,
    )
    _append_to_log(h)
    content = (tmp_path / "CURIOSITY.md").read_text()
    assert "SUPPORTED" in content
    assert "If A, then B." in content


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
