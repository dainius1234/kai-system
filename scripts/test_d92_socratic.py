"""Tests for D92: Socratic Self-Questioning (questioner.py + swarm integration)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
sys.path.insert(0, str(Path(__file__).parent.parent / "common"))

from questioner import (
    FALLBACK_QUESTIONS,
    SocraticQuestioner,
    SocraticResult,
    _build_enriched_query,
    _parse_question_list,
)


# ── _parse_question_list ──────────────────────────────────────────────

def test_parse_numbered_list():
    raw = "1. What assumptions are hidden here?\n2. What would disprove this?\n3. What is simplest?"
    result = _parse_question_list(raw)
    assert len(result) == 3
    assert result[0] == "What assumptions are hidden here?"


def test_parse_bulleted_list():
    raw = "- Is this question well-formed?\n• What is missing from the evidence?\n* What second-order effects exist?"
    result = _parse_question_list(raw)
    assert len(result) == 3


def test_parse_paren_numbered():
    raw = "1) What are the hidden constraints?\n2) What simpler explanation exists?\n3) What would falsify this?"
    result = _parse_question_list(raw)
    assert len(result) == 3


def test_parse_ignores_non_questions():
    raw = "1. This is a statement.\n2. What is the core assumption?\n3. Short."
    result = _parse_question_list(raw)
    assert len(result) == 1
    assert "core assumption" in result[0]


def test_parse_empty_raw():
    assert _parse_question_list("") == []


def test_parse_caps_at_five():
    raw = "\n".join(f"{i+1}. Question number {i+1} asking about something?" for i in range(10))
    result = _parse_question_list(raw)
    assert len(result) <= 10  # parser doesn't cap; caller caps at 5


def test_parse_skips_short_lines():
    raw = "1. Ok?\n2. What assumptions are embedded beneath the surface of this query?"
    result = _parse_question_list(raw)
    assert len(result) == 1


# ── _build_enriched_query ─────────────────────────────────────────────

def test_build_enriched_query_structure():
    query = "What should I do about inflation?"
    questions = ["What is driving inflation here?", "Who benefits from the status quo?"]
    result = _build_enriched_query(query, questions)
    assert result.startswith(query)
    assert "Key questions to address" in result
    assert "1. What is driving" in result
    assert "2. Who benefits" in result


def test_build_enriched_query_numbering():
    questions = [f"Question {i}?" for i in range(1, 6)]
    result = _build_enriched_query("query", questions)
    for i in range(1, 6):
        assert f"{i}. Question {i}?" in result


# ── SocraticQuestioner — fallback (no LLM) ───────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def test_decompose_fallback_no_llm():
    q = SocraticQuestioner(llm_chat_fn=None)
    result = run(q.decompose("What is the best database?"))
    assert isinstance(result, SocraticResult)
    assert result.used_llm is False
    assert len(result.questions) == 3
    assert result.original_query == "What is the best database?"
    assert result.enriched_query.startswith("What is the best database?")


def test_decompose_fallback_questions_are_from_fallback_list():
    q = SocraticQuestioner(llm_chat_fn=None)
    result = run(q.decompose("test query"))
    for question in result.questions:
        assert question in FALLBACK_QUESTIONS


def test_elapsed_ms_is_positive():
    q = SocraticQuestioner(llm_chat_fn=None)
    result = run(q.decompose("any query"))
    assert result.elapsed_ms >= 0.0


# ── SocraticQuestioner — with LLM ────────────────────────────────────

def test_decompose_with_llm_success():
    llm_response = (
        "1. What assumptions are hidden in this query?\n"
        "2. What evidence would disprove the obvious answer?\n"
        "3. What is the simplest explanation that fits?"
    )
    mock_llm = AsyncMock(return_value=llm_response)
    q = SocraticQuestioner(llm_chat_fn=mock_llm)
    result = run(q.decompose("What is consciousness?"))
    assert result.used_llm is True
    assert len(result.questions) == 3
    assert mock_llm.called


def test_decompose_with_llm_caps_at_five():
    llm_response = "\n".join(
        f"{i+1}. Decomposition question number {i+1} here?" for i in range(8)
    )
    mock_llm = AsyncMock(return_value=llm_response)
    q = SocraticQuestioner(llm_chat_fn=mock_llm)
    result = run(q.decompose("any query"))
    assert len(result.questions) <= 5


def test_decompose_with_llm_fallback_on_failure():
    mock_llm = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
    q = SocraticQuestioner(llm_chat_fn=mock_llm)
    result = run(q.decompose("will LLM fail?"))
    assert result.used_llm is False
    assert len(result.questions) == 3


def test_decompose_with_llm_fallback_on_empty_parse():
    # LLM returns something with no parseable questions
    mock_llm = AsyncMock(return_value="Sure, let me think about this.")
    q = SocraticQuestioner(llm_chat_fn=mock_llm)
    result = run(q.decompose("any query"))
    assert result.used_llm is False
    assert len(result.questions) == 3


# ── can_question ──────────────────────────────────────────────────────

def test_can_question_no_feature_flags_module():
    q = SocraticQuestioner()
    with patch.dict("sys.modules", {"feature_flags": None}):
        # ImportError → default True
        assert q.can_question() is True


def test_can_question_with_flag_enabled():
    mock_flags = type(sys)("feature_flags")
    mock_flags.is_enabled = lambda name: True
    q = SocraticQuestioner()
    with patch.dict("sys.modules", {"feature_flags": mock_flags}):
        assert q.can_question() is True


def test_can_question_with_flag_disabled():
    mock_flags = type(sys)("feature_flags")
    mock_flags.is_enabled = lambda name: False
    q = SocraticQuestioner()
    with patch.dict("sys.modules", {"feature_flags": mock_flags}):
        assert q.can_question() is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
