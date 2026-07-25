"""Tests for D94: Temporal Projection (forecaster.py)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from forecaster import (
    ForecastFan,
    ScenarioBranch,
    TemporalForecaster,
    _FALLBACK_BRANCHES,
    _parse_branches,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── ScenarioBranch ────────────────────────────────────────────────────

def test_scenario_branch_defaults():
    b = ScenarioBranch(label="base", narrative="things continue", probability=0.5)
    assert b.confidence_modifier == 0.0
    assert b.key_assumptions == []


# ── ForecastFan ───────────────────────────────────────────────────────

def test_forecast_fan_consensus_probability():
    fan = ForecastFan(
        query="test",
        base_claim="claim",
        branches=[
            ScenarioBranch(label="base", narrative="n", probability=0.55),
            ScenarioBranch(label="optimistic", narrative="n", probability=0.25),
        ],
    )
    assert fan.consensus_probability == 0.55


def test_forecast_fan_consensus_probability_no_base():
    fan = ForecastFan(query="test", base_claim="claim", branches=[])
    assert fan.consensus_probability == 0.5


def test_forecast_fan_to_dict():
    fan = ForecastFan(
        query="Q",
        base_claim="C",
        branches=[ScenarioBranch(label="base", narrative="n", probability=0.5)],
        elapsed_ms=12.3,
        used_llm=True,
    )
    d = fan.to_dict()
    assert d["query"] == "Q"
    assert d["used_llm"] is True
    assert len(d["branches"]) == 1
    assert d["branches"][0]["label"] == "base"


# ── _parse_branches ───────────────────────────────────────────────────

def test_parse_branches_valid_json():
    data = [
        {"label": "base", "narrative": "base case", "probability": 0.5, "key_assumptions": ["trend continues"]},
        {"label": "optimistic", "narrative": "best case", "probability": 0.25, "key_assumptions": []},
        {"label": "pessimistic", "narrative": "worst case", "probability": 0.2, "key_assumptions": []},
        {"label": "wild_card", "narrative": "surprise", "probability": 0.05, "key_assumptions": []},
    ]
    raw = json.dumps(data)
    branches = _parse_branches(raw)
    assert len(branches) == 4
    assert branches[0].label == "base"
    assert branches[0].probability == 0.5


def test_parse_branches_invalid_label_skipped():
    data = [
        {"label": "unknown_label", "narrative": "n", "probability": 0.5},
        {"label": "base", "narrative": "base", "probability": 0.5},
    ]
    branches = _parse_branches(json.dumps(data))
    assert len(branches) == 1
    assert branches[0].label == "base"


def test_parse_branches_malformed_json():
    branches = _parse_branches("not json at all")
    assert branches == []


def test_parse_branches_with_preamble():
    data = [{"label": "base", "narrative": "n", "probability": 0.5, "key_assumptions": []}]
    raw = f"Here are the scenarios: {json.dumps(data)} Hope that helps."
    branches = _parse_branches(raw)
    assert len(branches) == 1


# ── TemporalForecaster — fallback ────────────────────────────────────

def test_project_no_llm_returns_fallback():
    f = TemporalForecaster(llm_chat_fn=None)
    fan = run(f.project("will rates rise?", ["rates are high"]))
    assert fan.used_llm is False
    assert len(fan.branches) == 4
    labels = {b.label for b in fan.branches}
    assert labels == {"base", "optimistic", "pessimistic", "wild_card"}


def test_project_no_claims_uses_query_as_base():
    f = TemporalForecaster(llm_chat_fn=None)
    fan = run(f.project("what happens next?", []))
    assert fan.base_claim == "what happens next?"


def test_project_elapsed_ms_positive():
    f = TemporalForecaster(llm_chat_fn=None)
    fan = run(f.project("q", ["c"]))
    assert fan.elapsed_ms >= 0.0


# ── TemporalForecaster — with LLM ────────────────────────────────────

def test_project_llm_success():
    data = [
        {"label": "base", "narrative": "rates stay stable", "probability": 0.5, "key_assumptions": ["Fed holds"]},
        {"label": "optimistic", "narrative": "rates drop", "probability": 0.25, "key_assumptions": []},
        {"label": "pessimistic", "narrative": "rates spike", "probability": 0.2, "key_assumptions": []},
        {"label": "wild_card", "narrative": "currency crisis", "probability": 0.05, "key_assumptions": []},
    ]
    mock_llm = AsyncMock(return_value=json.dumps(data))
    f = TemporalForecaster(llm_chat_fn=mock_llm)
    fan = run(f.project("interest rate outlook", ["rates are elevated"]))
    assert fan.used_llm is True
    assert len(fan.branches) == 4


def test_project_llm_fallback_on_bad_parse():
    mock_llm = AsyncMock(return_value="not valid json")
    f = TemporalForecaster(llm_chat_fn=mock_llm)
    fan = run(f.project("q", ["c"]))
    assert fan.used_llm is False
    assert len(fan.branches) == 4


def test_project_llm_fallback_on_exception():
    mock_llm = AsyncMock(side_effect=TimeoutError("LLM timeout"))
    f = TemporalForecaster(llm_chat_fn=mock_llm)
    fan = run(f.project("q", ["c"]))
    assert fan.used_llm is False


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
