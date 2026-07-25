"""Test suite for D114 Cortex cognitive module (agentic/cortex.py).

Tests the Phase 0 stub contract:
  - Singleton factory works
  - can_operate() returns False when no service state has been fed
  - feed_service_state() populates SituationModel correctly
  - can_operate() returns True after fresh service state
  - bid_to_workspace() returns None when not operable, real bid when operable
  - tacit preferences applied correctly
  - tick() is a no-op in Phase 0 (FF_CORTEX_NPU=false)

Run: python scripts/test_cortex.py
"""
from __future__ import annotations

import sys
import os
import time

# Allow direct run from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agentic"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))

# Stub feature flags for the test environment
import types
_ff_stub = types.ModuleType("common.feature_flags")
_ff_stub.is_enabled = lambda flag: False
sys.modules["common.feature_flags"] = _ff_stub
sys.modules["common"] = types.ModuleType("common")

# Stub global_workspace so WorkspaceBid import works
import dataclasses

@dataclasses.dataclass
class _WorkspaceBid:
    module: str
    content: str
    urgency: float = 0.4
    relevance: float = 0.8
    surprise: float = 0.0
    confidence: float = 0.5
    emotional_salience: float = 0.3
    timestamp: float = dataclasses.field(default_factory=time.time)

_gw_stub = types.ModuleType("global_workspace")
_gw_stub.WorkspaceBid = _WorkspaceBid
_gw_stub.get_global_workspace = lambda: None
sys.modules["global_workspace"] = _gw_stub

import cortex as cortex_mod
from cortex import get_cortex, Cortex, SituationModel


def _reset_singleton() -> None:
    cortex_mod._cortex = None


def test_singleton() -> None:
    _reset_singleton()
    a = get_cortex()
    b = get_cortex()
    assert a is b, "get_cortex() must return the same instance"
    print("  PASS  singleton")


def test_cannot_operate_before_state() -> None:
    _reset_singleton()
    c = get_cortex()
    assert not c.can_operate(), "can_operate() must be False before any state is fed"
    print("  PASS  can_operate() → False before state")


def test_feed_service_state_populates() -> None:
    _reset_singleton()
    c = get_cortex()
    state = {
        "timestamp": "2026-07-25T12:00:00Z",
        "level1_facts": ["System: CPU 87%, RAM 62%", "Calendar: Meeting in 8 minutes"],
        "level2_summary": "System strained, hard stop approaching",
        "level3_implication": "Consider committing work before the meeting",
        "intent_fan": [
            {"label": "debugging", "confidence": 0.65, "context_hints": ["error logs"]},
        ],
        "bridge_active": False,
        "bridge_note": None,
        "tacit_rules": ["Prefers brief queries — default to bullet-point responses"],
        "sensor_credibility": {},
        "refresh_count": 3,
    }
    c.feed_service_state(state)

    assert c.situation.level_2_summary == "System strained, hard stop approaching"
    assert c.situation.level_3_implications == "Consider committing work before the meeting"
    assert c.situation.confidence == 1.0
    assert len(c.intent_shadow.active_intents) == 1
    assert c.intent_shadow.active_intents[0]["intent"] == "debugging"
    assert len(c.tacit_preferences) == 1
    print("  PASS  feed_service_state() populates SituationModel, IntentShadow, TacitPreferences")


def test_can_operate_after_fresh_state() -> None:
    _reset_singleton()
    c = get_cortex()
    state = {
        "timestamp": "2026-07-25T12:00:00Z",
        "level1_facts": [],
        "level2_summary": "Calm — no significant pressure signals",
        "level3_implication": "",
        "intent_fan": [],
        "bridge_active": False,
        "bridge_note": None,
        "tacit_rules": [],
        "sensor_credibility": {},
        "refresh_count": 1,
    }
    c.feed_service_state(state)
    assert c.can_operate(), "can_operate() must be True within 120s of fresh state"
    print("  PASS  can_operate() → True after fresh state")


def test_bid_none_when_not_operable() -> None:
    _reset_singleton()
    c = get_cortex()
    assert c.bid_to_workspace() is None, "bid_to_workspace() must return None when not operable"
    print("  PASS  bid_to_workspace() → None when not operable")


def test_bid_real_when_operable() -> None:
    _reset_singleton()
    c = get_cortex()
    state = {
        "timestamp": "2026-07-25T12:00:00Z",
        "level1_facts": ["System: CPU 87%"],
        "level2_summary": "System under load with services struggling",
        "level3_implication": "Restart unhealthy services",
        "intent_fan": [],
        "bridge_active": False,
        "bridge_note": None,
        "tacit_rules": [],
        "sensor_credibility": {},
        "refresh_count": 2,
    }
    c.feed_service_state(state)
    bid = c.bid_to_workspace()
    assert bid is not None, "bid_to_workspace() must return a bid when operable"
    assert bid.module == "cortex"
    assert "System under load" in bid.content
    assert bid.urgency == 0.6, f"expected 0.6 for strained, got {bid.urgency}"
    assert bid.relevance == 0.85
    print("  PASS  bid_to_workspace() → real WorkspaceBid when operable")


def test_urgency_scaling() -> None:
    _reset_singleton()
    c = get_cortex()
    for summary, expected_urgency in [
        ("Critical system issue with a hard deadline closing in", 0.9),
        ("Operator sprinting toward a hard deadline", 0.75),
        ("System strained, services struggling", 0.6),
        ("Calm — no significant pressure signals", 0.4),
    ]:
        c._service_last_ok = time.time()
        c.situation = SituationModel(
            level_2_summary=summary,
            level_3_implications="",
            confidence=1.0,
        )
        bid = c.bid_to_workspace()
        assert bid is not None
        assert bid.urgency == expected_urgency, (
            f"summary='{summary}' expected urgency={expected_urgency}, got {bid.urgency}"
        )
    print("  PASS  urgency scales correctly with situation severity")


def test_tick_noop_phase0() -> None:
    _reset_singleton()
    c = get_cortex()
    c.situation.level_2_summary = "before tick"
    c.tick({"cpu": "87%"})  # should be no-op in Phase 0
    assert c.situation.level_2_summary == "before tick", "tick() must be a no-op in Phase 0"
    print("  PASS  tick() is no-op in Phase 0 (FF_CORTEX_NPU=false)")


def test_tacit_preference_applied() -> None:
    _reset_singleton()
    c = get_cortex()
    from cortex import TacitPreference
    c.tacit_preferences = [TacitPreference(
        condition="observed", preferred_style="Prefers brief queries — default to bullet-point responses",
        observed_count=20, confidence=0.8,
    )]
    result = c.apply_tacit_preferences("prose", {})
    assert result == "bullet_points", f"expected 'bullet_points', got '{result}'"
    print("  PASS  apply_tacit_preferences() returns bullet_points for brief preference")


def test_get_current_situation() -> None:
    _reset_singleton()
    c = get_cortex()
    s = c.get_current_situation()
    assert isinstance(s, SituationModel)
    print("  PASS  get_current_situation() returns SituationModel")


if __name__ == "__main__":
    tests = [
        test_singleton,
        test_cannot_operate_before_state,
        test_feed_service_state_populates,
        test_can_operate_after_fresh_state,
        test_bid_none_when_not_operable,
        test_bid_real_when_operable,
        test_urgency_scaling,
        test_tick_noop_phase0,
        test_tacit_preference_applied,
        test_get_current_situation,
    ]

    print(f"\nD114 Cortex module — {len(tests)} tests\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:
            print(f"  ERROR {t.__name__}: {e}")

    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
