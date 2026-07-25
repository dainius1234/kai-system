"""Tests for D109: The Ohana Core — Situational Ethics & Loyalty Alignment.

Covers:
  - MoralFingerprint: defaults and field structure
  - MoralContext: defaults, to_prompt(), specific_stances / past_decisions
  - OhanaCore: build_moral_context, inject_into_prompt (stub no-op), record_decision,
               request_clarification, evaluate_action_alignment, progress, can_operate
  - Singleton: get_ohana_core returns same instance
"""
import sys
from pathlib import Path
import pytest


# ---------------------------------------------------------------------------
# Import helper
# ---------------------------------------------------------------------------

def _import_mc():
    sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))
    import moral_core
    return moral_core


# ---------------------------------------------------------------------------
# Importability
# ---------------------------------------------------------------------------

def test_moral_core_importable():
    mc = _import_mc()
    assert hasattr(mc, "MoralFingerprint")
    assert hasattr(mc, "MoralContext")
    assert hasattr(mc, "OhanaCore")
    assert hasattr(mc, "get_ohana_core")


# ---------------------------------------------------------------------------
# MoralFingerprint
# ---------------------------------------------------------------------------

def test_moral_fingerprint_defaults():
    mc = _import_mc()
    fp = mc.MoralFingerprint()
    assert "family safety" in fp.core_loyalties
    assert "survival" in fp.core_loyalties
    assert "autonomy" in fp.core_loyalties


def test_moral_fingerprint_harm_boundaries():
    mc = _import_mc()
    fp = mc.MoralFingerprint()
    assert len(fp.harm_boundaries) >= 1
    assert any("innocents" in b for b in fp.harm_boundaries)


def test_moral_fingerprint_loyalty_override_max():
    mc = _import_mc()
    fp = mc.MoralFingerprint()
    assert fp.loyalty_override == 1.0


def test_moral_fingerprint_rule_flexibility_range():
    mc = _import_mc()
    fp = mc.MoralFingerprint()
    assert 0.0 <= fp.rule_flexibility <= 1.0


def test_moral_fingerprint_situational_stances_empty_default():
    mc = _import_mc()
    fp = mc.MoralFingerprint()
    assert isinstance(fp.situational_stances, dict)


def test_moral_fingerprint_custom_stances():
    mc = _import_mc()
    fp = mc.MoralFingerprint(
        situational_stances={"reverse_engineering": "acceptable for learning"}
    )
    assert fp.situational_stances["reverse_engineering"] == "acceptable for learning"


# ---------------------------------------------------------------------------
# MoralContext
# ---------------------------------------------------------------------------

def test_moral_context_defaults():
    mc = _import_mc()
    ctx = mc.MoralContext()
    assert ctx.core_reminder != ""
    assert ctx.specific_stances == ""
    assert ctx.relevant_past_decisions == ""


def test_moral_context_to_prompt_contains_reminder():
    mc = _import_mc()
    ctx = mc.MoralContext()
    result = ctx.to_prompt()
    assert "[Ohana Context]" in result
    assert ctx.core_reminder in result


def test_moral_context_to_prompt_with_stances():
    mc = _import_mc()
    ctx = mc.MoralContext(specific_stances="survival takes priority")
    result = ctx.to_prompt()
    assert "survival takes priority" in result


def test_moral_context_to_prompt_with_past_decisions():
    mc = _import_mc()
    ctx = mc.MoralContext(relevant_past_decisions="chose family over rule in 2025-03")
    result = ctx.to_prompt()
    assert "2025-03" in result


def test_moral_context_to_prompt_minimal_when_empty():
    mc = _import_mc()
    ctx = mc.MoralContext()
    result = ctx.to_prompt()
    assert "Relevant stances" not in result
    assert "Past decisions" not in result


# ---------------------------------------------------------------------------
# OhanaCore — Phase 0 stub behaviour
# ---------------------------------------------------------------------------

def test_ohana_core_can_operate_false():
    mc = _import_mc()
    core = mc.OhanaCore()
    assert core.can_operate() is False


def test_ohana_core_build_moral_context_returns_default():
    mc = _import_mc()
    core = mc.OhanaCore()
    ctx = core.build_moral_context()
    assert isinstance(ctx, mc.MoralContext)
    assert ctx.core_reminder != ""


def test_ohana_core_build_moral_context_with_situation():
    mc = _import_mc()
    core = mc.OhanaCore()
    ctx = core.build_moral_context(situation={"query": "should I work late?"})
    assert isinstance(ctx, mc.MoralContext)


def test_ohana_core_inject_into_prompt_no_op():
    mc = _import_mc()
    core = mc.OhanaCore()
    original = "Answer this question: what time is it?"
    result = core.inject_into_prompt(original)
    assert result == original


def test_ohana_core_inject_with_situation_no_op():
    mc = _import_mc()
    core = mc.OhanaCore()
    original = "plan the day"
    result = core.inject_into_prompt(original, situation={"state": "morning"})
    assert result == original


def test_ohana_core_record_decision_no_exception():
    mc = _import_mc()
    core = mc.OhanaCore()
    core.record_decision(
        situation={"context": "traffic stop"},
        decision="cooperated fully",
        outcome="warning issued",
    )


def test_ohana_core_record_decision_increments_count():
    mc = _import_mc()
    core = mc.OhanaCore()
    core.record_decision({}, "chose survival")
    core.record_decision({}, "chose family")
    assert core._interaction_count == 2


def test_ohana_core_request_clarification_returns_none():
    mc = _import_mc()
    core = mc.OhanaCore()
    result = core.request_clarification({"contradiction": "prior vs new stance"})
    assert result is None


def test_ohana_core_evaluate_action_alignment_neutral():
    mc = _import_mc()
    core = mc.OhanaCore()
    score = core.evaluate_action_alignment({"action": "send email"})
    assert score == 0.5


def test_ohana_core_evaluate_action_range():
    mc = _import_mc()
    core = mc.OhanaCore()
    score = core.evaluate_action_alignment({"action": "complex decision"})
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# OhanaCore — progress
# ---------------------------------------------------------------------------

def test_ohana_core_progress_dict():
    mc = _import_mc()
    core = mc.OhanaCore()
    prog = core.progress()
    assert "can_operate" in prog
    assert "interaction_count" in prog
    assert "stances_learned" in prog
    assert "core_loyalties" in prog
    assert "loyalty_override" in prog
    assert prog["can_operate"] is False
    assert prog["loyalty_override"] == 1.0


def test_ohana_core_get_fingerprint_snapshot():
    mc = _import_mc()
    core = mc.OhanaCore()
    fp = core.get_fingerprint_snapshot()
    assert isinstance(fp, mc.MoralFingerprint)
    assert fp.loyalty_override == 1.0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

def test_get_ohana_core_returns_ohana_core():
    mc = _import_mc()
    mc._ohana_core = None
    core = mc.get_ohana_core()
    assert isinstance(core, mc.OhanaCore)


def test_get_ohana_core_is_singleton():
    mc = _import_mc()
    mc._ohana_core = None
    c1 = mc.get_ohana_core()
    c2 = mc.get_ohana_core()
    assert c1 is c2


def test_get_ohana_core_retains_interaction_count():
    mc = _import_mc()
    mc._ohana_core = None
    core = mc.get_ohana_core()
    core.record_decision({}, "test decision")
    core2 = mc.get_ohana_core()
    assert core2._interaction_count == 1
