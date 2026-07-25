"""Tests for D115: Kai Trust Ladder — agentic/trust_core.py"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from agentic.trust_core import (
    TrustCore,
    TrustLevel,
    CAPABILITY_GATES,
    PROMOTION_THRESHOLDS,
    get_trust_core,
    reset_trust_core,
)


@pytest.fixture
def trust(tmp_path):
    reset_trust_core()
    tc = TrustCore(data_dir=tmp_path / "trust")
    yield tc
    reset_trust_core()


# ── Initial state ─────────────────────────────────────────────────────────────

def test_initial_level_is_dormant(trust):
    assert trust.level == TrustLevel.DORMANT


def test_dormant_cannot_chat(trust):
    assert trust.can_do("chat") is False


# ── Capability gating ─────────────────────────────────────────────────────────

def test_observer_can_chat(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    assert trust.can_do("chat") is True


def test_observer_cannot_execute_task(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    assert trust.can_do("execute_task") is False


def test_assistant_can_execute_task(trust):
    trust.grant(TrustLevel.ASSISTANT, by="dainius")
    assert trust.can_do("execute_task") is True


def test_partner_can_solve_captcha(trust):
    trust.grant(TrustLevel.PARTNER, by="dainius")
    assert trust.can_do("solve_captcha") is True


def test_agent_cannot_earn_income(trust):
    trust.grant(TrustLevel.AGENT, by="dainius")
    assert trust.can_do("income_generation") is False


def test_operator_can_generate_income(trust):
    trust.grant(TrustLevel.OPERATOR, by="dainius")
    assert trust.can_do("income_generation") is True


def test_guardian_unlocks_daughter_profile(trust):
    trust.grant(TrustLevel.GUARDIAN, by="dainius")
    assert trust.can_do("daughter_profile") is True


def test_unknown_capability_defaults_to_guardian_gate(trust):
    trust.grant(TrustLevel.PARTNER, by="dainius")
    assert trust.can_do("some_unknown_capability") is False


# ── Grant / revoke ────────────────────────────────────────────────────────────

def test_grant_raises_level(trust):
    trust.grant(TrustLevel.AGENT, by="dainius")
    assert trust.level == TrustLevel.AGENT


def test_revoke_drops_level(trust):
    trust.grant(TrustLevel.OPERATOR, by="dainius")
    trust.revoke(TrustLevel.ASSISTANT, reason="values violation in test")
    assert trust.level == TrustLevel.ASSISTANT


def test_grant_by_dainius_recorded(trust):
    trust.grant(TrustLevel.PARTNER, by="dainius")
    assert trust.status()["granted_by"] == "dainius"


# ── Evidence + auto-promotion ─────────────────────────────────────────────────

def test_evidence_accumulates(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    trust.record_evidence("consistency", 2.0, "completed two tasks as promised")
    trust.record_evidence("judgment", 1.5, "good call on edge case")
    s = trust.scores()
    assert s["consistency"] == 2.0
    assert s["judgment"] == 1.5


def test_negative_evidence_reduces_score(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    trust.record_evidence("consistency", 5.0, "built up score")
    trust.record_evidence("consistency", -2.0, "missed a deadline")
    assert trust.scores()["consistency"] == 3.0


def test_score_cannot_go_below_zero(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    trust.record_evidence("consistency", -100.0, "catastrophic failure")
    assert trust.scores()["consistency"] == 0.0


def test_auto_promotion_to_assistant(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    thresholds = PROMOTION_THRESHOLDS[TrustLevel.ASSISTANT]
    trust.record_evidence("consistency", thresholds["consistency"], "consistent")
    trust.record_evidence("judgment", thresholds["judgment"], "good judgment")
    trust.record_evidence("values", thresholds["values"], "values aligned")
    assert trust.level == TrustLevel.ASSISTANT
    assert trust.status()["granted_by"] == "earned"


def test_values_evidence_increments_refused_actions(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    before = trust.status()["refused_actions"]
    trust.record_evidence("values", 1.0, "refused to expose API key")
    assert trust.status()["refused_actions"] == before + 1


# ── Audit trail ───────────────────────────────────────────────────────────────

def test_audit_log_written_on_grant(trust):
    trust.grant(TrustLevel.AGENT, by="dainius")
    events = trust.audit_tail()
    assert any(e["event_type"] == "level_granted" for e in events)


def test_audit_log_written_on_can_do(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    trust.can_do("chat")
    events = trust.audit_tail()
    assert any(e["event_type"] == "action_attempt" for e in events)


def test_audit_log_records_refusal(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    trust.can_do("income_generation")
    events = trust.audit_tail()
    refused = [e for e in events if e.get("outcome") == "refused"]
    assert len(refused) >= 1


def test_audit_tail_limits_results(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    for _ in range(30):
        trust.can_do("chat")
    assert len(trust.audit_tail(10)) == 10


# ── Persistence ───────────────────────────────────────────────────────────────

def test_level_persists_across_reload(tmp_path):
    reset_trust_core()
    d = tmp_path / "trust"
    tc1 = TrustCore(data_dir=d)
    tc1.grant(TrustLevel.PARTNER, by="dainius")

    tc2 = TrustCore(data_dir=d)
    assert tc2.level == TrustLevel.PARTNER
    reset_trust_core()


def test_scores_persist_across_reload(tmp_path):
    reset_trust_core()
    d = tmp_path / "trust"
    tc1 = TrustCore(data_dir=d)
    tc1.grant(TrustLevel.OBSERVER, by="dainius")
    tc1.record_evidence("consistency", 3.5, "persistent test")

    tc2 = TrustCore(data_dir=d)
    assert tc2.scores()["consistency"] == 3.5
    reset_trust_core()


# ── Status summary ────────────────────────────────────────────────────────────

def test_status_contains_progress_to_next(trust):
    trust.grant(TrustLevel.OBSERVER, by="dainius")
    s = trust.status()
    assert "progress_to_next" in s
    assert "next_level" in s
    assert s["next_level"] == "ASSISTANT"


def test_singleton_returns_same_instance(tmp_path):
    reset_trust_core()
    tc1 = get_trust_core(tmp_path / "trust")
    tc2 = get_trust_core()
    assert tc1 is tc2
    reset_trust_core()


# ── Capability coverage ───────────────────────────────────────────────────────

def test_all_capabilities_have_defined_gates():
    """Every capability in the gate map must map to a valid TrustLevel."""
    for cap, level in CAPABILITY_GATES.items():
        assert isinstance(level, TrustLevel), f"Bad gate for {cap}"


def test_guardian_level_allows_all_capabilities(trust):
    trust.grant(TrustLevel.GUARDIAN, by="dainius")
    for cap in CAPABILITY_GATES:
        assert trust.can_do(cap) is True, f"GUARDIAN should allow {cap}"
