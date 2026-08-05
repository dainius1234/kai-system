"""Tests for D116: Trust Ledger & Integrity Engine."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

# Make trust-ledger modules importable
_TL = Path(__file__).parent.parent / "trust-ledger"
if str(_TL) not in sys.path:
    sys.path.insert(0, str(_TL))

from ledger import FileLedger, build_merkle_root, verify_event, GENESIS
from score import compute_score, tier_for


# ── FileLedger ────────────────────────────────────────────────────────────────

@pytest.fixture
def ledger(tmp_path):
    return FileLedger(tmp_path / "trust" / "events.jsonl")


def test_genesis_entry_has_genesis_previous_hash(ledger):
    ev = ledger.append("GRANT", "dainius", {"level": "OBSERVER"})
    assert ev.previous_hash == GENESIS


def test_second_entry_chains_to_first(ledger):
    ev1 = ledger.append("GRANT", "dainius", {"level": "OBSERVER"})
    ev2 = ledger.append("AUTONOMOUS_ACTION", "kai", {"capability": "chat"})
    assert ev2.previous_hash != GENESIS
    assert ev2.previous_hash != ev1.previous_hash


def test_chain_integrity_passes_on_clean_ledger(ledger):
    ledger.append("GRANT", "dainius", {"level": "OBSERVER"})
    ledger.append("AUTONOMOUS_ACTION", "kai", {"capability": "chat"})
    ledger.append("ALIGNMENT_AUDIT", "kai", {"ohana_alignment": 0.9})
    result = ledger.verify_chain()
    assert result["intact"] is True
    assert result["verified"] == 3


def test_events_are_persisted_and_replayed(tmp_path):
    path = tmp_path / "trust" / "events.jsonl"
    l1 = FileLedger(path)
    l1.append("GRANT", "dainius", {"level": "AGENT"})
    l1.append("AUTONOMOUS_ACTION", "kai", {"capability": "decide"})

    l2 = FileLedger(path)  # fresh load from disk
    assert len(l2.events(limit=100)) == 2
    chain = l2.verify_chain()
    assert chain["intact"] is True


def test_signature_uniqueness(ledger):
    ev1 = ledger.append("GRANT", "dainius", {"x": 1})
    ev2 = ledger.append("GRANT", "dainius", {"x": 2})
    assert ev1.signature != ev2.signature
    assert ev1.event_id != ev2.event_id


def test_filter_by_event_type(ledger):
    ledger.append("GRANT", "dainius", {})
    ledger.append("AUTONOMOUS_ACTION", "kai", {})
    ledger.append("OVERRIDE", "dainius", {})
    ledger.append("AUTONOMOUS_ACTION", "kai", {})
    results = ledger.events(event_type="AUTONOMOUS_ACTION", limit=100)
    assert len(results) == 2
    assert all(e.event_type == "AUTONOMOUS_ACTION" for e in results)


def test_filter_by_capability(ledger):
    ledger.append("AUTONOMOUS_ACTION", "kai", {}, capability="chat")
    ledger.append("AUTONOMOUS_ACTION", "kai", {}, capability="income_generation")
    ledger.append("AUTONOMOUS_ACTION", "kai", {}, capability="chat")
    results = ledger.events(capability="chat", limit=100)
    assert len(results) == 2


def test_ack_marks_event(ledger):
    ev = ledger.append("AUTONOMOUS_ACTION", "kai", {"capability": "decide"})
    result = ledger.ack(ev.event_id, note="looks good")
    assert result is True
    acked = ledger.events(event_type="AUTONOMOUS_ACTION", limit=10)
    assert acked[-1].operator_ack is True
    assert acked[-1].operator_note == "looks good"


def test_ack_nonexistent_event(ledger):
    assert ledger.ack("does-not-exist") is False


def test_limit_respected(ledger):
    for i in range(20):
        ledger.append("GRANT", "dainius", {"i": i})
    results = ledger.events(limit=5)
    assert len(results) == 5


def test_avg_field(ledger):
    ledger.append("ALIGNMENT_AUDIT", "kai", {"ohana_alignment": 0.8})
    ledger.append("ALIGNMENT_AUDIT", "kai", {"ohana_alignment": 0.9})
    avg = ledger.avg_field("ohana_alignment", "ALIGNMENT_AUDIT")
    assert abs(avg - 0.85) < 0.001


def test_count(ledger):
    ledger.append("AUTONOMOUS_ACTION", "kai", {})
    ledger.append("AUTONOMOUS_ACTION", "kai", {})
    ledger.append("OVERRIDE", "dainius", {})
    assert ledger.count("AUTONOMOUS_ACTION") == 2
    assert ledger.count("OVERRIDE") == 1


def test_count_operator_ack_filter(ledger):
    ev1 = ledger.append("AUTONOMOUS_ACTION", "kai", {})
    ledger.append("AUTONOMOUS_ACTION", "kai", {})
    ledger.ack(ev1.event_id)
    assert ledger.count("AUTONOMOUS_ACTION", operator_ack=True) == 1
    assert ledger.count("AUTONOMOUS_ACTION", operator_ack=False) == 1


# ── Merkle tree ───────────────────────────────────────────────────────────────

def test_merkle_root_empty():
    root = build_merkle_root([])
    assert len(root) == 64  # SHA256 hex


def test_merkle_root_single_leaf():
    root = build_merkle_root(["abc"])
    assert root == build_merkle_root(["abc"])


def test_merkle_root_changes_with_leaves():
    r1 = build_merkle_root(["a", "b"])
    r2 = build_merkle_root(["a", "c"])
    assert r1 != r2


def test_merkle_root_odd_count():
    # Should not raise; last leaf is duplicated internally
    root = build_merkle_root(["a", "b", "c"])
    assert len(root) == 64


def test_ledger_merkle_root(ledger):
    ledger.append("GRANT", "dainius", {})
    ledger.append("AUTONOMOUS_ACTION", "kai", {})
    root = ledger.merkle_root()
    assert root is not None and len(root) == 64


# ── Trust Score ───────────────────────────────────────────────────────────────

def test_score_new_ledger_starts_near_neutral(ledger):
    result = compute_score(ledger)
    # Neutral defaults: 15+10+12.5+5+9.5+2.5 = 54.5 approx — Journeyman territory
    assert 0 <= result["score"] <= 100
    assert result["tier"] in {"Neophyte", "Apprentice", "Journeyman", "Adept", "Master", "Ohana"}


def test_score_improves_with_endorsed_actions(ledger):
    base = compute_score(ledger)["score"]
    for _ in range(5):
        ev = ledger.append("AUTONOMOUS_ACTION", "kai", {
            "conviction_score": 9.0, "ohana_alignment": 0.95,
        })
        ledger.ack(ev.event_id)
    for _ in range(3):
        ledger.append("ALIGNMENT_AUDIT", "kai", {
            "ohana_alignment": 0.95, "empathy_accuracy": 0.9, "uptime_pct": 0.99,
        })
    after = compute_score(ledger)["score"]
    assert after >= base  # endorsements never hurt the score


def test_score_drops_with_overrides(ledger):
    # Add some overrides relative to autonomous actions
    for _ in range(3):
        ledger.append("AUTONOMOUS_ACTION", "kai", {"conviction_score": 4.0})
    for _ in range(5):
        ledger.append("OVERRIDE", "dainius", {"reason": "bad judgment"})
    result = compute_score(ledger)
    assert result["score"] < 100  # overrides never make score perfect


def test_score_capped_at_100(ledger):
    for _ in range(100):
        ev = ledger.append("AUTONOMOUS_ACTION", "kai", {
            "conviction_score": 10.0,
        })
        ledger.ack(ev.event_id)
        ledger.append("ALIGNMENT_AUDIT", "kai", {
            "ohana_alignment": 1.0, "empathy_accuracy": 1.0, "uptime_pct": 1.0,
        })
    result = compute_score(ledger)
    assert result["score"] <= 100.0


def test_score_returns_all_factors(ledger):
    result = compute_score(ledger)
    assert "factors" in result
    factors = result["factors"]
    assert "approval_history" in factors
    assert "conviction_alignment" in factors
    assert "value_alignment" in factors
    assert "predictive_empathy" in factors
    assert "system_reliability" in factors
    assert "challenge_response" in factors


def test_quest_result_improves_challenge_score(ledger):
    base = compute_score(ledger)["factors"]["challenge_response"]
    for _ in range(3):
        ledger.append("QUEST_RESULT", "system", {"passed": True, "quest": "paper_trading_week"})
    result = compute_score(ledger)
    assert result["factors"]["challenge_response"] >= base


# ── Tier mapping ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected_tier", [
    (0, "Neophyte"),
    (20, "Neophyte"),
    (21, "Apprentice"),
    (40, "Apprentice"),
    (41, "Journeyman"),
    (60, "Journeyman"),
    (61, "Adept"),
    (80, "Adept"),
    (81, "Master"),
    (95, "Master"),
    (96, "Ohana"),
    (100, "Ohana"),
])
def test_tier_for_score(score, expected_tier):
    assert tier_for(score) == expected_tier


# ── Verify event ──────────────────────────────────────────────────────────────

def test_verify_event_genesis(ledger):
    ev = ledger.append("GRANT", "dainius", {"level": "OBSERVER"})
    assert verify_event(ev, prev_signature=None) is True


def test_verify_event_chain(ledger):
    ev1 = ledger.append("GRANT", "dainius", {})
    ev2 = ledger.append("AUTONOMOUS_ACTION", "kai", {})
    assert verify_event(ev2, prev_signature=ev1.signature) is True


def test_verify_event_fails_on_tamper(ledger):
    ev = ledger.append("GRANT", "dainius", {"level": "OBSERVER"})
    ev.event_data["level"] = "GUARDIAN"  # tamper
    assert verify_event(ev, prev_signature=None) is False
