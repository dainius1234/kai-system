"""Tests for D118: Trust Integration — agentic/trust_integration.py."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from trust_integration import (
    gate_autonomous_action,
    get_trust_status,
    record_alignment_audit,
    record_chat_response,
    _record_nonblocking,
)


# ── gate_autonomous_action ────────────────────────────────────────────────────

def test_gate_allows_when_trust_core_unavailable():
    """Fail-open: no trust core → allow."""
    with patch("trust_integration._get_trust_core", return_value=None), \
            patch("trust_integration._get_ohana", return_value=None), \
            patch("trust_integration._record_nonblocking"):
        allowed, reason = gate_autonomous_action("chat", {})
    assert allowed is True
    assert reason == "allowed"


def test_gate_refuses_when_trust_level_insufficient():
    trust = MagicMock()
    trust.can_do.return_value = False
    trust.level_name = "DORMANT"
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._get_ohana", return_value=None), \
            patch("trust_integration._record_nonblocking"):
        allowed, reason = gate_autonomous_action("income_generation", {})
    assert allowed is False
    assert "DORMANT" in reason
    assert "income_generation" in reason


def test_gate_refuses_when_ohana_blocks():
    trust = MagicMock()
    trust.can_do.return_value = True
    ohana = MagicMock()
    ohana.evaluate_action_alignment.return_value = 0.0
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._get_ohana", return_value=ohana), \
            patch("trust_integration._record_nonblocking"):
        allowed, reason = gate_autonomous_action("expose_api_key", {"action": "expose_api_key"})
    assert allowed is False
    assert "Ohana Core blocked" in reason


def test_gate_allows_with_low_alignment_but_warns(caplog):
    trust = MagicMock()
    trust.can_do.return_value = True
    ohana = MagicMock()
    ohana.evaluate_action_alignment.return_value = 0.3
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._get_ohana", return_value=ohana), \
            patch("trust_integration._record_nonblocking"):
        import logging
        with caplog.at_level(logging.WARNING, logger="kai.trust_integration"):
            allowed, reason = gate_autonomous_action("chat", {}, conviction=5.0)
    assert allowed is True
    assert "Low Ohana alignment" in caplog.text


def test_gate_records_to_ledger():
    with patch("trust_integration._get_trust_core", return_value=None), \
            patch("trust_integration._get_ohana", return_value=None), \
            patch("trust_integration._record_nonblocking") as mock_rec:
        gate_autonomous_action("chat", {"user": "dainius"}, conviction=8.0)
    mock_rec.assert_called_once()
    call_kwargs = mock_rec.call_args
    assert call_kwargs[1]["event_type"] == "AUTONOMOUS_ACTION" or call_kwargs[0][0] == "AUTONOMOUS_ACTION"


def test_gate_never_raises_on_broken_trust_core():
    broken = MagicMock()
    broken.can_do.side_effect = RuntimeError("db gone")
    with patch("trust_integration._get_trust_core", return_value=broken), \
            patch("trust_integration._get_ohana", return_value=None), \
            patch("trust_integration._record_nonblocking"):
        # Should NOT raise — gate is fail-open
        try:
            gate_autonomous_action("chat", {})
        except RuntimeError:
            pytest.fail("gate_autonomous_action raised — must never raise")


def test_gate_skips_ohana_check_when_trust_refused():
    trust = MagicMock()
    trust.can_do.return_value = False
    trust.level_name = "DORMANT"
    ohana = MagicMock()
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._get_ohana", return_value=ohana), \
            patch("trust_integration._record_nonblocking"):
        gate_autonomous_action("income_generation", {})
    ohana.evaluate_action_alignment.assert_not_called()


# ── record_chat_response ──────────────────────────────────────────────────────

def test_record_chat_response_fires_nonblocking():
    with patch("trust_integration._record_nonblocking") as mock_rec, \
            patch("trust_integration._get_trust_core", return_value=None):
        record_chat_response("hello", "hi there", conviction=8.0, specialist="mistral")
    mock_rec.assert_called_once()


def test_record_chat_response_feeds_consistency_evidence():
    trust = MagicMock()
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._record_nonblocking"):
        record_chat_response("hello", "hi", conviction=8.0, specialist="mistral")
    trust.record_evidence.assert_called_once()
    args = trust.record_evidence.call_args[0]
    assert args[0] == "consistency"


def test_record_chat_response_no_evidence_below_threshold():
    trust = MagicMock()
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._record_nonblocking"):
        record_chat_response("hello", "hi", conviction=6.9, specialist="mistral")
    trust.record_evidence.assert_not_called()


def test_record_chat_response_never_raises_on_broken_core():
    broken = MagicMock()
    broken.record_evidence.side_effect = Exception("storage fail")
    with patch("trust_integration._get_trust_core", return_value=broken), \
            patch("trust_integration._record_nonblocking"):
        record_chat_response("hello", "hi", conviction=9.0, specialist="mistral")


# ── record_alignment_audit ────────────────────────────────────────────────────

def test_record_alignment_audit_fires():
    with patch("trust_integration._record_nonblocking") as mock_rec:
        record_alignment_audit(ohana_alignment=0.95, uptime_pct=1.0, notes="daily check")
    mock_rec.assert_called_once()
    _, kwargs = mock_rec.call_args[0], mock_rec.call_args[1]
    event_data = mock_rec.call_args[1].get("event_data") or mock_rec.call_args[0][2]
    assert event_data["ohana_alignment"] == 0.95


# ── get_trust_status ──────────────────────────────────────────────────────────

def test_get_trust_status_returns_dict():
    with patch("trust_integration._get_trust_core", return_value=None), \
            patch("trust_integration._get_ledger", return_value=None):
        status = get_trust_status()
    assert isinstance(status, dict)


def test_get_trust_status_includes_score():
    with patch("trust_integration._get_trust_core", return_value=None), \
            patch("trust_integration._get_ledger", return_value=None):
        status = get_trust_status()
    assert "score" in status


def test_get_trust_status_enriched_by_trust_core():
    trust = MagicMock()
    trust.status.return_value = {
        "level": 2,
        "level_name": "ASSISTANT",
        "granted_by": "operator",
        "scores": {"consistency": 0.5},
        "next_level": "AGENT",
        "progress_to_next": 0.3,
    }
    with patch("trust_integration._get_trust_core", return_value=trust), \
            patch("trust_integration._get_ledger", return_value=None):
        status = get_trust_status()
    assert status["level_name"] == "ASSISTANT"
    assert status["progress_to_next"] == 0.3


def test_get_trust_status_never_raises():
    with patch("trust_integration._get_trust_core", side_effect=Exception("boom")):
        status = get_trust_status()
    assert "level_name" in status
    assert status["level_name"] == "DORMANT"


# ── _record_nonblocking ───────────────────────────────────────────────────────

def test_record_nonblocking_silently_ignores_no_ledger():
    with patch("trust_integration._get_ledger", return_value=None), \
            patch("trust_integration._get_trust_core", return_value=None):
        _record_nonblocking("AUTONOMOUS_ACTION", "kai", {"x": 1}, capability="chat")


def test_record_nonblocking_writes_when_ledger_available():
    ledger = MagicMock()
    with patch("trust_integration._get_ledger", return_value=ledger), \
            patch("trust_integration._get_trust_core", return_value=None):
        _record_nonblocking("AUTONOMOUS_ACTION", "kai", {"x": 1}, capability="chat")
    ledger.append.assert_called_once()


def test_record_nonblocking_includes_trust_tier_when_available():
    ledger = MagicMock()
    trust = MagicMock()
    trust.level_name = "AGENT"
    with patch("trust_integration._get_ledger", return_value=ledger), \
            patch("trust_integration._get_trust_core", return_value=trust):
        _record_nonblocking("AUTONOMOUS_ACTION", "kai", {"x": 1}, capability="chat")
    call_kwargs = ledger.append.call_args[1]
    assert call_kwargs.get("trust_tier") == "AGENT"
