"""Tests for D126: Trust Promotion Gate — agentic/trust_core.py + app.py endpoints."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from trust_core import (
    TrustCore,
    TrustLevel,
    PROMOTION_THRESHOLDS,
    get_trust_core,
    reset_trust_core,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _core(tmp_path: Path | None = None) -> TrustCore:
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp()) / "trust"
    return TrustCore(data_dir=tmp_path)


def _at_level(tmp_path: Path, level: TrustLevel) -> TrustCore:
    tc = _core(tmp_path)
    tc.grant(level, by="test")
    return tc


# ── promotion_readiness: GUARDIAN ceiling ────────────────────────────────────

def test_readiness_guardian_returns_no_next(tmp_path):
    tc = _at_level(tmp_path, TrustLevel.GUARDIAN)
    r = tc.promotion_readiness()
    assert r["next_level"] is None
    assert r["auto_eligible"] is False
    assert "GUARDIAN" in r["summary"]


def test_readiness_guardian_gaps_empty(tmp_path):
    tc = _at_level(tmp_path, TrustLevel.GUARDIAN)
    r = tc.promotion_readiness()
    assert r["gaps"] == {}
    assert r["thresholds"] == {}


# ── promotion_readiness: DORMANT → OBSERVER ───────────────────────────────────

def test_readiness_dormant_next_is_observer(tmp_path):
    tc = _core(tmp_path)
    r = tc.promotion_readiness()
    assert r["current_level"] == "DORMANT"
    assert r["next_level"] == "OBSERVER"


def test_readiness_observer_thresholds_all_low(tmp_path):
    """OBSERVER thresholds are tiny; fresh core is not yet eligible."""
    tc = _core(tmp_path)
    r = tc.promotion_readiness()
    # OBSERVER requires consistency=1.0, all others 0 — so we should have a gap
    assert r["gaps"]["consistency"] > 0


def test_readiness_auto_eligible_when_thresholds_met(tmp_path):
    tc = _core(tmp_path)
    # OBSERVER needs consistency=1.0 — recording it triggers auto-promotion to OBSERVER.
    # readiness() then reports OBSERVER → ASSISTANT (not yet eligible).
    tc.record_evidence("consistency", 1.0, "test")
    r = tc.promotion_readiness()
    assert r["current_level"] == "OBSERVER"   # auto-promoted
    assert r["next_level"] == "ASSISTANT"      # now reporting next hop


def test_readiness_summary_contains_gaps(tmp_path):
    tc = _core(tmp_path)
    r = tc.promotion_readiness()
    assert "Gaps:" in r["summary"] or "more needed" in r["summary"]


def test_readiness_summary_eligible_when_met(tmp_path):
    tc = _at_level(tmp_path, TrustLevel.OBSERVER)
    # At OBSERVER, recording enough evidence auto-promotes to ASSISTANT.
    # After that, readiness() reports ASSISTANT → AGENT (next hop), not yet eligible.
    tc.record_evidence("consistency", 5.0, "test")
    tc.record_evidence("judgment", 3.0, "test")
    tc.record_evidence("values", 2.0, "test")
    r = tc.promotion_readiness()
    assert r["current_level"] == "ASSISTANT"  # auto-promoted
    assert r["next_level"] == "AGENT"


# ── promotion_readiness: gaps are non-negative ────────────────────────────────

def test_readiness_gaps_never_negative(tmp_path):
    tc = _core(tmp_path)
    # Overshoot a dimension
    tc.record_evidence("consistency", 999.0, "test")
    r = tc.promotion_readiness()
    for gap in r["gaps"].values():
        assert gap >= 0.0


def test_readiness_includes_scores(tmp_path):
    tc = _core(tmp_path)
    tc.record_evidence("consistency", 0.5, "test")
    r = tc.promotion_readiness()
    assert "scores" in r
    assert r["scores"]["consistency"] >= 0.5


def test_readiness_level_ints_correct(tmp_path):
    tc = _core(tmp_path)
    r = tc.promotion_readiness()
    assert r["current_level_int"] == TrustLevel.DORMANT.value
    assert r["next_level_int"] == TrustLevel.OBSERVER.value


# ── HTTP endpoints via TestClient ─────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path):
    """TestClient with trust_core reset to a fresh tmp_path instance."""
    import os
    os.environ["FF_MODEL_COUNCIL"] = "false"
    os.environ["FF_WEB_SCOUT"] = "false"
    os.environ["FF_SERVICE_WATCHDOG"] = "false"
    os.environ["FF_PAPER_TRADING"] = "false"

    reset_trust_core()
    # Patch get_trust_core to return an isolated instance
    fresh = TrustCore(data_dir=tmp_path / "trust")

    with patch("app.get_trust_core", return_value=fresh):
        import app as app_module
        with TestClient(app_module.app) as c:
            yield c

    reset_trust_core()


def test_trust_status_endpoint(client):
    r = client.get("/trust/status")
    assert r.status_code == 200
    data = r.json()
    assert "level" in data
    assert "level_name" in data
    assert "scores" in data


def test_trust_readiness_endpoint(client):
    r = client.get("/trust/readiness")
    assert r.status_code == 200
    data = r.json()
    assert "current_level" in data
    assert "next_level" in data
    assert "auto_eligible" in data
    assert "summary" in data


def test_trust_promote_valid(client):
    r = client.post("/trust/promote", json={"level": TrustLevel.OBSERVER.value})
    assert r.status_code == 200
    data = r.json()
    assert data["granted"] == "OBSERVER"
    assert data["level"] == TrustLevel.OBSERVER.value


def test_trust_promote_invalid_level(client):
    r = client.post("/trust/promote", json={"level": 99})
    assert r.status_code == 422


def test_trust_demote_valid(client):
    # First promote to AGENT
    client.post("/trust/promote", json={"level": TrustLevel.AGENT.value})
    r = client.post("/trust/demote", json={"level": TrustLevel.OBSERVER.value, "reason": "test revoke"})
    assert r.status_code == 200
    data = r.json()
    assert data["revoked_to"] == "OBSERVER"
    assert data["reason"] == "test revoke"


def test_trust_demote_invalid_level(client):
    r = client.post("/trust/demote", json={"level": -1, "reason": "bad"})
    assert r.status_code == 422


def test_trust_demote_missing_reason(client):
    r = client.post("/trust/demote", json={"level": 1})
    assert r.status_code == 422


def test_trust_audit_endpoint(client):
    r = client.get("/trust/audit")
    assert r.status_code == 200
    data = r.json()
    assert "events" in data
    assert isinstance(data["events"], list)


def test_trust_audit_limit_param(client):
    # Generate some events
    for i in range(5):
        client.post("/trust/promote", json={"level": i % 4})
    r = client.get("/trust/audit?limit=3")
    assert r.status_code == 200
    assert len(r.json()["events"]) <= 3


def test_trust_promote_then_status_reflects_change(client):
    client.post("/trust/promote", json={"level": TrustLevel.PARTNER.value})
    r = client.get("/trust/status")
    data = r.json()
    assert data["level"] == TrustLevel.PARTNER.value
    assert data["level_name"] == "PARTNER"


def test_trust_readiness_after_promote_shows_next(client):
    client.post("/trust/promote", json={"level": TrustLevel.OBSERVER.value})
    r = client.get("/trust/readiness")
    data = r.json()
    assert data["current_level"] == "OBSERVER"
    assert data["next_level"] == "ASSISTANT"
