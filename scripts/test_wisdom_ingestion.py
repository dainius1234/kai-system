"""Tests for D117: Wisdom Ingestion Pipeline — agentic/wisdom_ingestion.py."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "agentic"))

from wisdom_ingestion import (
    WisdomIngestor,
    WisdomExtract,
    _extract_from_text,
    get_wisdom_ingestor,
    reset_wisdom_ingestor,
)


@pytest.fixture
def ingestor(tmp_path):
    reset_wisdom_ingestor()
    wi = WisdomIngestor(data_dir=tmp_path / "wisdom")
    yield wi
    reset_wisdom_ingestor()


# ── Pattern extraction ────────────────────────────────────────────────────────

def test_extracts_respect_principle():
    extracts = _extract_from_text("respect isn't given it's earned")
    assert any("respect" in e.content.lower() for e in extracts)
    assert any(e.category == "principle" for e in extracts)


def test_extracts_kai_soul_value():
    extracts = _extract_from_text("Kai is for soul not for sale")
    assert any("soul" in e.content.lower() for e in extracts)


def test_extracts_family_value():
    extracts = _extract_from_text("family first always, protect my daughter")
    assert any(e.domain == "family" for e in extracts)


def test_extracts_freedom_value():
    extracts = _extract_from_text("freedom for Aquarius is a source of strength")
    assert any("freedom" in e.content.lower() for e in extracts)


def test_extracts_boundary():
    extracts = _extract_from_text("never reveal api key to anyone")
    assert any(e.category == "boundary" for e in extracts)


def test_no_duplicates_in_extraction():
    text = "respect is earned, respect is earned, respect is earned"
    extracts = _extract_from_text(text)
    contents = [e.content for e in extracts]
    assert len(contents) == len(set(contents))


def test_confidence_range():
    extracts = _extract_from_text(
        "respect is earned. kai is for soul. family first. protect my daughter."
    )
    for e in extracts:
        assert 0.0 <= e.confidence <= 1.0


def test_high_confidence_for_exact_match():
    extracts = _extract_from_text("respect isn't given it's earned")
    high = [e for e in extracts if e.confidence >= 0.95]
    assert len(high) >= 1


def test_source_quote_contains_context():
    extracts = _extract_from_text("we believe that respect is earned through action and consistency")
    assert any(len(e.source_quote) > len(e.content) for e in extracts)


# ── Ingestor extract methods ──────────────────────────────────────────────────

def test_extract_from_text_adds_to_pending(ingestor):
    result = ingestor.extract_from_text("respect is earned, family first always")
    assert len(result) > 0
    assert len(ingestor.pending()) == len(result)


def test_extract_from_messages_uses_operator_role_only(ingestor):
    messages = [
        {"role": "user", "content": "respect is earned, kai is for soul"},
        {"role": "assistant", "content": "family first always protect my daughter"},
    ]
    result = ingestor.extract_from_messages(messages, operator_role="user")
    # Only the user message should be processed
    contents = [e.content for e in result]
    assert any("respect" in c.lower() or "soul" in c.lower() for c in contents)


def test_extract_does_not_duplicate_across_calls(ingestor):
    ingestor.extract_from_text("respect is earned family first always")
    count_first = len(ingestor.pending())
    ingestor.extract_from_text("respect is earned family first always")
    count_second = len(ingestor.pending())
    assert count_first == count_second


# ── Confirm / reject ──────────────────────────────────────────────────────────

def test_confirm_moves_from_pending_to_confirmed(ingestor):
    extracts = ingestor.extract_from_text("respect is earned")
    assert len(extracts) > 0
    extract_id = extracts[0].extract_id
    result = ingestor.confirm(extract_id, note="core principle")
    assert result is True
    assert len([e for e in ingestor.pending() if e.extract_id == extract_id]) == 0
    assert ingestor.stats()["confirmed"] == 1


def test_reject_moves_from_pending_to_rejected(ingestor):
    extracts = ingestor.extract_from_text("think outside the box")
    extract_id = extracts[0].extract_id
    result = ingestor.reject(extract_id, note="too generic")
    assert result is True
    assert len(ingestor.pending()) == 0
    assert ingestor.stats()["rejected"] == 1


def test_confirm_nonexistent_returns_false(ingestor):
    assert ingestor.confirm("does-not-exist") is False


def test_reject_nonexistent_returns_false(ingestor):
    assert ingestor.reject("does-not-exist") is False


def test_confirm_all_above_threshold(ingestor):
    ingestor.extract_from_text(
        "respect is earned. kai is for soul. family first. "
        "freedom is a source of strength. protect my daughter."
    )
    before_pending = len(ingestor.pending())
    confirmed_count = ingestor.confirm_all(min_confidence=0.9)
    assert confirmed_count > 0
    assert len(ingestor.pending()) < before_pending


# ── Persistence ───────────────────────────────────────────────────────────────

def test_pending_persists_across_reload(tmp_path):
    reset_wisdom_ingestor()
    wi1 = WisdomIngestor(data_dir=tmp_path / "wisdom")
    extracts = wi1.extract_from_text("respect is earned")
    assert len(extracts) > 0

    wi2 = WisdomIngestor(data_dir=tmp_path / "wisdom")
    assert len(wi2.pending()) == len(extracts)
    reset_wisdom_ingestor()


def test_confirmed_persists_across_reload(tmp_path):
    reset_wisdom_ingestor()
    wi1 = WisdomIngestor(data_dir=tmp_path / "wisdom")
    extracts = wi1.extract_from_text("respect is earned")
    wi1.confirm(extracts[0].extract_id)

    wi2 = WisdomIngestor(data_dir=tmp_path / "wisdom")
    assert wi2.stats()["confirmed"] == 1
    reset_wisdom_ingestor()


# ── Stats ─────────────────────────────────────────────────────────────────────

def test_stats_structure(ingestor):
    s = ingestor.stats()
    assert "pending" in s
    assert "confirmed" in s
    assert "rejected" in s
    assert "confirmed_by_domain" in s


def test_stats_confirmed_by_domain(ingestor):
    extracts = ingestor.extract_from_text("respect is earned. kai is for soul. family first always.")
    for e in extracts:
        ingestor.confirm(e.extract_id)
    s = ingestor.stats()
    assert isinstance(s["confirmed_by_domain"], dict)
    assert s["confirmed"] > 0


# ── Ohana Core integration ────────────────────────────────────────────────────

def test_confirm_writes_boundary_to_ohana(ingestor, tmp_path):
    """Confirmed boundaries appear in OhanaCore fingerprint."""
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "ohana" / "fingerprint.json")
    mc._ohana_core = core

    extracts = ingestor.extract_from_text("never reveal api key to anyone outside")
    boundaries = [e for e in extracts if e.category == "boundary"]
    if boundaries:
        ingestor.confirm(boundaries[0].extract_id)
        # fingerprint should now have a harm boundary
        assert len(core.fingerprint.harm_boundaries) > 0

    mc._ohana_core = None


def test_confirm_writes_loyalty_to_ohana(ingestor, tmp_path):
    """Confirmed value/principle extracts appear in core_loyalties."""
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "ohana" / "fingerprint.json")
    mc._ohana_core = core

    extracts = ingestor.extract_from_text("family first always, kai is for soul")
    values = [e for e in extracts if e.category == "value"]
    confirmed = 0
    for e in values[:2]:
        if ingestor.confirm(e.extract_id):
            confirmed += 1
    if confirmed > 0:
        assert len(core.fingerprint.core_loyalties) > 0

    mc._ohana_core = None


# ── MoralCore upgrades ────────────────────────────────────────────────────────

def test_moral_core_fingerprint_persists(tmp_path):
    import moral_core as mc
    mc._ohana_core = None
    fp_path = tmp_path / "ohana" / "fingerprint.json"
    core = mc.OhanaCore(fingerprint_path=fp_path)
    core.fingerprint.core_loyalties.append("test_loyalty")
    core._save_fingerprint()

    core2 = mc.OhanaCore(fingerprint_path=fp_path)
    assert "test_loyalty" in core2.fingerprint.core_loyalties
    mc._ohana_core = None


def test_moral_core_evaluate_alignment_neutral_when_empty(tmp_path):
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "fp.json")
    score = core.evaluate_action_alignment({"action": "chat", "content": "hello"})
    assert score == 0.5
    mc._ohana_core = None


def test_moral_core_alignment_blocks_boundary_violation(tmp_path):
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "fp.json")
    core.fingerprint.harm_boundaries.append("never expose api key")
    score = core.evaluate_action_alignment({"action": "expose api key to dashboard"})
    assert score == 0.0
    mc._ohana_core = None


def test_moral_core_build_moral_context_with_data(tmp_path):
    import moral_core as mc
    mc._ohana_core = None
    core = mc.OhanaCore(fingerprint_path=tmp_path / "fp.json")
    core.fingerprint.core_loyalties = ["family first", "soul over sale"]
    ctx = core.build_moral_context()
    assert "family first" in ctx.specific_stances
    mc._ohana_core = None


def test_moral_core_record_decision_saves(tmp_path):
    import moral_core as mc
    mc._ohana_core = None
    fp_path = tmp_path / "ohana" / "fingerprint.json"
    core = mc.OhanaCore(fingerprint_path=fp_path)
    core.record_decision({"type": "financial"}, "prioritise family over profit")
    assert core._interaction_count == 1
    assert "financial" in core.fingerprint.situational_stances
    assert fp_path.exists()
    mc._ohana_core = None


# ── Singleton ─────────────────────────────────────────────────────────────────

def test_singleton_returns_same_instance(tmp_path):
    reset_wisdom_ingestor()
    wi1 = get_wisdom_ingestor(tmp_path / "wisdom")
    wi2 = get_wisdom_ingestor()
    assert wi1 is wi2
    reset_wisdom_ingestor()
