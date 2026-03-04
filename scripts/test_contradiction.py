"""Tests for P4 — Contradiction Memory (TMC).

Validates the detect_contradiction engine in memu-core with
numeric drift, negation flip, and no-conflict scenarios.
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "memu-core"))
sys.path.insert(0, str(ROOT / "common"))

spec = importlib.util.spec_from_file_location("memu_app", ROOT / "memu-core" / "app.py")
memu = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memu
spec.loader.exec_module(memu)

detect_contradiction = memu.detect_contradiction
ContradictionResult = memu.ContradictionResult
_extract_numeric_claims = memu._extract_numeric_claims
MemoryRecord = memu.MemoryRecord


def _make_record(text: str, record_id: str = "rec-1", poisoned: bool = False) -> object:
    """Build a minimal MemoryRecord for testing."""
    return MemoryRecord(
        id=record_id,
        timestamp="2025-01-01T00:00:00",
        event_type="assertion",
        category="general",
        content={"result": text},
        embedding=[0.0] * 10,
        relevance=1.0,
        importance=0.5,
        poisoned=poisoned,
    )


class TestContradictionDetection(unittest.TestCase):
    """P4: TMC contradiction detection engine."""

    # ── Numeric drift ───────────────────────────────────────────────

    def test_numeric_drift_detects_different_values(self):
        """Same topic with different numbers = conflict."""
        existing = [_make_record("VAT registration threshold is £85,000 per year")]
        result = detect_contradiction("VAT registration threshold is £90,000 per year", existing)
        self.assertTrue(result.has_conflict)
        self.assertEqual(result.conflict_type, "numeric_drift")
        self.assertIn("85", result.explanation)
        self.assertIn("90", result.explanation)

    def test_numeric_drift_same_value_no_conflict(self):
        """Same number on same topic = no conflict."""
        existing = [_make_record("Risk limit is set to 5% maximum drawdown")]
        result = detect_contradiction("Risk limit is set to 5% maximum drawdown", existing)
        self.assertFalse(result.has_conflict)

    def test_numeric_drift_unrelated_topics(self):
        """Different topics with different numbers = no conflict (low overlap)."""
        existing = [_make_record("Temperature sensor reads 25 degrees")]
        result = detect_contradiction("Invoice amount is £25,000 for project Alpha", existing)
        self.assertFalse(result.has_conflict)

    # ── Negation flip ───────────────────────────────────────────────

    def test_negation_flip_detects_polarity_change(self):
        """'X is required' vs 'X is not required' = negation flip."""
        existing = [_make_record("Safety helmet is required on site at all times")]
        result = detect_contradiction("Safety helmet is not required on site at all times", existing)
        self.assertTrue(result.has_conflict)
        self.assertEqual(result.conflict_type, "negation_flip")

    def test_negation_flip_no_flip(self):
        """Both positive assertions on same topic = no conflict."""
        existing = [_make_record("Safety helmet is required on site")]
        result = detect_contradiction("Safety helmet is required on site always", existing)
        self.assertFalse(result.has_conflict)

    # ── No conflict scenarios ───────────────────────────────────────

    def test_no_conflict_empty_records(self):
        """No existing records = no conflict."""
        result = detect_contradiction("Any assertion about anything", [])
        self.assertFalse(result.has_conflict)

    def test_no_conflict_empty_text(self):
        """Empty new text = no conflict."""
        existing = [_make_record("Some existing memory")]
        result = detect_contradiction("", existing)
        self.assertFalse(result.has_conflict)

    def test_no_conflict_unrelated_topics(self):
        """Completely unrelated topics = no conflict."""
        existing = [_make_record("Python uses indentation for blocks")]
        result = detect_contradiction("The weather in London is rainy today", existing)
        self.assertFalse(result.has_conflict)

    def test_poisoned_records_skipped(self):
        """Quarantined records should not trigger contradictions."""
        rec = _make_record("VAT threshold is £85,000", poisoned=True)
        result = detect_contradiction("VAT threshold is £90,000", [rec])
        self.assertFalse(result.has_conflict)

    # ── Numeric extraction ──────────────────────────────────────────

    def test_extract_numeric_claims_currency(self):
        claims = _extract_numeric_claims("The cost is £85,000 and €50k deposit")
        self.assertGreaterEqual(len(claims), 2)

    def test_extract_numeric_claims_percentages(self):
        claims = _extract_numeric_claims("Growth rate is 5.5% over 90 days")
        self.assertGreaterEqual(len(claims), 1)

    def test_extract_numeric_claims_empty(self):
        claims = _extract_numeric_claims("No numbers here at all")
        self.assertEqual(len(claims), 0)

    # ── ContradictionResult defaults ────────────────────────────────

    def test_default_result_is_no_conflict(self):
        r = ContradictionResult()
        self.assertFalse(r.has_conflict)
        self.assertEqual(r.conflict_type, "")
        self.assertEqual(r.similarity, 0.0)


if __name__ == "__main__":
    unittest.main()
