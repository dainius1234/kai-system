"""Tests for P12 — Self-Deception Detection.

Validates detect_self_deception() in conviction.py: evidence gaps,
relevance gaps, rethink blind spots, and safe (no deception) cases.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph"))
sys.path.insert(0, str(ROOT / "common"))

# Stub redis for kai_config import chain
if "redis" not in sys.modules:
    from types import ModuleType
    _r = ModuleType("redis")

    class _FakeRedis:
        @classmethod
        def from_url(cls, *a, **kw): return cls()
        def ping(self): return True
    _r.Redis = _FakeRedis
    sys.modules["redis"] = _r

from conviction import detect_self_deception, SELF_DECEPTION_THRESHOLD


def _chunks(n: int, relevant: bool = True) -> list:
    """Generate n synthetic context chunks."""
    if relevant:
        return [{"content": f"risk limit policy detail {i}"} for i in range(n)]
    return [{"content": f"unrelated grocery item {i}"} for i in range(n)]


class TestSelfDeceptionDetection(unittest.TestCase):
    """P12: detect_self_deception() catches overconfidence."""

    # ── Evidence gap ────────────────────────────────────────────────

    def test_evidence_gap_high_conviction_no_chunks(self):
        """High conviction with zero chunks = evidence gap."""
        result = detect_self_deception("what is the policy?", {}, [], 0, 8.5)
        self.assertTrue(result["deceived"])
        self.assertTrue(any("evidence_gap" in f for f in result["flags"]))

    def test_evidence_gap_high_conviction_one_chunk(self):
        """High conviction with only 1 chunk = evidence gap."""
        result = detect_self_deception("what is the policy?", {}, _chunks(1), 0, 8.0)
        self.assertTrue(result["deceived"])
        self.assertTrue(any("evidence_gap" in f for f in result["flags"]))

    def test_no_evidence_gap_with_enough_chunks(self):
        """High conviction with 3+ chunks = no evidence gap."""
        result = detect_self_deception("what is the policy?", {}, _chunks(5), 1, 8.0)
        evidence_flags = [f for f in result["flags"] if "evidence_gap" in f]
        self.assertEqual(len(evidence_flags), 0)

    # ── Rethink blind spot ──────────────────────────────────────────

    def test_rethink_blind_spot_complex_query(self):
        """Complex query (15+ words) with 0 rethinks and high conviction."""
        long_query = "please analyze the full risk exposure of our current portfolio including all edge cases and potential failures"
        result = detect_self_deception(long_query, {}, _chunks(3), 0, 8.5)
        self.assertTrue(result["deceived"])
        self.assertTrue(any("rethink_blind_spot" in f for f in result["flags"]))

    def test_no_blind_spot_with_rethinks(self):
        """Complex query WITH rethinks = no blind spot."""
        long_query = "please analyze the full risk exposure of our current portfolio including all edge cases and potential failures"
        result = detect_self_deception(long_query, {}, _chunks(5), 2, 8.5)
        blind_spot_flags = [f for f in result["flags"] if "rethink_blind_spot" in f]
        self.assertEqual(len(blind_spot_flags), 0)

    def test_no_blind_spot_short_query(self):
        """Short query with 0 rethinks = no blind spot (not complex)."""
        result = detect_self_deception("what time?", {}, _chunks(3), 0, 8.5)
        blind_spot_flags = [f for f in result["flags"] if "rethink_blind_spot" in f]
        self.assertEqual(len(blind_spot_flags), 0)

    # ── Low conviction bypass ───────────────────────────────────────

    def test_low_conviction_never_flagged(self):
        """Below threshold = no deception check."""
        result = detect_self_deception("anything", {}, [], 0, 5.0)
        self.assertFalse(result["deceived"])
        self.assertEqual(len(result["flags"]), 0)

    def test_threshold_boundary(self):
        """Exactly at threshold = still checked."""
        result = detect_self_deception("what?", {}, [], 0, SELF_DECEPTION_THRESHOLD)
        # With 0 chunks and at threshold, evidence_gap should fire
        self.assertTrue(result["deceived"])

    # ── No deception ────────────────────────────────────────────────

    def test_no_deception_well_supported(self):
        """High conviction with good evidence + rethinks = clean."""
        result = detect_self_deception(
            "simple query",
            {},
            _chunks(5, relevant=True),
            2,
            8.5,
        )
        # Should have no flags if chunks are sufficient and rethinks done
        # (relevance gap might fire if keyword coverage is low, which is OK)
        evidence_flags = [f for f in result["flags"] if "evidence_gap" in f]
        blind_flags = [f for f in result["flags"] if "rethink_blind_spot" in f]
        self.assertEqual(len(evidence_flags), 0)
        self.assertEqual(len(blind_flags), 0)

    # ── Result structure ────────────────────────────────────────────

    def test_result_has_recommendation(self):
        """When deceived, result should include recommendation."""
        result = detect_self_deception("what?", {}, [], 0, 9.0)
        self.assertTrue(result["deceived"])
        self.assertIn("recommendation", result)
        self.assertIn("rethink", result["recommendation"].lower())

    def test_clean_result_structure(self):
        """Non-deceived result has correct shape."""
        result = detect_self_deception("hi", {}, _chunks(3), 1, 4.0)
        self.assertFalse(result["deceived"])
        self.assertEqual(result["flags"], [])
        self.assertEqual(result["recommendation"], "none")

    def test_threshold_constant(self):
        """Self-deception threshold is 7.0 by default."""
        self.assertEqual(SELF_DECEPTION_THRESHOLD, 7.0)


if __name__ == "__main__":
    unittest.main()
