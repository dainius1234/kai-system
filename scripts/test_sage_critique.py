"""P23 — SAGE Multi-Agent Critique Loop tests.

Tests both layers of SAGE self-critique:
  1. Verifier self-critique (_self_critique) — detects groupthink,
     thin-evidence, unsupported-material, and signal contradictions
  2. Adversary self-review (challenge_self_review) — detects false
     consensus, degraded groupthink, conflicting findings, over-optimism
"""
from __future__ import annotations

import sys
import os
import unittest

# ── path setup ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Stub common modules before importing app code
from types import ModuleType

_policy = ModuleType("common.policy")
_policy.policy_hash = "test"
_policy.policy_version = "test"
_policy.verifier_thresholds = lambda: {
    "pass_threshold": 0.65,
    "repair_threshold": 0.35,
    "min_strong_chunks": 2,
    "strong_chunk_threshold": 0.60,
}
sys.modules.setdefault("common.policy", _policy)

_runtime = ModuleType("common.runtime")


class _FakeLogger:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def debug(self, *a, **kw): pass


class _FakeBudget:
    def __init__(self, **kwargs): pass
    def record(self, code): pass
    def snapshot(self): return {}


_runtime.setup_json_logger = lambda *a, **kw: _FakeLogger()
_runtime.detect_device = lambda: "cpu"
_runtime.ErrorBudget = _FakeBudget
sys.modules.setdefault("common.runtime", _runtime)

# Import verifier code explicitly from its directory
import importlib.util
_verifier_spec = importlib.util.spec_from_file_location(
    "verifier_app", os.path.join(ROOT, "verifier", "app.py"),
    submodule_search_locations=[],
)
_verifier_app = importlib.util.module_from_spec(_verifier_spec)
sys.modules["verifier_app"] = _verifier_app
_verifier_spec.loader.exec_module(_verifier_app)

Signal = _verifier_app.Signal
MaterialClaim = _verifier_app.MaterialClaim
_self_critique = _verifier_app._self_critique
_aggregate = _verifier_app._aggregate
PASS_THRESHOLD = _verifier_app.PASS_THRESHOLD
REPAIR_THRESHOLD = _verifier_app.REPAIR_THRESHOLD
MIN_STRONG_CHUNKS = _verifier_app.MIN_STRONG_CHUNKS

# Import adversary from langgraph directory
sys.path.insert(0, os.path.join(ROOT, "langgraph"))
from adversary import ChallengeResult, challenge_self_review  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Part 1: Verifier Self-Critique Tests
# ═══════════════════════════════════════════════════════════════════

class TestSelfCritiqueEmpty(unittest.TestCase):
    """Edge cases with empty/missing inputs."""

    def test_no_signals(self):
        sig = _self_critique([], 0, [])
        self.assertEqual(sig.strategy, "self_critique")
        self.assertEqual(sig.score, 0.5)
        self.assertIn("no signals", sig.detail)

    def test_single_signal(self):
        sig = _self_critique(
            [Signal(strategy="a", score=0.7, detail="ok")],
            strong_chunks=3,
            material_claims=[],
        )
        self.assertEqual(sig.strategy, "self_critique")
        # single signal can't trigger groupthink
        self.assertNotIn("groupthink", sig.detail)


class TestSelfCritiqueGroupthink(unittest.TestCase):
    """Groupthink: all signals suspiciously uniform."""

    def test_uniform_scores_flagged(self):
        signals = [
            Signal(strategy="a", score=0.70, detail="x"),
            Signal(strategy="b", score=0.72, detail="x"),
            Signal(strategy="c", score=0.71, detail="x"),
        ]
        sig = _self_critique(signals, 3, [])
        self.assertIn("groupthink", sig.detail)
        self.assertLess(sig.score, 0.71)  # penalised

    def test_diverse_scores_ok(self):
        signals = [
            Signal(strategy="a", score=0.9, detail="x"),
            Signal(strategy="b", score=0.5, detail="x"),
            Signal(strategy="c", score=0.7, detail="x"),
        ]
        sig = _self_critique(signals, 3, [])
        self.assertNotIn("groupthink", sig.detail)

    def test_two_signals_no_groupthink(self):
        """Groupthink requires >= 3 signals."""
        signals = [
            Signal(strategy="a", score=0.70, detail="x"),
            Signal(strategy="b", score=0.71, detail="x"),
        ]
        sig = _self_critique(signals, 3, [])
        self.assertNotIn("groupthink", sig.detail)


class TestSelfCritiqueThinEvidence(unittest.TestCase):
    """Thin-evidence: high avg but too few strong chunks."""

    def test_thin_evidence_flagged(self):
        signals = [
            Signal(strategy="a", score=0.8, detail="x"),
            Signal(strategy="b", score=0.7, detail="x"),
            Signal(strategy="c", score=0.9, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=1, material_claims=[])
        self.assertIn("thin-evidence", sig.detail)

    def test_sufficient_chunks_ok(self):
        signals = [
            Signal(strategy="a", score=0.8, detail="x"),
            Signal(strategy="b", score=0.7, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=3, material_claims=[])
        self.assertNotIn("thin-evidence", sig.detail)


class TestSelfCritiqueUnsupportedMaterial(unittest.TestCase):
    """Unsupported material claims with zero chunks."""

    def test_material_no_chunks_flagged(self):
        claims = [
            MaterialClaim(claim_type="numbers", raw_text="42%", confidence=0.9),
        ]
        signals = [
            Signal(strategy="a", score=0.5, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=0, material_claims=claims)
        self.assertIn("unsupported-material", sig.detail)

    def test_material_with_chunks_ok(self):
        claims = [
            MaterialClaim(claim_type="numbers", raw_text="42%", confidence=0.9),
        ]
        signals = [
            Signal(strategy="a", score=0.5, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=2, material_claims=claims)
        self.assertNotIn("unsupported-material", sig.detail)

    def test_no_material_claims_ok(self):
        signals = [
            Signal(strategy="a", score=0.5, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=0, material_claims=[])
        self.assertNotIn("unsupported-material", sig.detail)


class TestSelfCritiqueContradiction(unittest.TestCase):
    """Signal contradiction: large divergence between strategies."""

    def test_large_divergence_flagged(self):
        signals = [
            Signal(strategy="keyword", score=0.9, detail="x"),
            Signal(strategy="material", score=0.3, detail="x"),
        ]
        sig = _self_critique(signals, 2, [])
        self.assertIn("contradiction", sig.detail)

    def test_small_divergence_ok(self):
        signals = [
            Signal(strategy="keyword", score=0.7, detail="x"),
            Signal(strategy="material", score=0.5, detail="x"),
        ]
        sig = _self_critique(signals, 2, [])
        self.assertNotIn("contradiction", sig.detail)


class TestSelfCritiqueMultipleIssues(unittest.TestCase):
    """Multiple issues compound the penalty."""

    def test_groupthink_plus_thin_evidence(self):
        signals = [
            Signal(strategy="a", score=0.70, detail="x"),
            Signal(strategy="b", score=0.72, detail="x"),
            Signal(strategy="c", score=0.71, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=0, material_claims=[])
        self.assertIn("groupthink", sig.detail)
        self.assertIn("thin-evidence", sig.detail)
        # Two issues → 2 * 0.15 = 0.30 penalty
        self.assertLess(sig.score, 0.5)

    def test_three_issues_floor_at_zero(self):
        claims = [MaterialClaim(claim_type="n", raw_text="1", confidence=0.8)]
        signals = [
            Signal(strategy="a", score=0.40, detail="x"),
            Signal(strategy="b", score=0.40, detail="x"),
            Signal(strategy="c", score=0.40, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=0, material_claims=claims)
        self.assertGreaterEqual(sig.score, 0.0)


class TestSelfCritiqueClean(unittest.TestCase):
    """Clean signals produce a healthy critique."""

    def test_clean_signals(self):
        signals = [
            Signal(strategy="memory_cross_ref", score=0.8, detail="x"),
            Signal(strategy="self_consistency", score=0.6, detail="x"),
            Signal(strategy="keyword_plausibility", score=0.7, detail="x"),
            Signal(strategy="material_claims", score=0.9, detail="x"),
        ]
        sig = _self_critique(signals, strong_chunks=3, material_claims=[])
        self.assertIn("coherent", sig.detail)
        self.assertGreater(sig.score, 0.5)


# ═══════════════════════════════════════════════════════════════════
# Part 2: Adversary Self-Review Tests
# ═══════════════════════════════════════════════════════════════════

def _make_result(strategy="test", passed=True, confidence=0.8,
                 finding="ok", modifier=0.2):
    return ChallengeResult(
        strategy=strategy, passed=passed, confidence=confidence,
        finding=finding, modifier=modifier,
    )


class TestSelfReviewEmpty(unittest.TestCase):

    def test_no_challenges(self):
        result = challenge_self_review([])
        self.assertEqual(result.strategy, "self_review")
        self.assertTrue(result.passed)
        self.assertEqual(result.modifier, 0.0)


class TestSelfReviewFalseConsensus(unittest.TestCase):
    """All passed but with low confidence."""

    def test_low_confidence_consensus(self):
        challenges = [
            _make_result("a", passed=True, confidence=0.3, modifier=0.1),
            _make_result("b", passed=True, confidence=0.4, modifier=0.1),
            _make_result("c", passed=True, confidence=0.2, modifier=0.1),
        ]
        result = challenge_self_review(challenges)
        self.assertFalse(result.passed)
        self.assertIn("false-consensus", result.finding)

    def test_high_confidence_consensus_ok(self):
        challenges = [
            _make_result("a", passed=True, confidence=0.8, modifier=0.2),
            _make_result("b", passed=True, confidence=0.7, modifier=0.2),
        ]
        result = challenge_self_review(challenges)
        self.assertNotIn("false-consensus", result.finding)


class TestSelfReviewDegradedGroupthink(unittest.TestCase):
    """Multiple challenges appear degraded (modifier=0, low conf)."""

    def test_two_degraded(self):
        challenges = [
            _make_result("verifier", passed=True, confidence=0.2, modifier=0.0),
            _make_result("policy", passed=True, confidence=0.2, modifier=0.0),
            _make_result("consistency", passed=True, confidence=0.85, modifier=0.2),
        ]
        result = challenge_self_review(challenges)
        self.assertIn("degraded-groupthink", result.finding)

    def test_one_degraded_ok(self):
        challenges = [
            _make_result("verifier", passed=True, confidence=0.2, modifier=0.0),
            _make_result("consistency", passed=True, confidence=0.85, modifier=0.2),
        ]
        result = challenge_self_review(challenges)
        self.assertNotIn("degraded-groupthink", result.finding)


class TestSelfReviewConflict(unittest.TestCase):
    """Conflicting findings between paired challenges."""

    def test_verifier_history_conflict(self):
        challenges = [
            _make_result("verifier", passed=True, confidence=0.8, modifier=0.3),
            _make_result("history", passed=False, confidence=0.9, modifier=-1.0),
        ]
        result = challenge_self_review(challenges)
        self.assertIn("conflict", result.finding)

    def test_same_direction_no_conflict(self):
        challenges = [
            _make_result("verifier", passed=True, confidence=0.8, modifier=0.3),
            _make_result("history", passed=True, confidence=0.7, modifier=0.2),
        ]
        result = challenge_self_review(challenges)
        self.assertNotIn("conflict", result.finding)

    def test_policy_consistency_conflict(self):
        challenges = [
            _make_result("policy", passed=False, confidence=0.9, modifier=-1.0),
            _make_result("consistency", passed=True, confidence=0.85, modifier=0.2),
        ]
        result = challenge_self_review(challenges)
        self.assertIn("conflict", result.finding)


class TestSelfReviewOverOptimism(unittest.TestCase):
    """Positive total modifier despite multiple failures."""

    def test_over_optimism_flagged(self):
        challenges = [
            _make_result("a", passed=True, confidence=0.8, modifier=2.0),
            _make_result("b", passed=False, confidence=0.6, modifier=-0.5),
            _make_result("c", passed=False, confidence=0.6, modifier=-0.5),
        ]
        # total modifier = 2.0 - 0.5 - 0.5 = 1.0 (positive)
        # but 2 failures
        result = challenge_self_review(challenges)
        self.assertIn("over-optimism", result.finding)

    def test_negative_modifier_no_over_optimism(self):
        challenges = [
            _make_result("a", passed=False, confidence=0.8, modifier=-1.0),
            _make_result("b", passed=False, confidence=0.7, modifier=-0.8),
        ]
        result = challenge_self_review(challenges)
        self.assertNotIn("over-optimism", result.finding)


class TestSelfReviewClean(unittest.TestCase):
    """All healthy — no issues."""

    def test_clean_challenges(self):
        challenges = [
            _make_result("history", passed=True, confidence=0.8, modifier=0.2),
            _make_result("verifier", passed=True, confidence=0.7, modifier=0.3),
            _make_result("policy", passed=True, confidence=0.8, modifier=0.1),
            _make_result("consistency", passed=True, confidence=0.85, modifier=0.2),
            _make_result("calibration", passed=True, confidence=0.7, modifier=0.1),
        ]
        result = challenge_self_review(challenges)
        self.assertTrue(result.passed)
        self.assertIn("coherent", result.finding)
        self.assertEqual(result.modifier, 0.1)


class TestSelfReviewMultipleIssues(unittest.TestCase):
    """Multiple meta-issues compound the penalty."""

    def test_two_issues(self):
        challenges = [
            _make_result("verifier", passed=True, confidence=0.2, modifier=0.0),
            _make_result("policy", passed=True, confidence=0.2, modifier=0.0),
            _make_result("history", passed=True, confidence=0.3, modifier=0.0),
        ]
        result = challenge_self_review(challenges)
        # Should flag: false-consensus (all passed, avg conf < 0.5)
        # and degraded-groupthink (3 with modifier=0, conf<=0.3)
        self.assertFalse(result.passed)
        issue_count = result.finding.count(":")
        self.assertGreater(len(result.evidence), 1)

    def test_penalty_capped_at_one(self):
        """Modifier should not exceed -1.0."""
        challenges = [
            _make_result("verifier", passed=True, confidence=0.2, modifier=0.0),
            _make_result("policy", passed=True, confidence=0.2, modifier=0.0),
            _make_result("history", passed=True, confidence=0.3, modifier=0.0),
            _make_result("consistency", passed=True, confidence=0.1, modifier=0.0),
        ]
        result = challenge_self_review(challenges)
        self.assertGreaterEqual(result.modifier, -1.0)


# ═══════════════════════════════════════════════════════════════════
# Part 3: Integration — self-critique affects aggregate verdict
# ═══════════════════════════════════════════════════════════════════

class TestSelfCritiqueAffectsVerdict(unittest.TestCase):
    """Verify that self-critique can change the aggregate verdict."""

    def test_groupthink_downgrades_pass_to_repair(self):
        """Uniform signals at threshold → self-critique pulls avg below PASS."""
        signals = [
            Signal(strategy="a", score=0.66, detail="x"),
            Signal(strategy="b", score=0.66, detail="x"),
            Signal(strategy="c", score=0.66, detail="x"),
        ]
        critique = _self_critique(signals, strong_chunks=0, material_claims=[])
        all_signals = signals + [critique]
        verdict, conf, summary = _aggregate(all_signals, strong_chunks=0)
        # groupthink + thin-evidence should pull avg down
        self.assertIn(verdict, ("REPAIR", "FAIL_CLOSED"))

    def test_clean_critique_preserves_pass(self):
        signals = [
            Signal(strategy="a", score=0.8, detail="x"),
            Signal(strategy="b", score=0.6, detail="x"),
            Signal(strategy="c", score=0.9, detail="x"),
            Signal(strategy="d", score=0.7, detail="x"),
        ]
        critique = _self_critique(signals, strong_chunks=3, material_claims=[])
        all_signals = signals + [critique]
        verdict, conf, summary = _aggregate(all_signals, strong_chunks=3)
        self.assertEqual(verdict, "PASS")


if __name__ == "__main__":
    unittest.main()
