"""Tests for P2: SELAUR — Uncertainty-Aware Learning Value (kai_config.py).

Covers:
  1. compute_learning_value scoring logic
  2. Uncertain failures = highest value (frontier of growth)
  3. Overconfident failures = high value (calibration error)
  4. Certain successes = low value (already competent)
  5. Rethink bonus
  6. Boundary values (conviction 0, 5, 10)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))

from kai_config import compute_learning_value


class TestSELAUR(unittest.TestCase):
    def test_uncertain_failure_highest_value(self):
        """Conviction ~5.0 (max uncertainty) + failure → highest learning value."""
        value = compute_learning_value(conviction=5.0, outcome_score=0.2)
        self.assertGreater(value, 0.8)

    def test_overconfident_failure_high_value(self):
        """Conviction 9.0 + failure → calibration error → high value."""
        value = compute_learning_value(conviction=9.0, outcome_score=0.2)
        self.assertGreater(value, 0.7)

    def test_certain_success_low_value(self):
        """Conviction 9.0 + success → already competent → low value."""
        value = compute_learning_value(conviction=9.0, outcome_score=0.9)
        self.assertLess(value, 0.4)

    def test_uncertain_success_moderate_value(self):
        """Conviction 5.0 + success → frontier validated → moderate."""
        value = compute_learning_value(conviction=5.0, outcome_score=0.8)
        self.assertGreater(value, 0.5)
        self.assertLess(value, 0.9)

    def test_rethink_bonus(self):
        """More rethinks → higher learning value (harder task)."""
        base = compute_learning_value(conviction=6.0, outcome_score=0.3, rethink_count=0)
        with_rethinks = compute_learning_value(conviction=6.0, outcome_score=0.3, rethink_count=3)
        self.assertGreater(with_rethinks, base)

    def test_rethink_bonus_capped(self):
        """Rethink bonus is capped at 0.2 (2 rethinks worth)."""
        r2 = compute_learning_value(conviction=5.0, outcome_score=0.2, rethink_count=2)
        r5 = compute_learning_value(conviction=5.0, outcome_score=0.2, rethink_count=5)
        self.assertEqual(r2, r5)  # both hit the cap

    def test_value_capped_at_one(self):
        """Learning value never exceeds 1.0."""
        value = compute_learning_value(conviction=5.0, outcome_score=0.1, rethink_count=10)
        self.assertLessEqual(value, 1.0)

    def test_value_never_negative(self):
        """Learning value is always >= 0."""
        for c in [0, 2.5, 5, 7.5, 10]:
            for o in [0.0, 0.3, 0.5, 0.7, 1.0]:
                value = compute_learning_value(conviction=c, outcome_score=o)
                self.assertGreaterEqual(value, 0.0, f"Negative for conviction={c}, outcome={o}")

    def test_conviction_zero_failure(self):
        """Conviction 0 + failure → certain failure (not uncertain)."""
        value = compute_learning_value(conviction=0.0, outcome_score=0.2)
        # Low uncertainty (conviction far from 5), so should be moderate
        self.assertLess(value, 0.8)

    def test_conviction_ten_success(self):
        """Conviction 10 + success → certain success → lowest bucket."""
        value = compute_learning_value(conviction=10.0, outcome_score=1.0)
        self.assertLess(value, 0.3)

    def test_returns_float(self):
        value = compute_learning_value(conviction=7.0, outcome_score=0.5)
        self.assertIsInstance(value, float)

    def test_ordering_uncertain_failure_beats_certain_success(self):
        """Uncertain failure always more valuable than certain success."""
        uncertain_fail = compute_learning_value(5.0, 0.2)
        certain_success = compute_learning_value(9.0, 0.9)
        self.assertGreater(uncertain_fail, certain_success)

    def test_overconfident_beats_underconfident_failure(self):
        """Overconfident failure (conv=9, fail) beats underconfident (conv=1, fail)."""
        overconfident = compute_learning_value(9.0, 0.2)
        underconfident = compute_learning_value(1.0, 0.2)
        self.assertGreater(overconfident, underconfident)


if __name__ == "__main__":
    unittest.main()
