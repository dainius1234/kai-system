"""Tests for P1: Failure Taxonomy + Metacognitive Rules (kai_config.py).

Covers:
  1. FailureClass enum values
  2. classify_failure — all 8 failure classes + success passthrough
  3. extract_metacognitive_rule — rule generation per class
  4. Edge cases (empty episodes, missing fields)
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))

from kai_config import (
    FailureClass,
    classify_failure,
    extract_metacognitive_rule,
)


class TestFailureClassEnum(unittest.TestCase):
    def test_all_classes_exist(self):
        expected = {
            "data_insufficient", "policy_blocked", "confidence_low",
            "operator_overridden", "service_unavailable",
            "contradicted_by_evidence", "time_expired", "scope_exceeded",
            "unknown",
        }
        actual = {fc.value for fc in FailureClass}
        self.assertEqual(actual, expected)

    def test_string_enum(self):
        self.assertEqual(str(FailureClass.DATA_INSUFFICIENT), "FailureClass.DATA_INSUFFICIENT")
        self.assertEqual(FailureClass.POLICY_BLOCKED.value, "policy_blocked")


class TestClassifyFailure(unittest.TestCase):
    def test_success_returns_unknown(self):
        ep = {"outcome_score": 0.8, "final_conviction": 9.0}
        self.assertEqual(classify_failure(ep), FailureClass.UNKNOWN)

    def test_policy_blocked(self):
        ep = {"outcome_score": 0.2}
        gate = {"approved": False, "status": "blocked", "reason": "policy violation"}
        self.assertEqual(classify_failure(ep, gate), FailureClass.POLICY_BLOCKED)

    def test_service_unavailable_from_gate(self):
        ep = {"outcome_score": 0.3}
        gate = {"status": "unavailable", "reason": "tool-gate unavailable"}
        self.assertEqual(classify_failure(ep, gate), FailureClass.SERVICE_UNAVAILABLE)

    def test_circuit_open_is_service_unavailable(self):
        ep = {"outcome_score": 0.1}
        gate = {"approved": False, "status": "blocked", "reason": "tool-gate circuit open"}
        self.assertEqual(classify_failure(ep, gate), FailureClass.SERVICE_UNAVAILABLE)

    def test_confidence_low(self):
        ep = {"outcome_score": 0.3, "final_conviction": 5.0, "rethink_count": 3}
        self.assertEqual(classify_failure(ep), FailureClass.CONFIDENCE_LOW)

    def test_contradicted_by_evidence(self):
        ep = {"outcome_score": 0.2, "verifier_verdict": "FAIL_CLOSED"}
        self.assertEqual(classify_failure(ep), FailureClass.CONTRADICTED_BY_EVIDENCE)

    def test_contradicted_repair(self):
        ep = {"outcome_score": 0.2, "verifier_verdict": "REPAIR"}
        self.assertEqual(classify_failure(ep), FailureClass.CONTRADICTED_BY_EVIDENCE)

    def test_data_insufficient(self):
        ep = {"outcome_score": 0.2, "offline_context_used": 0}
        self.assertEqual(classify_failure(ep), FailureClass.DATA_INSUFFICIENT)

    def test_operator_overridden(self):
        ep = {"outcome_score": 0.3, "conviction_override": "operator override matched"}
        self.assertEqual(classify_failure(ep), FailureClass.OPERATOR_OVERRIDDEN)

    def test_scope_exceeded(self):
        ep = {"outcome_score": 0.2, "scope_exceeded": True}
        self.assertEqual(classify_failure(ep), FailureClass.SCOPE_EXCEEDED)

    def test_time_expired(self):
        ep = {"outcome_score": 0.1, "time_expired": True}
        self.assertEqual(classify_failure(ep), FailureClass.TIME_EXPIRED)

    def test_low_conviction_no_gate_fallback(self):
        """Low outcome + moderate conviction → confidence_low fallback."""
        ep = {"outcome_score": 0.3, "final_conviction": 6.5}
        self.assertEqual(classify_failure(ep), FailureClass.CONFIDENCE_LOW)

    def test_gate_not_approved_is_policy(self):
        ep = {"outcome_score": 0.2}
        gate = {"approved": False, "status": "denied", "reason": "not allowed"}
        self.assertEqual(classify_failure(ep, gate), FailureClass.POLICY_BLOCKED)

    def test_empty_episode(self):
        """Missing fields → defaults → UNKNOWN (outcome defaults to 0.5)."""
        self.assertEqual(classify_failure({}), FailureClass.UNKNOWN)


class TestExtractMetacognitiveRule(unittest.TestCase):
    def test_unknown_returns_none(self):
        self.assertIsNone(extract_metacognitive_rule({}, FailureClass.UNKNOWN))

    def test_data_insufficient_rule(self):
        ep = {"input": "check the crypto prices for Bitcoin"}
        rule = extract_metacognitive_rule(ep, FailureClass.DATA_INSUFFICIENT)
        self.assertIn("always check memu-core", rule)
        self.assertIn("zero context chunks", rule)

    def test_policy_blocked_rule(self):
        ep = {"input": "execute trade on binance"}
        rule = extract_metacognitive_rule(ep, FailureClass.POLICY_BLOCKED)
        self.assertIn("verify tool-gate policy", rule)

    def test_confidence_low_rule(self):
        ep = {"input": "analyze quantum computing trends", "rethink_count": 2}
        rule = extract_metacognitive_rule(ep, FailureClass.CONFIDENCE_LOW)
        self.assertIn("gather more evidence", rule)
        self.assertIn("2 rethinks", rule)

    def test_contradicted_rule(self):
        ep = {"input": "the GDP of Lithuania is 100 billion"}
        rule = extract_metacognitive_rule(ep, FailureClass.CONTRADICTED_BY_EVIDENCE)
        self.assertIn("cross-check with verifier", rule)

    def test_service_unavailable_rule(self):
        ep = {"input": "fetch latest stock prices"}
        rule = extract_metacognitive_rule(ep, FailureClass.SERVICE_UNAVAILABLE)
        self.assertIn("pre-check service health", rule)

    def test_operator_overridden_rule(self):
        ep = {"input": "send email to client"}
        rule = extract_metacognitive_rule(ep, FailureClass.OPERATOR_OVERRIDDEN)
        self.assertIn("operator overrode", rule)

    def test_time_expired_rule(self):
        ep = {"input": "process entire dataset"}
        rule = extract_metacognitive_rule(ep, FailureClass.TIME_EXPIRED)
        self.assertIn("tighter time bounds", rule)

    def test_scope_exceeded_rule(self):
        ep = {"input": "rewrite the entire codebase"}
        rule = extract_metacognitive_rule(ep, FailureClass.SCOPE_EXCEEDED)
        self.assertIn("decompose into smaller scope", rule)

    def test_rule_contains_topic_words(self):
        ep = {"input": "calculate quarterly invoice totals for client"}
        rule = extract_metacognitive_rule(ep, FailureClass.DATA_INSUFFICIENT)
        # Should contain at least some topic words from the input
        self.assertIn("topic=", rule)

    def test_short_input_still_works(self):
        ep = {"input": "hi"}
        rule = extract_metacognitive_rule(ep, FailureClass.DATA_INSUFFICIENT)
        self.assertIsNotNone(rule)
        self.assertIn("this topic", rule)  # fallback when no 4+ char words


if __name__ == "__main__":
    unittest.main()
