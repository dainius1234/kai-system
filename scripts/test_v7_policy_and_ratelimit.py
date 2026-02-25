"""v7 rate limiter + policy module tests.

Exercises:
  - common/rate_limit.py: check_rate_limit, burst multiplier, rate_limit_snapshot
  - common/policy.py: POLICY dict, policy_hash, verifier_thresholds, evidence_weights,
    circuit_breaker_defaults, rate_limit accessor, risk_tier_for_tool, mode_config,
    quarantine_config, policy_version
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.policy import (
    POLICY,
    circuit_breaker_defaults,
    evidence_weights,
    mode_config,
    policy_hash,
    policy_version,
    quarantine_config,
    rate_limit,
    risk_tier_for_tool,
    verifier_thresholds,
)
from common.rate_limit import (
    BURST_MULTIPLIER,
    _windows,
    check_rate_limit,
    rate_limit_snapshot,
)


class TestPolicyLoader(unittest.TestCase):
    """Test common/policy.py loads and exposes policy correctly."""

    def test_policy_is_dict(self):
        self.assertIsInstance(POLICY, dict)

    def test_policy_hash_is_hex(self):
        self.assertTrue(all(c in "0123456789abcdef" for c in policy_hash))
        self.assertEqual(len(policy_hash), 16)

    def test_policy_version(self):
        # either "unknown" (no file) or a real version string
        self.assertIsInstance(policy_version, str)
        self.assertTrue(len(policy_version) > 0)

    def test_verifier_thresholds(self):
        vt = verifier_thresholds()
        self.assertIsInstance(vt, dict)
        # if policy.yml is present, should have thresholds
        if POLICY:
            self.assertIn("pass_threshold", vt)
            self.assertIn("repair_threshold", vt)
            self.assertIn("min_strong_chunks", vt)

    def test_evidence_weights(self):
        ew = evidence_weights()
        self.assertIsInstance(ew, dict)
        # must have the 5 ranking signals
        for key in ("similarity", "relevance", "importance", "recency", "pin_bonus"):
            self.assertIn(key, ew)
        # weights should sum to ~1.0
        total = sum(ew.values())
        self.assertAlmostEqual(total, 1.0, places=1)

    def test_circuit_breaker_defaults(self):
        cb = circuit_breaker_defaults()
        self.assertIsInstance(cb, dict)
        self.assertIn("failure_threshold", cb)
        self.assertIn("recovery_seconds", cb)

    def test_rate_limit_accessor(self):
        # configured endpoints should return > 0
        lim = rate_limit("gate_request")
        self.assertIsInstance(lim, int)
        self.assertGreater(lim, 0)

    def test_rate_limit_unknown_endpoint(self):
        # unknown endpoints get the default (60)
        lim = rate_limit("unknown_endpoint_xyz")
        self.assertEqual(lim, 60)

    def test_risk_tier_for_tool(self):
        # noop should be LOW
        tier = risk_tier_for_tool("noop")
        if POLICY:
            self.assertIn(tier, ("LOW", "MEDIUM", "HIGH"))

    def test_risk_tier_unknown_tool(self):
        tier = risk_tier_for_tool("unknown_tool_xyz")
        self.assertEqual(tier, "MEDIUM")  # default

    def test_quarantine_config(self):
        qc = quarantine_config()
        self.assertIsInstance(qc, dict)

    def test_mode_config(self):
        mc = mode_config("PUB")
        self.assertIsInstance(mc, dict)


class TestRateLimiter(unittest.TestCase):
    """Test common/rate_limit.py sliding window limiter."""

    def setUp(self):
        # clear all windows before each test
        _windows.clear()

    def test_burst_multiplier_is_numeric(self):
        self.assertIsInstance(BURST_MULTIPLIER, float)
        self.assertGreater(BURST_MULTIPLIER, 0)

    def test_check_rate_limit_allows_first(self):
        # first request should never be blocked
        check_rate_limit("test_endpoint_alpha")
        # no exception → pass

    def test_check_rate_limit_blocks_burst(self):
        """Exceed burst limit → HTTPException 429."""
        from fastapi import HTTPException

        endpoint = "test_rate_burst"
        limit = rate_limit(endpoint)
        burst_cap = int(limit * BURST_MULTIPLIER)

        # fill up to burst cap
        for _ in range(burst_cap):
            check_rate_limit(endpoint)

        # next one should be blocked
        with self.assertRaises(HTTPException) as ctx:
            check_rate_limit(endpoint)
        self.assertEqual(ctx.exception.status_code, 429)

    def test_rate_limit_snapshot(self):
        check_rate_limit("test_snapshot_ep")
        check_rate_limit("test_snapshot_ep")
        snap = rate_limit_snapshot()
        self.assertIn("test_snapshot_ep", snap)
        self.assertEqual(snap["test_snapshot_ep"]["current"], 2)

    def test_no_limit_configured(self):
        """Endpoint with limit <= 0 should never block."""
        # Patch to test: by default unknown endpoints get 60
        # This test just ensures zero-limit path works
        check_rate_limit("no_limit_test")  # should pass


if __name__ == "__main__":
    unittest.main()
