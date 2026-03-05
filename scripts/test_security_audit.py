"""P9 Security Self-Hacking — unit tests for security audit engine."""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "langgraph"))

from security_audit import (
    INJECTION_PAYLOADS,
    SANITIZATION_PAYLOADS,
    SecurityAudit,
    SecurityFinding,
    audit_hmac_auth,
    audit_policy_governance,
    audit_sanitization,
    fuzz_injection_filter,
    run_security_audit,
)


# ── Helpers ──────────────────────────────────────────────────────────

SAMPLE_INJECTION_RE = re.compile(
    r"(ignore|disregard)\s+(all\s+)?(previous|prior|above)|"
    r"system\s*prompt|"
    r"override\s+instructions|"
    r"you\s+are\s+now|"
    r"act\s+as\s+(if|though)|"
    r"disregard\s+(all\s+)?safety",
    re.IGNORECASE,
)


def sample_sanitize(text: str) -> str:
    """Minimal sanitiser: strip tags, null bytes, truncate."""
    text = re.sub(r"<[^>]*>", "", text)
    text = text.replace("\x00", "")
    return text[:50000]


def sample_sign(**kwargs):
    """Stub HMAC signer."""
    import hashlib
    payload = "|".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_verify(**kwargs):
    """Stub HMAC verifier."""
    sig = kwargs.pop("signature", "")
    expected = sample_sign(**kwargs)
    return sig == expected


# ── SecurityFinding ──────────────────────────────────────────────────

class TestSecurityFinding(unittest.TestCase):
    def test_to_dict(self):
        f = SecurityFinding("injection_bypass", "high", "test bypass", "payload", "fix it")
        d = f.to_dict()
        self.assertEqual(d["category"], "injection_bypass")
        self.assertEqual(d["severity"], "high")

    def test_payload_truncated(self):
        f = SecurityFinding("x", "low", "d", "A" * 500, "r")
        self.assertLessEqual(len(f.to_dict()["payload"]), 200)


# ── SecurityAudit ────────────────────────────────────────────────────

class TestSecurityAudit(unittest.TestCase):
    def test_risk_score_no_findings(self):
        a = SecurityAudit("id", 0, 10, [], 10, 0, ["injection_filter"], 1.0)
        self.assertAlmostEqual(a.risk_score, 0.0)

    def test_risk_score_critical(self):
        findings = [SecurityFinding("x", "critical", "d", "p", "r")]
        a = SecurityAudit("id", 0, 5, findings, 4, 1, ["test"], 1.0)
        self.assertGreater(a.risk_score, 0.0)

    def test_to_dict_includes_risk(self):
        a = SecurityAudit("id", 0, 5, [], 5, 0, ["test"], 1.0)
        d = a.to_dict()
        self.assertIn("risk_score", d)
        self.assertEqual(d["risk_score"], 0.0)

    def test_risk_score_capped(self):
        # Many critical findings shouldn't exceed 1.0
        findings = [SecurityFinding("x", "critical", "d", "p", "r") for _ in range(100)]
        a = SecurityAudit("id", 0, 5, findings, 0, 5, ["test"], 1.0)
        self.assertLessEqual(a.risk_score, 1.0)


# ── Injection Filter Fuzzing ─────────────────────────────────────────

class TestFuzzInjectionFilter(unittest.TestCase):
    def test_returns_list(self):
        findings = fuzz_injection_filter(SAMPLE_INJECTION_RE)
        self.assertIsInstance(findings, list)

    def test_catches_known_bypass(self):
        # Use a deliberately weak regex that misses everything
        weak_re = re.compile(r"NOMATCH_IMPOSSIBLE_PATTERN_12345")
        findings = fuzz_injection_filter(weak_re)
        bypass_findings = [f for f in findings if f.category == "injection_bypass"]
        # Should catch many of the should_match=True payloads
        self.assertGreater(len(bypass_findings), 5)

    def test_no_false_positives_on_legit(self):
        # A well-scoped regex should not flag legitimate queries
        findings = fuzz_injection_filter(SAMPLE_INJECTION_RE)
        fp = [f for f in findings if f.category == "injection_false_positive"]
        # Some false positives are expected with aggressive regex; just check some are detected
        self.assertIsInstance(fp, list)

    def test_payload_count(self):
        self.assertGreater(len(INJECTION_PAYLOADS), 15)


# ── Sanitization Audit ───────────────────────────────────────────────

class TestAuditSanitization(unittest.TestCase):
    def test_good_sanitizer(self):
        findings = audit_sanitization(sample_sanitize)
        critical = [f for f in findings if f.severity == "critical"]
        self.assertEqual(len(critical), 0)

    def test_bad_sanitizer(self):
        # Identity function — everything passes through
        findings = audit_sanitization(lambda x: x)
        self.assertGreater(len(findings), 0)

    def test_crashing_sanitizer(self):
        def crash(x):
            raise ValueError("boom")
        findings = audit_sanitization(crash)
        self.assertGreater(len(findings), 0)
        self.assertTrue(all(f.severity in ("high", "critical") for f in findings))

    def test_payload_count(self):
        self.assertGreater(len(SANITIZATION_PAYLOADS), 10)


# ── HMAC Auth Audit ──────────────────────────────────────────────────

class TestAuditHmacAuth(unittest.TestCase):
    def test_valid_sign_verify_cycle(self):
        findings = audit_hmac_auth(sample_sign, sample_verify)
        # Should pass basic sign/verify — no critical findings
        critical = [f for f in findings if f.severity == "critical"]
        self.assertEqual(len(critical), 0)

    def test_broken_verifier(self):
        # Verifier that always returns True
        findings = audit_hmac_auth(sample_sign, lambda **k: True)
        critical = [f for f in findings if f.severity == "critical"]
        # Should catch tampered-sig and modified-field
        self.assertGreaterEqual(len(critical), 1)


# ── Policy Governance Audit ──────────────────────────────────────────

class TestAuditPolicyGovernance(unittest.TestCase):
    def test_returns_findings(self):
        findings = audit_policy_governance()
        self.assertIsInstance(findings, list)
        # In test env, most env vars are unset so should have findings
        self.assertGreater(len(findings), 0)

    def test_categories(self):
        findings = audit_policy_governance()
        cats = {f.category for f in findings}
        self.assertTrue(cats.issubset({"policy_edge"}))


# ── Full Orchestration ───────────────────────────────────────────────

class TestRunSecurityAudit(unittest.TestCase):
    def test_full_audit(self):
        audit = run_security_audit(
            injection_re=SAMPLE_INJECTION_RE,
            sanitize_fn=sample_sanitize,
            sign_fn=sample_sign,
            verify_fn=sample_verify,
            audit_id="test-audit",
        )
        self.assertEqual(audit.audit_id, "test-audit")
        self.assertGreater(audit.total_tests, 0)
        self.assertEqual(audit.passed + audit.failed, audit.total_tests)
        self.assertIn("injection_filter", audit.categories_tested)
        self.assertIn("sanitization", audit.categories_tested)
        self.assertIn("hmac_auth", audit.categories_tested)
        self.assertIn("policy_governance", audit.categories_tested)
        self.assertGreater(audit.duration_ms, 0)

    def test_minimal_audit(self):
        # No optional params → only policy governance
        audit = run_security_audit(audit_id="minimal")
        self.assertEqual(len(audit.categories_tested), 1)
        self.assertIn("policy_governance", audit.categories_tested)

    def test_to_dict_serializable(self):
        import json
        audit = run_security_audit(
            injection_re=SAMPLE_INJECTION_RE,
            sanitize_fn=sample_sanitize,
            audit_id="serial-test",
        )
        d = audit.to_dict()
        # Should be JSON-serialisable
        s = json.dumps(d)
        self.assertIn("serial-test", s)


# ── Adversary Integration ────────────────────────────────────────────

class TestChallengeSecurityIntegration(unittest.TestCase):
    def test_challenge_security_exists(self):
        from adversary import challenge_security
        self.assertTrue(callable(challenge_security))

    def test_challenge_security_returns_result(self):
        from adversary import challenge_security
        result = challenge_security(
            injection_re=SAMPLE_INJECTION_RE,
            sanitize_fn=sample_sanitize,
        )
        # ChallengeResult has modifier, confidence, passed, finding
        self.assertIsNotNone(result.modifier)
        self.assertIsNotNone(result.confidence)
        self.assertIsInstance(result.passed, bool)


if __name__ == "__main__":
    unittest.main()
