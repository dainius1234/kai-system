"""v7 verifier tests — verdict logic, material claim extraction, aggregation.

Exercises:
  - PASS / REPAIR / FAIL_CLOSED verdict paths
  - Material claim extraction (numbers, dates, identifiers, safety, etc.)
  - _aggregate logic with strong chunk requirements
  - /verify endpoint via TestClient
  - /metrics verdict counter tracking
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# import verifier module
module_path = ROOT / "verifier" / "app.py"
spec = importlib.util.spec_from_file_location("verifier_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from fastapi.testclient import TestClient

client = TestClient(mod.app)


class TestMaterialClaimExtraction(unittest.TestCase):
    """Test extract_material_claims() against real construction domain text."""

    def test_extracts_numbers(self):
        text = "The slab is 150mm thick and weighs 2.5 tonnes"
        claims = mod.extract_material_claims(text)
        types = {c.claim_type for c in claims}
        self.assertIn("numbers", types)

    def test_extracts_dates(self):
        text = "Inspection date 15/03/2025, completed week 12"
        claims = mod.extract_material_claims(text)
        types = {c.claim_type for c in claims}
        self.assertIn("dates", types)

    def test_extracts_identifiers(self):
        text = "Refer to DWG-1234 and NCR-001 for details"
        claims = mod.extract_material_claims(text)
        types = {c.claim_type for c in claims}
        self.assertIn("identifiers", types)

    def test_extracts_safety(self):
        text = "RAMS submitted. PPE required in exclusion zone. COSHH assessment pending."
        claims = mod.extract_material_claims(text)
        types = {c.claim_type for c in claims}
        self.assertIn("safety", types)

    def test_no_duplicates(self):
        text = "DWG-1234 and DWG-1234 mentioned twice"
        claims = mod.extract_material_claims(text)
        raw_texts = [c.raw_text for c in claims]
        self.assertEqual(len(raw_texts), len(set(raw_texts)))

    def test_empty_text(self):
        claims = mod.extract_material_claims("")
        self.assertEqual(claims, [])

    def test_confidence_levels(self):
        text = "Grid A1 is at chainage 250m. RAMS required."
        claims = mod.extract_material_claims(text)
        for c in claims:
            self.assertGreaterEqual(c.confidence, 0.0)
            self.assertLessEqual(c.confidence, 1.0)


class TestSelfConsistency(unittest.TestCase):
    """Test _self_consistency_check()."""

    def test_no_plan(self):
        signal = mod._self_consistency_check(None)
        self.assertEqual(signal.strategy, "self_consistency")
        self.assertEqual(signal.score, 0.5)

    def test_valid_plan(self):
        plan = {
            "summary": "Install drainage",
            "steps": [
                {"action": "excavate trench"},
                {"action": "lay pipe"},
                {"action": "backfill"},
            ],
        }
        signal = mod._self_consistency_check(plan)
        self.assertEqual(signal.score, 1.0)

    def test_plan_missing_actions(self):
        plan = {
            "summary": "Some work",
            "steps": [{"action": "dig"}, {"tool": "spade"}],
        }
        signal = mod._self_consistency_check(plan)
        self.assertLess(signal.score, 1.0)

    def test_plan_no_summary(self):
        plan = {"steps": [{"action": "dig"}]}
        signal = mod._self_consistency_check(plan)
        self.assertLess(signal.score, 1.0)


class TestKeywordPlausibility(unittest.TestCase):
    """Test _keyword_plausibility()."""

    def test_hedged_language(self):
        signal = mod._keyword_plausibility("This suggests possible subsidence", None)
        self.assertGreaterEqual(signal.score, 0.6)

    def test_absolute_language(self):
        signal = mod._keyword_plausibility("This will always certainly work", None)
        self.assertLessEqual(signal.score, 0.5)

    def test_context_improves_score(self):
        s1 = mod._keyword_plausibility("Good foundations", None)
        s2 = mod._keyword_plausibility("Good foundations", "Based on structural survey report dated 2024")
        self.assertGreater(s2.score, s1.score)


class TestAggregate(unittest.TestCase):
    """Test _aggregate() verdict determination."""

    def _signal(self, strategy: str, score: float) -> mod.Signal:
        return mod.Signal(strategy=strategy, score=score, detail="test")

    def test_pass_with_strong_chunks(self):
        signals = [
            self._signal("mem", 0.8),
            self._signal("kw", 0.7),
            self._signal("mc", 0.9),
        ]
        verdict, conf, summary = mod._aggregate(signals, strong_chunks=3)
        self.assertEqual(verdict, "PASS")
        self.assertGreaterEqual(conf, mod.PASS_THRESHOLD)

    def test_repair_low_strong_chunks(self):
        """Score meets threshold but not enough strong chunks → REPAIR."""
        signals = [
            self._signal("mem", 0.8),
            self._signal("kw", 0.7),
            self._signal("mc", 0.9),
        ]
        verdict, conf, summary = mod._aggregate(signals, strong_chunks=0)
        self.assertEqual(verdict, "REPAIR")

    def test_repair_medium_score(self):
        """Average between REPAIR and PASS thresholds → REPAIR."""
        signals = [
            self._signal("mem", 0.4),
            self._signal("kw", 0.5),
        ]
        verdict, conf, summary = mod._aggregate(signals, strong_chunks=0)
        self.assertEqual(verdict, "REPAIR")

    def test_fail_closed_low_score(self):
        signals = [
            self._signal("mem", 0.1),
            self._signal("kw", 0.2),
        ]
        verdict, conf, summary = mod._aggregate(signals, strong_chunks=0)
        self.assertEqual(verdict, "FAIL_CLOSED")

    def test_empty_signals(self):
        verdict, conf, summary = mod._aggregate([], strong_chunks=0)
        self.assertEqual(verdict, "FAIL_CLOSED")
        self.assertEqual(conf, 0.0)

    def test_evidence_summary_present(self):
        signals = [self._signal("mem", 0.5)]
        verdict, conf, summary = mod._aggregate(signals, strong_chunks=1)
        self.assertIn("avg=", summary)
        self.assertIn("strong_chunks=", summary)


class TestMaterialClaimSignal(unittest.TestCase):
    """Test _material_claim_signal()."""

    def test_no_claims(self):
        signal = mod._material_claim_signal([], 0)
        self.assertEqual(signal.score, 0.7)

    def test_claims_fully_supported(self):
        claims = [
            mod.MaterialClaim(claim_type="numbers", raw_text="50m", confidence=0.9),
        ]
        signal = mod._material_claim_signal(claims, strong_chunks=mod.MIN_STRONG_CHUNKS)
        self.assertGreaterEqual(signal.score, 0.8)

    def test_claims_no_evidence(self):
        claims = [
            mod.MaterialClaim(claim_type="numbers", raw_text="50m", confidence=0.9),
            mod.MaterialClaim(claim_type="dates", raw_text="15/03/2025", confidence=0.9),
        ]
        signal = mod._material_claim_signal(claims, strong_chunks=0)
        self.assertLessEqual(signal.score, 0.3)


class TestVerifyEndpoint(unittest.TestCase):
    """Test /verify via HTTP."""

    def test_health(self):
        resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("policy_version", data)
        self.assertIn("policy_hash", data)

    def test_verify_simple_claim(self):
        resp = client.post("/verify", json={
            "claim": "The slab is 150mm thick",
            "source": "test",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn(data["verdict"], ("PASS", "REPAIR", "FAIL_CLOSED"))
        self.assertIn("confidence", data)
        self.assertIn("signals", data)
        self.assertIn("material_claims", data)
        self.assertIn("claim_hash", data)
        self.assertIn("strong_chunks", data)
        self.assertIn("evidence_summary", data)

    def test_verify_with_evidence_pack(self):
        """Supply pre-built evidence pack — skips memu call."""
        resp = client.post("/verify", json={
            "claim": "Drainage at grid B5 is 200mm deep",
            "source": "test",
            "evidence_pack": [
                {"content": "Drainage at grid B5 is 200mm deep per DWG-005",
                 "relevance": 0.9, "importance": 0.8},
                {"content": "Grid B5 drainage confirmed at 200mm",
                 "relevance": 0.85, "importance": 0.7},
                {"content": "B5 drainage spec 200mm as-built",
                 "relevance": 0.8, "importance": 0.75},
            ],
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # with strong matching evidence, expect PASS or at least REPAIR
        self.assertIn(data["verdict"], ("PASS", "REPAIR"))

    def test_verify_suspect_claim(self):
        """Absolute language with no evidence → low score."""
        resp = client.post("/verify", json={
            "claim": "This is always guaranteed to be 100% impossible to fail",
            "source": "test",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # keyword plausibility should drag down the score
        self.assertLessEqual(data["confidence"], 0.6)

    def test_metrics_tracks_verdicts(self):
        # reset counters
        mod._verdict_counts.clear()
        mod._verdict_counts.update({"PASS": 0, "REPAIR": 0, "FAIL_CLOSED": 0})

        client.post("/verify", json={"claim": "test claim", "source": "test"})

        resp = client.get("/metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("verdicts", data)
        total = sum(data["verdicts"].values())
        self.assertGreaterEqual(total, 1)


if __name__ == "__main__":
    unittest.main()
