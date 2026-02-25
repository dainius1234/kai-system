"""v7 memu-core quarantine + evidence-pack tests.

Exercises:
  - POST /memory/quarantine — mark record as poisoned
  - POST /memory/quarantine/clear — unmark poisoned record
  - GET /memory/quarantine/list — list poisoned records
  - GET /memory/evidence-pack — scored evidence pack for verifier
  - Quarantined records excluded from /memory/retrieve
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# env overrides BEFORE import
os.environ["MAX_MEMORY_RECORDS"] = "100"
os.environ.setdefault("VECTOR_STORE", "memory")

module_path = ROOT / "memu-core" / "app.py"
spec = importlib.util.spec_from_file_location("memu_core_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from fastapi.testclient import TestClient

client = TestClient(mod.app)


def _memorize(text: str, event_type: str = "test") -> str:
    """Helper: memorize a record and return its ID."""
    resp = client.post("/memory/memorize", json={
        "timestamp": "2025-01-01T00:00:00Z",
        "event_type": event_type,
        "result_raw": text,
        "user_id": "keeper",
    })
    assert resp.status_code == 200, f"memorize failed: {resp.text}"
    return resp.json()["id"]


class TestQuarantine(unittest.TestCase):
    """Test quarantine lifecycle: mark → list → clear."""

    def test_quarantine_and_list(self):
        rid = _memorize("Record to quarantine")

        # quarantine it
        resp = client.post("/memory/quarantine", json={
            "record_id": rid,
            "reason": "suspected confabulation",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "quarantined")

        # list quarantined
        resp = client.get("/memory/quarantine/list")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["count"], 1)
        ids = [r["id"] for r in data["quarantined"]]
        self.assertIn(rid, ids)

    def test_quarantine_clear(self):
        rid = _memorize("Record to quarantine then clear")

        # quarantine
        client.post("/memory/quarantine", json={
            "record_id": rid,
            "reason": "test",
        })

        # clear
        resp = client.post("/memory/quarantine/clear", json={
            "record_id": rid,
            "reason": "cleared after review",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "cleared")

    def test_quarantine_nonexistent_record(self):
        resp = client.post("/memory/quarantine", json={
            "record_id": "nonexistent-id-999",
            "reason": "test",
        })
        self.assertEqual(resp.status_code, 404)

    def test_quarantined_excluded_from_retrieval(self):
        """Quarantined records should not appear in retrieve results."""
        rid = _memorize("unique_quarantine_test_token_xyz123")

        # quarantine it
        client.post("/memory/quarantine", json={
            "record_id": rid,
            "reason": "bad data",
        })

        # retrieve should not include it
        resp = client.get("/memory/retrieve", params={
            "query": "unique_quarantine_test_token_xyz123",
            "user_id": "keeper",
        })
        self.assertEqual(resp.status_code, 200)
        results = resp.json()
        result_ids = [r.get("id") for r in results if isinstance(r, dict)]
        self.assertNotIn(rid, result_ids)


class TestEvidencePack(unittest.TestCase):
    """Test /memory/evidence-pack endpoint."""

    def test_evidence_pack_returns_structure(self):
        # store something first
        _memorize("Evidence pack test record about drainage at grid B5")

        resp = client.get("/memory/evidence-pack", params={
            "query": "drainage grid B5",
            "user_id": "keeper",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("query", data)
        self.assertIn("pack_size", data)
        self.assertIn("evidence", data)
        self.assertIsInstance(data["evidence"], list)

    def test_evidence_pack_fields(self):
        _memorize("Steel reinforcement 200mm at grid C3 per DWG-007")

        resp = client.get("/memory/evidence-pack", params={
            "query": "steel reinforcement",
            "user_id": "keeper",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        if data["pack_size"] > 0:
            rec = data["evidence"][0]
            # check expected fields are present
            for field in ("id", "rank_score", "trust_tier", "source_id",
                          "category", "relevance", "importance", "content"):
                self.assertIn(field, rec, f"missing field: {field}")

    def test_evidence_pack_structure_on_miss(self):
        """Query with unlikely match — verify structure, not exact count.

        The in-memory store may have residual records from other tests,
        so we only assert the response schema is correct.
        """
        resp = client.get("/memory/evidence-pack", params={
            "query": "completely_nonexistent_xzy_789",
            "user_id": "keeper",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("pack_size", data)
        self.assertIn("evidence", data)
        self.assertIsInstance(data["evidence"], list)
        self.assertEqual(data["pack_size"], len(data["evidence"]))


class TestMemoryRecordV7Fields(unittest.TestCase):
    """Test that MemoryRecord has v7 fields."""

    def test_record_has_v7_fields(self):
        rec = mod.MemoryRecord(
            id="test-1",
            timestamp="2025-01-01T00:00:00Z",
            event_type="test",
            content={"result": "hello"},
            embedding=[0.0] * 10,
            relevance=0.5,
            importance=0.5,
            access_count=0,
            rank_score=0.75,
            trust_tier="verified",
            source_id="unit-test",
            poisoned=True,
            quarantine_reason="test reason",
        )
        self.assertEqual(rec.rank_score, 0.75)
        self.assertEqual(rec.trust_tier, "verified")
        self.assertEqual(rec.source_id, "unit-test")
        self.assertTrue(rec.poisoned)
        self.assertEqual(rec.quarantine_reason, "test reason")

    def test_record_defaults(self):
        rec = mod.MemoryRecord(
            id="test-2",
            timestamp="2025-01-01T00:00:00Z",
            event_type="test",
            content={"result": "hello"},
            embedding=[0.0] * 10,
            relevance=0.5,
            importance=0.5,
            access_count=0,
        )
        self.assertIsNone(rec.rank_score)
        self.assertEqual(rec.trust_tier, "unverified")
        self.assertFalse(rec.poisoned)
        self.assertIsNone(rec.quarantine_reason)


if __name__ == "__main__":
    unittest.main()
