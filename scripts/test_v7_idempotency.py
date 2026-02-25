"""v7 tool-gate idempotency tests.

Exercises:
  - Duplicate requests with same idempotency_key return cached response
  - Different keys return independent results
  - Stale cache entries are pruned
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.auth import sign_gate_request

# temp ledger dir
_TMPDIR = tempfile.mkdtemp(prefix="idempotency-test-")
os.environ["LEDGER_PATH"] = str(Path(_TMPDIR) / "ledger.jsonl")

module_path = ROOT / "tool-gate" / "app.py"
spec = importlib.util.spec_from_file_location("tool_gate_idem", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from fastapi.testclient import TestClient

AUTH_TOKEN = "idem-test-token"
AUTH_HEADER = {"Authorization": f"Bearer {AUTH_TOKEN}"}

mod.TRUSTED_TOKENS = {AUTH_TOKEN}
mod.TOKEN_SCOPES = {AUTH_TOKEN: {"executor"}}
mod.SEEN_NONCES.clear()
mod.ledger = mod.PersistentLedger(Path(_TMPDIR) / "test-ledger.jsonl")
mod.policy = mod.GatePolicy()
mod.policy.mode = "WORK"
mod.policy.allowed_tools.add("executor")

client = TestClient(mod.app)


def _make_request(confidence: float, idem_key: str | None = None, nonce: str | None = None):
    """Build a gate request payload with HMAC signature."""
    now = time.time()
    n = nonce or f"n{now}"
    payload = {
        "tool": "executor",
        "actor_did": "langgraph",
        "session_id": AUTH_TOKEN,
        "confidence": confidence,
        "nonce": n,
        "ts": now,
        "signature": sign_gate_request(
            actor_did="langgraph",
            session_id=AUTH_TOKEN,
            tool="executor",
            nonce=n,
            ts=now,
        ),
    }
    if idem_key:
        payload["idempotency_key"] = idem_key
    return payload


class TestIdempotency(unittest.TestCase):
    """Test tool-gate idempotency_key caching."""

    def setUp(self):
        mod._idempotency_cache.clear()
        mod.SEEN_NONCES.clear()

    def test_same_key_returns_cached(self):
        """Two requests with the same idempotency_key should return same decision."""
        idem = "idem-001"
        payload1 = _make_request(0.95, idem_key=idem, nonce="nonce-a1")
        resp1 = client.post("/gate/request", json=payload1, headers=AUTH_HEADER)
        self.assertEqual(resp1.status_code, 200)
        decision1 = resp1.json()

        # second request with same key but different nonce
        payload2 = _make_request(0.95, idem_key=idem, nonce="nonce-a2")
        resp2 = client.post("/gate/request", json=payload2, headers=AUTH_HEADER)
        self.assertEqual(resp2.status_code, 200)
        decision2 = resp2.json()

        # decisions should be identical (cached)
        self.assertEqual(decision1["approved"], decision2["approved"])
        self.assertEqual(decision1["evaluated_at"], decision2["evaluated_at"])

    def test_different_keys_independent(self):
        """Different idempotency keys produce independent evaluations."""
        payload1 = _make_request(0.95, idem_key="key-A", nonce="nonce-b1")
        resp1 = client.post("/gate/request", json=payload1, headers=AUTH_HEADER)
        self.assertEqual(resp1.status_code, 200)

        payload2 = _make_request(0.2, idem_key="key-B", nonce="nonce-b2")
        resp2 = client.post("/gate/request", json=payload2, headers=AUTH_HEADER)
        self.assertEqual(resp2.status_code, 200)

        # both evaluated independently
        d1, d2 = resp1.json(), resp2.json()
        self.assertNotEqual(d1["evaluated_at"], d2["evaluated_at"])

    def test_no_key_no_caching(self):
        """Requests without idempotency_key are not cached."""
        payload = _make_request(0.95, nonce="nonce-c1")
        resp = client.post("/gate/request", json=payload, headers=AUTH_HEADER)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(mod._idempotency_cache), 0)

    def test_stale_cache_entry_evicted(self):
        """Expired cache entries should be pruned."""
        idem = "idem-stale"
        payload = _make_request(0.95, idem_key=idem, nonce="nonce-d1")
        resp = client.post("/gate/request", json=payload, headers=AUTH_HEADER)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(idem, mod._idempotency_cache)

        # manually expire the entry
        decision, _ = mod._idempotency_cache[idem]
        mod._idempotency_cache[idem] = (decision, time.time() - 1)

        # next request with same key should NOT get cached version
        payload2 = _make_request(0.95, idem_key=idem, nonce="nonce-d2")
        resp2 = client.post("/gate/request", json=payload2, headers=AUTH_HEADER)
        self.assertEqual(resp2.status_code, 200)
        # a new evaluation should have been made
        self.assertNotEqual(
            resp.json()["evaluated_at"],
            resp2.json()["evaluated_at"],
        )


if __name__ == "__main__":
    unittest.main()
