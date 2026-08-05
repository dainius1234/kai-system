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
os.environ.setdefault("HMAC_ALLOW_DEV_SECRET", "true")

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
mod._mode_override_until[0] = time.time() + 3600 * 4
mod.policy.allowed_tools.add("executor")

client = TestClient(mod.app)


def _make_request(conviction: float, idem_key: str | None = None, nonce: str | None = None):
    """Build a gate request payload with HMAC signature."""
    now = time.time()
    n = nonce or f"n{now}"
    payload = {
        "tool": "executor",
        "actor_did": "agentic",
        "session_id": AUTH_TOKEN,
        "conviction": conviction,
        "nonce": n,
        "ts": now,
        "signature": sign_gate_request(
            actor_did="agentic",
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
        payload1 = _make_request(9.5, idem_key=idem, nonce="nonce-a1")
        resp1 = client.post("/gate/request", json=payload1, headers=AUTH_HEADER)
        self.assertEqual(resp1.status_code, 200)
        decision1 = resp1.json()

        # second request with same key but different nonce
        payload2 = _make_request(9.5, idem_key=idem, nonce="nonce-a2")
        resp2 = client.post("/gate/request", json=payload2, headers=AUTH_HEADER)
        self.assertEqual(resp2.status_code, 200)
        decision2 = resp2.json()

        # decisions should be identical (cached)
        self.assertEqual(decision1["approved"], decision2["approved"])
        self.assertEqual(decision1["evaluated_at"], decision2["evaluated_at"])

    def test_different_keys_independent(self):
        """Different idempotency keys produce independent evaluations."""
        payload1 = _make_request(9.5, idem_key="key-A", nonce="nonce-b1")
        resp1 = client.post("/gate/request", json=payload1, headers=AUTH_HEADER)
        self.assertEqual(resp1.status_code, 200)

        payload2 = _make_request(2.0, idem_key="key-B", nonce="nonce-b2")
        resp2 = client.post("/gate/request", json=payload2, headers=AUTH_HEADER)
        self.assertEqual(resp2.status_code, 200)

        # both evaluated independently
        d1, d2 = resp1.json(), resp2.json()
        self.assertNotEqual(d1["evaluated_at"], d2["evaluated_at"])

    def test_no_key_no_caching(self):
        """Requests without idempotency_key are not cached."""
        payload = _make_request(9.5, nonce="nonce-c1")
        resp = client.post("/gate/request", json=payload, headers=AUTH_HEADER)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(mod._idempotency_cache), 0)

    def test_stale_cache_entry_evicted(self):
        """Expired cache entries should be pruned."""
        idem = "idem-stale"
        payload = _make_request(9.5, idem_key=idem, nonce="nonce-d1")
        resp = client.post("/gate/request", json=payload, headers=AUTH_HEADER)
        self.assertEqual(resp.status_code, 200)
        self.assertIn(idem, mod._idempotency_cache)

        # force-expire the entry in whichever store(s) are actually backing it
        # (poking the in-memory dict alone is not enough once Redis is configured —
        # _idem_get prefers Redis, so the entry must be evicted from both)
        mod._idem_evict(idem)

        # next request with same key should NOT get cached version
        payload2 = _make_request(9.5, idem_key=idem, nonce="nonce-d2")
        resp2 = client.post("/gate/request", json=payload2, headers=AUTH_HEADER)
        self.assertEqual(resp2.status_code, 200)
        # a new evaluation should have been made
        self.assertNotEqual(
            resp.json()["evaluated_at"],
            resp2.json()["evaluated_at"],
        )


if __name__ == "__main__":
    unittest.main()


class TestParkingIsIdempotent(unittest.TestCase):
    """H-6: a retry must not park a second co-sign entry.

    The gate executes nothing — it decides and parks. So a duplicate
    *decision* is cheap, and a duplicate *park* is not: two entries for
    one intent means the operator can confirm the same destructive
    action twice, and neither confirmation looks wrong from the inside.

    Both park sites used to write `_pending_cosign[entry.request_id]`
    directly, so a retry landed a second entry under a new request_id.
    """

    def setUp(self):
        mod._idempotency_cache.clear()
        mod._pending_cosign.clear()
        mod._pending_by_idem.clear()
        mod.SEEN_NONCES.clear()
        mod._idem_mark(True)

    def _park(self, idem, nonce):
        """Force the co-sign path with a conviction below the threshold."""
        payload = _make_request(0.5, idem_key=idem, nonce=nonce)
        return client.post("/gate/request", json=payload, headers=AUTH_HEADER)

    def test_a_retry_reuses_the_existing_park(self):
        self._park("park-idem-1", "nonce-p1")
        first = dict(mod._pending_cosign)
        assert len(first) == 1, first

        # Same idempotency key, fresh nonce — a genuine retry. The
        # decision cache is cleared so the request is re-evaluated, which
        # is what a retry landing on another replica would do.
        mod._idempotency_cache.clear()
        self._park("park-idem-1", "nonce-p2")
        assert len(mod._pending_cosign) == 1, mod._pending_cosign
        assert set(mod._pending_cosign) == set(first), (mod._pending_cosign, first)

    def test_distinct_intents_still_park_separately(self):
        """The fix must not collapse two genuinely different requests."""
        self._park("park-idem-a", "nonce-pa")
        mod._idempotency_cache.clear()
        self._park("park-idem-b", "nonce-pb")
        assert len(mod._pending_cosign) == 2, mod._pending_cosign

    def test_a_request_without_a_key_is_unaffected(self):
        """No idempotency key means no claim of sameness — park each."""
        self._park(None, "nonce-n1")
        self._park(None, "nonce-n2")
        assert len(mod._pending_cosign) == 2, mod._pending_cosign

    def test_confirming_drops_the_reverse_index(self):
        """Otherwise a later retry resolves to an already-confirmed park —
        the double-approval arriving by the back door."""
        self._park("park-idem-c", "nonce-pc")
        rid = next(iter(mod._pending_cosign))
        resp = client.post("/gate/cosign",
                           json={"request_id": rid, "approved": True, "reason": "ok"},
                           headers=AUTH_HEADER)
        assert resp.status_code == 200, resp.text
        assert "park-idem-c" not in mod._pending_by_idem, mod._pending_by_idem


class TestIrreversibleNeedsSharedIdempotency(unittest.TestCase):
    """H-6: scoped by blast radius, not applied to everything.

    A reversible tool has no park to duplicate, so a Redis blip costs it
    nothing and it keeps working. An irreversible one is refused, but
    only after the grace window — a 200ms blip must not reject a call
    that would have succeeded.
    """

    def setUp(self):
        mod._idempotency_cache.clear()
        mod._pending_cosign.clear()
        mod._pending_by_idem.clear()
        mod.SEEN_NONCES.clear()
        mod._idem_mark(True)

    def tearDown(self):
        mod._idem_mark(True)

    def _shell_request(self, nonce):
        now = time.time()
        return {
            "tool": "shell", "actor_did": "agentic", "session_id": AUTH_TOKEN,
            "conviction": 9.9, "nonce": nonce, "ts": now,
            "signature": sign_gate_request(
                actor_did="agentic", session_id=AUTH_TOKEN, tool="shell",
                nonce=nonce, ts=now),
        }

    def test_healthy_store_permits_an_irreversible_request(self):
        mod.TOKEN_SCOPES[AUTH_TOKEN] = {"executor", "shell"}
        resp = client.post("/gate/request", json=self._shell_request("nonce-s0"),
                           headers=AUTH_HEADER)
        assert resp.status_code == 200, resp.text

    def test_a_brief_blip_does_not_refuse(self):
        """Inside the grace window the call proceeds."""
        mod.TOKEN_SCOPES[AUTH_TOKEN] = {"executor", "shell"}
        mod._idem_mark(False)  # failed just now
        resp = client.post("/gate/request", json=self._shell_request("nonce-s1"),
                           headers=AUTH_HEADER)
        assert resp.status_code == 200, resp.text

    def test_a_sustained_outage_refuses_the_irreversible_action(self):
        mod.TOKEN_SCOPES[AUTH_TOKEN] = {"executor", "shell"}
        mod._idem_mark(False)
        mod._idem_shared_failed_at = time.time() - (mod._IDEM_GRACE_SECONDS + 1)
        resp = client.post("/gate/request", json=self._shell_request("nonce-s2"),
                           headers=AUTH_HEADER)
        assert resp.status_code == 503, resp.text
        assert resp.json()["detail"]["reason_code"] == "IDEMPOTENCY_STORE_UNAVAILABLE"

    def test_a_reversible_tool_is_unaffected_by_the_same_outage(self):
        """The whole point of scoping this by blast radius."""
        mod._idem_mark(False)
        mod._idem_shared_failed_at = time.time() - (mod._IDEM_GRACE_SECONDS + 1)
        resp = client.post("/gate/request",
                           json=_make_request(9.5, nonce="nonce-r1"),
                           headers=AUTH_HEADER)
        assert resp.status_code == 200, resp.text

    def test_recovery_lifts_the_refusal(self):
        mod.TOKEN_SCOPES[AUTH_TOKEN] = {"executor", "shell"}
        mod._idem_mark(False)
        mod._idem_shared_failed_at = time.time() - (mod._IDEM_GRACE_SECONDS + 1)
        assert mod._shared_idem_unavailable()
        mod._idem_mark(True)
        assert not mod._shared_idem_unavailable()
        resp = client.post("/gate/request", json=self._shell_request("nonce-s3"),
                           headers=AUTH_HEADER)
        assert resp.status_code == 200, resp.text
