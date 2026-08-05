"""Integration chain test — exercises the full sovereign AI pipeline within a single process.

Chain: memorize → retrieve → evidence-pack → verify → quarantine → clear

This test loads memu-core and verifier side-by-side via importlib and
exercises the real code paths (no mocks). The verifier is configured to
use the evidence-pack from memu-core directly rather than calling it
over the network.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── env overrides BEFORE loading modules ────────────────────────────
os.environ["MAX_MEMORY_RECORDS"] = "200"
os.environ.setdefault("VECTOR_STORE", "memory")


from scripts.module_stubs import stubbed  # noqa: E402


def _load_module(name: str, path: Path, extra=None):
    """Load a service by path without leaving its name claimed.

    `app` and `introspect_app` are each the name of two different files
    in this repository. Registering one permanently gives the next file
    to want that name the other service, and the resulting error names
    the innocent file. Scoped to the exec, which is the only moment the
    name is actually needed — `introspect_app.py` does
    `from app import store` at its own import time.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    with stubbed({name: mod, **(extra or {})}):
        spec.loader.exec_module(mod)
    return mod


# Evict any MagicMock stub injected by test_agentic_routes.py and load the
# real local lakefs_client so memu-core/app.py gets proper LakeFSClient/VersionCommit.
sys.modules.pop("lakefs_client", None)
_lc_spec = importlib.util.spec_from_file_location("lakefs_client", ROOT / "memu-core" / "lakefs_client.py")
_lc_mod = importlib.util.module_from_spec(_lc_spec)
sys.modules["lakefs_client"] = _lc_mod
_lc_spec.loader.exec_module(_lc_mod)

memu = _load_module("app", ROOT / "memu-core" / "app.py")
verifier = _load_module("verifier_integ", ROOT / "verifier" / "app.py")
# Quarantine endpoints moved to memu-core-introspect (see DECISIONS.md D21).
# introspect_app.py does `from app import store`, so memu is handed to it
# under that name for the duration of the exec and no longer.
introspect = _load_module(
    "introspect_app", ROOT / "memu-core" / "introspect_app.py", {"app": memu})

from fastapi.testclient import TestClient

memu_client = TestClient(memu.app)
introspect_client = TestClient(introspect.app)
verifier_client = TestClient(verifier.app)


class TestIntegrationChain(unittest.TestCase):
    """End-to-end chain: memorize → retrieve → evidence-pack → verify → quarantine."""

    def test_full_chain(self):
        # ── Step 1: Memorize some construction domain records ────────
        records = [
            "Drainage at grid B5 installed at 200mm depth per DWG-005",
            "Concrete pour at grid B5 completed 15/03/2025, 150mm slab",
            "RAMS submitted for B5 drainage works, PPE required",
        ]
        record_ids = []
        for text in records:
            resp = memu_client.post("/memory/memorize", json={
                "timestamp": "2025-03-15T10:00:00Z",
                "event_type": "site_log",
                "result_raw": text,
                "user_id": "keeper",
            })
            self.assertEqual(resp.status_code, 200)
            record_ids.append(resp.json()["id"])

        self.assertEqual(len(record_ids), 3)

        # ── Step 2: Retrieve by query ───────────────────────────────
        resp = memu_client.get("/memory/retrieve", params={
            "query": "drainage grid B5",
            "user_id": "keeper",
            "top_k": 5,
        })
        self.assertEqual(resp.status_code, 200)
        results = resp.json()
        self.assertGreater(len(results), 0)

        # ── Step 3: Get evidence pack ───────────────────────────────
        resp = memu_client.get("/memory/evidence-pack", params={
            "query": "drainage grid B5 200mm depth",
            "user_id": "keeper",
            "top_k": 10,
        })
        self.assertEqual(resp.status_code, 200)
        evidence = resp.json()
        self.assertGreater(evidence["pack_size"], 0)

        # ── Step 4: Verify a claim using the evidence pack ──────────
        resp = verifier_client.post("/verify", json={
            "claim": "Drainage at grid B5 was installed at 200mm depth",
            "source": "integration-test",
            "evidence_pack": evidence["evidence"],
        })
        self.assertEqual(resp.status_code, 200)
        verdict_data = resp.json()
        self.assertIn(verdict_data["verdict"], ("PASS", "REPAIR", "FAIL_CLOSED"))
        self.assertIn("signals", verdict_data)
        self.assertIn("material_claims", verdict_data)
        self.assertGreater(len(verdict_data["signals"]), 0)

        # ── Step 5: Quarantine a record ─────────────────────────────
        target_id = record_ids[0]
        resp = introspect_client.post("/memory/quarantine", json={
            "record_id": target_id,
            "reason": "integration test quarantine",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "quarantined")

        # verify it's in quarantine list
        resp = introspect_client.get("/memory/quarantine/list")
        self.assertEqual(resp.status_code, 200)
        q_ids = [r["id"] for r in resp.json()["quarantined"]]
        self.assertIn(target_id, q_ids)

        # ── Step 6: Clear quarantine ────────────────────────────────
        resp = introspect_client.post("/memory/quarantine/clear", json={
            "record_id": target_id,
            "reason": "cleared after review",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "cleared")

    def test_verify_unsubstantiated_claim(self):
        """A claim with no matching evidence should score low."""
        resp = verifier_client.post("/verify", json={
            "claim": "The bridge at grid Z99 collapsed on 01/01/2099",
            "source": "integration-test",
            "evidence_pack": [],
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # with no evidence pack, memory cross ref gets 0
        self.assertIn(data["verdict"], ("REPAIR", "FAIL_CLOSED"))

    def test_health_chain(self):
        """Both services respond healthy."""
        r1 = memu_client.get("/health")
        r2 = verifier_client.get("/health")
        self.assertEqual(r1.status_code, 200)
        self.assertEqual(r2.status_code, 200)
        self.assertEqual(r1.json()["status"], "ok")
        self.assertEqual(r2.json()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
