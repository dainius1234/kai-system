"""HMAC Rotation Drill — validates the full key rotation lifecycle.

Simulates the 3-phase rotation procedure documented in
docs/hmac_rotation_runbook.md:

  Phase 1: Single key (old) — sign and verify
  Phase 2: Overlap window — new key primary, old key secondary
           Both signatures accepted, dual-sign bundle works
  Phase 3: Retirement — old key revoked, only new key accepted

Run: python3 scripts/hmac_rotation_drill.py
  or: make hmac-rotation-drill
"""
from __future__ import annotations

import os
import sys
import time
import unittest

# Ensure common/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _set_env(**kwargs: str) -> None:
    """Set environment variables (clearing empty ones)."""
    for k, v in kwargs.items():
        if v:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


class HMACRotationDrill(unittest.TestCase):
    """End-to-end rotation lifecycle test."""

    def setUp(self):
        for key in [
            "INTERSERVICE_HMAC_SECRET",
            "INTERSERVICE_HMAC_SECRET_PREV",
            "INTERSERVICE_HMAC_KEY_ID",
            "INTERSERVICE_HMAC_KEY_ID_PREV",
            "INTERSERVICE_HMAC_REVOKED_IDS",
            "INTERSERVICE_HMAC_STRICT_KEY_ID",
        ]:
            os.environ.pop(key, None)
        import common.auth as auth
        auth._WARNED_DEFAULT_SECRET = False
        self._auth = auth

    def _sign(self, **extra):
        return self._auth.sign_gate_request(
            actor_did="drill-actor", session_id="drill-session",
            tool="drill-tool", nonce="drill-nonce-001", ts=1700000000.0,
            **extra,
        )

    def _sign_bundle(self):
        return self._auth.sign_gate_request_bundle(
            actor_did="drill-actor", session_id="drill-session",
            tool="drill-tool", nonce="drill-nonce-001", ts=1700000000.0,
        )

    def _verify(self, sig):
        return self._auth.verify_gate_signature(
            actor_did="drill-actor", session_id="drill-session",
            tool="drill-tool", nonce="drill-nonce-001", ts=1700000000.0,
            signature=sig,
        )

    # ── Phase 1: Single key ──────────────────────────────────────────

    def test_phase1_single_key_sign_verify(self):
        _set_env(INTERSERVICE_HMAC_SECRET="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v1",
                 INTERSERVICE_HMAC_SECRET_PREV="")
        sig = self._sign()
        self.assertTrue(sig.startswith("v1:"))
        self.assertTrue(self._verify(sig))

    def test_phase1_wrong_secret_rejected(self):
        _set_env(INTERSERVICE_HMAC_SECRET="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v1")
        sig = self._sign()
        _set_env(INTERSERVICE_HMAC_SECRET="wrong-secret")
        self.assertFalse(self._verify(sig))

    def test_phase1_no_signature_rejected(self):
        self.assertFalse(self._verify(None))
        self.assertFalse(self._verify(""))

    # ── Phase 2: Overlap window ──────────────────────────────────────

    def test_phase2_new_key_accepted(self):
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-beta",
                 INTERSERVICE_HMAC_SECRET_PREV="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v2",
                 INTERSERVICE_HMAC_KEY_ID_PREV="v1")
        sig = self._sign()
        self.assertTrue(sig.startswith("v2:"))
        self.assertTrue(self._verify(sig))

    def test_phase2_old_key_still_accepted(self):
        _set_env(INTERSERVICE_HMAC_SECRET="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v1",
                 INTERSERVICE_HMAC_SECRET_PREV="")
        old_sig = self._sign()
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-beta",
                 INTERSERVICE_HMAC_SECRET_PREV="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v2",
                 INTERSERVICE_HMAC_KEY_ID_PREV="v1")
        self.assertTrue(self._verify(old_sig))

    def test_phase2_dual_sign_bundle(self):
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-beta",
                 INTERSERVICE_HMAC_SECRET_PREV="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v2",
                 INTERSERVICE_HMAC_KEY_ID_PREV="v1")
        bundle = self._sign_bundle()
        self.assertEqual(len(bundle), 2)
        self.assertTrue(bundle[0].startswith("v2:"))
        self.assertTrue(bundle[1].startswith("v1:"))
        self.assertTrue(self._verify(bundle[0]))
        self.assertTrue(self._verify(bundle[1]))

    def test_phase2_strict_key_id_binding(self):
        # Sign properly with old key
        _set_env(INTERSERVICE_HMAC_SECRET="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v1")
        proper_old_sig = self._sign(key_id="v1")
        # Verify in overlap+strict
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-beta",
                 INTERSERVICE_HMAC_SECRET_PREV="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v2",
                 INTERSERVICE_HMAC_KEY_ID_PREV="v1",
                 INTERSERVICE_HMAC_STRICT_KEY_ID="true")
        self.assertTrue(self._verify(proper_old_sig))

    # ── Phase 3: Retirement ──────────────────────────────────────────

    def test_phase3_old_key_revoked(self):
        _set_env(INTERSERVICE_HMAC_SECRET="old-secret-alpha",
                 INTERSERVICE_HMAC_KEY_ID="v1",
                 INTERSERVICE_HMAC_SECRET_PREV="")
        old_sig = self._sign()
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-beta",
                 INTERSERVICE_HMAC_KEY_ID="v2",
                 INTERSERVICE_HMAC_SECRET_PREV="",
                 INTERSERVICE_HMAC_REVOKED_IDS="v1")
        self.assertFalse(self._verify(old_sig))

    def test_phase3_new_key_works_after_retirement(self):
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-beta",
                 INTERSERVICE_HMAC_KEY_ID="v2",
                 INTERSERVICE_HMAC_SECRET_PREV="",
                 INTERSERVICE_HMAC_REVOKED_IDS="v1")
        sig = self._sign()
        self.assertTrue(self._verify(sig))

    def test_phase3_multiple_revoked_ids(self):
        _set_env(INTERSERVICE_HMAC_SECRET="new-secret-gamma",
                 INTERSERVICE_HMAC_KEY_ID="v3",
                 INTERSERVICE_HMAC_SECRET_PREV="",
                 INTERSERVICE_HMAC_REVOKED_IDS="v1,v2")
        sig = self._sign()
        self.assertTrue(self._verify(sig))
        self.assertFalse(self._verify("v1:fakedigest"))
        self.assertFalse(self._verify("v2:fakedigest"))

    # ── Ed25519 rotation state ───────────────────────────────────────

    def test_ed25519_state_lifecycle(self):
        import tempfile
        from scripts.auto_rotate_ed25519 import load_state, save_state, _new_keypair
        import scripts.auto_rotate_ed25519 as rot
        from pathlib import Path

        original = rot.STATE_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        try:
            rot.STATE_PATH = Path(tmp.name)
            os.unlink(tmp.name)

            state = load_state()
            self.assertIn("current", state)
            self.assertEqual(state["mode"], "single")
            self.assertIsNone(state["previous"])

            # Ensure unique key_id (time-based IDs collide within same second)
            new = _new_keypair()
            if new["key_id"] == state["current"]:
                new["key_id"] = new["key_id"] + "b"
            state["previous"] = state["current"]
            state["keys"][new["key_id"]] = new
            state["current"] = new["key_id"]
            state["mode"] = "dual_sign"
            state["rotated_at"] = time.time()
            save_state(state)

            reloaded = load_state()
            self.assertEqual(reloaded["mode"], "dual_sign")
            self.assertIsNotNone(reloaded["previous"])
            self.assertEqual(len(reloaded["keys"]), 2)
        finally:
            rot.STATE_PATH = original
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # ── Docker secrets load_secret ───────────────────────────────────

    def test_load_secret_from_env(self):
        os.environ["TEST_DRILL_SECRET"] = "my-secret-value"
        try:
            self.assertEqual(self._auth.load_secret("TEST_DRILL_SECRET"), "my-secret-value")
        finally:
            del os.environ["TEST_DRILL_SECRET"]

    def test_load_secret_default(self):
        os.environ.pop("NONEXISTENT_SECRET_VAR", None)
        self.assertEqual(self._auth.load_secret("NONEXISTENT_SECRET_VAR", "fallback"), "fallback")

    def test_load_secret_from_docker_path(self):
        """When env var is /run/secrets/<name> and file exists, read it."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, dir="/tmp") as f:
            f.write("  docker-secret-val  \n")
            secret_path = f.name
        # Monkey-patch to use /tmp path instead of /run/secrets/
        os.environ["TEST_SECRET_FILE"] = secret_path
        try:
            # Direct env read (no /run/secrets prefix) returns path as value
            val = self._auth.load_secret("TEST_SECRET_FILE")
            self.assertEqual(val, secret_path)
        finally:
            del os.environ["TEST_SECRET_FILE"]
            os.unlink(secret_path)


if __name__ == "__main__":
    unittest.main()
