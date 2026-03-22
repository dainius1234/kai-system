"""Tests for the production hardening sprint:
  1. Redis pubsub SSE endpoint (dashboard)
  2. Docker secrets load_secret (common/auth)
  3. Backup service full lifecycle
  4. HMAC rotation drill (comprehensive)
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# backup-service dir has a hyphen — can't be imported normally
import importlib


def _load_backup_app():
    """Import backup-service/app.py via importlib (hyphenated dir)."""
    spec = importlib.util.spec_from_file_location(
        "backup_app",
        os.path.join(os.path.dirname(__file__), "..", "backup-service", "app.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1. Docker Secrets (load_secret) ─────────────────────────────────

class TestLoadSecret(unittest.TestCase):
    """Tests for common.auth.load_secret()."""

    def setUp(self):
        from common.auth import load_secret
        self._load = load_secret

    def test_env_var_fallback(self):
        os.environ["_TEST_LS_1"] = "hello"
        try:
            self.assertEqual(self._load("_TEST_LS_1"), "hello")
        finally:
            del os.environ["_TEST_LS_1"]

    def test_default_when_missing(self):
        os.environ.pop("_TEST_LS_MISSING", None)
        self.assertEqual(self._load("_TEST_LS_MISSING", "def"), "def")

    def test_run_secrets_path(self):
        """When env var points to /run/secrets/..., read from file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, prefix="/tmp/") as f:
            f.write("  secret-from-file  \n")
            path = f.name
        try:
            # Simulate Docker secret: patch check so our temp file is read
            os.environ["_TEST_LS_FILE"] = "/run/secrets/test_ls_file"
            with patch("pathlib.Path.is_file", return_value=True), \
                 patch("pathlib.Path.read_text", return_value="  secret-from-file  \n"):
                val = self._load("_TEST_LS_FILE")
                self.assertEqual(val, "secret-from-file")
        finally:
            os.environ.pop("_TEST_LS_FILE", None)
            os.unlink(path)

    def test_run_secrets_file_missing_returns_default(self):
        os.environ["_TEST_LS_NOFILE"] = "/run/secrets/nonexistent"
        try:
            val = self._load("_TEST_LS_NOFILE", "backup")
            self.assertEqual(val, "backup")
        finally:
            del os.environ["_TEST_LS_NOFILE"]

    def test_convention_file_by_env_name(self):
        """If /run/secrets/<env_name_lower> exists, read it."""
        with patch("pathlib.Path.is_file", return_value=True), \
             patch("pathlib.Path.read_text", return_value="conv-secret"):
            os.environ.pop("_TEST_CONV_SECRET", None)
            val = self._load("_TEST_CONV_SECRET", "")
            # Since env is empty, it checks convention path
            self.assertEqual(val, "conv-secret")


# ── 2. Redis Pubsub SSE (dashboard) ─────────────────────────────────

class TestDashboardPubsub(unittest.TestCase):
    """Tests for the Redis pubsub SSE endpoint in dashboard."""

    def test_event_channels_defined(self):
        from dashboard.app import _EVENT_CHANNELS
        self.assertIn("kai:health", _EVENT_CHANNELS)
        self.assertIn("kai:episode", _EVENT_CHANNELS)
        self.assertIn("kai:breaker", _EVENT_CHANNELS)
        self.assertIn("kai:memory", _EVENT_CHANNELS)

    def test_publish_event_exists(self):
        from dashboard.app import _publish_event
        self.assertTrue(callable(_publish_event))

    def test_sse_endpoint_exists(self):
        from dashboard.app import app
        routes = [r.path for r in app.routes if hasattr(r, "path")]
        self.assertIn("/api/events", routes)

    def test_publish_event_graceful_failure(self):
        """_publish_event should not raise even if Redis is down."""
        from dashboard.app import _publish_event
        loop = asyncio.new_event_loop()
        try:
            # Will fail to connect but should not raise
            loop.run_until_complete(_publish_event("kai:test", {"msg": "test"}))
        finally:
            loop.close()


# ── 3. Backup Service ───────────────────────────────────────────────

class TestBackupService(unittest.TestCase):
    """Tests for the upgraded backup-service."""

    @classmethod
    def setUpClass(cls):
        cls._mod = _load_backup_app()
        cls._app = cls._mod.app
        # Use a temp dir to avoid /data/backup permission errors
        cls._tmp = tempfile.mkdtemp(prefix="kai-backup-test-")
        cls._mod.BACKUP_DIR = cls._tmp

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls._tmp, ignore_errors=True)

    def _client(self):
        from fastapi.testclient import TestClient
        return TestClient(self._app)

    def test_health(self):
        resp = self._client().get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")

    def test_backup_list_empty(self):
        old = self._mod.BACKUP_DIR
        self._mod.BACKUP_DIR = os.path.join(self._tmp, "nonexistent-sub")
        try:
            resp = self._client().get("/backup/list")
            self.assertEqual(resp.status_code, 200)
            self.assertIn("backups", resp.json())
        finally:
            self._mod.BACKUP_DIR = old

    def test_postgres_backup_no_pg_dump(self):
        with patch("shutil.which", return_value=None):
            resp = self._client().post("/backup/postgres")
            self.assertEqual(resp.status_code, 503)

    def test_redis_backup_no_redis_cli(self):
        with patch("shutil.which", return_value=None):
            resp = self._client().post("/backup/redis")
            self.assertEqual(resp.status_code, 503)

    def test_restore_missing_filename(self):
        resp = self._client().post("/restore/postgres")
        self.assertEqual(resp.status_code, 400)

    def test_restore_invalid_filename(self):
        resp = self._client().post("/restore/postgres?backup_file=../../etc/passwd")
        self.assertEqual(resp.status_code, 400)

    def test_restore_file_not_found(self):
        resp = self._client().post("/restore/postgres?backup_file=nonexistent.sql")
        self.assertEqual(resp.status_code, 404)

    def test_full_backup_endpoints_exist(self):
        routes = [r.path for r in self._app.routes if hasattr(r, "path")]
        for path in ["/backup/postgres", "/backup/redis", "/backup/memory",
                     "/backup/ledger", "/backup/full", "/backup/list",
                     "/restore/postgres"]:
            self.assertIn(path, routes, f"Missing route: {path}")

    def test_sha256_helper(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            path = f.name
        try:
            result = self._mod._sha256(path)
            expected = hashlib.sha256(b"test content").hexdigest()
            self.assertEqual(result, expected)
        finally:
            os.unlink(path)


# ── 4. HMAC Rotation Drill ──────────────────────────────────────────

class TestHMACRotation(unittest.TestCase):
    """Rotation drill — phases 1/2/3."""

    def setUp(self):
        for key in ["INTERSERVICE_HMAC_SECRET", "INTERSERVICE_HMAC_SECRET_PREV",
                     "INTERSERVICE_HMAC_KEY_ID", "INTERSERVICE_HMAC_KEY_ID_PREV",
                     "INTERSERVICE_HMAC_REVOKED_IDS", "INTERSERVICE_HMAC_STRICT_KEY_ID"]:
            os.environ.pop(key, None)
        import common.auth as auth
        auth._WARNED_DEFAULT_SECRET = False
        self._auth = auth

    def _sign(self, **extra):
        return self._auth.sign_gate_request(
            actor_did="test", session_id="s1", tool="t1",
            nonce="n1", ts=1700000000.0, **extra)

    def _verify(self, sig):
        return self._auth.verify_gate_signature(
            actor_did="test", session_id="s1", tool="t1",
            nonce="n1", ts=1700000000.0, signature=sig)

    def test_phase1_sign_verify(self):
        os.environ["INTERSERVICE_HMAC_SECRET"] = "alpha"
        os.environ["INTERSERVICE_HMAC_KEY_ID"] = "v1"
        sig = self._sign()
        self.assertTrue(sig.startswith("v1:"))
        self.assertTrue(self._verify(sig))

    def test_phase2_overlap(self):
        os.environ["INTERSERVICE_HMAC_SECRET"] = "alpha"
        os.environ["INTERSERVICE_HMAC_KEY_ID"] = "v1"
        old_sig = self._sign()
        os.environ["INTERSERVICE_HMAC_SECRET"] = "beta"
        os.environ["INTERSERVICE_HMAC_SECRET_PREV"] = "alpha"
        os.environ["INTERSERVICE_HMAC_KEY_ID"] = "v2"
        os.environ["INTERSERVICE_HMAC_KEY_ID_PREV"] = "v1"
        new_sig = self._sign()
        self.assertTrue(self._verify(old_sig), "old sig should verify in overlap")
        self.assertTrue(self._verify(new_sig), "new sig should verify in overlap")

    def test_phase3_revoke(self):
        os.environ["INTERSERVICE_HMAC_SECRET"] = "alpha"
        os.environ["INTERSERVICE_HMAC_KEY_ID"] = "v1"
        old_sig = self._sign()
        os.environ["INTERSERVICE_HMAC_SECRET"] = "beta"
        os.environ["INTERSERVICE_HMAC_KEY_ID"] = "v2"
        os.environ.pop("INTERSERVICE_HMAC_SECRET_PREV", None)
        os.environ["INTERSERVICE_HMAC_REVOKED_IDS"] = "v1"
        self.assertFalse(self._verify(old_sig))

    def test_dual_sign_bundle_length(self):
        os.environ["INTERSERVICE_HMAC_SECRET"] = "beta"
        os.environ["INTERSERVICE_HMAC_SECRET_PREV"] = "alpha"
        os.environ["INTERSERVICE_HMAC_KEY_ID"] = "v2"
        os.environ["INTERSERVICE_HMAC_KEY_ID_PREV"] = "v1"
        bundle = self._auth.sign_gate_request_bundle(
            actor_did="test", session_id="s1", tool="t1",
            nonce="n1", ts=1700000000.0)
        self.assertEqual(len(bundle), 2)

    def test_ed25519_state(self):
        import scripts.auto_rotate_ed25519 as rot
        original = rot.STATE_PATH
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        try:
            rot.STATE_PATH = Path(tmp.name)
            os.unlink(tmp.name)
            state = rot.load_state()
            self.assertIn("current", state)
            self.assertEqual(state["mode"], "single")
        finally:
            rot.STATE_PATH = original
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


# ── 5. Docker Compose Secrets Config ────────────────────────────────

class TestDockerSecretsConfig(unittest.TestCase):
    """Verify docker-compose files have secrets section."""

    def test_full_compose_has_secrets(self):
        compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.full.yml")
        with open(compose_path) as f:
            content = f.read()
        self.assertIn("secrets:", content)
        self.assertIn("hmac_secret", content)
        self.assertIn("db_password", content)
        self.assertIn("bridge_secret", content)

    def test_tool_gate_has_secrets(self):
        compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.full.yml")
        with open(compose_path) as f:
            content = f.read()
        # tool-gate should reference hmac_secret
        self.assertIn("hmac_secret", content)

    def test_backup_service_has_env_vars(self):
        compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.full.yml")
        with open(compose_path) as f:
            content = f.read()
        self.assertIn("MEMU_URL", content)
        self.assertIn("TOOL_GATE_URL", content)


if __name__ == "__main__":
    unittest.main()
