"""Tests for sandboxes/shell/app.py — allowlist-gated shell execution."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

sys.path.insert(0, str(ROOT / "sandboxes" / "shell"))
from app import app  # noqa: E402

client = TestClient(app)


class TestShellSandboxHealth(unittest.TestCase):
    def test_health(self):
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_allowlist_endpoint(self):
        r = client.get("/allowlist")
        self.assertEqual(r.status_code, 200)
        cmds = r.json()["commands"]
        self.assertIn("echo", cmds)
        self.assertIn("ls", cmds)
        self.assertNotIn("rm", cmds)
        self.assertNotIn("curl", cmds)
        self.assertNotIn("env", cmds)
        self.assertNotIn("printenv", cmds)


class TestShellSandboxAllowed(unittest.TestCase):
    def test_echo(self):
        r = client.post("/run", json={"command": "echo hello"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("hello", data["stdout"])
        self.assertEqual(data["returncode"], 0)

    def test_date(self):
        r = client.post("/run", json={"command": "date"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["returncode"], 0)

    def test_pwd(self):
        r = client.post("/run", json={"command": "pwd"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("/", r.json()["stdout"])

    def test_whoami(self):
        r = client.post("/run", json={"command": "whoami"})
        self.assertEqual(r.status_code, 200)
        self.assertGreater(len(r.json()["stdout"]), 0)

    def test_echo_with_args(self):
        r = client.post("/run", json={"command": "echo foo bar baz"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("foo bar baz", r.json()["stdout"])

    def test_nonzero_returncode(self):
        r = client.post("/run", json={"command": "ls /tmp/nonexistent_xyz_123"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertNotEqual(data["returncode"], 0)
        self.assertEqual(data["status"], "error")

    def test_command_field_in_response(self):
        r = client.post("/run", json={"command": "echo test"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("command", r.json())


class TestShellSandboxPathRestriction(unittest.TestCase):
    """Path-argument commands must be restricted to SAFE_DIRS."""

    def test_cat_passwd_blocked(self):
        r = client.post("/run", json={"command": "cat /etc/passwd"})
        self.assertEqual(r.status_code, 403)
        self.assertIn("outside allowed", r.json()["detail"])

    def test_cat_proc_environ_blocked(self):
        r = client.post("/run", json={"command": "cat /proc/1/environ"})
        self.assertEqual(r.status_code, 403)

    def test_head_etc_shadow_blocked(self):
        r = client.post("/run", json={"command": "head /etc/shadow"})
        self.assertEqual(r.status_code, 403)

    def test_tail_private_key_blocked(self):
        r = client.post("/run", json={"command": "tail /root/.ssh/id_rsa"})
        self.assertEqual(r.status_code, 403)

    def test_ls_root_blocked(self):
        r = client.post("/run", json={"command": "ls /root"})
        self.assertEqual(r.status_code, 403)

    def test_du_etc_blocked(self):
        r = client.post("/run", json={"command": "du /etc"})
        self.assertEqual(r.status_code, 403)

    def test_wc_passwd_blocked(self):
        r = client.post("/run", json={"command": "wc -l /etc/passwd"})
        self.assertEqual(r.status_code, 403)

    def test_cat_tmp_allowed(self):
        # /tmp is in SAFE_DIRS — a cat on a /tmp path should pass the path check
        # (may fail with non-zero exit if file doesn't exist, which is fine)
        r = client.post("/run", json={"command": "cat /tmp/nonexistent_test_file"})
        # Path check passes; subprocess returns non-zero but HTTP is 200
        self.assertIn(r.status_code, (200, 404))

    def test_ls_with_flag_only_allowed(self):
        # ls with only flags (no path arg) — safe, no path to check
        r = client.post("/run", json={"command": "ls -la"})
        self.assertEqual(r.status_code, 200)

    def test_cat_proc_self_allowed(self):
        r = client.post("/run", json={"command": "cat /proc/self/status"})
        # Path check passes (/proc/self is in SAFE_DIRS)
        self.assertIn(r.status_code, (200,))

    def test_allowlist_exposes_safe_dirs(self):
        r = client.get("/allowlist")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("safe_dirs", data)
        self.assertIn("path_restricted_commands", data)
        self.assertIn("cat", data["path_restricted_commands"])


class TestShellSandboxBlocked(unittest.TestCase):
    def test_rm_blocked(self):
        r = client.post("/run", json={"command": "rm -rf /"})
        self.assertEqual(r.status_code, 403)

    def test_curl_blocked(self):
        r = client.post("/run", json={"command": "curl http://example.com"})
        self.assertEqual(r.status_code, 403)

    def test_python_blocked(self):
        r = client.post("/run", json={"command": "python3 -c 'import os; os.system(\"id\")'"}
)
        self.assertEqual(r.status_code, 403)

    def test_sudo_blocked(self):
        r = client.post("/run", json={"command": "sudo whoami"})
        self.assertEqual(r.status_code, 403)

    def test_bash_blocked(self):
        r = client.post("/run", json={"command": "bash -c 'echo pwned'"})
        self.assertEqual(r.status_code, 403)

    def test_env_blocked(self):
        r = client.post("/run", json={"command": "env"})
        self.assertEqual(r.status_code, 403)

    def test_printenv_blocked(self):
        r = client.post("/run", json={"command": "printenv"})
        self.assertEqual(r.status_code, 403)

    def test_empty_command(self):
        r = client.post("/run", json={"command": ""})
        self.assertEqual(r.status_code, 400)

    def test_invalid_syntax(self):
        r = client.post("/run", json={"command": "echo 'unclosed"})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
