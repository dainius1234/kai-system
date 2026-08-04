"""Tests for git-watcher service."""
import importlib.util
import sys
from pathlib import Path as _Path

# Derived from this file's location, never written out. Three test
# files carried the literal path of one developer's machine, so they
# raised FileNotFoundError on every other machine — 26 failures on
# CI's first complete run, in files that pass here.
_REPO = _Path(__file__).resolve().parents[1]
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from scripts.module_stubs import stubbed  # noqa: E402
import types
import unittest

from fastapi.testclient import TestClient  # must be imported before any stubs


def _load_module():
    _stubs = {}
    for key in list(sys.modules.keys()):
        if "git_watcher" in key:
            del sys.modules[key]

    runtime = types.ModuleType("common.runtime")
    runtime.setup_json_logger = lambda *_, **__: __import__("logging").getLogger("git-test")
    runtime.ErrorBudget = type("ErrorBudget", (), {
        "__init__": lambda self, **_: None,
        "record": lambda self, *a, **k: None,
        "snapshot": lambda self: {},
    })
    sys.modules.setdefault("common", types.ModuleType("common"))
    _stubs["common.runtime"] = runtime

    spec = importlib.util.spec_from_file_location(
        "git_watcher",
        str(_REPO / "git-watcher" / "app.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    with stubbed(_stubs):
        spec.loader.exec_module(mod)
    return mod


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_health_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("repo_count", data)
        self.assertIn("uptime_seconds", data)

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)


class TestReposEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_repos_empty(self):
        self.mod._repos = []
        r = self.client.get("/repos")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["count"], 0)
        self.assertIsInstance(data["repos"], list)

    def test_repos_with_data(self):
        fake_repo = {
            "path": "/workspace",
            "branch": "main",
            "commit_hash": "abc1234",
            "commit_message": "initial commit",
            "commit_author": "Test User",
            "commit_date": "2026-07-24 10:00:00 +0000",
            "uncommitted_changes": 0,
            "untracked_files": 0,
            "ahead": 0,
            "behind": 0,
            "stash_count": 0,
            "error": None,
        }
        self.mod._repos = [fake_repo]
        r = self.client.get("/repos")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["repos"][0]["branch"], "main")


class TestDirtyEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_dirty_none(self):
        self.mod._repos = [
            {
                "path": "/workspace",
                "branch": "main",
                "commit_hash": "abc1234",
                "commit_message": "fix",
                "commit_author": "Dev",
                "commit_date": "2026-07-24 10:00:00 +0000",
                "uncommitted_changes": 0,
                "untracked_files": 0,
                "ahead": 0,
                "behind": 0,
                "stash_count": 0,
                "error": None,
            }
        ]
        r = self.client.get("/dirty")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["count"], 0)
        self.assertEqual(data["repos"], [])

    def test_dirty_with_changes(self):
        self.mod._repos = [
            {
                "path": "/workspace",
                "branch": "feat",
                "commit_hash": "def5678",
                "commit_message": "wip",
                "commit_author": "Dev",
                "commit_date": "2026-07-24 10:00:00 +0000",
                "uncommitted_changes": 3,
                "untracked_files": 1,
                "ahead": 0,
                "behind": 0,
                "stash_count": 0,
                "error": None,
            }
        ]
        r = self.client.get("/dirty")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["repos"][0]["uncommitted_changes"], 3)


class TestSummaryEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_summary_not_polled(self):
        self.mod._repos = []
        self.mod._last_poll = 0
        r = self.client.get("/summary")
        self.assertEqual(r.status_code, 200)
        self.assertIn("not yet polled", r.json()["summary"])

    def test_summary_clean(self):
        self.mod._repos = [
            {
                "path": "/workspace",
                "branch": "main",
                "commit_hash": "abc1234",
                "commit_message": "fix",
                "commit_author": "Dev",
                "commit_date": "2026-07-24 10:00:00 +0000",
                "uncommitted_changes": 0,
                "untracked_files": 0,
                "ahead": 0,
                "behind": 0,
                "stash_count": 0,
                "error": None,
            }
        ]
        self.mod._last_poll = 1.0
        r = self.client.get("/summary")
        self.assertEqual(r.status_code, 200)
        summary = r.json()["summary"]
        self.assertIn("main", summary)
        self.assertIn("no uncommitted changes", summary)

    def test_summary_dirty(self):
        self.mod._repos = [
            {
                "path": "/workspace",
                "branch": "feat",
                "commit_hash": "def5678",
                "commit_message": "wip",
                "commit_author": "Dev",
                "commit_date": "2026-07-24 10:00:00 +0000",
                "uncommitted_changes": 2,
                "untracked_files": 0,
                "ahead": 0,
                "behind": 0,
                "stash_count": 0,
                "error": None,
            }
        ]
        self.mod._last_poll = 1.0
        r = self.client.get("/summary")
        self.assertEqual(r.status_code, 200)
        summary = r.json()["summary"]
        self.assertIn("2 uncommitted changes", summary)


class TestInspectRepo(unittest.TestCase):
    def _get_inspect(self):
        """Load a fresh module and return the _inspect_repo function."""
        mod = _load_module()
        return mod._inspect_repo

    def test_not_a_git_repo(self):
        inspect = self._get_inspect()
        result = inspect("/nonexistent/path/that/does/not/exist")
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["uncommitted_changes"], 0)
        self.assertEqual(result["untracked_files"], 0)

    def test_valid_repo(self):
        inspect = self._get_inspect()
        result = inspect(str(_REPO))
        self.assertIsNone(result["error"], f"Unexpected error: {result['error']}")
        self.assertTrue(result["branch"], "branch should be non-empty")
        self.assertTrue(result["commit_hash"], "commit_hash should be non-empty")


if __name__ == "__main__":
    unittest.main()
