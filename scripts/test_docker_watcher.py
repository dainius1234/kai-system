"""Tests for docker-watcher service."""
import importlib.util
import sys
import types
import unittest

from fastapi.testclient import TestClient  # must be imported before httpx stub


def _load_module(with_sdk=True, containers=None):
    for key in list(sys.modules.keys()):
        if "docker_watcher" in key:
            del sys.modules[key]

    if with_sdk:
        docker_stub = types.ModuleType("docker")

        class _FakeImage:
            tags = ["myimage:latest"]
            short_id = "abc123"

        class _FakeContainer:
            short_id = "aabb"
            name = "test-container"
            image = _FakeImage()
            status = "running"
            attrs = {
                "State": {
                    "Health": {"Status": "healthy"},
                    "RestartCount": 0,
                    "StartedAt": "2026-07-24T00:00:00Z",
                    "ExitCode": 0,
                },
                "HostConfig": {},
            }

        _clist = containers if containers is not None else [_FakeContainer()]

        class _Containers:
            def list(self, all=False):
                return _clist

        class _Client:
            containers = _Containers()

        docker_stub.from_env = lambda: _Client()
        sys.modules["docker"] = docker_stub
    else:
        sys.modules.pop("docker", None)

    runtime = types.ModuleType("common.runtime")
    runtime.setup_json_logger = lambda *_, **__: __import__("logging").getLogger("docker-test")
    runtime.ErrorBudget = type("ErrorBudget", (), {
        "__init__": lambda self, **_: None,
        "record": lambda self, *a, **k: None,
        "snapshot": lambda self: {},
    })
    sys.modules.setdefault("common", types.ModuleType("common"))
    sys.modules["common.runtime"] = runtime

    spec = importlib.util.spec_from_file_location(
        "docker_watcher",
        "/home/user/kai-system/docker-watcher/app.py",
    )
    mod = importlib.util.module_from_spec(spec)
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
        self.assertIn("uptime_seconds", data)
        self.assertIn("docker_sdk", data)

    def test_metrics(self):
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)


class TestContainersEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.mod._containers = [
            {"id": "aa", "name": "web", "image": "nginx:latest", "status": "running",
             "health": "healthy", "restarts": 0, "started_at": "", "exit_code": 0},
            {"id": "bb", "name": "db", "image": "postgres:15", "status": "running",
             "health": "none", "restarts": 0, "started_at": "", "exit_code": 0},
        ]
        self.client = TestClient(self.mod.app)

    def test_containers_list(self):
        r = self.client.get("/containers")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["total"], 2)
        self.assertIsInstance(data["containers"], list)

    def test_containers_empty(self):
        self.mod._containers = []
        r = self.client.get("/containers")
        self.assertEqual(r.json()["total"], 0)


class TestUnhealthyEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_all_healthy(self):
        self.mod._containers = [
            {"id": "aa", "name": "web", "image": "nginx", "status": "running",
             "health": "healthy", "restarts": 0, "started_at": "", "exit_code": 0}
        ]
        r = self.client.get("/unhealthy")
        self.assertEqual(r.json()["count"], 0)

    def test_unhealthy_container(self):
        self.mod._containers = [
            {"id": "aa", "name": "broken", "image": "nginx", "status": "running",
             "health": "unhealthy", "restarts": 0, "started_at": "", "exit_code": 0}
        ]
        r = self.client.get("/unhealthy")
        self.assertEqual(r.json()["count"], 1)

    def test_high_restarts_flagged(self):
        self.mod._containers = [
            {"id": "aa", "name": "crashloop", "image": "nginx", "status": "running",
             "health": "none", "restarts": 10, "started_at": "", "exit_code": 0}
        ]
        r = self.client.get("/unhealthy")
        self.assertEqual(r.json()["count"], 1)


class TestSummaryEndpoint(unittest.TestCase):
    def setUp(self):
        self.mod = _load_module()
        self.client = TestClient(self.mod.app)

    def test_summary_empty(self):
        self.mod._containers = []
        r = self.client.get("/summary")
        self.assertIn("not yet", r.json()["summary"])

    def test_summary_all_running(self):
        self.mod._containers = [
            {"id": "aa", "name": "web", "image": "nginx", "status": "running",
             "health": "none", "restarts": 0, "started_at": "", "exit_code": 0}
        ]
        r = self.client.get("/summary")
        self.assertIn("running", r.json()["summary"])

    def test_summary_with_issues(self):
        self.mod._containers = [
            {"id": "aa", "name": "bad", "image": "nginx", "status": "running",
             "health": "unhealthy", "restarts": 0, "started_at": "", "exit_code": 0}
        ]
        r = self.client.get("/summary")
        self.assertIn("issues", r.json()["summary"])


class TestPollViaSdk(unittest.TestCase):
    def test_sdk_available(self):
        mod = _load_module(with_sdk=True)
        self.assertTrue(mod._DOCKER_SDK)

    def test_sdk_not_available(self):
        mod = _load_module(with_sdk=False)
        self.assertFalse(mod._DOCKER_SDK)

    def test_poll_sdk_returns_list(self):
        mod = _load_module(with_sdk=True)
        result = mod._poll_via_sdk()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["name"], "test-container")


if __name__ == "__main__":
    unittest.main()
