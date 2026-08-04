"""Tests for perception/files/app.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.module_stubs import stubbed  # noqa: E402

_SVC = Path(__file__).resolve().parents[1] / "perception" / "files" / "app.py"


def _load():
    spec = importlib.util.spec_from_file_location("files_app", _SVC)
    mod = importlib.util.module_from_spec(spec)
    runtime_stub = MagicMock()
    runtime_stub.setup_json_logger.return_value = MagicMock()
    runtime_stub.ErrorBudget = MagicMock(return_value=MagicMock(record=MagicMock(), snapshot=MagicMock(return_value={})))
    # Stub watchdog so tests run without it installed. Scoped to the
    # import: see scripts/module_stubs.py.
    watchdog_stub = MagicMock()
    stubs = {
        "common.runtime": runtime_stub,
        "watchdog": watchdog_stub,
        "watchdog.observers": watchdog_stub,
        "watchdog.events": watchdog_stub,
    }
    if "common" not in sys.modules:
        stubs["common"] = MagicMock()
    with stubbed(stubs):
        spec.loader.exec_module(mod)
    return mod


_mod = _load()
client = TestClient(_mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "watching" in data
    assert "events_buffered" in data


def test_watching():
    resp = client.get("/watching")
    assert resp.status_code == 200
    assert "directories" in resp.json()


def test_events_empty():
    _mod._events.clear()
    resp = client.get("/events")
    assert resp.status_code == 200
    data = resp.json()
    assert data["events"] == []
    assert data["total"] == 0


def test_events_populated():
    _mod._events.clear()
    import time
    _mod._events.append({"path": "/tmp/test.py", "event": "modified", "timestamp": time.time()})
    _mod._events.append({"path": "/tmp/other.py", "event": "created", "timestamp": time.time()})
    resp = client.get("/events")
    assert resp.status_code == 200
    events = resp.json()["events"]
    assert len(events) == 2


def test_events_limit():
    _mod._events.clear()
    import time
    for i in range(10):
        _mod._events.append({"path": f"/tmp/f{i}.py", "event": "modified", "timestamp": time.time()})
    resp = client.get("/events?limit=3")
    assert len(resp.json()["events"]) == 3


def test_events_filter_by_type():
    _mod._events.clear()
    import time
    _mod._events.append({"path": "/tmp/a.py", "event": "created", "timestamp": time.time()})
    _mod._events.append({"path": "/tmp/b.py", "event": "deleted", "timestamp": time.time()})
    resp = client.get("/events?event_type=created")
    events = resp.json()["events"]
    assert all(e["event"] == "created" for e in events)


def test_watch_no_watchdog():
    """When watchdog is not available, POST /watch returns 503."""
    _mod._WATCHDOG_OK = False
    resp = client.post("/watch", json={"directory": "/tmp"})
    assert resp.status_code == 503
    _mod._WATCHDOG_OK = True


def test_remove_watch_not_watching():
    resp = client.request("DELETE", "/watch", json={"directory": "/nonexistent"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is False


def test_remove_watch_present():
    _mod._watching.append("/tmp/test-dir")
    resp = client.request("DELETE", "/watch", json={"directory": "/tmp/test-dir"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert "/tmp/test-dir" not in _mod._watching


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
