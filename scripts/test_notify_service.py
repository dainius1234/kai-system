"""Tests for output/notify/app.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

_SVC = Path(__file__).resolve().parents[1] / "output" / "notify" / "app.py"


def _load():
    spec = importlib.util.spec_from_file_location("notify_app", _SVC)
    mod = importlib.util.module_from_spec(spec)
    runtime_stub = MagicMock()
    runtime_stub.setup_json_logger.return_value = MagicMock()
    runtime_stub.ErrorBudget = MagicMock(return_value=MagicMock(record=MagicMock(), snapshot=MagicMock(return_value={})))
    sys.modules.setdefault("common", MagicMock())
    # The service now imports common.service_auth; because `common` is
    # stubbed above, the submodule must be stubbed too.  Auth itself is
    # covered by scripts/test_service_auth.py.
    _auth_stub = MagicMock()
    _auth_stub.require_service_auth = lambda operation: (lambda: None)
    sys.modules["common.service_auth"] = _auth_stub
    sys.modules["common.runtime"] = runtime_stub
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
client = TestClient(_mod.app)


def _reset():
    _mod._pending.clear()
    _mod._counter = 0


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_notify_no_title():
    _reset()
    resp = client.post("/notify", json={"title": "  ", "body": "test"})
    assert resp.status_code == 400


def test_notify_queued_when_no_notify_send():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        resp = client.post("/notify", json={"title": "Test", "body": "Hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["channel"] == "queue"
    assert data["id"] == 1


def test_notify_sent_via_notify_send():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=True):
        resp = client.post("/notify", json={"title": "Test", "body": "Hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["channel"] == "notify-send"
    assert data["id"] is None


def test_pending_empty():
    _reset()
    resp = client.get("/pending")
    assert resp.status_code == 200
    assert resp.json()["notifications"] == []


def test_pending_shows_unread():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        client.post("/notify", json={"title": "A", "body": "msg A"})
        client.post("/notify", json={"title": "B", "body": "msg B"})
    resp = client.get("/pending")
    assert resp.status_code == 200
    notes = resp.json()["notifications"]
    assert len(notes) == 2


def test_pending_all():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        client.post("/notify", json={"title": "X", "body": "msg"})
    client.delete("/pending/1")
    resp = client.get("/pending?unread_only=false")
    assert resp.json()["total"] == 1


def test_dismiss_single():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        client.post("/notify", json={"title": "Dismiss me", "body": "test"})
    resp = client.delete("/pending/1")
    assert resp.status_code == 200
    assert resp.json()["cleared"] is True
    # unread count should now be 0
    assert client.get("/pending").json()["total"] == 0


def test_dismiss_not_found():
    _reset()
    resp = client.delete("/pending/999")
    assert resp.status_code == 404


def test_dismiss_all():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        client.post("/notify", json={"title": "A", "body": "1"})
        client.post("/notify", json={"title": "B", "body": "2"})
    resp = client.delete("/pending")
    assert resp.status_code == 200
    assert client.get("/pending").json()["total"] == 0


def test_urgency_clamp():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        resp = client.post("/notify", json={"title": "T", "body": "b", "urgency": "invalid"})
    assert resp.status_code == 200


def test_timeout_clamp():
    _reset()
    with patch.object(_mod, "_try_notify_send", return_value=False):
        resp = client.post("/notify", json={"title": "T", "body": "b", "timeout_ms": 0})
    assert resp.status_code == 200


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
