"""Tests for perception/clipboard/app.py."""
import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

_SVC = Path(__file__).resolve().parents[1] / "perception" / "clipboard" / "app.py"


def _load():
    spec = importlib.util.spec_from_file_location("clipboard_app", _SVC)
    mod = importlib.util.module_from_spec(spec)
    # Stub common.runtime before import
    runtime_stub = MagicMock()
    runtime_stub.setup_json_logger.return_value = MagicMock()
    runtime_stub.ErrorBudget = MagicMock(return_value=MagicMock(record=MagicMock(), snapshot=MagicMock(return_value={})))
    sys.modules.setdefault("common", MagicMock())
    sys.modules["common.runtime"] = runtime_stub
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
client = TestClient(_mod.app)


def _reset():
    _mod._history.clear()
    _mod._counter = 0


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_push_basic():
    _reset()
    resp = client.post("/push", json={"content": "hello world"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["id"] == 1


def test_push_empty_ignored():
    _reset()
    resp = client.post("/push", json={"content": "   "})
    assert resp.status_code == 200
    assert resp.json()["id"] is None


def test_push_dedup():
    _reset()
    client.post("/push", json={"content": "abc"})
    resp = client.post("/push", json={"content": "abc"})
    assert resp.status_code == 200
    assert resp.json().get("note") == "duplicate"
    assert len(_mod._history) == 1


def test_push_exceeds_size_limit():
    _reset()
    big = "x" * (_mod.MAX_CONTENT_BYTES + 1)
    resp = client.post("/push", json={"content": big})
    assert resp.status_code == 413


def test_latest_empty():
    _reset()
    resp = client.get("/latest")
    assert resp.status_code == 404


def test_latest_returns_last():
    _reset()
    client.post("/push", json={"content": "first"})
    client.post("/push", json={"content": "second"})
    resp = client.get("/latest")
    assert resp.status_code == 200
    assert resp.json()["content"] == "second"


def test_history():
    _reset()
    client.post("/push", json={"content": "a"})
    client.post("/push", json={"content": "b"})
    resp = client.get("/history")
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 2
    assert entries[0]["content"] == "b"  # reversed


def test_history_limit():
    _reset()
    for i in range(5):
        client.post("/push", json={"content": f"item-{i}"})
    resp = client.get("/history?limit=2")
    assert len(resp.json()["entries"]) == 2


def test_clear_history():
    _reset()
    client.post("/push", json={"content": "to clear"})
    resp = client.delete("/history")
    assert resp.status_code == 200
    assert resp.json()["cleared"] is True
    assert len(_mod._history) == 0


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_source_default():
    _reset()
    client.post("/push", json={"content": "no source"})
    resp = client.get("/latest")
    assert resp.json()["source"] == "browser"
