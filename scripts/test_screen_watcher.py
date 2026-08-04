"""Screen-watcher service tests — screen-capture HTTP is stubbed."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from scripts.module_stubs import stubbed  # noqa: E402
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient


# ── Stubs ─────────────────────────────────────────────────────────────────────

def _make_httpx_stub(screenshot_bytes=b"\x89PNG\r\n" + b"\x00" * 100):
    stub = types.ModuleType("httpx")

    class FakeResp:
        def __init__(self):
            self.status_code = 200
            self.content = screenshot_bytes

        def raise_for_status(self):
            pass

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        async def get(self, url, **kwargs):
            return FakeResp()

        async def post(self, url, **kwargs):
            return FakeResp()

    stub.AsyncClient = FakeClient
    stub.HTTPStatusError = Exception
    stub.RequestError = Exception
    return stub


def _load_module(monkeypatch):
    _stubs = {}
    mod_name = "screen_watcher_app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    _stubs["httpx"] = _make_httpx_stub()

    monkeypatch.setenv("PORT", "8036")
    monkeypatch.setenv("WATCH_INTERVAL_SECONDS", "10")
    monkeypatch.setenv("CHANGE_THRESHOLD", "0.05")

    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent.parent / "screen-watcher" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    with stubbed(_stubs):
        spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client(monkeypatch):
    mod = _load_module(monkeypatch)
    return TestClient(mod.app), mod


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["watching"] is False
    assert body["uptime_seconds"] >= 0


def test_metrics(client):
    c, _ = client
    r = c.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_status_initial(client):
    c, _ = client
    r = c.get("/status")
    assert r.status_code == 200
    body = r.json()
    assert body["watching"] is False
    assert body["interval_seconds"] == 10
    assert body["threshold"] == 0.05
    assert body["last_capture_ts"] == 0.0
    assert body["last_change_ts"] == 0.0
    assert body["last_diff_score"] == 0.0


def test_snapshot_no_data(client):
    c, _ = client
    r = c.get("/snapshot")
    assert r.status_code == 404


def test_snapshot_returns_image(client):
    c, mod = client
    mod._last_screenshot = b"\x89PNG\r\n" + b"\x00" * 100
    r = c.get("/snapshot")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0


def test_watch_start(client):
    c, mod = client
    r = c.post("/watch/start", json={"interval_seconds": 5, "threshold": 0.1})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["interval_seconds"] == 5
    assert body["threshold"] == 0.1
    assert mod._watching is True
    # Cleanup
    mod._watching = False
    if mod._watch_task:
        mod._watch_task.cancel()


def test_watch_start_already_watching(client):
    c, mod = client
    mod._watching = True
    r = c.post("/watch/start")
    assert r.status_code == 200
    assert r.json()["already_watching"] is True
    mod._watching = False


def test_watch_stop(client):
    c, mod = client
    mod._watching = True
    r = c.post("/watch/stop")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert mod._watching is False


def test_watch_stop_when_not_running(client):
    c, mod = client
    mod._watching = False
    r = c.post("/watch/stop")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_image_hash_deterministic(client):
    _, mod = client
    data = b"\x00\x01\x02" * 1000
    h1 = mod._image_hash(data)
    h2 = mod._image_hash(data)
    assert h1 == h2


def test_image_hash_different_images(client):
    _, mod = client
    h1 = mod._image_hash(b"\x00" * 1000)
    h2 = mod._image_hash(b"\xff" * 1000)
    assert h1 != h2


def test_diff_score_identical(client):
    _, mod = client
    h = mod._image_hash(b"\xAB" * 1000)
    assert mod._diff_score(h, h) == 0.0


def test_diff_score_different(client):
    _, mod = client
    h1 = mod._image_hash(b"\x00" * 1000)
    h2 = mod._image_hash(b"\xff" * 1000)
    score = mod._diff_score(h1, h2)
    assert 0.0 <= score <= 1.0
    assert score > 0.0


def test_diff_score_empty_hashes(client):
    _, mod = client
    score = mod._diff_score("", "")
    assert score == 1.0


def test_watch_interval_min_clamped(client):
    c, mod = client
    r = c.post("/watch/start", json={"interval_seconds": 0})
    assert r.status_code == 200
    assert mod._interval >= 2  # clamped to minimum
    mod._watching = False
    if mod._watch_task:
        mod._watch_task.cancel()


def test_watch_threshold_clamped_high(client):
    c, mod = client
    r = c.post("/watch/start", json={"threshold": 2.0})
    assert r.status_code == 200
    assert mod._threshold <= 1.0
    mod._watching = False
    if mod._watch_task:
        mod._watch_task.cancel()


def test_watch_threshold_clamped_low(client):
    c, mod = client
    r = c.post("/watch/start", json={"threshold": -1.0})
    assert r.status_code == 200
    assert mod._threshold >= 0.0
    mod._watching = False
    if mod._watch_task:
        mod._watch_task.cancel()
