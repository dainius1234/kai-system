from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

# import tts app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("tts_app", ROOT / "output" / "tts" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)

# Stub audio bytes returned by the fake edge-tts backend
_FAKE_AUDIO = b"\xFF\xFB" + b"\x00" * 200  # minimal MP3-ish header + padding


def _make_fake_communicate(audio_bytes: bytes = _FAKE_AUDIO):
    """Return a mock edge_tts.Communicate that yields fake audio without network."""

    async def _fake_stream():
        yield {"type": "audio", "data": audio_bytes}

    mock_comm = MagicMock()
    mock_comm.stream = _fake_stream
    return MagicMock(return_value=mock_comm)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") in ("ok", "degraded")
    assert "backend" in j
    assert "default_voice" in j


def test_synthesize_offline_happy_path():
    """Synthesis succeeds when edge-tts is available and returns audio bytes."""
    fake_edge_tts = ModuleType("edge_tts")
    fake_edge_tts.Communicate = _make_fake_communicate()
    original = mod._edge_available
    mod._edge_available = True
    mod.edge_tts = fake_edge_tts
    try:
        resp = client.post("/synthesize", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.headers.get("content-type", "").startswith("audio/")
        assert len(resp.content) > 100
    finally:
        mod._edge_available = original


def test_synthesize_network_failure_returns_503():
    """A network error from edge-tts surfaces as HTTP 503 (upstream unavailable)."""

    class _ErrorStream:
        """Async iterable that immediately raises a network-style exception."""
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise Exception("Invalid response status, url='wss://speech.platform.bing.com/...'")

    fake_comm = MagicMock()
    fake_comm.stream = _ErrorStream
    fake_edge_tts = ModuleType("edge_tts")
    fake_edge_tts.Communicate = MagicMock(return_value=fake_comm)

    original = mod._edge_available
    mod._edge_available = True
    mod.edge_tts = fake_edge_tts
    try:
        resp = client.post("/synthesize", json={"text": "hello"})
        assert resp.status_code == 503
    finally:
        mod._edge_available = original


def test_synthesize_no_backend_returns_503():
    """When edge-tts is not installed the endpoint returns 503."""
    original = mod._edge_available
    mod._edge_available = False
    try:
        resp = client.post("/synthesize", json={"text": "hello"})
        assert resp.status_code == 503
    finally:
        mod._edge_available = original


def test_voices():
    resp = client.get("/voices")
    assert resp.status_code == 200
    j = resp.json()
    assert "presets" in j
    assert "kai-default" in j["presets"]


if __name__ == "__main__":
    test_health()
    test_synthesize_offline_happy_path()
    test_synthesize_network_failure_returns_503()
    test_synthesize_no_backend_returns_503()
    test_voices()
    print("tts service tests passed")
