from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") in ("ok", "degraded")
    assert "backend" in j
    assert "default_voice" in j


def test_synthesize():
    resp = client.post("/synthesize", json={"text": "hello"})
    if resp.status_code == 200:
        # Real TTS returns audio bytes (edge-tts available)
        assert resp.headers.get("content-type", "").startswith("audio/")
        assert len(resp.content) > 100
    else:
        # edge-tts not installed in test env → 503
        assert resp.status_code == 503


def test_voices():
    resp = client.get("/voices")
    assert resp.status_code == 200
    j = resp.json()
    assert "presets" in j
    assert "kai-default" in j["presets"]


if __name__ == "__main__":
    test_health()
    test_synthesize()
    test_voices()
    print("tts service tests passed")
