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
    assert j.get("status") == "ok"
    assert "model" in j


def test_synthesize():
    resp = client.post("/synthesize", json={"text": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert data.get("audio_path")


if __name__ == "__main__":
    test_health()
    test_synthesize()
    print("tts service tests passed")
