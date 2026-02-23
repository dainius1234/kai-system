from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# import heartbeat app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("heartbeat_app", ROOT / "heartbeat" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status").startswith("ok") or resp.json().get("status").startswith("running")


def test_status():
    resp = client.get("/status")
    assert resp.status_code == 200
    payload = resp.json()
    assert "status" in payload
    assert "elapsed_seconds" in payload


def test_tick_and_event():
    r1 = client.post("/tick")
    assert r1.status_code == 200
    r2 = client.post("/event", json={"status": "test", "reason": "unit"})
    assert r2.status_code == 200

if __name__ == "__main__":
    test_health()
    test_status()
    test_tick_and_event()
    print("heartbeat tests passed")
