from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# import camera app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("camera_app", ROOT / "perception" / "camera" / "app.py")
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
    # expect camera_device/virtual_device keys rather than generic device
    assert "camera_device" in j and "virtual_device" in j


def test_capture():
    resp = client.post("/process")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


if __name__ == "__main__":
    test_health()
    test_capture()
    print("camera service tests passed")
