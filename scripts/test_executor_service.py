from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# import executor app
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("executor_app", ROOT / "executor" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_health_alive():
    for path in ("/health", "/alive"):
        resp = client.get(path)
        assert resp.status_code == 200
        assert resp.json().get("status") == "ok"


def test_execute_noop():
    resp = client.post("/execute", json={"tool": "noop", "params": {}, "task_id": "t1", "device": "cpu"})
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "completed"


if __name__ == "__main__":
    test_health_alive()
    test_execute_noop()
    print("executor service tests passed")
