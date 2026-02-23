from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# import langgraph app
ROOT = Path(__file__).resolve().parents[1]
# include both workspace root and langgraph directory so config module can be imported
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "langgraph"))
# redis is an optional dependency; provide a dummy module so import succeeds
import types
sys.modules.setdefault("redis", types.SimpleNamespace())
spec = importlib.util.spec_from_file_location("langgraph_app", ROOT / "langgraph" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"



if __name__ == "__main__":
    test_health()
    print("langgraph service tests passed")
