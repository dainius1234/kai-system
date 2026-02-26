from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# import dashboard module
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("dashboard_app", ROOT / "dashboard" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_ui_page():
    resp = client.get("/ui")
    assert resp.status_code == 200
    text = resp.text
    assert "Sovereign Dashboard" in text
    assert "<div id=\"nodes\"" in text
    # static index should also be reachable directly
    resp2 = client.get("/static/index.html")
    assert resp2.status_code == 200
    assert "Sovereign AI" in resp2.text

if __name__ == "__main__":
    test_ui_page()
    print("dashboard UI test passed")
