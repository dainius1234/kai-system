from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("EPISODE_STORE", "memory")

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))
spec = importlib.util.spec_from_file_location("introspect_app", ROOT / "agentic" / "introspect_app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


def test_dream_insufficient_data():
    # Fresh in-memory episode store has nothing to dream on.
    resp = client.post("/dream")
    assert resp.status_code == 200
    assert resp.json().get("status") == "insufficient_data"


def test_evolve_suggestions_empty():
    resp = client.get("/evolve/suggestions")
    assert resp.status_code == 200
    j = resp.json()
    assert j.get("status") == "ok"
    assert "reports" in j


def test_security_audit():
    resp = client.get("/security/audit")
    assert resp.status_code == 200
    j = resp.json()
    assert "findings" in j
    assert "risk_score" in j


def test_core_routes_not_present():
    # /chat and /run belong to agentic core, not introspect — this is the
    # actual proof the split happened, not just that the file was renamed.
    assert client.post("/chat", json={}).status_code == 404
    assert client.post("/run", json={}).status_code == 404


if __name__ == "__main__":
    test_health()
    test_dream_insufficient_data()
    test_evolve_suggestions_empty()
    test_security_audit()
    test_core_routes_not_present()
    print("agentic-introspect tests passed")
