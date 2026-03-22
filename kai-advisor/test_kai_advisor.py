import importlib
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure we import kai-advisor's app, not another service's app module
_kai_advisor_dir = str(Path(__file__).resolve().parent)
if _kai_advisor_dir not in sys.path:
    sys.path.insert(0, _kai_advisor_dir)
# Force fresh import from kai-advisor directory
if "app" in sys.modules:
    _saved = sys.modules.pop("app")
_mod = importlib.import_module("app")
sys.modules.setdefault("app", _mod)

client = TestClient(_mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"
    assert "model" in data


def test_ask_empty():
    resp = client.post("/ask", json={"question": ""})
    assert resp.status_code == 400


def test_ask_echo():
    resp = client.post("/ask", json={"question": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["question"] == "hello"
    assert "answer" in data
