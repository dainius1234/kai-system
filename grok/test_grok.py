from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

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
