from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

# ensure workspace root is on the import path for common/ and services
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from common.auth import sign_gate_request

# import the tool-gate module directly from source path
module_path = Path(__file__).resolve().parents[1] / "tool-gate" / "app.py"
spec = importlib.util.spec_from_file_location("tool_gate_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# initialize env for tests
mod.TRUSTED_TOKENS = {"bootstrap-token-1"}
mod.TOKEN_SCOPES = {"bootstrap-token-1": {"executor"}}
mod.SEEN_NONCES.clear()
mod.ledger = mod.InMemoryLedger()  # reset ledger
mod.policy = mod.GatePolicy()  # fresh policy
mod.policy.mode = "WORK"  # start in WORK so approval logic is exercised
mod.policy.allowed_tools.add("executor")  # ensure our test tool is allowed

client = TestClient(mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "mode" in body


def make_request(confidence: float, cosign: bool = False):
    now = time.time()
    req = {
        "tool": "executor",
        "actor_did": "langgraph",
        "session_id": "bootstrap-token-1",
        "confidence": confidence,
        "nonce": f"n{now}",
        "ts": now,
        "signature": sign_gate_request(
            actor_did="langgraph",
            session_id="bootstrap-token-1",
            tool="executor",
            nonce=f"n{now}",
            ts=now,
        ),
        "cosign": cosign,
    }
    return req


def test_gate_block_low_confidence():
    resp = client.post("/gate/request", json=make_request(0.1))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is False
    assert data["reason_code"] == "LOW_CONFIDENCE"


def test_gate_approve_confidence():
    resp = client.post("/gate/request", json=make_request(0.9))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["reason_code"] == "APPROVED"


def test_gate_cosign_overrides():
    resp = client.post("/gate/request", json=make_request(0.1, cosign=True))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["reason_code"] == "APPROVED"


def test_mode_switch_and_public():
    # switch to PUB then back to WORK to exercise both
    resp = client.post("/gate/mode", json={"mode": "PUB", "reason": "testing"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "PUB"
    h = client.get("/health").json()
    assert h["mode"] == "PUB"
    resp2 = client.post("/gate/mode", json={"mode": "WORK", "reason": "resume"})
    assert resp2.status_code == 200
    assert resp2.json()["mode"] == "WORK"
    h2 = client.get("/health").json()
    assert h2["mode"] == "WORK"


def test_ledger_endpoints():
    # ledger stats should be >0 after previous decisions
    stats = client.get("/ledger/stats").json()
    assert stats["count"] >= 3
    tail = client.get("/ledger/tail", params={"limit": 2}).json()
    assert isinstance(tail, list) and len(tail) <= 2
    verify = client.get("/ledger/verify").json()
    assert verify["valid"] is True
    merkle = client.get("/ledger/merkle").json()
    assert merkle["valid"] is True


def test_replay_detection():
    # reuse same nonce should be rejected
    req = make_request(0.9)
    resp1 = client.post("/gate/request", json=req)
    assert resp1.status_code == 200
    resp2 = client.post("/gate/request", json=req)
    assert resp2.status_code == 409 or resp2.status_code == 400


if __name__ == "__main__":
    test_health()
    test_gate_block_low_confidence()
    test_gate_approve_confidence()
    test_gate_cosign_overrides()
    test_mode_switch_and_public()
    test_ledger_endpoints()
    test_replay_detection()
    print("tool-gate api tests passed")
