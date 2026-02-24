from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

# ── ensure workspace root is on the import path for common/ and services ──
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.auth import sign_gate_request

# ── set LEDGER_PATH to temp dir BEFORE importing tool-gate ──────────────
# PersistentLedger creates the path on module load; /data/ is not writable
# in CI, so we redirect to an ephemeral directory.
_TMPDIR = tempfile.mkdtemp(prefix="tool-gate-test-")
os.environ["LEDGER_PATH"] = str(Path(_TMPDIR) / "ledger.jsonl")

# import the tool-gate module directly from source path
module_path = ROOT / "tool-gate" / "app.py"
spec = importlib.util.spec_from_file_location("tool_gate_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from fastapi.testclient import TestClient

# ── initialize env for tests ────────────────────────────────────────────
AUTH_TOKEN = "bootstrap-token-1"
AUTH_HEADER = {"Authorization": f"Bearer {AUTH_TOKEN}"}

mod.TRUSTED_TOKENS = {AUTH_TOKEN}
mod.TOKEN_SCOPES = {AUTH_TOKEN: {"executor"}}
mod.SEEN_NONCES.clear()
mod.ledger = mod.PersistentLedger(Path(_TMPDIR) / "test-ledger.jsonl")
mod.policy = mod.GatePolicy()
mod.policy.mode = "WORK"
mod.policy.allowed_tools.add("executor")

client = TestClient(mod.app)


# ── helpers ──────────────────────────────────────────────────────────────

def make_request(confidence: float, cosign: bool = False):
    now = time.time()
    nonce = f"n{now}"
    return {
        "tool": "executor",
        "actor_did": "langgraph",
        "session_id": AUTH_TOKEN,
        "confidence": confidence,
        "nonce": nonce,
        "ts": now,
        "signature": sign_gate_request(
            actor_did="langgraph",
            session_id=AUTH_TOKEN,
            tool="executor",
            nonce=nonce,
            ts=now,
        ),
        "cosign": cosign,
    }


# ── tests ────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "mode" in body


def test_gate_pending_cosign_very_low():
    """Very low confidence (< COSIGN_CONFIDENCE_THRESHOLD) parks for co-sign."""
    resp = client.post("/gate/request", json=make_request(0.1))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is False
    assert data["reason_code"] == "PENDING_COSIGN"


def test_gate_block_low_confidence():
    """Mid-range confidence (above cosign threshold but below required) is blocked."""
    resp = client.post("/gate/request", json=make_request(0.6))
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
    assert data["reason_code"] == "COSIGNED"


def test_mode_switch_and_public():
    # switch to PUB then back to WORK — requires auth
    resp = client.post("/gate/mode", json={"mode": "PUB", "reason": "testing"}, headers=AUTH_HEADER)
    assert resp.status_code == 200
    assert resp.json()["mode"] == "PUB"
    h = client.get("/health").json()
    assert h["mode"] == "PUB"
    resp2 = client.post("/gate/mode", json={"mode": "WORK", "reason": "resume"}, headers=AUTH_HEADER)
    assert resp2.status_code == 200
    assert resp2.json()["mode"] == "WORK"
    h2 = client.get("/health").json()
    assert h2["mode"] == "WORK"


def test_mode_switch_requires_auth():
    """Unauthenticated mode switch must be rejected."""
    resp = client.post("/gate/mode", json={"mode": "PUB", "reason": "unauthed"})
    assert resp.status_code == 401


def test_ledger_endpoints():
    # ledger stats should be >0 after previous decisions
    stats = client.get("/ledger/stats").json()
    assert stats["count"] >= 3
    tail = client.get("/ledger/tail", params={"limit": 2}, headers=AUTH_HEADER).json()
    assert isinstance(tail, list) and len(tail) <= 2
    verify = client.get("/ledger/verify").json()
    assert verify["valid"] is True
    merkle = client.get("/ledger/merkle").json()
    assert merkle["valid"] is True


def test_ledger_tail_requires_auth():
    """Unauthenticated ledger read must be rejected."""
    resp = client.get("/ledger/tail")
    assert resp.status_code == 401


def test_replay_detection():
    # reuse same nonce should be rejected
    req = make_request(0.9)
    resp1 = client.post("/gate/request", json=req)
    assert resp1.status_code == 200
    resp2 = client.post("/gate/request", json=req)
    assert resp2.status_code == 409 or resp2.status_code == 400


if __name__ == "__main__":
    test_health()
    test_gate_pending_cosign_very_low()
    test_gate_block_low_confidence()
    test_gate_approve_confidence()
    test_gate_cosign_overrides()
    test_mode_switch_and_public()
    test_mode_switch_requires_auth()
    test_ledger_endpoints()
    test_ledger_tail_requires_auth()
    test_replay_detection()
    print("tool-gate api tests passed")
