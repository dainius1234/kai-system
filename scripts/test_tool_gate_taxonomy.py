"""Tool-gate irreversible-action taxonomy tests (Phase 0, Step B/C).

Exercises:
  - Irreversible tools (e.g. "shell") require BOTH conviction >=
    IRREVERSIBLE_MIN_CONVICTION AND explicit cosign confirmation.
  - Confirmation alone never substitutes for the conviction floor.
  - Non-irreversible tools are unaffected by this taxonomy.
  - PUB mode raises the general conviction bar; WORK mode does not.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HMAC_ALLOW_DEV_SECRET", "true")

from common.auth import sign_gate_request

_TMPDIR = tempfile.mkdtemp(prefix="tool-gate-taxonomy-test-")
os.environ["LEDGER_PATH"] = str(Path(_TMPDIR) / "ledger.jsonl")

module_path = ROOT / "tool-gate" / "app.py"
spec = importlib.util.spec_from_file_location("tool_gate_taxonomy", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

from fastapi.testclient import TestClient

AUTH_TOKEN = "taxonomy-test-token"
AUTH_HEADER = {"Authorization": f"Bearer {AUTH_TOKEN}"}

mod.TRUSTED_TOKENS = {AUTH_TOKEN}
mod.TOKEN_SCOPES = {AUTH_TOKEN: {"executor", "shell"}}
mod.SEEN_NONCES.clear()
mod.ledger = mod.PersistentLedger(Path(_TMPDIR) / "test-ledger.jsonl")
mod.policy = mod.GatePolicy()
mod.policy.mode = "WORK"
mod._mode_override_until[0] = time.time() + 3600 * 4
mod.policy.allowed_tools.update({"executor", "shell"})

client = TestClient(mod.app)


def make_request(tool: str, conviction: float, cosign: bool = False, nonce: str | None = None):
    now = time.time()
    n = nonce or f"n{now}"
    return {
        "tool": tool,
        "actor_did": "agentic",
        "session_id": AUTH_TOKEN,
        "conviction": conviction,
        "nonce": n,
        "ts": now,
        "signature": sign_gate_request(
            actor_did="agentic", session_id=AUTH_TOKEN, tool=tool, nonce=n, ts=now,
        ),
        "cosign": cosign,
    }


def test_irreversible_high_conviction_no_cosign_denied():
    resp = client.post("/gate/request", json=make_request("shell", 9.5))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is False
    assert data["reason_code"] == "IRREVERSIBLE_REQUIRES_CONFIRMATION"


def test_irreversible_high_conviction_with_cosign_approved():
    resp = client.post("/gate/request", json=make_request("shell", 9.5, cosign=True))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["reason_code"] == "IRREVERSIBLE_CONFIRMED"


def test_irreversible_low_conviction_with_cosign_still_denied():
    """Confirmation is never a substitute for the conviction floor."""
    resp = client.post("/gate/request", json=make_request("shell", 3.0, cosign=True))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is False
    assert data["reason_code"] == "IRREVERSIBLE_REQUIRES_CONFIRMATION"


def test_non_irreversible_tool_unaffected():
    resp = client.post("/gate/request", json=make_request("executor", 8.5))
    assert resp.status_code == 200
    data = resp.json()
    assert data["approved"] is True
    assert data["reason_code"] == "APPROVED"


def test_pub_mode_raises_bar_work_does_not():
    moderate = 8.5
    mod.policy.mode = "WORK"
    mod._mode_override_until[0] = time.time() + 3600 * 4
    work_resp = client.post("/gate/request", json=make_request("executor", moderate, nonce="pub-work-1"))
    assert work_resp.json()["reason_code"] == "APPROVED"

    mod.policy.mode = "PUB"
    mod._mode_override_until[0] = time.time() + 3600 * 4
    pub_resp = client.post("/gate/request", json=make_request("executor", moderate, nonce="pub-work-2"))
    assert pub_resp.json()["reason_code"] == "LOW_CONVICTION"

    near_max = 9.8
    pub_resp_high = client.post("/gate/request", json=make_request("executor", near_max, nonce="pub-work-3"))
    assert pub_resp_high.json()["reason_code"] == "APPROVED"

    # restore WORK for any subsequent tests in this module
    mod.policy.mode = "WORK"
    mod._mode_override_until[0] = time.time() + 3600 * 4


if __name__ == "__main__":
    test_irreversible_high_conviction_no_cosign_denied()
    test_irreversible_high_conviction_with_cosign_approved()
    test_irreversible_low_conviction_with_cosign_still_denied()
    test_non_irreversible_tool_unaffected()
    test_pub_mode_raises_bar_work_does_not()
    print("tool-gate taxonomy tests passed")
