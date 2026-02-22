from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

from fastapi import HTTPException

from common.auth import sign_gate_request

module_path = Path(__file__).resolve().parents[1] / "tool-gate" / "app.py"
spec = importlib.util.spec_from_file_location("tool_gate_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.TRUSTED_TOKENS = {"bootstrap-token-1"}
mod.TOKEN_SCOPES = {"bootstrap-token-1": {"executor"}}
mod.SEEN_NONCES.clear()
assert mod._is_tool_allowed("bootstrap-token-1", "executor")
assert not mod._is_tool_allowed("bootstrap-token-1", "memu-core")

now = time.time()
req = mod.GateRequest(
    tool="executor",
    actor_did="langgraph",
    session_id="bootstrap-token-1",
    confidence=0.9,
    nonce="n1",
    ts=now,
    signature=sign_gate_request(actor_did="langgraph", session_id="bootstrap-token-1", tool="executor", nonce="n1", ts=now),
)
mod._validate_nonce_and_sig(req)

try:
    mod._validate_nonce_and_sig(req)
    raise AssertionError("expected replay detection")
except HTTPException as exc:
    assert exc.status_code == 409

print("tool-gate security tests passed")
