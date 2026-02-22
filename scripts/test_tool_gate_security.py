from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

from fastapi import HTTPException

module_path = Path(__file__).resolve().parents[1] / "tool-gate" / "app.py"
spec = importlib.util.spec_from_file_location("tool_gate_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.TRUSTED_TOKENS = {"bootstrap-token-1"}
mod.TOKEN_SCOPES = {"bootstrap-token-1": {"*"}}
assert mod._is_tool_allowed("bootstrap-token-1", "executor")

req = mod.GateRequest(tool="executor", actor_did="langgraph", session_id="bootstrap-token-1", confidence=0.9, nonce="n1", ts=time.time())
mod._validate_nonce(req)

try:
    mod._validate_nonce(req)
    raise AssertionError("expected replay detection")
except HTTPException as exc:
    assert exc.status_code == 409

print("tool-gate security tests passed")
