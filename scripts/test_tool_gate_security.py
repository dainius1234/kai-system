from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import time
from pathlib import Path

from fastapi import HTTPException

os.environ["HMAC_ALLOW_DEV_SECRET"] = "true"
from common.auth import sign_gate_request

# Set LEDGER_PATH to a temp dir before importing tool-gate (it creates the dir on import)
_tmpdir = tempfile.mkdtemp()
os.environ["LEDGER_PATH"] = os.path.join(_tmpdir, "ledger.jsonl")

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
    actor_did="agentic",
    session_id="bootstrap-token-1",
    conviction=9.0,
    nonce="n1",
    ts=now,
    signature=sign_gate_request(actor_did="agentic", session_id="bootstrap-token-1", tool="executor", nonce="n1", ts=now),
)
mod._validate_nonce_and_sig(req)

try:
    mod._validate_nonce_and_sig(req)
    raise AssertionError("expected replay detection")
except HTTPException as exc:
    assert exc.status_code == 409

print("tool-gate security tests passed")


# ── Every allowlisted tool must be classified ────────────────────────
#
# A tool on neither classification list was treated as reversible, and
# `n8n` sat in that gap for the life of this gate. It runs arbitrary
# workflows — it can fire any actuator, reach any external service and
# mutate any state this system can reach — and because
# `IRREVERSIBLE_TOOLS_JSON` is set in no compose file, no env file and no
# Makefile target, the code default *was* the configuration everywhere.
# So n8n needed conviction and no operator confirmation, purely because
# nobody had named it.
#
# "Unclassified" and "safe" being the same state is the fail-open this
# programme keeps finding. These assertions make them different states.

_unclassified = mod.unclassified_tools()
assert _unclassified == set(), (
    f"allowlisted but classified as neither irreversible nor explicitly "
    f"reversible: {sorted(_unclassified)}. Add it to IRREVERSIBLE_TOOLS_JSON "
    f"or REVERSIBLE_TOOLS_JSON — leaving it out is a decision to treat it "
    f"as safe.")

# ...and the detector notices one, or the assertion above is vacuous.
_original_allowed = set(mod.GatePolicy.allowed_tools) if isinstance(
    getattr(mod.GatePolicy, "allowed_tools", None), set) else None
_probe = mod.GatePolicy()
_probe_original = set(_probe.allowed_tools)
_probe.allowed_tools.add("brand-new-unclassified-tool")
try:
    _classified = set().union(*mod.IRREVERSIBLE_CATEGORIES.values())
    _seen = set(_probe.allowed_tools) - _classified - mod.REVERSIBLE_TOOLS
    assert "brand-new-unclassified-tool" in _seen, (
        "the unclassified-tool detector does not react to an unclassified "
        "tool, so the assertion above proves nothing")
finally:
    _probe.allowed_tools.clear()
    _probe.allowed_tools.update(_probe_original)

assert mod._classify_irreversibility("n8n") is not None, \
    "n8n runs arbitrary workflows and must require operator confirmation"
assert mod._classify_irreversibility("shell") is not None
# The classification must stay meaningful: if everything were
# irreversible the co-sign requirement would be noise.
assert mod._classify_irreversibility("noop") is None

print("  tool classification: every allowlisted tool is classified")
