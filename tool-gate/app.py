from __future__ import annotations

import hashlib
import json
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from common.auth import verify_gate_signature
from common.runtime import AuditStream, ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("tool-gate", os.getenv("LOG_PATH", "/tmp/tool-gate.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Tool Gate", version="0.4.0")
TOKENS_PATH = Path(os.getenv("TRUSTED_TOKENS_PATH", "/config/trusted_tokens.txt"))
NONCE_CACHE_PATH = Path(os.getenv("NONCE_CACHE_PATH", "/tmp/tool-gate-nonces.json"))
TRUSTED_TOKENS: Set[str] = set()
TOKEN_SCOPES: Dict[str, Set[str]] = {}
NONCE_TTL_SECONDS = int(os.getenv("NONCE_TTL_SECONDS", "300"))
SIGNATURE_SKEW_SECONDS = int(os.getenv("SIGNATURE_SKEW_SECONDS", str(NONCE_TTL_SECONDS)))
REQUIRE_SIGNATURE = os.getenv("REQUIRE_SIGNATURE", "true").lower() == "true"
SEEN_NONCES: Dict[str, float] = {}
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("tool-gate", required=os.getenv("AUDIT_REQUIRED", "false").lower() == "true")


class GateRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    actor_did: str
    session_id: str
    cosign: bool = False
    rationale: Optional[str] = None
    device: Optional[str] = None
    nonce: Optional[str] = None
    ts: Optional[float] = None
    signature: Optional[str] = None
    signatures: List[str] = Field(default_factory=list)
    request_source: Optional[str] = None
    trace_id: Optional[str] = None


class GateDecision(BaseModel):
    request_id: str
    approved: bool
    status: str
    reason: str
    ledger_hash: str
    reason_code: str
    evaluated_at: float
    policy_version: str


class ModeChange(BaseModel):
    mode: str
    reason: str


@dataclass
class LedgerEntry:
    request_id: str
    timestamp: float
    payload: Dict[str, Any]
    approved: bool
    reason: str
    prev_hash: str
    entry_hash: str


class InMemoryLedger:
    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []

    def append(self, payload: Dict[str, Any], approved: bool, reason: str) -> LedgerEntry:
        request_id = hashlib.sha256(f"{time.time_ns()}-{payload}".encode()).hexdigest()
        prev_hash = self._entries[-1].entry_hash if self._entries else "GENESIS"
        entry_data = {"request_id": request_id, "timestamp": time.time(), "payload": payload, "approved": approved, "reason": reason, "prev_hash": prev_hash}
        entry_hash = hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()
        entry = LedgerEntry(request_id=request_id, timestamp=entry_data["timestamp"], payload=payload, approved=approved, reason=reason, prev_hash=prev_hash, entry_hash=entry_hash)
        self._entries.append(entry)
        return entry

    def tail(self, limit: int = 10) -> List[LedgerEntry]:
        return self._entries[-limit:]

    def count(self) -> int:
        return len(self._entries)


ledger = InMemoryLedger()


def _merkle_root(entries: List[LedgerEntry]) -> str:
    if not entries:
        return "GENESIS"
    level = [e.entry_hash for e in entries]
    while len(level) > 1:
        nxt: List[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256(f"{left}{right}".encode()).hexdigest())
        level = nxt
    return level[0]


def _persist_nonces() -> None:
    try:
        NONCE_CACHE_PATH.write_text(json.dumps(SEEN_NONCES), encoding="utf-8")
    except Exception:
        logger.warning("Failed to persist nonce cache")


def _restore_nonces() -> None:
    if not NONCE_CACHE_PATH.exists():
        return
    try:
        payload = json.loads(NONCE_CACHE_PATH.read_text(encoding="utf-8"))
        SEEN_NONCES.update({str(k): float(v) for k, v in payload.items()})
    except Exception:
        logger.warning("Failed to restore nonce cache")


def load_trusted_tokens() -> None:
    global TRUSTED_TOKENS, TOKEN_SCOPES
    TRUSTED_TOKENS = set()
    TOKEN_SCOPES = {}
    if not TOKENS_PATH.exists():
        return
    for raw in TOKENS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            token, scope_raw = line.split(":", 1)
            token = token.strip()
            scopes = {x.strip() for x in scope_raw.split(",") if x.strip()}
            TRUSTED_TOKENS.add(token)
            TOKEN_SCOPES[token] = scopes
        else:
            TRUSTED_TOKENS.add(line)
            TOKEN_SCOPES[line] = {"*"}


def _reload_tokens(_signum: int, _frame: Any) -> None:
    load_trusted_tokens()


def _cleanup_nonces(now: float) -> None:
    stale = [k for k, ts in SEEN_NONCES.items() if now - ts > NONCE_TTL_SECONDS]
    for key in stale:
        SEEN_NONCES.pop(key, None)


def _validate_nonce_and_sig(request: GateRequest) -> None:
    if request.nonce is None and request.ts is None and not REQUIRE_SIGNATURE:
        return
    if not request.nonce or request.ts is None:
        raise HTTPException(status_code=400, detail="nonce and ts must be provided together")
    now = time.time()
    if abs(now - request.ts) > SIGNATURE_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="request timestamp outside allowed window")
    if REQUIRE_SIGNATURE:
        candidates = []
        if request.signature:
            candidates.append(request.signature)
        candidates.extend(request.signatures)
        is_valid = any(
            verify_gate_signature(
                actor_did=request.actor_did,
                session_id=request.session_id,
                tool=request.tool,
                nonce=request.nonce,
                ts=request.ts,
                signature=candidate,
            )
            for candidate in candidates
        )
        if not is_valid:
            raise HTTPException(status_code=401, detail="invalid request signature")
    _cleanup_nonces(now)
    nonce_key = f"{request.session_id}:{request.nonce}"
    if nonce_key in SEEN_NONCES:
        raise HTTPException(status_code=409, detail="replay detected")
    SEEN_NONCES[nonce_key] = now
    _persist_nonces()


def _is_tool_allowed(token: str, tool: str) -> bool:
    scopes = TOKEN_SCOPES.get(token, set())
    return "*" in scopes or tool in scopes


class GatePolicy:
    def __init__(self) -> None:
        self.mode = os.getenv("MODE", "PUB").upper()
        self.required_confidence = float(os.getenv("REQUIRED_CONFIDENCE", "0.7"))
        self.policy_version = os.getenv("POLICY_VERSION", "phase1-v1")
        self.allowed_tools = {"shell", "qgis", "n8n", "noop"}

    def evaluate(self, request: GateRequest) -> GateDecision:
        if self.mode == "PUB":
            approved, reason, reason_code = False, "Tool Gate in PUB mode (execution disabled).", "PUB_MODE"
        elif request.confidence >= self.required_confidence or request.cosign:
            approved, reason, reason_code = True, "Approved by confidence threshold or co-sign.", "APPROVED"
        else:
            approved, reason, reason_code = False, "Insufficient confidence; co-sign required.", "LOW_CONFIDENCE"

        entry = ledger.append(request.model_dump(), approved, reason)
        return GateDecision(
            request_id=entry.request_id,
            approved=approved,
            status="approved" if approved else "blocked",
            reason=reason,
            ledger_hash=entry.entry_hash,
            reason_code=reason_code,
            evaluated_at=time.time(),
            policy_version=self.policy_version,
        )


policy = GatePolicy()
load_trusted_tokens()
_restore_nonces()
signal.signal(signal.SIGHUP, _reload_tokens)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        audit.log("info", f"{request.method} {request.url.path} -> {response.status_code}")
        return response
    except Exception:
        budget.record(500)
        audit.log("error", f"{request.method} {request.url.path} -> 500")
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "mode": policy.mode, "device": DEVICE, "trusted_tokens": str(len(TRUSTED_TOKENS))}


@app.get("/metrics")
async def metrics() -> Dict[str, float]:
    return budget.snapshot()


@app.post("/gate/request", response_model=GateDecision)
async def gate_request(request: GateRequest) -> GateDecision:
    request = request.model_copy(
        update={
            "tool": sanitize_string(request.tool),
            "actor_did": sanitize_string(request.actor_did),
            "session_id": sanitize_string(request.session_id),
            "rationale": sanitize_string(request.rationale) if request.rationale else None,
            "nonce": sanitize_string(request.nonce) if request.nonce else None,
            "signature": sanitize_string(request.signature) if request.signature else None,
            "signatures": [sanitize_string(x) for x in request.signatures],
            "request_source": sanitize_string(request.request_source) if request.request_source else "unknown",
            "trace_id": sanitize_string(request.trace_id) if request.trace_id else None,
        }
    )
    if not request.tool:
        raise HTTPException(status_code=400, detail={"reason_code": "ALLOWLIST_BLOCK", "message": "Tool name is required."})
    if not request.actor_did:
        raise HTTPException(status_code=400, detail={"reason_code": "ALLOWLIST_BLOCK", "message": "actor_did is required."})
    if request.tool not in policy.allowed_tools:
        raise HTTPException(status_code=400, detail={"reason_code": "ALLOWLIST_BLOCK", "message": f"Tool '{request.tool}' is not allowlisted."})
    if request.session_id not in TRUSTED_TOKENS:
        raise HTTPException(status_code=401, detail="Session not trusted")
    if not _is_tool_allowed(request.session_id, request.tool):
        raise HTTPException(status_code=403, detail="Token not authorized for this tool")
    _validate_nonce_and_sig(request)
    if request.device is None:
        request = request.model_copy(update={"device": DEVICE})
    return policy.evaluate(request)


@app.post("/gate/mode")
async def set_mode(change: ModeChange) -> Dict[str, str]:
    normalized = sanitize_string(change.mode).upper()
    if normalized not in {"PUB", "WORK"}:
        raise HTTPException(status_code=400, detail="Mode must be PUB or WORK.")
    policy.mode = normalized
    ledger.append({"mode": normalized, "reason": sanitize_string(change.reason)}, True, "Mode updated")
    return {"status": "ok", "mode": policy.mode}


@app.get("/ledger/tail")
async def ledger_tail(limit: int = 10) -> List[Dict[str, Any]]:
    return [{"request_id": e.request_id, "timestamp": e.timestamp, "payload": e.payload, "approved": e.approved, "reason": e.reason, "prev_hash": e.prev_hash, "entry_hash": e.entry_hash} for e in ledger.tail(limit)]


@app.get("/ledger/stats")
async def ledger_stats() -> Dict[str, Any]:
    return {"status": "ok", "count": ledger.count()}


@app.get("/ledger/verify")
async def ledger_verify() -> Dict[str, Any]:
    entries = ledger.tail(ledger.count())
    prev_hash = "GENESIS"
    for entry in entries:
        entry_data = {
            "request_id": entry.request_id,
            "timestamp": entry.timestamp,
            "payload": entry.payload,
            "approved": entry.approved,
            "reason": entry.reason,
            "prev_hash": entry.prev_hash,
        }
        expected_hash = hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()
        if entry.prev_hash != prev_hash or entry.entry_hash != expected_hash:
            return {"status": "error", "valid": False, "failed_request_id": entry.request_id}
        prev_hash = entry.entry_hash
    return {"status": "ok", "valid": True, "count": len(entries)}


@app.get("/ledger/merkle")
async def ledger_merkle() -> Dict[str, Any]:
    entries = ledger.tail(ledger.count())
    verify = await ledger_verify()
    return {
        "status": "ok" if verify.get("valid") else "error",
        "valid": bool(verify.get("valid")),
        "count": len(entries),
        "merkle_root": _merkle_root(entries),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
