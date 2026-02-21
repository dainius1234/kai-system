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

from common.runtime import AuditStream, ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("tool-gate", os.getenv("LOG_PATH", "/tmp/tool-gate.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Tool Gate", version="0.4.0")
TOKENS_PATH = Path(os.getenv("TRUSTED_TOKENS_PATH", "/config/trusted_tokens.txt"))
TRUSTED_TOKENS: Set[str] = set()
budget = ErrorBudget(window_seconds=300)
audit = AuditStream("tool-gate", required=os.getenv("AUDIT_REQUIRED", "false").lower()=="true")


class GateRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    actor_did: str
    session_id: str
    cosign: bool = False
    rationale: Optional[str] = None
    device: Optional[str] = None


class GateDecision(BaseModel):
    request_id: str
    approved: bool
    status: str
    reason: str
    ledger_hash: str


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


def load_trusted_tokens() -> None:
    global TRUSTED_TOKENS
    TRUSTED_TOKENS = {t.strip() for t in TOKENS_PATH.read_text(encoding="utf-8").splitlines() if t.strip()} if TOKENS_PATH.exists() else set()


def _reload_tokens(_signum: int, _frame: Any) -> None:
    load_trusted_tokens()


class GatePolicy:
    def __init__(self) -> None:
        self.mode = os.getenv("MODE", "PUB").upper()
        self.required_confidence = float(os.getenv("REQUIRED_CONFIDENCE", "0.7"))

    def evaluate(self, request: GateRequest) -> GateDecision:
        if self.mode == "PUB":
            approved, reason = False, "Tool Gate in PUB mode (execution disabled)."
        elif request.confidence >= self.required_confidence or request.cosign:
            approved, reason = True, "Approved by confidence threshold or co-sign."
        else:
            approved, reason = False, "Insufficient confidence; co-sign required."

        entry = ledger.append(request.model_dump(), approved, reason)
        return GateDecision(request_id=entry.request_id, approved=approved, status="approved" if approved else "blocked", reason=reason, ledger_hash=entry.entry_hash)


policy = GatePolicy()
load_trusted_tokens()
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
    request = request.model_copy(update={"tool": sanitize_string(request.tool), "actor_did": sanitize_string(request.actor_did), "session_id": sanitize_string(request.session_id), "rationale": sanitize_string(request.rationale) if request.rationale else None})
    if not request.tool:
        raise HTTPException(status_code=400, detail="Tool name is required.")
    if request.session_id not in TRUSTED_TOKENS:
        raise HTTPException(status_code=401, detail="Session not trusted")
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
