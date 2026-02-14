from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Tool Gate", version="0.2.0")

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

DEVICE = "cuda" if torch and torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("No GPU — running on CPU only")

ALLOWED_TOOLS = {"shell", "qgis", "n8n", "noop"}
POLICY_VERSION = "2026.02-phase1"


class GateRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    actor_did: str
    cosign: bool = False
    rationale: Optional[str] = None
    device: Optional[str] = None
    request_source: Optional[str] = "unknown"
    trace_id: Optional[str] = None


class GateDecision(BaseModel):
    request_id: str
    approved: bool
    status: str
    reason: str
    reason_code: str
    ledger_hash: str
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
    reason_code: str
    prev_hash: str
    entry_hash: str


class InMemoryLedger:
    def __init__(self) -> None:
        self._entries: List[LedgerEntry] = []

    def append(self, payload: Dict[str, Any], approved: bool, reason: str, reason_code: str) -> LedgerEntry:
        request_id = hashlib.sha256(f"{time.time_ns()}-{payload}".encode()).hexdigest()
        prev_hash = self._entries[-1].entry_hash if self._entries else "GENESIS"
        entry_data = {
            "request_id": request_id,
            "timestamp": time.time(),
            "payload": payload,
            "approved": approved,
            "reason": reason,
            "reason_code": reason_code,
            "prev_hash": prev_hash,
        }
        entry_hash = hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()
        entry = LedgerEntry(
            request_id=request_id,
            timestamp=entry_data["timestamp"],
            payload=payload,
            approved=approved,
            reason=reason,
            reason_code=reason_code,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def tail(self, limit: int = 10) -> List[LedgerEntry]:
        return self._entries[-limit:]

    def count(self) -> int:
        return len(self._entries)

    def verify(self) -> Dict[str, Any]:
        prev = "GENESIS"
        for idx, entry in enumerate(self._entries):
            if entry.prev_hash != prev:
                return {"valid": False, "index": idx, "error": "prev_hash_mismatch"}
            check_data = {
                "request_id": entry.request_id,
                "timestamp": entry.timestamp,
                "payload": entry.payload,
                "approved": entry.approved,
                "reason": entry.reason,
                "reason_code": entry.reason_code,
                "prev_hash": entry.prev_hash,
            }
            check_hash = hashlib.sha256(json.dumps(check_data, sort_keys=True).encode()).hexdigest()
            if check_hash != entry.entry_hash:
                return {"valid": False, "index": idx, "error": "entry_hash_mismatch"}
            prev = entry.entry_hash
        return {"valid": True, "count": len(self._entries)}


ledger = InMemoryLedger()


class GatePolicy:
    def __init__(self) -> None:
        self.mode = os.getenv("MODE", "PUB").upper()
        self.required_confidence = float(os.getenv("REQUIRED_CONFIDENCE", "0.7"))

    def evaluate(self, request: GateRequest) -> GateDecision:
        if not request.actor_did.strip():
            raise HTTPException(status_code=400, detail={"reason_code": "INVALID_ACTOR", "reason": "actor_did is required"})

        if request.tool not in ALLOWED_TOOLS:
            raise HTTPException(
                status_code=400,
                detail={"reason_code": "ALLOWLIST_BLOCK", "reason": f"Tool '{request.tool}' is not allowlisted."},
            )

        reason_code = "APPROVED"
        if self.mode == "PUB":
            approved = False
            reason = "Tool Gate in PUB mode (execution disabled)."
            reason_code = "PUB_MODE"
        elif request.confidence >= self.required_confidence or request.cosign:
            approved = True
            reason = "Approved by confidence threshold or co-sign."
            reason_code = "APPROVED"
        else:
            approved = False
            reason = "Insufficient confidence; co-sign required."
            reason_code = "LOW_CONFIDENCE"

        entry = ledger.append(request.model_dump(), approved, reason, reason_code)
        return GateDecision(
            request_id=entry.request_id,
            approved=approved,
            status="approved" if approved else "blocked",
            reason=reason,
            reason_code=reason_code,
            ledger_hash=entry.entry_hash,
            evaluated_at=entry.timestamp,
            policy_version=POLICY_VERSION,
        )


policy = GatePolicy()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "mode": policy.mode, "device": DEVICE, "policy_version": POLICY_VERSION}


@app.post("/gate/request", response_model=GateDecision)
async def gate_request(request: GateRequest) -> GateDecision:
    if not request.tool:
        raise HTTPException(status_code=400, detail="Tool name is required.")
    if request.device is None:
        request = request.model_copy(update={"device": DEVICE})
    return policy.evaluate(request)


@app.post("/gate/mode")
async def set_mode(change: ModeChange) -> Dict[str, str]:
    normalized = change.mode.upper()
    if normalized not in {"PUB", "WORK"}:
        raise HTTPException(status_code=400, detail="Mode must be PUB or WORK.")
    policy.mode = normalized
    ledger.append({"mode": normalized, "reason": change.reason}, True, "Mode updated", "MODE_CHANGE")
    return {"status": "ok", "mode": policy.mode}


@app.get("/ledger/tail")
async def ledger_tail(limit: int = 10) -> List[Dict[str, Any]]:
    entries = ledger.tail(limit=limit)
    return [
        {
            "request_id": entry.request_id,
            "timestamp": entry.timestamp,
            "payload": entry.payload,
            "approved": entry.approved,
            "reason": entry.reason,
            "reason_code": entry.reason_code,
            "prev_hash": entry.prev_hash,
            "entry_hash": entry.entry_hash,
        }
        for entry in entries
    ]


@app.get("/ledger/stats")
async def ledger_stats() -> Dict[str, Any]:
    return {"status": "ok", "count": ledger.count()}


@app.get("/ledger/verify")
async def ledger_verify() -> Dict[str, Any]:
    result = ledger.verify()
    return {"status": "ok" if result["valid"] else "error", **result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
