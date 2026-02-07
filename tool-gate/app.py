from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


app = FastAPI(title="Tool Gate", version="0.1.0")


class GateRequest(BaseModel):
    tool: str
    params: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)
    actor_did: str
    cosign: bool = False
    rationale: Optional[str] = None


class GateDecision(BaseModel):
    request_id: str
    approved: bool
    status: str
    reason: str
    ledger_hash: str
    required_cosign: bool


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
        entry_data = {
            "request_id": request_id,
            "timestamp": time.time(),
            "payload": payload,
            "approved": approved,
            "reason": reason,
            "prev_hash": prev_hash,
        }
        entry_hash = hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()
        entry = LedgerEntry(
            request_id=request_id,
            timestamp=entry_data["timestamp"],
            payload=payload,
            approved=approved,
            reason=reason,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )
        self._entries.append(entry)
        return entry

    def tail(self, limit: int = 10) -> List[LedgerEntry]:
        return self._entries[-limit:]


ledger = InMemoryLedger()


class GatePolicy:
    def __init__(self) -> None:
        self.mode = os.getenv("MODE", "PUB").upper()
        self.required_confidence = float(os.getenv("REQUIRED_CONFIDENCE", "0.7"))
        self.high_risk_tools: Set[str] = {
            tool.strip()
            for tool in os.getenv("HIGH_RISK_TOOLS", "shell,n8n,qgis").split(",")
            if tool.strip()
        }

    def evaluate(self, request: GateRequest) -> GateDecision:
        requires_cosign = request.tool in self.high_risk_tools
        if self.mode == "PUB":
            approved = False
            reason = "Tool Gate in PUB mode (execution disabled)."
        elif requires_cosign and not request.cosign:
            approved = False
            reason = "High-risk tool requires co-sign."
        elif request.confidence >= self.required_confidence or request.cosign:
            approved = True
            reason = "Approved by confidence threshold or co-sign."
        else:
            approved = False
            reason = "Insufficient confidence; co-sign required."

        entry = ledger.append(request.model_dump(), approved, reason)
        return GateDecision(
            request_id=entry.request_id,
            approved=approved,
            status="approved" if approved else "blocked",
            reason=reason,
            ledger_hash=entry.entry_hash,
            required_cosign=requires_cosign,
        )


policy = GatePolicy()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "mode": policy.mode}


@app.post("/gate/request", response_model=GateDecision)
async def gate_request(request: GateRequest) -> GateDecision:
    if not request.tool:
        raise HTTPException(status_code=400, detail="Tool name is required.")
    return policy.evaluate(request)


@app.post("/gate/mode")
async def set_mode(change: ModeChange) -> Dict[str, str]:
    normalized = change.mode.upper()
    if normalized not in {"PUB", "WORK"}:
        raise HTTPException(status_code=400, detail="Mode must be PUB or WORK.")
    policy.mode = normalized
    ledger.append({"mode": normalized, "reason": change.reason}, True, "Mode updated")
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
            "prev_hash": entry.prev_hash,
            "entry_hash": entry.entry_hash,
        }
        for entry in entries
    ]


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
