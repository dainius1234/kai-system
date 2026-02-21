from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import time
from collections import deque
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import shutil
import subprocess
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


LOG_PATH = os.getenv("LOG_PATH", "/tmp/tool-gate.json.log")
handler = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=30)
handler.setFormatter(logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","service":"%(name)s","msg":"%(message)s"}'))
logger = logging.getLogger("tool-gate")
logger.setLevel(logging.INFO)
logger.handlers = [handler]
logger.propagate = False

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Tool Gate", version="0.3.0")

TOKENS_PATH = Path(os.getenv("TRUSTED_TOKENS_PATH", "/config/trusted_tokens.txt"))
TRUSTED_TOKENS: Set[str] = set()
SERVICE_CONTAINER = os.getenv("SERVICE_CONTAINER", "sovereign-tool-gate")
ERROR_WINDOW_SECONDS = 300
_metrics: Deque[Tuple[float, int]] = deque()


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

    def count(self) -> int:
        return len(self._entries)


ledger = InMemoryLedger()


def sanitize_string(value: str) -> str:
    sanitized = re.sub(r"[;|&]", "", value)
    return sanitized[:1024]


def _prune_metrics(now: float) -> None:
    while _metrics and now - _metrics[0][0] > ERROR_WINDOW_SECONDS:
        _metrics.popleft()


def _record_status(code: int) -> None:
    now = time.time()
    _metrics.append((now, code))
    _prune_metrics(now)


def _error_budget() -> Dict[str, float]:
    now = time.time()
    _prune_metrics(now)
    total = len(_metrics)
    if total == 0:
        return {"error_ratio": 0.0, "total": 0}
    errors = sum(1 for _, code in _metrics if code in {429, 500, 408})
    return {"error_ratio": errors / total, "total": total}


def _maybe_restart() -> None:
    budget = _error_budget()
    if budget["total"] < 10:
        return
    if budget["error_ratio"] > 0.03 and shutil.which("docker"):
        subprocess.run(["docker", "restart", SERVICE_CONTAINER], check=False)


def load_trusted_tokens() -> None:
    global TRUSTED_TOKENS
    if TOKENS_PATH.exists():
        TRUSTED_TOKENS = {
            token.strip() for token in TOKENS_PATH.read_text(encoding="utf-8").splitlines() if token.strip()
        }
    else:
        TRUSTED_TOKENS = set()


def _reload_tokens(_signum: int, _frame: Any) -> None:
    load_trusted_tokens()


class GatePolicy:
    def __init__(self) -> None:
        self.mode = os.getenv("MODE", "PUB").upper()
        self.required_confidence = float(os.getenv("REQUIRED_CONFIDENCE", "0.7"))

    def evaluate(self, request: GateRequest) -> GateDecision:
        if self.mode == "PUB":
            approved = False
            reason = "Tool Gate in PUB mode (execution disabled)."
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
        )


policy = GatePolicy()
load_trusted_tokens()
signal.signal(signal.SIGHUP, _reload_tokens)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        _record_status(response.status_code)
        _maybe_restart()
        return response
    except Exception:
        _record_status(500)
        _maybe_restart()
        raise


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "mode": policy.mode,
        "device": DEVICE,
        "trusted_tokens": str(len(TRUSTED_TOKENS)),
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return _error_budget()


@app.post("/gate/request", response_model=GateDecision)
async def gate_request(request: GateRequest) -> GateDecision:
    request = request.model_copy(
        update={
            "tool": sanitize_string(request.tool),
            "actor_did": sanitize_string(request.actor_did),
            "session_id": sanitize_string(request.session_id),
            "rationale": sanitize_string(request.rationale) if request.rationale else None,
        }
    )
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


@app.get("/ledger/stats")
async def ledger_stats() -> Dict[str, Any]:
    return {"status": "ok", "count": ledger.count()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
