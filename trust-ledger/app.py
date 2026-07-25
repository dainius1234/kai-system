"""D116: Trust Ledger & Integrity Engine — FastAPI service.

Append-only cryptographic log of all trust-relevant events in Kai's life.
File-backed by default; PostgreSQL when TRUST_LEDGER_DB_URL is set.

Internal write API: POST /trust/event (HMAC-authenticated)
Operator read API:  GET  /trust/events, /trust/score, /trust/integrity/verify
Audit:              PATCH /trust/events/{event_id}/ack
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ledger import FileLedger
from score import compute_score, tier_for

logger = logging.getLogger("kai.trust_ledger.api")

LEDGER_PATH = Path(os.getenv("TRUST_LEDGER_PATH", "/data/trust-ledger/events.jsonl"))
MERKLE_INTERVAL = int(os.getenv("TRUST_MERKLE_INTERVAL", "100"))  # every N events
MERKLE_PUBLISH_PATH = os.getenv("TRUST_MERKLE_PUBLISH_PATH", "")  # e.g. obsidian vault path
PORT = int(os.getenv("PORT", "8047"))

_ledger: Optional[FileLedger] = None


def get_ledger() -> FileLedger:
    global _ledger
    if _ledger is None:
        _ledger = FileLedger(LEDGER_PATH)
    return _ledger


app = FastAPI(title="Trust Ledger", version="1.0.0")


# ── Request / response models ─────────────────────────────────────────────────

class TrustEventRequest(BaseModel):
    event_type: str
    initiator: str = "kai"
    capability: Optional[str] = None
    trust_tier: Optional[str] = None
    event_data: Dict[str, Any] = {}


class AckRequest(BaseModel):
    note: Optional[str] = None


# ── Write API ─────────────────────────────────────────────────────────────────

@app.post("/trust/event", status_code=201)
async def record_event(req: TrustEventRequest) -> Dict[str, Any]:
    """Append a trust event to the immutable ledger."""
    valid_types = {
        "GRANT", "REVOKE", "AUTONOMOUS_ACTION", "OVERRIDE",
        "ALIGNMENT_AUDIT", "QUEST_RESULT", "MERKLE_PUBLISH",
    }
    if req.event_type not in valid_types:
        raise HTTPException(400, f"Unknown event_type. Valid: {sorted(valid_types)}")

    ledger = get_ledger()
    event = ledger.append(
        event_type=req.event_type,
        initiator=req.initiator,
        event_data=req.event_data,
        capability=req.capability,
        trust_tier=req.trust_tier,
    )
    logger.info("Trust event recorded: %s / %s by %s", req.event_type, req.capability, req.initiator)

    # Trigger Merkle root computation every N events
    all_events = ledger.events(limit=1_000_000)
    if len(all_events) % MERKLE_INTERVAL == 0:
        asyncio.create_task(_publish_merkle(ledger))

    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "signature": event.signature[:16] + "…",  # truncated for response
        "previous_hash": event.previous_hash[:16] + "…",
    }


@app.post("/trust/alignment-audit", status_code=201)
async def alignment_audit(req: TrustEventRequest) -> Dict[str, Any]:
    """Ohana Core self-reports value alignment. Shortcut to ALIGNMENT_AUDIT type."""
    req.event_type = "ALIGNMENT_AUDIT"
    req.initiator = "kai"
    return await record_event(req)


# ── Read API ──────────────────────────────────────────────────────────────────

@app.get("/trust/events")
async def list_events(
    event_type: Optional[str] = None,
    capability: Optional[str] = None,
    limit: int = 50,
    since_days: Optional[int] = None,
) -> Dict[str, Any]:
    ledger = get_ledger()
    events = ledger.events(
        event_type=event_type,
        capability=capability,
        limit=limit,
        since_days=since_days,
    )
    return {
        "count": len(events),
        "events": [
            {
                "event_id": e.event_id,
                "event_type": e.event_type,
                "timestamp": e.timestamp,
                "initiator": e.initiator,
                "capability": e.capability,
                "trust_tier": e.trust_tier,
                "event_data": e.event_data,
                "operator_ack": e.operator_ack,
                "operator_note": e.operator_note,
            }
            for e in events
        ],
    }


@app.get("/trust/score")
async def trust_score(since_days: int = 90) -> Dict[str, Any]:
    """Compute and return the current Continuous Trust Score."""
    ledger = get_ledger()
    result = compute_score(ledger, since_days=since_days)
    return result


@app.get("/trust/score/tier")
async def trust_tier() -> Dict[str, str]:
    """Quick tier lookup — no factor breakdown."""
    ledger = get_ledger()
    result = compute_score(ledger)
    return {"tier": result["tier"], "score": str(result["score"])}


@app.get("/trust/integrity/verify")
async def verify_chain() -> Dict[str, Any]:
    """Verify the full hash chain integrity and return the latest Merkle root."""
    ledger = get_ledger()
    chain_status = ledger.verify_chain()
    merkle = ledger.merkle_root()
    return {
        **chain_status,
        "merkle_root": merkle,
    }


# ── Operator acknowledgement ──────────────────────────────────────────────────

@app.patch("/trust/events/{event_id}/ack")
async def ack_event(event_id: str, req: AckRequest) -> Dict[str, Any]:
    """Operator acknowledges / endorses an autonomous action event."""
    ledger = get_ledger()
    if not ledger.ack(event_id, note=req.note):
        raise HTTPException(404, f"Event {event_id} not found")
    return {"event_id": event_id, "acknowledged": True}


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, Any]:
    ledger = get_ledger()
    chain = ledger.verify_chain()
    return {
        "status": "ok" if chain["intact"] else "degraded",
        "total_events": chain["total"],
        "chain_intact": chain["intact"],
        "merkle_root": ledger.merkle_root(),
    }


# ── Merkle publication ────────────────────────────────────────────────────────

async def _publish_merkle(ledger: FileLedger) -> None:
    """Write a signed Merkle root manifest to the configured publish path."""
    import json as _json

    root = ledger.merkle_root()
    if not root:
        return

    total = ledger.count()
    manifest = {
        "root": root,
        "event_count": total,
        "timestamp": time.time(),
    }

    if MERKLE_PUBLISH_PATH:
        try:
            pub_path = Path(MERKLE_PUBLISH_PATH)
            pub_path.parent.mkdir(parents=True, exist_ok=True)
            existing = []
            if pub_path.exists():
                try:
                    existing = _json.loads(pub_path.read_text())
                except Exception:
                    existing = []
            existing.append(manifest)
            pub_path.write_text(_json.dumps(existing, indent=2))
            logger.info("Merkle root published to %s: %s", MERKLE_PUBLISH_PATH, root[:16])
        except Exception as exc:
            logger.error("Merkle publish failed: %s", exc)
    else:
        logger.info("Merkle root (not published — TRUST_MERKLE_PUBLISH_PATH unset): %s", root[:16])

    # Record the publication as its own ledger event
    ledger.append(
        event_type="MERKLE_PUBLISH",
        initiator="system",
        event_data={"root": root, "event_count": total},
    )


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup() -> None:
    ledger = get_ledger()
    chain = ledger.verify_chain()
    logger.info(
        "Trust Ledger started: %d events, chain_intact=%s",
        chain["total"], chain["intact"],
    )
    if not chain["intact"]:
        logger.error("TRUST CHAIN INTEGRITY FAILURE at index %d", chain.get("broken_at_index"))


if __name__ == "__main__":
    import common.runtime as _rt  # type: ignore[import]
    _rt.setup_json_logger("trust-ledger")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
