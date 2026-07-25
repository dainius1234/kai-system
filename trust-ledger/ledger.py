"""Trust Ledger — append-only cryptographic hash chain.

Two storage backends share the same interface:
    FileLedger   — JSONL file, no external deps (default for local dev / tests)
    PostgresLedger — PostgreSQL-backed (production)

Both produce identical signatures and chain hashes. The backend is selected
by the TRUST_LEDGER_DB_URL env var: if set, PostgresLedger is used.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("kai.trust_ledger")

GENESIS = "GENESIS"
_HMAC_KEY = os.getenv("TRUST_LEDGER_HMAC_SECRET", "trust-dev-secret").encode()


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class TrustEvent:
    event_id: str
    event_type: str       # GRANT | REVOKE | AUTONOMOUS_ACTION | OVERRIDE |
                          # ALIGNMENT_AUDIT | QUEST_RESULT | MERKLE_PUBLISH
    timestamp: float
    initiator: str        # 'operator' | 'kai' | 'system'
    capability: Optional[str]
    trust_tier: Optional[str]
    event_data: Dict[str, Any]
    signature: str
    previous_hash: str
    operator_ack: bool = False
    operator_note: Optional[str] = None


# ── Cryptographic helpers ─────────────────────────────────────────────────────

def _sign(event_id: str, timestamp: float, event_type: str,
          initiator: str, event_data: Dict[str, Any]) -> str:
    """HMAC-SHA512 signature over core event fields."""
    payload = f"{event_id}|{timestamp}|{event_type}|{initiator}|{json.dumps(event_data, sort_keys=True)}"
    return hmac.new(_HMAC_KEY, payload.encode(), hashlib.sha512).hexdigest()


def _chain_hash(prev_signature: str) -> str:
    """SHA256 of the previous event's signature — forms the chain link."""
    return hashlib.sha256(prev_signature.encode()).hexdigest()


def verify_event(event: TrustEvent, prev_signature: Optional[str]) -> bool:
    """Return True if the event's signature and chain hash are valid."""
    expected_sig = _sign(
        event.event_id, event.timestamp, event.event_type,
        event.initiator, event.event_data,
    )
    if not hmac.compare_digest(event.signature, expected_sig):
        return False
    if prev_signature is None:
        return event.previous_hash == GENESIS
    expected_chain = _chain_hash(prev_signature)
    return hmac.compare_digest(event.previous_hash, expected_chain)


# ── Merkle tree ───────────────────────────────────────────────────────────────

def build_merkle_root(leaves: List[str]) -> str:
    """Build a Merkle tree from a list of leaf hashes and return the root."""
    if not leaves:
        return hashlib.sha256(b"EMPTY").hexdigest()
    nodes = list(leaves)
    while len(nodes) > 1:
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])  # duplicate last leaf if odd count
        nodes = [
            hashlib.sha256((nodes[i] + nodes[i + 1]).encode()).hexdigest()
            for i in range(0, len(nodes), 2)
        ]
    return nodes[0]


# ── File-backed ledger ────────────────────────────────────────────────────────

class FileLedger:
    """JSONL append-only ledger for local dev and tests."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[TrustEvent] = []
        self._replay()

    def _replay(self) -> None:
        if not self._path.exists():
            return
        prev_sig: Optional[str] = None
        for lineno, raw in enumerate(self._path.read_text().splitlines(), 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
                ev = TrustEvent(**obj)
            except Exception:
                logger.warning("Ledger line %d: corrupt — skipped", lineno)
                continue
            if not verify_event(ev, prev_sig):
                logger.warning("Ledger line %d: integrity mismatch — skipped", lineno)
                continue
            self._events.append(ev)
            prev_sig = ev.signature
        logger.info("Ledger replayed: %d events", len(self._events))

    def _persist(self, event: TrustEvent) -> None:
        with self._path.open("a") as fh:
            fh.write(json.dumps(asdict(event)) + "\n")

    def append(
        self,
        event_type: str,
        initiator: str,
        event_data: Dict[str, Any],
        capability: Optional[str] = None,
        trust_tier: Optional[str] = None,
    ) -> TrustEvent:
        prev = self._events[-1] if self._events else None
        prev_sig = prev.signature if prev else None
        prev_hash = _chain_hash(prev_sig) if prev_sig else GENESIS

        event_id = str(uuid.uuid4())
        ts = time.time()
        sig = _sign(event_id, ts, event_type, initiator, event_data)

        ev = TrustEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=ts,
            initiator=initiator,
            capability=capability,
            trust_tier=trust_tier,
            event_data=event_data,
            signature=sig,
            previous_hash=prev_hash,
        )
        self._events.append(ev)
        self._persist(ev)
        return ev

    def events(
        self,
        event_type: Optional[str] = None,
        capability: Optional[str] = None,
        limit: int = 50,
        since_days: Optional[int] = None,
    ) -> List[TrustEvent]:
        cutoff = (time.time() - since_days * 86400) if since_days else 0
        result = [
            ev for ev in self._events
            if (event_type is None or ev.event_type == event_type)
            and (capability is None or ev.capability == capability)
            and ev.timestamp >= cutoff
        ]
        return result[-limit:]

    def verify_chain(self) -> Dict[str, Any]:
        ok = 0
        broken_at: Optional[int] = None
        prev_sig: Optional[str] = None
        for i, ev in enumerate(self._events):
            if verify_event(ev, prev_sig):
                ok += 1
                prev_sig = ev.signature
            else:
                broken_at = i
                break
        return {
            "total": len(self._events),
            "verified": ok,
            "intact": broken_at is None,
            "broken_at_index": broken_at,
        }

    def merkle_root(self, batch_size: int = 100) -> Optional[str]:
        if not self._events:
            return None
        leaves = [ev.signature for ev in self._events[-batch_size:]]
        return build_merkle_root(leaves)

    def ack(self, event_id: str, note: Optional[str] = None) -> bool:
        for ev in self._events:
            if ev.event_id == event_id:
                ev.operator_ack = True
                ev.operator_note = note
                return True
        return False

    def avg_field(self, field: str, event_type: str, since_days: int = 90) -> float:
        evs = self.events(event_type=event_type, since_days=since_days, limit=10000)
        vals = [ev.event_data.get(field) for ev in evs if ev.event_data.get(field) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def count(self, event_type: Optional[str] = None, since_days: int = 90,
              operator_ack: Optional[bool] = None) -> int:
        evs = self.events(event_type=event_type, since_days=since_days, limit=100000)
        if operator_ack is not None:
            evs = [e for e in evs if e.operator_ack == operator_ack]
        return len(evs)
