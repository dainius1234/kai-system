"""Base contract types and shared infrastructure for canonical schemas.

Design rules:
  - All contracts extend ContractBase with common metadata fields.
  - Schema version is embedded in every serialised instance.
  - Content digests use SHA-256 over canonicalised JSON (sorted keys, no
    whitespace, UTF-8).
  - Executable control fields use ``model_config = {"extra": "forbid"}``
    so unrecognised extras are rejected at parse time.
  - Narrative text fields (str) are never parsed as hidden control
    authority — they are opaque payloads.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


# ── Enums ───────────────────────────────────────────────────────────────

class RiskTier(str, Enum):
    OBSERVE = "observe"
    ADVISE = "advise"
    PROPOSE = "propose"
    ACT_SUPERVISED = "act_supervised"
    ACT_AUTONOMOUS = "act_autonomous"


class ContractState(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    REVOKED = "revoked"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    CONFLICT = "conflict"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    ESCALATED = "escalated"


class VerificationVerdict(str, Enum):
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    INCONCLUSIVE = "inconclusive"
    UNAVAILABLE = "unavailable"


# ── Shared sub-models ──────────────────────────────────────────────────

class Principal(BaseModel):
    model_config = {"extra": "forbid"}

    identity: str
    role: str
    delegation_chain: List[str] = Field(default_factory=list)


class Provenance(BaseModel):
    model_config = {"extra": "forbid"}

    source: str
    transformation: Optional[str] = None
    upstream_ids: List[str] = Field(default_factory=list)
    independence_group: Optional[str] = None


class ContractDigest(BaseModel):
    model_config = {"extra": "forbid"}

    algorithm: str = "sha256"
    value: str


# ── Base contract ──────────────────────────────────────────────────────

class ContractBase(BaseModel):
    model_config = {"extra": "forbid"}

    schema_version: str = "1.0.0"
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    principal: Principal
    purpose: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    observed_at: Optional[datetime] = None
    received_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    classification: str = "internal"
    revision: int = 1
    state: ContractState = ContractState.ACTIVE
    provenance: Provenance
    digest: Optional[ContractDigest] = None

    @model_validator(mode="after")
    def _compute_digest(self) -> "ContractBase":
        if self.digest is None:
            self.digest = self._make_digest()
        return self

    def _make_digest(self) -> ContractDigest:
        data = self.model_dump(exclude={"digest"})
        canonical = _canonical_json(data)
        h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return ContractDigest(algorithm="sha256", value=h)

    def verify_digest(self) -> bool:
        expected = self._make_digest()
        return self.digest is not None and self.digest.value == expected.value


# ── Canonical serialisation ────────────────────────────────────────────

def _canonical_json(obj: Any) -> str:
    return json.dumps(
        _prepare(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _prepare(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _prepare(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_prepare(v) for v in obj]
    return obj


# ── Approval matrix ───────────────────────────────────────────────────

APPROVAL_MATRIX: Dict[RiskTier, Dict[str, Any]] = {
    RiskTier.OBSERVE: {
        "requires_human_approval": False,
        "auto_approve": True,
        "max_value_usd": None,
        "cooldown_seconds": 0,
    },
    RiskTier.ADVISE: {
        "requires_human_approval": False,
        "auto_approve": True,
        "max_value_usd": None,
        "cooldown_seconds": 0,
    },
    RiskTier.PROPOSE: {
        "requires_human_approval": False,
        "auto_approve": False,
        "max_value_usd": 0,
        "cooldown_seconds": 60,
    },
    RiskTier.ACT_SUPERVISED: {
        "requires_human_approval": True,
        "auto_approve": False,
        "max_value_usd": 100,
        "cooldown_seconds": 300,
    },
    RiskTier.ACT_AUTONOMOUS: {
        "requires_human_approval": True,
        "auto_approve": False,
        "max_value_usd": 50,
        "cooldown_seconds": 600,
    },
}
