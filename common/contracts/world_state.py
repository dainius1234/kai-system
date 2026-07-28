"""World state contracts — facts, claims, evidence, and snapshots."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from common.contracts.base import (
    ContractBase,
    ContractState,
    Provenance,
    VerificationVerdict,
)


class FreshnessStatus(str, Enum):
    CURRENT = "current"
    STALE = "stale"
    UNKNOWN = "unknown"
    EXPIRED = "expired"


class Claim(ContractBase):
    """A single factual claim derived from evidence.

    Claims are the atomic units of the world model. Each claim has a
    verification status and links back to the evidence that supports or
    contradicts it.
    """

    claim_text: str
    domain: str
    evidence_ids: List[str] = Field(default_factory=list)
    verification: VerificationVerdict = VerificationVerdict.INCONCLUSIVE
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    contradicts: List[str] = Field(default_factory=list)
    supersedes: Optional[str] = None
    freshness: FreshnessStatus = FreshnessStatus.UNKNOWN


class EvidenceRecord(ContractBase):
    """A piece of evidence supporting or contradicting a claim."""

    content: str
    evidence_type: str
    source_event_id: Optional[str] = None
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    direction: str = "supports"
    claim_ids: List[str] = Field(default_factory=list)
    raw_data: Optional[Dict[str, Any]] = None
    freshness: FreshnessStatus = FreshnessStatus.UNKNOWN


class WorldStateSnapshot(ContractBase):
    """An immutable point-in-time view of the world model.

    Snapshots are scoped to a principal, purpose, and data classification.
    They are reproducible given the same event sequence.
    """

    snapshot_at: datetime
    scope_principal: str
    scope_purpose: str
    scope_classification: str = "internal"
    claims: List[Claim] = Field(default_factory=list)
    evidence: List[EvidenceRecord] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    degraded_sources: List[str] = Field(default_factory=list)
    event_sequence_digest: Optional[str] = None
