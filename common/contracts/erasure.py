"""Erasure contracts — subject-scoped deletion with surviving audit.

Deletion has two requirements that pull against each other: the subject's
data must genuinely go, and the audit trail must survive to prove what
happened.  The resolution is a **tombstone** — the audit reference is
kept with the payload redacted, so the record's existence and lineage
remain provable while its content does not.

A tombstone therefore carries no subject content by construction: it has
a digest of what was removed, never the removed data itself.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

from pydantic import Field

from common.contracts.base import ContractBase


class ErasureLayer(str, Enum):
    SOURCE_EVENTS = "source_events"
    WORLD_STATE = "world_state"
    PROPOSALS = "proposals"
    AUDIT_REFERENCES = "audit_references"
    LEARNING_DERIVATIVES = "learning_derivatives"


class ErasureStatus(str, Enum):
    PENDING = "pending"
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ErasureRequest(ContractBase):
    """A request to erase everything belonging to one subject."""

    subject_identity: str
    reason: str
    requested_by: str
    layers: List[ErasureLayer] = Field(default_factory=list)


class Tombstone(ContractBase):
    """Proof that a record existed and was erased.

    Deliberately carries ``content_digest`` rather than content: the
    tombstone must not become a backdoor copy of the deleted data.
    """

    layer: ErasureLayer
    original_id: str
    content_digest: str
    erased_at: datetime
    erasure_request_id: str


class LayerResult(ContractBase):
    """Outcome of erasing one layer."""

    layer: ErasureLayer
    records_examined: int = 0
    records_erased: int = 0
    tombstones_created: int = 0
    residue_found: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class ErasureReceipt(ContractBase):
    """The verified outcome of an erasure request."""

    request_id: str
    subject_identity: str
    status: ErasureStatus
    layer_results: List[LayerResult] = Field(default_factory=list)
    total_erased: int = 0
    total_tombstones: int = 0
    verified: bool = False
    verification_residue: List[str] = Field(default_factory=list)
    completed_at: Optional[datetime] = None
