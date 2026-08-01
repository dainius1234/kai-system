"""Autonomy contracts — evidence grading, scoped authority, calibration.

UH-8 replaces the Trust Ledger's single scalar trust level with scoped,
bounded, expiring and revocable autonomy grants, each backed by evidence
that is graded on whether it can grant trust at all.

The central rule: self-generated text and simulation cannot grant trust.
``EvidenceGrade.qualifies()`` is the single place that decides, and every
grant path routes through it.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

from pydantic import Field

from common.contracts.base import ContractBase, RiskTier


class EvidenceGrade(str, Enum):
    """How evidence was obtained — decides whether it can grant trust."""

    EXTERNAL_OBSERVED = "external_observed"
    VERIFIED_OUTCOME = "verified_outcome"
    HUMAN_CONFIRMED = "human_confirmed"
    MODEL_GENERATED = "model_generated"
    SIMULATED = "simulated"
    UNKNOWN = "unknown"

    def qualifies(self) -> bool:
        """Whether evidence of this grade may contribute to trust.

        Self-generated text (MODEL_GENERATED) and simulation output
        (SIMULATED) never qualify, regardless of volume or confidence.
        """
        return self in (
            EvidenceGrade.EXTERNAL_OBSERVED,
            EvidenceGrade.VERIFIED_OUTCOME,
            EvidenceGrade.HUMAN_CONFIRMED,
        )


class AutonomyLevel(IntEnum):
    """Scoped autonomy authority. Higher levels require stronger evidence."""

    A0_NONE = 0
    A1_OBSERVE = 1
    A2_REVERSIBLE = 2
    A3_SUPERVISED = 3
    A4_HIGH_CONSEQUENCE = 4


# Minimum qualifying-outcome count and calibration accuracy required to
# hold each level.  A0 is the floor and needs nothing.
AUTONOMY_REQUIREMENTS: Dict[AutonomyLevel, Dict[str, Any]] = {
    AutonomyLevel.A0_NONE: {
        "min_qualifying_outcomes": 0,
        "min_accuracy": 0.0,
        "min_independent_verifiers": 0,
        "max_grant_seconds": 0,
        "requires_human_confirmation": False,
    },
    AutonomyLevel.A1_OBSERVE: {
        "min_qualifying_outcomes": 10,
        "min_accuracy": 0.70,
        "min_independent_verifiers": 1,
        "max_grant_seconds": 86400,
        "requires_human_confirmation": False,
    },
    AutonomyLevel.A2_REVERSIBLE: {
        "min_qualifying_outcomes": 25,
        "min_accuracy": 0.80,
        "min_independent_verifiers": 1,
        "max_grant_seconds": 43200,
        "requires_human_confirmation": False,
    },
    AutonomyLevel.A3_SUPERVISED: {
        "min_qualifying_outcomes": 50,
        "min_accuracy": 0.90,
        "min_independent_verifiers": 2,
        "max_grant_seconds": 14400,
        "requires_human_confirmation": True,
    },
    AutonomyLevel.A4_HIGH_CONSEQUENCE: {
        "min_qualifying_outcomes": 100,
        "min_accuracy": 0.95,
        "min_independent_verifiers": 3,
        "max_grant_seconds": 3600,
        "requires_human_confirmation": True,
    },
}


class GradedEvidence(ContractBase):
    """An immutable piece of evidence with an explicit grade.

    Stored append-only.  The grade is assigned at write time from the
    source's nature, never inferred later from content.
    """

    grade: EvidenceGrade
    claim_id: Optional[str] = None
    outcome_id: Optional[str] = None
    domain: str
    task_type: str
    content: Dict[str, Any] = Field(default_factory=dict)
    observed_by: str
    supersedes: Optional[str] = None


class VerifierRegistration(ContractBase):
    """A registered outcome verifier.

    ``independence_group`` must differ from the actuator's group for a
    verification to count — an executor cannot verify its own success.
    """

    verifier_identity: str
    display_name: str
    domains: List[str] = Field(default_factory=list)
    independence_group: str
    active: bool = True
    suspended_reason: Optional[str] = None


class CalibrationRecord(ContractBase):
    """Calibration for one (task_type, domain, revision) triple."""

    task_type: str
    domain: str
    revision: str
    total_predictions: int = 0
    correct_predictions: int = 0
    qualifying_outcomes: int = 0
    rejected_non_qualifying: int = 0
    brier_sum: float = 0.0
    accuracy: float = 0.0
    brier_score: float = 1.0


class AutonomyGrant(ContractBase):
    """A scoped, bounded, expiring and revocable autonomy authorisation.

    A grant authorises exactly one capability in exactly one domain, for
    a bounded number of invocations, until an explicit expiry.  It can be
    revoked at any moment and never renews itself.
    """

    level: AutonomyLevel
    capability: str
    domain: str
    granted_by: str
    granted_at: datetime
    expires_at_grant: datetime
    max_invocations: int
    invocations_used: int = 0
    evidence_ids: List[str] = Field(default_factory=list)
    calibration_id: Optional[str] = None
    revoked: bool = False
    revoked_at: Optional[datetime] = None
    revoked_reason: Optional[str] = None
    human_confirmation_id: Optional[str] = None


class ValueConfirmation(ContractBase):
    """Explicit human confirmation of a value-laden decision.

    Cannot be inferred from normal chat — requires an explicit prompt and
    an explicit response bound to a specific proposal digest.
    """

    subject_digest: str
    subject_kind: str
    prompt_shown: str
    confirmed: bool
    confirmed_by: str
    confirmed_at: datetime
    nonce: str


class ReleaseBundle(ContractBase):
    """A signed, capability-specific release authorisation.

    Binds a capability to a code revision and an autonomy level.  The
    signature covers the bundle's payload; tampering with any field
    invalidates it.
    """

    capability: str
    code_revision: str
    autonomy_level: AutonomyLevel
    domains: List[str] = Field(default_factory=list)
    signature: str
    signed_by: str
    signed_at: datetime
    valid_until: datetime
    revoked: bool = False


class WisdomNode(ContractBase):
    """A node in the wisdom graph with explicit lineage."""

    statement: str
    domain: str
    derived_from: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    contradicts: List[str] = Field(default_factory=list)
    superseded_by: Optional[str] = None
