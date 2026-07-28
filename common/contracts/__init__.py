"""Canonical contracts for the Kai system (UH-1).

All inter-service data exchange uses these versioned schemas.
Executable control fields reject unrecognised extras.
Narrative text fields are never parsed as hidden control authority.
"""

from common.contracts.base import (
    ContractBase,
    ContractDigest,
    RiskTier,
    ContractState,
    Principal,
    Provenance,
)
from common.contracts.perception import PerceptionEvent
from common.contracts.world_state import WorldStateSnapshot, Claim, EvidenceRecord
from common.contracts.action import (
    ActionProposal,
    ConstraintAssessment,
    PolicyDecision,
    ApprovalRecord,
    ActionCapability,
    ActionWorkflow,
    ActuatorReceipt,
    VerifiedOutcome,
    LearningUpdate,
    CapabilityReleaseRecord,
)

SCHEMA_VERSION = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "ContractBase",
    "ContractDigest",
    "RiskTier",
    "ContractState",
    "Principal",
    "Provenance",
    "PerceptionEvent",
    "WorldStateSnapshot",
    "Claim",
    "EvidenceRecord",
    "ActionProposal",
    "ConstraintAssessment",
    "PolicyDecision",
    "ApprovalRecord",
    "ActionCapability",
    "ActionWorkflow",
    "ActuatorReceipt",
    "VerifiedOutcome",
    "LearningUpdate",
    "CapabilityReleaseRecord",
]
