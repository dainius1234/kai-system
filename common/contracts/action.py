"""Action contracts — proposals, approvals, capabilities, and outcomes."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from common.contracts.base import (
    ApprovalStatus,
    ContractBase,
    RiskTier,
    VerificationVerdict,
)


class ActionProposal(ContractBase):
    """A proposed action that requires assessment and approval."""

    action_type: str
    description: str
    risk_tier: RiskTier
    rationale: str
    alternatives: List[str] = Field(default_factory=list)
    no_action_rationale: Optional[str] = None
    assumptions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    estimated_value: Optional[float] = None
    estimated_risk: Optional[float] = None
    world_state_snapshot_id: Optional[str] = None
    bidder_id: Optional[str] = None


class ConstraintAssessment(ContractBase):
    """Assessment of constraints on a proposed action."""

    proposal_id: str
    constraints_checked: List[str] = Field(default_factory=list)
    violations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    risk_tier_override: Optional[RiskTier] = None
    assessment_passed: bool = True


class PolicyDecision(ContractBase):
    """A policy evaluation result for a proposed action."""

    proposal_id: str
    policy_version: str
    rules_evaluated: List[str] = Field(default_factory=list)
    result: str
    overrides: List[str] = Field(default_factory=list)
    conviction_score: Optional[float] = None
    mode: Optional[str] = None


class ApprovalRecord(ContractBase):
    """Record of a human or system approval decision."""

    proposal_id: str
    status: ApprovalStatus
    approver: str
    risk_tier: RiskTier
    conditions: List[str] = Field(default_factory=list)
    approved_at: Optional[datetime] = None
    expires_at_approval: Optional[datetime] = None
    nonce: Optional[str] = None


class ActionCapability(ContractBase):
    """A one-time capability token authorising a specific action."""

    proposal_id: str
    approval_id: str
    capability_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    risk_tier: RiskTier
    max_retries: int = 0
    timeout_seconds: int = 30
    used: bool = False
    used_at: Optional[datetime] = None


class ActionWorkflow(ContractBase):
    """Tracks the lifecycle of an action from proposal to completion."""

    proposal_id: str
    approval_id: Optional[str] = None
    capability_id: Optional[str] = None
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    current_step: int = 0
    status: str = "pending"
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class ActuatorReceipt(ContractBase):
    """Receipt from an actuator confirming action execution."""

    capability_id: str
    workflow_id: str
    actuator: str
    action_taken: str
    result: Dict[str, Any] = Field(default_factory=dict)
    side_effects: List[str] = Field(default_factory=list)
    reversible: bool = False
    executed_at: datetime


class VerifiedOutcome(ContractBase):
    """Independent verification of an action's outcome."""

    workflow_id: str
    receipt_id: str
    verifier: str
    verdict: VerificationVerdict
    expected_state: Optional[Dict[str, Any]] = None
    actual_state: Optional[Dict[str, Any]] = None
    discrepancies: List[str] = Field(default_factory=list)
    verified_at: datetime


class LearningUpdate(ContractBase):
    """A learning signal derived from a verified outcome."""

    outcome_id: str
    update_type: str
    domain: str
    signal: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class CapabilityReleaseRecord(ContractBase):
    """Record of a capability being released (consumed or revoked)."""

    capability_id: str
    release_type: str
    released_at: datetime
    reason: str
