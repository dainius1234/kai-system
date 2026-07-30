"""Proposal-only workspace — deliberation without execution.

The workspace receives proposals from registered bidders, validates them
against evidence and assumptions, detects contradictions and missing evidence,
and produces deterministic proposal envelopes.

Critically, the workspace:
  - CANNOT issue capabilities
  - CANNOT trigger execution
  - CANNOT import or call actuators
  - produces ActionProposal contracts that must traverse the policy/approval
    chain before any execution occurs

A winning proposal is a recommendation, not an authorization.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from common.contracts.base import (
    ContractState,
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.action import ActionProposal
from common.contracts.world_state import Claim

from common.proposal_workspace.bidder import BidderRegistry, BidderRegistration


class WorkspaceStatus(str, Enum):
    OPEN = "open"
    DEGRADED = "degraded"
    CLOSED = "closed"


class ProposalSubmission(BaseModel):
    model_config = {"extra": "forbid"}

    bidder_id: str
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


class EvidenceGap(BaseModel):
    model_config = {"extra": "forbid"}

    description: str
    referenced_by: List[str] = Field(default_factory=list)
    severity: str = "warning"


class ProposalEnvelope(BaseModel):
    model_config = {"extra": "forbid"}

    proposal: ActionProposal
    submission: ProposalSubmission
    evidence_gaps: List[EvidenceGap] = Field(default_factory=list)
    contradictions: List[Dict[str, Any]] = Field(default_factory=list)
    correlated_bidders: List[str] = Field(default_factory=list)
    qualifying_consensus: bool = False
    workspace_status: WorkspaceStatus = WorkspaceStatus.OPEN
    envelope_digest: str = ""


class ProposalWorkspace:
    """Deliberation workspace — proposal-only, no execution authority.

    Parameters:
        principal: the workspace's operating principal
        registry: bidder registry for authentication
        available_claims: current claims from the world state (for
            evidence validation and contradiction detection)
    """

    def __init__(
        self,
        principal: Principal,
        registry: BidderRegistry,
        available_claims: Optional[List[Claim]] = None,
    ) -> None:
        self._principal = principal
        self._registry = registry
        self._claims: Dict[str, Claim] = {}
        if available_claims:
            for claim in available_claims:
                self._claims[claim.id] = claim
        self._proposals: List[ProposalEnvelope] = []
        self._status = WorkspaceStatus.OPEN

    @property
    def status(self) -> WorkspaceStatus:
        return self._status

    def update_claims(self, claims: List[Claim]) -> None:
        self._claims = {c.id: c for c in claims}

    def submit(self, submission: ProposalSubmission) -> ProposalEnvelope:
        if self._status == WorkspaceStatus.CLOSED:
            raise RuntimeError("workspace is closed — proposals blocked")

        if not self._registry.is_registered(submission.bidder_id):
            raise ValueError(
                f"unregistered bidder: {submission.bidder_id}"
            )

        if not submission.alternatives and submission.no_action_rationale is None:
            raise ValueError(
                "proposal must include alternatives or no_action_rationale"
            )

        evidence_gaps = self._check_evidence(submission)
        contradictions = self._check_contradictions(submission)
        correlated = self._registry.correlated_bidders(submission.bidder_id)

        proposal = ActionProposal(
            action_type=submission.action_type,
            description=submission.description,
            risk_tier=submission.risk_tier,
            rationale=submission.rationale,
            alternatives=submission.alternatives,
            no_action_rationale=submission.no_action_rationale,
            assumptions=submission.assumptions,
            dependencies=submission.dependencies,
            evidence_ids=submission.evidence_ids,
            estimated_value=submission.estimated_value,
            estimated_risk=submission.estimated_risk,
            world_state_snapshot_id=submission.world_state_snapshot_id,
            bidder_id=submission.bidder_id,
            principal=self._principal,
            purpose="proposal",
            provenance=Provenance(
                source=f"workspace:{submission.bidder_id}",
                independence_group=(
                    self._registry.get(submission.bidder_id).independence_group
                    if self._registry.get(submission.bidder_id) else None
                ),
            ),
        )

        envelope = ProposalEnvelope(
            proposal=proposal,
            submission=submission,
            evidence_gaps=evidence_gaps,
            contradictions=contradictions,
            correlated_bidders=correlated,
            qualifying_consensus=self._registry.has_qualifying_diversity(),
            workspace_status=self._status,
            envelope_digest=self._compute_envelope_digest(proposal, submission),
        )

        self._proposals.append(envelope)
        return envelope

    def _check_evidence(self, submission: ProposalSubmission) -> List[EvidenceGap]:
        gaps: List[EvidenceGap] = []

        if not submission.evidence_ids:
            gaps.append(EvidenceGap(
                description="no evidence cited",
                referenced_by=[submission.bidder_id],
                severity="warning",
            ))
            return gaps

        for eid in submission.evidence_ids:
            if eid not in self._claims:
                gaps.append(EvidenceGap(
                    description=f"cited evidence {eid} not found in world state",
                    referenced_by=[submission.bidder_id],
                    severity="error",
                ))

        return gaps

    def _check_contradictions(
        self, submission: ProposalSubmission
    ) -> List[Dict[str, Any]]:
        contradictions: List[Dict[str, Any]] = []

        cited_claims = [
            self._claims[eid] for eid in submission.evidence_ids
            if eid in self._claims
        ]
        for claim in cited_claims:
            if claim.contradicts:
                for contra_id in claim.contradicts:
                    if contra_id in self._claims:
                        contradictions.append({
                            "claim_id": claim.id,
                            "contradicts": contra_id,
                            "claim_text": claim.claim_text[:100],
                            "contra_text": self._claims[contra_id].claim_text[:100],
                        })

        return contradictions

    def _compute_envelope_digest(
        self, proposal: ActionProposal, submission: ProposalSubmission
    ) -> str:
        canonical = json.dumps({
            "bidder_id": submission.bidder_id,
            "action_type": submission.action_type,
            "risk_tier": submission.risk_tier.value,
            "evidence_ids": sorted(submission.evidence_ids),
            "alternatives": sorted(submission.alternatives),
        }, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get_proposals(self) -> List[ProposalEnvelope]:
        return list(self._proposals)

    def close(self) -> None:
        self._status = WorkspaceStatus.CLOSED

    def degrade(self) -> None:
        self._status = WorkspaceStatus.DEGRADED

    def proposal_count(self) -> int:
        return len(self._proposals)
