"""Policy-as-code engine — evaluates proposals against risk classification.

The engine:
  - classifies proposal risk tier
  - evaluates policy rules
  - produces typed PolicyDecision contracts
  - fails closed on outage (unavailable → deny)
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from common.contracts.base import (
    APPROVAL_MATRIX,
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.action import ActionProposal, PolicyDecision

POLICY_VERSION = "1.0.0"


class PolicyEvaluation:
    __slots__ = ("decision", "rules_evaluated", "reason")

    def __init__(
        self,
        decision: PolicyDecision,
        rules_evaluated: List[str],
        reason: str = "",
    ):
        self.decision = decision
        self.rules_evaluated = rules_evaluated
        self.reason = reason


class PolicyEngine:
    """Evaluates proposals against risk classification and approval matrix.

    Fails closed: if the engine is unavailable or encounters an error,
    the result is deny.
    """

    def __init__(
        self,
        principal: Principal,
        policy_version: str = POLICY_VERSION,
        assessors: Optional["AssessorRegistry"] = None,
    ) -> None:
        self._principal = principal
        self._version = policy_version
        self._available = True
        self._assessors = assessors

    def set_available(self, available: bool) -> None:
        self._available = available

    @property
    def available(self) -> bool:
        return self._available

    def evaluate(self, proposal: ActionProposal) -> PolicyEvaluation:
        rules: List[str] = []

        if not self._available:
            decision = self._make_decision(
                proposal, "deny", rules=["engine_unavailable"],
            )
            return PolicyEvaluation(
                decision=decision,
                rules_evaluated=["engine_unavailable"],
                reason="policy engine unavailable — fail closed",
            )

        rules.append("risk_tier_classification")
        tier = proposal.risk_tier
        matrix_entry = APPROVAL_MATRIX.get(tier)
        if matrix_entry is None:
            decision = self._make_decision(
                proposal, "deny", rules=["unknown_risk_tier"],
            )
            return PolicyEvaluation(
                decision=decision,
                rules_evaluated=["unknown_risk_tier"],
                reason=f"unknown risk tier: {tier}",
            )

        rules.append("approval_matrix_check")

        if matrix_entry["auto_approve"]:
            result = "allow"
            rules.append("auto_approved")
        elif matrix_entry["requires_human_approval"]:
            result = "requires_approval"
            rules.append("requires_human_approval")
        else:
            result = "allow_advisory"
            rules.append("advisory_tier")

        rules.append("value_limit_check")
        max_val = matrix_entry.get("max_value_usd")
        if max_val is not None and proposal.estimated_value is not None:
            if proposal.estimated_value > max_val:
                result = "deny"
                rules.append("value_exceeds_limit")

        rules.append("digest_binding_check")
        if proposal.digest is None or not proposal.verify_digest():
            result = "deny"
            rules.append("invalid_digest")

        # Constraint assessments are consulted last and can only ever
        # tighten the outcome.  An assessor cannot turn a deny into an
        # allow — it has no way to express permission (roadmap §7.4).
        assessment_reason = ""
        if self._assessors is not None:
            rules.append("constraint_assessment_check")
            aggregate = self._assessors.aggregate(proposal)
            if aggregate.blocked:
                assessment_reason = aggregate.reason
                if aggregate.blocking_assessors:
                    result = "deny"
                    rules.append("assessor_block")
                elif result != "deny":
                    result = "requires_approval"
                    rules.append("assessor_requires_human")

        decision = self._make_decision(proposal, result, rules=rules)
        reason = f"policy evaluation: {result}"
        if assessment_reason:
            reason += f" — {assessment_reason}"
        return PolicyEvaluation(
            decision=decision,
            rules_evaluated=rules,
            reason=reason,
        )

    def _make_decision(
        self,
        proposal: ActionProposal,
        result: str,
        rules: List[str],
    ) -> PolicyDecision:
        return PolicyDecision(
            proposal_id=proposal.id,
            policy_version=self._version,
            rules_evaluated=rules,
            result=result,
            principal=self._principal,
            purpose="policy_evaluation",
            provenance=Provenance(
                source=f"policy_engine:{self._version}",
                upstream_ids=[proposal.id],
            ),
        )
