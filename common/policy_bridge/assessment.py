"""Assessor registry — typed constraint assessments, fail-closed.

Sits between the proposal workspace and the policy engine.  Assessors
(Ohana values, safety, privacy, risk, domain) each return a typed
assessment; the registry aggregates them under rules that cannot be
argued around:

  - a BLOCK from any assessor is final — no amount of loyalty,
    conviction or advisory approval outweighs it;
  - an assessor that is registered as required but unavailable fails
    closed, blocking rather than being skipped;
  - assessors can only ever advise allow.  ``aggregate()`` returns
    whether anything blocks, never a permission.

The last point is the important one structurally: this module has no way
to express "allowed".  Permission is the policy engine's and the human's
to give, so an assessor cannot manufacture one.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from common.contracts.base import Principal, Provenance
from common.contracts.action import ActionProposal
from common.contracts.assessment import (
    AssessmentResult,
    AssessmentType,
    ConstraintAssessmentRecord,
)


class AssessmentError(Exception):
    pass


class AggregateAssessment:
    """The combined verdict of every assessor consulted."""

    __slots__ = ("blocked", "reason", "assessments", "blocking_assessors",
                 "requires_human", "cautions")

    def __init__(
        self,
        blocked: bool,
        reason: str,
        assessments: List[ConstraintAssessmentRecord],
        blocking_assessors: List[str],
        requires_human: bool,
        cautions: List[str],
    ) -> None:
        self.blocked = blocked
        self.reason = reason
        self.assessments = assessments
        self.blocking_assessors = blocking_assessors
        self.requires_human = requires_human
        self.cautions = cautions


class AssessorRegistry:
    """Registry of constraint assessors with fail-closed aggregation.

    Parameters:
        principal: owning principal
        values_revision: the values/policy revision assessments are
            stamped with, so a stale assessment is detectable
    """

    def __init__(
        self,
        principal: Principal,
        values_revision: str = "1.0.0",
    ) -> None:
        self._principal = principal
        self._values_revision = values_revision
        self._assessors: Dict[str, AssessmentType] = {}
        self._required: Set[str] = set()
        self._handlers: Dict[str, Callable[[ActionProposal], AssessmentResult]] = {}
        self._available: Dict[str, bool] = {}

    def register(
        self,
        assessor_identity: str,
        assessment_type: AssessmentType,
        handler: Optional[Callable[[ActionProposal], AssessmentResult]] = None,
        required: bool = False,
    ) -> None:
        if not assessor_identity or not assessor_identity.strip():
            raise AssessmentError("assessor identity must not be empty")
        if assessor_identity in self._assessors:
            raise AssessmentError(
                f"assessor already registered: {assessor_identity}"
            )

        self._assessors[assessor_identity] = assessment_type
        self._available[assessor_identity] = True
        if handler is not None:
            self._handlers[assessor_identity] = handler
        if required:
            self._required.add(assessor_identity)

    def set_available(self, assessor_identity: str, available: bool) -> None:
        if assessor_identity not in self._assessors:
            raise AssessmentError(f"unknown assessor: {assessor_identity}")
        self._available[assessor_identity] = available

    # ── Assessment ──────────────────────────────────────────────────

    def assess_one(
        self,
        assessor_identity: str,
        proposal: ActionProposal,
    ) -> ConstraintAssessmentRecord:
        if assessor_identity not in self._assessors:
            raise AssessmentError(f"unknown assessor: {assessor_identity}")

        assessment_type = self._assessors[assessor_identity]
        reasons: List[str] = []

        if not self._available.get(assessor_identity, False):
            result = AssessmentResult.UNAVAILABLE
            reasons.append(f"{assessor_identity} is unavailable")
        else:
            handler = self._handlers.get(assessor_identity)
            if handler is None:
                result = AssessmentResult.ALLOW_ADVISORY
                reasons.append("no handler registered — advisory only")
            else:
                try:
                    result = handler(proposal)
                except Exception as exc:
                    # A crashing assessor is an unavailable assessor.  It
                    # must never be read as approval.
                    result = AssessmentResult.UNAVAILABLE
                    reasons.append(f"{assessor_identity} raised: {exc}")

        if not isinstance(result, AssessmentResult):
            result = AssessmentResult.UNAVAILABLE
            reasons.append(
                f"{assessor_identity} returned a non-assessment value"
            )

        return ConstraintAssessmentRecord(
            proposal_digest=proposal.digest.value if proposal.digest else "",
            assessor_identity=assessor_identity,
            assessment_type=assessment_type,
            result=result,
            reasons=reasons,
            policy_or_values_revision=self._values_revision,
            principal=self._principal,
            purpose="constraint_assessment",
            provenance=Provenance(source=f"assessor:{assessor_identity}"),
        )

    def aggregate(self, proposal: ActionProposal) -> AggregateAssessment:
        """Consult every assessor and combine the results, fail-closed."""
        assessments: List[ConstraintAssessmentRecord] = []
        blocking: List[str] = []
        cautions: List[str] = []
        requires_human = False
        reasons: List[str] = []

        for identity in sorted(self._assessors):
            record = self.assess_one(identity, proposal)
            assessments.append(record)

            if record.result == AssessmentResult.BLOCK:
                blocking.append(identity)
                reasons.append(f"{identity}: blocked")
            elif record.result == AssessmentResult.REQUIRES_HUMAN:
                requires_human = True
                reasons.append(f"{identity}: requires human review")
            elif record.result == AssessmentResult.UNAVAILABLE:
                # Only a *required* assessor blocks when unavailable.
                if identity in self._required:
                    blocking.append(identity)
                    reasons.append(f"{identity}: required but unavailable")
                else:
                    cautions.append(f"{identity} unavailable (not required)")
            elif record.result == AssessmentResult.CAUTION:
                cautions.append(f"{identity}: caution")

        blocked = bool(blocking) or requires_human
        return AggregateAssessment(
            blocked=blocked,
            reason="; ".join(reasons) if reasons else "no objections",
            assessments=assessments,
            blocking_assessors=blocking,
            requires_human=requires_human,
            cautions=cautions,
        )

    # ── Inspection ──────────────────────────────────────────────────

    def is_required(self, assessor_identity: str) -> bool:
        return assessor_identity in self._required

    def list_assessors(self) -> List[str]:
        return sorted(self._assessors)
