"""Assessment contracts — typed constraint assessments.

Ohana, safety, privacy, risk and domain controls all return the same
shape (roadmap §7.4).  The rules that make this shape safe:

  - Ohana never creates a security allow by itself;
  - a hard safety/security block cannot be outweighed by loyalty or
    conviction;
  - an unavailable required assessment fails closed;
  - value assessment and factual verification remain separate dimensions.

``ALLOW_ADVISORY`` is deliberately named: an assessor can advise that it
sees no objection, but it cannot grant permission.  Permission comes from
the policy engine and human approval, never from an assessor.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import Field

from common.contracts.base import ContractBase


class AssessmentResult(str, Enum):
    ALLOW_ADVISORY = "allow_advisory"
    CAUTION = "caution"
    BLOCK = "block"
    REQUIRES_HUMAN = "requires_human"
    UNAVAILABLE = "unavailable"

    def is_blocking(self) -> bool:
        """Whether this result stops an action proceeding automatically.

        UNAVAILABLE counts as blocking: a required assessment that could
        not be obtained fails closed rather than being skipped.
        """
        return self in (
            AssessmentResult.BLOCK,
            AssessmentResult.REQUIRES_HUMAN,
            AssessmentResult.UNAVAILABLE,
        )


class AssessmentType(str, Enum):
    VALUES = "values"
    SAFETY = "safety"
    PRIVACY = "privacy"
    RISK = "risk"
    DOMAIN = "domain"


class ConstraintAssessmentRecord(ContractBase):
    """A typed assessment of one proposal by one assessor."""

    proposal_digest: str
    assessor_identity: str
    assessment_type: AssessmentType
    result: AssessmentResult
    constraints: List[str] = Field(default_factory=list)
    reasons: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    policy_or_values_revision: str
    expires_at_assessment: Optional[datetime] = None
