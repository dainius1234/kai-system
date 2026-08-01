"""Immutable claim/evidence service.

Append-only storage for graded evidence.  Records are never mutated in
place: a correction is a new record whose ``supersedes`` points at the
old one, so the full lineage survives.

The service is the single gate on the exit-gate rule that self-generated
text and simulation cannot grant trust.  Every record is graded at write
time, and ``qualifying_evidence()`` filters on that grade rather than on
anything inferred from content later.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from common.contracts.base import Principal, Provenance
from common.contracts.autonomy import EvidenceGrade, GradedEvidence


# Provenance source prefixes that can never produce qualifying evidence,
# whatever grade the caller claims.  A model writing its own transcript
# into the evidence store must not be able to launder it into trust.
SELF_GENERATED_SOURCES = (
    "llm:",
    "model:",
    "claude",
    "gpt",
    "kai:self",
    "self:",
    "simulation:",
    "simulated:",
    "paper-trader",
)


class EvidenceError(Exception):
    pass


class EvidenceService:
    """Append-only graded-evidence store.

    Parameters:
        principal: owning principal for cross-principal isolation
    """

    def __init__(self, principal: Principal) -> None:
        self._principal = principal
        self._records: Dict[str, GradedEvidence] = {}
        self._order: List[str] = []
        self._superseded: Set[str] = set()

    # ── Write path ──────────────────────────────────────────────────

    def record(
        self,
        grade: EvidenceGrade,
        domain: str,
        task_type: str,
        observed_by: str,
        content: Optional[Dict] = None,
        claim_id: Optional[str] = None,
        outcome_id: Optional[str] = None,
        supersedes: Optional[str] = None,
        provenance: Optional[Provenance] = None,
        principal: Optional[Principal] = None,
    ) -> GradedEvidence:
        if not domain or not domain.strip():
            raise EvidenceError("evidence must specify a domain")
        if not task_type or not task_type.strip():
            raise EvidenceError("evidence must specify a task_type")
        if not observed_by or not observed_by.strip():
            raise EvidenceError("evidence must name its observer")

        prov = provenance or Provenance(source=observed_by)
        effective_grade = self._downgrade_if_self_generated(grade, prov, observed_by)

        if supersedes is not None:
            if supersedes not in self._records:
                raise EvidenceError(f"cannot supersede unknown record: {supersedes}")
            if supersedes in self._superseded:
                raise EvidenceError(
                    f"record already superseded: {supersedes}"
                )

        record = GradedEvidence(
            grade=effective_grade,
            claim_id=claim_id,
            outcome_id=outcome_id,
            domain=domain,
            task_type=task_type,
            content=content or {},
            observed_by=observed_by,
            supersedes=supersedes,
            principal=principal or self._principal,
            purpose="evidence",
            provenance=prov,
        )

        self._records[record.id] = record
        self._order.append(record.id)
        if supersedes is not None:
            self._superseded.add(supersedes)

        return record

    def _downgrade_if_self_generated(
        self,
        claimed_grade: EvidenceGrade,
        provenance: Provenance,
        observed_by: str,
    ) -> EvidenceGrade:
        """Force a non-qualifying grade when the source is self-generated.

        A caller may not label its own model output EXTERNAL_OBSERVED to
        get it past the trust gate.
        """
        if not claimed_grade.qualifies():
            return claimed_grade

        haystack = f"{provenance.source} {observed_by}".lower()
        for marker in SELF_GENERATED_SOURCES:
            if marker in haystack:
                if "simul" in marker or "paper" in marker:
                    return EvidenceGrade.SIMULATED
                return EvidenceGrade.MODEL_GENERATED

        return claimed_grade

    # ── Immutability ────────────────────────────────────────────────

    def correct(
        self,
        record_id: str,
        grade: EvidenceGrade,
        content: Dict,
        observed_by: str,
        provenance: Optional[Provenance] = None,
    ) -> GradedEvidence:
        """Correct a record by appending a superseding one.

        The original is never edited — it stays readable for audit.
        """
        original = self._records.get(record_id)
        if original is None:
            raise EvidenceError(f"unknown record: {record_id}")

        return self.record(
            grade=grade,
            domain=original.domain,
            task_type=original.task_type,
            observed_by=observed_by,
            content=content,
            claim_id=original.claim_id,
            outcome_id=original.outcome_id,
            supersedes=record_id,
            provenance=provenance,
        )

    # ── Read path ───────────────────────────────────────────────────

    def get(self, record_id: str) -> Optional[GradedEvidence]:
        return self._records.get(record_id)

    def is_superseded(self, record_id: str) -> bool:
        return record_id in self._superseded

    def lineage(self, record_id: str) -> List[GradedEvidence]:
        """Full supersession chain ending at ``record_id``, oldest first."""
        chain: List[GradedEvidence] = []
        seen: Set[str] = set()
        current = self._records.get(record_id)
        while current is not None and current.id not in seen:
            chain.append(current)
            seen.add(current.id)
            if current.supersedes is None:
                break
            current = self._records.get(current.supersedes)
        return list(reversed(chain))

    def all_evidence(
        self,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
        include_superseded: bool = False,
    ) -> List[GradedEvidence]:
        results = [self._records[rid] for rid in self._order]
        if not include_superseded:
            results = [r for r in results if r.id not in self._superseded]
        if domain is not None:
            results = [r for r in results if r.domain == domain]
        if task_type is not None:
            results = [r for r in results if r.task_type == task_type]
        return results

    def qualifying_evidence(
        self,
        domain: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[GradedEvidence]:
        """Evidence that may contribute to trust.

        This is the only path a grant may use to count evidence.
        """
        return [
            r for r in self.all_evidence(domain=domain, task_type=task_type)
            if r.grade.qualifies()
        ]

    def grade_breakdown(
        self,
        domain: Optional[str] = None,
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for record in self.all_evidence(domain=domain):
            counts[record.grade.value] = counts.get(record.grade.value, 0) + 1
        return counts

    @property
    def count(self) -> int:
        return len(self._records)
