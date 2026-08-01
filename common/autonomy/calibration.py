"""Calibration by task type, domain and code revision.

Tracks how well predictions match verified outcomes, keyed on
(task_type, domain, revision).  Keying on revision matters: a code change
invalidates the track record that preceded it, so a new revision starts
uncalibrated rather than inheriting the old one's credit.

Only qualifying evidence moves the numbers.  Model-generated and
simulated records are counted separately as ``rejected_non_qualifying``
so the rejection is visible rather than silent.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from common.contracts.base import Principal, Provenance
from common.contracts.autonomy import (
    CalibrationRecord,
    EvidenceGrade,
    GradedEvidence,
)


class CalibrationError(Exception):
    pass


def _key(task_type: str, domain: str, revision: str) -> Tuple[str, str, str]:
    return (task_type, domain, revision)


class CalibrationTracker:
    """Per-(task, domain, revision) prediction calibration."""

    def __init__(self, principal: Principal) -> None:
        self._principal = principal
        self._records: Dict[Tuple[str, str, str], CalibrationRecord] = {}

    def _ensure(
        self, task_type: str, domain: str, revision: str
    ) -> CalibrationRecord:
        key = _key(task_type, domain, revision)
        if key not in self._records:
            self._records[key] = CalibrationRecord(
                task_type=task_type,
                domain=domain,
                revision=revision,
                principal=self._principal,
                purpose="calibration",
                provenance=Provenance(source="calibration_tracker"),
            )
        return self._records[key]

    def observe(
        self,
        task_type: str,
        domain: str,
        revision: str,
        predicted_confidence: float,
        evidence: GradedEvidence,
        was_correct: bool,
    ) -> CalibrationRecord:
        """Record one prediction against its graded outcome evidence.

        Non-qualifying evidence increments the rejection counter and
        leaves accuracy untouched.
        """
        if not 0.0 <= predicted_confidence <= 1.0:
            raise CalibrationError(
                f"predicted_confidence out of range: {predicted_confidence}"
            )

        record = self._ensure(task_type, domain, revision)

        if not evidence.grade.qualifies():
            record.rejected_non_qualifying += 1
            record.digest = record._make_digest()
            return record

        record.total_predictions += 1
        record.qualifying_outcomes += 1
        if was_correct:
            record.correct_predictions += 1

        actual = 1.0 if was_correct else 0.0
        record.brier_sum += (predicted_confidence - actual) ** 2

        record.accuracy = record.correct_predictions / record.total_predictions
        record.brier_score = record.brier_sum / record.total_predictions
        record.digest = record._make_digest()
        return record

    def get(
        self, task_type: str, domain: str, revision: str
    ) -> Optional[CalibrationRecord]:
        return self._records.get(_key(task_type, domain, revision))

    def accuracy(
        self, task_type: str, domain: str, revision: str
    ) -> float:
        record = self.get(task_type, domain, revision)
        return record.accuracy if record else 0.0

    def qualifying_count(
        self, task_type: str, domain: str, revision: str
    ) -> int:
        record = self.get(task_type, domain, revision)
        return record.qualifying_outcomes if record else 0

    def list_records(
        self,
        domain: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> List[CalibrationRecord]:
        results = list(self._records.values())
        if domain is not None:
            results = [r for r in results if r.domain == domain]
        if revision is not None:
            results = [r for r in results if r.revision == revision]
        return sorted(results, key=lambda r: (r.domain, r.task_type, r.revision))

    def revisions_for(self, task_type: str, domain: str) -> List[str]:
        return sorted(
            rev for (t, d, rev) in self._records
            if t == task_type and d == domain
        )
