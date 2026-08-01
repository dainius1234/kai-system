"""UH-8: Outcome-based learning and autonomy requalification.

Components:
  - evidence_service:  append-only graded evidence; self-generated text
                       and simulation are structurally non-qualifying
  - verifier_registry: independent outcome verifiers; no self-verification
  - calibration:       accuracy and Brier score by task/domain/revision
  - authority:         A0–A4 scoped, bounded, expiring, revocable grants
  - release_bundle:    capability-specific signed release authorisation
  - wisdom_graph:      lineage and contradiction tracking
"""

from common.autonomy.evidence_service import EvidenceError, EvidenceService
from common.autonomy.verifier_registry import VerifierError, VerifierRegistry
from common.autonomy.calibration import CalibrationError, CalibrationTracker
from common.autonomy.authority import AutonomyAuthority, AutonomyError
from common.autonomy.release_bundle import (
    ReleaseBundleError,
    ReleaseBundleService,
)
from common.autonomy.wisdom_graph import WisdomError, WisdomGraph

__all__ = [
    "EvidenceError",
    "EvidenceService",
    "VerifierError",
    "VerifierRegistry",
    "CalibrationError",
    "CalibrationTracker",
    "AutonomyAuthority",
    "AutonomyError",
    "ReleaseBundleError",
    "ReleaseBundleService",
    "WisdomError",
    "WisdomGraph",
]
