"""Scoped autonomy authority — the Trust Ledger replacement.

The old Trust Ledger held one scalar level that applied everywhere and
never expired.  This replaces it with grants that are:

  - **scoped**    to exactly one (capability, domain) pair
  - **bounded**   by a maximum invocation count
  - **expiring**  at an explicit deadline, capped per level
  - **revocable** at any moment, with revocation beating every other check
  - **earned**    from qualifying evidence and calibration, never from
                  self-generated text or simulation

Requalification is not automatic.  A grant that expires is gone; issuing
a new one re-runs every requirement against current evidence.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from common.contracts.base import Principal, Provenance
from common.contracts.autonomy import (
    AUTONOMY_REQUIREMENTS,
    AutonomyGrant,
    AutonomyLevel,
    EvidenceGrade,
    ValueConfirmation,
)
from common.autonomy.calibration import CalibrationTracker
from common.autonomy.evidence_service import EvidenceService


class AutonomyError(Exception):
    pass


class AutonomyAuthority:
    """Issues, checks and revokes scoped autonomy grants.

    Parameters:
        principal: owning principal
        evidence: the graded-evidence store grants are earned from
        calibration: per-(task, domain, revision) calibration tracker
    """

    def __init__(
        self,
        principal: Principal,
        evidence: EvidenceService,
        calibration: CalibrationTracker,
    ) -> None:
        self._principal = principal
        self._evidence = evidence
        self._calibration = calibration
        self._grants: Dict[str, AutonomyGrant] = {}
        self._confirmations: Dict[str, ValueConfirmation] = {}
        self._used_nonces: set[str] = set()

    # ── Value confirmation ──────────────────────────────────────────

    def confirm_value(
        self,
        subject_digest: str,
        subject_kind: str,
        prompt_shown: str,
        confirmed: bool,
        confirmed_by: str,
        nonce: Optional[str] = None,
    ) -> ValueConfirmation:
        """Record an explicit human value confirmation.

        Confirmation is never inferred from ordinary conversation: it
        requires a prompt that was actually shown and a response bound to
        a specific digest, with a single-use nonce.
        """
        if not confirmed_by or not confirmed_by.strip():
            raise AutonomyError("value confirmation cannot be anonymous")
        if not prompt_shown or not prompt_shown.strip():
            raise AutonomyError(
                "value confirmation requires the prompt that was shown"
            )
        if not subject_digest or not subject_digest.strip():
            raise AutonomyError("value confirmation must bind to a digest")

        used_nonce = nonce or secrets.token_hex(16)
        if used_nonce in self._used_nonces:
            raise AutonomyError("value confirmation replay detected")
        self._used_nonces.add(used_nonce)

        record = ValueConfirmation(
            subject_digest=subject_digest,
            subject_kind=subject_kind,
            prompt_shown=prompt_shown,
            confirmed=confirmed,
            confirmed_by=confirmed_by,
            confirmed_at=datetime.now(timezone.utc),
            nonce=used_nonce,
            principal=self._principal,
            purpose="value_confirmation",
            provenance=Provenance(source=f"human:{confirmed_by}"),
        )
        self._confirmations[record.id] = record
        return record

    # ── Qualification ───────────────────────────────────────────────

    def check_qualification(
        self,
        level: AutonomyLevel,
        capability: str,
        domain: str,
        task_type: str,
        revision: str,
        independent_verifier_count: int = 0,
        human_confirmation_id: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Whether the evidence on hand supports this level right now."""
        if level == AutonomyLevel.A0_NONE:
            return True, "A0 requires no qualification"

        req = AUTONOMY_REQUIREMENTS[level]

        qualifying = self._evidence.qualifying_evidence(
            domain=domain, task_type=task_type
        )
        if len(qualifying) < req["min_qualifying_outcomes"]:
            return False, (
                f"insufficient qualifying evidence: {len(qualifying)} < "
                f"{req['min_qualifying_outcomes']}"
            )

        accuracy = self._calibration.accuracy(task_type, domain, revision)
        if accuracy < req["min_accuracy"]:
            return False, (
                f"calibration accuracy {accuracy:.2f} below required "
                f"{req['min_accuracy']:.2f} for revision '{revision}'"
            )

        if independent_verifier_count < req["min_independent_verifiers"]:
            return False, (
                f"insufficient independent verifiers: "
                f"{independent_verifier_count} < {req['min_independent_verifiers']}"
            )

        if req["requires_human_confirmation"]:
            if human_confirmation_id is None:
                return False, f"{level.name} requires explicit human confirmation"
            confirmation = self._confirmations.get(human_confirmation_id)
            if confirmation is None:
                return False, "human confirmation not found"
            if not confirmation.confirmed:
                return False, "human confirmation was declined"

        return True, "qualified"

    # ── Grant issuance ──────────────────────────────────────────────

    def grant(
        self,
        level: AutonomyLevel,
        capability: str,
        domain: str,
        task_type: str,
        revision: str,
        granted_by: str,
        max_invocations: int,
        duration_seconds: Optional[int] = None,
        independent_verifier_count: int = 0,
        human_confirmation_id: Optional[str] = None,
    ) -> AutonomyGrant:
        if not granted_by or not granted_by.strip():
            raise AutonomyError("grant cannot be issued anonymously")
        if max_invocations < 1:
            raise AutonomyError("grant must allow at least one invocation")

        qualified, reason = self.check_qualification(
            level=level,
            capability=capability,
            domain=domain,
            task_type=task_type,
            revision=revision,
            independent_verifier_count=independent_verifier_count,
            human_confirmation_id=human_confirmation_id,
        )
        if not qualified:
            raise AutonomyError(f"not qualified for {level.name}: {reason}")

        req = AUTONOMY_REQUIREMENTS[level]
        max_seconds = req["max_grant_seconds"]
        if max_seconds <= 0:
            raise AutonomyError(f"{level.name} cannot hold a standing grant")

        requested = duration_seconds if duration_seconds is not None else max_seconds
        if requested > max_seconds:
            raise AutonomyError(
                f"requested duration {requested}s exceeds "
                f"{level.name} maximum {max_seconds}s"
            )
        if requested < 1:
            raise AutonomyError("grant duration must be positive")

        now = datetime.now(timezone.utc)
        evidence_ids = [
            e.id for e in self._evidence.qualifying_evidence(
                domain=domain, task_type=task_type
            )
        ]
        calibration = self._calibration.get(task_type, domain, revision)

        grant = AutonomyGrant(
            level=level,
            capability=capability,
            domain=domain,
            granted_by=granted_by,
            granted_at=now,
            expires_at_grant=now + timedelta(seconds=requested),
            max_invocations=max_invocations,
            evidence_ids=evidence_ids,
            calibration_id=calibration.id if calibration else None,
            human_confirmation_id=human_confirmation_id,
            principal=self._principal,
            purpose="autonomy_grant",
            provenance=Provenance(
                source=f"autonomy_authority:{granted_by}",
                upstream_ids=evidence_ids[:16],
            ),
        )
        self._grants[grant.id] = grant
        return grant

    # ── Grant use ───────────────────────────────────────────────────

    def check_grant(
        self,
        grant_id: str,
        capability: str,
        domain: str,
    ) -> Tuple[bool, str]:
        """Whether a grant currently authorises this capability and domain."""
        grant = self._grants.get(grant_id)
        if grant is None:
            return False, f"grant not found: {grant_id}"

        if grant.revoked:
            return False, f"grant revoked: {grant.revoked_reason}"

        now = datetime.now(timezone.utc)
        if now >= grant.expires_at_grant:
            return False, "grant expired"

        if grant.invocations_used >= grant.max_invocations:
            return False, (
                f"grant exhausted: {grant.invocations_used}/"
                f"{grant.max_invocations} invocations used"
            )

        if grant.capability != capability:
            return False, (
                f"capability mismatch: grant is for '{grant.capability}', "
                f"not '{capability}'"
            )

        if grant.domain != domain:
            return False, (
                f"domain mismatch: grant is for '{grant.domain}', "
                f"not '{domain}'"
            )

        return True, "valid"

    def consume_grant(
        self,
        grant_id: str,
        capability: str,
        domain: str,
    ) -> AutonomyGrant:
        valid, reason = self.check_grant(grant_id, capability, domain)
        if not valid:
            raise AutonomyError(f"grant unusable: {reason}")

        grant = self._grants[grant_id]
        grant.invocations_used += 1
        grant.digest = grant._make_digest()
        return grant

    def revoke(self, grant_id: str, reason: str) -> AutonomyGrant:
        grant = self._grants.get(grant_id)
        if grant is None:
            raise AutonomyError(f"grant not found: {grant_id}")

        grant.revoked = True
        grant.revoked_at = datetime.now(timezone.utc)
        grant.revoked_reason = reason
        grant.digest = grant._make_digest()
        return grant

    def revoke_all(self, reason: str) -> int:
        """Emergency stop — revoke every outstanding grant."""
        count = 0
        for grant in self._grants.values():
            if not grant.revoked:
                self.revoke(grant.id, reason)
                count += 1
        return count

    # ── Inspection ──────────────────────────────────────────────────

    def get_grant(self, grant_id: str) -> Optional[AutonomyGrant]:
        return self._grants.get(grant_id)

    def active_grants(
        self,
        capability: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[AutonomyGrant]:
        now = datetime.now(timezone.utc)
        results = [
            g for g in self._grants.values()
            if not g.revoked
            and now < g.expires_at_grant
            and g.invocations_used < g.max_invocations
        ]
        if capability is not None:
            results = [g for g in results if g.capability == capability]
        if domain is not None:
            results = [g for g in results if g.domain == domain]
        return sorted(results, key=lambda g: g.granted_at)

    def effective_level(self, capability: str, domain: str) -> AutonomyLevel:
        """Highest level currently authorised for this capability and domain."""
        grants = self.active_grants(capability=capability, domain=domain)
        if not grants:
            return AutonomyLevel.A0_NONE
        return max(g.level for g in grants)
