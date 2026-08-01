"""Outcome verifier registry.

Verifiers are registered with an independence group.  A verification only
counts when the verifier's group differs from the actuator's — an
executor self-verifying its own success is an explicitly rejected
anti-pattern, and so is a panel of verifiers that all share one group.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from common.contracts.base import (
    Principal,
    Provenance,
    VerificationVerdict,
)
from common.contracts.action import VerifiedOutcome
from common.contracts.autonomy import VerifierRegistration


class VerifierError(Exception):
    pass


class VerifierRegistry:
    """Registry of independent outcome verifiers.

    Parameters:
        principal: owning principal
        actuator_groups: actuator identity → independence group, used to
            reject self-verification
    """

    def __init__(
        self,
        principal: Principal,
        actuator_groups: Optional[Dict[str, str]] = None,
    ) -> None:
        self._principal = principal
        self._verifiers: Dict[str, VerifierRegistration] = {}
        self._actuator_groups: Dict[str, str] = dict(actuator_groups or {})

    # ── Registration ────────────────────────────────────────────────

    def register(
        self,
        verifier_identity: str,
        display_name: str,
        domains: List[str],
        independence_group: str,
        principal: Optional[Principal] = None,
    ) -> VerifierRegistration:
        if not verifier_identity or not verifier_identity.strip():
            raise VerifierError("verifier identity must not be empty")
        if verifier_identity in self._verifiers:
            raise VerifierError(f"verifier already registered: {verifier_identity}")
        if not domains:
            raise VerifierError("verifier must declare at least one domain")
        if not independence_group or not independence_group.strip():
            raise VerifierError("verifier must declare an independence group")

        registration = VerifierRegistration(
            verifier_identity=verifier_identity,
            display_name=display_name,
            domains=list(domains),
            independence_group=independence_group,
            principal=principal or self._principal,
            purpose="verifier_registration",
            provenance=Provenance(
                source="verifier_registry",
                independence_group=independence_group,
            ),
        )
        self._verifiers[verifier_identity] = registration
        return registration

    def set_actuator_group(self, actuator_identity: str, group: str) -> None:
        self._actuator_groups[actuator_identity] = group

    def suspend(self, verifier_identity: str, reason: str) -> None:
        reg = self._verifiers.get(verifier_identity)
        if reg is None:
            raise VerifierError(f"unknown verifier: {verifier_identity}")
        reg.active = False
        reg.suspended_reason = reason

    # ── Verification ────────────────────────────────────────────────

    def can_verify(
        self,
        verifier_identity: str,
        actuator_identity: str,
        domain: str,
    ) -> tuple[bool, str]:
        """Whether this verifier may verify this actuator in this domain."""
        reg = self._verifiers.get(verifier_identity)
        if reg is None:
            return False, f"verifier not registered: {verifier_identity}"
        if not reg.active:
            return False, f"verifier suspended: {reg.suspended_reason}"
        if domain not in reg.domains:
            return False, f"verifier does not cover domain '{domain}'"

        if verifier_identity == actuator_identity:
            return False, "self-verification: verifier is the actuator"

        actuator_group = self._actuator_groups.get(actuator_identity)
        if actuator_group is not None and actuator_group == reg.independence_group:
            return False, (
                f"verifier shares independence group '{actuator_group}' "
                f"with actuator '{actuator_identity}'"
            )

        return True, "independent"

    def verify(
        self,
        verifier_identity: str,
        actuator_identity: str,
        domain: str,
        workflow_id: str,
        receipt_id: str,
        verdict: VerificationVerdict,
        expected_state: Optional[Dict] = None,
        actual_state: Optional[Dict] = None,
        discrepancies: Optional[List[str]] = None,
        principal: Optional[Principal] = None,
    ) -> VerifiedOutcome:
        allowed, reason = self.can_verify(verifier_identity, actuator_identity, domain)
        if not allowed:
            raise VerifierError(f"verification rejected: {reason}")

        reg = self._verifiers[verifier_identity]
        return VerifiedOutcome(
            workflow_id=workflow_id,
            receipt_id=receipt_id,
            verifier=verifier_identity,
            verdict=verdict,
            expected_state=expected_state,
            actual_state=actual_state,
            discrepancies=list(discrepancies or []),
            verified_at=datetime.now(timezone.utc),
            principal=principal or self._principal,
            purpose="outcome_verification",
            provenance=Provenance(
                source=f"verifier:{verifier_identity}",
                upstream_ids=[workflow_id, receipt_id],
                independence_group=reg.independence_group,
            ),
        )

    # ── Independence accounting ─────────────────────────────────────

    def independent_verifier_count(
        self,
        actuator_identity: str,
        domain: str,
    ) -> int:
        """How many distinct independence groups can verify this actuator."""
        groups: Set[str] = set()
        for identity, reg in self._verifiers.items():
            allowed, _ = self.can_verify(identity, actuator_identity, domain)
            if allowed:
                groups.add(reg.independence_group)
        return len(groups)

    def distinct_groups(self, outcomes: List[VerifiedOutcome]) -> int:
        """Distinct independence groups represented in a set of outcomes.

        A panel that all shares one group counts as one, not as many.
        """
        groups: Set[str] = set()
        for outcome in outcomes:
            group = outcome.provenance.independence_group
            if group is not None:
                groups.add(group)
        return len(groups)

    def get(self, verifier_identity: str) -> Optional[VerifierRegistration]:
        return self._verifiers.get(verifier_identity)

    def list_verifiers(
        self,
        domain: Optional[str] = None,
        active_only: bool = True,
    ) -> List[VerifierRegistration]:
        results = list(self._verifiers.values())
        if active_only:
            results = [r for r in results if r.active]
        if domain is not None:
            results = [r for r in results if domain in r.domains]
        return sorted(results, key=lambda r: r.verifier_identity)
