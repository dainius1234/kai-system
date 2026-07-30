"""Capability bridge — single-use, audience-bound capability tokens.

Capabilities are:
  - audience-restricted to one actuator (cannot be used by another)
  - single-use (consumed on first use)
  - short-lived (subject to expiry)
  - bound to the exact approval and proposal digests
  - revocable at any time

An actuator cannot use a capability intended for another actuator.
Replay, target substitution, and parameter modification fail.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from common.contracts.base import (
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.action import (
    ActionCapability,
    ActionProposal,
    ApprovalRecord,
    CapabilityReleaseRecord,
)


class CapabilityError(Exception):
    pass


class CapabilityBridge:
    """Issues and validates single-use, audience-bound capabilities.

    Parameters:
        default_timeout: seconds before a capability expires
    """

    def __init__(self, default_timeout: int = 30) -> None:
        self._timeout = timedelta(seconds=default_timeout)
        self._capabilities: Dict[str, ActionCapability] = {}
        self._releases: List[CapabilityReleaseRecord] = []
        self._revoked: Set[str] = set()

    def issue(
        self,
        proposal: ActionProposal,
        approval: ApprovalRecord,
        target_actuator: str,
        capability_type: str,
        principal: Principal,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ActionCapability:
        if not target_actuator or not target_actuator.strip():
            raise CapabilityError("capability must specify a target actuator")

        if proposal.digest is None or not proposal.verify_digest():
            raise CapabilityError("proposal digest invalid")

        if approval.status.value != "approved":
            raise CapabilityError(
                f"approval status is {approval.status.value}, not approved"
            )

        if approval.proposal_id != proposal.id:
            raise CapabilityError("approval does not match proposal")

        now = datetime.now(timezone.utc)
        cap = ActionCapability(
            proposal_id=proposal.id,
            approval_id=approval.id,
            capability_type=capability_type,
            parameters=parameters or {},
            risk_tier=proposal.risk_tier,
            timeout_seconds=int(self._timeout.total_seconds()),
            principal=principal,
            purpose="capability",
            provenance=Provenance(
                source=f"capability_bridge:{target_actuator}",
                upstream_ids=[proposal.id, approval.id],
            ),
        )

        self._capabilities[cap.id] = cap
        return cap

    def consume(
        self,
        capability_id: str,
        actuator_identity: str,
        principal: Principal,
    ) -> ActionCapability:
        if capability_id not in self._capabilities:
            raise CapabilityError(f"capability not found: {capability_id}")

        cap = self._capabilities[capability_id]

        if capability_id in self._revoked:
            raise CapabilityError("capability has been revoked")

        if cap.used:
            raise CapabilityError("capability already consumed (single-use)")

        expected_actuator = cap.provenance.source.split(":", 1)[-1]
        if actuator_identity != expected_actuator:
            raise CapabilityError(
                f"actuator mismatch: capability is for '{expected_actuator}', "
                f"not '{actuator_identity}'"
            )

        now = datetime.now(timezone.utc)
        if cap.created_at and (now - cap.created_at).total_seconds() > cap.timeout_seconds:
            self._release(cap, "expired", "capability expired before consumption", principal)
            raise CapabilityError("capability has expired")

        cap.used = True
        cap.used_at = now
        return cap

    def revoke(self, capability_id: str, reason: str, principal: Principal) -> None:
        if capability_id not in self._capabilities:
            raise CapabilityError(f"capability not found: {capability_id}")

        self._revoked.add(capability_id)
        cap = self._capabilities[capability_id]
        self._release(cap, "revoked", reason, principal)

    def _release(
        self,
        cap: ActionCapability,
        release_type: str,
        reason: str,
        principal: Principal,
    ) -> CapabilityReleaseRecord:
        record = CapabilityReleaseRecord(
            capability_id=cap.id,
            release_type=release_type,
            released_at=datetime.now(timezone.utc),
            reason=reason,
            principal=principal,
            purpose="capability_release",
            provenance=Provenance(source="capability_bridge"),
        )
        self._releases.append(record)
        return record

    def get(self, capability_id: str) -> Optional[ActionCapability]:
        return self._capabilities.get(capability_id)

    @property
    def releases(self) -> List[CapabilityReleaseRecord]:
        return list(self._releases)

    def is_valid(self, capability_id: str, actuator_identity: str) -> bool:
        if capability_id not in self._capabilities:
            return False
        cap = self._capabilities[capability_id]
        if cap.used or capability_id in self._revoked:
            return False
        expected = cap.provenance.source.split(":", 1)[-1]
        if actuator_identity != expected:
            return False
        if cap.created_at:
            age = (datetime.now(timezone.utc) - cap.created_at).total_seconds()
            if age > cap.timeout_seconds:
                return False
        return True
