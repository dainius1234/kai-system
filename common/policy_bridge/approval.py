"""Protected approval — digest-bound, single-use, authenticated.

Approval records are:
  - bound to the exact proposal digest (cannot be applied to a changed proposal)
  - single-use (nonce-protected against replay)
  - scoped to a specific risk tier
  - subject to expiry
  - revocable (denial takes precedence)

Anonymous, low-scope, and replay approvals are rejected.
"""
from __future__ import annotations

import hashlib
import secrets
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

from common.contracts.base import (
    APPROVAL_MATRIX,
    ApprovalStatus,
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.action import ActionProposal, ApprovalRecord


class ApprovalError(Exception):
    pass


class ApprovalGate:
    """Protected approval gate — validates and records approval decisions.

    Parameters:
        default_expiry_seconds: how long an approval remains valid
    """

    def __init__(self, default_expiry_seconds: int = 300) -> None:
        self._expiry = timedelta(seconds=default_expiry_seconds)
        self._used_nonces: Set[str] = set()
        self._records: Dict[str, ApprovalRecord] = {}
        self._revocations: Set[str] = set()

    def approve(
        self,
        proposal: ActionProposal,
        approver: str,
        principal: Principal,
        nonce: Optional[str] = None,
    ) -> ApprovalRecord:
        if not approver or not approver.strip():
            raise ApprovalError("anonymous approval rejected")

        if proposal.digest is None or not proposal.verify_digest():
            raise ApprovalError("proposal digest invalid — cannot approve")

        if nonce is None:
            nonce = secrets.token_hex(16)

        if nonce in self._used_nonces:
            raise ApprovalError(f"replay detected — nonce already used: {nonce}")

        matrix_entry = APPROVAL_MATRIX.get(proposal.risk_tier)
        if matrix_entry is None:
            raise ApprovalError(f"unknown risk tier: {proposal.risk_tier}")

        now = datetime.now(timezone.utc)
        record = ApprovalRecord(
            proposal_id=proposal.id,
            status=ApprovalStatus.APPROVED,
            approver=approver,
            risk_tier=proposal.risk_tier,
            approved_at=now,
            expires_at_approval=now + self._expiry,
            nonce=nonce,
            principal=principal,
            purpose="approval",
            provenance=Provenance(
                source="approval_gate",
                upstream_ids=[proposal.id],
            ),
        )

        self._used_nonces.add(nonce)
        self._records[record.id] = record
        return record

    def deny(
        self,
        proposal: ActionProposal,
        approver: str,
        principal: Principal,
        reason: str = "",
    ) -> ApprovalRecord:
        if not approver or not approver.strip():
            raise ApprovalError("anonymous denial rejected")

        record = ApprovalRecord(
            proposal_id=proposal.id,
            status=ApprovalStatus.DENIED,
            approver=approver,
            risk_tier=proposal.risk_tier,
            conditions=[reason] if reason else [],
            principal=principal,
            purpose="approval",
            provenance=Provenance(
                source="approval_gate",
                upstream_ids=[proposal.id],
            ),
        )

        self._records[record.id] = record
        self._revocations.add(proposal.id)
        return record

    def revoke(self, proposal_id: str) -> None:
        self._revocations.add(proposal_id)

    def is_approved(self, proposal: ActionProposal) -> bool:
        if proposal.id in self._revocations:
            return False

        for record in self._records.values():
            if record.proposal_id != proposal.id:
                continue
            if record.status == ApprovalStatus.DENIED:
                return False
            if record.status == ApprovalStatus.APPROVED:
                if record.expires_at_approval:
                    if datetime.now(timezone.utc) > record.expires_at_approval:
                        continue
                return True
        return False

    def get_record(self, proposal_id: str) -> Optional[ApprovalRecord]:
        for record in reversed(list(self._records.values())):
            if record.proposal_id == proposal_id:
                return record
        return None

    def validate_for_capability(
        self, proposal: ActionProposal
    ) -> ApprovalRecord:
        if proposal.id in self._revocations:
            raise ApprovalError("proposal has been revoked/denied")

        record = self.get_record(proposal.id)
        if record is None:
            raise ApprovalError("no approval record found")

        if record.status != ApprovalStatus.APPROVED:
            raise ApprovalError(f"approval status is {record.status.value}")

        if record.expires_at_approval:
            if datetime.now(timezone.utc) > record.expires_at_approval:
                raise ApprovalError("approval has expired")

        if proposal.digest is None or not proposal.verify_digest():
            raise ApprovalError("proposal digest changed after approval")

        return record
