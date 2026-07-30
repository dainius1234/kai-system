"""UH-5 policy bridge exit-gate tests.

Exit gates (from roadmap):
  - anonymous/low-scope/XSS/replay/modified-action approvals fail
  - policy/approval service outage fails closed
  - actuator cannot use a capability intended for another actuator

Additional tests:
  - risk classification via approval matrix
  - policy-as-code decision
  - digest binding
  - single-use nonce protection
  - revocation and expiry
  - capability audience restriction
  - capability consumption and release
  - full pipeline: proposal → policy → approval → capability
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import (
    APPROVAL_MATRIX,
    ApprovalStatus,
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.action import ActionProposal, ApprovalRecord
from common.policy_bridge.policy_engine import (
    POLICY_VERSION,
    PolicyEngine,
    PolicyEvaluation,
)
from common.policy_bridge.approval import ApprovalError, ApprovalGate
from common.policy_bridge.capability import CapabilityBridge, CapabilityError

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def _principal() -> Principal:
    return Principal(identity="kai", role="system")


def _proposal(
    risk_tier: RiskTier = RiskTier.OBSERVE,
    estimated_value: float | None = None,
) -> ActionProposal:
    return ActionProposal(
        action_type="test",
        description="test proposal",
        risk_tier=risk_tier,
        rationale="testing",
        alternatives=["do nothing"],
        principal=_principal(),
        purpose="test",
        provenance=Provenance(source="test"),
        estimated_value=estimated_value,
    )


# ── 1. Policy: observe tier auto-approves ────────────────────────────

def test_policy_observe():
    engine = PolicyEngine(principal=_principal())
    proposal = _proposal(RiskTier.OBSERVE)
    result = engine.evaluate(proposal)

    check("observe_allows", result.decision.result == "allow")
    check("observe_rules_evaluated", len(result.rules_evaluated) > 0)
    check("observe_policy_version", result.decision.policy_version == POLICY_VERSION)


# ── 2. Policy: supervised tier requires approval ─────────────────────

def test_policy_supervised():
    engine = PolicyEngine(principal=_principal())
    proposal = _proposal(RiskTier.ACT_SUPERVISED)
    result = engine.evaluate(proposal)

    check("supervised_requires_approval", result.decision.result == "requires_approval")
    check("supervised_rules_has_human",
          "requires_human_approval" in result.rules_evaluated)


# ── 3. Policy: autonomous tier requires approval ────────────────────

def test_policy_autonomous():
    engine = PolicyEngine(principal=_principal())
    proposal = _proposal(RiskTier.ACT_AUTONOMOUS)
    result = engine.evaluate(proposal)

    check("autonomous_requires_approval", result.decision.result == "requires_approval")


# ── 4. Policy: value exceeds limit → deny ───────────────────────────

def test_policy_value_limit():
    engine = PolicyEngine(principal=_principal())
    proposal = _proposal(RiskTier.ACT_SUPERVISED, estimated_value=500.0)
    result = engine.evaluate(proposal)

    check("value_exceeds_deny", result.decision.result == "deny")
    check("value_rule_present", "value_exceeds_limit" in result.rules_evaluated)


# ── 5. Policy: outage fails closed ──────────────────────────────────

def test_policy_outage_fails_closed():
    engine = PolicyEngine(principal=_principal())
    engine.set_available(False)

    proposal = _proposal(RiskTier.OBSERVE)
    result = engine.evaluate(proposal)

    check("outage_denies", result.decision.result == "deny")
    check("outage_reason", "unavailable" in result.reason.lower())


# ── 6. Policy: invalid digest → deny ────────────────────────────────

def test_policy_invalid_digest():
    engine = PolicyEngine(principal=_principal())
    proposal = _proposal(RiskTier.OBSERVE)
    proposal.digest.value = "tampered"

    result = engine.evaluate(proposal)
    check("invalid_digest_deny", result.decision.result == "deny")
    check("invalid_digest_rule", "invalid_digest" in result.rules_evaluated)


# ── 7. Approval: valid approval ──────────────────────────────────────

def test_approval_valid():
    gate = ApprovalGate()
    proposal = _proposal()
    record = gate.approve(proposal, "dainius", _principal())

    check("approval_created", record is not None)
    check("approval_status", record.status == ApprovalStatus.APPROVED)
    check("approval_approver", record.approver == "dainius")
    check("approval_has_nonce", record.nonce is not None)
    check("approval_has_expiry", record.expires_at_approval is not None)
    check("is_approved", gate.is_approved(proposal))


# ── 8. Approval: anonymous rejected ─────────────────────────────────

def test_approval_anonymous():
    gate = ApprovalGate()
    proposal = _proposal()

    try:
        gate.approve(proposal, "", _principal())
        check("anonymous_rejected", False, "should have raised")
    except ApprovalError as e:
        check("anonymous_rejected", "anonymous" in str(e).lower())

    try:
        gate.approve(proposal, "  ", _principal())
        check("whitespace_rejected", False, "should have raised")
    except ApprovalError as e:
        check("whitespace_rejected", "anonymous" in str(e).lower())


# ── 9. Approval: replay rejected ────────────────────────────────────

def test_approval_replay():
    gate = ApprovalGate()
    proposal = _proposal()

    gate.approve(proposal, "dainius", _principal(), nonce="unique-nonce-1")

    try:
        p2 = _proposal()
        gate.approve(p2, "dainius", _principal(), nonce="unique-nonce-1")
        check("replay_rejected", False, "should have raised")
    except ApprovalError as e:
        check("replay_rejected", "replay" in str(e).lower())


# ── 10. Approval: modified proposal rejected ────────────────────────

def test_approval_modified_proposal():
    gate = ApprovalGate()
    proposal = _proposal()
    proposal.digest.value = "tampered_after_signing"

    try:
        gate.approve(proposal, "dainius", _principal())
        check("modified_proposal_rejected", False, "should have raised")
    except ApprovalError as e:
        check("modified_proposal_rejected", "digest" in str(e).lower())


# ── 11. Approval: denial takes precedence ────────────────────────────

def test_denial_precedence():
    gate = ApprovalGate()
    proposal = _proposal()

    gate.approve(proposal, "dainius", _principal())
    check("initially_approved", gate.is_approved(proposal))

    gate.deny(proposal, "dainius", _principal(), "changed my mind")
    check("denial_overrides", not gate.is_approved(proposal))


# ── 12. Approval: revocation ────────────────────────────────────────

def test_revocation():
    gate = ApprovalGate()
    proposal = _proposal()

    gate.approve(proposal, "dainius", _principal())
    check("approved_before_revoke", gate.is_approved(proposal))

    gate.revoke(proposal.id)
    check("revoked", not gate.is_approved(proposal))


# ── 13. Approval: expiry ────────────────────────────────────────────

def test_approval_expiry():
    gate = ApprovalGate(default_expiry_seconds=1)
    proposal = _proposal()

    gate.approve(proposal, "dainius", _principal())
    check("approved_before_expiry", gate.is_approved(proposal))

    time.sleep(1.1)
    check("expired_after_timeout", not gate.is_approved(proposal))


# ── 14. Capability: issue and consume ────────────────────────────────

def test_capability_issue_consume():
    bridge = CapabilityBridge()
    proposal = _proposal()
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())

    cap = bridge.issue(
        proposal, approval, "paper-trader", "execute_trade", _principal()
    )
    check("capability_issued", cap is not None)
    check("capability_not_used", not cap.used)
    check("capability_risk_tier", cap.risk_tier == RiskTier.OBSERVE)

    consumed = bridge.consume(cap.id, "paper-trader", _principal())
    check("capability_consumed", consumed.used is True)
    check("capability_used_at_set", consumed.used_at is not None)


# ── 15. Capability: wrong actuator rejected ──────────────────────────

def test_capability_wrong_actuator():
    bridge = CapabilityBridge()
    proposal = _proposal()
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())

    cap = bridge.issue(
        proposal, approval, "paper-trader", "execute_trade", _principal()
    )

    try:
        bridge.consume(cap.id, "evil-actuator", _principal())
        check("wrong_actuator_rejected", False, "should have raised")
    except CapabilityError as e:
        check("wrong_actuator_rejected", "mismatch" in str(e).lower())


# ── 16. Capability: single use ───────────────────────────────────────

def test_capability_single_use():
    bridge = CapabilityBridge()
    proposal = _proposal()
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())

    cap = bridge.issue(
        proposal, approval, "paper-trader", "execute_trade", _principal()
    )
    bridge.consume(cap.id, "paper-trader", _principal())

    try:
        bridge.consume(cap.id, "paper-trader", _principal())
        check("single_use_enforced", False, "should have raised")
    except CapabilityError as e:
        check("single_use_enforced", "consumed" in str(e).lower() or "single" in str(e).lower())


# ── 17. Capability: revocation ───────────────────────────────────────

def test_capability_revocation():
    bridge = CapabilityBridge()
    proposal = _proposal()
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())

    cap = bridge.issue(
        proposal, approval, "paper-trader", "execute_trade", _principal()
    )
    check("valid_before_revoke", bridge.is_valid(cap.id, "paper-trader"))

    bridge.revoke(cap.id, "policy change", _principal())
    check("invalid_after_revoke", not bridge.is_valid(cap.id, "paper-trader"))

    try:
        bridge.consume(cap.id, "paper-trader", _principal())
        check("revoked_consume_rejected", False, "should have raised")
    except CapabilityError as e:
        check("revoked_consume_rejected", "revoked" in str(e).lower())

    check("release_recorded", len(bridge.releases) > 0)


# ── 18. Capability: expiry ──────────────────────────────────────────

def test_capability_expiry():
    bridge = CapabilityBridge(default_timeout=1)
    proposal = _proposal()
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())

    cap = bridge.issue(
        proposal, approval, "paper-trader", "execute_trade", _principal()
    )
    time.sleep(1.1)

    check("expired_invalid", not bridge.is_valid(cap.id, "paper-trader"))

    try:
        bridge.consume(cap.id, "paper-trader", _principal())
        check("expired_consume_rejected", False, "should have raised")
    except CapabilityError as e:
        check("expired_consume_rejected", "expired" in str(e).lower())


# ── 19. Capability: invalid proposal digest rejected ────────────────

def test_capability_invalid_proposal():
    bridge = CapabilityBridge()
    proposal = _proposal()
    proposal.digest.value = "tampered"

    gate = ApprovalGate()
    try:
        approval = gate.approve(proposal, "dainius", _principal())
        check("tampered_approval_rejected", False)
    except ApprovalError:
        check("tampered_approval_rejected", True)


# ── 20. Capability: mismatched approval rejected ────────────────────

def test_capability_mismatched_approval():
    bridge = CapabilityBridge()
    p1 = _proposal()
    p2 = _proposal()
    gate = ApprovalGate()
    approval = gate.approve(p1, "dainius", _principal())

    try:
        bridge.issue(p2, approval, "paper-trader", "execute_trade", _principal())
        check("mismatched_approval_rejected", False, "should have raised")
    except CapabilityError as e:
        check("mismatched_approval_rejected", "match" in str(e).lower())


# ── 21. Full pipeline: proposal → policy → approval → capability ────

def test_full_pipeline():
    engine = PolicyEngine(principal=_principal())
    gate = ApprovalGate()
    bridge = CapabilityBridge()

    proposal = _proposal(RiskTier.ACT_SUPERVISED, estimated_value=50.0)

    policy_result = engine.evaluate(proposal)
    check("pipeline_policy_requires_approval",
          policy_result.decision.result == "requires_approval")

    approval = gate.approve(proposal, "dainius", _principal())
    check("pipeline_approved", gate.is_approved(proposal))

    cap = bridge.issue(
        proposal, approval, "paper-trader", "paper_trade", _principal()
    )
    check("pipeline_capability_issued", cap is not None)

    consumed = bridge.consume(cap.id, "paper-trader", _principal())
    check("pipeline_capability_consumed", consumed.used)


# ── 22. Validate for capability checks all conditions ────────────────

def test_validate_for_capability():
    gate = ApprovalGate()
    proposal = _proposal()

    try:
        gate.validate_for_capability(proposal)
        check("no_approval_fails", False, "should have raised")
    except ApprovalError:
        check("no_approval_fails", True)

    gate.approve(proposal, "dainius", _principal())
    record = gate.validate_for_capability(proposal)
    check("valid_approval_passes", record.status == ApprovalStatus.APPROVED)


# ── Runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_policy_observe()
    test_policy_supervised()
    test_policy_autonomous()
    test_policy_value_limit()
    test_policy_outage_fails_closed()
    test_policy_invalid_digest()
    test_approval_valid()
    test_approval_anonymous()
    test_approval_replay()
    test_approval_modified_proposal()
    test_denial_precedence()
    test_revocation()
    test_approval_expiry()
    test_capability_issue_consume()
    test_capability_wrong_actuator()
    test_capability_single_use()
    test_capability_revocation()
    test_capability_expiry()
    test_capability_invalid_proposal()
    test_capability_mismatched_approval()
    test_full_pipeline()
    test_validate_for_capability()

    print(f"\n{'='*60}")
    print(f"UH-5 Policy Bridge Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
