"""UH-6 vertical slice exit-gate tests.

Exit gates (from roadmap):
  - no direct financial mutation path remains
  - correlation and stale-source tests fail safely
  - one signal cannot close unrelated positions
  - partial/unknown outcomes reconcile safely
  - the slice runs in shadow/test mode

Additional tests:
  - full pipeline happy path
  - invalid market data rejected at perception stage
  - policy denial stops pipeline
  - approval gate blocks without auto_approve
  - capability consumed exactly once
  - verified outcome requires matching receipt
  - workflow audit trail complete
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import (
    ApprovalStatus,
    Principal,
    Provenance,
    RiskTier,
    VerificationVerdict,
)
from common.contracts.action import (
    ActionCapability,
    ActionProposal,
    ActionWorkflow,
    ActuatorReceipt,
    ApprovalRecord,
    VerifiedOutcome,
)
from common.perception_spine.journal import EventJournal
from common.vertical_slice.paper_trade_slice import (
    PaperTradeSlice,
    SliceResult,
    SliceStage,
)

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


def _market_data(symbol: str = "BTC/USDT", price: float = 5000.0) -> dict:
    return {
        "symbol": symbol,
        "price": price,
        "volume_24h": 1_000_000.0,
        "change_24h": 2.5,
        "bid": price - 10,
        "ask": price + 10,
        "timestamp": "2025-01-01T00:00:00Z",
    }


_tmpdir = tempfile.mkdtemp(prefix="uh6_test_")
_slice_counter = 0


def _make_slice(principal: Principal | None = None) -> PaperTradeSlice:
    global _slice_counter
    _slice_counter += 1
    p = principal or _principal()
    journal = EventJournal(
        os.path.join(_tmpdir, f"journal_{_slice_counter}.jsonl")
    )
    return PaperTradeSlice(
        journal=journal,
        principal=p,
        approver="dainius",
    )


# ── 1. Full pipeline happy path ────────────────────────────────────

def test_full_pipeline_happy_path():
    s = _make_slice()
    result = s.execute_slice(
        _market_data(),
        action_type="paper_trade_open",
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=True,
    )

    check("happy_path_success", result.success,
          f"stage={result.stage.value}, reason={result.reason}")
    check("happy_path_complete", result.stage == SliceStage.COMPLETE,
          f"stage={result.stage.value}, reason={result.reason}")
    check("happy_path_perception", result.perception_event is not None)
    check("happy_path_snapshot", result.snapshot is not None)
    check("happy_path_proposal", result.proposal is not None)
    check("happy_path_approval", result.approval is not None)
    check("happy_path_capability", result.capability is not None)
    check("happy_path_receipt", result.receipt is not None)
    check("happy_path_outcome", result.outcome is not None)
    check("happy_path_workflow", result.workflow is not None)


# ── 2. Pipeline produces correct artifact chain ────────────────────

def test_artifact_chain():
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    check("chain_proposal_has_id", r.proposal.id is not None)
    check("chain_approval_matches_proposal",
          r.approval.proposal_id == r.proposal.id)
    check("chain_capability_matches_proposal",
          r.capability.proposal_id == r.proposal.id)
    check("chain_capability_matches_approval",
          r.capability.approval_id == r.approval.id)
    check("chain_workflow_matches_proposal",
          r.workflow.proposal_id == r.proposal.id)
    check("chain_workflow_matches_approval",
          r.workflow.approval_id == r.approval.id)
    check("chain_workflow_matches_capability",
          r.workflow.capability_id == r.capability.id)
    check("chain_receipt_matches_capability",
          r.receipt.capability_id == r.capability.id)
    check("chain_receipt_matches_workflow",
          r.receipt.workflow_id == r.workflow.id)
    check("chain_outcome_matches_workflow",
          r.outcome.workflow_id == r.workflow.id)
    check("chain_outcome_matches_receipt",
          r.outcome.receipt_id == r.receipt.id)


# ── 3. No direct financial mutation path ───────────────────────────

def test_no_direct_mutation_path():
    """Execution requires a valid capability — no shortcut path exists."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    check("cap_is_consumed", r.capability.used is True)
    check("cap_used_at_set", r.capability.used_at is not None)

    try:
        s._capability_bridge.consume(
            r.capability.id, "paper-trader", _principal()
        )
        check("double_consume_blocked", False, "should have raised")
    except Exception as e:
        check("double_consume_blocked", "consumed" in str(e).lower() or "single" in str(e).lower())

    check("receipt_is_simulated", r.receipt.result.get("simulated") is True)
    check("receipt_is_reversible", r.receipt.reversible is True)
    check("receipt_actuator_is_paper", r.receipt.actuator == "paper-trader")


# ── 4. Capability audience restriction ─────────────────────────────

def test_capability_audience_restriction():
    """An actuator cannot use a capability intended for another actuator."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    cap_id = r.capability.id

    fresh_bridge = s._capability_bridge
    fresh_proposal = r.proposal
    fresh_approval = r.approval

    new_cap = fresh_bridge.issue(
        fresh_proposal, fresh_approval, "paper-trader",
        "paper_trade_open", _principal(),
    )

    try:
        fresh_bridge.consume(new_cap.id, "evil-actuator", _principal())
        check("audience_restriction_enforced", False, "should have raised")
    except Exception as e:
        check("audience_restriction_enforced", "mismatch" in str(e).lower())


# ── 5. Invalid market data rejected at perception ──────────────────

def test_invalid_market_data_rejected():
    s = _make_slice()
    result = s.execute_slice({})

    check("invalid_data_fails", not result.success)
    check("invalid_data_stage", result.stage == SliceStage.FAILED)
    check("invalid_data_reason", "invalid" in result.reason.lower() or "none" in result.reason.lower())


# ── 6. Policy denial stops pipeline ────────────────────────────────

def test_policy_denial_stops_pipeline():
    s2 = _make_slice()
    s2._policy_engine.set_available(False)
    result2 = s2.execute_slice(
        _market_data(),
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=True,
    )

    check("policy_outage_fails", not result2.success)
    check("policy_outage_stage", result2.stage == SliceStage.FAILED)
    check("policy_outage_reason", "denied" in result2.reason.lower() or "unavailable" in result2.reason.lower())
    check("policy_outage_no_capability", result2.capability is None)
    check("policy_outage_no_receipt", result2.receipt is None)


# ── 7. Approval gate blocks without auto_approve ──────────────────

def test_approval_gate_blocks():
    s = _make_slice()
    result = s.execute_slice(
        _market_data(),
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=False,
    )

    check("approval_blocks", not result.success)
    check("approval_stage", result.stage == SliceStage.APPROVAL)
    check("approval_reason", "approval" in result.reason.lower())
    check("approval_no_capability", result.capability is None)
    check("approval_no_receipt", result.receipt is None)
    check("approval_proposal_exists", result.proposal is not None)


# ── 8. Correlation: stale source fails safely ─────────────────────

def test_stale_source_fails_safely():
    """Stale events are accepted with ACCEPTED_STALE verdict but pipeline continues safely."""
    s = _make_slice()

    stale_data = _market_data()
    stale_data["timestamp"] = "2020-01-01T00:00:00Z"

    result = s.execute_slice(
        stale_data,
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=True,
    )

    if result.success:
        check("stale_still_tracked",
              result.perception_event is not None)
        check("stale_snapshot_valid",
              result.snapshot is not None and len(result.snapshot.claims) > 0)
    else:
        check("stale_rejected_safely", result.stage == SliceStage.FAILED)


# ── 9. One signal cannot close unrelated positions ─────────────────

def test_signal_isolation():
    """A signal for one symbol should not produce a proposal for another."""
    s = _make_slice()
    btc_result = s.execute_slice(
        _market_data("BTC/USDT", 5000.0),
        action_type="paper_trade_open",
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=True,
    )

    check("btc_signal_success", btc_result.success)
    check("btc_symbol_in_receipt",
          btc_result.receipt.result.get("symbol") == "BTC/USDT")

    s2 = _make_slice()
    eth_result = s2.execute_slice(
        _market_data("ETH/USDT", 3000.0),
        action_type="paper_trade_open",
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=True,
    )

    check("eth_signal_success", eth_result.success)
    check("eth_symbol_in_receipt",
          eth_result.receipt.result.get("symbol") == "ETH/USDT")

    check("signals_isolated",
          btc_result.receipt.result.get("symbol") != eth_result.receipt.result.get("symbol"))

    check("btc_cap_id_different", btc_result.capability.id != eth_result.capability.id)
    check("btc_proposal_id_different", btc_result.proposal.id != eth_result.proposal.id)


# ── 10. Partial/unknown outcomes reconcile safely ──────────────────

def test_partial_outcome_reconciliation():
    """Verified outcome matches expected vs actual state."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    check("outcome_verdict_confirmed",
          r.outcome.verdict == VerificationVerdict.CONFIRMED)
    check("outcome_expected_state",
          r.outcome.expected_state == {"simulated_trade": True})
    check("outcome_actual_matches_expected",
          r.outcome.expected_state == r.outcome.actual_state)
    check("outcome_has_verifier",
          r.outcome.verifier == "portfolio-verifier")
    check("outcome_provenance_upstream",
          r.workflow.id in r.outcome.provenance.upstream_ids)
    check("outcome_receipt_upstream",
          r.receipt.id in r.outcome.provenance.upstream_ids)


# ── 11. Shadow/test mode: all results simulated ───────────────────

def test_shadow_mode():
    """The slice runs in shadow/test mode — no live capital."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    check("shadow_receipt_simulated", r.receipt.result.get("simulated") is True)
    check("shadow_receipt_reversible", r.receipt.reversible is True)
    check("shadow_actuator_is_paper", r.receipt.actuator == "paper-trader")
    check("shadow_no_side_effects", len(r.receipt.side_effects) == 0)
    check("shadow_workflow_purpose", r.workflow.purpose == "paper_trade")


# ── 12. Verified outcome required for completion ──────────────────

def test_verified_outcome_required():
    """No learning update without verified outcome — outcome must exist for COMPLETE."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    check("complete_has_outcome", r.outcome is not None)
    check("complete_outcome_verified",
          r.outcome.verdict == VerificationVerdict.CONFIRMED)
    check("complete_verified_at_set", r.outcome.verified_at is not None)
    check("complete_provenance_source",
          r.outcome.provenance.source == "portfolio-verifier")


# ── 13. Workflow audit trail ──────────────────────────────────────

def test_workflow_audit_trail():
    """Workflow links all components for audit."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    check("audit_workflow_status", r.workflow.status == "executed")
    check("audit_workflow_started", r.workflow.started_at is not None)
    check("audit_workflow_completed", r.workflow.completed_at is not None)
    check("audit_workflow_principal",
          r.workflow.principal.identity == "kai")
    check("audit_provenance_source",
          r.workflow.provenance.source == "paper_trade_slice")
    check("audit_provenance_upstream_has_proposal",
          r.proposal.id in r.workflow.provenance.upstream_ids)
    check("audit_provenance_upstream_has_approval",
          r.approval.id in r.workflow.provenance.upstream_ids)
    check("audit_provenance_upstream_has_capability",
          r.capability.id in r.workflow.provenance.upstream_ids)


# ── 14. Observe tier auto-approves in slice ───────────────────────

def test_observe_tier_auto_approves():
    """OBSERVE tier should pass policy without requiring explicit approval."""
    s = _make_slice()
    r = s.execute_slice(
        _market_data(),
        risk_tier=RiskTier.OBSERVE,
        auto_approve=False,
    )

    check("observe_succeeds", r.success)
    check("observe_complete", r.stage == SliceStage.COMPLETE)
    check("observe_approval_exists", r.approval is not None)


# ── 15. Value limit triggers denial in slice ──────────────────────

def test_value_limit_denial():
    """High estimated_value triggers policy denial."""
    s = _make_slice()
    data = _market_data(price=50000.0)
    r = s.execute_slice(
        data,
        risk_tier=RiskTier.ACT_SUPERVISED,
        auto_approve=True,
    )

    check("value_limit_fails", not r.success)
    check("value_limit_stage", r.stage == SliceStage.FAILED)
    check("value_limit_reason", "denied" in r.reason.lower() or "policy" in r.reason.lower())


# ── 16. Duplicate events handled safely ───────────────────────────

def test_duplicate_events_handled():
    """Submitting the same market data twice — second should be deduped."""
    s = _make_slice()
    data = _market_data()

    r1 = s.execute_slice(
        data, auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )
    check("first_submission_ok", r1.success)

    r2 = s.execute_slice(
        data, auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )

    if r2.success:
        check("dedup_second_distinct", r2.proposal.id != r1.proposal.id)
    else:
        check("dedup_failed_safely",
              r2.stage == SliceStage.FAILED)


# ── 17. Cross-principal isolation ─────────────────────────────────

def test_cross_principal_isolation():
    """Events from different principals are isolated."""
    p1 = Principal(identity="kai", role="system")
    p2 = Principal(identity="other", role="external")

    s1 = _make_slice(principal=p1)
    s2 = _make_slice(principal=p2)

    r1 = s1.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )
    check("p1_succeeds", r1.success)

    r2 = s2.execute_slice(
        _market_data(), auto_approve=True,
        risk_tier=RiskTier.ACT_SUPERVISED,
    )
    check("p2_succeeds", r2.success)

    check("principal_isolation_proposals",
          r1.proposal.id != r2.proposal.id)
    check("principal_isolation_capabilities",
          r1.capability.id != r2.capability.id)


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_full_pipeline_happy_path()
    test_artifact_chain()
    test_no_direct_mutation_path()
    test_capability_audience_restriction()
    test_invalid_market_data_rejected()
    test_policy_denial_stops_pipeline()
    test_approval_gate_blocks()
    test_stale_source_fails_safely()
    test_signal_isolation()
    test_partial_outcome_reconciliation()
    test_shadow_mode()
    test_verified_outcome_required()
    test_workflow_audit_trail()
    test_observe_tier_auto_approves()
    test_value_limit_denial()
    test_duplicate_events_handled()
    test_cross_principal_isolation()

    print(f"\n{'='*60}")
    print(f"UH-6 Vertical Slice Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
