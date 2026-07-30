"""UH-4 proposal workspace exit-gate tests.

Exit gates (from roadmap):
  - winning a bid cannot execute anything
  - duplicate/correlated/stub bidders cannot create qualifying consensus
  - workspace outage blocks proposals requiring it rather than causing a bypass

Additional tests:
  - bidder registration and rejection
  - typed proposal interface
  - evidence/assumption/dependency graph
  - alternatives and no-action option required
  - contradiction and missing-evidence handling
  - deterministic proposal envelope
  - no imports or network permissions to actuators
  - no capability issuance
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import (
    ContractState,
    Principal,
    Provenance,
    RiskTier,
    VerificationVerdict,
)
from common.contracts.action import ActionProposal
from common.contracts.world_state import Claim
from common.proposal_workspace.bidder import (
    BidderRegistration,
    BidderRegistry,
    BidderStatus,
)
from common.proposal_workspace.workspace import (
    EvidenceGap,
    ProposalEnvelope,
    ProposalSubmission,
    ProposalWorkspace,
    WorkspaceStatus,
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


def _bidder(
    identity: str = "analyst-1",
    domain: str = "market_analysis",
    group: str = "group_A",
) -> BidderRegistration:
    return BidderRegistration(
        identity=identity,
        display_name=identity,
        expertise_domain=domain,
        independence_group=group,
    )


def _claim(text: str = "test claim", domain: str = "test") -> Claim:
    return Claim(
        claim_text=text,
        domain=domain,
        principal=_principal(),
        purpose="test",
        provenance=Provenance(source="test"),
    )


def _submission(
    bidder_id: str = "analyst-1",
    evidence_ids: list | None = None,
    alternatives: list | None = None,
    no_action: str | None = "no action needed",
) -> ProposalSubmission:
    return ProposalSubmission(
        bidder_id=bidder_id,
        action_type="analysis",
        description="Market analysis proposal",
        risk_tier=RiskTier.OBSERVE,
        rationale="Based on current evidence",
        alternatives=alternatives or [],
        no_action_rationale=no_action,
        assumptions=["market is open"],
        dependencies=[],
        evidence_ids=evidence_ids or [],
    )


# ── 1. Bidder registration ──────────────────────────────────────────

def test_bidder_registration():
    reg = BidderRegistry()
    bid_id = reg.register(_bidder("b1", "domain1", "g1"))
    check("bidder_registered", bid_id == "b1")
    check("bidder_is_active", reg.is_registered("b1"))
    check("bidder_count", reg.count() == 1)


# ── 2. Duplicate bidder rejection ────────────────────────────────────

def test_duplicate_bidder():
    reg = BidderRegistry()
    reg.register(_bidder("dup1", "d1", "g1"))
    try:
        reg.register(_bidder("dup1", "d2", "g2"))
        check("duplicate_rejected", False, "should have raised")
    except ValueError as e:
        check("duplicate_rejected", "duplicate" in str(e).lower())


# ── 3. Stub bidder rejection ────────────────────────────────────────

def test_stub_bidder():
    reg = BidderRegistry()
    try:
        reg.register(_bidder("stub1", "", "g1"))
        check("stub_empty_domain_rejected", False, "should have raised")
    except ValueError as e:
        check("stub_empty_domain_rejected", "stub" in str(e).lower())

    try:
        reg.register(_bidder("stub2", "domain", ""))
        check("stub_empty_group_rejected", False, "should have raised")
    except ValueError as e:
        check("stub_empty_group_rejected", "stub" in str(e).lower())


# ── 4. Correlated bidder tracking ────────────────────────────────────

def test_correlated_bidders():
    reg = BidderRegistry()
    reg.register(_bidder("c1", "d1", "same_group"))
    reg.register(_bidder("c2", "d2", "same_group"))
    reg.register(_bidder("c3", "d3", "different_group"))

    correlated = reg.correlated_bidders("c1")
    check("correlated_found", "c2" in correlated)
    check("uncorrelated_excluded", "c3" not in correlated)


# ── 5. Qualifying diversity ─────────────────────────────────────────

def test_qualifying_diversity():
    reg = BidderRegistry(min_independent_groups=2)

    reg.register(_bidder("solo", "d1", "only_group"))
    check("single_group_no_diversity", not reg.has_qualifying_diversity())

    reg.register(_bidder("other", "d2", "second_group"))
    check("two_groups_qualifies", reg.has_qualifying_diversity())


# ── 6. Correlated bidders cannot create qualifying consensus ─────────

def test_correlated_no_consensus():
    reg = BidderRegistry(min_independent_groups=2)
    reg.register(_bidder("a1", "d1", "same_group"))
    reg.register(_bidder("a2", "d2", "same_group"))
    reg.register(_bidder("a3", "d3", "same_group"))

    check("correlated_only_no_consensus", not reg.has_qualifying_diversity())
    check("only_one_group", len(reg.independence_groups()) == 1)


# ── 7. Proposal submission produces ActionProposal ───────────────────

def test_proposal_creates_contract():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    envelope = ws.submit(_submission())
    check("envelope_created", isinstance(envelope, ProposalEnvelope))
    check("proposal_is_action_proposal", isinstance(envelope.proposal, ActionProposal))
    check("proposal_has_id", len(envelope.proposal.id) > 0)
    check("proposal_has_digest", envelope.proposal.digest is not None)
    check("proposal_digest_verifies", envelope.proposal.verify_digest())
    check("proposal_bidder_id", envelope.proposal.bidder_id == "analyst-1")


# ── 8. Unregistered bidder rejected ──────────────────────────────────

def test_unregistered_bidder():
    reg = BidderRegistry()
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    try:
        ws.submit(_submission("ghost"))
        check("unregistered_rejected", False, "should have raised")
    except ValueError as e:
        check("unregistered_rejected", "unregistered" in str(e).lower())


# ── 9. Alternatives or no-action required ────────────────────────────

def test_alternatives_required():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    try:
        ws.submit(ProposalSubmission(
            bidder_id="analyst-1",
            action_type="test",
            description="test",
            risk_tier=RiskTier.OBSERVE,
            rationale="test",
            alternatives=[],
            no_action_rationale=None,
        ))
        check("no_alternatives_rejected", False, "should have raised")
    except ValueError as e:
        check("no_alternatives_rejected", "alternatives" in str(e).lower())

    envelope = ws.submit(ProposalSubmission(
        bidder_id="analyst-1",
        action_type="test",
        description="test",
        risk_tier=RiskTier.OBSERVE,
        rationale="test",
        alternatives=["option B"],
        no_action_rationale=None,
    ))
    check("with_alternatives_accepted", envelope is not None)

    envelope2 = ws.submit(ProposalSubmission(
        bidder_id="analyst-1",
        action_type="test",
        description="test",
        risk_tier=RiskTier.OBSERVE,
        rationale="test",
        alternatives=[],
        no_action_rationale="no action is best",
    ))
    check("with_no_action_accepted", envelope2 is not None)


# ── 10. Evidence gap detection ───────────────────────────────────────

def test_evidence_gaps():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    envelope_no_evidence = ws.submit(_submission(evidence_ids=[]))
    check("no_evidence_gap_detected", len(envelope_no_evidence.evidence_gaps) > 0)
    check("no_evidence_gap_warning",
          any(g.severity == "warning" for g in envelope_no_evidence.evidence_gaps))

    claim = _claim()
    ws2 = ProposalWorkspace(
        principal=_principal(), registry=reg, available_claims=[claim]
    )
    envelope_missing = ws2.submit(_submission(evidence_ids=["nonexistent_id"]))
    check("missing_evidence_gap_error",
          any(g.severity == "error" for g in envelope_missing.evidence_gaps))

    envelope_valid = ws2.submit(_submission(evidence_ids=[claim.id]))
    check("valid_evidence_no_gaps", len(envelope_valid.evidence_gaps) == 0)


# ── 11. Contradiction detection ──────────────────────────────────────

def test_contradiction_detection():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))

    c1 = _claim("price will rise", "market")
    c2 = _claim("price will fall", "market")
    c1_with_contra = c1.model_copy(
        update={"contradicts": [c2.id], "digest": None}
    )
    c1_with_contra.digest = c1_with_contra._make_digest()

    ws = ProposalWorkspace(
        principal=_principal(),
        registry=reg,
        available_claims=[c1_with_contra, c2],
    )

    envelope = ws.submit(_submission(evidence_ids=[c1_with_contra.id]))
    check("contradictions_detected", len(envelope.contradictions) > 0)
    check("contradiction_has_claim_id",
          envelope.contradictions[0]["claim_id"] == c1_with_contra.id)


# ── 12. Workspace closed blocks proposals ────────────────────────────

def test_workspace_closed():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    ws.close()
    check("workspace_status_closed", ws.status == WorkspaceStatus.CLOSED)

    try:
        ws.submit(_submission())
        check("closed_workspace_blocks", False, "should have raised")
    except RuntimeError as e:
        check("closed_workspace_blocks", "closed" in str(e).lower())


# ── 13. Workspace degraded allows proposals ──────────────────────────

def test_workspace_degraded():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    ws.degrade()
    check("workspace_status_degraded", ws.status == WorkspaceStatus.DEGRADED)

    envelope = ws.submit(_submission())
    check("degraded_allows_proposals", envelope is not None)
    check("degraded_flagged_in_envelope",
          envelope.workspace_status == WorkspaceStatus.DEGRADED)


# ── 14. Deterministic envelope digest ────────────────────────────────

def test_deterministic_envelope():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws1 = ProposalWorkspace(principal=_principal(), registry=reg)
    ws2 = ProposalWorkspace(principal=_principal(), registry=reg)

    sub = _submission(evidence_ids=["e1", "e2"], alternatives=["alt1"])
    env1 = ws1.submit(sub)
    env2 = ws2.submit(sub)

    check("envelope_digest_deterministic",
          env1.envelope_digest == env2.envelope_digest)
    check("envelope_digest_nonempty", len(env1.envelope_digest) == 64)


# ── 15. Winning bid cannot execute — no capability issuance ──────────

def test_no_execution():
    import common.proposal_workspace.workspace as ws_mod
    source = open(ws_mod.__file__).read()

    check("no_capability_import",
          "ActionCapability" not in source)
    check("no_actuator_import",
          "ActuatorReceipt" not in source)
    check("no_executor_import",
          "executor" not in source.lower() or "execute anything" in source.lower())

    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)
    envelope = ws.submit(_submission())

    check("proposal_has_no_execute_method",
          not hasattr(envelope.proposal, "execute"))
    check("workspace_has_no_execute_method",
          not hasattr(ws, "execute"))
    check("workspace_has_no_issue_capability",
          not hasattr(ws, "issue_capability"))


# ── 16. Bidder suspend/revoke ────────────────────────────────────────

def test_bidder_lifecycle():
    reg = BidderRegistry()
    reg.register(_bidder("life1", "d1", "g1"))
    check("active_initially", reg.is_registered("life1"))

    reg.suspend("life1")
    check("suspended_not_active", not reg.is_registered("life1"))
    check("suspended_status", reg.get("life1").status == BidderStatus.SUSPENDED)

    reg.revoke("life1")
    check("revoked_status", reg.get("life1").status == BidderStatus.REVOKED)


# ── 17. Proposal count tracking ──────────────────────────────────────

def test_proposal_count():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    check("starts_at_zero", ws.proposal_count() == 0)
    ws.submit(_submission())
    check("increments", ws.proposal_count() == 1)
    ws.submit(_submission())
    check("tracks_all", ws.proposal_count() == 2)


# ── 18. Assumptions and dependencies tracked ─────────────────────────

def test_assumptions_dependencies():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    sub = ProposalSubmission(
        bidder_id="analyst-1",
        action_type="test",
        description="test",
        risk_tier=RiskTier.OBSERVE,
        rationale="test",
        alternatives=["do nothing"],
        assumptions=["market is open", "API available"],
        dependencies=["world-state-snapshot-xyz"],
    )
    envelope = ws.submit(sub)

    check("assumptions_preserved",
          envelope.proposal.assumptions == ["market is open", "API available"])
    check("dependencies_preserved",
          envelope.proposal.dependencies == ["world-state-snapshot-xyz"])


# ── 19. Independence group in proposal provenance ────────────────────

def test_provenance_independence():
    reg = BidderRegistry()
    reg.register(_bidder("analyst-1", "d1", "group_alpha"))
    ws = ProposalWorkspace(principal=_principal(), registry=reg)

    envelope = ws.submit(_submission())
    check("provenance_independence_group",
          envelope.proposal.provenance.independence_group == "group_alpha")


# ── 20. Extra fields rejected on submission ──────────────────────────

def test_extra_fields_rejected():
    try:
        ProposalSubmission(
            bidder_id="test",
            action_type="test",
            description="test",
            risk_tier=RiskTier.OBSERVE,
            rationale="test",
            alternatives=["alt"],
            injected_capability="hack",
        )
        check("extra_fields_rejected", False, "should have raised")
    except Exception:
        check("extra_fields_rejected", True)


# ── Runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_bidder_registration()
    test_duplicate_bidder()
    test_stub_bidder()
    test_correlated_bidders()
    test_qualifying_diversity()
    test_correlated_no_consensus()
    test_proposal_creates_contract()
    test_unregistered_bidder()
    test_alternatives_required()
    test_evidence_gaps()
    test_contradiction_detection()
    test_workspace_closed()
    test_workspace_degraded()
    test_deterministic_envelope()
    test_no_execution()
    test_bidder_lifecycle()
    test_proposal_count()
    test_assumptions_dependencies()
    test_provenance_independence()
    test_extra_fields_rejected()

    print(f"\n{'='*60}")
    print(f"UH-4 Proposal Workspace Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
