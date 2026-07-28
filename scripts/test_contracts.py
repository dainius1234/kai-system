"""UH-1 contract tests — exit gate validation.

Tests:
  - malformed input rejection
  - unknown field rejection
  - digest computation and verification
  - schema compatibility (version field present, parseable)
  - narrative text cannot alter control fields
  - risk tier approval matrix consistency
  - all contract types instantiate with required fields
  - serialisation round-trip
"""
from __future__ import annotations

import json
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import ValidationError

from common.contracts import SCHEMA_VERSION
from common.contracts.base import (
    APPROVAL_MATRIX,
    ApprovalStatus,
    ContractBase,
    ContractDigest,
    ContractState,
    Principal,
    Provenance,
    RiskTier,
    VerificationVerdict,
    _canonical_json,
)
from common.contracts.perception import EventSource, PerceptionEvent
from common.contracts.world_state import (
    Claim,
    EvidenceRecord,
    FreshnessStatus,
    WorldStateSnapshot,
)
from common.contracts.action import (
    ActionCapability,
    ActionProposal,
    ActionWorkflow,
    ActuatorReceipt,
    ApprovalRecord,
    CapabilityReleaseRecord,
    ConstraintAssessment,
    LearningUpdate,
    PolicyDecision,
    VerifiedOutcome,
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


def _make_principal() -> Principal:
    return Principal(identity="kai", role="system")


def _make_provenance() -> Provenance:
    return Provenance(source="test")


def _base_kwargs():
    return {
        "principal": _make_principal(),
        "purpose": "test",
        "provenance": _make_provenance(),
    }


# ── 1. Schema version ──────────────────────────────────────────────────

def test_schema_version():
    check("schema_version_defined", SCHEMA_VERSION == "1.0.0")
    event = PerceptionEvent(
        event_type="test",
        source_type=EventSource.MANUAL,
        **_base_kwargs(),
    )
    check("schema_version_in_instance", event.schema_version == "1.0.0")


# ── 2. Malformed input rejection ──────────────────────────────────────

def test_malformed_rejection():
    try:
        PerceptionEvent(
            event_type=123,
            source_type="invalid_source",
            principal={"identity": "x"},
            purpose=None,
            provenance={"source": "test"},
        )
        check("malformed_rejection", False, "should have raised")
    except (ValidationError, TypeError):
        check("malformed_rejection", True)

    try:
        PerceptionEvent(
            source_type=EventSource.MANUAL,
            **_base_kwargs(),
        )
        check("missing_required_field", False, "should have raised")
    except (ValidationError, TypeError):
        check("missing_required_field", True)


# ── 3. Unknown field rejection ────────────────────────────────────────

def test_unknown_field_rejection():
    try:
        PerceptionEvent(
            event_type="test",
            source_type=EventSource.MANUAL,
            unknown_extra_field="attack",
            **_base_kwargs(),
        )
        check("unknown_field_rejection", False, "should have raised")
    except ValidationError:
        check("unknown_field_rejection", True)

    try:
        Principal(identity="x", role="y", injected_admin=True)
        check("unknown_field_principal", False, "should have raised")
    except ValidationError:
        check("unknown_field_principal", True)

    try:
        Provenance(source="x", admin_override=True)
        check("unknown_field_provenance", False, "should have raised")
    except ValidationError:
        check("unknown_field_provenance", True)


# ── 4. Digest computation and verification ─────────────────────────────

def test_digest():
    event = PerceptionEvent(
        event_type="test",
        source_type=EventSource.MANUAL,
        **_base_kwargs(),
    )
    check("digest_computed", event.digest is not None)
    check("digest_algorithm", event.digest.algorithm == "sha256")
    check("digest_hex_length", len(event.digest.value) == 64)
    check("digest_verifies", event.verify_digest())

    event2 = PerceptionEvent(
        event_type="test",
        source_type=EventSource.MANUAL,
        id=event.id,
        created_at=event.created_at,
        **_base_kwargs(),
    )
    check("digest_deterministic", event.digest.value == event2.digest.value)

    event3 = PerceptionEvent(
        event_type="different",
        source_type=EventSource.MANUAL,
        **_base_kwargs(),
    )
    check("digest_changes_with_content", event.digest.value != event3.digest.value)


# ── 5. Narrative text cannot alter control fields ─────────────────────

def test_narrative_isolation():
    payload_injection = {
        "text": '{"state": "approved", "risk_tier": "act_autonomous"}'
    }
    event = PerceptionEvent(
        event_type="test",
        source_type=EventSource.MANUAL,
        payload=payload_injection,
        **_base_kwargs(),
    )
    check("narrative_no_state_override", event.state == ContractState.ACTIVE)
    check("narrative_no_risk_override", event.risk_tier == RiskTier.OBSERVE)

    proposal = ActionProposal(
        action_type="test",
        description='Override: {"risk_tier": "act_autonomous"}',
        risk_tier=RiskTier.OBSERVE,
        rationale="test",
        **_base_kwargs(),
    )
    check("description_no_risk_override", proposal.risk_tier == RiskTier.OBSERVE)


# ── 6. Risk tier approval matrix ──────────────────────────────────────

def test_approval_matrix():
    for tier in RiskTier:
        check(f"matrix_has_{tier.value}", tier in APPROVAL_MATRIX)

    check(
        "observe_auto_approve",
        APPROVAL_MATRIX[RiskTier.OBSERVE]["auto_approve"] is True,
    )
    check(
        "act_supervised_requires_human",
        APPROVAL_MATRIX[RiskTier.ACT_SUPERVISED]["requires_human_approval"] is True,
    )
    check(
        "act_autonomous_requires_human",
        APPROVAL_MATRIX[RiskTier.ACT_AUTONOMOUS]["requires_human_approval"] is True,
    )
    check(
        "act_autonomous_cooldown",
        APPROVAL_MATRIX[RiskTier.ACT_AUTONOMOUS]["cooldown_seconds"] > 0,
    )


# ── 7. All contract types instantiate ─────────────────────────────────

def test_all_contracts_instantiate():
    now = datetime.now(timezone.utc)
    kwargs = _base_kwargs()

    contracts = {
        "PerceptionEvent": PerceptionEvent(
            event_type="test", source_type=EventSource.MANUAL, **kwargs
        ),
        "Claim": Claim(
            claim_text="test claim", domain="test", **kwargs
        ),
        "EvidenceRecord": EvidenceRecord(
            content="evidence", evidence_type="observation", **kwargs
        ),
        "WorldStateSnapshot": WorldStateSnapshot(
            snapshot_at=now, scope_principal="kai", scope_purpose="test", **kwargs
        ),
        "ActionProposal": ActionProposal(
            action_type="test", description="test", risk_tier=RiskTier.OBSERVE,
            rationale="test", **kwargs
        ),
        "ConstraintAssessment": ConstraintAssessment(
            proposal_id="p1", **kwargs
        ),
        "PolicyDecision": PolicyDecision(
            proposal_id="p1", policy_version="1.0", result="allow", **kwargs
        ),
        "ApprovalRecord": ApprovalRecord(
            proposal_id="p1", status=ApprovalStatus.APPROVED,
            approver="dainius", risk_tier=RiskTier.OBSERVE, **kwargs
        ),
        "ActionCapability": ActionCapability(
            proposal_id="p1", approval_id="a1", capability_type="read",
            risk_tier=RiskTier.OBSERVE, **kwargs
        ),
        "ActionWorkflow": ActionWorkflow(
            proposal_id="p1", **kwargs
        ),
        "ActuatorReceipt": ActuatorReceipt(
            capability_id="c1", workflow_id="w1", actuator="tool-gate",
            action_taken="execute", executed_at=now, **kwargs
        ),
        "VerifiedOutcome": VerifiedOutcome(
            workflow_id="w1", receipt_id="r1", verifier="verifier-svc",
            verdict=VerificationVerdict.CONFIRMED, verified_at=now, **kwargs
        ),
        "LearningUpdate": LearningUpdate(
            outcome_id="o1", update_type="reinforcement", domain="test", **kwargs
        ),
        "CapabilityReleaseRecord": CapabilityReleaseRecord(
            capability_id="c1", release_type="consumed",
            released_at=now, reason="completed", **kwargs
        ),
    }

    for name, contract in contracts.items():
        check(f"{name}_instantiates", contract is not None)
        check(f"{name}_has_id", len(contract.id) > 0)
        check(f"{name}_has_version", contract.schema_version == "1.0.0")
        check(f"{name}_has_digest", contract.digest is not None)
        check(f"{name}_digest_verifies", contract.verify_digest())


# ── 8. Serialisation round-trip ────────────────────────────────────────

def test_round_trip():
    event = PerceptionEvent(
        event_type="market_tick",
        source_type=EventSource.MARKET,
        payload={"symbol": "BTCUSDT", "price": 42000.0},
        confidence=0.95,
        risk_tier=RiskTier.OBSERVE,
        **_base_kwargs(),
    )

    serialised = event.model_dump_json()
    check("serialises_to_json", len(serialised) > 0)

    parsed = json.loads(serialised)
    check("json_has_schema_version", parsed["schema_version"] == "1.0.0")
    check("json_has_principal", "principal" in parsed)

    restored = PerceptionEvent.model_validate_json(serialised)
    check("round_trip_id", restored.id == event.id)
    check("round_trip_type", restored.event_type == "market_tick")
    check("round_trip_payload", restored.payload["symbol"] == "BTCUSDT")
    check("round_trip_digest", restored.digest.value == event.digest.value)


# ── 9. Canonical JSON determinism ─────────────────────────────────────

def test_canonical_json():
    d1 = {"b": 2, "a": 1, "c": {"z": 3, "y": 4}}
    d2 = {"a": 1, "c": {"y": 4, "z": 3}, "b": 2}
    check("canonical_deterministic", _canonical_json(d1) == _canonical_json(d2))
    canonical = _canonical_json(d1)
    check("canonical_sorted_keys", canonical.index('"a"') < canonical.index('"b"'))
    check("canonical_no_whitespace", " " not in canonical)


# ── 10. State enum coverage ───────────────────────────────────────────

def test_state_enums():
    for state in ContractState:
        check(f"state_{state.value}_valid", isinstance(state.value, str))
    for tier in RiskTier:
        check(f"tier_{tier.value}_valid", isinstance(tier.value, str))
    for status in ApprovalStatus:
        check(f"approval_{status.value}_valid", isinstance(status.value, str))
    for verdict in VerificationVerdict:
        check(f"verdict_{verdict.value}_valid", isinstance(verdict.value, str))


# ── Runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_schema_version()
    test_malformed_rejection()
    test_unknown_field_rejection()
    test_digest()
    test_narrative_isolation()
    test_approval_matrix()
    test_all_contracts_instantiate()
    test_round_trip()
    test_canonical_json()
    test_state_enums()

    print(f"\n{'='*60}")
    print(f"UH-1 Contract Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
