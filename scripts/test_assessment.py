"""Constraint assessment tests — Ohana unavailable and poisoned values.

Closes roadmap §16.13 ("Ohana unavailable or poisoned values state").

Enforces the §7.4 rules:
  - Ohana never creates a security allow by itself;
  - a hard safety/security block cannot be outweighed by loyalty or
    conviction;
  - an unavailable required assessment fails closed;
  - value assessment and factual verification remain separate dimensions.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.action import ActionProposal
from common.contracts.assessment import (
    AssessmentResult,
    AssessmentType,
    ConstraintAssessmentRecord,
)
from common.policy_bridge.assessment import (
    AssessmentError,
    AssessorRegistry,
)
from common.policy_bridge.policy_engine import PolicyEngine

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


def _proposal(risk_tier: RiskTier = RiskTier.OBSERVE) -> ActionProposal:
    return ActionProposal(
        action_type="test", description="test proposal",
        risk_tier=risk_tier, rationale="testing",
        alternatives=["do nothing"], principal=_principal(),
        purpose="test", provenance=Provenance(source="test"),
    )


def _registry(values_revision: str = "1.0.0") -> AssessorRegistry:
    return AssessorRegistry(
        principal=_principal(), values_revision=values_revision
    )


# ── 1. Result semantics ─────────────────────────────────────────────

def test_blocking_results():
    check("block_is_blocking", AssessmentResult.BLOCK.is_blocking())
    check("requires_human_is_blocking",
          AssessmentResult.REQUIRES_HUMAN.is_blocking())
    check("unavailable_is_blocking",
          AssessmentResult.UNAVAILABLE.is_blocking())
    check("allow_advisory_not_blocking",
          not AssessmentResult.ALLOW_ADVISORY.is_blocking())
    check("caution_not_blocking", not AssessmentResult.CAUTION.is_blocking())


def test_no_bare_allow_exists():
    """There is no way for an assessor to say 'allowed'."""
    values = {r.value for r in AssessmentResult}
    check("no_allow_value", "allow" not in values)
    check("no_approved_value", "approved" not in values)
    check("no_permit_value", "permit" not in values)
    check("advisory_is_explicit", "allow_advisory" in values)


# ── 2. Registration ─────────────────────────────────────────────────

def test_registration():
    registry = _registry()
    registry.register("ohana", AssessmentType.VALUES, required=True)
    check("assessor_registered", "ohana" in registry.list_assessors())
    check("required_tracked", registry.is_required("ohana"))

    registry.register("privacy", AssessmentType.PRIVACY)
    check("optional_not_required", not registry.is_required("privacy"))

    try:
        registry.register("ohana", AssessmentType.VALUES)
        check("duplicate_rejected", False, "should have raised")
    except AssessmentError as e:
        check("duplicate_rejected", "already registered" in str(e))

    try:
        registry.register("", AssessmentType.VALUES)
        check("empty_identity_rejected", False, "should have raised")
    except AssessmentError:
        check("empty_identity_rejected", True)


# ── 3. EXIT GATE: Ohana unavailable fails closed ────────────────────

def test_required_assessor_unavailable_blocks():
    registry = _registry()
    registry.register("ohana", AssessmentType.VALUES, required=True)
    registry.set_available("ohana", False)

    result = registry.aggregate(_proposal())
    check("unavailable_required_blocks", result.blocked)
    check("unavailable_named", "ohana" in result.blocking_assessors)
    check("unavailable_reason", "unavailable" in result.reason)


def test_optional_assessor_unavailable_does_not_block():
    registry = _registry()
    registry.register("nice-to-have", AssessmentType.DOMAIN, required=False)
    registry.set_available("nice-to-have", False)

    result = registry.aggregate(_proposal())
    check("unavailable_optional_allows", not result.blocked)
    check("unavailable_optional_cautions", len(result.cautions) == 1)


def test_crashing_assessor_treated_as_unavailable():
    """An assessor that raises must never be read as approval."""
    def exploding(proposal):
        raise RuntimeError("values store corrupt")

    registry = _registry()
    registry.register("ohana", AssessmentType.VALUES,
                      handler=exploding, required=True)

    record = registry.assess_one("ohana", _proposal())
    check("crash_is_unavailable", record.result == AssessmentResult.UNAVAILABLE)
    check("crash_reason_captured", any("corrupt" in r for r in record.reasons))

    result = registry.aggregate(_proposal())
    check("crash_blocks", result.blocked)


def test_garbage_return_treated_as_unavailable():
    """A poisoned assessor returning junk cannot smuggle through an allow."""
    for junk in ("allow", True, 1, None, {"result": "allow"}):
        registry = _registry()
        registry.register(
            "ohana", AssessmentType.VALUES,
            handler=lambda p, j=junk: j, required=True,
        )
        record = registry.assess_one("ohana", _proposal())
        check(f"junk_{type(junk).__name__}_is_unavailable",
              record.result == AssessmentResult.UNAVAILABLE)


def test_policy_fails_closed_on_unavailable_ohana():
    """End to end: unavailable required Ohana denies at the policy engine."""
    registry = _registry()
    registry.register("ohana", AssessmentType.VALUES, required=True)
    registry.set_available("ohana", False)

    engine = PolicyEngine(principal=_principal(), assessors=registry)
    evaluation = engine.evaluate(_proposal(RiskTier.OBSERVE))

    check("policy_denies_on_unavailable_ohana",
          evaluation.decision.result == "deny", evaluation.reason)
    check("policy_names_assessor_block",
          "assessor_block" in evaluation.rules_evaluated)


# ── 4. EXIT GATE: poisoned values cannot grant permission ───────────

def test_ohana_cannot_grant_allow():
    """Ohana advising allow does not by itself produce an allow."""
    registry = _registry()
    registry.register(
        "ohana", AssessmentType.VALUES,
        handler=lambda p: AssessmentResult.ALLOW_ADVISORY, required=True,
    )

    engine = PolicyEngine(principal=_principal(), assessors=registry)

    # A supervised-tier proposal still requires approval — Ohana's
    # advisory allow cannot downgrade it to a plain allow.
    evaluation = engine.evaluate(_proposal(RiskTier.ACT_SUPERVISED))
    check("ohana_cannot_downgrade_to_allow",
          evaluation.decision.result == "requires_approval",
          evaluation.reason)


def test_ohana_allow_cannot_override_policy_deny():
    """A policy deny stands even when every assessor advises allow."""
    registry = _registry()
    for name in ("ohana", "safety", "privacy"):
        registry.register(
            name,
            AssessmentType.VALUES if name == "ohana" else AssessmentType.SAFETY,
            handler=lambda p: AssessmentResult.ALLOW_ADVISORY,
        )

    engine = PolicyEngine(principal=_principal(), assessors=registry)
    proposal = _proposal(RiskTier.OBSERVE)
    proposal.digest.value = "tampered"

    evaluation = engine.evaluate(proposal)
    check("assessors_cannot_override_deny",
          evaluation.decision.result == "deny", evaluation.reason)


def test_safety_block_beats_values_allow():
    """Loyalty cannot outweigh a hard safety block."""
    registry = _registry()
    registry.register(
        "ohana", AssessmentType.VALUES,
        handler=lambda p: AssessmentResult.ALLOW_ADVISORY, required=True,
    )
    registry.register(
        "safety", AssessmentType.SAFETY,
        handler=lambda p: AssessmentResult.BLOCK, required=True,
    )

    result = registry.aggregate(_proposal())
    check("safety_block_wins", result.blocked)
    check("safety_named", "safety" in result.blocking_assessors)
    check("ohana_not_blocking", "ohana" not in result.blocking_assessors)

    engine = PolicyEngine(principal=_principal(), assessors=registry)
    evaluation = engine.evaluate(_proposal(RiskTier.OBSERVE))
    check("policy_denies_on_safety_block",
          evaluation.decision.result == "deny", evaluation.reason)


def test_requires_human_escalates_not_denies():
    registry = _registry()
    registry.register(
        "ohana", AssessmentType.VALUES,
        handler=lambda p: AssessmentResult.REQUIRES_HUMAN,
    )

    result = registry.aggregate(_proposal())
    check("requires_human_blocks_auto", result.blocked)
    check("requires_human_flagged", result.requires_human)
    check("requires_human_not_hard_block",
          "ohana" not in result.blocking_assessors)

    engine = PolicyEngine(principal=_principal(), assessors=registry)
    evaluation = engine.evaluate(_proposal(RiskTier.OBSERVE))
    check("policy_escalates_to_approval",
          evaluation.decision.result == "requires_approval", evaluation.reason)


# ── 5. Separation of dimensions ─────────────────────────────────────

def test_assessment_record_shape():
    """The record carries the §7.4 fields, including revision binding."""
    registry = _registry(values_revision="2.1.0")
    registry.register("ohana", AssessmentType.VALUES,
                      handler=lambda p: AssessmentResult.CAUTION)
    proposal = _proposal()
    record = registry.assess_one("ohana", proposal)

    check("record_binds_digest",
          record.proposal_digest == proposal.digest.value)
    check("record_names_assessor", record.assessor_identity == "ohana")
    check("record_has_type", record.assessment_type == AssessmentType.VALUES)
    check("record_has_result", record.result == AssessmentResult.CAUTION)
    check("record_binds_revision",
          record.policy_or_values_revision == "2.1.0")
    check("record_has_provenance",
          record.provenance.source == "assessor:ohana")


def test_values_do_not_affect_factual_confidence():
    """A values assessment carries no factual weight.

    ConstraintAssessmentRecord has no confidence or evidence-quality
    field, so a values verdict cannot be mistaken for evidence.
    """
    fields = set(ConstraintAssessmentRecord.model_fields)
    check("no_confidence_field", "confidence" not in fields)
    check("no_evidence_quality_field", "evidence_quality" not in fields)
    check("no_certainty_field", "certainty" not in fields)
    check("has_evidence_refs_only", "evidence_refs" in fields)


def test_no_assessors_means_no_objection():
    """An engine with no assessor registry behaves as before."""
    engine = PolicyEngine(principal=_principal())
    evaluation = engine.evaluate(_proposal(RiskTier.OBSERVE))
    check("no_assessors_allows", evaluation.decision.result == "allow")

    registry = _registry()
    engine2 = PolicyEngine(principal=_principal(), assessors=registry)
    evaluation2 = engine2.evaluate(_proposal(RiskTier.OBSERVE))
    check("empty_registry_allows", evaluation2.decision.result == "allow")


def test_aggregate_consults_every_assessor():
    calls = []

    def make(name, result):
        def handler(p):
            calls.append(name)
            return result
        return handler

    registry = _registry()
    registry.register("ohana", AssessmentType.VALUES,
                      handler=make("ohana", AssessmentResult.ALLOW_ADVISORY))
    registry.register("privacy", AssessmentType.PRIVACY,
                      handler=make("privacy", AssessmentResult.CAUTION))
    registry.register("risk", AssessmentType.RISK,
                      handler=make("risk", AssessmentResult.ALLOW_ADVISORY))

    result = registry.aggregate(_proposal())
    check("all_assessors_consulted", sorted(calls) == ["ohana", "privacy", "risk"])
    check("all_assessments_returned", len(result.assessments) == 3)
    check("caution_surfaced", len(result.cautions) == 1)


def test_unknown_assessor_rejected():
    registry = _registry()
    try:
        registry.assess_one("ghost", _proposal())
        check("unknown_assessor_rejected", False, "should have raised")
    except AssessmentError as e:
        check("unknown_assessor_rejected", "unknown assessor" in str(e))

    try:
        registry.set_available("ghost", False)
        check("unknown_availability_rejected", False, "should have raised")
    except AssessmentError:
        check("unknown_availability_rejected", True)


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_blocking_results()
    test_no_bare_allow_exists()
    test_registration()
    test_required_assessor_unavailable_blocks()
    test_optional_assessor_unavailable_does_not_block()
    test_crashing_assessor_treated_as_unavailable()
    test_garbage_return_treated_as_unavailable()
    test_policy_fails_closed_on_unavailable_ohana()
    test_ohana_cannot_grant_allow()
    test_ohana_allow_cannot_override_policy_deny()
    test_safety_block_beats_values_allow()
    test_requires_human_escalates_not_denies()
    test_assessment_record_shape()
    test_values_do_not_affect_factual_confidence()
    test_no_assessors_means_no_objection()
    test_aggregate_consults_every_assessor()
    test_unknown_assessor_rejected()

    print(f"\n{'='*60}")
    print(f"Constraint Assessment Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
