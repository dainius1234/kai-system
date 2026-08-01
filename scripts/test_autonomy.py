"""UH-8 outcome-based learning and autonomy requalification tests.

Exit gates (from roadmap):
  - self-generated text or simulation cannot grant trust
  - high-consequence domains pass separate attack-chain tests
  - autonomy remains bounded, expiring and revocable

Deliverable coverage:
  - immutable claim/evidence service
  - outcome verifier registry
  - calibration by task/domain/revision
  - Trust Ledger replacement (A0-A4 scoped authority)
  - explicit value confirmation workflow
  - Wisdom Graph lineage and contradiction
  - capability-specific signed release bundles
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import (
    Principal,
    Provenance,
    VerificationVerdict,
)
from common.contracts.autonomy import (
    AUTONOMY_REQUIREMENTS,
    AutonomyLevel,
    EvidenceGrade,
)
from common.autonomy.evidence_service import EvidenceError, EvidenceService
from common.autonomy.verifier_registry import VerifierError, VerifierRegistry
from common.autonomy.calibration import CalibrationError, CalibrationTracker
from common.autonomy.authority import AutonomyAuthority, AutonomyError
from common.autonomy.release_bundle import (
    ReleaseBundleError,
    ReleaseBundleService,
)
from common.autonomy.wisdom_graph import WisdomError, WisdomGraph

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


def _evidence() -> EvidenceService:
    return EvidenceService(principal=_principal())


def _seed_qualifying(
    service: EvidenceService,
    count: int,
    domain: str = "trading",
    task_type: str = "price_forecast",
) -> list:
    return [
        service.record(
            grade=EvidenceGrade.VERIFIED_OUTCOME,
            domain=domain,
            task_type=task_type,
            observed_by="portfolio-verifier",
            content={"i": i},
            provenance=Provenance(source="verifier:portfolio-verifier"),
        )
        for i in range(count)
    ]


def _seed_calibration(
    tracker: CalibrationTracker,
    service: EvidenceService,
    count: int,
    accuracy: float,
    domain: str = "trading",
    task_type: str = "price_forecast",
    revision: str = "r1",
) -> None:
    correct_target = int(round(count * accuracy))
    for i in range(count):
        ev = service.record(
            grade=EvidenceGrade.VERIFIED_OUTCOME,
            domain=domain,
            task_type=task_type,
            observed_by="portfolio-verifier",
            provenance=Provenance(source="verifier:portfolio-verifier"),
        )
        tracker.observe(
            task_type=task_type,
            domain=domain,
            revision=revision,
            predicted_confidence=0.9,
            evidence=ev,
            was_correct=i < correct_target,
        )


# ═══════════════════════════════════════════════════════════════════
# EXIT GATE 1: self-generated text or simulation cannot grant trust
# ═══════════════════════════════════════════════════════════════════

def test_grade_qualification_rules():
    check("external_qualifies", EvidenceGrade.EXTERNAL_OBSERVED.qualifies())
    check("verified_qualifies", EvidenceGrade.VERIFIED_OUTCOME.qualifies())
    check("human_qualifies", EvidenceGrade.HUMAN_CONFIRMED.qualifies())
    check("model_generated_never_qualifies",
          not EvidenceGrade.MODEL_GENERATED.qualifies())
    check("simulated_never_qualifies",
          not EvidenceGrade.SIMULATED.qualifies())
    check("unknown_never_qualifies", not EvidenceGrade.UNKNOWN.qualifies())


def test_model_output_cannot_be_relabelled():
    """A caller cannot label its own model output as external observation."""
    service = _evidence()

    record = service.record(
        grade=EvidenceGrade.EXTERNAL_OBSERVED,
        domain="trading",
        task_type="forecast",
        observed_by="llm:claude",
        provenance=Provenance(source="llm:claude-fable-5"),
    )
    check("llm_source_downgraded",
          record.grade == EvidenceGrade.MODEL_GENERATED)
    check("llm_downgraded_not_qualifying", not record.grade.qualifies())

    record2 = service.record(
        grade=EvidenceGrade.VERIFIED_OUTCOME,
        domain="trading",
        task_type="forecast",
        observed_by="kai:self-reflection",
        provenance=Provenance(source="kai:self"),
    )
    check("self_source_downgraded",
          record2.grade == EvidenceGrade.MODEL_GENERATED)

    check("no_qualifying_from_self",
          len(service.qualifying_evidence(domain="trading")) == 0)


def test_simulation_cannot_be_relabelled():
    service = _evidence()

    record = service.record(
        grade=EvidenceGrade.VERIFIED_OUTCOME,
        domain="trading",
        task_type="paper_trade",
        observed_by="paper-trader",
        provenance=Provenance(source="paper-trader"),
    )
    check("paper_trader_downgraded", record.grade == EvidenceGrade.SIMULATED)
    check("simulated_not_qualifying", not record.grade.qualifies())

    record2 = service.record(
        grade=EvidenceGrade.EXTERNAL_OBSERVED,
        domain="trading",
        task_type="backtest",
        observed_by="backtester",
        provenance=Provenance(source="simulation:backtest-engine"),
    )
    check("simulation_source_downgraded",
          record2.grade == EvidenceGrade.SIMULATED)

    check("no_qualifying_from_simulation",
          len(service.qualifying_evidence(domain="trading")) == 0)


def test_self_generated_cannot_grant_autonomy():
    """The full chain: model output cannot produce an autonomy grant."""
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    authority = AutonomyAuthority(_principal(), service, tracker)

    for i in range(200):
        service.record(
            grade=EvidenceGrade.EXTERNAL_OBSERVED,
            domain="trading",
            task_type="forecast",
            observed_by="llm:claude",
            provenance=Provenance(source="llm:claude"),
        )

    check("many_records_stored", service.count == 200)
    check("none_qualify", len(service.qualifying_evidence(domain="trading")) == 0)

    qualified, reason = authority.check_qualification(
        level=AutonomyLevel.A1_OBSERVE,
        capability="market_read",
        domain="trading",
        task_type="forecast",
        revision="r1",
        independent_verifier_count=5,
    )
    check("self_generated_fails_qualification", not qualified)
    check("self_generated_reason", "insufficient qualifying evidence" in reason)

    try:
        authority.grant(
            level=AutonomyLevel.A1_OBSERVE,
            capability="market_read",
            domain="trading",
            task_type="forecast",
            revision="r1",
            granted_by="dainius",
            max_invocations=10,
            independent_verifier_count=5,
        )
        check("self_generated_grant_rejected", False, "should have raised")
    except AutonomyError as e:
        check("self_generated_grant_rejected", "not qualified" in str(e))


def test_calibration_rejects_non_qualifying():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())

    sim = service.record(
        grade=EvidenceGrade.SIMULATED,
        domain="trading",
        task_type="forecast",
        observed_by="backtester",
    )
    record = tracker.observe(
        task_type="forecast", domain="trading", revision="r1",
        predicted_confidence=0.9, evidence=sim, was_correct=True,
    )

    check("non_qualifying_not_counted", record.total_predictions == 0)
    check("non_qualifying_tracked", record.rejected_non_qualifying == 1)
    check("non_qualifying_accuracy_zero", record.accuracy == 0.0)
    check("non_qualifying_no_outcomes", record.qualifying_outcomes == 0)


# ═══════════════════════════════════════════════════════════════════
# Immutable evidence service
# ═══════════════════════════════════════════════════════════════════

def test_evidence_is_append_only():
    service = _evidence()
    original = service.record(
        grade=EvidenceGrade.EXTERNAL_OBSERVED,
        domain="weather", task_type="temp_read",
        observed_by="weather-service",
        content={"temp": 20},
    )

    corrected = service.correct(
        original.id,
        grade=EvidenceGrade.EXTERNAL_OBSERVED,
        content={"temp": 22},
        observed_by="weather-service",
    )

    check("correction_is_new_record", corrected.id != original.id)
    check("original_unchanged", service.get(original.id).content == {"temp": 20})
    check("original_superseded", service.is_superseded(original.id))
    check("correction_points_back", corrected.supersedes == original.id)
    check("both_records_retained", service.count == 2)

    active = service.all_evidence(domain="weather")
    check("only_correction_active", len(active) == 1)
    check("active_is_correction", active[0].id == corrected.id)


def test_evidence_lineage():
    service = _evidence()
    v1 = service.record(
        grade=EvidenceGrade.EXTERNAL_OBSERVED, domain="d", task_type="t",
        observed_by="sensor", content={"v": 1},
    )
    v2 = service.correct(v1.id, EvidenceGrade.EXTERNAL_OBSERVED, {"v": 2}, "sensor")
    v3 = service.correct(v2.id, EvidenceGrade.EXTERNAL_OBSERVED, {"v": 3}, "sensor")

    lineage = service.lineage(v3.id)
    check("lineage_length", len(lineage) == 3)
    check("lineage_oldest_first", lineage[0].id == v1.id)
    check("lineage_newest_last", lineage[2].id == v3.id)


def test_double_supersede_rejected():
    service = _evidence()
    original = service.record(
        grade=EvidenceGrade.EXTERNAL_OBSERVED, domain="d", task_type="t",
        observed_by="sensor",
    )
    service.correct(original.id, EvidenceGrade.EXTERNAL_OBSERVED, {}, "sensor")

    try:
        service.correct(original.id, EvidenceGrade.EXTERNAL_OBSERVED, {}, "sensor")
        check("double_supersede_rejected", False, "should have raised")
    except EvidenceError as e:
        check("double_supersede_rejected", "already superseded" in str(e))


def test_evidence_validation():
    service = _evidence()
    for field, kwargs in [
        ("domain", {"domain": "", "task_type": "t", "observed_by": "s"}),
        ("task_type", {"domain": "d", "task_type": "", "observed_by": "s"}),
        ("observed_by", {"domain": "d", "task_type": "t", "observed_by": ""}),
    ]:
        try:
            service.record(grade=EvidenceGrade.EXTERNAL_OBSERVED, **kwargs)
            check(f"empty_{field}_rejected", False, "should have raised")
        except EvidenceError:
            check(f"empty_{field}_rejected", True)


def test_grade_breakdown():
    service = _evidence()
    service.record(grade=EvidenceGrade.EXTERNAL_OBSERVED, domain="d",
                   task_type="t", observed_by="sensor")
    service.record(grade=EvidenceGrade.EXTERNAL_OBSERVED, domain="d",
                   task_type="t", observed_by="sensor")
    service.record(grade=EvidenceGrade.SIMULATED, domain="d",
                   task_type="t", observed_by="sim")

    breakdown = service.grade_breakdown(domain="d")
    check("breakdown_external", breakdown.get("external_observed") == 2)
    check("breakdown_simulated", breakdown.get("simulated") == 1)


# ═══════════════════════════════════════════════════════════════════
# Verifier registry — no self-verification
# ═══════════════════════════════════════════════════════════════════

def test_verifier_registration():
    registry = VerifierRegistry(principal=_principal())
    reg = registry.register(
        "portfolio-verifier", "Portfolio Verifier",
        ["trading"], "finance-independent",
    )
    check("verifier_registered", reg is not None)
    check("verifier_active", reg.active)
    check("verifier_group", reg.independence_group == "finance-independent")

    try:
        registry.register("portfolio-verifier", "dup", ["trading"], "g")
        check("duplicate_verifier_rejected", False, "should have raised")
    except VerifierError as e:
        check("duplicate_verifier_rejected", "already registered" in str(e))


def test_verifier_validation():
    registry = VerifierRegistry(principal=_principal())
    for name, args in [
        ("empty_identity", ("", "n", ["d"], "g")),
        ("empty_domains", ("v", "n", [], "g")),
        ("empty_group", ("v", "n", ["d"], "")),
    ]:
        try:
            registry.register(*args)
            check(f"verifier_{name}_rejected", False, "should have raised")
        except VerifierError:
            check(f"verifier_{name}_rejected", True)


def test_self_verification_rejected():
    """An actuator cannot verify its own success."""
    registry = VerifierRegistry(principal=_principal())
    registry.register("paper-trader", "Paper Trader", ["trading"], "execution")
    registry.set_actuator_group("paper-trader", "execution")

    allowed, reason = registry.can_verify("paper-trader", "paper-trader", "trading")
    check("self_verification_rejected", not allowed)
    check("self_verification_reason", "self-verification" in reason)

    try:
        registry.verify(
            "paper-trader", "paper-trader", "trading",
            "wf-1", "rc-1", VerificationVerdict.CONFIRMED,
        )
        check("self_verify_raises", False, "should have raised")
    except VerifierError as e:
        check("self_verify_raises", "self-verification" in str(e))


def test_correlated_verifier_rejected():
    """A verifier sharing the actuator's independence group does not count."""
    registry = VerifierRegistry(principal=_principal())
    registry.register("trade-checker", "Trade Checker", ["trading"], "execution")
    registry.set_actuator_group("paper-trader", "execution")

    allowed, reason = registry.can_verify("trade-checker", "paper-trader", "trading")
    check("correlated_verifier_rejected", not allowed)
    check("correlated_reason", "shares independence group" in reason)


def test_independent_verifier_accepted():
    registry = VerifierRegistry(principal=_principal())
    registry.register(
        "portfolio-verifier", "Portfolio Verifier",
        ["trading"], "finance-independent",
    )
    registry.set_actuator_group("paper-trader", "execution")

    allowed, reason = registry.can_verify(
        "portfolio-verifier", "paper-trader", "trading"
    )
    check("independent_verifier_allowed", allowed)
    check("independent_reason", reason == "independent")

    outcome = registry.verify(
        "portfolio-verifier", "paper-trader", "trading",
        "wf-1", "rc-1", VerificationVerdict.CONFIRMED,
        expected_state={"x": 1}, actual_state={"x": 1},
    )
    check("outcome_produced", outcome is not None)
    check("outcome_verdict", outcome.verdict == VerificationVerdict.CONFIRMED)
    check("outcome_carries_group",
          outcome.provenance.independence_group == "finance-independent")


def test_domain_scoping():
    registry = VerifierRegistry(principal=_principal())
    registry.register("weather-verifier", "Weather", ["weather"], "env")

    allowed, reason = registry.can_verify("weather-verifier", "some-actuator", "trading")
    check("wrong_domain_rejected", not allowed)
    check("wrong_domain_reason", "does not cover domain" in reason)


def test_suspended_verifier_rejected():
    registry = VerifierRegistry(principal=_principal())
    registry.register("v1", "V1", ["trading"], "g1")
    registry.suspend("v1", "failed audit")

    allowed, reason = registry.can_verify("v1", "actuator", "trading")
    check("suspended_verifier_rejected", not allowed)
    check("suspended_reason", "suspended" in reason)


def test_distinct_group_counting():
    """A panel that all shares one group counts as one verifier."""
    registry = VerifierRegistry(principal=_principal())
    registry.register("v1", "V1", ["trading"], "group-a")
    registry.register("v2", "V2", ["trading"], "group-a")
    registry.register("v3", "V3", ["trading"], "group-b")
    registry.set_actuator_group("paper-trader", "execution")

    count = registry.independent_verifier_count("paper-trader", "trading")
    check("distinct_groups_counted", count == 2, f"got {count}")

    outcomes = [
        registry.verify(v, "paper-trader", "trading", "wf", "rc",
                        VerificationVerdict.CONFIRMED)
        for v in ("v1", "v2")
    ]
    check("correlated_panel_counts_once", registry.distinct_groups(outcomes) == 1)

    outcomes.append(
        registry.verify("v3", "paper-trader", "trading", "wf", "rc",
                        VerificationVerdict.CONFIRMED)
    )
    check("diverse_panel_counts_all", registry.distinct_groups(outcomes) == 2)


# ═══════════════════════════════════════════════════════════════════
# Calibration by task / domain / revision
# ═══════════════════════════════════════════════════════════════════

def test_calibration_accuracy():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())

    for i in range(10):
        ev = service.record(
            grade=EvidenceGrade.VERIFIED_OUTCOME, domain="trading",
            task_type="forecast", observed_by="verifier",
            provenance=Provenance(source="verifier:portfolio"),
        )
        tracker.observe("forecast", "trading", "r1", 0.8, ev, was_correct=i < 8)

    record = tracker.get("forecast", "trading", "r1")
    check("calibration_total", record.total_predictions == 10)
    check("calibration_correct", record.correct_predictions == 8)
    check("calibration_accuracy", abs(record.accuracy - 0.8) < 1e-9)
    check("calibration_brier_computed", record.brier_score > 0)


def test_calibration_keyed_by_revision():
    """A new code revision starts uncalibrated rather than inheriting."""
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())

    _seed_calibration(tracker, service, 20, 1.0, revision="r1")
    check("r1_calibrated",
          tracker.accuracy("price_forecast", "trading", "r1") == 1.0)
    check("r2_uncalibrated",
          tracker.accuracy("price_forecast", "trading", "r2") == 0.0)
    check("r2_no_outcomes",
          tracker.qualifying_count("price_forecast", "trading", "r2") == 0)


def test_calibration_keyed_by_domain_and_task():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())

    _seed_calibration(tracker, service, 10, 1.0,
                      domain="trading", task_type="forecast", revision="r1")
    check("trading_calibrated",
          tracker.accuracy("forecast", "trading", "r1") == 1.0)
    check("other_domain_uncalibrated",
          tracker.accuracy("forecast", "weather", "r1") == 0.0)
    check("other_task_uncalibrated",
          tracker.accuracy("classify", "trading", "r1") == 0.0)


def test_calibration_confidence_range():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    ev = service.record(
        grade=EvidenceGrade.VERIFIED_OUTCOME, domain="d", task_type="t",
        observed_by="verifier", provenance=Provenance(source="verifier:v"),
    )

    for bad in (-0.1, 1.1):
        try:
            tracker.observe("t", "d", "r1", bad, ev, True)
            check(f"confidence_{bad}_rejected", False, "should have raised")
        except CalibrationError:
            check(f"confidence_{bad}_rejected", True)


# ═══════════════════════════════════════════════════════════════════
# EXIT GATE 3: autonomy is bounded, expiring and revocable
# ═══════════════════════════════════════════════════════════════════

def _qualified_authority(
    outcomes: int = 30, accuracy: float = 1.0
) -> tuple:
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, outcomes, accuracy,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)
    return service, tracker, authority


def test_grant_is_bounded():
    _, _, authority = _qualified_authority()
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=3,
        independent_verifier_count=2,
    )

    for i in range(3):
        authority.consume_grant(grant.id, "market_read", "trading")
    check("grant_fully_consumed", grant.invocations_used == 3)

    valid, reason = authority.check_grant(grant.id, "market_read", "trading")
    check("exhausted_grant_invalid", not valid)
    check("exhausted_reason", "exhausted" in reason)

    try:
        authority.consume_grant(grant.id, "market_read", "trading")
        check("exhausted_consume_rejected", False, "should have raised")
    except AutonomyError as e:
        check("exhausted_consume_rejected", "exhausted" in str(e))


def test_grant_expires():
    _, _, authority = _qualified_authority()
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=100,
        duration_seconds=1, independent_verifier_count=2,
    )

    valid, _ = authority.check_grant(grant.id, "market_read", "trading")
    check("grant_valid_before_expiry", valid)

    time.sleep(1.1)
    valid, reason = authority.check_grant(grant.id, "market_read", "trading")
    check("grant_expired", not valid)
    check("expired_reason", "expired" in reason)

    try:
        authority.consume_grant(grant.id, "market_read", "trading")
        check("expired_consume_rejected", False, "should have raised")
    except AutonomyError as e:
        check("expired_consume_rejected", "expired" in str(e))


def test_grant_is_revocable():
    _, _, authority = _qualified_authority()
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=100,
        independent_verifier_count=2,
    )

    valid, _ = authority.check_grant(grant.id, "market_read", "trading")
    check("grant_valid_before_revoke", valid)

    authority.revoke(grant.id, "policy change")
    valid, reason = authority.check_grant(grant.id, "market_read", "trading")
    check("revoked_grant_invalid", not valid)
    check("revoked_reason", "revoked" in reason)
    check("revoked_timestamped", authority.get_grant(grant.id).revoked_at is not None)


def test_revoke_all_is_emergency_stop():
    _, _, authority = _qualified_authority()
    grants = [
        authority.grant(
            level=AutonomyLevel.A2_REVERSIBLE, capability=f"cap_{i}",
            domain="trading", task_type="price_forecast", revision="r1",
            granted_by="dainius", max_invocations=10,
            independent_verifier_count=2,
        )
        for i in range(3)
    ]

    count = authority.revoke_all("emergency stop")
    check("revoke_all_count", count == 3)
    check("no_active_grants", authority.active_grants() == [])
    for g in grants:
        valid, _ = authority.check_grant(g.id, g.capability, "trading")
        check(f"grant_{g.capability}_revoked", not valid)


def test_grant_duration_capped_by_level():
    _, _, authority = _qualified_authority(outcomes=120, accuracy=1.0)
    max_a2 = AUTONOMY_REQUIREMENTS[AutonomyLevel.A2_REVERSIBLE]["max_grant_seconds"]

    try:
        authority.grant(
            level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
            domain="trading", task_type="price_forecast", revision="r1",
            granted_by="dainius", max_invocations=10,
            duration_seconds=max_a2 + 1, independent_verifier_count=2,
        )
        check("duration_cap_enforced", False, "should have raised")
    except AutonomyError as e:
        check("duration_cap_enforced", "exceeds" in str(e))


def test_a0_cannot_hold_grant():
    _, _, authority = _qualified_authority()
    try:
        authority.grant(
            level=AutonomyLevel.A0_NONE, capability="anything",
            domain="trading", task_type="price_forecast", revision="r1",
            granted_by="dainius", max_invocations=1,
        )
        check("a0_no_standing_grant", False, "should have raised")
    except AutonomyError as e:
        check("a0_no_standing_grant", "cannot hold a standing grant" in str(e))


def test_grant_validation():
    _, _, authority = _qualified_authority()
    try:
        authority.grant(
            level=AutonomyLevel.A2_REVERSIBLE, capability="c", domain="trading",
            task_type="price_forecast", revision="r1", granted_by="",
            max_invocations=1, independent_verifier_count=2,
        )
        check("anonymous_grant_rejected", False, "should have raised")
    except AutonomyError as e:
        check("anonymous_grant_rejected", "anonymously" in str(e))

    try:
        authority.grant(
            level=AutonomyLevel.A2_REVERSIBLE, capability="c", domain="trading",
            task_type="price_forecast", revision="r1", granted_by="dainius",
            max_invocations=0, independent_verifier_count=2,
        )
        check("zero_invocations_rejected", False, "should have raised")
    except AutonomyError as e:
        check("zero_invocations_rejected", "at least one invocation" in str(e))


# ═══════════════════════════════════════════════════════════════════
# EXIT GATE 2: high-consequence attack chains
# ═══════════════════════════════════════════════════════════════════

def test_attack_grant_scope_escape_capability():
    """A grant for one capability cannot be used for another."""
    _, _, authority = _qualified_authority()
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=10,
        independent_verifier_count=2,
    )

    valid, reason = authority.check_grant(grant.id, "place_order", "trading")
    check("capability_scope_enforced", not valid)
    check("capability_scope_reason", "capability mismatch" in reason)

    try:
        authority.consume_grant(grant.id, "place_order", "trading")
        check("capability_escape_blocked", False, "should have raised")
    except AutonomyError:
        check("capability_escape_blocked", True)


def test_attack_grant_scope_escape_domain():
    """A grant for one domain cannot be used in another."""
    _, _, authority = _qualified_authority()
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=10,
        independent_verifier_count=2,
    )

    valid, reason = authority.check_grant(grant.id, "market_read", "medical")
    check("domain_scope_enforced", not valid)
    check("domain_scope_reason", "domain mismatch" in reason)


def test_attack_escalation_without_evidence():
    """A1-qualifying evidence cannot reach A4."""
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, 12, 1.0,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)

    qualified, _ = authority.check_qualification(
        AutonomyLevel.A1_OBSERVE, "market_read", "trading",
        "price_forecast", "r1", independent_verifier_count=1,
    )
    check("a1_qualified_with_12", qualified)

    confirmation = authority.confirm_value(
        subject_digest="abc123", subject_kind="autonomy_grant",
        prompt_shown="Grant A4 autonomy for trading?",
        confirmed=True, confirmed_by="dainius",
    )
    qualified, reason = authority.check_qualification(
        AutonomyLevel.A4_HIGH_CONSEQUENCE, "place_order", "trading",
        "price_forecast", "r1", independent_verifier_count=3,
        human_confirmation_id=confirmation.id,
    )
    check("a4_blocked_on_evidence", not qualified)
    check("a4_evidence_reason", "insufficient qualifying evidence" in reason)


def test_attack_a4_requires_human_confirmation():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, 120, 1.0,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)

    qualified, reason = authority.check_qualification(
        AutonomyLevel.A4_HIGH_CONSEQUENCE, "place_order", "trading",
        "price_forecast", "r1", independent_verifier_count=3,
    )
    check("a4_needs_confirmation", not qualified)
    check("a4_confirmation_reason", "human confirmation" in reason)

    confirmation = authority.confirm_value(
        subject_digest="d1", subject_kind="autonomy_grant",
        prompt_shown="Grant A4?", confirmed=True, confirmed_by="dainius",
    )
    qualified, reason = authority.check_qualification(
        AutonomyLevel.A4_HIGH_CONSEQUENCE, "place_order", "trading",
        "price_forecast", "r1", independent_verifier_count=3,
        human_confirmation_id=confirmation.id,
    )
    check("a4_qualified_with_confirmation", qualified, reason)


def test_attack_declined_confirmation_does_not_grant():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, 120, 1.0,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)

    declined = authority.confirm_value(
        subject_digest="d1", subject_kind="autonomy_grant",
        prompt_shown="Grant A4?", confirmed=False, confirmed_by="dainius",
    )
    qualified, reason = authority.check_qualification(
        AutonomyLevel.A4_HIGH_CONSEQUENCE, "place_order", "trading",
        "price_forecast", "r1", independent_verifier_count=3,
        human_confirmation_id=declined.id,
    )
    check("declined_confirmation_blocks", not qualified)
    check("declined_reason", "declined" in reason)


def test_attack_insufficient_verifier_diversity():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, 120, 1.0,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)
    confirmation = authority.confirm_value(
        subject_digest="d1", subject_kind="grant", prompt_shown="?",
        confirmed=True, confirmed_by="dainius",
    )

    qualified, reason = authority.check_qualification(
        AutonomyLevel.A4_HIGH_CONSEQUENCE, "place_order", "trading",
        "price_forecast", "r1", independent_verifier_count=2,
        human_confirmation_id=confirmation.id,
    )
    check("insufficient_verifiers_blocked", not qualified)
    check("verifier_diversity_reason", "independent verifiers" in reason)


def test_attack_calibration_below_threshold():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, 120, 0.60,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)

    qualified, reason = authority.check_qualification(
        AutonomyLevel.A2_REVERSIBLE, "market_read", "trading",
        "price_forecast", "r1", independent_verifier_count=2,
    )
    check("low_accuracy_blocked", not qualified)
    check("low_accuracy_reason", "accuracy" in reason)


def test_attack_stale_revision_calibration():
    """Calibration on an old revision does not authorise the new one."""
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    _seed_calibration(tracker, service, 120, 1.0,
                      task_type="price_forecast", revision="r1")
    authority = AutonomyAuthority(_principal(), service, tracker)

    qualified, _ = authority.check_qualification(
        AutonomyLevel.A2_REVERSIBLE, "market_read", "trading",
        "price_forecast", "r1", independent_verifier_count=2,
    )
    check("r1_qualified", qualified)

    qualified, reason = authority.check_qualification(
        AutonomyLevel.A2_REVERSIBLE, "market_read", "trading",
        "price_forecast", "r2", independent_verifier_count=2,
    )
    check("r2_not_qualified", not qualified)
    check("r2_reason_revision", "r2" in reason)


def test_attack_confirmation_replay():
    _, _, authority = _qualified_authority()
    authority.confirm_value(
        subject_digest="d1", subject_kind="grant", prompt_shown="?",
        confirmed=True, confirmed_by="dainius", nonce="nonce-1",
    )
    try:
        authority.confirm_value(
            subject_digest="d2", subject_kind="grant", prompt_shown="?",
            confirmed=True, confirmed_by="dainius", nonce="nonce-1",
        )
        check("confirmation_replay_rejected", False, "should have raised")
    except AutonomyError as e:
        check("confirmation_replay_rejected", "replay" in str(e))


def test_attack_confirmation_cannot_be_inferred():
    """Confirmation needs an explicit prompt, a named human and a digest."""
    _, _, authority = _qualified_authority()

    for name, kwargs in [
        ("anonymous", {"confirmed_by": "", "prompt_shown": "p", "subject_digest": "d"}),
        ("no_prompt", {"confirmed_by": "dainius", "prompt_shown": "", "subject_digest": "d"}),
        ("no_digest", {"confirmed_by": "dainius", "prompt_shown": "p", "subject_digest": ""}),
    ]:
        try:
            authority.confirm_value(
                subject_kind="grant", confirmed=True, **kwargs
            )
            check(f"confirmation_{name}_rejected", False, "should have raised")
        except AutonomyError:
            check(f"confirmation_{name}_rejected", True)


def test_effective_level_defaults_to_a0():
    _, _, authority = _qualified_authority()
    check("no_grant_means_a0",
          authority.effective_level("anything", "trading") == AutonomyLevel.A0_NONE)

    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="market_read",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=10,
        independent_verifier_count=2,
    )
    check("grant_raises_level",
          authority.effective_level("market_read", "trading")
          == AutonomyLevel.A2_REVERSIBLE)

    authority.revoke(grant.id, "done")
    check("revocation_drops_to_a0",
          authority.effective_level("market_read", "trading")
          == AutonomyLevel.A0_NONE)


# ═══════════════════════════════════════════════════════════════════
# Signed release bundles
# ═══════════════════════════════════════════════════════════════════

def test_release_bundle_sign_and_verify():
    service = ReleaseBundleService(_principal(), b"secret-key")
    bundle = service.sign(
        capability="place_order", code_revision="abc123",
        autonomy_level=AutonomyLevel.A3_SUPERVISED,
        domains=["trading"], signed_by="dainius",
    )

    valid, reason = service.verify(bundle, "place_order", "abc123", "trading")
    check("bundle_verifies", valid, reason)
    check("bundle_has_signature", len(bundle.signature) == 64)


def test_release_bundle_tampering_detected():
    service = ReleaseBundleService(_principal(), b"secret-key")
    bundle = service.sign(
        capability="place_order", code_revision="abc123",
        autonomy_level=AutonomyLevel.A3_SUPERVISED,
        domains=["trading"], signed_by="dainius",
    )

    bundle.autonomy_level = AutonomyLevel.A4_HIGH_CONSEQUENCE
    valid, reason = service.verify(bundle, "place_order", "abc123", "trading")
    check("level_tampering_detected", not valid)
    check("tampering_reason", "signature invalid" in reason)


def test_release_bundle_revision_binding():
    """A bundle signed for one revision does not carry to the next."""
    service = ReleaseBundleService(_principal(), b"secret-key")
    bundle = service.sign(
        capability="place_order", code_revision="abc123",
        autonomy_level=AutonomyLevel.A3_SUPERVISED,
        domains=["trading"], signed_by="dainius",
    )

    valid, reason = service.verify(bundle, "place_order", "def456", "trading")
    check("revision_binding_enforced", not valid)
    check("revision_reason", "revision mismatch" in reason)


def test_release_bundle_capability_binding():
    service = ReleaseBundleService(_principal(), b"secret-key")
    bundle = service.sign(
        capability="market_read", code_revision="abc123",
        autonomy_level=AutonomyLevel.A1_OBSERVE,
        domains=["trading"], signed_by="dainius",
    )

    valid, reason = service.verify(bundle, "place_order", "abc123", "trading")
    check("bundle_capability_binding", not valid)
    check("bundle_capability_reason", "capability mismatch" in reason)


def test_release_bundle_domain_scoping():
    service = ReleaseBundleService(_principal(), b"secret-key")
    bundle = service.sign(
        capability="place_order", code_revision="abc123",
        autonomy_level=AutonomyLevel.A3_SUPERVISED,
        domains=["trading"], signed_by="dainius",
    )

    valid, reason = service.verify(bundle, "place_order", "abc123", "medical")
    check("bundle_domain_scoping", not valid)
    check("bundle_domain_reason", "does not cover domain" in reason)


def test_release_bundle_expiry_and_revocation():
    service = ReleaseBundleService(_principal(), b"secret-key")

    expiring = service.sign(
        capability="c", code_revision="r", autonomy_level=AutonomyLevel.A1_OBSERVE,
        domains=["d"], signed_by="dainius", valid_for_seconds=1,
    )
    valid, _ = service.verify(expiring, "c", "r", "d")
    check("bundle_valid_before_expiry", valid)
    time.sleep(1.1)
    valid, reason = service.verify(expiring, "c", "r", "d")
    check("bundle_expired", not valid)
    check("bundle_expiry_reason", "expired" in reason)

    revocable = service.sign(
        capability="c2", code_revision="r", autonomy_level=AutonomyLevel.A1_OBSERVE,
        domains=["d"], signed_by="dainius",
    )
    service.revoke(revocable.id, "compromised")
    valid, reason = service.verify(revocable, "c2", "r", "d")
    check("bundle_revoked", not valid)
    check("bundle_revoked_reason", "revoked" in reason)


def test_release_bundle_wrong_key_fails():
    signer = ReleaseBundleService(_principal(), b"real-key")
    attacker = ReleaseBundleService(_principal(), b"forged-key")

    bundle = signer.sign(
        capability="place_order", code_revision="abc123",
        autonomy_level=AutonomyLevel.A3_SUPERVISED,
        domains=["trading"], signed_by="dainius",
    )

    valid, reason = attacker.verify(bundle, "place_order", "abc123", "trading")
    check("wrong_key_rejected", not valid)
    check("wrong_key_reason", "signature invalid" in reason)


def test_release_bundle_validation():
    service = ReleaseBundleService(_principal(), b"key")
    base = dict(
        capability="c", code_revision="r",
        autonomy_level=AutonomyLevel.A1_OBSERVE,
        domains=["d"], signed_by="dainius",
    )
    for name, override in [
        ("empty_capability", {"capability": ""}),
        ("empty_revision", {"code_revision": ""}),
        ("anonymous", {"signed_by": ""}),
        ("no_domains", {"domains": []}),
    ]:
        try:
            service.sign(**{**base, **override})
            check(f"bundle_{name}_rejected", False, "should have raised")
        except ReleaseBundleError:
            check(f"bundle_{name}_rejected", True)

    try:
        ReleaseBundleService(_principal(), b"")
        check("empty_key_rejected", False, "should have raised")
    except ReleaseBundleError:
        check("empty_key_rejected", True)


# ═══════════════════════════════════════════════════════════════════
# Wisdom graph — lineage and contradiction
# ═══════════════════════════════════════════════════════════════════

def test_wisdom_requires_qualifying_support():
    service = _evidence()
    graph = WisdomGraph(_principal(), service)

    sim = service.record(
        grade=EvidenceGrade.SIMULATED, domain="trading",
        task_type="backtest", observed_by="backtester",
    )
    node = graph.add(
        "Momentum works in bull markets", "trading",
        evidence_ids=[sim.id], confidence=0.95,
    )
    check("simulated_support_zero_confidence", node.confidence == 0.0)

    real = service.record(
        grade=EvidenceGrade.VERIFIED_OUTCOME, domain="trading",
        task_type="live_trade", observed_by="portfolio-verifier",
        provenance=Provenance(source="verifier:portfolio"),
    )
    node2 = graph.add(
        "Momentum works in bull markets", "trading",
        evidence_ids=[real.id], confidence=0.95,
    )
    check("verified_support_keeps_confidence", node2.confidence == 0.95)


def test_wisdom_confidence_not_laundered_through_chain():
    """A chain of inferences from simulation stays at zero confidence."""
    service = _evidence()
    graph = WisdomGraph(_principal(), service)

    sim = service.record(
        grade=EvidenceGrade.SIMULATED, domain="trading",
        task_type="backtest", observed_by="backtester",
    )
    n1 = graph.add("Base claim", "trading", evidence_ids=[sim.id], confidence=0.9)
    n2 = graph.add("Derived claim", "trading", derived_from=[n1.id], confidence=0.9)
    n3 = graph.add("Further claim", "trading", derived_from=[n2.id], confidence=0.9)

    check("chain_base_zero", n1.confidence == 0.0)
    check("chain_derived_zero", n2.confidence == 0.0)
    check("chain_further_zero", n3.confidence == 0.0)


def test_wisdom_lineage():
    service = _evidence()
    graph = WisdomGraph(_principal(), service)
    real = service.record(
        grade=EvidenceGrade.VERIFIED_OUTCOME, domain="d", task_type="t",
        observed_by="verifier", provenance=Provenance(source="verifier:v"),
    )

    n1 = graph.add("Root", "d", evidence_ids=[real.id], confidence=0.8)
    n2 = graph.add("Middle", "d", derived_from=[n1.id], confidence=0.7)
    n3 = graph.add("Leaf", "d", derived_from=[n2.id], confidence=0.6)

    lineage = graph.lineage(n3.id)
    check("lineage_has_ancestors", len(lineage) == 2)
    check("lineage_nearest_first", lineage[0].id == n2.id)
    check("lineage_includes_root", lineage[1].id == n1.id)
    check("derived_inherits_support", n3.confidence == 0.6)


def test_wisdom_contradiction():
    service = _evidence()
    graph = WisdomGraph(_principal(), service)
    n1 = graph.add("Rates will rise", "econ")
    n2 = graph.add("Rates will fall", "econ")

    graph.record_contradiction(n1.id, n2.id)
    check("contradiction_recorded_a", n2.id in graph.get(n1.id).contradicts)
    check("contradiction_symmetric", n1.id in graph.get(n2.id).contradicts)
    check("contradiction_listed", len(graph.contradictions()) == 1)

    try:
        graph.record_contradiction(n1.id, n1.id)
        check("self_contradiction_rejected", False, "should have raised")
    except WisdomError:
        check("self_contradiction_rejected", True)


def test_wisdom_supersession():
    service = _evidence()
    graph = WisdomGraph(_principal(), service)
    old = graph.add("Old understanding", "d")
    new = graph.add("New understanding", "d")

    graph.supersede(old.id, new.id)
    check("supersede_recorded", graph.get(old.id).superseded_by == new.id)

    active = graph.active_nodes(domain="d")
    check("superseded_not_active", len(active) == 1)
    check("active_is_new", active[0].id == new.id)


def test_wisdom_validation():
    service = _evidence()
    graph = WisdomGraph(_principal(), service)

    try:
        graph.add("", "d")
        check("empty_statement_rejected", False, "should have raised")
    except WisdomError:
        check("empty_statement_rejected", True)

    try:
        graph.add("s", "d", evidence_ids=["nonexistent"])
        check("unknown_evidence_rejected", False, "should have raised")
    except WisdomError:
        check("unknown_evidence_rejected", True)

    try:
        graph.add("s", "d", derived_from=["nonexistent"])
        check("unknown_parent_rejected", False, "should have raised")
    except WisdomError:
        check("unknown_parent_rejected", True)

    try:
        graph.add("s", "d", confidence=1.5)
        check("bad_confidence_rejected", False, "should have raised")
    except WisdomError:
        check("bad_confidence_rejected", True)


# ═══════════════════════════════════════════════════════════════════
# Integration: full requalification cycle
# ═══════════════════════════════════════════════════════════════════

def test_full_requalification_cycle():
    service = _evidence()
    tracker = CalibrationTracker(principal=_principal())
    registry = VerifierRegistry(principal=_principal())
    authority = AutonomyAuthority(_principal(), service, tracker)

    registry.register("v1", "V1", ["trading"], "group-a")
    registry.register("v2", "V2", ["trading"], "group-b")
    registry.set_actuator_group("paper-trader", "execution")
    verifier_count = registry.independent_verifier_count("paper-trader", "trading")
    check("cycle_two_independent_verifiers", verifier_count == 2)

    _seed_calibration(tracker, service, 60, 0.95,
                      task_type="price_forecast", revision="r1")

    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="paper_trade_open",
        domain="trading", task_type="price_forecast", revision="r1",
        granted_by="dainius", max_invocations=5,
        independent_verifier_count=verifier_count,
    )
    check("cycle_grant_issued", grant is not None)
    check("cycle_grant_bounded", grant.max_invocations == 5)
    check("cycle_grant_expires", grant.expires_at_grant > grant.granted_at)
    check("cycle_grant_cites_evidence", len(grant.evidence_ids) >= 60)
    check("cycle_grant_cites_calibration", grant.calibration_id is not None)

    authority.consume_grant(grant.id, "paper_trade_open", "trading")
    check("cycle_grant_consumed", authority.get_grant(grant.id).invocations_used == 1)

    # A new code revision invalidates the track record.
    qualified, reason = authority.check_qualification(
        AutonomyLevel.A2_REVERSIBLE, "paper_trade_open", "trading",
        "price_forecast", "r2", independent_verifier_count=verifier_count,
    )
    check("cycle_new_revision_requalifies", not qualified)
    check("cycle_requalify_reason", "r2" in reason)


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_grade_qualification_rules()
    test_model_output_cannot_be_relabelled()
    test_simulation_cannot_be_relabelled()
    test_self_generated_cannot_grant_autonomy()
    test_calibration_rejects_non_qualifying()
    test_evidence_is_append_only()
    test_evidence_lineage()
    test_double_supersede_rejected()
    test_evidence_validation()
    test_grade_breakdown()
    test_verifier_registration()
    test_verifier_validation()
    test_self_verification_rejected()
    test_correlated_verifier_rejected()
    test_independent_verifier_accepted()
    test_domain_scoping()
    test_suspended_verifier_rejected()
    test_distinct_group_counting()
    test_calibration_accuracy()
    test_calibration_keyed_by_revision()
    test_calibration_keyed_by_domain_and_task()
    test_calibration_confidence_range()
    test_grant_is_bounded()
    test_grant_expires()
    test_grant_is_revocable()
    test_revoke_all_is_emergency_stop()
    test_grant_duration_capped_by_level()
    test_a0_cannot_hold_grant()
    test_grant_validation()
    test_attack_grant_scope_escape_capability()
    test_attack_grant_scope_escape_domain()
    test_attack_escalation_without_evidence()
    test_attack_a4_requires_human_confirmation()
    test_attack_declined_confirmation_does_not_grant()
    test_attack_insufficient_verifier_diversity()
    test_attack_calibration_below_threshold()
    test_attack_stale_revision_calibration()
    test_attack_confirmation_replay()
    test_attack_confirmation_cannot_be_inferred()
    test_effective_level_defaults_to_a0()
    test_release_bundle_sign_and_verify()
    test_release_bundle_tampering_detected()
    test_release_bundle_revision_binding()
    test_release_bundle_capability_binding()
    test_release_bundle_domain_scoping()
    test_release_bundle_expiry_and_revocation()
    test_release_bundle_wrong_key_fails()
    test_release_bundle_validation()
    test_wisdom_requires_qualifying_support()
    test_wisdom_confidence_not_laundered_through_chain()
    test_wisdom_lineage()
    test_wisdom_contradiction()
    test_wisdom_supersession()
    test_wisdom_validation()
    test_full_requalification_cycle()

    print(f"\n{'='*60}")
    print(f"UH-8 Autonomy Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
