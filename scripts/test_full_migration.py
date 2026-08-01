"""Full-catalogue migration — tiers 2-8 (UH tracker gap G-09).

Tier 1 is read-only; everything above it causes real side effects.  Three
properties matter more here than they did for reads:

  - **Legacy closure is verified, not asserted.**  `disable_legacy_path()`
    alone is a promise. The migration driver now requires the legacy path
    to be *provably* closed first, checked against the source tree.
  - **Side effects are declared.**  Each mutating action names what it
    changes, and the receipt records it.
  - **Uncertain effects are flagged.**  A POST that errors may still have
    caused its effect; that is recorded so reconciliation is possible
    rather than assumed away.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.action import ActionProposal
from common.actuator_registry.catalog import build_catalog
from common.actuator_registry.legacy_verification import (
    LEGACY_CHECKS,
    open_legacy_paths,
    verify_all,
    verify_legacy_closed,
)
from common.actuator_registry.migration import migrate_tier, migration_plan
from common.actuator_registry.mutating_handlers import (
    MUTATING_ENDPOINTS,
    MutatingHandlerError,
    attach_all_handlers,
    attach_mutating_handlers,
    build_mutating_handler,
    side_effects_for,
)
from common.actuator_registry.registry import (
    ActuatorDispatchError,
    MigrationStatus,
    MigrationTier,
)
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge

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


def _catalog(with_handlers: bool = True, post=None):
    catalog = build_catalog(_principal())
    if with_handlers:
        attach_all_handlers(
            catalog,
            http_get=lambda u: {"read": True},
            http_post=post or (lambda u, b, h: {"posted": True}),
        )
    return catalog


def _migrate_all(catalog, **kwargs):
    results = {}
    for tier in MigrationTier:
        results[tier] = migrate_tier(catalog, tier, _principal(), **kwargs)
    return results


# ═══════════════════════════════════════════════════════════════════
# 1. Legacy closure is a verified fact
# ═══════════════════════════════════════════════════════════════════

def test_all_legacy_paths_verified_closed():
    results = verify_all()
    still_open = {n: r for n, (c, r) in results.items() if not c}
    check("all_legacy_paths_closed", not still_open, str(still_open))
    check("legacy_checks_cover_catalogue", len(results) == len(LEGACY_CHECKS))
    check("open_legacy_paths_empty", open_legacy_paths() == {})


def test_legacy_closure_explains_itself():
    for actuator in ("db-restore", "browser-actor", "executor-shell",
                     "vault-sync", "telegram-bot"):
        closed, reason = verify_legacy_closed(actuator)
        check(f"{actuator}_closed", closed, reason)
        check(f"{actuator}_reason_meaningful", len(reason) > 10, reason)


def test_paper_trader_legacy_is_symbol_removal():
    """auto_trade() was deleted outright, not merely authenticated."""
    closed, reason = verify_legacy_closed("paper-trader")
    check("paper_trader_closed", closed, reason)
    check("paper_trader_symbol_gone", "removed" in reason, reason)


def test_unknown_actuator_passes_trivially():
    closed, reason = verify_legacy_closed("no-such-actuator")
    check("no_legacy_path_passes", closed)
    check("no_legacy_path_explained", "no legacy path" in reason)


def test_verification_catches_an_unauthenticated_route():
    """The checker must actually be able to fail."""
    from common.actuator_registry.legacy_verification import _route_is_authenticated

    # /health is deliberately unauthenticated everywhere.
    closed, reason = _route_is_authenticated(
        "backup-service/app.py", "get", "/health"
    )
    check("unauthenticated_route_detected", not closed, reason)
    check("detection_explains", "unauthenticated" in reason, reason)


def test_verification_catches_missing_route():
    from common.actuator_registry.legacy_verification import _route_is_authenticated

    closed, reason = _route_is_authenticated(
        "backup-service/app.py", "post", "/no-such-route"
    )
    check("missing_route_detected", not closed)
    check("missing_route_explains", "not found" in reason)


def test_migration_blocks_on_unverified_legacy():
    """A migration must refuse when the legacy path cannot be proven closed."""
    import common.actuator_registry.migration as mig

    original = mig.verify_legacy_closed
    mig.verify_legacy_closed = lambda name: (False, "pretend still open")
    try:
        catalog = _catalog()
        result = migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
        blocked_with_legacy = [
            i for i in result.blocked
            if "legacy path still open" in result.errors.get(i, "")
        ]
        # Tier 1 has no legacy paths, so nothing should block there.
        check("tier1_unaffected_by_legacy_check", result.ok, str(result.errors))

        tier3 = migrate_tier(catalog, MigrationTier.LOCAL_TEST, _principal())
        tier3 = migrate_tier(catalog, MigrationTier.DOCUMENT_READ, _principal())
        check("tier3_blocked_on_open_legacy",
              "browser-reader" in tier3.blocked, str(tier3.errors))
        check("tier3_error_names_legacy",
              "legacy path still open" in tier3.errors.get("browser-reader", ""))
    finally:
        mig.verify_legacy_closed = original


def test_verification_can_be_waived_explicitly():
    import common.actuator_registry.migration as mig

    original = mig.verify_legacy_closed
    mig.verify_legacy_closed = lambda name: (False, "pretend still open")
    try:
        catalog = _catalog()
        migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
        migrate_tier(catalog, MigrationTier.LOCAL_TEST, _principal())
        result = migrate_tier(
            catalog, MigrationTier.DOCUMENT_READ, _principal(),
            verify_legacy=False,
        )
        check("waiver_bypasses_verification", result.ok, str(result.errors))
    finally:
        mig.verify_legacy_closed = original


# ═══════════════════════════════════════════════════════════════════
# 2. Full-catalogue migration
# ═══════════════════════════════════════════════════════════════════

def test_all_tiers_migrate():
    catalog = _catalog()
    results = _migrate_all(catalog)

    for tier, result in results.items():
        check(f"tier{tier.value}_ok", result.ok,
              f"{tier.name}: {result.errors}")

    report = catalog.migration_report()
    check("everything_migrated",
          report["migrated"] == report["total_actuators"],
          f"{report['migrated']}/{report['total_actuators']}")
    check("nothing_remaining", report["remaining"] == 0)
    check("no_legacy_paths_open", report["legacy_paths_open"] == [])


def test_catalogue_size():
    catalog = _catalog(with_handlers=False)
    report = catalog.migration_report()
    check("catalogue_has_34", report["total_actuators"] == 34,
          str(report["total_actuators"]))
    check("all_eight_tiers", len(report["by_tier"]) == 8)


def test_every_actuator_has_a_handler():
    catalog = _catalog()
    missing = [
        e.identity for e in catalog.list_actuators()
        if catalog.handler_for(e.identity) is None
    ]
    check("every_actuator_has_handler", not missing, str(missing))


def test_plan_reports_complete_after_full_migration():
    catalog = _catalog()
    _migrate_all(catalog)
    plan = migration_plan(catalog)
    check("plan_complete", plan["complete"] is True)
    check("plan_no_next_tier", plan["next_tier"] is None)
    check("plan_no_candidates", plan["candidates"] == [])


def test_ordering_still_enforced_across_all_tiers():
    """Tier 8 must remain unreachable until 1-7 are done."""
    catalog = _catalog()
    from common.actuator_registry.migration import MigrationError

    try:
        migrate_tier(catalog, MigrationTier.FINANCIAL_DESTRUCTIVE, _principal())
        check("tier8_blocked_from_cold", False, "should have raised")
    except MigrationError as e:
        check("tier8_blocked_from_cold", "lower tiers" in str(e))


def test_migration_without_handlers_blocks_every_tier():
    catalog = _catalog(with_handlers=False)
    result = migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
    check("no_handlers_blocks", not result.ok)
    check("no_handlers_all_blocked", len(result.blocked) == 11)


# ═══════════════════════════════════════════════════════════════════
# 3. Mutating dispatch
# ═══════════════════════════════════════════════════════════════════

def _dispatch(catalog, bridge, actuator, action, params=None):
    proposal = ActionProposal(
        action_type=action, description="t", risk_tier=RiskTier.OBSERVE,
        rationale="t", alternatives=["n"], principal=_principal(),
        purpose="test", provenance=Provenance(source="test"),
    )
    approval = ApprovalGate().approve(proposal, "dainius", _principal())
    cap = bridge.issue(
        proposal, approval, actuator, action, _principal(),
        parameters=params or {},
    )
    bridge.consume(cap.id, actuator, _principal())
    return catalog.dispatch(cap, actuator, action, "wf", _principal())


def test_mutating_dispatch_posts():
    calls = []
    catalog = _catalog(post=lambda u, b, h: calls.append((u, b)) or {"ok": 1})
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    receipt = _dispatch(catalog, bridge, "notify-service", "notify_desktop",
                        {"title": "hi", "body": "there"})
    check("mutating_called_service", len(calls) == 1)
    check("mutating_hit_notify", calls and calls[0][0].endswith("/notify"))
    check("mutating_sent_body", calls and calls[0][1] == {"title": "hi", "body": "there"})
    check("mutating_receipt_ok", receipt.result.get("ok") is True)
    check("mutating_method_recorded", receipt.result.get("method") == "POST")


def test_side_effects_declared_on_receipt():
    catalog = _catalog()
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    receipt = _dispatch(catalog, bridge, "db-restore", "db_restore",
                        {"backup_file": "b.sql"})
    effects = receipt.result.get("side_effects", [])
    check("db_restore_declares_effects", "database_overwrite" in effects)
    check("db_restore_marked_irreversible", "irreversible" in effects)


def test_side_effects_lookup():
    check("notify_effect",
          "desktop_notification" in side_effects_for("notify-service", "notify_desktop"))
    check("browser_click_irreversible",
          "irreversible" in side_effects_for("browser-actor", "browser_click"))
    check("noop_has_no_effects",
          side_effects_for("executor-sandbox", "noop") == [])


def test_failed_post_flags_uncertain_effect():
    """A POST that errors may still have caused its effect."""
    def exploding(url, body, headers):
        raise ConnectionError("connection reset mid-request")

    catalog = _catalog(post=exploding)
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    receipt = _dispatch(catalog, bridge, "db-restore", "db_restore",
                        {"backup_file": "b.sql"})
    check("failed_post_not_ok", receipt.result.get("ok") is False)
    check("failed_post_flags_uncertainty",
          receipt.result.get("effect_uncertain") is True)
    check("failed_post_keeps_effects",
          "database_overwrite" in receipt.result.get("side_effects", []))
    check("failed_post_records_error",
          "ConnectionError" in receipt.result.get("error", ""))


def test_effect_free_action_not_flagged_uncertain():
    def exploding(url, body, headers):
        raise ConnectionError("down")

    catalog = _catalog(post=exploding)
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    receipt = _dispatch(catalog, bridge, "executor-sandbox", "noop")
    check("noop_failure_not_uncertain",
          receipt.result.get("effect_uncertain") is False)


def test_mutating_dispatch_requires_capability():
    catalog = _catalog()
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    proposal = ActionProposal(
        action_type="db_restore", description="t", risk_tier=RiskTier.OBSERVE,
        rationale="t", alternatives=["n"], principal=_principal(),
        purpose="test", provenance=Provenance(source="test"),
    )
    approval = ApprovalGate().approve(proposal, "dainius", _principal())
    cap = bridge.issue(proposal, approval, "db-restore", "db_restore", _principal())
    # deliberately not consumed
    try:
        catalog.dispatch(cap, "db-restore", "db_restore", "wf", _principal())
        check("unconsumed_capability_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("unconsumed_capability_rejected", "consumed" in str(e))


def test_mutating_audience_enforced():
    catalog = _catalog()
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    proposal = ActionProposal(
        action_type="notify_desktop", description="t",
        risk_tier=RiskTier.OBSERVE, rationale="t", alternatives=["n"],
        principal=_principal(), purpose="test",
        provenance=Provenance(source="test"),
    )
    approval = ApprovalGate().approve(proposal, "dainius", _principal())
    cap = bridge.issue(
        proposal, approval, "notify-service", "notify_desktop", _principal()
    )
    bridge.consume(cap.id, "notify-service", _principal())

    try:
        catalog.dispatch(cap, "db-restore", "db_restore", "wf", _principal())
        check("mutating_audience_enforced", False, "should have raised")
    except ActuatorDispatchError as e:
        check("mutating_audience_enforced", "audience mismatch" in str(e))


def test_path_parameters_not_duplicated_in_body():
    calls = []
    catalog = _catalog(post=lambda u, b, h: calls.append((u, b)) or {})
    _migrate_all(catalog)
    bridge = CapabilityBridge()

    _dispatch(catalog, bridge, "checkpoint-manager", "checkpoint_restore",
              {"checkpoint_id": "ck-1", "reason": "rollback"})
    check("path_param_in_url",
          calls and "ck-1" in calls[0][0], str(calls))
    check("path_param_not_in_body",
          calls and "checkpoint_id" not in calls[0][1], str(calls))
    check("other_params_in_body",
          calls and calls[0][1].get("reason") == "rollback")


def test_missing_path_parameter_raises():
    handler = build_mutating_handler(
        "checkpoint-manager", http_post=lambda u, b, h: {}
    )
    try:
        handler({}, "checkpoint_restore")
        check("missing_path_param_raises", False, "should have raised")
    except MutatingHandlerError as e:
        check("missing_path_param_raises", "requires parameter" in str(e))


def test_unknown_mutating_actuator_rejected():
    try:
        build_mutating_handler("sysmetrics")
        check("read_only_not_mutating", False, "should have raised")
    except MutatingHandlerError as e:
        check("read_only_not_mutating", "not a registered mutating" in str(e))


def test_auth_header_attached_when_token_set():
    calls = []
    saved = os.environ.get("KAI_SERVICE_TOKEN")
    os.environ["KAI_SERVICE_TOKEN"] = "tok-123"
    try:
        handler = build_mutating_handler(
            "notify-service",
            http_post=lambda u, b, h: calls.append(h) or {},
        )
        handler({"title": "x"}, "notify_desktop")
        check("auth_header_attached",
              calls and calls[0].get("Authorization") == "Bearer tok-123",
              str(calls))
    finally:
        if saved is None:
            os.environ.pop("KAI_SERVICE_TOKEN", None)
        else:
            os.environ["KAI_SERVICE_TOKEN"] = saved


def test_all_mutating_endpoints_declared():
    """Every non-tier-1 actuator must appear in MUTATING_ENDPOINTS."""
    catalog = build_catalog(_principal())
    missing = []
    for entry in catalog.list_actuators():
        if entry.migration_tier == MigrationTier.READ_ONLY:
            continue
        if entry.identity not in MUTATING_ENDPOINTS:
            missing.append(entry.identity)
    check("all_mutating_declared", not missing, str(missing))


def test_every_declared_action_has_an_endpoint():
    catalog = build_catalog(_principal())
    missing = []
    for entry in catalog.list_actuators():
        if entry.migration_tier == MigrationTier.READ_ONLY:
            continue
        _, _, actions = MUTATING_ENDPOINTS.get(entry.identity, (None, None, {}))
        for action in entry.action_types:
            if action not in actions:
                missing.append(f"{entry.identity}:{action}")
    check("every_action_has_endpoint", not missing, str(missing))


def test_destructive_actions_declare_irreversible():
    """The most dangerous actions must say so on the receipt."""
    for actuator, action in [
        ("db-restore", "db_restore"),
        ("browser-actor", "browser_click"),
        ("browser-actor", "browser_type"),
        ("executor-shell", "shell_exec"),
        ("executor-shell", "script_exec"),
    ]:
        effects = side_effects_for(actuator, action)
        check(f"{actuator}_{action}_irreversible", "irreversible" in effects,
              str(effects))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_all_legacy_paths_verified_closed()
    test_legacy_closure_explains_itself()
    test_paper_trader_legacy_is_symbol_removal()
    test_unknown_actuator_passes_trivially()
    test_verification_catches_an_unauthenticated_route()
    test_verification_catches_missing_route()
    test_migration_blocks_on_unverified_legacy()
    test_verification_can_be_waived_explicitly()
    test_all_tiers_migrate()
    test_catalogue_size()
    test_every_actuator_has_a_handler()
    test_plan_reports_complete_after_full_migration()
    test_ordering_still_enforced_across_all_tiers()
    test_migration_without_handlers_blocks_every_tier()
    test_mutating_dispatch_posts()
    test_side_effects_declared_on_receipt()
    test_side_effects_lookup()
    test_failed_post_flags_uncertain_effect()
    test_effect_free_action_not_flagged_uncertain()
    test_mutating_dispatch_requires_capability()
    test_mutating_audience_enforced()
    test_path_parameters_not_duplicated_in_body()
    test_missing_path_parameter_raises()
    test_unknown_mutating_actuator_rejected()
    test_auth_header_attached_when_token_set()
    test_all_mutating_endpoints_declared()
    test_every_declared_action_has_an_endpoint()
    test_destructive_actions_declare_irreversible()

    print(f"\n{'='*60}")
    print(f"Full Migration Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
