"""UH-7 actuator registry exit-gate tests.

Exit gate (from roadmap):
  - each migration disables the old path before the new path is
    marked verified

Additional tests:
  - no dispatch without registration
  - no dispatch while LEGACY or MIGRATING
  - capability audience, type and consumption enforced
  - action_type ownership is exclusive
  - migration transitions are a strict state machine
  - risk ordering: lower tiers migrate first
  - catalog covers every audited actuator surface
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.action import ActionProposal
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge
from common.actuator_registry.registry import (
    ActuatorDispatchError,
    ActuatorRegistration,
    ActuatorRegistry,
    MigrationStatus,
    MigrationTier,
)
from common.actuator_registry.catalog import ALL_ACTUATORS, build_catalog

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


def _registration(
    identity: str = "test-actuator",
    action_types: list | None = None,
    tier: MigrationTier = MigrationTier.READ_ONLY,
    legacy_path: str | None = None,
) -> ActuatorRegistration:
    return ActuatorRegistration(
        identity=identity,
        display_name=identity.title(),
        description="test actuator",
        risk_tier=RiskTier.OBSERVE,
        migration_tier=tier,
        action_types=action_types or ["test_action"],
        reversible=True,
        legacy_path=legacy_path,
    )


def _capability(
    bridge: CapabilityBridge,
    actuator: str,
    action_type: str,
    consume: bool = True,
):
    proposal = ActionProposal(
        action_type=action_type,
        description="test",
        risk_tier=RiskTier.OBSERVE,
        rationale="testing",
        alternatives=["nothing"],
        principal=_principal(),
        purpose="test",
        provenance=Provenance(source="test"),
    )
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())
    cap = bridge.issue(
        proposal, approval, actuator, action_type, _principal()
    )
    if consume:
        bridge.consume(cap.id, actuator, _principal())
    return cap


def _ready_registry(
    identity: str = "test-actuator",
    action_types: list | None = None,
) -> ActuatorRegistry:
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration(identity, action_types))
    reg.advance_migration(identity, MigrationStatus.MIGRATING, _principal())
    reg.advance_migration(identity, MigrationStatus.VERIFIED, _principal())
    return reg


# ── 1. Registration basics ──────────────────────────────────────────

def test_registration():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())

    entry = reg.get("test-actuator")
    check("registration_stored", entry is not None)
    check("registration_starts_legacy",
          entry.migration_status == MigrationStatus.LEGACY)
    check("registration_by_action_type",
          reg.get_by_action_type("test_action").identity == "test-actuator")
    check("unknown_action_type_none",
          reg.get_by_action_type("nope") is None)


def test_duplicate_registration_rejected():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    try:
        reg.register(_registration())
        check("duplicate_identity_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("duplicate_identity_rejected", "already registered" in str(e))


def test_action_type_collision_rejected():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration("actuator-a", ["shared_action"]))
    try:
        reg.register(_registration("actuator-b", ["shared_action"]))
        check("action_type_collision_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("action_type_collision_rejected", "already claimed" in str(e))


def test_empty_registration_rejected():
    try:
        ActuatorRegistration(
            identity="", display_name="x", description="x",
            risk_tier=RiskTier.OBSERVE,
            migration_tier=MigrationTier.READ_ONLY,
            action_types=["a"],
        )
        check("empty_identity_rejected", False, "should have raised")
    except ValueError:
        check("empty_identity_rejected", True)

    try:
        ActuatorRegistration(
            identity="x", display_name="x", description="x",
            risk_tier=RiskTier.OBSERVE,
            migration_tier=MigrationTier.READ_ONLY,
            action_types=[],
        )
        check("empty_action_types_rejected", False, "should have raised")
    except ValueError:
        check("empty_action_types_rejected", True)


# ── 2. Migration state machine ──────────────────────────────────────

def test_migration_state_machine():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    p = _principal()

    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)
    check("legacy_to_migrating",
          reg.get("test-actuator").migration_status == MigrationStatus.MIGRATING)

    reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
    check("migrating_to_verified",
          reg.get("test-actuator").migration_status == MigrationStatus.VERIFIED)

    reg.advance_migration("test-actuator", MigrationStatus.ACTIVE, p)
    check("verified_to_active",
          reg.get("test-actuator").migration_status == MigrationStatus.ACTIVE)


def test_migration_skip_rejected():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    p = _principal()

    try:
        reg.advance_migration("test-actuator", MigrationStatus.ACTIVE, p)
        check("legacy_to_active_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("legacy_to_active_rejected", "invalid migration" in str(e))

    try:
        reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
        check("legacy_to_verified_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("legacy_to_verified_rejected", "invalid migration" in str(e))


def test_disabled_is_terminal():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    p = _principal()

    reg.advance_migration("test-actuator", MigrationStatus.DISABLED, p)
    try:
        reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)
        check("disabled_is_terminal", False, "should have raised")
    except ActuatorDispatchError:
        check("disabled_is_terminal", True)


def test_unknown_actuator_migration_rejected():
    reg = ActuatorRegistry(principal=_principal())
    try:
        reg.advance_migration("ghost", MigrationStatus.MIGRATING, _principal())
        check("unknown_actuator_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("unknown_actuator_rejected", "not registered" in str(e))


# ── 3. EXIT GATE: legacy path disabled before verified ──────────────

def test_legacy_path_blocks_verification():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration(legacy_path="old:/direct/path"))
    p = _principal()

    entry = reg.get("test-actuator")
    check("legacy_path_recorded", entry.legacy_path == "old:/direct/path")
    check("legacy_starts_enabled", entry.legacy_disabled is False)

    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)

    try:
        reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
        check("verify_blocked_by_legacy", False, "should have raised")
    except ActuatorDispatchError as e:
        check("verify_blocked_by_legacy", "legacy path" in str(e).lower())

    check("still_migrating",
          reg.get("test-actuator").migration_status == MigrationStatus.MIGRATING)

    reg.disable_legacy_path("test-actuator", p)
    check("legacy_now_disabled", reg.get("test-actuator").legacy_disabled is True)
    check("legacy_disabled_timestamped",
          reg.get("test-actuator").legacy_disabled_at is not None)

    reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
    check("verify_allowed_after_disable",
          reg.get("test-actuator").migration_status == MigrationStatus.VERIFIED)


def test_no_legacy_path_verifies_directly():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration(legacy_path=None))
    p = _principal()

    check("no_legacy_counts_disabled",
          reg.get("test-actuator").legacy_disabled is True)

    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)
    reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
    check("no_legacy_verifies",
          reg.get("test-actuator").migration_status == MigrationStatus.VERIFIED)


def test_catalog_legacy_paths_all_open_initially():
    catalog = build_catalog(_principal())
    report = catalog.migration_report()

    check("catalog_has_open_legacy_paths",
          len(report["legacy_paths_open"]) > 0)
    check("catalog_paper_trader_legacy_open",
          "paper-trader" in report["legacy_paths_open"])
    check("catalog_db_restore_legacy_open",
          "db-restore" in report["legacy_paths_open"])
    check("catalog_browser_actor_legacy_open",
          "browser-actor" in report["legacy_paths_open"])


# ── 4. Dispatch gating ──────────────────────────────────────────────

def test_dispatch_requires_registration():
    reg = ActuatorRegistry(principal=_principal())
    bridge = CapabilityBridge()
    cap = _capability(bridge, "ghost-actuator", "test_action")

    try:
        reg.dispatch(cap, "ghost-actuator", "test_action", "wf-1", _principal())
        check("dispatch_requires_registration", False, "should have raised")
    except ActuatorDispatchError as e:
        check("dispatch_requires_registration", "not registered" in str(e))


def test_dispatch_blocked_while_legacy():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "test_action")

    try:
        reg.dispatch(cap, "test-actuator", "test_action", "wf-1", _principal())
        check("dispatch_blocked_legacy", False, "should have raised")
    except ActuatorDispatchError as e:
        check("dispatch_blocked_legacy", "not ready" in str(e))


def test_dispatch_blocked_while_migrating():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, _principal())
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "test_action")

    try:
        reg.dispatch(cap, "test-actuator", "test_action", "wf-1", _principal())
        check("dispatch_blocked_migrating", False, "should have raised")
    except ActuatorDispatchError as e:
        check("dispatch_blocked_migrating", "not ready" in str(e))


def test_dispatch_blocked_when_disabled():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    p = _principal()
    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)
    reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
    reg.advance_migration("test-actuator", MigrationStatus.DISABLED, p)

    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "test_action")

    try:
        reg.dispatch(cap, "test-actuator", "test_action", "wf-1", p)
        check("dispatch_blocked_disabled", False, "should have raised")
    except ActuatorDispatchError as e:
        check("dispatch_blocked_disabled", "not ready" in str(e))


def test_dispatch_succeeds_when_verified():
    reg = _ready_registry()
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "test_action")

    receipt = reg.dispatch(cap, "test-actuator", "test_action", "wf-1", _principal())
    check("dispatch_returns_receipt", receipt is not None)
    check("receipt_actuator", receipt.actuator == "test-actuator")
    check("receipt_action", receipt.action_taken == "test_action")
    check("receipt_capability_id", receipt.capability_id == cap.id)
    check("receipt_workflow_id", receipt.workflow_id == "wf-1")
    check("receipt_logged", len(reg.dispatch_log) == 1)


# ── 5. Capability enforcement at dispatch ───────────────────────────

def test_unconsumed_capability_rejected():
    reg = _ready_registry()
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "test_action", consume=False)

    try:
        reg.dispatch(cap, "test-actuator", "test_action", "wf-1", _principal())
        check("unconsumed_capability_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("unconsumed_capability_rejected", "consumed" in str(e))


def test_capability_audience_enforced_at_dispatch():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration("actuator-a", ["action_a"]))
    reg.register(_registration("actuator-b", ["action_b"]))
    p = _principal()
    for ident in ("actuator-a", "actuator-b"):
        reg.advance_migration(ident, MigrationStatus.MIGRATING, p)
        reg.advance_migration(ident, MigrationStatus.VERIFIED, p)

    bridge = CapabilityBridge()
    cap = _capability(bridge, "actuator-a", "action_a")

    try:
        reg.dispatch(cap, "actuator-b", "action_b", "wf-1", p)
        check("audience_enforced_at_dispatch", False, "should have raised")
    except ActuatorDispatchError as e:
        check("audience_enforced_at_dispatch", "audience mismatch" in str(e))


def test_capability_type_mismatch_rejected():
    reg = _ready_registry("test-actuator", ["action_one", "action_two"])
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "action_one")

    try:
        reg.dispatch(cap, "test-actuator", "action_two", "wf-1", _principal())
        check("capability_type_mismatch_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("capability_type_mismatch_rejected", "type mismatch" in str(e))


def test_unsupported_action_type_rejected():
    reg = _ready_registry()
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "unsupported_action")

    try:
        reg.dispatch(cap, "test-actuator", "unsupported_action", "wf-1", _principal())
        check("unsupported_action_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("unsupported_action_rejected", "does not support" in str(e))


# ── 6. Handlers ─────────────────────────────────────────────────────

def test_handler_invoked():
    reg = ActuatorRegistry(principal=_principal())
    calls = []

    def handler(params, action_type):
        calls.append((params, action_type))
        return {"handled": True, "echo": params.get("k")}

    reg.register(_registration(), handler=handler)
    p = _principal()
    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)
    reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)

    bridge = CapabilityBridge()
    proposal = ActionProposal(
        action_type="test_action", description="t", risk_tier=RiskTier.OBSERVE,
        rationale="t", alternatives=["n"], principal=p, purpose="test",
        provenance=Provenance(source="test"),
    )
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", p)
    cap = bridge.issue(
        proposal, approval, "test-actuator", "test_action", p,
        parameters={"k": "v"},
    )
    bridge.consume(cap.id, "test-actuator", p)

    receipt = reg.dispatch(cap, "test-actuator", "test_action", "wf-1", p)
    check("handler_called", len(calls) == 1)
    check("handler_got_params", calls[0][0] == {"k": "v"})
    check("handler_result_in_receipt", receipt.result.get("handled") is True)
    check("handler_echo", receipt.result.get("echo") == "v")


def test_default_handler_simulates():
    reg = _ready_registry()
    bridge = CapabilityBridge()
    cap = _capability(bridge, "test-actuator", "test_action")

    receipt = reg.dispatch(cap, "test-actuator", "test_action", "wf-1", _principal())
    check("default_handler_simulated", receipt.result.get("simulated") is True)


def test_set_handler_unknown_actuator_rejected():
    reg = ActuatorRegistry(principal=_principal())
    try:
        reg.set_handler("ghost", lambda p, a: {})
        check("set_handler_unknown_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("set_handler_unknown_rejected", "not registered" in str(e))


# ── 7. Risk-ordered migration ───────────────────────────────────────

def test_next_candidates_are_lowest_tier():
    catalog = build_catalog(_principal())
    candidates = catalog.next_migration_candidates()

    check("candidates_exist", len(candidates) > 0)
    check("candidates_all_tier_1",
          all(c.migration_tier == MigrationTier.READ_ONLY for c in candidates))
    check("candidates_include_broker_bridge",
          any(c.identity == "broker-bridge" for c in candidates))


def test_candidates_advance_with_tier():
    catalog = build_catalog(_principal())
    p = _principal()

    for entry in catalog.list_actuators(migration_tier=MigrationTier.READ_ONLY):
        catalog.advance_migration(entry.identity, MigrationStatus.MIGRATING, p)
        catalog.disable_legacy_path(entry.identity, p)
        catalog.advance_migration(entry.identity, MigrationStatus.VERIFIED, p)
        catalog.advance_migration(entry.identity, MigrationStatus.ACTIVE, p)

    candidates = catalog.next_migration_candidates()
    check("tier_2_next_after_tier_1",
          all(c.migration_tier == MigrationTier.LOCAL_TEST for c in candidates))


def test_no_candidates_when_fully_migrated():
    reg = ActuatorRegistry(principal=_principal())
    reg.register(_registration())
    p = _principal()
    reg.advance_migration("test-actuator", MigrationStatus.MIGRATING, p)
    reg.advance_migration("test-actuator", MigrationStatus.VERIFIED, p)
    reg.advance_migration("test-actuator", MigrationStatus.ACTIVE, p)

    check("no_candidates_when_done", reg.next_migration_candidates() == [])


# ── 8. Catalog coverage ─────────────────────────────────────────────

def test_catalog_coverage():
    catalog = build_catalog(_principal())
    report = catalog.migration_report()

    check("catalog_populated", report["total_actuators"] == len(ALL_ACTUATORS))
    check("catalog_nothing_migrated_initially", report["migrated"] == 0)
    check("catalog_all_remaining",
          report["remaining"] == report["total_actuators"])
    check("catalog_all_eight_tiers", len(report["by_tier"]) == 8)


def test_catalog_high_risk_is_tier_8():
    catalog = build_catalog(_principal())
    tier_8 = {
        r.identity
        for r in catalog.list_actuators(migration_tier=MigrationTier.FINANCIAL_DESTRUCTIVE)
    }

    check("paper_trader_tier_8", "paper-trader" in tier_8)
    check("db_restore_tier_8", "db-restore" in tier_8)
    check("browser_actor_tier_8", "browser-actor" in tier_8)
    check("executor_shell_tier_8", "executor-shell" in tier_8)


def test_catalog_read_only_is_tier_1():
    catalog = build_catalog(_principal())
    tier_1 = {
        r.identity
        for r in catalog.list_actuators(migration_tier=MigrationTier.READ_ONLY)
    }

    check("broker_bridge_tier_1", "broker-bridge" in tier_1)
    check("git_watcher_tier_1", "git-watcher" in tier_1)
    check("email_reader_tier_1", "email-reader" in tier_1)
    check("no_paper_trader_in_tier_1", "paper-trader" not in tier_1)


def test_catalog_action_types_unique():
    catalog = build_catalog(_principal())
    seen = {}
    duplicates = []
    for entry in catalog.list_actuators():
        for action_type in entry.action_types:
            if action_type in seen:
                duplicates.append(action_type)
            seen[action_type] = entry.identity

    check("catalog_action_types_unique", not duplicates,
          f"duplicates: {duplicates}")


def test_catalog_destructive_marked_irreversible():
    catalog = build_catalog(_principal())

    check("db_restore_irreversible",
          catalog.get("db-restore").reversible is False)
    check("browser_actor_irreversible",
          catalog.get("browser-actor").reversible is False)
    check("executor_shell_irreversible",
          catalog.get("executor-shell").reversible is False)
    check("paper_trader_reversible",
          catalog.get("paper-trader").reversible is True)


# ── 9. Listing and reporting ────────────────────────────────────────

def test_list_filters():
    catalog = build_catalog(_principal())
    p = _principal()
    catalog.advance_migration("git-watcher", MigrationStatus.MIGRATING, p)

    migrating = catalog.list_actuators(migration_status=MigrationStatus.MIGRATING)
    check("list_filter_by_status", len(migrating) == 1)
    check("list_filter_status_match", migrating[0].identity == "git-watcher")

    tier_1 = catalog.list_actuators(migration_tier=MigrationTier.READ_ONLY)
    check("list_filter_by_tier", len(tier_1) == 11)

    both = catalog.list_actuators(
        migration_status=MigrationStatus.MIGRATING,
        migration_tier=MigrationTier.READ_ONLY,
    )
    check("list_filter_combined", len(both) == 1)


def test_report_tracks_progress():
    catalog = build_catalog(_principal())
    p = _principal()

    before = catalog.migration_report()["migrated"]
    catalog.advance_migration("git-watcher", MigrationStatus.MIGRATING, p)
    catalog.advance_migration("git-watcher", MigrationStatus.VERIFIED, p)
    after = catalog.migration_report()["migrated"]

    check("report_tracks_progress", after == before + 1)


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_registration()
    test_duplicate_registration_rejected()
    test_action_type_collision_rejected()
    test_empty_registration_rejected()
    test_migration_state_machine()
    test_migration_skip_rejected()
    test_disabled_is_terminal()
    test_unknown_actuator_migration_rejected()
    test_legacy_path_blocks_verification()
    test_no_legacy_path_verifies_directly()
    test_catalog_legacy_paths_all_open_initially()
    test_dispatch_requires_registration()
    test_dispatch_blocked_while_legacy()
    test_dispatch_blocked_while_migrating()
    test_dispatch_blocked_when_disabled()
    test_dispatch_succeeds_when_verified()
    test_unconsumed_capability_rejected()
    test_capability_audience_enforced_at_dispatch()
    test_capability_type_mismatch_rejected()
    test_unsupported_action_type_rejected()
    test_handler_invoked()
    test_default_handler_simulates()
    test_set_handler_unknown_actuator_rejected()
    test_next_candidates_are_lowest_tier()
    test_candidates_advance_with_tier()
    test_no_candidates_when_fully_migrated()
    test_catalog_coverage()
    test_catalog_high_risk_is_tier_8()
    test_catalog_read_only_is_tier_1()
    test_catalog_action_types_unique()
    test_catalog_destructive_marked_irreversible()
    test_list_filters()
    test_report_tracks_progress()

    print(f"\n{'='*60}")
    print(f"UH-7 Actuator Registry Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
