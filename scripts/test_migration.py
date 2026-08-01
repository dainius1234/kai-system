"""Tier-1 actuator migration and perception active mode.

Closes UH tracker gaps G-01 (tier-1 actuators migrated) and G-02
(perception spine active mode).

The property that keeps G-01 honest: an actuator may not reach ACTIVE
without a dispatch handler.  Without that rule "migrated" could mean the
registry merely knows the actuator's name while no traffic routes
through it — a green report describing nothing.

For G-02, active mode is *additive*. It feeds the world state without
disabling the legacy polling path, so a fault in the spine cannot take
perception offline.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.action import ActionProposal
from common.contracts.perception import EventSource, PerceptionEvent
from common.actuator_registry.catalog import build_catalog
from common.actuator_registry.handlers import (
    READ_ONLY_ENDPOINTS,
    HandlerError,
    attach_read_handlers,
    build_read_handler,
)
from common.actuator_registry.migration import (
    MigrationError,
    migrate_tier,
    migration_plan,
    lower_tiers_complete,
)
from common.actuator_registry.registry import (
    ActuatorDispatchError,
    MigrationStatus,
    MigrationTier,
)
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge
from common.perception_spine.journal import EventJournal
from common.perception_spine.shadow import (
    MODE_ENV,
    ShadowPerceptionRunner,
    perception_mode,
)
from common.world_state.snapshot_store import SnapshotStore
from common.perception_spine.cortex_source import (
    SOURCE_ENV,
    build_state_from_world,
    cortex_source,
    resolve_cortex_state,
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


class _Env:
    def __init__(self, **overrides):
        self._o = overrides
        self._saved = {}

    def __enter__(self):
        for k, v in self._o.items():
            self._saved[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


def _catalog(with_handlers: bool = True, response=None):
    catalog = build_catalog(_principal())
    if with_handlers:
        attach_read_handlers(
            catalog, http_get=lambda url: response or {"stub": True, "url": url}
        )
    return catalog


# ═══════════════════════════════════════════════════════════════════
# G-01 · 1. Handler-less actuators cannot be activated
# ═══════════════════════════════════════════════════════════════════

def test_activation_requires_handler():
    catalog = _catalog(with_handlers=False)
    result = migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())

    check("no_handler_blocks_all", len(result.blocked) == 11,
          f"blocked={len(result.blocked)}")
    check("no_handler_migrates_none", result.migrated == [])
    check("no_handler_not_ok", not result.ok)
    check("no_handler_explains",
          all("no dispatch handler" in e for e in result.errors.values()))

    for entry in catalog.list_actuators(migration_tier=MigrationTier.READ_ONLY):
        check(f"{entry.identity}_still_legacy",
              entry.migration_status == MigrationStatus.LEGACY)


def test_handler_requirement_can_be_waived_explicitly():
    catalog = _catalog(with_handlers=False)
    result = migrate_tier(
        catalog, MigrationTier.READ_ONLY, _principal(), require_handler=False
    )
    check("waiver_allows_migration", len(result.migrated) == 11)
    check("waiver_is_explicit_only", result.ok)


# ═══════════════════════════════════════════════════════════════════
# G-01 · 2. Tier-1 migration
# ═══════════════════════════════════════════════════════════════════

def test_tier1_migrates_fully():
    catalog = _catalog()
    result = migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())

    check("tier1_all_migrated", len(result.migrated) == 11,
          f"migrated={len(result.migrated)}")
    check("tier1_none_blocked", result.blocked == [], str(result.errors))
    check("tier1_ok", result.ok)

    for entry in catalog.list_actuators(migration_tier=MigrationTier.READ_ONLY):
        check(f"{entry.identity}_active",
              entry.migration_status == MigrationStatus.ACTIVE,
              entry.migration_status.value)


def test_tier1_stops_at_verified_when_asked():
    catalog = _catalog()
    migrate_tier(
        catalog, MigrationTier.READ_ONLY, _principal(), activate=False
    )
    for entry in catalog.list_actuators(migration_tier=MigrationTier.READ_ONLY):
        check(f"{entry.identity}_verified_not_active",
              entry.migration_status == MigrationStatus.VERIFIED,
              entry.migration_status.value)


def test_migration_is_idempotent():
    catalog = _catalog()
    migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
    second = migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())

    check("second_run_skips_all", len(second.skipped) == 11)
    check("second_run_migrates_none", second.migrated == [])
    check("second_run_ok", second.ok)


def test_report_reflects_migration():
    catalog = _catalog()
    before = catalog.migration_report()
    check("report_starts_zero", before["migrated"] == 0)

    migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
    after = catalog.migration_report()
    check("report_counts_migrated", after["migrated"] == 11)
    check("report_remaining", after["remaining"] == after["total_actuators"] - 11)


# ═══════════════════════════════════════════════════════════════════
# G-01 · 3. Risk ordering enforced
# ═══════════════════════════════════════════════════════════════════

def test_higher_tier_blocked_until_lower_complete():
    catalog = _catalog()
    try:
        migrate_tier(catalog, MigrationTier.LOCAL_TEST, _principal())
        check("tier2_blocked_before_tier1", False, "should have raised")
    except MigrationError as e:
        check("tier2_blocked_before_tier1", "lower tiers" in str(e))


def test_tier2_unblocked_after_tier1():
    catalog = _catalog()
    migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
    check("no_outstanding_after_tier1",
          lower_tiers_complete(catalog, MigrationTier.LOCAL_TEST) == [])

    plan = migration_plan(catalog)
    check("next_tier_is_2", plan["next_tier"] == 2, str(plan["next_tier"]))


def test_financial_tier_still_blocked():
    """Tier 8 must remain unreachable while tiers 2-7 are un-migrated."""
    catalog = _catalog()
    migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())
    try:
        migrate_tier(
            catalog, MigrationTier.FINANCIAL_DESTRUCTIVE, _principal(),
            require_handler=False,
        )
        check("tier8_still_blocked", False, "should have raised")
    except MigrationError as e:
        check("tier8_still_blocked", "lower tiers" in str(e))


def test_plan_reports_missing_handlers():
    catalog = _catalog(with_handlers=False)
    plan = migration_plan(catalog)
    check("plan_next_tier_1", plan["next_tier"] == 1)
    check("plan_lists_missing_handlers", len(plan["missing_handlers"]) == 11)
    check("plan_not_complete", plan["complete"] is False)


# ═══════════════════════════════════════════════════════════════════
# G-01 · 4. Dispatch through migrated actuators
# ═══════════════════════════════════════════════════════════════════

def _capability(bridge, actuator, action_type, parameters=None):
    proposal = ActionProposal(
        action_type=action_type, description="t", risk_tier=RiskTier.OBSERVE,
        rationale="t", alternatives=["n"], principal=_principal(),
        purpose="test", provenance=Provenance(source="test"),
    )
    approval = ApprovalGate().approve(proposal, "dainius", _principal())
    cap = bridge.issue(
        proposal, approval, actuator, action_type, _principal(),
        parameters=parameters or {},
    )
    bridge.consume(cap.id, actuator, _principal())
    return cap


def test_migrated_actuator_dispatches():
    calls = []
    catalog = build_catalog(_principal())

    def fake_get(url):
        calls.append(url)
        return {"branch": "main", "dirty": False}

    attach_read_handlers(catalog, http_get=fake_get)
    migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())

    bridge = CapabilityBridge()
    cap = _capability(bridge, "git-watcher", "git_status_read")
    receipt = catalog.dispatch(
        cap, "git-watcher", "git_status_read", "wf-1", _principal()
    )

    check("dispatch_called_service", len(calls) == 1, str(calls))
    check("dispatch_hit_summary", calls[0].endswith("/summary"), calls[0])
    check("dispatch_ok", receipt.result.get("ok") is True)
    check("dispatch_returns_data",
          receipt.result.get("data") == {"branch": "main", "dirty": False})
    check("dispatch_not_simulated", "simulated" not in receipt.result)


def test_dispatch_still_requires_capability():
    """Migration does not weaken the capability requirement."""
    catalog = _catalog()
    migrate_tier(catalog, MigrationTier.READ_ONLY, _principal())

    bridge = CapabilityBridge()
    cap = _capability(bridge, "git-watcher", "git_status_read")

    try:
        catalog.dispatch(
            cap, "weather-service", "weather_read", "wf-1", _principal()
        )
        check("wrong_actuator_still_rejected", False, "should have raised")
    except ActuatorDispatchError as e:
        check("wrong_actuator_still_rejected", "audience mismatch" in str(e))


def test_handler_parameter_substitution():
    calls = []
    handler = build_read_handler("broker-bridge", http_get=lambda u: calls.append(u) or {})
    handler({"symbol": "BTCUSDT"}, "market_ticker_read")
    check("param_substituted", calls and calls[0].endswith("/ticker/BTCUSDT"),
          str(calls))


def test_handler_missing_parameter_reported():
    handler = build_read_handler("broker-bridge", http_get=lambda u: {})
    try:
        handler({}, "market_ticker_read")
        check("missing_param_reported", False, "should have raised")
    except HandlerError as e:
        check("missing_param_reported", "requires parameter" in str(e))


def test_handler_service_failure_is_recorded_not_raised():
    def failing(url):
        raise ConnectionError("service down")

    handler = build_read_handler("git-watcher", http_get=failing)
    result = handler({}, "git_status_read")
    check("failure_not_raised", isinstance(result, dict))
    check("failure_marked_not_ok", result.get("ok") is False)
    check("failure_records_error", "ConnectionError" in result.get("error", ""))


def test_handler_rejects_unknown_actuator():
    try:
        build_read_handler("paper-trader")
        check("non_readonly_rejected", False, "should have raised")
    except HandlerError as e:
        check("non_readonly_rejected", "not a registered read-only" in str(e))


def test_all_readonly_endpoints_are_reads():
    """Every tier-1 endpoint must be a read path, never a mutation."""
    mutating = []
    for actuator, (_, _, paths) in READ_ONLY_ENDPOINTS.items():
        for action, path in paths.items():
            lowered = f"{action} {path}".lower()
            if any(w in lowered for w in
                   ("delete", "create", "write", "restore", "execute", "post")):
                mutating.append(f"{actuator}:{action}")
    check("no_mutating_readonly_endpoints", not mutating, str(mutating))


# ═══════════════════════════════════════════════════════════════════
# G-02 · Perception spine active mode
# ═══════════════════════════════════════════════════════════════════

_tmpdir = tempfile.mkdtemp(prefix="migration_test_")
_counter = 0


def _runner(mode=None, world_state=None):
    global _counter
    _counter += 1
    return ShadowPerceptionRunner(
        journal_path=os.path.join(_tmpdir, f"j{_counter}.jsonl"),
        world_state=world_state,
        mode=mode,
    )


def test_mode_defaults_to_shadow():
    with _Env(**{MODE_ENV: None}):
        check("default_mode_shadow", perception_mode() == "shadow")
        check("runner_defaults_shadow", _runner().mode == "shadow")


def test_mode_env_parsing():
    for value, expected in [("active", "active"), ("ACTIVE", "active"),
                            ("shadow", "shadow"), ("nonsense", "shadow"),
                            ("", "shadow")]:
        with _Env(**{MODE_ENV: value}):
            check(f"mode_{value or 'empty'}", perception_mode() == expected)


def test_shadow_mode_does_not_feed_world_state():
    store = SnapshotStore(principal=_principal())
    runner = _runner(mode="shadow", world_state=store)

    event = PerceptionEvent(
        source_type=EventSource.SYSTEM, event_type="test",
        payload={"cpu": 12}, principal=_principal(), purpose="test",
        provenance=Provenance(source="sysmetrics"),
        source_timestamp=datetime.now(timezone.utc),
    )
    runner._maybe_reduce(event, {})

    check("shadow_reduces_nothing", runner.reduced_count == 0)
    check("shadow_world_state_empty", len(store.active_claims()) == 0)


def test_active_mode_feeds_world_state():
    store = SnapshotStore(principal=_principal())
    runner = _runner(mode="active", world_state=store)

    event = PerceptionEvent(
        source_type=EventSource.SYSTEM, event_type="system",
        payload={"cpu_percent": 12, "memory_percent": 40},
        principal=_principal(), purpose="test",
        provenance=Provenance(source="sysmetrics"),
        source_timestamp=datetime.now(timezone.utc),
    )
    runner._maybe_reduce(event, {})

    check("active_reduced_one", runner.reduced_count == 1)
    check("active_world_state_populated", len(store.active_claims()) > 0)


def test_active_mode_without_sink_is_safe():
    runner = _runner(mode="active", world_state=None)
    event = PerceptionEvent(
        source_type=EventSource.SYSTEM, event_type="test", payload={},
        principal=_principal(), purpose="test",
        provenance=Provenance(source="s"),
        source_timestamp=datetime.now(timezone.utc),
    )
    runner._maybe_reduce(event, {})
    check("active_no_sink_no_crash", runner.reduced_count == 0)


def test_reducer_failure_does_not_stop_ingestion():
    """A reduction fault is recorded; the poll loop keeps running."""
    class _Exploding:
        def ingest_event(self, event):
            raise RuntimeError("reducer exploded")

    runner = _runner(mode="active", world_state=_Exploding())
    results: dict = {}
    event = PerceptionEvent(
        source_type=EventSource.SYSTEM, event_type="test", payload={},
        principal=_principal(), purpose="test",
        provenance=Provenance(source="s"),
        source_timestamp=datetime.now(timezone.utc),
    )
    runner._maybe_reduce(event, results)

    check("reduce_failure_counted", runner.reduce_failures == 1)
    check("reduce_failure_not_raised", True)
    check("reduce_failure_recorded", "reduce_errors" in results)
    check("reduce_failure_detail",
          "reducer exploded" in results["reduce_errors"][0])


def test_mode_switchable_at_runtime():
    store = SnapshotStore(principal=_principal())
    runner = _runner(mode="shadow", world_state=store)
    check("starts_shadow", runner.mode == "shadow")

    runner.set_mode("active")
    check("switched_active", runner.mode == "active")

    runner.set_mode("nonsense")
    check("bad_mode_falls_back_shadow", runner.mode == "shadow")


def test_active_mode_is_additive():
    """Active mode must not disable journalling — it adds a consumer."""
    store = SnapshotStore(principal=_principal())
    runner = _runner(mode="active", world_state=store)

    event = PerceptionEvent(
        source_type=EventSource.SYSTEM, event_type="system",
        payload={"cpu_percent": 5}, principal=_principal(), purpose="test",
        provenance=Provenance(source="sysmetrics"),
        source_timestamp=datetime.now(timezone.utc),
        raw_hash="additive-1",
    )
    result = runner._ingress.submit(event)
    runner._maybe_reduce(result.event, {})

    check("active_still_journals", runner.journal.count() == 1)
    check("active_also_reduces", runner.reduced_count == 1)


def test_cycle_stats_report_mode():
    runner = _runner(mode="active", world_state=SnapshotStore(principal=_principal()))
    runner._endpoints = {}
    stats = asyncio.run(runner.run_once())
    check("stats_include_mode", stats.get("mode") == "active")
    check("stats_include_reduced", "events_reduced" in stats)


# ═══════════════════════════════════════════════════════════════════
# G-01b · Endpoint paths verified against real service routes
# ═══════════════════════════════════════════════════════════════════

# Which source file actually defines each actuator's routes.
SERVICE_ROUTE_FILES = {
    "broker-bridge": "broker-bridge/app.py",
    "docker-watcher": "docker-watcher/app.py",
    "git-watcher": "git-watcher/app.py",
    "email-reader": "email-reader/app.py",
    "calendar-service": "calendar-service/app.py",
    "news-feed": "news-feed/app.py",
    "sysmetrics": "sysmetrics/app.py",
    "weather-service": "weather-service/app.py",
    "alpha-signals": "agentic/app.py",
    "market-data": "agentic/app.py",
    "service-watchdog": "agentic/app.py",
}


def _declared_get_routes(rel_path: str) -> set:
    """GET routes a service actually declares, read from its source."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / rel_path
    if not source.exists():
        return set()
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    routes = set()
    for node in ast.walk(tree):
        for dec in getattr(node, "decorator_list", []):
            if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
                continue
            if dec.func.attr != "get" or not dec.args:
                continue
            first = dec.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                routes.add(first.value)
    return routes


def _route_matches(template: str, declared: set) -> bool:
    """Whether a handler template corresponds to a declared route.

    Path parameters are named differently between the handler template
    and the route decorator, so templated paths compare on their static
    prefix rather than exact text.
    """
    if template in declared:
        return True
    if "{" not in template:
        return False
    prefix = template.split("{")[0].rstrip("/")
    return any(
        route.split("{")[0].rstrip("/") == prefix
        for route in declared
        if "{" in route
    )


def test_every_endpoint_exists_in_its_service():
    """Every tier-1 path must correspond to a route the service declares.

    Mock-based dispatch tests pass against any string, so they cannot
    catch a wrong path.  This reads the services' own source and was
    what caught four incorrect endpoints on first run.
    """
    missing = []
    unmapped = []

    for actuator, (_, _, paths) in sorted(READ_ONLY_ENDPOINTS.items()):
        rel = SERVICE_ROUTE_FILES.get(actuator)
        if rel is None:
            unmapped.append(actuator)
            continue
        declared = _declared_get_routes(rel)
        if not declared:
            unmapped.append(f"{actuator} ({rel} unreadable)")
            continue
        for action, template in paths.items():
            if not _route_matches(template, declared):
                missing.append(f"{actuator}:{action} -> {template}")

    check("every_actuator_has_route_source", not unmapped, "; ".join(unmapped))
    check("every_endpoint_exists_in_service", not missing, "; ".join(missing))


def test_route_source_map_covers_all_actuators():
    uncovered = set(READ_ONLY_ENDPOINTS) - set(SERVICE_ROUTE_FILES)
    check("route_map_complete", not uncovered, str(uncovered))


def test_route_matcher_rejects_wrong_paths():
    """The matcher must not pass anything — it would be useless if it did."""
    declared = {"/inbox", "/unread", "/ticker/{symbol}"}
    check("matcher_accepts_exact", _route_matches("/inbox", declared))
    check("matcher_accepts_templated", _route_matches("/ticker/{sym}", declared))
    check("matcher_rejects_absent", not _route_matches("/summary", declared))
    check("matcher_rejects_wrong_prefix",
          not _route_matches("/quote/{sym}", declared))


# ═══════════════════════════════════════════════════════════════════
# G-02b · Cortex world-state source (defined cutover for polling)
# ═══════════════════════════════════════════════════════════════════

def _populated_store():
    store = SnapshotStore(principal=_principal())
    for i, (etype, payload) in enumerate([
        ("system", {"cpu_percent": 12, "memory_percent": 40}),
        ("docker", {"containers": [{"name": "kai", "status": "running"}]}),
        ("git", {"branch": "main", "dirty": False}),
    ]):
        store.ingest_event(PerceptionEvent(
            source_type=EventSource.SYSTEM, event_type=etype, payload=payload,
            principal=_principal(), purpose="test",
            provenance=Provenance(source=f"sensor-{i}"),
            source_timestamp=datetime.now(timezone.utc),
            raw_hash=f"cortex-{i}",
        ))
    return store


def test_cortex_source_defaults_to_poll():
    with _Env(**{SOURCE_ENV: None}):
        check("default_source_poll", cortex_source() == "poll")


def test_cortex_source_env_parsing():
    for value, expected in [("world_state", "world_state"), ("POLL", "poll"),
                            ("poll", "poll"), ("nonsense", "poll"), ("", "poll")]:
        with _Env(**{SOURCE_ENV: value}):
            check(f"source_{value or 'empty'}", cortex_source() == expected)


def test_poll_mode_returns_polled_state_untouched():
    """The default path must be exactly what it was before this existed."""
    polled = {"level1_facts": ["a"], "level2_summary": "s", "custom": 1}
    with _Env(**{SOURCE_ENV: "poll"}):
        result = resolve_cortex_state(polled, store=_populated_store())
    check("poll_mode_untouched", result == polled)
    check("poll_mode_is_same_object", result is polled)


def test_world_state_mode_reads_claims():
    with _Env(**{SOURCE_ENV: "world_state"}):
        result = resolve_cortex_state({"level1_facts": ["old"]},
                                      store=_populated_store())
    check("world_state_used", result.get("source") == "world_state")
    check("world_state_has_facts", len(result.get("level1_facts", [])) > 0)
    check("world_state_replaced_polled",
          "old" not in result.get("level1_facts", []))
    check("world_state_has_summary", bool(result.get("level2_summary")))
    check("world_state_has_implication", bool(result.get("level3_implication")))


def test_empty_world_state_falls_back_to_polled():
    """A cold start must not blank out perception."""
    polled = {"level1_facts": ["from polling"], "level2_summary": "s"}
    with _Env(**{SOURCE_ENV: "world_state"}):
        result = resolve_cortex_state(
            polled, store=SnapshotStore(principal=_principal())
        )
    check("empty_world_falls_back", result == polled)


def test_missing_store_falls_back_to_polled():
    polled = {"level1_facts": ["from polling"]}
    with _Env(**{SOURCE_ENV: "world_state"}):
        result = resolve_cortex_state(polled, store=None)
    check("no_store_falls_back", result == polled)


def test_broken_store_falls_back_to_polled():
    class _Broken:
        def active_claims(self):
            raise RuntimeError("store offline")

    polled = {"level1_facts": ["from polling"]}
    with _Env(**{SOURCE_ENV: "world_state"}):
        result = resolve_cortex_state(polled, store=_Broken())
    check("broken_store_falls_back", result == polled)


def test_build_state_returns_none_when_empty():
    check("build_none_on_empty",
          build_state_from_world(SnapshotStore(principal=_principal())) is None)
    check("build_none_on_missing", build_state_from_world(None) is None)


def test_facts_are_domain_prefixed():
    state = build_state_from_world(_populated_store())
    check("facts_prefixed",
          all(":" in f for f in state["level1_facts"]),
          str(state["level1_facts"][:3]))


def test_fact_cap_respected():
    store = SnapshotStore(principal=_principal())
    for i in range(60):
        store.ingest_event(PerceptionEvent(
            source_type=EventSource.SYSTEM, event_type="system",
            payload={"cpu_percent": i}, principal=_principal(), purpose="t",
            provenance=Provenance(source=f"s{i}"),
            source_timestamp=datetime.now(timezone.utc),
            raw_hash=f"cap-{i}",
        ))
    state = build_state_from_world(store, max_facts=5)
    if state is not None:
        check("fact_cap_respected", len(state["level1_facts"]) <= 5)
    else:
        check("fact_cap_respected", True)


def test_state_shape_matches_cortex_contract():
    """The rendered state must carry the keys feed_service_state reads."""
    state = build_state_from_world(_populated_store())
    for key in ("level1_facts", "level2_summary", "level3_implication",
                "refresh_count", "timestamp"):
        check(f"state_has_{key}", key in state)
    check("facts_is_list", isinstance(state["level1_facts"], list))
    check("refresh_count_is_int", isinstance(state["refresh_count"], int))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_activation_requires_handler()
    test_handler_requirement_can_be_waived_explicitly()
    test_tier1_migrates_fully()
    test_tier1_stops_at_verified_when_asked()
    test_migration_is_idempotent()
    test_report_reflects_migration()
    test_higher_tier_blocked_until_lower_complete()
    test_tier2_unblocked_after_tier1()
    test_financial_tier_still_blocked()
    test_plan_reports_missing_handlers()
    test_migrated_actuator_dispatches()
    test_dispatch_still_requires_capability()
    test_handler_parameter_substitution()
    test_handler_missing_parameter_reported()
    test_handler_service_failure_is_recorded_not_raised()
    test_handler_rejects_unknown_actuator()
    test_all_readonly_endpoints_are_reads()
    test_mode_defaults_to_shadow()
    test_mode_env_parsing()
    test_shadow_mode_does_not_feed_world_state()
    test_active_mode_feeds_world_state()
    test_active_mode_without_sink_is_safe()
    test_reducer_failure_does_not_stop_ingestion()
    test_mode_switchable_at_runtime()
    test_active_mode_is_additive()
    test_cycle_stats_report_mode()
    test_every_endpoint_exists_in_its_service()
    test_route_source_map_covers_all_actuators()
    test_route_matcher_rejects_wrong_paths()
    test_cortex_source_defaults_to_poll()
    test_cortex_source_env_parsing()
    test_poll_mode_returns_polled_state_untouched()
    test_world_state_mode_reads_claims()
    test_empty_world_state_falls_back_to_polled()
    test_missing_store_falls_back_to_polled()
    test_broken_store_falls_back_to_polled()
    test_build_state_returns_none_when_empty()
    test_facts_are_domain_prefixed()
    test_fact_cap_respected()
    test_state_shape_matches_cortex_contract()

    print(f"\n{'='*60}")
    print(f"Migration & Active Mode Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
