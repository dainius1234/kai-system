"""UH-3 world state exit-gate tests.

Exit gates (from roadmap):
  - snapshots are reproducible
  - conflicting sources remain visible
  - deleted/superseded records do not remain active in derivatives

Additional tests:
  - deterministic reducers produce claims from events
  - event-to-fact lineage (evidence_ids link claims to source events)
  - conflict/unknown/stale semantics
  - principal/purpose/data-class scoped views
  - bounded retention and deletion lineage
  - snapshot digest verification
  - reducer registry and generic fallback
  - domain cap enforcement
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import (
    ContractState,
    Principal,
    Provenance,
    RiskTier,
    VerificationVerdict,
)
from common.contracts.perception import EventSource, PerceptionEvent
from common.contracts.world_state import (
    Claim,
    EvidenceRecord,
    FreshnessStatus,
    WorldStateSnapshot,
)
from common.world_state.reducers import (
    REDUCER_REVISION,
    ReducerOutput,
    ReducerRegistry,
    reduce_weather,
    reduce_system,
    reduce_docker,
    reduce_git,
    reduce_generic,
)
from common.world_state.snapshot_store import SnapshotStore

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


def _principal(identity: str = "kai") -> Principal:
    return Principal(identity=identity, role="system")


def _event(
    event_type: str,
    source: EventSource,
    payload: dict,
    principal: Principal | None = None,
    stale: bool = False,
    source_timestamp: datetime | None = None,
) -> PerceptionEvent:
    return PerceptionEvent(
        event_type=event_type,
        source_type=source,
        principal=principal or _principal(),
        purpose="test",
        provenance=Provenance(source="test"),
        payload=payload,
        stale=stale,
        source_timestamp=source_timestamp,
    )


# ── 1. Reducer produces claims from weather event ─────────────────────

def test_reducer_weather():
    p = _principal()
    event = _event("weather_reading", EventSource.WEATHER, {"summary": "Sunny 22°C", "temp_c": 22})
    output = reduce_weather(event, p)

    check("weather_produces_claims", len(output.claims) > 0)
    check("weather_produces_evidence", len(output.evidence) > 0)

    claim = output.claims[0]
    check("weather_claim_has_text", "Sunny" in claim.claim_text)
    check("weather_claim_domain", claim.domain == "environment")
    check("weather_claim_has_evidence_ids", len(claim.evidence_ids) > 0)
    check("weather_claim_principal", claim.principal.identity == "kai")

    evidence = output.evidence[0]
    check("weather_evidence_source_event", evidence.source_event_id == event.id)
    check("weather_evidence_freshness", evidence.freshness is not None)


# ── 2. Reducer produces claims from system metrics ───────────────────

def test_reducer_system():
    p = _principal()
    event = _event("system_metrics", EventSource.SYSTEM, {
        "cpu_percent": 75, "memory_percent": 60, "disk_percent": 45
    })
    output = reduce_system(event, p)

    check("system_produces_multiple_claims", len(output.claims) == 3)
    domains = {c.domain for c in output.claims}
    check("system_claims_infrastructure", domains == {"infrastructure"})
    texts = [c.claim_text for c in output.claims]
    check("system_has_cpu_claim", any("CPU" in t for t in texts))
    check("system_has_memory_claim", any("Memory" in t for t in texts))
    check("system_has_disk_claim", any("Disk" in t for t in texts))


# ── 3. Generic reducer fallback ──────────────────────────────────────

def test_reducer_generic():
    p = _principal()
    event = _event("unknown_type", EventSource.MANUAL, {"summary": "hello"})
    output = reduce_generic(event, p)

    check("generic_produces_claim", len(output.claims) == 1)
    check("generic_claim_has_source", event.source_type.value in output.claims[0].claim_text)


# ── 4. Reducer registry dispatches correctly ─────────────────────────

def test_registry_dispatch():
    registry = ReducerRegistry()

    p = _principal()
    weather_ev = _event("weather_reading", EventSource.WEATHER, {"summary": "Rain"})
    out = registry.reduce(weather_ev, p)
    check("registry_dispatches_weather", len(out.claims) > 0)
    check("registry_weather_domain", out.claims[0].domain == "environment")

    system_ev = _event("system_metrics", EventSource.SYSTEM, {"cpu_percent": 50})
    out2 = registry.reduce(system_ev, p)
    check("registry_dispatches_system", len(out2.claims) > 0)

    unknown_ev = _event("unregistered_type", EventSource.MANUAL, {"summary": "test"})
    out3 = registry.reduce(unknown_ev, p)
    check("registry_falls_back_to_generic", len(out3.claims) > 0)


# ── 5. Stale event produces stale claim ──────────────────────────────

def test_stale_claim():
    p = _principal()
    event = _event("weather_reading", EventSource.WEATHER, {"summary": "Old data"}, stale=True)
    output = reduce_weather(event, p)

    check("stale_event_produces_claim", len(output.claims) > 0)
    check("stale_evidence_marked", output.evidence[0].freshness == FreshnessStatus.STALE)
    check("stale_claim_marked", output.claims[0].freshness == FreshnessStatus.STALE)


# ── 6. Event-to-fact lineage ─────────────────────────────────────────

def test_event_fact_lineage():
    p = _principal()
    event = _event("docker_status", EventSource.DOCKER, {"summary": "5 running"})
    output = reduce_docker(event, p)

    evidence = output.evidence[0]
    claim = output.claims[0]

    check("lineage_evidence_to_event", evidence.source_event_id == event.id)
    check("lineage_claim_to_evidence", evidence.id in claim.evidence_ids)
    check("lineage_provenance_upstream", event.id in evidence.provenance.upstream_ids)
    check("lineage_claim_provenance", evidence.id in claim.provenance.upstream_ids)


# ── 7. Snapshot store produces snapshot ──────────────────────────────

def test_snapshot_basic():
    store = SnapshotStore(principal=_principal())

    event = _event("weather_reading", EventSource.WEATHER, {"summary": "Clear"})
    store.ingest_event(event)
    snapshot = store.take_snapshot()

    check("snapshot_created", snapshot is not None)
    check("snapshot_has_claims", len(snapshot.claims) > 0)
    check("snapshot_has_evidence", len(snapshot.evidence) > 0)
    check("snapshot_scoped_principal", snapshot.scope_principal == "kai")
    check("snapshot_scoped_purpose", snapshot.scope_purpose == "world_state")
    check("snapshot_has_digest", snapshot.digest is not None)
    check("snapshot_digest_verifies", snapshot.verify_digest())
    check("snapshot_has_sequence_digest", snapshot.event_sequence_digest is not None)


# ── 8. Snapshot immutability ─────────────────────────────────────────

def test_snapshot_immutability():
    store = SnapshotStore(principal=_principal())
    store.ingest_event(_event("weather_reading", EventSource.WEATHER, {"summary": "A"}))
    snap1 = store.take_snapshot()
    snap1_claims = len(snap1.claims)
    snap1_digest = snap1.digest.value

    store.ingest_event(_event("docker_status", EventSource.DOCKER, {"summary": "B"}))
    snap2 = store.take_snapshot()

    check("snap1_unchanged_claims", len(snap1.claims) == snap1_claims)
    check("snap1_unchanged_digest", snap1.digest.value == snap1_digest)
    check("snap2_has_more_claims", len(snap2.claims) > snap1_claims)
    check("snap2_different_digest", snap2.digest.value != snap1_digest)


# ── 9. Supersession — newer claim supersedes older ───────────────────

def test_supersession():
    store = SnapshotStore(principal=_principal())

    e1 = _event("weather_reading", EventSource.WEATHER, {"summary": "Rain"})
    e1 = e1.model_copy(update={
        "provenance": Provenance(source="test", independence_group="weather-src"),
        "digest": None,
    })
    e1.digest = e1._make_digest()
    store.ingest_event(e1)

    e2 = _event("weather_reading", EventSource.WEATHER, {"summary": "Sun"})
    e2 = e2.model_copy(update={
        "provenance": Provenance(source="test", independence_group="weather-src"),
        "digest": None,
    })
    e2.digest = e2._make_digest()
    store.ingest_event(e2)

    active = store.active_claims()
    active_env = [c for c in active if c.domain == "environment"]
    check("superseded_reduces_active", len(active_env) <= 2,
          f"got {len(active_env)}")

    snapshot = store.take_snapshot()
    superseded = [c for c in store._claims.values() if c.state == ContractState.SUPERSEDED]
    check("superseded_claims_exist", len(superseded) > 0)
    check("superseded_not_in_snapshot_active",
          all(c.id not in [a.id for a in snapshot.claims] for c in superseded))


# ── 10. Conflicts from different independence groups ─────────────────

def test_conflicts_visible():
    store = SnapshotStore(principal=_principal())

    e1 = _event("weather_reading", EventSource.WEATHER, {"summary": "Rainy"})
    e1 = e1.model_copy(update={
        "provenance": Provenance(source="test", independence_group="source_A"),
        "digest": None,
    })
    e1.digest = e1._make_digest()
    store.ingest_event(e1)

    e2 = _event("weather_reading", EventSource.WEATHER, {"summary": "Sunny"})
    e2 = e2.model_copy(update={
        "provenance": Provenance(source="test", independence_group="source_B"),
        "digest": None,
    })
    e2.digest = e2._make_digest()
    store.ingest_event(e2)

    conflicts = store.conflicts()
    check("conflicts_detected", len(conflicts) > 0,
          f"got {len(conflicts)} conflicts")
    if conflicts:
        check("conflict_has_domain", "domain" in conflicts[0])
        check("conflict_has_groups", "groups" in conflicts[0])
        check("conflict_multiple_groups", len(conflicts[0]["groups"]) > 1)

    snapshot = store.take_snapshot()
    check("conflicts_in_snapshot", len(snapshot.conflicts) > 0)
    env_claims = [c for c in snapshot.claims if c.domain == "environment"]
    check("conflicting_claims_both_active", len(env_claims) >= 2,
          f"got {len(env_claims)}")


# ── 11. Principal-scoped views ───────────────────────────────────────

def test_scoped_views():
    store = SnapshotStore(principal=_principal("kai"), purpose="world_state")
    store.ingest_event(_event("weather_reading", EventSource.WEATHER, {"summary": "Fine"}))
    store.take_snapshot()

    view_kai = store.scoped_view("kai")
    check("scoped_view_kai_ok", view_kai is not None)

    view_other = store.scoped_view("other_principal")
    check("scoped_view_other_blocked", view_other is None)

    view_wrong_purpose = store.scoped_view("kai", purpose="trading")
    check("scoped_view_wrong_purpose", view_wrong_purpose is None)


# ── 12. Snapshot reproducibility ─────────────────────────────────────

def test_snapshot_reproducibility():
    events = [
        _event("weather_reading", EventSource.WEATHER, {"summary": "Clear 20°C", "temp_c": 20}),
        _event("system_metrics", EventSource.SYSTEM, {"cpu_percent": 40, "memory_percent": 55}),
        _event("docker_status", EventSource.DOCKER, {"summary": "10 containers"}),
    ]

    store1 = SnapshotStore(principal=_principal())
    store1.ingest_events(events)
    snap1 = store1.take_snapshot()

    store2 = SnapshotStore(principal=_principal())
    store2.ingest_events(events)
    snap2 = store2.take_snapshot()

    check("reproducible_same_claim_count", len(snap1.claims) == len(snap2.claims))
    check("reproducible_same_evidence_count", len(snap1.evidence) == len(snap2.evidence))
    check("reproducible_same_sequence_digest",
          snap1.event_sequence_digest == snap2.event_sequence_digest)

    snap1_domains = sorted(c.domain for c in snap1.claims)
    snap2_domains = sorted(c.domain for c in snap2.claims)
    check("reproducible_same_domains", snap1_domains == snap2_domains)


# ── 13. Replay from events ───────────────────────────────────────────

def test_replay_from_events():
    events = [
        _event("weather_reading", EventSource.WEATHER, {"summary": "Warm"}),
        _event("git_status", EventSource.GIT, {"summary": "2 repos"}),
    ]

    store = SnapshotStore(principal=_principal())
    store.ingest_events(events)
    original = store.take_snapshot()

    replayed = store.replay_from_events(events)

    check("replay_same_claim_count", len(replayed.claims) == len(original.claims))
    check("replay_same_sequence_digest",
          replayed.event_sequence_digest == original.event_sequence_digest)


# ── 14. Bounded retention ────────────────────────────────────────────

def test_bounded_retention():
    store = SnapshotStore(principal=_principal(), max_snapshots=3)

    for i in range(5):
        store.ingest_event(
            _event("weather_reading", EventSource.WEATHER, {"summary": f"Day {i}"})
        )
        store.take_snapshot()

    check("retention_capped", store.snapshot_count() == 3)
    check("retention_deleted_tracked", len(store.deleted_snapshot_ids) == 2)


# ── 15. Deleted/superseded not active in snapshot ────────────────────

def test_deleted_superseded_not_active():
    store = SnapshotStore(principal=_principal())

    e1 = _event("weather_reading", EventSource.WEATHER, {"summary": "Old"})
    e1 = e1.model_copy(update={
        "provenance": Provenance(source="test", independence_group="wx"),
        "digest": None,
    })
    e1.digest = e1._make_digest()
    store.ingest_event(e1)

    e2 = _event("weather_reading", EventSource.WEATHER, {"summary": "New"})
    e2 = e2.model_copy(update={
        "provenance": Provenance(source="test", independence_group="wx"),
        "digest": None,
    })
    e2.digest = e2._make_digest()
    store.ingest_event(e2)

    snapshot = store.take_snapshot()
    superseded_ids = {
        cid for cid, c in store._claims.items()
        if c.state == ContractState.SUPERSEDED
    }

    for claim in snapshot.claims:
        check(f"active_claim_{claim.id[:8]}_not_superseded",
              claim.id not in superseded_ids)


# ── 16. Domain cap enforcement ───────────────────────────────────────

def test_domain_cap():
    store = SnapshotStore(principal=_principal(), max_claims_per_domain=3)

    for i in range(6):
        store.ingest_event(
            _event("docker_status", EventSource.DOCKER, {"summary": f"Container set {i}"})
        )

    active = [c for c in store.active_claims() if c.domain == "infrastructure"]
    check("domain_cap_enforced", len(active) <= 3,
          f"got {len(active)} active infra claims")


# ── 17. Degraded sources in snapshot ─────────────────────────────────

def test_degraded_sources():
    store = SnapshotStore(principal=_principal())

    stale_event = _event("weather_reading", EventSource.WEATHER, {"summary": "Stale"}, stale=True)
    store.ingest_event(stale_event)

    snapshot = store.take_snapshot()
    check("degraded_sources_listed", len(snapshot.degraded_sources) > 0)


# ── 18. Reducer revision tracked ─────────────────────────────────────

def test_reducer_revision():
    registry = ReducerRegistry()
    check("reducer_revision_set", registry.revision == REDUCER_REVISION)
    check("reducer_revision_is_semver", len(REDUCER_REVISION.split(".")) == 3)


# ── 19. Multiple domains don't interfere ─────────────────────────────

def test_domain_isolation():
    store = SnapshotStore(principal=_principal())

    store.ingest_event(_event("weather_reading", EventSource.WEATHER, {"summary": "Rain"}))
    store.ingest_event(_event("docker_status", EventSource.DOCKER, {"summary": "5 up"}))
    store.ingest_event(_event("git_status", EventSource.GIT, {"summary": "clean"}))

    snapshot = store.take_snapshot()
    domains = {c.domain for c in snapshot.claims}
    check("multiple_domains_present", len(domains) >= 3)
    check("environment_domain", "environment" in domains)
    check("infrastructure_domain", "infrastructure" in domains)
    check("development_domain", "development" in domains)


# ── 20. Snapshot serialisation round-trip ─────────────────────────────

def test_snapshot_round_trip():
    store = SnapshotStore(principal=_principal())
    store.ingest_event(_event("weather_reading", EventSource.WEATHER, {"summary": "Warm"}))
    snapshot = store.take_snapshot()

    serialised = snapshot.model_dump_json()
    check("snapshot_serialises", len(serialised) > 0)

    restored = WorldStateSnapshot.model_validate_json(serialised)
    check("snapshot_round_trip_id", restored.id == snapshot.id)
    check("snapshot_round_trip_claims", len(restored.claims) == len(snapshot.claims))
    check("snapshot_round_trip_digest", restored.digest.value == snapshot.digest.value)


# ── Runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_reducer_weather()
    test_reducer_system()
    test_reducer_generic()
    test_registry_dispatch()
    test_stale_claim()
    test_event_fact_lineage()
    test_snapshot_basic()
    test_snapshot_immutability()
    test_supersession()
    test_conflicts_visible()
    test_scoped_views()
    test_snapshot_reproducibility()
    test_replay_from_events()
    test_bounded_retention()
    test_deleted_superseded_not_active()
    test_domain_cap()
    test_degraded_sources()
    test_reducer_revision()
    test_domain_isolation()
    test_snapshot_round_trip()

    print(f"\n{'='*60}")
    print(f"UH-3 World State Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
