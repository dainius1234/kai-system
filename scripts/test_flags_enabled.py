"""Every migration flag ON, end to end (UH tracker gap G-11).

All four migration flags default to the legacy path, which is correct for
deployment but leaves an obvious question: does the *new* path actually
work when they are switched on?

Every other suite tests one flag in isolation. This one turns them all on
together and runs perception → world state → Cortex → proposal → policy →
approval → capability → actuator → verification in a single pass, because
flags that each work alone can still interact badly.

    KAI_PERCEPTION_MODE=active     spine feeds the world state
    KAI_CORTEX_SOURCE=world_state  Cortex reads the world state
    KAI_AUTONOMY_ENFORCE=true      scoped grants bind
    KAI_SERVICE_TOKEN=<set>        downstream endpoints authenticate
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import (
    Principal,
    Provenance,
    RiskTier,
    VerificationVerdict,
)
from common.contracts.action import ActionProposal
from common.contracts.autonomy import AutonomyLevel, EvidenceGrade
from common.contracts.perception import EventSource, PerceptionEvent
from common.actuator_registry.catalog import build_catalog
from common.actuator_registry.migration import migrate_tier
from common.actuator_registry.mutating_handlers import attach_all_handlers
from common.actuator_registry.registry import MigrationTier
from common.autonomy.authority import AutonomyAuthority
from common.autonomy.calibration import CalibrationTracker
from common.autonomy.evidence_service import EvidenceService
from common.autonomy.legacy_bridge import ENFORCE_ENV, LegacyTrustBridge
from common.autonomy.verifier_registry import VerifierRegistry
from common.perception_spine.cortex_source import SOURCE_ENV, resolve_cortex_state
from common.perception_spine.ingress import IngressVerdict
from common.perception_spine.shadow import MODE_ENV, ShadowPerceptionRunner
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge
from common.policy_bridge.policy_engine import PolicyEngine
from common.service_auth import TOKEN_ENV, check_token
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


def _principal() -> Principal:
    return Principal(identity="kai", role="system")


class _AllFlagsOn:
    """Every migration flag enabled for the duration of the block."""

    FLAGS = {
        MODE_ENV: "active",
        SOURCE_ENV: "world_state",
        ENFORCE_ENV: "true",
        TOKEN_ENV: "integration-test-token",
    }

    def __enter__(self):
        self._saved = {k: os.environ.get(k) for k in self.FLAGS}
        os.environ.update(self.FLAGS)
        return self

    def __exit__(self, *exc):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return False


_tmpdir = tempfile.mkdtemp(prefix="flags_on_")
_counter = 0


def _journal_path() -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"j{_counter}.jsonl")


def _event(i: int, etype: str = "system") -> PerceptionEvent:
    payloads = {
        "system": {"cpu_percent": 20 + i, "memory_percent": 50},
        "docker": {"containers": [{"name": f"svc-{i}", "status": "running"}]},
        "git": {"branch": "main", "dirty": False},
    }
    return PerceptionEvent(
        source_type=EventSource.SYSTEM, event_type=etype,
        payload=payloads.get(etype, {"v": i}),
        principal=_principal(), purpose="integration",
        provenance=Provenance(source=f"sensor-{etype}"),
        source_timestamp=datetime.now(timezone.utc),
        raw_hash=f"flags-{etype}-{i}",
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Perception → world state → Cortex, all flags on
# ═══════════════════════════════════════════════════════════════════

def test_perception_feeds_world_state_and_cortex():
    with _AllFlagsOn():
        store = SnapshotStore(principal=_principal())
        runner = ShadowPerceptionRunner(
            journal_path=_journal_path(), world_state=store,
        )
        check("runner_picked_up_active", runner.mode == "active")

        for i, etype in enumerate(["system", "docker", "git"]):
            result = runner._ingress.submit(_event(i, etype))
            check(f"{etype}_accepted",
                  result.verdict == IngressVerdict.ACCEPTED, result.reason)
            runner._maybe_reduce(result.event, {})

        check("all_three_reduced", runner.reduced_count == 3,
              str(runner.reduced_count))
        check("world_state_populated", len(store.active_claims()) >= 3)
        check("journal_durable", runner.journal.count() == 3)

        polled = {"level1_facts": ["stale polled fact"]}
        state = resolve_cortex_state(polled, store=store)
        check("cortex_used_world_state", state.get("source") == "world_state")
        check("cortex_dropped_polled",
              "stale polled fact" not in state.get("level1_facts", []))
        check("cortex_has_real_facts", len(state.get("level1_facts", [])) >= 3)


def test_cortex_still_falls_back_with_flags_on():
    """Flags on must not mean perception can go blank."""
    with _AllFlagsOn():
        polled = {"level1_facts": ["polled"]}
        state = resolve_cortex_state(
            polled, store=SnapshotStore(principal=_principal())
        )
        check("empty_world_still_falls_back", state == polled)


# ═══════════════════════════════════════════════════════════════════
# 2. Autonomy enforcement binds when the flag is on
# ═══════════════════════════════════════════════════════════════════

def _authority_with_grant(capability="paper_trade_open", domain="trading"):
    principal = _principal()
    evidence = EvidenceService(principal=principal)
    calibration = CalibrationTracker(principal=principal)
    for _ in range(30):
        record = evidence.record(
            grade=EvidenceGrade.VERIFIED_OUTCOME, domain=domain,
            task_type="trade", observed_by="portfolio-verifier",
            provenance=Provenance(source="verifier:portfolio"),
        )
        calibration.observe("trade", domain, "r1", 0.9, record, True)
    authority = AutonomyAuthority(principal, evidence, calibration)
    authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability=capability,
        domain=domain, task_type="trade", revision="r1",
        granted_by="dainius", max_invocations=10,
        independent_verifier_count=2,
    )
    return authority


def test_enforcement_denies_without_grant():
    with _AllFlagsOn():
        principal = _principal()
        authority = AutonomyAuthority(
            principal,
            EvidenceService(principal=principal),
            CalibrationTracker(principal=principal),
        )
        bridge = LegacyTrustBridge(authority, principal)
        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=True, legacy_reason="trust ok",
        )
        check("enforcing_denies_ungranted", not allowed)
        check("enforcing_names_scoped", "scoped authority denies" in reason)


def test_enforcement_permits_with_grant():
    with _AllFlagsOn():
        bridge = LegacyTrustBridge(_authority_with_grant(), _principal())
        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=True, legacy_reason="trust ok",
        )
        check("enforcing_permits_granted", allowed, reason)


def test_enforcement_still_cannot_widen():
    """Even with a grant, a legacy denial stands."""
    with _AllFlagsOn():
        bridge = LegacyTrustBridge(_authority_with_grant(), _principal())
        allowed, reason = bridge.gate(
            "paper_trade_open", legacy_allowed=False, legacy_reason="legacy no",
        )
        check("legacy_deny_survives_enforcement", not allowed)
        check("legacy_deny_named", "legacy gate denies" in reason)


# ═══════════════════════════════════════════════════════════════════
# 3. Service auth active
# ═══════════════════════════════════════════════════════════════════

def test_service_auth_enforced_with_token_set():
    with _AllFlagsOn():
        ok, status, _ = check_token("Bearer integration-test-token", "op")
        check("valid_token_accepted", ok)
        check("valid_token_200", status == 200)

        ok, status, _ = check_token("Bearer wrong", "op")
        check("wrong_token_rejected", not ok)
        check("wrong_token_403", status == 403)

        ok, status, _ = check_token(None, "op")
        check("no_header_rejected", not ok)
        check("no_header_401", status == 401)


# ═══════════════════════════════════════════════════════════════════
# 4. Full pipeline with every flag on
# ═══════════════════════════════════════════════════════════════════

def test_full_pipeline_all_flags_on():
    """Perception through to verified outcome, nothing mocked away."""
    with _AllFlagsOn():
        principal = _principal()

        store = SnapshotStore(principal=principal)
        runner = ShadowPerceptionRunner(
            journal_path=_journal_path(), world_state=store,
        )
        result = runner._ingress.submit(_event(99, "system"))
        runner._maybe_reduce(result.event, {})
        check("pipeline_world_state_ready", len(store.active_claims()) > 0)

        posts = []
        catalog = build_catalog(principal)
        attach_all_handlers(
            catalog,
            http_get=lambda u: {"read": True},
            http_post=lambda u, b, h: posts.append((u, h)) or {"done": True},
        )
        for tier in MigrationTier:
            migrate_tier(catalog, tier, principal)
        report = catalog.migration_report()
        check("pipeline_all_migrated",
              report["migrated"] == report["total_actuators"])

        proposal = ActionProposal(
            action_type="notify_desktop", description="integration",
            risk_tier=RiskTier.ACT_SUPERVISED, rationale="flags-on test",
            alternatives=["stay silent"], principal=principal,
            purpose="integration", provenance=Provenance(source="test"),
            estimated_value=10.0,
            world_state_snapshot_id=store.take_snapshot().id,
        )
        policy = PolicyEngine(principal=principal).evaluate(proposal)
        check("pipeline_policy_requires_approval",
              policy.decision.result == "requires_approval", policy.reason)

        approval = ApprovalGate().approve(proposal, "dainius", principal)
        bridge = CapabilityBridge()
        cap = bridge.issue(
            proposal, approval, "notify-service", "notify_desktop",
            principal, parameters={"title": "flags on", "body": "ok"},
        )
        bridge.consume(cap.id, "notify-service", principal)

        receipt = catalog.dispatch(
            cap, "notify-service", "notify_desktop", "wf-flags", principal
        )
        check("pipeline_dispatched", receipt.result.get("ok") is True)
        check("pipeline_posted", len(posts) == 1)
        check("pipeline_sent_auth_header",
              posts and posts[0][1].get("Authorization")
              == "Bearer integration-test-token")
        check("pipeline_declared_effects",
              "desktop_notification" in receipt.result.get("side_effects", []))

        verifiers = VerifierRegistry(principal=principal)
        verifiers.register(
            "desktop-verifier", "Desktop Verifier", ["notification"],
            "observation",
        )
        verifiers.set_actuator_group("notify-service", "execution")
        outcome = verifiers.verify(
            "desktop-verifier", "notify-service", "notification",
            "wf-flags", receipt.id, VerificationVerdict.CONFIRMED,
        )
        check("pipeline_verified",
              outcome.verdict == VerificationVerdict.CONFIRMED)
        check("pipeline_verifier_independent",
              outcome.provenance.independence_group == "observation")


def test_flags_off_restores_legacy_behaviour():
    """With flags cleared, every path reverts to the legacy default."""
    for key in (MODE_ENV, SOURCE_ENV, ENFORCE_ENV, TOKEN_ENV):
        os.environ.pop(key, None)

    from common.autonomy.legacy_bridge import enforcing
    from common.perception_spine.cortex_source import cortex_source
    from common.perception_spine.shadow import perception_mode

    check("mode_back_to_shadow", perception_mode() == "shadow")
    check("cortex_back_to_poll", cortex_source() == "poll")
    check("enforcement_back_off", enforcing() is False)

    polled = {"level1_facts": ["polled"]}
    store = SnapshotStore(principal=_principal())
    store.ingest_event(_event(1, "system"))
    check("cortex_ignores_world_state_when_off",
          resolve_cortex_state(polled, store=store) == polled)

    principal = _principal()
    bridge = LegacyTrustBridge(
        AutonomyAuthority(
            principal,
            EvidenceService(principal=principal),
            CalibrationTracker(principal=principal),
        ),
        principal,
    )
    allowed, reason = bridge.gate(
        "paper_trade_open", legacy_allowed=True, legacy_reason="legacy ok",
    )
    check("advisory_mode_keeps_legacy", allowed)
    check("advisory_reason_is_legacy", reason == "legacy ok")


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_perception_feeds_world_state_and_cortex()
    test_cortex_still_falls_back_with_flags_on()
    test_enforcement_denies_without_grant()
    test_enforcement_permits_with_grant()
    test_enforcement_still_cannot_widen()
    test_service_auth_enforced_with_token_set()
    test_full_pipeline_all_flags_on()
    test_flags_off_restores_legacy_behaviour()

    print(f"\n{'='*60}")
    print(f"Flags-Enabled Integration Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
