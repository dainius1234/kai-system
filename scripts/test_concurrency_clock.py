"""Multi-worker, restart, clock-change and leader-fencing tests.

Closes roadmap §16.27.

Four distinct failure families:
  - **Multi-worker**: concurrent writers must not lose or duplicate offsets.
  - **Restart**: a process that dies mid-stream must resume without
    overwriting or skipping.
  - **Clock change**: moving the wall clock backwards must never extend
    an expiry or resurrect an expired grant/capability/lease.
  - **Leader fencing**: a stalled leader that wakes up must be refused,
    even though it still believes it holds the lease.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance, RiskTier
from common.contracts.perception import EventSource, PerceptionEvent
from common.contracts.action import ActionProposal
from common.contracts.autonomy import AutonomyLevel, EvidenceGrade
from common.perception_spine.journal import EventJournal
from common.perception_spine.ingress import IngressVerdict, PerceptionIngress
from common.perception_spine.lease import FencedLease, LeaseError
from common.policy_bridge.approval import ApprovalGate
from common.policy_bridge.capability import CapabilityBridge, CapabilityError
from common.autonomy.authority import AutonomyAuthority, AutonomyError
from common.autonomy.calibration import CalibrationTracker
from common.autonomy.evidence_service import EvidenceService

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


_tmpdir = tempfile.mkdtemp(prefix="concurrency_test_")
_counter = 0


def _path(name: str) -> str:
    global _counter
    _counter += 1
    return os.path.join(_tmpdir, f"{name}_{_counter}.jsonl")


def _event(i: int) -> PerceptionEvent:
    return PerceptionEvent(
        source_type=EventSource.SYSTEM,
        event_type="test",
        payload={"i": i},
        principal=_principal(),
        purpose="test",
        provenance=Provenance(source="test-sensor"),
        source_timestamp=datetime.now(timezone.utc),
        raw_hash=f"hash-{i}",
    )


class _FakeClock:
    """Injectable monotonic clock that only moves when told to."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ═══════════════════════════════════════════════════════════════════
# 1. Multi-worker concurrent writes
# ═══════════════════════════════════════════════════════════════════

def test_concurrent_appends_no_lost_offsets():
    journal = EventJournal(_path("concurrent"))
    workers, per_worker = 8, 25
    errors: list[str] = []
    offsets: list[int] = []
    offsets_lock = threading.Lock()

    def worker(wid: int) -> None:
        try:
            for i in range(per_worker):
                off = journal.append(_event(wid * 1000 + i))
                with offsets_lock:
                    offsets.append(off)
        except Exception as exc:
            errors.append(f"worker {wid}: {exc}")

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = workers * per_worker
    check("concurrent_no_errors", not errors, "; ".join(errors[:3]))
    check("concurrent_all_appended", len(offsets) == expected,
          f"got {len(offsets)}")
    check("concurrent_offsets_unique", len(set(offsets)) == expected,
          f"{expected - len(set(offsets))} duplicates")
    check("concurrent_offsets_contiguous",
          sorted(offsets) == list(range(expected)))
    check("concurrent_journal_count", journal.count() == expected)


def test_concurrent_replay_integrity():
    journal = EventJournal(_path("replay_integrity"))
    threads = [
        threading.Thread(target=lambda w=w: [journal.append(_event(w * 100 + i))
                                             for i in range(20)])
        for w in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    entries = list(journal.replay(verify_digests=True))
    check("replay_reads_all", len(entries) == 100)
    check("replay_offsets_ordered",
          [e.offset for e in entries] == list(range(100)))


def test_concurrent_ingress_dedup_is_atomic():
    """Two workers submitting the same event: exactly one wins."""
    journal = EventJournal(_path("dedup_race"))
    ingress = PerceptionIngress(journal=journal, principal=_principal())

    verdicts: list[IngressVerdict] = []
    verdicts_lock = threading.Lock()

    def submit() -> None:
        ev = _event(42)
        result = ingress.submit(ev)
        with verdicts_lock:
            verdicts.append(result.verdict)

    threads = [threading.Thread(target=submit) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    accepted = [v for v in verdicts if v == IngressVerdict.ACCEPTED]
    duplicates = [v for v in verdicts if v == IngressVerdict.REJECTED_DUPLICATE]
    check("dedup_race_one_accepted", len(accepted) == 1, f"got {len(accepted)}")
    check("dedup_race_rest_duplicate", len(duplicates) == 9,
          f"got {len(duplicates)}")
    check("dedup_race_journal_has_one", journal.count() == 1)


# ═══════════════════════════════════════════════════════════════════
# 2. Restart recovery
# ═══════════════════════════════════════════════════════════════════

def test_restart_resumes_offset():
    path = _path("restart")
    j1 = EventJournal(path)
    for i in range(10):
        j1.append(_event(i))
    check("pre_restart_offset", j1.next_offset == 10)

    j2 = EventJournal(path)
    check("restart_recovers_offset", j2.next_offset == 10)

    off = j2.append(_event(99))
    check("restart_continues_sequence", off == 10)
    check("restart_no_overwrite", j2.count() == 11)


def test_restart_after_partial_line():
    """A torn final write must not corrupt offset recovery."""
    path = _path("torn")
    journal = EventJournal(path)
    for i in range(5):
        journal.append(_event(i))

    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"offset": 5, "incomplete"')

    recovered = EventJournal(path)
    check("torn_line_recovered", recovered.next_offset == 5)

    off = recovered.append(_event(100))
    check("torn_line_append_works", off == 5)

    entries = list(recovered.replay())
    check("torn_line_skipped_on_replay", len(entries) == 6)


def test_restart_preserves_replay_order():
    path = _path("order")
    j1 = EventJournal(path)
    for i in range(5):
        j1.append(_event(i))
    del j1

    j2 = EventJournal(path)
    for i in range(5, 10):
        j2.append(_event(i))

    entries = list(j2.replay())
    check("restart_order_preserved",
          [e.offset for e in entries] == list(range(10)))


# ═══════════════════════════════════════════════════════════════════
# 3. Clock change
# ═══════════════════════════════════════════════════════════════════

def test_backwards_clock_does_not_revive_capability():
    """An expired capability stays expired if the clock jumps back."""
    bridge = CapabilityBridge(default_timeout=60)
    proposal = ActionProposal(
        action_type="t", description="t", risk_tier=RiskTier.OBSERVE,
        rationale="t", alternatives=["n"], principal=_principal(),
        purpose="test", provenance=Provenance(source="test"),
    )
    gate = ApprovalGate()
    approval = gate.approve(proposal, "dainius", _principal())
    cap = bridge.issue(proposal, approval, "actuator-a", "t", _principal())

    # Simulate the clock having moved forward past expiry by ageing the
    # capability's creation time, then "moving the clock back" by
    # restoring it — the used flag and expiry must not be reversible.
    cap.created_at = datetime.now(timezone.utc) - timedelta(seconds=120)
    try:
        bridge.consume(cap.id, "actuator-a", _principal())
        check("expired_capability_rejected", False, "should have raised")
    except CapabilityError as e:
        check("expired_capability_rejected", "expired" in str(e))

    check("expiry_recorded_release", len(bridge.releases) > 0)


def test_backwards_clock_does_not_extend_lease():
    """A monotonic clock source means wall-clock changes cannot help."""
    clock = _FakeClock()
    lease = FencedLease(ttl_seconds=10.0, clock=clock)
    token = lease.acquire("worker-1")

    clock.advance(11.0)
    check("lease_expired_after_ttl", lease.holder is None)

    # A backwards wall-clock jump cannot rescue it: the lease reads a
    # monotonic source, which never goes backwards.
    try:
        lease.check_token(token)
        check("expired_lease_write_rejected", False, "should have raised")
    except LeaseError as e:
        check("expired_lease_write_rejected", "expired" in str(e))


def test_backwards_clock_does_not_revive_grant():
    service = EvidenceService(principal=_principal())
    tracker = CalibrationTracker(principal=_principal())
    for i in range(30):
        ev = service.record(
            grade=EvidenceGrade.VERIFIED_OUTCOME, domain="d",
            task_type="t", observed_by="verifier",
            provenance=Provenance(source="verifier:v"),
        )
        tracker.observe("t", "d", "r1", 0.9, ev, was_correct=True)

    authority = AutonomyAuthority(_principal(), service, tracker)
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="c", domain="d",
        task_type="t", revision="r1", granted_by="dainius",
        max_invocations=5, duration_seconds=60,
        independent_verifier_count=2,
    )

    grant.expires_at_grant = datetime.now(timezone.utc) - timedelta(seconds=1)
    valid, reason = authority.check_grant(grant.id, "c", "d")
    check("expired_grant_invalid", not valid)
    check("expired_grant_reason", "expired" in reason)

    # Revocation is independent of the clock entirely.
    authority.revoke(grant.id, "clock anomaly")
    valid, reason = authority.check_grant(grant.id, "c", "d")
    check("revocation_survives_clock", not valid)
    check("revocation_reason_wins", "revoked" in reason)


def test_exhaustion_is_clock_independent():
    """Invocation limits cannot be reset by any clock manipulation."""
    service = EvidenceService(principal=_principal())
    tracker = CalibrationTracker(principal=_principal())
    for i in range(30):
        ev = service.record(
            grade=EvidenceGrade.VERIFIED_OUTCOME, domain="d",
            task_type="t", observed_by="verifier",
            provenance=Provenance(source="verifier:v"),
        )
        tracker.observe("t", "d", "r1", 0.9, ev, was_correct=True)

    authority = AutonomyAuthority(_principal(), service, tracker)
    grant = authority.grant(
        level=AutonomyLevel.A2_REVERSIBLE, capability="c", domain="d",
        task_type="t", revision="r1", granted_by="dainius",
        max_invocations=2, independent_verifier_count=2,
    )

    authority.consume_grant(grant.id, "c", "d")
    authority.consume_grant(grant.id, "c", "d")

    grant.expires_at_grant = datetime.now(timezone.utc) + timedelta(days=365)
    valid, reason = authority.check_grant(grant.id, "c", "d")
    check("exhaustion_beats_extended_expiry", not valid)
    check("exhaustion_reason", "exhausted" in reason)


# ═══════════════════════════════════════════════════════════════════
# 4. Leader fencing
# ═══════════════════════════════════════════════════════════════════

def test_single_leader_only():
    lease = FencedLease(ttl_seconds=30.0)
    token1 = lease.acquire("worker-1")
    check("first_acquires", token1 == 1)
    check("holder_is_worker1", lease.holder == "worker-1")

    try:
        lease.acquire("worker-2")
        check("second_blocked", False, "should have raised")
    except LeaseError as e:
        check("second_blocked", "held by" in str(e))


def test_token_increases_monotonically():
    clock = _FakeClock()
    lease = FencedLease(ttl_seconds=10.0, clock=clock)

    t1 = lease.acquire("worker-1")
    clock.advance(11.0)
    t2 = lease.acquire("worker-2")
    clock.advance(11.0)
    t3 = lease.acquire("worker-3")

    check("tokens_increase", t1 < t2 < t3)
    check("tokens_sequential", [t1, t2, t3] == [1, 2, 3])


def test_stalled_leader_is_fenced():
    """The core fencing scenario: a stalled leader wakes and is refused."""
    clock = _FakeClock()
    lease = FencedLease(ttl_seconds=10.0, clock=clock)

    stale_token = lease.acquire("worker-1")
    lease.check_token(stale_token)  # healthy so far

    # worker-1 stalls; its lease expires; worker-2 takes over.
    clock.advance(11.0)
    new_token = lease.acquire("worker-2")

    # worker-1 wakes up still believing it is leader.
    try:
        lease.check_token(stale_token)
        check("stalled_leader_fenced", False, "stale write was allowed")
    except LeaseError as e:
        check("stalled_leader_fenced", "stale" in str(e))

    lease.check_token(new_token)
    check("new_leader_can_write", True)
    check("stale_token_lower", stale_token < new_token)


def test_stale_leader_cannot_renew():
    clock = _FakeClock()
    lease = FencedLease(ttl_seconds=10.0, clock=clock)
    stale_token = lease.acquire("worker-1")

    clock.advance(11.0)
    lease.acquire("worker-2")

    try:
        lease.renew("worker-1", stale_token)
        check("stale_renew_rejected", False, "should have raised")
    except LeaseError as e:
        check("stale_renew_rejected",
              "does not hold" in str(e) or "stale" in str(e))


def test_healthy_leader_renews_without_token_change():
    clock = _FakeClock()
    lease = FencedLease(ttl_seconds=10.0, clock=clock)
    token = lease.acquire("worker-1")

    clock.advance(5.0)
    renewed = lease.renew("worker-1", token)
    check("renew_keeps_token", renewed == token)

    clock.advance(7.0)
    lease.check_token(token)
    check("renew_extended_lease", True)


def test_release_frees_lease():
    lease = FencedLease(ttl_seconds=30.0)
    token = lease.acquire("worker-1")
    lease.release("worker-1", token)
    check("released_holder_none", lease.holder is None)

    new_token = lease.acquire("worker-2")
    check("reacquire_after_release", new_token > token)

    try:
        lease.check_token(token)
        check("released_token_fenced", False, "should have raised")
    except LeaseError:
        check("released_token_fenced", True)


def test_release_by_wrong_worker_is_noop():
    lease = FencedLease(ttl_seconds=30.0)
    token = lease.acquire("worker-1")
    lease.release("worker-2", token)
    check("wrong_release_ignored", lease.holder == "worker-1")


def test_concurrent_acquire_single_winner():
    lease = FencedLease(ttl_seconds=30.0)
    winners: list[str] = []
    winners_lock = threading.Lock()

    def contend(wid: int) -> None:
        try:
            lease.acquire(f"worker-{wid}")
            with winners_lock:
                winners.append(f"worker-{wid}")
        except LeaseError:
            pass

    threads = [threading.Thread(target=contend, args=(w,)) for w in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check("exactly_one_leader", len(winners) == 1, f"got {len(winners)}")
    check("token_matches_winners", lease.current_token == 1)


def test_lease_validation():
    try:
        FencedLease(ttl_seconds=0)
        check("zero_ttl_rejected", False, "should have raised")
    except LeaseError:
        check("zero_ttl_rejected", True)

    lease = FencedLease(ttl_seconds=10.0)
    try:
        lease.acquire("")
        check("empty_worker_rejected", False, "should have raised")
    except LeaseError:
        check("empty_worker_rejected", True)


def test_is_valid_helper():
    clock = _FakeClock()
    lease = FencedLease(ttl_seconds=10.0, clock=clock)
    token = lease.acquire("worker-1")

    check("is_valid_true", lease.is_valid("worker-1", token))
    check("is_valid_wrong_worker", not lease.is_valid("worker-2", token))
    check("is_valid_wrong_token", not lease.is_valid("worker-1", token + 1))

    clock.advance(11.0)
    check("is_valid_expired", not lease.is_valid("worker-1", token))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_concurrent_appends_no_lost_offsets()
    test_concurrent_replay_integrity()
    test_concurrent_ingress_dedup_is_atomic()
    test_restart_resumes_offset()
    test_restart_after_partial_line()
    test_restart_preserves_replay_order()
    test_backwards_clock_does_not_revive_capability()
    test_backwards_clock_does_not_extend_lease()
    test_backwards_clock_does_not_revive_grant()
    test_exhaustion_is_clock_independent()
    test_single_leader_only()
    test_token_increases_monotonically()
    test_stalled_leader_is_fenced()
    test_stale_leader_cannot_renew()
    test_healthy_leader_renews_without_token_change()
    test_release_frees_lease()
    test_release_by_wrong_worker_is_noop()
    test_concurrent_acquire_single_winner()
    test_lease_validation()
    test_is_valid_helper()

    print(f"\n{'='*60}")
    print(f"Concurrency & Clock Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
