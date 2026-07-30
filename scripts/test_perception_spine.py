"""UH-2 perception spine exit-gate tests.

Exit gates (from roadmap):
  - invalid/stale/duplicate events are rejected or explicitly classified
  - restart/replay reproduces the same accepted event sequence
  - cross-principal events cannot leak

Additional tests:
  - journal append and replay
  - journal digest verification on replay
  - adapter coverage (all registered adapters produce valid events)
  - dedup by raw_hash and by event id
  - staleness detection and marking
  - ingress stats tracking
  - shadow runner report structure
  - adapter null/empty handling
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pydantic import ValidationError

from common.contracts.base import (
    ContractState,
    Principal,
    Provenance,
    RiskTier,
)
from common.contracts.perception import EventSource, PerceptionEvent
from common.perception_spine.journal import EventJournal, JournalEntry
from common.perception_spine.ingress import (
    IngressResult,
    IngressVerdict,
    PerceptionIngress,
)
from common.perception_spine.adapters import (
    ADAPTER_REGISTRY,
    adapt_weather,
    adapt_calendar,
    adapt_docker,
    adapt_git,
    adapt_system_metrics,
    adapt_screen,
    adapt_clipboard,
    adapt_email,
    adapt_news,
    adapt_telegram,
    adapt_market,
    _hash_payload,
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


def _principal(identity: str = "kai") -> Principal:
    return Principal(identity=identity, role="system")


def _event(
    event_type: str = "test",
    source: EventSource = EventSource.MANUAL,
    principal: Principal | None = None,
    payload: dict | None = None,
    raw_hash: str | None = None,
    source_timestamp: datetime | None = None,
) -> PerceptionEvent:
    return PerceptionEvent(
        event_type=event_type,
        source_type=source,
        principal=principal or _principal(),
        purpose="test",
        provenance=Provenance(source="test"),
        payload=payload or {},
        raw_hash=raw_hash,
        source_timestamp=source_timestamp,
    )


# ── 1. Journal: append and replay ─────────────────────────────────────

def test_journal_append_replay():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")

        check("journal_starts_empty", journal.count() == 0)

        e1 = _event("ev1", payload={"k": "v1"})
        off1 = journal.append(e1)
        check("journal_first_offset_zero", off1 == 0)
        check("journal_count_after_one", journal.count() == 1)

        e2 = _event("ev2", payload={"k": "v2"})
        off2 = journal.append(e2)
        check("journal_second_offset_one", off2 == 1)
        check("journal_count_after_two", journal.count() == 2)

        entries = list(journal.replay())
        check("replay_returns_all", len(entries) == 2)
        check("replay_first_offset", entries[0].offset == 0)
        check("replay_second_offset", entries[1].offset == 1)
        check("replay_preserves_event_type", entries[0].event.event_type == "ev1")
        check("replay_preserves_payload", entries[1].event.payload == {"k": "v2"})


# ── 2. Journal: replay from offset ───────────────────────────────────

def test_journal_replay_from_offset():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        for i in range(5):
            journal.append(_event(f"ev{i}"))

        entries = list(journal.replay(from_offset=3))
        check("replay_from_offset_count", len(entries) == 2)
        check("replay_from_offset_first", entries[0].offset == 3)
        check("replay_from_offset_last", entries[1].offset == 4)


# ── 3. Journal: digest verification on replay ────────────────────────

def test_journal_digest_verification():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        journal.append(_event("verified"))

        entries = list(journal.replay(verify_digests=True))
        check("digest_verified_on_replay", len(entries) == 1)
        check("digest_entry_valid", entries[0].event.verify_digest())


# ── 4. Journal: recovery after restart ───────────────────────────────

def test_journal_restart_recovery():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.jsonl"

        j1 = EventJournal(path)
        j1.append(_event("before_restart_1"))
        j1.append(_event("before_restart_2"))
        j1.append(_event("before_restart_3"))

        j2 = EventJournal(path)
        check("recovery_offset_correct", j2.next_offset == 3)

        off = j2.append(_event("after_restart"))
        check("recovery_continues_from_correct_offset", off == 3)

        entries = list(j2.replay())
        check("recovery_all_entries_present", len(entries) == 4)
        check("recovery_sequence_preserved",
              [e.event.event_type for e in entries] ==
              ["before_restart_1", "before_restart_2", "before_restart_3", "after_restart"])


# ── 5. Journal: replay reproduces same sequence ─────────────────────

def test_replay_determinism():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "test.jsonl"
        journal = EventJournal(path)
        events = [_event(f"det{i}", payload={"i": i}) for i in range(10)]
        for ev in events:
            journal.append(ev)

        replay1 = [(e.offset, e.event.id, e.event.event_type) for e in journal.replay()]
        replay2 = [(e.offset, e.event.id, e.event.event_type) for e in journal.replay()]
        check("replay_deterministic", replay1 == replay2)

        j2 = EventJournal(path)
        replay3 = [(e.offset, e.event.id, e.event.event_type) for e in j2.replay()]
        check("replay_after_restart_deterministic", replay1 == replay3)


# ── 6. Ingress: invalid events rejected ──────────────────────────────

def test_ingress_rejects_invalid():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(journal=journal, principal=_principal())

        result = ingress.submit_raw({"event_type": 123, "bad": True})
        check("invalid_event_rejected", result.verdict == IngressVerdict.REJECTED_INVALID)
        check("invalid_no_offset", result.offset is None)
        check("invalid_has_reason", len(result.reason) > 0)

        result2 = ingress.submit_raw({})
        check("empty_event_rejected", result2.verdict == IngressVerdict.REJECTED_INVALID)


# ── 7. Ingress: duplicate detection by raw_hash ─────────────────────

def test_ingress_dedup_by_hash():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(journal=journal, principal=_principal())

        e1 = _event("dup_test", raw_hash="abc123")
        r1 = ingress.submit(e1)
        check("first_submit_accepted", r1.verdict == IngressVerdict.ACCEPTED)

        e2 = _event("dup_test_2", raw_hash="abc123")
        r2 = ingress.submit(e2)
        check("duplicate_hash_rejected", r2.verdict == IngressVerdict.REJECTED_DUPLICATE)
        check("duplicate_reason_has_key", "abc123" in r2.reason)


# ── 8. Ingress: duplicate detection by event id ─────────────────────

def test_ingress_dedup_by_id():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(journal=journal, principal=_principal())

        e1 = _event("id_dup")
        r1 = ingress.submit(e1)
        check("id_first_accepted", r1.verdict == IngressVerdict.ACCEPTED)

        e2 = _event("id_dup_2")
        e2 = e2.model_copy(update={"id": e1.id})
        r2 = ingress.submit(e2)
        check("duplicate_id_rejected", r2.verdict == IngressVerdict.REJECTED_DUPLICATE)


# ── 9. Ingress: stale events classified ──────────────────────────────

def test_ingress_stale_events():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(
            journal=journal, principal=_principal(), freshness_seconds=60
        )

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        e1 = _event("stale_test", source_timestamp=old_time)
        r1 = ingress.submit(e1)
        check("stale_event_accepted_stale", r1.verdict == IngressVerdict.ACCEPTED_STALE)
        check("stale_event_has_offset", r1.offset is not None)
        check("stale_event_marked", r1.event is not None and r1.event.stale is True)


# ── 10. Ingress: fresh events accepted normally ──────────────────────

def test_ingress_fresh_events():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(
            journal=journal, principal=_principal(), freshness_seconds=600
        )

        recent_time = datetime.now(timezone.utc) - timedelta(seconds=30)
        e1 = _event("fresh_test", source_timestamp=recent_time)
        r1 = ingress.submit(e1)
        check("fresh_event_accepted", r1.verdict == IngressVerdict.ACCEPTED)
        check("fresh_event_not_stale", r1.event is not None and r1.event.stale is False)


# ── 11. Ingress: cross-principal rejection ───────────────────────────

def test_cross_principal_isolation():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(journal=journal, principal=_principal("kai"))

        foreign_event = _event("foreign", principal=_principal("attacker"))
        r = ingress.submit(foreign_event)
        check("cross_principal_rejected",
              r.verdict == IngressVerdict.REJECTED_CROSS_PRINCIPAL)
        check("cross_principal_no_journal", journal.count() == 0)
        check("cross_principal_reason_has_identity", "attacker" in r.reason)


# ── 12. Ingress: stats tracking ──────────────────────────────────────

def test_ingress_stats():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "test.jsonl")
        ingress = PerceptionIngress(
            journal=journal, principal=_principal(), freshness_seconds=60
        )

        ingress.submit(_event("a"))
        ingress.submit(_event("b", raw_hash="same"))
        ingress.submit(_event("c", raw_hash="same"))
        old = datetime.now(timezone.utc) - timedelta(seconds=120)
        ingress.submit(_event("d", source_timestamp=old))
        ingress.submit(_event("foreign", principal=_principal("other")))
        ingress.submit_raw({"bad": True})

        stats = ingress.stats
        check("stats_accepted", stats["accepted"] == 2)
        check("stats_duplicate", stats["rejected_duplicate"] == 1)
        check("stats_stale", stats["accepted_stale"] == 1)
        check("stats_cross_principal", stats["rejected_cross_principal"] == 1)
        check("stats_invalid", stats["rejected_invalid"] == 1)


# ── 13. Adapters: all registered adapters exist ──────────────────────

def test_adapter_registry():
    expected = {"weather", "calendar", "docker", "git", "system",
                "screen", "clipboard", "email", "news", "telegram", "market"}
    check("adapter_registry_complete", set(ADAPTER_REGISTRY.keys()) == expected)
    for name, fn in ADAPTER_REGISTRY.items():
        check(f"adapter_{name}_callable", callable(fn))


# ── 14. Adapters: produce valid PerceptionEvents ─────────────────────

def test_adapters_produce_valid_events():
    p = _principal()

    cases = {
        "weather": {"summary": "Sunny 22°C"},
        "calendar": {"summary": "2 events today"},
        "docker": {"summary": "12 containers running"},
        "git": {"summary": "3 repos, 1 dirty"},
        "system": {"cpu_percent": 45, "memory": {"percent": 62}},
        "screen": {"watching": True, "last_diff_score": 0.3},
        "clipboard": {"content": "hello world", "id": "c1"},
        "email": {"unread_count": 5, "subjects": ["test"]},
        "news": {"articles": [{"title": "Headline 1"}, {"title": "Headline 2"}]},
        "telegram": {"text": "Hello Kai", "chat_id": "12345"},
        "market": {"symbol": "BTCUSDT", "price": 42000.0},
    }

    for name, data in cases.items():
        adapter = ADAPTER_REGISTRY[name]
        event = adapter(data, p)
        check(f"adapter_{name}_returns_event", event is not None)
        if event is not None:
            check(f"adapter_{name}_valid_type", isinstance(event, PerceptionEvent))
            check(f"adapter_{name}_has_raw_hash", event.raw_hash is not None)
            check(f"adapter_{name}_has_tags", len(event.tags) > 0)
            check(f"adapter_{name}_digest_verifies", event.verify_digest())
            check(f"adapter_{name}_correct_principal", event.principal.identity == "kai")


# ── 15. Adapters: null/empty handling ────────────────────────────────

def test_adapters_handle_empty():
    p = _principal()
    for name, adapter in ADAPTER_REGISTRY.items():
        result_none = adapter(None, p)
        check(f"adapter_{name}_none_returns_none", result_none is None)
        result_empty = adapter({}, p)
        check(f"adapter_{name}_empty_returns_none", result_empty is None)


# ── 16. Adapters: screen ignores inactive ────────────────────────────

def test_screen_adapter_ignores_inactive():
    p = _principal()
    result = adapt_screen({"watching": False, "last_diff_score": 0.5}, p)
    check("screen_not_watching_returns_none", result is None)


# ── 17. Hash payload determinism ─────────────────────────────────────

def test_hash_determinism():
    d1 = {"b": 2, "a": 1}
    d2 = {"a": 1, "b": 2}
    check("hash_payload_deterministic", _hash_payload(d1) == _hash_payload(d2))
    d3 = {"a": 1, "b": 3}
    check("hash_payload_changes_with_value", _hash_payload(d1) != _hash_payload(d3))


# ── 18. End-to-end: adapter → ingress → journal → replay ────────────

def test_end_to_end_flow():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "e2e.jsonl"
        journal = EventJournal(path)
        ingress = PerceptionIngress(journal=journal, principal=_principal())

        weather_data = {"summary": "Rainy 15°C", "temp_c": 15}
        event = adapt_weather(weather_data, _principal())
        check("e2e_adapter_produces_event", event is not None)

        r = ingress.submit(event)
        check("e2e_ingress_accepts", r.verdict == IngressVerdict.ACCEPTED)
        check("e2e_has_offset", r.offset == 0)

        entries = list(journal.replay(verify_digests=True))
        check("e2e_journal_has_entry", len(entries) == 1)
        check("e2e_replay_event_type", entries[0].event.event_type == "weather_reading")
        check("e2e_replay_payload_intact",
              entries[0].event.payload["summary"] == "Rainy 15°C")
        check("e2e_replay_digest_valid", entries[0].event.verify_digest())


# ── 19. End-to-end: multiple sensors in sequence ─────────────────────

def test_multi_sensor_sequence():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "multi.jsonl")
        ingress = PerceptionIngress(journal=journal, principal=_principal())

        sensor_data = [
            ("weather", {"summary": "Clear 20°C"}),
            ("docker", {"summary": "10 containers"}),
            ("git", {"summary": "2 repos clean"}),
            ("system", {"cpu_percent": 30, "memory": {"percent": 55}}),
        ]

        for name, data in sensor_data:
            event = ADAPTER_REGISTRY[name](data, _principal())
            if event:
                ingress.submit(event)

        check("multi_sensor_all_journaled", journal.count() == 4)

        entries = list(journal.replay())
        types = [e.event.event_type for e in entries]
        check("multi_sensor_order_preserved",
              types == ["weather_reading", "docker_status", "git_status", "system_metrics"])

        offsets = [e.offset for e in entries]
        check("multi_sensor_offsets_sequential", offsets == [0, 1, 2, 3])


# ── 20. Cross-principal: second principal cannot read first ──────────

def test_cross_principal_journal_isolation():
    with tempfile.TemporaryDirectory() as td:
        j_kai = EventJournal(Path(td) / "kai.jsonl")
        j_other = EventJournal(Path(td) / "other.jsonl")
        i_kai = PerceptionIngress(journal=j_kai, principal=_principal("kai"))
        i_other = PerceptionIngress(journal=j_other, principal=_principal("other"))

        e_kai = _event("kai_only")
        r = i_kai.submit(e_kai)
        check("kai_event_accepted", r.verdict == IngressVerdict.ACCEPTED)

        e_cross = _event("cross_attempt", principal=_principal("kai"))
        r_cross = i_other.submit(e_cross)
        check("cross_attempt_rejected",
              r_cross.verdict == IngressVerdict.REJECTED_CROSS_PRINCIPAL)
        check("other_journal_empty", j_other.count() == 0)
        check("kai_journal_has_event", j_kai.count() == 1)


# ── 21. Journal truncate ─────────────────────────────────────────────

def test_journal_truncate():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "trunc.jsonl")
        journal.append(_event("a"))
        journal.append(_event("b"))
        check("pre_truncate_count", journal.count() == 2)

        journal.truncate()
        check("post_truncate_count", journal.count() == 0)
        check("post_truncate_replay_empty", list(journal.replay()) == [])

        off = journal.append(_event("c"))
        check("post_truncate_offset_resets", off == 0)


# ── 22. Ingress received_at is set ───────────────────────────────────

def test_received_at_stamped():
    with tempfile.TemporaryDirectory() as td:
        journal = EventJournal(Path(td) / "ts.jsonl")
        ingress = PerceptionIngress(journal=journal, principal=_principal())

        before = datetime.now(timezone.utc)
        e = _event("ts_test")
        r = ingress.submit(e)
        after = datetime.now(timezone.utc)

        check("received_at_set", r.event.received_at is not None)
        if r.event.received_at:
            check("received_at_in_range",
                  before <= r.event.received_at <= after)


# ── Runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_journal_append_replay()
    test_journal_replay_from_offset()
    test_journal_digest_verification()
    test_journal_restart_recovery()
    test_replay_determinism()
    test_ingress_rejects_invalid()
    test_ingress_dedup_by_hash()
    test_ingress_dedup_by_id()
    test_ingress_stale_events()
    test_ingress_fresh_events()
    test_cross_principal_isolation()
    test_ingress_stats()
    test_adapter_registry()
    test_adapters_produce_valid_events()
    test_adapters_handle_empty()
    test_screen_adapter_ignores_inactive()
    test_hash_determinism()
    test_end_to_end_flow()
    test_multi_sensor_sequence()
    test_cross_principal_journal_isolation()
    test_journal_truncate()
    test_received_at_stamped()

    print(f"\n{'='*60}")
    print(f"UH-2 Perception Spine Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
