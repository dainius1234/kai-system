"""Oversized / deep / high-cardinality payload tests.

Closes roadmap §16.4 ("oversized/deep/high-cardinality event payloads").

A compromised or faulty sensor must not be able to exhaust memory in the
reducers by sending one enormous event.  The ingress rejects before the
event reaches the journal or the world state.
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.contracts.base import Principal, Provenance
from common.contracts.perception import EventSource, PerceptionEvent
from common.perception_spine.ingress import (
    MAX_PAYLOAD_BYTES,
    MAX_PAYLOAD_DEPTH,
    MAX_PAYLOAD_KEYS,
    MAX_STRING_LENGTH,
    IngressVerdict,
    PerceptionIngress,
    check_payload_bounds,
)
from common.perception_spine.journal import EventJournal

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


_tmpdir = tempfile.mkdtemp(prefix="bounds_test_")
_counter = 0


def _ingress() -> PerceptionIngress:
    global _counter
    _counter += 1
    return PerceptionIngress(
        journal=EventJournal(os.path.join(_tmpdir, f"j{_counter}.jsonl")),
        principal=_principal(),
    )


def _event(payload: dict) -> PerceptionEvent:
    return PerceptionEvent(
        source_type=EventSource.SYSTEM,
        event_type="test",
        payload=payload,
        principal=_principal(),
        purpose="test",
        provenance=Provenance(source="test-sensor"),
        source_timestamp=datetime.now(timezone.utc),
    )


def _deep(depth: int) -> dict:
    root: dict = {}
    cursor = root
    for _ in range(depth):
        cursor["n"] = {}
        cursor = cursor["n"]
    return root


# ── 1. Depth ────────────────────────────────────────────────────────

def test_depth_within_limit_accepted():
    result = _ingress().submit(_event(_deep(MAX_PAYLOAD_DEPTH - 2)))
    check("shallow_payload_accepted",
          result.verdict == IngressVerdict.ACCEPTED, result.reason)


def test_excessive_depth_rejected():
    result = _ingress().submit(_event(_deep(MAX_PAYLOAD_DEPTH + 10)))
    check("deep_payload_rejected",
          result.verdict == IngressVerdict.REJECTED_OVERSIZED)
    check("deep_reason", "too deep" in result.reason)


def test_deeply_nested_lists_rejected():
    payload: object = "leaf"
    for _ in range(MAX_PAYLOAD_DEPTH + 10):
        payload = [payload]
    result = _ingress().submit(_event({"nested": payload}))
    check("deep_list_rejected",
          result.verdict == IngressVerdict.REJECTED_OVERSIZED)


def test_depth_check_does_not_stack_overflow():
    """The bounds check must survive a payload built to blow the stack."""
    try:
        reason = check_payload_bounds(_deep(2000))
        check("deep_check_survives", reason is not None)
    except RecursionError:
        check("deep_check_survives", False, "RecursionError in bounds check")


# ── 2. Cardinality ──────────────────────────────────────────────────

def test_high_cardinality_rejected():
    payload = {f"k{i}": i for i in range(MAX_PAYLOAD_KEYS + 100)}
    result = _ingress().submit(_event(payload))
    check("high_cardinality_rejected",
          result.verdict == IngressVerdict.REJECTED_OVERSIZED)
    check("cardinality_reason", "high-cardinality" in result.reason)


def test_normal_cardinality_accepted():
    payload = {f"k{i}": i for i in range(50)}
    result = _ingress().submit(_event(payload))
    check("normal_cardinality_accepted",
          result.verdict == IngressVerdict.ACCEPTED, result.reason)


def test_large_list_rejected():
    result = _ingress().submit(_event({"items": list(range(MAX_PAYLOAD_KEYS + 100))}))
    check("large_list_rejected",
          result.verdict == IngressVerdict.REJECTED_OVERSIZED)


# ── 3. Size ─────────────────────────────────────────────────────────

def test_oversized_string_rejected():
    result = _ingress().submit(_event({"blob": "x" * (MAX_STRING_LENGTH + 1000)}))
    check("oversized_string_rejected",
          result.verdict == IngressVerdict.REJECTED_OVERSIZED)
    check("string_reason", "string too long" in result.reason)


def test_oversized_total_rejected():
    chunk = "y" * 1000
    payload = {f"k{i}": chunk for i in range(400)}
    reason = check_payload_bounds(payload)
    check("oversized_total_rejected", reason is not None)
    check("total_reason_is_size_or_cardinality",
          "too large" in (reason or "") or "cardinality" in (reason or ""),
          reason or "")


def test_normal_payload_accepted():
    result = _ingress().submit(_event({
        "temp": 21.5, "humidity": 60, "conditions": "clear",
        "location": {"lat": 51.5, "lon": -0.12},
    }))
    check("normal_payload_accepted",
          result.verdict == IngressVerdict.ACCEPTED, result.reason)


# ── 4. Rejection is total — nothing reaches the journal ─────────────

def test_rejected_payload_not_journalled():
    journal = EventJournal(os.path.join(_tmpdir, "reject.jsonl"))
    ingress = PerceptionIngress(journal=journal, principal=_principal())

    before = journal.count()
    result = ingress.submit(_event(_deep(MAX_PAYLOAD_DEPTH + 10)))
    after = journal.count()

    check("oversized_rejected", result.verdict == IngressVerdict.REJECTED_OVERSIZED)
    check("oversized_not_journalled", after == before)
    check("oversized_no_offset", result.offset is None)


def test_oversized_counted_in_stats():
    ingress = _ingress()
    ingress.submit(_event(_deep(MAX_PAYLOAD_DEPTH + 10)))
    check("oversized_counted",
          ingress.stats[IngressVerdict.REJECTED_OVERSIZED.value] == 1)


def test_oversized_does_not_poison_dedup():
    """A rejected event must not consume its dedup key.

    Otherwise one oversized event could permanently block the legitimate
    event that shares its hash.
    """
    ingress = _ingress()
    big = _event(_deep(MAX_PAYLOAD_DEPTH + 10))
    ingress.submit(big)

    ok = _event({"temp": 20})
    result = ingress.submit(ok)
    check("dedup_not_poisoned", result.verdict == IngressVerdict.ACCEPTED,
          result.reason)


# ── 5. Bounds helper directly ───────────────────────────────────────

def test_bounds_helper_accepts_normal():
    check("helper_accepts_empty", check_payload_bounds({}) is None)
    check("helper_accepts_flat", check_payload_bounds({"a": 1, "b": "x"}) is None)
    check("helper_accepts_nested",
          check_payload_bounds({"a": {"b": {"c": 1}}}) is None)


def test_bounds_helper_reports_reason():
    reason = check_payload_bounds(_deep(MAX_PAYLOAD_DEPTH + 5))
    check("helper_returns_string", isinstance(reason, str))
    check("helper_reason_actionable", "too deep" in (reason or ""))


# ── Runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_depth_within_limit_accepted()
    test_excessive_depth_rejected()
    test_deeply_nested_lists_rejected()
    test_depth_check_does_not_stack_overflow()
    test_high_cardinality_rejected()
    test_normal_cardinality_accepted()
    test_large_list_rejected()
    test_oversized_string_rejected()
    test_oversized_total_rejected()
    test_normal_payload_accepted()
    test_rejected_payload_not_journalled()
    test_oversized_counted_in_stats()
    test_oversized_does_not_poison_dedup()
    test_bounds_helper_accepts_normal()
    test_bounds_helper_reports_reason()

    print(f"\n{'='*60}")
    print(f"Payload Bounds Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
