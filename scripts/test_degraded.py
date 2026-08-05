"""Degraded-state envelope tests — Wave 1 Track D.

Covers `common/degraded.py` and the dashboard behaviour built on it:
`KAI-DASH-016`, `061`, `063`–`067`, `080`, `082`.

The whole point of this mechanism is that an outage cannot be mistaken
for an answer, so the tests that matter are the ones asserting the two
signals stay distinguishable: HTTP status **and** an in-body marker. A
degraded envelope that answered 200, or a healthy read that carried
`degraded: true`, would each defeat it on their own.
"""
from __future__ import annotations

import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.degraded import (
    DEGRADED_STATUS_CODE,
    STATUS_UNAVAILABLE,
    degradation_report,
    degraded_body,
    degraded_response,
    is_degraded,
    record_degradation,
    reset_degradations,
    unavailable_metric,
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


def _payload(response) -> dict:
    return json.loads(bytes(response.body).decode())


# ── The envelope ─────────────────────────────────────────────────────

def test_body_carries_every_marker():
    body = degraded_body("memu", "connection refused")
    for key in ("status", "degraded", "source", "reason", "observed_at"):
        check(f"degraded body carries {key}", key in body, str(body))
    check("status is unavailable", body["status"] == STATUS_UNAVAILABLE)
    check("degraded is exactly True", body["degraded"] is True)
    check("source names the failing backend", body["source"] == "memu")


def test_shape_is_preserved_so_adopting_this_breaks_nothing():
    body = degraded_body("memu", "boom", {"nudges": [], "count": 0})
    check("caller's keys survive",
          body["nudges"] == [] and body["count"] == 0, str(body))
    check("markers are added alongside", body["degraded"] is True)


def test_markers_win_over_a_conflicting_shape():
    """A backend must not be able to declare itself healthy."""
    body = degraded_body("memu", "boom", {"status": "ok", "degraded": False})
    check("a backend cannot override status",
          body["status"] == STATUS_UNAVAILABLE, str(body))
    check("a backend cannot override the degraded flag",
          body["degraded"] is True, str(body))


def test_shape_is_not_mutated():
    shape = {"nudges": []}
    degraded_body("memu", "boom", shape)
    check("the caller's dict is left alone", shape == {"nudges": []}, str(shape))


def test_reason_is_bounded():
    body = degraded_body("memu", "x" * 5000)
    check("reason cannot blow up the response",
          len(body["reason"]) <= 300, str(len(body["reason"])))


def test_timestamp_is_timezone_aware():
    """A naive stamp is its own finding (083/084)."""
    body = degraded_body("memu", "boom")
    stamp = body["observed_at"]
    check("observed_at is timezone-aware",
          stamp.endswith("+00:00") or stamp.endswith("Z"), stamp)


# ── The response ─────────────────────────────────────────────────────

def test_response_answers_503_not_200():
    response = degraded_response("memu", "boom", {"nudges": []})
    check("a degraded read does not answer 200",
          response.status_code == DEGRADED_STATUS_CODE,
          str(response.status_code))


def test_response_body_is_the_envelope():
    response = degraded_response("verifier", "timeout", {"corrections": []})
    payload = _payload(response)
    check("body keeps the expected shape", payload["corrections"] == [])
    check("body carries the marker", payload["degraded"] is True)
    check("body names the source", payload["source"] == "verifier")


def test_both_channels_are_independent():
    """Either signal alone must be enough to detect the outage."""
    response = degraded_response("memu", "boom", {"nudges": []})
    payload = _payload(response)
    check("status alone reveals it", response.status_code >= 500)
    check("body alone reveals it", is_degraded(payload))


# ── Distinguishing outage from emptiness ─────────────────────────────

def test_empty_is_not_degraded():
    """The bug this exists to prevent: empty data reading as an outage."""
    check("a genuinely empty result is not degraded",
          not is_degraded({"nudges": []}))
    check("a healthy status is not degraded",
          not is_degraded({"status": "ok", "records": []}))


def test_degraded_is_detected_either_way():
    check("detected via the flag", is_degraded({"degraded": True}))
    check("detected via the status", is_degraded({"status": STATUS_UNAVAILABLE}))


def test_non_mappings_are_not_degraded():
    for value in (None, [], "unavailable", 0, object()):
        check(f"{type(value).__name__} is not treated as degraded",
              not is_degraded(value))


# ── Declining to measure ─────────────────────────────────────────────

def test_unavailable_metric_is_explicitly_absent():
    metric = unavailable_metric("recent_approved_decisions", "no credential")
    check("the metric names itself",
          metric["metric"] == "recent_approved_decisions")
    check("it is marked unavailable", metric["available"] is False)
    check("it says why", "credential" in metric["reason"])
    check("it carries no value that could be mistaken for a measurement",
          "value" not in metric and "count" not in metric, str(metric))


def test_unavailable_metric_reason_is_bounded():
    metric = unavailable_metric("m", "y" * 5000)
    check("metric reason is bounded", len(metric["reason"]) <= 300)


# ── There is deliberately no 200 shortcut ────────────────────────────

def test_no_success_shaped_helper_exists():
    """A `degraded_ok()` returning 200 would undo the whole mechanism."""
    import common.degraded as module
    offenders = [n for n in dir(module)
                 if "degraded" in n.lower() and n.endswith("_ok")]
    check("no helper returns a degraded 200", not offenders, str(offenders))


# ── The degradation registry (H-6) ───────────────────────────────────
#
# This is the half of the mechanism that covers failures which are
# deliberately *survived* rather than reported — memu-core's Redis
# fallback and its 33 siblings. The envelope above answers "the caller
# asked and I could not tell them"; this answers "nobody asked, and it
# has been broken since Tuesday".


def test_a_survived_failure_is_recorded():
    reset_degradations()
    record_degradation("redis", "p20_put_value", RuntimeError("connection refused"))
    report = degradation_report()
    check("one entry recorded", len(report) == 1, str(report))
    check("source is named", report[0]["source"] == "redis")
    check("operation is named", report[0]["operation"] == "p20_put_value")
    check("reason names the exception type",
          "RuntimeError" in report[0]["reason"], report[0]["reason"])
    check("reason names the message",
          "connection refused" in report[0]["reason"], report[0]["reason"])


def test_repeats_aggregate_rather_than_accumulate():
    """The Q2 requirement: ten seconds must be distinguishable from ten
    days, and a hot loop must not produce a hundred thousand entries."""
    reset_degradations()
    for _ in range(500):
        record_degradation("redis", "p20_put_value", RuntimeError("nope"))
    report = degradation_report()
    check("500 failures are one entry", len(report) == 1, str(len(report)))
    check("the count is kept", report[0]["count"] == 500, str(report[0]["count"]))


def test_distinct_operations_stay_distinct():
    reset_degradations()
    record_degradation("redis", "read", RuntimeError("a"))
    record_degradation("redis", "write", RuntimeError("b"))
    record_degradation("postgres", "read", RuntimeError("c"))
    report = degradation_report()
    check("three keys", len(report) == 3, str(len(report)))
    keys = {(r["source"], r["operation"]) for r in report}
    check("keyed by source and operation",
          keys == {("redis", "read"), ("redis", "write"), ("postgres", "read")},
          str(keys))


def test_duration_is_reported_not_just_a_count():
    """A count alone cannot separate a burst from a chronic outage."""
    reset_degradations()
    record_degradation("redis", "read", RuntimeError("x"))
    report = degradation_report()[0]
    check("failing_for_seconds present", "failing_for_seconds" in report)
    check("first_seen present", "first_seen" in report)
    check("last_seen present", "last_seen" in report)
    check("stale_seconds present", "stale_seconds" in report)
    check("timestamps are timezone-aware",
          report["first_seen"].endswith("+00:00"), report["first_seen"])


def test_a_string_reason_is_accepted():
    """Not every survived failure has an exception object to hand."""
    reset_degradations()
    record_degradation("gpu", "probe", "no device")
    check("string reason kept", degradation_report()[0]["reason"] == "no device")


def test_the_registry_is_bounded():
    """An unbounded registry would be a leak in the leak detector."""
    reset_degradations()
    import common.degraded as module
    for i in range(module._MAX_TRACKED + 50):
        record_degradation("src", f"op{i}", "x")
    check("registry is capped",
          len(degradation_report()) <= module._MAX_TRACKED,
          str(len(degradation_report())))


def test_recording_never_raises():
    """A failure in the failure recorder must not become the outage."""
    reset_degradations()

    class Hostile:
        def __str__(self):
            raise ValueError("I refuse to be described")

    ok = True
    try:
        record_degradation("weird", "op", Hostile())  # type: ignore[arg-type]
    except Exception:
        ok = False
    check("hostile input does not propagate", ok)


def test_logging_is_rate_limited():
    """These sit in hot loops; an unthrottled warning is its own outage."""
    reset_degradations()
    import logging as _logging

    seen = []

    class Capture(_logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log = _logging.getLogger("test.degraded.ratelimit")
    log.addHandler(Capture())
    log.setLevel(_logging.WARNING)
    for _ in range(200):
        record_degradation("redis", "hot_loop", RuntimeError("x"), logger=log)
    check("logs once, not two hundred times", len(seen) == 1, str(len(seen)))
    check("the one line carries the count",
          seen and "count=1" in seen[0], str(seen[:1]))


def test_the_log_line_is_machine_parseable():
    """Aggregatable means a field=value shape, not prose."""
    reset_degradations()
    import logging as _logging

    seen = []

    class Capture(_logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    log = _logging.getLogger("test.degraded.parse")
    log.addHandler(Capture())
    log.setLevel(_logging.WARNING)
    record_degradation("redis", "p21_hash_put", RuntimeError("boom"), logger=log)
    line = seen[0] if seen else ""
    for field in ("source=", "operation=", "count=", "failing_for_seconds=", "reason="):
        check(f"log line carries {field}", field in line, line)


def test_concurrent_records_do_not_lose_counts():
    """`memu-core` is threaded and these sit in per-request paths.

    A read-modify-write on a shared dict without a lock loses increments
    under contention, and the failure mode would be a count that
    understates a chronic outage — the exact thing the count exists to
    make visible. The lock is claimed in the module docstring; this is
    the claim being checked rather than asserted.
    """
    import threading
    reset_degradations()
    threads = 16
    each = 500
    log = logging.getLogger("test.degraded.concurrency")
    log.disabled = True

    def worker(i: int) -> None:
        for _ in range(each):
            record_degradation("redis", f"op{i % 4}", RuntimeError("x"), logger=log)

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()

    report = degradation_report()
    check("four distinct operations, not one per thread",
          len(report) == 4, str(len(report)))
    check("no increment was lost under contention",
          sum(e["count"] for e in report) == threads * each,
          str(sum(e["count"] for e in report)))
    log.disabled = False
    reset_degradations()


def run() -> None:
    test_body_carries_every_marker()
    test_shape_is_preserved_so_adopting_this_breaks_nothing()
    test_markers_win_over_a_conflicting_shape()
    test_shape_is_not_mutated()
    test_reason_is_bounded()
    test_timestamp_is_timezone_aware()
    test_response_answers_503_not_200()
    test_response_body_is_the_envelope()
    test_both_channels_are_independent()
    test_empty_is_not_degraded()
    test_degraded_is_detected_either_way()
    test_non_mappings_are_not_degraded()
    test_unavailable_metric_is_explicitly_absent()
    test_unavailable_metric_reason_is_bounded()
    test_no_success_shaped_helper_exists()
    test_a_survived_failure_is_recorded()
    test_repeats_aggregate_rather_than_accumulate()
    test_distinct_operations_stay_distinct()
    test_duration_is_reported_not_just_a_count()
    test_a_string_reason_is_accepted()
    test_the_registry_is_bounded()
    test_recording_never_raises()
    test_logging_is_rate_limited()
    test_the_log_line_is_machine_parseable()
    test_concurrent_records_do_not_lose_counts()
    reset_degradations()


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Degraded State Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
