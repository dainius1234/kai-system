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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common.degraded import (
    DEGRADED_STATUS_CODE,
    STATUS_UNAVAILABLE,
    degraded_body,
    degraded_response,
    is_degraded,
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


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Degraded State Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
