"""Repo-wide suite floor tests — driven entirely by synthetic logs.

The gate this guards records what the repository's own test run produced:
4,208 passed, 0 failed, 0 errors. Nothing here reads that run. Every input
is a log string built in this file, for the reason set out as class B in
`kai-pm/TEST_WRITING_REVIEW.md`: a test guarded on a number the repository
owns breaks when the repository improves, and the sharpest defect in that
review was a test that required its own bug to persist.

The one assertion that touches the real floor file checks a *property* —
that failures and errors are recorded at zero and the pass floor is
positive — which can only be made stronger by the suite getting better.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_suite_floor as gate  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 11
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def log(text: str) -> Path:
    path = Path(tempfile.mkdtemp()) / "run.log"
    path.write_text(text, encoding="utf-8")
    return path


def run(text: str) -> int:
    argv = sys.argv
    sys.argv = ["check_suite_floor.py", "--from-log", str(log(text))]
    try:
        return gate.main()
    finally:
        sys.argv = argv


# ── Parsing ──────────────────────────────────────────────────────────

def test_a_clean_summary_parses() -> None:
    scenario("clean summary parses")
    counts = gate.parse("4208 passed, 6 skipped, 322 warnings in 152.32s")
    check("passed read", counts.get("passed") == 4208, str(counts))
    check("absent keys stay absent", "failed" not in counts, str(counts))


def test_failures_and_errors_parse() -> None:
    scenario("failures and errors parse")
    counts = gate.parse("4100 passed, 8 failed, 3 errors, 6 skipped in 1s")
    check("failed read", counts.get("failed") == 8, str(counts))
    check("errors read", counts.get("errors") == 3, str(counts))


def test_singular_error_parses_as_errors() -> None:
    """pytest writes "1 error", not "1 errors"."""
    scenario("singular error")
    counts = gate.parse("10 passed, 1 error in 1s")
    check("one error counted", counts.get("errors") == 1, str(counts))


def test_a_log_with_no_summary_is_not_zero_failures() -> None:
    """The exact shape of the outage: a run that reported nothing."""
    scenario("no summary is not a pass")
    check("returns None rather than {}", gate.parse("nothing here at all") is None)


# ── The ratchet ──────────────────────────────────────────────────────

def test_the_recorded_result_passes() -> None:
    scenario("recorded result passes")
    floor = json.loads(gate.FLOOR.read_text(encoding="utf-8"))
    text = f"{floor['min_passed']} passed, 6 skipped in 1s"
    check("at the floor, passes", run(text) == 0)


def test_a_new_failure_fails() -> None:
    scenario("new failure fails")
    floor = json.loads(gate.FLOOR.read_text(encoding="utf-8"))
    check("one failure is enough",
          run(f"{floor['min_passed']} passed, 1 failed in 1s") == 1)


def test_a_new_error_fails() -> None:
    scenario("new error fails")
    floor = json.loads(gate.FLOOR.read_text(encoding="utf-8"))
    check("one error is enough",
          run(f"{floor['min_passed']} passed, 1 error in 1s") == 1)


def test_deleting_tests_does_not_pass_the_gate() -> None:
    """Without a pass floor, `rm` would be a way to go green."""
    scenario("shrinking fails")
    floor = json.loads(gate.FLOOR.read_text(encoding="utf-8"))
    check("fewer passes fails",
          run(f"{floor['min_passed'] - 1} passed, 6 skipped in 1s") == 1)


def test_an_aborted_collection_fails() -> None:
    scenario("aborted collection fails")
    check("interrupted collection is not a pass",
          run("Interrupted: 6 errors during collection\n"
              "0 passed, 6 errors in 5s") == 1)


def test_a_missing_log_fails() -> None:
    scenario("missing log fails")
    argv = sys.argv
    sys.argv = ["check_suite_floor.py", "--from-log",
                str(Path(tempfile.mkdtemp()) / "absent.log")]
    try:
        status = gate.main()
    finally:
        sys.argv = argv
    check("a missing log is not a passing suite", status == 1, f"exit={status}")


def test_the_floor_records_zero_and_a_positive_pass_count() -> None:
    """One-way: these can only be made stronger by the suite improving."""
    scenario("floor is at zero")
    floor = json.loads(gate.FLOOR.read_text(encoding="utf-8"))
    check("no failures tolerated", floor["max_failed"] == 0, str(floor["max_failed"]))
    check("no errors tolerated", floor["max_errors"] == 0, str(floor["max_errors"]))
    check("a real pass floor is recorded", floor["min_passed"] > 4000,
          str(floor["min_passed"]))
    check("the history keeps the starting point",
          any(entry["passed"] == 0 for entry in floor.get("history", [])),
          "the run that executed no tests should stay on the record")


def run_all() -> None:
    test_a_clean_summary_parses()
    test_failures_and_errors_parse()
    test_singular_error_parses_as_errors()
    test_a_log_with_no_summary_is_not_zero_failures()
    test_the_recorded_result_passes()
    test_a_new_failure_fails()
    test_a_new_error_fails()
    test_deleting_tests_does_not_pass_the_gate()
    test_an_aborted_collection_fails()
    test_a_missing_log_fails()
    test_the_floor_records_zero_and_a_positive_pass_count()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Suite Floor Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
