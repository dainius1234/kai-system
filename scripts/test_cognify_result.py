#!/usr/bin/env python3
"""Calibration for KAI-GATE-050's success predicate.

The load-bearing assertions:

1. **Both directions, on the shapes cognee actually returns.** Terminal
   success must pass; terminal failure must not. A predicate that only
   ever refuses is as useless as one that only ever accepts — and the
   refusing kind would break idempotent re-ingest.

2. **The predicate is a CLASS, not the observed instance.** A status
   nobody has seen yet — a future cognee state, a run left mid-flight —
   must refuse success without this file having been taught its name.
   Hard-coding `PipelineRunFailedError` would have fixed run 9 and stayed
   blind to the next failure mode.

3. **Fail closed on a shape it cannot read (I-1).** If cognee changes its
   return type, that must become a loud failure rather than a silent
   success — which is precisely how the original defect worked.

4. **`AlreadyCompleted` is a success.** Re-ingesting unchanged data is a
   legitimate no-op; calling it a failure would break idempotent callers
   in order to fix a different bug.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "memu-graph"))

import cognify_result as c  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 7
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


class RunInfo:
    """The shape cognee returns: a pydantic model carrying `.status`."""

    def __init__(self, status: str) -> None:
        self.status = status


DATASET = "084851e5-f7c8-5162-b6c8-1b0c10687939"
OTHER = "7464f1cd-2fc7-5a3e-94c6-6937a1b97020"


def test_terminal_success_passes() -> None:
    """The known-positive. Without it the predicate could simply always
    refuse and every test above would still be green."""
    scenario("terminal success passes")
    ok, states, why = c.evaluate({DATASET: RunInfo("PipelineRunCompleted")})
    check("a completed pipeline is success", ok, why)
    check("and the status is reported", states[DATASET] == "PipelineRunCompleted",
          str(states))
    check("with no reason-not-to", why == "", why)
    ok, _, why = c.evaluate({DATASET: RunInfo("PipelineRunCompleted"),
                             OTHER: RunInfo("PipelineRunCompleted")})
    check("two completed pipelines are success", ok, why)


def test_the_reproduced_failure_is_refused() -> None:
    """Run 9 and run 11's actual shape: add completed, cognify errored."""
    scenario("reproduced failure refused")
    ok, states, why = c.evaluate({OTHER: RunInfo("PipelineRunCompleted"),
                                  DATASET: RunInfo("PipelineRunErrored")})
    check("a mixed result is NOT success", not ok, why)
    check("it names the failing dataset", DATASET in why, why)
    check("and its status", "PipelineRunErrored" in why, why)
    check("and counts the population", "1 of 2" in why, why)
    check("while still reporting the good one",
          states[OTHER] == "PipelineRunCompleted", str(states))


def test_a_status_this_file_has_never_seen_refuses_success() -> None:
    """THE class assertion. The predicate asks 'is this terminal
    success?', never 'is this the error we saw'. A status invented here
    and nowhere in the source must still refuse."""
    scenario("unknown status refuses")
    for status in ("PipelineRunStarted", "PipelineRunYield",
                   "PipelineRunCancelled", "SomeFutureCogneeStatus"):
        ok, _, why = c.evaluate({DATASET: RunInfo(status)})
        check(f"{status!r} is not terminal success", not ok, why)
    check("and the failure text never names the observed exception",
          "PipelineRunFailedError" not in
          c.evaluate({DATASET: RunInfo("PipelineRunErrored")})[2],
          c.evaluate({DATASET: RunInfo("PipelineRunErrored")})[2])


def test_already_completed_is_a_success() -> None:
    """Idempotent re-ingest must not be broken to fix a different bug."""
    scenario("already-completed is success")
    ok, _, why = c.evaluate({DATASET: RunInfo("PipelineRunAlreadyCompleted")})
    check("AlreadyCompleted is success", ok, why)


def test_unreadable_shapes_fail_closed() -> None:
    """I-1. A result this code cannot understand is a failure to answer,
    not a pass — the exact mechanism of the original defect."""
    scenario("unreadable shapes fail closed")
    ok, _, why = c.evaluate(None)
    check("None is not success", not ok, why)
    check("and says so", "None" in why, why)
    ok, _, why = c.evaluate({})
    check("an empty dict is not success", not ok, why)
    check("and says no pipeline ran", "no pipeline ran" in why, why)
    ok, _, why = c.evaluate({DATASET: object()})
    check("an entry with no status is not success", not ok, why)
    check("and says the shape is not understood",
          "not understood" in why, why)
    ok, _, why = c.evaluate("a string cognee never returns")
    check("an unrecognised type is not success", not ok, why)


def test_a_bare_run_info_is_handled() -> None:
    """run_pipeline_blocking returns the run_info ITSELF when it carries
    no dataset_id — a real branch of the contract, not a hypothetical."""
    scenario("bare run_info handled")
    ok, states, why = c.evaluate(RunInfo("PipelineRunCompleted"))
    check("a bare completed run_info is success", ok, why)
    check("and is keyed as having no dataset id",
          "<no-dataset-id>" in states, str(states))
    ok, _, why = c.evaluate(RunInfo("PipelineRunErrored"))
    check("a bare errored run_info is NOT success", not ok, why)


def test_dict_shaped_run_info_is_read_too() -> None:
    """Defensive: if a run_info arrives as a plain dict rather than a
    model, its status must still be read rather than silently missed."""
    scenario("dict-shaped run_info")
    ok, _, why = c.evaluate({DATASET: {"status": "PipelineRunCompleted"}})
    check("a dict run_info reads as success", ok, why)
    ok, _, why = c.evaluate({DATASET: {"status": "PipelineRunErrored"}})
    check("and an errored one does not", not ok, why)


def run_all() -> None:
    test_terminal_success_passes()
    test_the_reproduced_failure_is_refused()
    test_a_status_this_file_has_never_seen_refuses_success()
    test_already_completed_is_a_success()
    test_unreadable_shapes_fail_closed()
    test_a_bare_run_info_is_handled()
    test_dict_shaped_run_info_is_read_too()

    # I-2: the population this predicate accepts, stated out loud.
    print(f"  inspected: {len(c.TERMINAL_SUCCESS)} terminal-success "
          f"status(es) accepted: {sorted(c.TERMINAL_SUCCESS)}")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Cognify Result Predicate Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
