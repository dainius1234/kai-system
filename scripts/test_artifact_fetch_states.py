#!/usr/bin/env python3
"""Calibration for the artifact-fetch state classifier.

One property, and it is the operator's: **the five failure causes must
stay distinguishable.** A classifier that maps them all to one string is
the same defect as one abort message covering three selftest states — the
report names a condition that may not have occurred, and the reader
cannot tell which did.

So the assertions here are mostly *inequalities*: this input must NOT
produce that state. Each state also gets the case that produces it, and
the ordering is pinned, because "the run is still going" and "the run
finished and produced nothing" are the same absence seen at different
times and only the order of the questions tells them apart.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import classify_artifact_fetch as f  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 3
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


def state(**over) -> str:
    args = dict(http_status=200, run_status="completed",
                artifact_names=["memu-graph-llm-contract"],
                artifact_name="memu-graph-llm-contract",
                artifact_expired=False, expected_file_present=False)
    args.update(over)
    return f.classify_fetch(**args)[0]


def test_each_cause_has_its_own_state() -> None:
    scenario("one state per cause")
    check("the file being present is ARTIFACT_PRESENT",
          state(expected_file_present=True) == f.ARTIFACT_PRESENT)
    check("an unmade request is NETWORK_FAILURE",
          state(http_status=None) == f.NETWORK_FAILURE)
    check("401 is ACCESS_DENIED", state(http_status=401) == f.ACCESS_DENIED)
    check("403 is ACCESS_DENIED", state(http_status=403) == f.ACCESS_DENIED)
    check("a 5xx is NETWORK_FAILURE", state(http_status=502) == f.NETWORK_FAILURE)
    check("a run still going is SUBJECT_RUN_INCOMPLETE",
          state(run_status="in_progress") == f.SUBJECT_RUN_INCOMPLETE)
    check("a queued run is SUBJECT_RUN_INCOMPLETE",
          state(run_status="queued") == f.SUBJECT_RUN_INCOMPLETE)
    check("completed with no such artifact is ARTIFACT_ABSENT",
          state(artifact_names=["something-else"]) == f.ARTIFACT_ABSENT)
    check("an expired artifact is ARTIFACT_EXPIRED",
          state(artifact_expired=True) == f.ARTIFACT_EXPIRED)
    check("present-but-wrong-contents is ARTIFACT_MALFORMED",
          state() == f.ARTIFACT_MALFORMED)
    check("an unlistable artifact set is NETWORK_FAILURE, not ABSENT",
          state(artifact_names=None) == f.NETWORK_FAILURE)
    check("every state has a distinct exit code",
          len(set(f.EXIT.values())) == len(f.EXIT), str(f.EXIT))
    check("only ARTIFACT_PRESENT exits 0",
          [k for k, v in f.EXIT.items() if v == 0] == [f.ARTIFACT_PRESENT])


def test_the_states_do_not_collapse_into_each_other() -> None:
    scenario("no collapsing")
    # THE distinction the operator named. Same observable absence, two
    # very different meanings, and only the order of questions separates
    # them: one is "wait", the other is a finding about the subject run.
    waiting = state(run_status="in_progress", artifact_names=[])
    finished = state(run_status="completed", artifact_names=[])
    check("a run in progress is not ARTIFACT_ABSENT",
          waiting != f.ARTIFACT_ABSENT, waiting)
    check("a completed run with nothing IS ARTIFACT_ABSENT",
          finished == f.ARTIFACT_ABSENT, finished)
    check("and the two differ", waiting != finished)

    # A permissions failure here must never be reported as the subject
    # having produced nothing there.
    denied = state(http_status=403, run_status="completed", artifact_names=[])
    check("access denied is not ARTIFACT_ABSENT", denied != f.ARTIFACT_ABSENT)
    check("access denied is not NETWORK_FAILURE", denied != f.NETWORK_FAILURE)
    check("and it is ACCESS_DENIED", denied == f.ACCESS_DENIED)

    # Expired is not absent: the run DID its job.
    check("expired is not absent",
          state(artifact_expired=True) != f.ARTIFACT_ABSENT)
    # Malformed is not absent either.
    check("malformed is not absent", state() != f.ARTIFACT_ABSENT)
    check("malformed is not expired", state() != f.ARTIFACT_EXPIRED)

    # Transport is asked BEFORE the subject's state, because a run status
    # we could not fetch cannot be trusted to say "completed".
    check("a network failure outranks a stale run_status",
          state(http_status=None, run_status="completed") == f.NETWORK_FAILURE)

    # And every distinct cause really does map somewhere distinct.
    seen = {state(expected_file_present=True), state(http_status=None),
            state(http_status=403), state(run_status="in_progress"),
            state(artifact_names=[]), state(artifact_expired=True), state()}
    check("seven inputs produce seven distinct states", len(seen) == 7,
          str(sorted(seen)))


def test_every_state_says_what_it_means() -> None:
    scenario("each state explains itself")
    for name in f.EXIT:
        check(f"{name} has a meaning recorded", name in f.MEANING, name)
        check(f"{name}'s meaning is a sentence, not a label",
              len(f.MEANING.get(name, "")) > 40, name)
    _, why = f.classify_fetch(
        http_status=200, run_status="completed", artifact_names=[],
        artifact_name="x", artifact_expired=None, expected_file_present=False)
    check("ARTIFACT_ABSENT says it may be a finding about the subject",
          "fact about the subject run" in why, why)
    _, why = f.classify_fetch(
        http_status=403, run_status=None, artifact_names=None,
        artifact_name="x", artifact_expired=None, expected_file_present=False)
    check("ACCESS_DENIED says it is about THIS job, not the subject",
          "not about the subject run" in why, why)


def run_all() -> None:
    test_each_cause_has_its_own_state()
    test_the_states_do_not_collapse_into_each_other()
    test_every_state_says_what_it_means()
    print(f"  inspected: {len(f.EXIT)} fetch state(s) discriminated")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Artifact Fetch State Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
