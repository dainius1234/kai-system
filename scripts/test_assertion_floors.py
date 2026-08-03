"""Assertion-floor gate tests — the guard on the guard.

`scripts/security/check_assertion_floors.py` exists because five checks
in this programme quietly stopped checking. A gate written to catch that
is worthless if it can do the same thing, so this suite's job is to prove
the gate **fails** in every shape of erosion it claims to catch.

Two design rules follow from the pattern itself, and both are deliberate:

  1. **Synthetic inputs only.** Every case here is driven from a hand-
     written log string and a temporary floors file. Nothing reads the
     repository's real counts. That is the fix for the pattern's root
     cause — case 5 eroded precisely because its preconditions were read
     from live state that the system kept improving.
  2. **A meta-assertion on this file.** `EXPECTED_SCENARIOS` is checked
     against the number of cases actually executed. If someone deletes a
     case, or an exception swallows one, the count stops matching and
     this suite says so. A test file that silently runs fewer tests is
     the same defect one level up.

The invariant checks at the end are the second half of rule 2: they
assert *structural* facts about the gate's configuration — every sampled
suite is floored, every floor is a positive integer — rather than facts
about current counts, which change every time a test is added.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_assertion_floors as gate  # noqa: E402

passed = 0
failed = 0

# Meta-assertion (see module docstring). Raise this deliberately when a
# scenario is added; it is the thing that notices when one disappears.
EXPECTED_SCENARIOS = 20
executed: list[str] = []


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


def scenario(name: str) -> None:
    """Record that a case actually ran. Counted by the meta-assertion."""
    executed.append(name)


# ── Synthetic fixtures ───────────────────────────────────────────────

def log_of(**suites: int) -> str:
    """A synthetic test-run log. Never the repository's real output."""
    lines = ["make[1]: Entering directory '/synthetic'"]
    for name, count in suites.items():
        label = name.replace("_", " ")
        lines.append("=" * 60)
        lines.append(f"{label}: {count} passed, 0 failed")
        lines.append("EXIT GATE: PASS")
    lines.append("All Unified Hunter suites passed.")
    return "\n".join(lines) + "\n"


@contextlib.contextmanager
def floors_of(**suites: int):
    """Point the gate at a temporary floors file for the duration."""
    original = gate.FLOORS
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "floors.json"
        recorded = {n.replace("_", " "): c for n, c in suites.items()}
        path.write_text(json.dumps({
            "note": "synthetic",
            "floors": recorded,
            "total": sum(recorded.values()),
        }), encoding="utf-8")
        gate.FLOORS = path
        try:
            yield path
        finally:
            gate.FLOORS = original


def run_gate(log: str, *extra_args: str) -> tuple[int, str]:
    """Invoke the gate's CLI against a synthetic log; capture its report."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "run.log"
        log_path.write_text(log, encoding="utf-8")
        argv = ["check_assertion_floors.py", "--from-log", str(log_path),
                *extra_args]
        buffer = io.StringIO()
        original_argv = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(buffer):
                code = gate.main()
        finally:
            sys.argv = original_argv
        return code, buffer.getvalue()


# ── Parsing ──────────────────────────────────────────────────────────

def test_counts_are_parsed_from_a_run():
    scenario("parse")
    counts = gate.parse_counts(log_of(Alpha_Tests=12, Beta_Tests=7))
    check("both suites parsed", counts == {"Alpha Tests": 12, "Beta Tests": 7},
          str(counts))


def test_surrounding_noise_is_not_mistaken_for_a_suite():
    scenario("parse-noise")
    noise = ("Ran 26 tests in 0.9s\n"
             "note: 3 passed, 1 failed on the previous branch\n"
             "  Indented Thing: 5 passed, 0 failed\n")
    counts = gate.parse_counts(noise)
    check("prose is not parsed as a suite", "note" not in counts, str(counts))
    check("indented suite lines still count",
          counts.get("Indented Thing") == 5, str(counts))


# ── Erosion over time: a suite that shrinks ──────────────────────────

def test_a_shrinking_suite_fails():
    scenario("shrink")
    with floors_of(Hygiene_Gate_Tests=39):
        code, out = run_gate(log_of(Hygiene_Gate_Tests=31))
    check("shrinking suite fails the gate", code == 1)
    check("the report says coverage fell", "COVERAGE FELL" in out, out)
    check("the shrinking suite is named", "Hygiene Gate Tests" in out, out)
    check("the delta is shown", "39 → 31" in out, out)


def test_a_growing_suite_passes():
    scenario("grow")
    with floors_of(Hygiene_Gate_Tests=39):
        code, out = run_gate(log_of(Hygiene_Gate_Tests=44))
    check("a suite exercising more passes", code == 0, out)
    check("the pass is stated", "met its floor" in out, out)


def test_meeting_the_floor_exactly_passes():
    scenario("exact")
    with floors_of(Hygiene_Gate_Tests=39):
        code, _ = run_gate(log_of(Hygiene_Gate_Tests=39))
    check("floor is a minimum, not a target", code == 0)


# ── Erosion by disappearance ─────────────────────────────────────────

def test_a_vanished_suite_fails():
    """Case 4's shape: a suite that reports nothing looks like silence."""
    scenario("vanish")
    with floors_of(Alpha_Tests=10, Beta_Tests=20):
        code, out = run_gate(log_of(Alpha_Tests=10))
    check("a vanished suite fails the gate", code == 1)
    check("the report says the suite vanished", "SUITES VANISHED" in out, out)
    check("the vanished suite is named", "Beta Tests" in out, out)


def test_an_empty_run_fails():
    scenario("empty")
    with floors_of(Alpha_Tests=10):
        code, out = run_gate("make: nothing to be done\n")
    check("an empty run fails the gate", code == 1)
    check("silence is not treated as success",
          "not a run that passed" in out, out)


def test_an_unfloored_suite_fails():
    scenario("unfloored")
    with floors_of(Alpha_Tests=10):
        code, out = run_gate(log_of(Alpha_Tests=10, Newcomer_Tests=5))
    check("an unwatched suite fails the gate", code == 1)
    check("the report names it unfloored", "UNFLOORED SUITES" in out, out)
    check("the new suite is named", "Newcomer Tests" in out, out)


# ── Erosion at a point in time: a context-dependent count ────────────

class _FakeRun:
    """Stands in for `subprocess.run`, returning canned suite output."""

    def __init__(self, standalone: dict[str, int | None]):
        self.standalone = standalone
        self.calls: list[str] = []

    def __call__(self, argv, **_kwargs):
        target = argv[-1]
        self.calls.append(target)
        label = next((k for k, v in gate.SUITE_TARGETS.items() if v == target),
                     None)
        count = self.standalone.get(label)
        text = "" if count is None else f"{label}: {count} passed, 0 failed\n"

        class _Result:
            stdout = text
            stderr = ""
            returncode = 0

        return _Result()


@contextlib.contextmanager
def sampled(targets: dict[str, str], standalone: dict[str, int | None]):
    original_targets, original_run = gate.SUITE_TARGETS, gate.subprocess.run
    gate.SUITE_TARGETS = targets
    gate.subprocess.run = _FakeRun(standalone)
    try:
        yield
    finally:
        gate.SUITE_TARGETS = original_targets
        gate.subprocess.run = original_run


def test_a_count_that_changes_with_context_is_reported():
    """Case 5's actual signature: 34 alone, 31 under the aggregate run."""
    scenario("drift")
    with sampled({"Alpha Tests": "test-alpha"}, {"Alpha Tests": 34}):
        drifted = gate.check_determinism({"Alpha Tests": 31})
    check("context-dependent count is caught", len(drifted) == 1, str(drifted))
    check("both counts are shown",
          drifted and "34 alone vs 31" in drifted[0], str(drifted))


def test_a_stable_count_is_not_reported():
    scenario("stable")
    with sampled({"Alpha Tests": "test-alpha"}, {"Alpha Tests": 31}):
        drifted = gate.check_determinism({"Alpha Tests": 31})
    check("a context-free count is clean", drifted == [], str(drifted))


def test_a_sample_that_reports_nothing_alone_is_caught():
    scenario("silent-alone")
    with sampled({"Alpha Tests": "test-alpha"}, {"Alpha Tests": None}):
        drifted = gate.check_determinism({"Alpha Tests": 31})
    check("silence when run alone is caught", len(drifted) == 1, str(drifted))
    check("the reason is stated",
          drifted and "no count when run alone" in drifted[0], str(drifted))


def test_a_renamed_sample_fails_rather_than_being_skipped():
    """The gate's own self-consuming-guard case.

    `if label not in aggregate: continue` would make this file sample
    fewer suites every time one was renamed — silently, and with a green
    result. That is case 5 rewritten inside the fix for case 5.
    """
    scenario("renamed-sample")
    with sampled({"Gone Tests": "test-gone"}, {}):
        drifted = gate.check_determinism({"Alpha Tests": 31})
    check("a sample naming a missing suite fails", len(drifted) == 1,
          str(drifted))
    check("the reason points at a rename",
          drifted and "absent from the aggregate" in drifted[0], str(drifted))


# ── The ratchet only turns one way ───────────────────────────────────

def test_lowering_a_floor_is_refused():
    scenario("refuse-lower")
    with floors_of(Alpha_Tests=50) as path:
        code, out = run_gate(log_of(Alpha_Tests=40), "--update-floors")
        after = json.loads(path.read_text(encoding="utf-8"))["floors"]
    check("lowering is refused", code == 1)
    check("the refusal is explicit", "REFUSED" in out, out)
    check("the floor was not written", after["Alpha Tests"] == 50, str(after))


def test_raising_a_floor_is_recorded():
    scenario("raise")
    with floors_of(Alpha_Tests=50) as path:
        code, _ = run_gate(log_of(Alpha_Tests=60), "--update-floors")
        after = json.loads(path.read_text(encoding="utf-8"))["floors"]
    check("raising succeeds", code == 0)
    check("the higher floor is recorded", after["Alpha Tests"] == 60,
          str(after))


def test_updating_never_drops_a_suite_absent_from_this_run():
    """A partial run must not quietly delete floors it did not observe."""
    scenario("update-preserves")
    with floors_of(Alpha_Tests=50, Beta_Tests=20) as path:
        run_gate(log_of(Alpha_Tests=55), "--update-floors")
        after = json.loads(path.read_text(encoding="utf-8"))["floors"]
    check("unobserved floors survive an update", after.get("Beta Tests") == 20,
          str(after))


def test_a_failing_aggregate_run_is_reported_as_itself():
    scenario("red-run")
    original = gate.run_suites
    gate.run_suites = lambda: (log_of(Alpha_Tests=10), 2)
    try:
        with floors_of(Alpha_Tests=10, Beta_Tests=20):
            argv, sys.argv = sys.argv, ["check_assertion_floors.py"]
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    code = gate.main()
            finally:
                sys.argv = argv
    finally:
        gate.run_suites = original
    out = buffer.getvalue()
    check("a red test run fails the gate", code == 1)
    check("it is not disguised as erosion", "SUITES VANISHED" not in out, out)
    check("the real cause is named", "aggregate test run failed" in out, out)


# ── Invariants of the real configuration ─────────────────────────────
#
# Structural, not numeric: they stay true as counts change, which is what
# stops them eroding the way the checks they guard did.

def test_every_sampled_suite_is_floored():
    scenario("invariant-sampled")
    recorded = json.loads(gate.FLOORS.read_text(encoding="utf-8"))["floors"]
    orphans = [label for label in gate.SUITE_TARGETS if label not in recorded]
    check("every determinism sample has a floor", not orphans, str(orphans))


def test_every_sampled_target_exists_in_the_makefile():
    scenario("invariant-targets")
    makefile = (Path(__file__).resolve().parent.parent / "Makefile").read_text(
        encoding="utf-8")
    missing = [t for t in gate.SUITE_TARGETS.values()
               if f"\n{t}:" not in makefile]
    check("every sampled target is a real make target", not missing,
          str(missing))


def test_every_floor_is_a_positive_integer():
    scenario("invariant-positive")
    recorded = json.loads(gate.FLOORS.read_text(encoding="utf-8"))["floors"]
    bad = {n: v for n, v in recorded.items()
           if not isinstance(v, int) or v <= 0}
    check("no floor is zero, negative or non-integer", not bad, str(bad))


def test_the_recorded_total_matches_the_recorded_floors():
    scenario("invariant-total")
    data = json.loads(gate.FLOORS.read_text(encoding="utf-8"))
    check("the stated total is the sum of the floors",
          data.get("total") == sum(data["floors"].values()),
          f"{data.get('total')} vs {sum(data['floors'].values())}")


def run() -> None:
    test_counts_are_parsed_from_a_run()
    test_surrounding_noise_is_not_mistaken_for_a_suite()
    test_a_shrinking_suite_fails()
    test_a_growing_suite_passes()
    test_meeting_the_floor_exactly_passes()
    test_a_vanished_suite_fails()
    test_an_empty_run_fails()
    test_an_unfloored_suite_fails()
    test_a_count_that_changes_with_context_is_reported()
    test_a_stable_count_is_not_reported()
    test_a_sample_that_reports_nothing_alone_is_caught()
    test_a_renamed_sample_fails_rather_than_being_skipped()
    test_lowering_a_floor_is_refused()
    test_raising_a_floor_is_recorded()
    test_updating_never_drops_a_suite_absent_from_this_run()
    test_a_failing_aggregate_run_is_reported_as_itself()
    test_every_sampled_suite_is_floored()
    test_every_sampled_target_exists_in_the_makefile()
    test_every_floor_is_a_positive_integer()
    test_the_recorded_total_matches_the_recorded_floors()

    # The meta-assertion. Two scenarios cover parsing, so the count is
    # not the number of test functions — it is the number of cases that
    # reported having run.
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Assertion Floor Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
