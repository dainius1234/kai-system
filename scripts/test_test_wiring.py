"""Dead-test detector tests — including its own three wrong answers.

The operator asked for a review of how this programme writes tests, after
a great deal of time went into fixing tests that gave false readings. The
detector this suite guards is the review's main output, and building it
reproduced the exact failure being reviewed: **three confident wrong
answers before the right one.**

  attempt 1  "any `test_` never called is dead"          1,555
  attempt 2  "...unless the file has unittest.main()"    1,813
  attempt 3  "look inside the file's run() function"        54
  attempt 4  ask the Makefile how the file is invoked       10

The first three read a *proxy* — the file's contents — for the thing that
actually decides the answer: the command that runs it. `python -m pytest
x.py` collects every test; `python x.py` runs only what the file calls.

So the detector is calibrated. `test_calibration_catches_a_broken_detector`
asserts that a wrong detector is *refused* rather than believed, which is
the property the first three attempts lacked.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_test_wiring as wiring  # noqa: E402

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


def suite(body: str) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "test_synthetic.py"
    tmp.write_text(body)
    return tmp


# ── The detector itself ──────────────────────────────────────────────

def test_an_undispatched_test_is_found():
    scenario("orphan")
    found = wiring.orphans(suite(
        "def test_a():\n    pass\n\ndef test_b():\n    pass\n\n"
        "def run():\n    test_a()\n"))
    check("the undispatched test is found", found == ["test_b"], str(found))


def test_a_fully_dispatched_suite_is_clean():
    scenario("clean")
    found = wiring.orphans(suite(
        "def test_a():\n    pass\n\ndef test_b():\n    pass\n\n"
        "def run():\n    test_a()\n    test_b()\n"))
    check("a dispatched suite reports nothing", found == [], str(found))


def test_a_helper_named_run_does_not_confuse_it():
    """Attempt 3 took `dispatch[0]` and matched a helper called `run(...)`
    instead of `run_all()`, reporting 54 phantom orphans."""
    scenario("helper-named-run")
    found = wiring.orphans(suite(
        "def run(x):\n    return x\n\n"
        "def test_a():\n    pass\n\ndef test_b():\n    pass\n\n"
        "def run_all():\n    test_a()\n    test_b()\n"))
    check("a helper named run() is not mistaken for the dispatcher",
          found == [], str(found))


def test_unittest_collection_is_not_reported():
    """Attempts 1 and 2: unittest and pytest COLLECT tests. Nothing calls
    them by name, and reporting that as death gave 1,555 and 1,813."""
    scenario("unittest")
    found = wiring.orphans(suite(
        "import unittest\n\nclass T(unittest.TestCase):\n"
        "    def test_a(self):\n        pass\n\n"
        "if __name__ == '__main__':\n    unittest.main()\n"))
    check("unittest-collected tests are not reported dead", found == [],
          str(found))


# ── Invocation is read from the Makefile, not guessed ────────────────

def test_pytest_and_script_invocations_are_told_apart():
    scenario("invocation")
    collected, self_run = wiring.invocations()
    check("some suites are collected by pytest", len(collected) > 10,
          str(len(collected)))
    check("some suites are run as plain scripts", len(self_run) > 10,
          str(len(self_run)))
    check("the two sets are disjoint", not (collected & self_run),
          str(collected & self_run))


# ── Calibration: the property the first three attempts lacked ────────

def test_the_real_repository_is_clean():
    scenario("real-clean")
    _, self_run = wiring.invocations()
    dead = {n: wiring.orphans(wiring.REPO / "scripts" / n)
            for n in self_run if (wiring.REPO / "scripts" / n).exists()}
    dead = {n: v for n, v in dead.items() if v}
    check("no self-run suite has an undispatched test", not dead, str(dead))


def test_calibration_passes_on_the_real_detector():
    scenario("calibration-ok")
    check("the detector agrees with every known-good suite",
          wiring.calibrate() == [], str(wiring.calibrate()))


def test_calibration_catches_a_broken_detector():
    """A detector that cannot reproduce a known answer must be refused,
    not believed. This is what would have stopped 1,555 and 1,813."""
    scenario("calibration-catches")
    original = wiring.orphans
    wiring.orphans = lambda path: ["test_phantom"]      # a wrong detector
    try:
        wrong = wiring.calibrate()
    finally:
        wiring.orphans = original
    check("a broken detector fails calibration",
          len(wrong) == len(wiring.CALIBRATION), str(wrong))
    check("and the real one still passes", wiring.calibrate() == [], "")


def test_an_exit_gate_suite_that_nothing_runs_is_reported():
    """The other half of "a test that is never called is not a test".

    A `check()`/`EXIT GATE` suite counts failures in a module global and
    only exits non-zero under `__main__`. Collected by pytest instead,
    every test function returns normally whatever check() recorded, so
    pytest reports pass while the suite is failing. Proven on
    2026-08-05 by breaking one assertion in test_service_tokens.py:

        python -m pytest scripts/test_service_tokens.py -> 10 passed
        python    scripts/test_service_tokens.py        -> 1 failed,
                                                           EXIT GATE: FAIL

    That suite was in no Makefile recipe, so its 24 assertions could not
    fail anything.
    """
    scenario("unrun exit-gate suite")
    unrun = wiring.unrun_exit_gates()
    check("every EXIT GATE suite is run as a script today", unrun == [],
          str(unrun))


def test_the_rule_covers_the_suites_that_carry_the_marker():
    """Measured before enforcing: 42 suites carry `EXIT GATE` and 41 were
    already wired, so the rule costs nothing and caught the one that was
    not. A floor, so new suites of this shape are covered too."""
    scenario("exit-gate denominator")
    marked = [f for f in (wiring.REPO / "scripts").glob("test_*.py")
              if "EXIT GATE" in f.read_text(encoding="utf-8", errors="replace")]
    check("a real number of suites use this pattern", len(marked) >= 40,
          str(len(marked)))


def run_all() -> None:
    test_an_undispatched_test_is_found()
    test_a_fully_dispatched_suite_is_clean()
    test_a_helper_named_run_does_not_confuse_it()
    test_unittest_collection_is_not_reported()
    test_pytest_and_script_invocations_are_told_apart()
    test_the_real_repository_is_clean()
    test_calibration_passes_on_the_real_detector()
    test_calibration_catches_a_broken_detector()

    test_an_exit_gate_suite_that_nothing_runs_is_reported()
    test_the_rule_covers_the_suites_that_carry_the_marker()

    # Dispatched EXPLICITLY, on purpose. This suite tests the
    # dynamic-dispatch exemption, so it must not be exempt by it — a
    # check whose own fixtures satisfy it proves nothing.
    test_dynamic_dispatch_is_recognised_and_cannot_be_faked()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


# ── I-8 for the dynamic-dispatch exemption ───────────────────────────
#
# A suite that collects its own tests from `globals()` is a collector,
# like unittest.main(): a new test is dispatched by existing. Enumerating
# call sites cannot see that and reports every test in the file.
#
# The risk of the exemption is that it becomes a way to HIDE a suite, so
# both a known-positive and a known-negative are pinned here, and the
# two halves are proven insufficient alone.

def test_dynamic_dispatch_is_recognised_and_cannot_be_faked():
    scenario("dynamic-dispatch")
    import tempfile
    from scripts.security.check_test_wiring import (orphans,
                                                    _dispatches_dynamically)
    tmp = Path(tempfile.mkdtemp()) / "t.py"

    # known-negative: a genuinely undispatched test is STILL reported
    tmp.write_text("def test_a():\n    pass\ndef test_b():\n    pass\n"
                   "def run():\n    test_a()\n")
    check("an undispatched test is still an orphan", orphans(tmp) == ["test_b"])

    # known-positive: dynamic dispatch reports none
    tmp.write_text('def test_a():\n    pass\ndef run():\n'
                   '    [f() for n, f in globals().items() '
                   'if n.startswith("test_")]\n')
    check("a suite that collects from globals() has no orphans",
          orphans(tmp) == [])

    # neither half alone may buy the exemption, or it becomes a way to
    # hide a suite from this check by mentioning a word.
    check("mentioning globals() alone is not dispatch",
          not _dispatches_dynamically("x = globals()"))
    check("a test_ prefix filter alone is not dispatch",
          not _dispatches_dynamically('n.startswith("test_")'))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Test Wiring Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
