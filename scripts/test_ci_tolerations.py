"""CI toleration gate tests — a step that passes while doing nothing.

The operator's rule, encoded by the gate this suite guards:

> Zero tolerance for silent failure. High tolerance for documented skips
> with a reason and an owner.

Two of these cases are regression tests for defects the gate itself had
while being written:

  - Its first suppression pattern included ``if cmd; then`` and matched
    every ordinary shell conditional in the repository. Structural
    guessing was replaced by an explicit `# ci-toleration:` marker,
    cross-checked in both directions.
  - Adding the parse check found **three workflows that do not parse as
    YAML**, including `core-tests.yml`. A workflow that does not parse
    runs nothing, and running nothing is indistinguishable from having no
    failures.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_ci_tolerations as ci  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 12
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
    executed.append(name)


def with_workflows(files: dict):
    """Point the gate at a synthetic .github/workflows directory."""
    tmp = Path(tempfile.mkdtemp()) / "workflows"
    tmp.mkdir(parents=True)
    for name, body in files.items():
        (tmp / name).write_text(body)
    return tmp


def survey_in(files: dict):
    original = ci.WORKFLOWS
    ci.WORKFLOWS = with_workflows(files)
    try:
        return ci.survey(), ci.unparseable(), ci.markers()
    finally:
        ci.WORKFLOWS = original


VALID = ("name: x\non: [push]\njobs:\n  j:\n    runs-on: ubuntu-latest\n"
         "    steps:\n      - name: Step\n        run: echo hi\n")


# ── Suppression detection ────────────────────────────────────────────

def test_a_swallowed_exit_code_is_found():
    scenario("suppress-echo")
    (found, _), _, _ = survey_in({"a.yml": VALID.replace(
        "run: echo hi", 'run: make thing || echo "::warning::oops"')})
    check("`|| echo` is detected", len(found) == 1, str(found))


def test_continue_on_error_is_found():
    scenario("suppress-coe")
    (found, _), _, _ = survey_in({"a.yml": VALID.replace(
        "        run: echo hi", "        continue-on-error: true\n        run: echo hi")})
    check("`continue-on-error: true` is detected", len(found) == 1, str(found))


def test_install_tolerance_is_not_a_suppression():
    """`pip install psutil || true` — the test after it is still the gate.
    Verified empirically: none of the five optional-dep suites skips on a
    missing import."""
    scenario("install-ok")
    (found, _), _, _ = survey_in({"a.yml": VALID.replace(
        "run: echo hi", "run: pip install psutil --quiet || true")})
    check("install tolerance is not flagged", not found, str(found))


def test_an_icon_ternary_is_not_a_suppression():
    scenario("icon-ok")
    (found, _), _, _ = survey_in({"a.yml": VALID.replace(
        "run: echo hi", 'run: icon=$( [ "$C" = "0" ] && echo "OK" || echo "BAD" )')})
    check("a report ternary is not flagged", not found, str(found))


def test_an_ordinary_shell_conditional_is_not_a_suppression():
    """Regression: the first pattern matched every `if ...; then` in the
    repository — a false-positive machine."""
    scenario("conditional-ok")
    (found, _), _, _ = survey_in({"a.yml": VALID.replace(
        "run: echo hi", "run: |\n          if [ -n \"$X\" ]; then\n            echo yes\n          fi")})
    check("a plain conditional is not flagged", not found, str(found))


# ── The parse check ──────────────────────────────────────────────────

def test_an_unparseable_workflow_is_caught():
    """A workflow that does not parse runs nothing."""
    scenario("unparseable")
    broken_yaml = ("name: x\non: [push]\njobs:\n  j:\n    steps:\n"
                   "      - run: |\n          echo start\n"
                   "import sys, json\n")
    _, broken, _ = survey_in({"bad.yml": broken_yaml})
    check("an unparseable workflow is caught", len(broken) == 1, str(broken))
    check("it is named", broken and "bad.yml" in broken[0], str(broken))


def test_a_valid_workflow_parses():
    scenario("parseable")
    _, broken, _ = survey_in({"a.yml": VALID})
    check("a valid workflow is not flagged", not broken, str(broken))


# ── Markers and declarations must agree, both ways ───────────────────

def test_a_marker_is_read_from_the_workflow():
    scenario("marker-read")
    _, _, marks = survey_in({"a.yml": VALID.replace(
        "      - name: Step", "      # ci-toleration: needs-owner\n      - name: Step")})
    check("the marker and its bucket are read",
          marks == [("a.yml", "needs-owner")], str(marks))


def test_the_real_repository_agrees_in_both_directions():
    """Every declaration has a marker, and every marker a declaration."""
    scenario("real-agreement")
    marks = set(ci.markers())
    declared = {(d.workflow, d.bucket) for d in ci.DECLARED}
    check("no declaration lacks a marker", not (declared - marks),
          str(declared - marks))
    check("no marker lacks a declaration", not (marks - declared),
          str(marks - declared))
    check("every declaration carries an owner and a date",
          all(d.owner and d.review_by for d in ci.DECLARED), "")
    check("no toleration sits in the defect bucket",
          not [d for d in ci.DECLARED if d.bucket == ci.DEFECT],
          str([d.step for d in ci.DECLARED if d.bucket == ci.DEFECT]))


# ── the third directive: nothing repeats unexplained ─────────────────

def test_every_warning_emitter_is_declared():
    """A step that prints `::warning::` and carries on has decided
    something is not worth failing over — the same decision as swallowing
    an exit code, needing the same reason, owner and date.

    Added after `The "DB_PASSWORD" variable is not set` printed on
    every
    compose invocation in every log for a day and was read past, while
    postgres refused to start because of it. On its first run this rule
    found one undeclared emitter, written earlier the same day.
    """
    scenario("warnings declared")
    check("no step warns without an owner", ci.undeclared_warnings() == [],
          str(ci.undeclared_warnings()))


def test_the_warning_detector_finds_the_emitters():
    """A denominator. If this ever returns nothing the rule has gone
    blind rather than the workflows having gone quiet."""
    scenario("warning denominator")
    emitters = ci.warning_emitters()
    check("emitters are found", len(emitters) >= 5, str(len(emitters)))
    check("each is (workflow, step)",
          all(len(e) == 2 and e[0].endswith(".yml") for e in emitters),
          str(emitters[:3]))


def test_an_undeclared_emitter_is_reported():
    """Calibration: the rule must be able to fail."""
    scenario("undeclared emitter fails")
    original = ci.DECLARED
    try:
        ci.DECLARED = ()
        found = ci.undeclared_warnings()
        check("with nothing declared, every emitter is reported",
              len(found) >= 5, str(len(found)))
        check("and each says why",
              all("nobody reads" in f for f in found), str(found[:1]))
    finally:
        ci.DECLARED = original


def run_all() -> None:
    test_a_swallowed_exit_code_is_found()
    test_continue_on_error_is_found()
    test_install_tolerance_is_not_a_suppression()
    test_an_icon_ternary_is_not_a_suppression()
    test_an_ordinary_shell_conditional_is_not_a_suppression()
    test_an_unparseable_workflow_is_caught()
    test_a_valid_workflow_parses()
    test_a_marker_is_read_from_the_workflow()
    test_the_real_repository_agrees_in_both_directions()

    test_every_warning_emitter_is_declared()
    test_the_warning_detector_finds_the_emitters()
    test_an_undeclared_emitter_is_reported()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"CI Toleration Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
