"""Tests for `check_workflow_filters` — the layer no other gate could see.

`drift-detector.yml` failed all 15 of its scheduled runs from 2026-04-27
to 2026-08-05 on a jq filter whose quotes were escaped for a shell
context that does not escape them. The file was valid YAML, so the
toleration gate's `unparseable()` passed it. The script was valid bash,
so `bash -n` passed it. The defect lived in a string that only jq ever
parses.

It survived three and a half months because a **scheduled** workflow has
no author watching the result. A push workflow that breaks gets noticed
by whoever pushed; a scheduled one fails into an empty room.

Every assertion below drives synthetic workflow text. The two that read
the repository say so.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_workflow_filters as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 10
executed: list = []

BROKEN = r'''[.[] | select(.title | startswith(\"Weekly drift report (\"))] | .[0]'''
FIXED = r'''[.[] | select(.title | startswith("Weekly drift report ("))] | .[0]'''


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def _workflow(root: Path, name: str, body: str) -> None:
    (root / name).write_text(body, encoding="utf-8")


def _run(root: Path):
    original = gate.WORKFLOWS
    try:
        gate.WORKFLOWS = root
        return gate.audit()
    finally:
        gate.WORKFLOWS = original


# ── extraction ───────────────────────────────────────────────────────

def test_both_invocation_forms_are_found() -> None:
    """`gh ... --jq '...'` and a bare `jq '...'` in a pipeline."""
    scenario("extract both forms")
    text = ("run: |\n"
            "  gh api x --jq '.a.b'\n"
            "  cat f | jq '.c'\n")
    found = gate.filters_in(text)
    check("both are extracted", found == [".a.b", ".c"], str(found))


def test_a_word_ending_in_jq_is_not_mistaken_for_jq() -> None:
    """`yq '...'` and `--foo-jq '...'` are other tools. A detector that
    claims them would report failures in filters it never read."""
    scenario("no false extraction")
    text = "run: |\n  yq '.a'\n  myjq '.b'\n"
    check("neither is claimed", gate.filters_in(text) == [],
          str(gate.filters_in(text)))


# ── compilation ──────────────────────────────────────────────────────

def test_the_real_defect_is_caught() -> None:
    """The exact filter that killed 15 scheduled runs."""
    scenario("real defect caught")
    ok, detail = gate.compiles(BROKEN)
    check("it does not compile", not ok, detail)
    check("and jq says why", "syntax error" in detail, detail)


def test_the_corrected_filter_compiles() -> None:
    scenario("corrected filter compiles")
    ok, detail = gate.compiles(FIXED)
    check("it compiles", ok, detail)


def test_a_runtime_error_is_not_a_syntax_error() -> None:
    """`.a.b` against `null` is fine; `.[0]` against `null` is fine too.
    A filter that compiles and then fails on unrepresentative input is
    not a finding — inventing input for every filter would be a
    false-positive machine, which is the failure mode this repository
    works hardest to avoid."""
    scenario("runtime is not syntax")
    ok, _ = gate.compiles('.commit.committer.date')
    check("a valid filter passes regardless of input shape", ok, "")


# ── the gate ─────────────────────────────────────────────────────────

def test_a_broken_filter_fails_the_gate() -> None:
    scenario("broken filter fails")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _workflow(root, "a.yml", f"run: |\n  gh api x --jq '{BROKEN}'\n")
        findings, checked, workflows = _run(root)
        check("it fails", len(findings) == 1, str(findings))
        check("names the workflow",
              findings and "a.yml" in findings[0], str(findings))
        check("counts what it looked at", checked == 1, str(checked))
        check("and how many files", workflows == 1, str(workflows))


def test_a_clean_workflow_passes() -> None:
    scenario("clean workflow passes")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _workflow(root, "a.yml", f"run: |\n  gh api x --jq '{FIXED}'\n")
        findings, checked, _ = _run(root)
        check("no findings", findings == [], str(findings))
        check("one filter checked", checked == 1, str(checked))


def test_a_double_quoted_filter_is_reported_not_skipped() -> None:
    """It cannot be compiled without knowing the shell variables, so it
    is named rather than quietly passed over. Silence is the defect."""
    scenario("double-quoted reported")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _workflow(root, "a.yml", 'run: |\n  gh api x --jq ".a.$VAR"\n')
        findings, checked, _ = _run(root)
        check("it is reported", len(findings) == 1, str(findings))
        check("and says why",
              findings and "shell expansion" in findings[0], str(findings))
        check("and is not counted as checked", checked == 0, str(checked))


def test_no_filters_is_reported_as_no_filters() -> None:
    """I-2. Zero checked must be distinguishable from all-passing."""
    scenario("zero denominator")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _workflow(root, "a.yml", "run: |\n  echo hello\n")
        findings, checked, _ = _run(root)
        check("nothing is reported", findings == [], str(findings))
        check("and the denominator is zero, not hidden", checked == 0,
              str(checked))


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_filters_all_compile() -> None:
    scenario("repository passes today")
    findings, checked, workflows = gate.audit()
    check("no broken filters", findings == [], str(findings))
    check("and something was actually checked", checked > 0, str(checked))
    check("across every workflow", workflows >= 9, str(workflows))


def run_all() -> None:
    if shutil.which("jq") is None:
        # Fail rather than skip: this suite exists to prove a jq-backed
        # gate works, and it cannot prove that without jq.
        print("  FAIL: jq is not installed, so none of this could be verified")
        print("\n" + "=" * 60)
        print("Workflow Filter Tests: 0 passed, 1 failed")
        print("EXIT GATE: FAIL")
        sys.exit(1)

    test_both_invocation_forms_are_found()
    test_a_word_ending_in_jq_is_not_mistaken_for_jq()
    test_the_real_defect_is_caught()
    test_the_corrected_filter_compiles()
    test_a_runtime_error_is_not_a_syntax_error()
    test_a_broken_filter_fails_the_gate()
    test_a_clean_workflow_passes()
    test_a_double_quoted_filter_is_reported_not_skipped()
    test_no_filters_is_reported_as_no_filters()
    test_the_repository_filters_all_compile()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Workflow Filter Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
