"""Hygiene ratchet tests — H-5.

The ratchet is the only thing standing between a one-time cleanup and a
slow return to 136. If it cannot fail, it is decoration.

The cases that matter are the ones where it must *refuse*: a count that
has risen, and an attempt to hide that rise by raising the ceiling. A
gate whose baseline can be edited upward by the same change that breaks
it protects nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import hygiene_survey as hs

passed = 0
failed = 0

REPO = Path(__file__).resolve().parent.parent
SURVEY = REPO / "scripts" / "security" / "hygiene_survey.py"


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


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SURVEY), *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


class _Baseline:
    """Swap the baseline file, restoring the real one afterwards."""

    def __init__(self, totals):
        self.totals = totals
        self._saved = None

    def __enter__(self):
        self._saved = hs.BASELINE.read_text(encoding="utf-8")
        hs.BASELINE.write_text(
            json.dumps({"totals": self.totals,
                        "grand_total": sum(self.totals.values())}) + "\n",
            encoding="utf-8")
        return self

    def __exit__(self, *exc):
        hs.BASELINE.write_text(self._saved, encoding="utf-8")
        return False


# ── The ratchet ──────────────────────────────────────────────────────

def test_current_tree_passes_its_own_baseline():
    result = _run("--gate")
    check("the recorded baseline matches the tree",
          result.returncode == 0, result.stdout[-400:])


def test_a_risen_count_fails_the_gate():
    totals = hs.survey()
    actual = {c: sum(r[c] for r in totals.values()) for c in hs.COLUMNS}
    lowered = dict(actual)
    lowered["clients"] = max(0, actual["clients"] - 1)
    with _Baseline(lowered):
        result = _run("--gate")
    check("a count above the baseline fails the gate",
          result.returncode == 1, str(result.returncode))
    check("the failure names the column that rose",
          "clients:" in result.stdout, result.stdout[-300:])


def test_an_improved_count_passes():
    totals = hs.survey()
    actual = {c: sum(r[c] for r in totals.values()) for c in hs.COLUMNS}
    generous = {c: v + 5 for c, v in actual.items()}
    with _Baseline(generous):
        result = _run("--gate")
    check("a count below the baseline passes", result.returncode == 0,
          result.stdout[-300:])


def test_every_column_is_ratcheted():
    """A column absent from the ratchet is a column that can rise freely.

    Driven against `ratchet()` with synthetic totals rather than the real
    tree. Subprocess runs can only exercise columns whose current count is
    above zero — and once a column is driven to zero (as
    `naive_timestamps` now is by H-1) that test silently stops testing it,
    which is precisely the failure mode this suite exists to prevent.
    """
    baseline = {c: 10 for c in hs.COLUMNS}
    with _Baseline(baseline):
        for column in hs.COLUMNS:
            risen = dict(baseline)
            risen[column] = 11
            reported = hs.ratchet(risen)
            check(f"{column} is ratcheted",
                  any(column in line for line in reported),
                  f"a rise in {column} was not reported: {reported}")

            unchanged = dict(baseline)
            check(f"{column} does not false-positive when unchanged",
                  not hs.ratchet(unchanged),
                  str(hs.ratchet(unchanged)))

            improved = dict(baseline)
            improved[column] = 9
            check(f"{column} does not fail when improved",
                  not hs.ratchet(improved), str(hs.ratchet(improved)))


def test_a_column_missing_from_the_baseline_fails():
    """An unrecorded column would otherwise be unratcheted and invisible."""
    partial = {c: 10 for c in hs.COLUMNS if c != "naive_timestamps"}
    with _Baseline(partial):
        reported = hs.ratchet({c: 10 for c in hs.COLUMNS})
    check("a column absent from the baseline is reported",
          any("naive_timestamps" in line for line in reported), str(reported))


def test_a_zeroed_column_is_still_ratcheted():
    """H-1 drove naive_timestamps to zero; it must not become free to rise."""
    baseline = {c: 0 for c in hs.COLUMNS}
    with _Baseline(baseline):
        reported = hs.ratchet({**baseline, "naive_timestamps": 1})
    check("a column at zero still fails when it rises",
          any("naive_timestamps" in line for line in reported), str(reported))


def test_the_baseline_cannot_be_raised():
    """Otherwise the change that breaks the gate can also silence it."""
    totals = hs.survey()
    actual = {c: sum(r[c] for r in totals.values()) for c in hs.COLUMNS}
    lowered = dict(actual)
    lowered["clients"] = max(0, actual["clients"] - 1)
    with _Baseline(lowered):
        result = _run("--update-baseline")
        after = json.loads(hs.BASELINE.read_text(encoding="utf-8"))["totals"]
    check("raising the ceiling is refused", result.returncode == 1,
          result.stdout[-200:])
    check("the refusal says why", "only be lowered" in result.stdout,
          result.stdout[-200:])
    check("the baseline file is left unchanged",
          after["clients"] == lowered["clients"], str(after))


def test_the_baseline_can_be_lowered():
    totals = hs.survey()
    actual = {c: sum(r[c] for r in totals.values()) for c in hs.COLUMNS}
    generous = {c: v + 5 for c, v in actual.items()}
    with _Baseline(generous):
        result = _run("--update-baseline")
        after = json.loads(hs.BASELINE.read_text(encoding="utf-8"))["totals"]
    check("an improvement can be locked in", result.returncode == 0,
          result.stdout[-200:])
    check("the new ceiling is the improved count",
          after == actual, f"{after} != {actual}")


def test_a_missing_baseline_fails_rather_than_passes():
    saved = hs.BASELINE.read_text(encoding="utf-8")
    hs.BASELINE.unlink()
    try:
        result = _run("--gate")
    finally:
        hs.BASELINE.write_text(saved, encoding="utf-8")
    check("no baseline is a failure, not a free pass",
          result.returncode == 1, str(result.returncode))


# ── The survey underneath it ─────────────────────────────────────────

def test_survey_counts_only_route_handlers_for_success_on_failure():
    """A helper returning a dict is not an HTTP 200."""
    helper = ("def _helper():\n    try:\n        pass\n"
              "    except Exception:\n        return {'a': 1}\n")
    routed = ("@app.get('/x')\nasync def x():\n    try:\n        pass\n"
              "    except Exception:\n        return {'a': 1}\n")
    check("a bare helper is not counted", hs._success_on_failure(helper) == 0)
    check("a route handler is counted", hs._success_on_failure(routed) == 1)


def test_survey_ignores_remediated_failure_paths():
    fixed = ("@app.get('/x')\nasync def x():\n    try:\n        pass\n"
             "    except Exception as exc:\n"
             "        return degraded_response('s', str(exc), {'a': 1})\n")
    check("a degraded response is not counted",
          hs._success_on_failure(fixed) == 0)


def test_survey_reaches_nested_services():
    services = {str(p.parent.relative_to(REPO)) for p in hs._service_files()}
    check("nested service paths are surveyed",
          any("/" in s for s in services), str(sorted(services)[:5]))


def test_adoption_is_reported_not_just_debt():
    results = hs.survey()
    adopted = sum(r["pooled"] for r in results.values())
    check("adoption of the shared pool is visible", adopted > 0, str(adopted))


def run() -> None:
    test_current_tree_passes_its_own_baseline()
    test_a_risen_count_fails_the_gate()
    test_an_improved_count_passes()
    test_every_column_is_ratcheted()
    test_a_column_missing_from_the_baseline_fails()
    test_a_zeroed_column_is_still_ratcheted()
    test_the_baseline_cannot_be_raised()
    test_the_baseline_can_be_lowered()
    test_a_missing_baseline_fails_rather_than_passes()
    test_survey_counts_only_route_handlers_for_success_on_failure()
    test_survey_ignores_remediated_failure_paths()
    test_survey_reaches_nested_services()
    test_adoption_is_reported_not_just_debt()


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Hygiene Gate Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
