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


class _Survey:
    """Replace the survey with synthetic counts.

    Lets the gate's guarantees be asserted independently of how much real
    debt happens to remain. Tests that only work while the tree is dirty
    lose coverage exactly as the system gets healthier — which is how the
    `naive_timestamps` and `clients` ratchets each went quietly untested.
    """

    def __init__(self, totals):
        self.totals = totals
        self._saved = None

    def __enter__(self):
        self._saved = hs.survey
        hs.survey = lambda: {"synthetic": {**self.totals, "pooled": 0, "bounded": 0}}
        return self

    def __exit__(self, *exc):
        hs.survey = self._saved
        return False


def _main(*argv) -> int:
    """Run the survey's own main() in-process, capturing its exit code."""
    saved = sys.argv
    sys.argv = ["hygiene_survey.py", *argv]
    try:
        return hs.main()
    finally:
        sys.argv = saved


# ── The ratchet ──────────────────────────────────────────────────────

def test_current_tree_passes_its_own_baseline():
    result = _run("--gate")
    check("the recorded baseline matches the tree",
          result.returncode == 0, result.stdout[-400:])


def _actual_totals():
    surveyed = hs.survey()
    return {c: sum(r[c] for r in surveyed.values()) for c in hs.COLUMNS}


def test_a_risen_count_fails_the_gate():
    """Driven from an all-zero baseline rather than by decrementing a
    named column.

    Picking a column by name and lowering it assumes that column is
    non-zero — and as the debt is paid down, such a test silently stops
    testing. That has now happened twice in this suite (`naive_timestamps`
    after H-1, `clients` after H-4). An all-zero baseline fails whenever
    *any* debt remains, and when none does, `ratchet()` is exercised
    directly instead.
    """
    actual = _actual_totals()
    with _Baseline({c: 0 for c in hs.COLUMNS}):
        result = _run("--gate")
    if any(actual.values()):
        check("a count above the baseline fails the gate",
              result.returncode == 1, str(result.returncode))
        risen = [c for c, v in actual.items() if v]
        check("the failure names a column that rose",
              any(f"{c}:" in result.stdout for c in risen),
              result.stdout[-300:])
    else:
        check("with no debt left, the gate passes an all-zero baseline",
              result.returncode == 0, result.stdout[-200:])
        reported = hs.ratchet({**{c: 0 for c in hs.COLUMNS},
                               hs.COLUMNS[0]: 1})
        check("and ratchet() still reports a synthetic rise", bool(reported))


def test_an_improved_count_passes():
    actual = _actual_totals()
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
    """Otherwise the change that breaks the gate can also silence it.

    Driven from synthetic counts so the guarantee holds whether the tree
    is clean or filthy.
    """
    baseline = {c: 1 for c in hs.COLUMNS}
    risen = {c: 2 for c in hs.COLUMNS}
    with _Baseline(baseline), _Survey(risen):
        code = _main("--update-baseline")
        after = json.loads(hs.BASELINE.read_text(encoding="utf-8"))["totals"]
    check("raising the ceiling is refused", code == 1, str(code))
    check("the baseline file is left unchanged", after == baseline, str(after))


def test_the_baseline_can_be_lowered_synthetically():
    baseline = {c: 5 for c in hs.COLUMNS}
    improved = {c: 2 for c in hs.COLUMNS}
    with _Baseline(baseline), _Survey(improved):
        code = _main("--update-baseline")
        after = json.loads(hs.BASELINE.read_text(encoding="utf-8"))["totals"]
    check("an improvement is accepted", code == 0, str(code))
    check("the new ceiling is the improved count", after == improved, str(after))


def test_a_rise_in_any_single_column_is_refused():
    """One column rising while the others fall must still be refused."""
    baseline = {c: 5 for c in hs.COLUMNS}
    for column in hs.COLUMNS:
        mixed = {c: 1 for c in hs.COLUMNS}
        mixed[column] = 6
        with _Baseline(baseline), _Survey(mixed):
            code = _main("--update-baseline")
        check(f"a rise in {column} is refused even as others improve",
              code == 1, f"{column} accepted")


def test_the_gate_fails_on_synthetic_debt():
    """Independent of the real tree, so it keeps working at zero."""
    with _Baseline({c: 0 for c in hs.COLUMNS}), _Survey({c: 1 for c in hs.COLUMNS}):
        code = _main("--gate")
    check("the gate fails when synthetic counts exceed the baseline",
          code == 1, str(code))
    with _Baseline({c: 5 for c in hs.COLUMNS}), _Survey({c: 1 for c in hs.COLUMNS}):
        code = _main("--gate")
    check("the gate passes when synthetic counts are below the baseline",
          code == 0, str(code))


def test_the_baseline_can_be_lowered():
    actual = _actual_totals()
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


# ── Calibration: every detector must still detect ────────────────────
#
# A ratchet only catches a count that RISES. A detector that quietly
# stops detecting takes every count to zero and the gate reports
# improvement — the one failure a ratchet is structurally incapable of
# seeing.
#
# Not hypothetical. On 2026-08-05, routing the textual detectors through
# a tokeniser to stop `common/http_hygiene.py`'s docstring being counted
# as an unbounded body took `clients` from 16 to 0 and adoption from 149
# to 0, because that first tokeniser rebuilt the source with newlines
# between tokens. **The gate passed.** Nothing was wrong with the
# ratchet; it was doing exactly what a ratchet does.
#
# So each detector is pointed at input whose answer is known before it is
# pointed at the repository. The denominator is `DETECTORS`, so adding a
# detector without a sample fails rather than going uncalibrated.

_CALIBRATION = {
    "clients": (
        "async def f():\n"
        "    async with httpx.AsyncClient(timeout=1.0) as c:\n"
        "        await c.get('http://x')\n"
    ),
    "unbounded_bodies": (
        "async def f(request):\n"
        "    body = await request.json()\n"
        "    return body\n"
    ),
    "naive_timestamps": "def f():\n    return datetime.utcnow()\n",
    "success_on_failure": (
        "@app.get('/x')\n"
        "async def handler():\n"
        "    try:\n"
        "        return {'ok': True}\n"
        "    except Exception:\n"
        "        return {'ok': True}\n"
    ),
    "silent_swallows": (
        "def f():\n"
        "    try:\n"
        "        g()\n"
        "    except Exception:\n"
        "        pass\n"
    ),
}

# Prose that mentions every pattern without being any of them. A detector
# that fires on this has the false-positive problem the tokeniser exists
# to prevent — and the module that fixes a defect necessarily describes
# it, which is how `http_hygiene.py` came to be counted as debt.
_PROSE_ONLY = "\n".join([
    'MODULE_DOC = """A module documenting what it replaces.',
    '',
    'Avoid `async with httpx.AsyncClient(` inside a handler; avoid an',
    'unbounded `await request.json()`; never use `datetime.utcnow()`.',
    'Do not write `except Exception:` followed by `pass`.',
    '"""',
    '',
    '# async with httpx.AsyncClient( in a comment is also not a client.',
    'def f():',
    '    return 1',
    '',
])


def test_every_detector_has_a_calibration_sample():
    """The denominator is DETECTORS, so a new detector needs a sample."""
    missing = sorted(set(hs.DETECTORS) - set(_CALIBRATION))
    check("no detector lacks a known-positive sample", not missing, str(missing))


def test_every_detector_fires_on_its_known_positive():
    """The check a ratchet cannot make: does this still detect anything?"""
    for name, sample in sorted(_CALIBRATION.items()):
        detector = hs.DETECTORS.get(name)
        if detector is None:
            check(f"{name} still exists", False, "detector removed")
            continue
        found = detector(sample)
        check(f"{name} fires on a known positive", found >= 1,
              f"returned {found} for:\n{sample}")


def test_no_detector_fires_on_prose_alone():
    """The other half. A survey with false positives invites someone to
    'fix' correct code, and a module that fixes a defect has to describe
    it."""
    for name, detector in sorted(hs.DETECTORS.items()):
        found = detector(_PROSE_ONLY)
        check(f"{name} ignores a docstring describing the pattern",
              found == 0, f"returned {found}")


def test_adoption_detectors_are_calibrated_too():
    """Adoption is how progress is reported. Silently zeroing it would
    read as 'nobody has adopted the fix'."""
    pooled = "async def f():\n    async with pooled_client() as c:\n        pass\n"
    check("pooled fires", hs.ADOPTION_DETECTORS["pooled"](pooled) >= 1,
          str(hs.ADOPTION_DETECTORS["pooled"](pooled)))
    bounded = "async def f(request):\n    body = await bounded_json(request)\n"
    check("bounded fires", hs.ADOPTION_DETECTORS["bounded"](bounded) >= 1,
          str(hs.ADOPTION_DETECTORS["bounded"](bounded)))


def test_the_survey_covers_library_modules_not_only_entry_points():
    """The scope was twice a hand-written list, and twice too narrow.

    It began as `*/app.py`, which missed `agentic/introspect_app.py`.
    Widened to four globs, it then missed all 117 library modules — and a
    defect in `common/llm.py` reaches every service, so the least-covered
    files had the widest blast radius. Widening surfaced 16 per-request
    clients and 30 silent swallows the ratchet had never been able to
    see.

    This pins the property that matters: the denominator is derived from
    the tree, so adding a module cannot leave it unscanned.
    """
    scanned = {str(p.relative_to(REPO)) for p in hs._service_files()}
    for library in ("common/llm.py", "common/policy.py", "agentic/kai_config.py"):
        check(f"{library} is surveyed", library in scanned, str(len(scanned)))
    check("tests are still excluded",
          not any(f.startswith("scripts/") for f in scanned), "")
    check("vendored code is still excluded",
          not any("site-packages" in f or "node_modules" in f for f in scanned), "")


def test_widening_the_scope_needs_a_written_reason():
    """A rise is either a regression or a change of denominator, and the
    two must not be spelled the same way.

    `--update-baseline` refuses a rise, which is what makes the ratchet a
    ratchet. `--widen-scope` allows one and records why, so the rise
    shows up in review as a deliberate act rather than as a number that
    quietly went the wrong way.
    """
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/security/hygiene_survey.py"), "--help"],
        capture_output=True, text=True, timeout=60)
    check("--widen-scope exists", "--widen-scope" in proc.stdout, proc.stdout[-200:])
    check("it takes a reason rather than being a bare flag",
          "REASON" in proc.stdout, proc.stdout[-200:])

    baseline = json.loads(hs.BASELINE.read_text(encoding="utf-8"))
    widened = baseline.get("scope_widened")
    if widened is not None:
        check("a recorded widening carries its reason",
              bool(widened.get("reason", "").strip()), str(widened))
        check("and names which columns it raised",
              bool(widened.get("raised")), str(widened))


def test_adoption_is_reported_not_just_debt():
    results = hs.survey()
    adopted = sum(r["pooled"] for r in results.values())
    check("adoption of the shared pool is visible", adopted > 0, str(adopted))


def test_a_nested_helper_is_not_counted_as_a_route_failure():
    """A helper returning a dict is not the route's HTTP 200.

    `agentic`'s per-node `_ping()` returns {"reachable": False} and the
    route around it succeeds while reporting which nodes are down. That
    is correct code, and counting it invited someone to "fix" it.
    """
    nested = ("@app.get('/x')\n"
              "async def x():\n"
              "    async def _ping(u):\n"
              "        try:\n"
              "            return await get(u)\n"
              "        except Exception:\n"
              "            return {'reachable': False}\n"
              "    return {'pings': await _ping('u')}\n")
    own = ("@app.get('/y')\n"
           "async def y():\n"
           "    try:\n"
           "        return await get('u')\n"
           "    except Exception:\n"
           "        return {'status': 'unavailable'}\n")
    check("a nested helper's dict return is not counted",
          hs._success_on_failure(nested) == 0,
          str(hs._success_on_failure(nested)))
    check("the route's own dict return is still counted",
          hs._success_on_failure(own) == 1,
          str(hs._success_on_failure(own)))


def test_lambdas_and_classes_are_separate_scopes():
    src = ("@app.get('/z')\n"
           "async def z():\n"
           "    class Inner:\n"
           "        def go(self):\n"
           "            try:\n"
           "                pass\n"
           "            except Exception:\n"
           "                return {'a': 1}\n"
           "    return Inner().go()\n")
    check("a nested class body is a separate scope",
          hs._success_on_failure(src) == 0, str(hs._success_on_failure(src)))


def run() -> None:
    test_current_tree_passes_its_own_baseline()
    test_a_risen_count_fails_the_gate()
    test_an_improved_count_passes()
    test_every_column_is_ratcheted()
    test_a_column_missing_from_the_baseline_fails()
    test_a_zeroed_column_is_still_ratcheted()
    test_the_baseline_cannot_be_raised()
    test_the_baseline_can_be_lowered_synthetically()
    test_a_rise_in_any_single_column_is_refused()
    test_the_gate_fails_on_synthetic_debt()
    test_the_baseline_can_be_lowered()
    test_a_missing_baseline_fails_rather_than_passes()
    test_survey_counts_only_route_handlers_for_success_on_failure()
    test_survey_ignores_remediated_failure_paths()
    test_a_nested_helper_is_not_counted_as_a_route_failure()
    test_lambdas_and_classes_are_separate_scopes()
    test_survey_reaches_nested_services()
    test_every_detector_has_a_calibration_sample()
    test_every_detector_fires_on_its_known_positive()
    test_no_detector_fires_on_prose_alone()
    test_adoption_detectors_are_calibrated_too()
    test_the_survey_covers_library_modules_not_only_entry_points()
    test_widening_the_scope_needs_a_written_reason()
    test_adoption_is_reported_not_just_debt()


if __name__ == "__main__":
    run()
    print(f"\n{'='*60}")
    print(f"Hygiene Gate Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
