"""Cross-file isolation gate tests — driven entirely by synthetic reports.

The gate this suite guards was written after the repo-wide pytest run was
found to have executed **zero** tests, on every run, for at least a week.
It aborted during collection because one file replaced `sys.modules
["common"]` and five later files could not import `common.<anything>`.

Two rules from `kai-pm/TEST_WRITING_REVIEW.md` shape this file:

  - **Class B — never guard on state the repository owns.** Not one
    assertion here reads the real repository. Every input is a synthetic
    report built in-process, so nothing in this suite can break because
    a leak was *fixed*. That defect (a test that required its own bug to
    persist) is the sharpest one in the review.
  - **Class D — derive fixtures from the source of truth.** `_report()`
    builds entries with every key the plugin emits, taken from the
    plugin's own writer, so a new key cannot silently go untested the way
    `inert` and then `lapsed` did.

The one assertion that does touch reality is deliberate and one-way:
`test_the_real_repository_replaces_nothing`. `replaced` is at zero and
enforced, so that assertion can only ever be made *stronger* by the
repository improving.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_test_isolation as gate  # noqa: E402
from scripts.security import isolation_plugin as plugin  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

passed = 0
failed = 0

EXPECTED_SCENARIOS = 15
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


# The keys the plugin actually writes. Derived, not retyped: when the
# plugin grows a category, every fixture below grows with it.
_KEYS = ("replaced", "added", "env_set", "env_changed", "path_added")


def _entry(**kwargs) -> dict:
    entry = {key: [] for key in _KEYS}
    entry.update(kwargs)
    return entry


def _report(**files) -> dict:
    """A synthetic plugin report, keyed by absolute path as pytest emits."""
    return {str(gate.REPO / "scripts" / name): entry
            for name, entry in files.items()}


def _baseline(**files) -> dict:
    return {"replaced_allowed": 0,
            "leaky_files": {f"scripts/{name}": counts
                            for name, counts in files.items()}}


# ── The rule that is at zero and enforced ────────────────────────────

def test_a_replaced_module_fails() -> None:
    scenario("replaced fails")
    replacements, grown, _ = gate.compare(
        _report(**{"test_a.py": _entry(
            replaced=["httpx (mod:/real/httpx.py -> mod:none)"])}),
        _baseline())
    check("a replacement is reported", len(replacements) == 1, str(replacements))
    check("and is not confused with growth", grown == [], str(grown))


def test_a_replacement_fails_even_when_declared() -> None:
    """There is no way to baseline a replacement. That is the point."""
    scenario("replacement cannot be declared away")
    replacements, _, _ = gate.compare(
        _report(**{"test_a.py": _entry(replaced=["fastapi (mod:/x -> mod:none)"])}),
        _baseline(**{"test_a.py": {"added": 99, "env_set": 99}}))
    check("declaring the file does not excuse the replacement",
          len(replacements) == 1, str(replacements))


def test_the_replacement_message_names_the_file() -> None:
    scenario("message names the culprit")
    replacements, _, _ = gate.compare(
        _report(**{"test_culprit.py": _entry(replaced=["httpx (mod:/a -> mod:b)"])}),
        _baseline())
    check("the culprit is named, not the victim",
          any("test_culprit.py" in line for line in replacements),
          str(replacements))


# ── The ratchet ──────────────────────────────────────────────────────

def test_declared_leakage_at_its_baseline_passes() -> None:
    scenario("baseline passes")
    _, grown, _ = gate.compare(
        _report(**{"test_a.py": _entry(env_set=["ONE", "TWO"], added=["x"])}),
        _baseline(**{"test_a.py": {"env_set": 2, "added": 1}}))
    check("a leak at its declared size passes", grown == [], str(grown))


def test_leakage_that_shrinks_passes() -> None:
    scenario("shrinking passes")
    _, grown, _ = gate.compare(
        _report(**{"test_a.py": _entry(env_set=["ONE"])}),
        _baseline(**{"test_a.py": {"env_set": 2, "added": 0}}))
    check("a leak that shrank passes", grown == [], str(grown))


def test_leakage_that_grows_fails() -> None:
    scenario("growth fails")
    _, grown, _ = gate.compare(
        _report(**{"test_a.py": _entry(env_set=["ONE", "TWO", "THREE"])}),
        _baseline(**{"test_a.py": {"env_set": 2, "added": 0}}))
    check("a leak that grew fails", len(grown) == 1, str(grown))


def test_an_undeclared_leaky_file_fails() -> None:
    """The week-long outage was invisible. Silence is not a pass."""
    scenario("undeclared fails")
    _, grown, _ = gate.compare(
        _report(**{"test_new.py": _entry(added=["zzz"])}),
        _baseline())
    check("a file nobody declared fails", len(grown) == 1, str(grown))


def test_a_clean_file_is_not_reported() -> None:
    scenario("clean is clean")
    replacements, grown, totals = gate.compare(_report(), _baseline())
    check("nothing reported", replacements == [] and grown == [], "")
    check("totals are zero", totals == {"replaced": 0, "added": 0, "env_set": 0},
          str(totals))


# ── Fail-closed behaviour ────────────────────────────────────────────

def test_a_missing_report_is_a_failure_not_a_pass() -> None:
    """Boundary blindness: no report must not read as no leaks."""
    scenario("missing report fails")
    missing = Path(tempfile.mkdtemp()) / "absent.json"
    argv = sys.argv
    sys.argv = ["check_test_isolation.py", "--from-report", str(missing)]
    try:
        status = gate.main()
    finally:
        sys.argv = argv
    check("a missing report exits nonzero", status == 1, f"exit={status}")


def test_an_empty_report_against_a_declared_baseline_fails() -> None:
    """CI's own finding: the gate warned "this is not a pass", then passed.

    On 2026-08-04 the suite aborted during collection, the plugin wrote an
    empty report, and this check printed a warning and exited 0 — boundary
    blindness in the file written to prevent boundary blindness. A warning
    was doing the work of a rule.
    """
    scenario("empty report fails")
    report = Path(tempfile.mkdtemp()) / "empty.json"
    report.write_text("{}", encoding="utf-8")
    argv = sys.argv
    sys.argv = ["check_test_isolation.py", "--from-report", str(report)]
    try:
        status = gate.main()
    finally:
        sys.argv = argv
    check("an empty report is not a clean repository", status == 1,
          f"exit={status}")


def test_write_baseline_refuses_to_record_a_replacement() -> None:
    scenario("baseline refuses a replacement")
    report = Path(tempfile.mkdtemp()) / "report.json"
    report.write_text(json.dumps(
        _report(**{"test_a.py": _entry(replaced=["httpx (mod:/a -> mod:none)"])})),
        encoding="utf-8")
    before = gate.BASELINE.read_text(encoding="utf-8")
    argv = sys.argv
    sys.argv = ["check_test_isolation.py", "--from-report", str(report),
                "--write-baseline"]
    try:
        status = gate.main()
    finally:
        sys.argv = argv
    check("recording is refused", status == 1, f"exit={status}")
    check("and the baseline is untouched",
          gate.BASELINE.read_text(encoding="utf-8") == before, "")


# ── The plugin's own judgement ───────────────────────────────────────

def test_the_plugin_tells_a_swap_from_an_import() -> None:
    """Importing a module is what tests are for; swapping one is not."""
    scenario("swap vs import")
    import types
    real = types.ModuleType("x")
    real.__spec__ = type("S", (), {"origin": "/usr/lib/x.py"})()
    stub = types.ModuleType("x")
    check("a real module and a stub differ",
          plugin._fingerprint(real) != plugin._fingerprint(stub), "")
    check("a module is not mistaken for a mock",
          plugin._fingerprint(real).startswith("mod:"), "")


def test_the_real_repository_replaces_nothing() -> None:
    """One-way: `replaced` is at zero, so this can only get stronger."""
    scenario("repo has no replacements")
    baseline = json.loads(gate.BASELINE.read_text(encoding="utf-8"))
    check("the baseline permits no replacements",
          baseline.get("replaced_allowed") == 0, str(baseline.get("replaced_allowed")))
    check("and declares every leaky file with both counts",
          all({"env_set", "added"} <= set(v)
              for v in baseline.get("leaky_files", {}).values()),
          str(baseline.get("leaky_files")))


def test_a_collection_time_leak_is_seen() -> None:
    """The gap that cost a full-suite abort.

    The plugin originally hooked only `pytest_runtest_protocol`. Pytest
    imports every test module during *collection*, before the first such
    hook fires — so every module-scope `os.environ[...] = ...` and
    `sys.modules[...] = MagicMock()` in the repository had already
    happened when the plugin took its first snapshot, and sat inside the
    baseline it measured against. It reported `{}` for a file whose very
    first lines poison the interpreter.

    Run as a real pytest subprocess against a two-file fixture: the
    behaviour under test is a pytest hook ordering, and asserting on it
    any other way would be asserting on my model of pytest rather than
    on pytest.
    """
    scenario("collection-time leak is seen")
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "test_aaa_leaker.py").write_text(
            "import os, sys\n"
            "from unittest.mock import MagicMock\n"
            "os.environ['KAI_ISO_PROBE'] = 'set-at-import'\n"
            "sys.modules['kai_iso_probe_stub'] = MagicMock()\n"
            "def test_ok():\n    assert True\n", encoding="utf-8")
        (root / "test_bbb_clean.py").write_text(
            "def test_ok():\n    assert True\n", encoding="utf-8")
        report = root / "report.json"
        env = dict(os.environ)
        env["KAI_ISOLATION_REPORT"] = str(report)
        env["PYTHONPATH"] = str(REPO)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(root), "-q",
             "-p", "scripts.security.isolation_plugin", "-p", "no:cacheprovider"],
            capture_output=True, text=True, env=env, cwd=str(root), timeout=180)
        check("the fixture suite itself passes", proc.returncode == 0,
              proc.stdout[-400:])
        check("a report was written", report.exists(), proc.stdout[-400:])
        found = json.loads(report.read_text(encoding="utf-8")) if report.exists() else {}
        leaker = next((v for k, v in found.items() if "aaa_leaker" in k), None)
        check("the leaking file is reported", leaker is not None, str(found))
        check("its import-time env write is seen",
              bool(leaker) and "KAI_ISO_PROBE" in leaker.get("env_set", []),
              str(leaker))
        check("its import-time module stub is seen",
              bool(leaker) and "kai_iso_probe_stub" in leaker.get("added", []),
              str(leaker))
        check("the clean file is not blamed",
              not any("bbb_clean" in k for k in found), str(found))


def test_one_file_loaded_twice_is_not_a_replacement() -> None:
    """`dashboard/app.py` and `scripts/../dashboard/app.py` are one file.

    Compared as raw strings they look like a module being swapped, and
    three of the first five findings the widened plugin produced were
    exactly that. A detector with false positives gets somebody to
    "fix" correct code, so the path is normalised.
    """
    scenario("two spellings of one path")
    import types
    here = str(REPO / "conftest.py")
    detour = str(REPO / "scripts" / ".." / "conftest.py")
    a, b = types.ModuleType("m"), types.ModuleType("m")
    a.__spec__ = type("S", (), {"origin": here})()
    b.__spec__ = type("S", (), {"origin": detour})()
    check("the two spellings fingerprint identically",
          plugin._fingerprint(a) == plugin._fingerprint(b),
          f"{plugin._fingerprint(a)} vs {plugin._fingerprint(b)}")


def run_all() -> None:
    test_a_replaced_module_fails()
    test_a_replacement_fails_even_when_declared()
    test_the_replacement_message_names_the_file()
    test_declared_leakage_at_its_baseline_passes()
    test_leakage_that_shrinks_passes()
    test_leakage_that_grows_fails()
    test_an_undeclared_leaky_file_fails()
    test_a_clean_file_is_not_reported()
    test_a_missing_report_is_a_failure_not_a_pass()
    test_an_empty_report_against_a_declared_baseline_fails()
    test_write_baseline_refuses_to_record_a_replacement()
    test_the_plugin_tells_a_swap_from_an_import()
    test_the_real_repository_replaces_nothing()
    test_a_collection_time_leak_is_seen()
    test_one_file_loaded_twice_is_not_a_replacement()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Test Isolation Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
