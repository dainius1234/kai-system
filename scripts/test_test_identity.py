"""Tests for `check_test_identity` — a suite that ran as the wrong user.

`scripts/security_fuzz_upload.py` opened with

    _os.environ.setdefault("KAI_DASHBOARD_ROLE", "keeper")

which reads as *"this suite runs as keeper"*. It does not. `setdefault`
means **unless somebody else already decided**, and on 2026-08-06
somebody did: `KAI_DASHBOARD_ROLE: operator` went into `core-tests.yml`
at job scope so the live smoke could authenticate.

`/api/upload` requires `Scope.WRITE_EXTERNAL` — `keeper` has it,
`operator` does not. Eight of fourteen tests stopped asserting upload
validation and started asserting authorisation:

    FAILED test_one_byte_over_limit_returns_413
    FAILED test_oversized_payload_returns_413
    FAILED test_no_filename_rejected               … five more

Neither the tests nor the endpoint were wrong. The suite's **identity**
was chosen by an environment variable it did not control, so what it
verified depended on who ran it — and under CI it had only ever run at
whatever privilege happened to be ambient.

The systemic finding aimed at a test rather than a check: its scope —
*which privilege am I exercising* — was inherited rather than stated,
and it reported success over a question it never asked.

Calibrated against the real file: `git show
1689e19:scripts/security_fuzz_upload.py` gives exactly three findings.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_test_identity as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 9
executed: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def found(src: str) -> list:
    return gate.findings_in(ast.parse(src), "t.py")


# ── the defect itself ────────────────────────────────────────────────

def test_the_real_defect_is_caught() -> None:
    scenario("real defect caught")
    src = ('import os as _os\n'
           '_os.environ.setdefault("KAI_DASHBOARD_ROLE", "keeper")\n')
    f = found(src)
    check("it is reported", len(f) == 1, str(f))
    check("the variable is named",
          f and "KAI_DASHBOARD_ROLE" in f[0], str(f))
    check("and the remedy is assignment",
          f and "environ['KAI_DASHBOARD_ROLE'] = …" in f[0], str(f))


def test_assignment_passes() -> None:
    scenario("assignment passes")
    src = ('import os\nos.environ["KAI_DASHBOARD_ROLE"] = "keeper"\n')
    check("nothing reported", found(src) == [], str(found(src)))


def test_every_identity_word_is_recognised() -> None:
    """Knowing only ROLE would leave TOKEN and SECRET ambient — a gate
    whose scope is smaller than its name, which is the thing this whole
    programme exists to stop."""
    scenario("all identity words")
    for name in ("KAI_DASHBOARD_TOKEN", "KAI_SERVICE_TOKEN", "SOME_ROLE",
                 "X_IDENTITY", "MEMU_HMAC_KEY", "APP_SECRET",
                 "PRINCIPALS_JSON", "AUTH_MODE", "API_KEY"):
        src = f'import os\nos.environ.setdefault("{name}", "x")\n'
        check(f"{name} is identity-bearing", len(found(src)) == 1,
              f"{name}: {found(src)}")


def test_a_path_or_flag_may_still_setdefault() -> None:
    """A test that tolerates an ambient `/tmp` path is not thereby
    testing something different. Flagging these would report a defect in
    code that works — 44 such calls exist in this tree."""
    scenario("paths and flags allowed")
    for name in ("LEDGER_PATH", "NONCE_CACHE_PATH", "VECTOR_STORE",
                 "MEMU_ALLOW_FAKE_EMBEDDINGS", "CACHING",
                 "TELEMETRY_DISABLED"):
        src = f'import os\nos.environ.setdefault("{name}", "x")\n'
        check(f"{name} is allowed", found(src) == [],
              f"{name}: {found(src)}")


def test_a_setdefault_on_something_else_is_ignored() -> None:
    """`d.setdefault("token", …)` on a plain dict is not the environment."""
    scenario("non-environ ignored")
    src = ('d = {}\nd.setdefault("KAI_DASHBOARD_TOKEN", "x")\n')
    check("nothing reported", found(src) == [], str(found(src)))


def test_a_non_literal_key_is_not_guessed_at() -> None:
    """`environ.setdefault(name, …)` names something this gate cannot
    read. Reporting on a guess is worse than saying nothing."""
    scenario("dynamic key skipped")
    src = ('import os\nname = "KAI_DASHBOARD_ROLE"\n'
           'os.environ.setdefault(name, "keeper")\n')
    check("nothing reported", found(src) == [], str(found(src)))


# ── I-1: zero inputs is not a pass ───────────────────────────────────

def test_a_tree_with_no_tests_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, calls, files = gate.audit(Path(tmp))
        check("it fails rather than passing", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", (calls, files) == (0, 0),
              f"{calls}, {files}")


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, calls, files = gate.audit()
    check("no ambient privilege", findings == [], str(findings))
    check("across a real number of test files", files > 100, str(files))
    check("and setdefault is still in use for benign things",
          calls > 10, str(calls))


def test_conftest_is_surveyed() -> None:
    """It sets the environment every suite inherits, so an ambient
    identity there reaches all of them at once."""
    scenario("conftest surveyed")
    names = [p.name for p in gate.test_files()]
    check("conftest.py is in the denominator", "conftest.py" in names,
          str(names[:5]))
    check("and so are fuzz suites named without a test_ prefix",
          any("_fuzz_" in n for n in names), str([n for n in names if "fuzz" in n]))


def run_all() -> None:
    test_the_real_defect_is_caught()
    test_assignment_passes()
    test_every_identity_word_is_recognised()
    test_a_path_or_flag_may_still_setdefault()
    test_a_setdefault_on_something_else_is_ignored()
    test_a_non_literal_key_is_not_guessed_at()
    test_a_tree_with_no_tests_refuses()
    test_the_repository_passes_today()
    test_conftest_is_surveyed()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Test Identity Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
