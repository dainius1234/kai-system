#!/usr/bin/env python3
"""Calibration for the declared-prerequisite gate.

The rule under test is the doctrine's spine — *nothing is true because it
was true last time* — in its one decidable form: **a declared condition
must be in force where the service is actually started.**

A gate for that has three ways to be wrong, and all three are exercised
here because two of them were live defects in the first draft:

  * **too narrow** — it resolved a site's service against only the
    services that HAVE conditions, so a service with none read as
    unresolvable instead of trivially clean;
  * **too generous** — it printed "none declared" for a compose file it
    could not resolve. Unknown wearing the clothes of clean, written
    twenty minutes after the principle it was implementing;
  * **too wide** — banning `--no-deps` outright would fail the Stage-1
    preflight and the model-readiness probe, both of which are correct.

Fixtures are built on disk and scanned, rather than the functions being
stubbed, because the defect being guarded lives in reading real files.
The live tree is then asserted against too — a check calibrated only on
fixtures has never met the thing it polices.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import check_declared_prerequisites as dp  # noqa: E402

GATE = REPO / "scripts" / "security" / "check_declared_prerequisites.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 5
executed: list[str] = []

COMPOSE = """\
services:
  puller:
    image: busybox
  worker:
    image: busybox
    depends_on:
      puller:
        condition: service_completed_successfully
  loner:
    image: busybox
"""


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def tree(tmp: Path, invocation: str, compose_name="docker-compose.full.yml",
         preamble="") -> Path:
    """A miniature repo: one compose file, one workflow that runs it."""
    root = tmp / f"t{len(list(tmp.iterdir()))}"
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / compose_name).write_text(COMPOSE)
    (root / ".github" / "workflows" / "job.yml").write_text(
        "jobs:\n  a:\n    steps:\n      - run: |\n"
        f"{preamble}          {invocation}\n")
    return root


def verdict(root: Path, declared=()) -> tuple[int, str]:
    """Run the shipped main() against a fixture, capturing what it says."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = dp.main(root=root, declared=tuple(declared))
    return code, buf.getvalue()


def test_a_bypassed_condition_is_found() -> None:
    scenario("undeclared bypass is found")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = tree(tmp, "docker compose -f docker-compose.full.yml run "
                         "--rm --no-deps -T worker echo hi")
        code, out = verdict(root)
        check("an undeclared bypass FAILS", code == 1, out)
        check("and names the condition it skipped",
              "puller: service_completed_successfully" in out, out)
        check("and says what a declaration must carry",
              "reason, owner, review date" in out, out)


def test_a_declared_bypass_passes() -> None:
    scenario("declared bypass passes")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        root = tree(tmp, "docker compose -f docker-compose.full.yml run "
                         "--rm --no-deps -T worker echo hi")
        good = dp.Bypass(file=".github/workflows/job.yml", service="worker",
                         dependency="puller", reason="fixture",
                         owner="orion", review_by="2099-01-01")
        code, out = verdict(root, [good])
        check("a declared bypass PASSES", code == 0, out)
        check("and is counted as accounted for", "accounted for" in out, out)
        # the declaration must be specific: a different dependency is not it
        wrong = dp.Bypass(file=".github/workflows/job.yml", service="worker",
                          dependency="something-else", reason="fixture",
                          owner="orion", review_by="2099-01-01")
        code, out = verdict(root, [wrong])
        check("a declaration for the WRONG dependency does not cover it",
              code == 1, out)


def test_the_scope_is_not_too_narrow_or_too_wide() -> None:
    scenario("scope is exactly right")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # TOO WIDE: --no-deps on a service with no conditions is fine
        root = tree(tmp, "docker compose -f docker-compose.full.yml run "
                         "--rm --no-deps -T loner echo hi")
        code, out = verdict(root)
        check("--no-deps on a service with no conditions is CLEAN",
              code == 0, out)
        check("and it is not reported as unresolved",
              "UNRESOLVED" not in out, out)
        # TOO NARROW: no --no-deps at all is not a site
        root = tree(tmp, "docker compose -f docker-compose.full.yml run "
                         "--rm -T worker echo hi")
        code, out = verdict(root)
        check("an invocation WITHOUT --no-deps is not a bypass", code == 0,
              out)
        check("and the site count reflects that",
              "--no-deps execution site(s): 0" in out, out)


def test_unknown_is_never_clean() -> None:
    scenario("unknown is not clean")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # a compose file named by a variable that is never assigned
        root = tree(tmp, 'docker compose -f "$MYSTERY" run --rm --no-deps '
                         '-T worker echo hi')
        code, out = verdict(root)
        check("an unresolvable compose file FAILS", code == 1, out)
        check("it is reported as UNRESOLVED", "UNRESOLVED" in out, out)
        check("and says so in as many words",
              "Unknown is NOT clean" in out, out)
        check("it is NOT folded into the clean total",
              "PASS:" not in out, out)

        # ...but a plain literal assignment IS resolvable
        root = tree(tmp, 'docker compose -f "$COMPOSE_FILE" run --rm '
                         '--no-deps -T worker echo hi',
                    preamble='          COMPOSE_FILE=docker-compose.full.yml\n')
        code, out = verdict(root)
        check("a literal assignment resolves rather than refusing",
              "UNRESOLVED" not in out, out)
        check("and the bypass behind it is then found", code == 1, out)
        check("literals_in reads a plain assignment",
              dp.literals_in('X="a.yml"\n').get("X") == "a.yml")
        check("and refuses one built from another variable",
              "Y" not in dp.literals_in('Y="$Z/a.yml"\n'))


def test_it_has_met_the_tree_it_polices() -> None:
    """A gate calibrated only on fixtures has never met the real thing."""
    scenario("live tree")
    r = subprocess.run([sys.executable, str(GATE)], capture_output=True,
                       text=True)
    check("the shipped entry point runs", "inspected:" in r.stdout, r.stderr)
    check("the live denominator is derived, not typed",
          len(dp.declarations()) >= 3 and len(dp.all_services()) > 10,
          f"{len(dp.declarations())} compose file(s), "
          f"{len(dp.all_services())} service(s)")
    # The Stage-1 bypasses are DECLARED, so they must never appear as
    # findings. That holds whatever else the tree contains.
    undeclared = r.stdout.split("UNDECLARED")[-1] if "UNDECLARED" in r.stdout \
        else ""
    check("the declared Stage-1 bypasses are not reported as findings",
          "stage1-replay.yml" not in undeclared, undeclared[:300])
    # Deliberately NOT asserted: that a particular finding is present.
    # The first draft required verify_identity_in_containers.sh to appear,
    # which would have failed the suite the moment that finding was
    # judged and declared -- a test that depends on an open finding
    # staying open pressures the operator to leave it open. What IS
    # asserted is the structural property: whatever is reported carries
    # the condition it skipped, so a finding can never be a bare name.
    for line in undeclared.splitlines():
        if line.strip().startswith("- ") and "--no-deps skips" in line:
            check("every reported finding names the condition it skipped",
                  ":" in line.split("--no-deps skips")[-1], line)


def run_all() -> None:
    test_a_bypassed_condition_is_found()
    test_a_declared_bypass_passes()
    test_the_scope_is_not_too_narrow_or_too_wide()
    test_unknown_is_never_clean()
    test_it_has_met_the_tree_it_polices()
    print(f"  inspected: {EXPECTED_SCENARIOS} declared-prerequisite "
          f"scenario(s) across 1 gate")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Declared Prerequisites Calibration: {passed} passed, "
          f"{failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
