"""Tests for the two guards the first full-profile bring-up needed.

On 2026-08-06 the full profile was brought up for the first time ever
and died:

    level=warning msg="secret file kai-system_db_password does not exist"
    Container kai-system-backup-service-1  Error response from daemon:
      invalid mount config for type "bind": bind source path does not
      exist: …/runtime-secrets/db_password

Two separate defects, both in code that had never executed — the day's
only pattern:

  1. `docker-compose.full.yml` declares three file-backed Docker secrets
     and nothing creates them. A comment tells a human to. Five
     service-secret bindings across four services could not have worked.

  2. The bring-up guard was `grep -q 'variable is not set'`, written out
     separately in two steps. It is *named* for the class "compose
     warned about something missing" and matched one member of it, so
     the warning above passed in silence and the failure arrived one
     line later as a mount error. Sixteenth venue of the systemic
     finding.

The empty-log case below is here because the first draft of
`assert_clean_bringup` failed it: it printed "WARNING: the log was
empty" and returned 0. A bring-up producing no output at all was
reported as a clean bring-up. Found by calibration — pointing the
detector at input whose answer was known — not by review.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ci import assert_clean_bringup as guard   # noqa: E402
from scripts.ci import make_dev_secrets as secrets     # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 10
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


def log(body: str) -> Path:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    tmp.write(body)
    tmp.close()
    return Path(tmp.name)


# ── assert_clean_bringup ─────────────────────────────────────────────

def test_the_warning_that_slipped_past_the_old_guard() -> None:
    scenario("secret-file warning caught")
    path = log('time="2026-08-06T12:46:48Z" level=warning '
               'msg="secret file kai-system_db_password does not exist"\n'
               ' Container kai-system-tool-gate-1  Creating\n')
    findings, _, _ = guard.audit([path])
    check("it is reported", len(findings) == 1, str(findings))
    check("and names the missing file",
          findings and "db_password" in findings[0], str(findings))


def test_the_warning_the_old_guard_did_catch_still_fails() -> None:
    """Widening a rule must not drop what it already covered."""
    scenario("variable-not-set still caught")
    path = log('WARN[0000] The "DB_PASSWORD" variable is not set. '
               'Defaulting to a blank string.\n')
    findings, _, _ = guard.audit([path])
    check("still reported", len(findings) == 1, str(findings))


def test_the_obsolete_version_warning_fails() -> None:
    """`version:` was removed from all three profiles, so this warning
    should no longer appear. It had printed on every compose call in
    every run — a signal that repeats forever is one nobody reads."""
    scenario("obsolete version caught")
    path = log('time="…" level=warning msg="the attribute `version` is '
               'obsolete, it will be ignored"\n')
    findings, _, _ = guard.audit([path])
    check("reported rather than tolerated", len(findings) == 1, str(findings))


def test_an_ordinary_bring_up_passes() -> None:
    """The inverse defect matters more: a guard that fails on healthy
    output sends people to break working code."""
    scenario("clean bring-up passes")
    path = log(" Container kai-system-postgres-1  Created\n"
               " Container kai-system-redis-1  Healthy\n"
               " Container kai-system-memu-core-1  Started\n")
    findings, lines, read = guard.audit([path])
    check("nothing reported", findings == [], str(findings))
    check("and it says how much it read", (lines, read) == (3, 1),
          f"{lines}, {read}")


def test_a_declared_benign_warning_passes() -> None:
    scenario("benign warning allowed")
    path = log(" Container x Created\n"
               "Node.js 20 is deprecated. The following actions ...\n")
    findings, _, _ = guard.audit([path])
    check("the declared one is skipped", findings == [], str(findings))


# ── I-1, both shapes ─────────────────────────────────────────────────

def test_an_empty_log_is_a_finding() -> None:
    """The defect the first draft shipped with."""
    scenario("empty log refuses")
    findings, lines, read = guard.audit([log("")])
    check("it fails rather than passing", findings != [], str(findings))
    check("and says the redirect did not capture",
          any("did not capture" in f for f in findings), str(findings))
    check("with a zero denominator", lines == 0, str(lines))


def test_a_missing_log_is_a_finding() -> None:
    scenario("missing log refuses")
    findings, _, read = guard.audit([Path("/nonexistent/bringup.log")])
    check("it fails rather than passing", findings != [], str(findings))
    check("and nothing was read", read == 0, str(read))


# ── make_dev_secrets ─────────────────────────────────────────────────

def test_secret_names_are_read_from_compose() -> None:
    """Derived, not listed. Three `echo` lines in the workflow would be
    the list-beside-the-thing pattern created deliberately."""
    scenario("secrets derived")
    declared = secrets.declared_secrets(
        secrets.REPO / "docker-compose.full.yml")
    names = {n for n, _ in declared}
    check("every declared secret is found",
          names == {"hmac_secret", "db_password", "bridge_secret"},
          str(names))
    # `Path` normalises the leading `./` away, so assert on the resolved
    # directory rather than on the literal text of the declaration.
    check("and the ${SECRETS_DIR:-…} default is resolved to a real dir",
          all(p.parent.name == "runtime-secrets" for _, p in declared),
          str([str(p) for _, p in declared]))
    check("with no unexpanded variable left in it",
          not any("$" in str(p) for _, p in declared),
          str([str(p) for _, p in declared]))


def test_provisioning_creates_every_declared_secret() -> None:
    scenario("secrets created")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        created, kept = secrets.provision(
            secrets.REPO / "docker-compose.full.yml", root)
        check("all three created", len(created) == 3, str(created))
        check("none pre-existing", kept == [], str(kept))
        body = (root / "runtime-secrets" / "db_password").read_text()
        check("and the value says what it is",
              "not-a-real-secret" in body, body.strip())


def test_an_existing_secret_is_never_overwritten() -> None:
    """This writes files named `db_password`. On a machine holding real
    secret material, leaving it alone is the only acceptable behaviour."""
    scenario("existing secret kept")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        real = root / "runtime-secrets" / "db_password"
        real.parent.mkdir(parents=True)
        real.write_text("a-real-password-that-must-survive\n")
        created, kept = secrets.provision(
            secrets.REPO / "docker-compose.full.yml", root)
        check("it is reported as kept", kept == ["runtime-secrets/db_password"],
              str(kept))
        check("and its contents are untouched",
              real.read_text().strip() == "a-real-password-that-must-survive",
              real.read_text())
        check("while the others are created", len(created) == 2, str(created))


def run_all() -> None:
    test_the_warning_that_slipped_past_the_old_guard()
    test_the_warning_the_old_guard_did_catch_still_fails()
    test_the_obsolete_version_warning_fails()
    test_an_ordinary_bring_up_passes()
    test_a_declared_benign_warning_passes()
    test_an_empty_log_is_a_finding()
    test_a_missing_log_is_a_finding()
    test_secret_names_are_read_from_compose()
    test_provisioning_creates_every_declared_secret()
    test_an_existing_secret_is_never_overwritten()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Bring-up Guard Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
