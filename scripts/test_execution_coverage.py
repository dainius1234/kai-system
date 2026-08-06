"""Tests for `report_execution_coverage` — measuring what never runs.

Ten defects were found on 2026-08-06 and every one lived in code that
had **never executed**. Not one was code that used to work and broke.
So the remaining exposure sits wherever execution has not reached, and
this report measures that surface rather than guessing at it.

The report's own first run got it wrong, which is why these tests exist
in the shape they do. It took every non-flag token after `up -d` as a
service name, and the minimal bring-up is

    docker compose -f … up -d --build \\
      2>&1 | tee /tmp/bringup.log

so it read the trailing `\\` as a service, concluded the step named one
service, and reported **3** covered instead of **15**. That is exactly
the defect `check_compose_env` was given fail-closed handling for
earlier the same day — a line continuation parsed as a name —
reproduced by me hours later in a new file.

It was caught because the number disagreed with a hand count. An
instrument is not trusted because it is new.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import report_execution_coverage as rep  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 8
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


def tree(profile_body: str, workflow_run: str):
    """A synthetic repo: one compose profile and one workflow."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "docker-compose.minimal.yml").write_text(profile_body)
    wf = root / ".github" / "workflows"
    wf.mkdir(parents=True)
    (wf / "core-tests.yml").write_text(
        "jobs:\n  test:\n    steps:\n      - name: up\n        run: |\n"
        + "".join(f"          {ln}\n" for ln in workflow_run.splitlines()))
    return tmp, root


PROFILE = """services:
  alpha:
    build: {context: ., dockerfile: alpha/Dockerfile}
  beta:
    build: {context: ., dockerfile: beta/Dockerfile}
  gated:
    profiles: ["extras"]
    build: {context: ., dockerfile: gated/Dockerfile}
"""


# ── the defect the report shipped with ───────────────────────────────

def test_a_line_continuation_is_not_a_service_name() -> None:
    scenario("continuation not a service")
    tmp, root = tree(PROFILE,
                     "docker compose -f docker-compose.minimal.yml up -d "
                     "--build \\\n2>&1 | tee /tmp/bringup.log")
    with tmp:
        started = rep.started_by_ci(root)
        check("the backslash is not counted as a service",
              "\\" not in started, str(started))
        check("a bare `up` starts every ungated service",
              started == {"alpha", "beta"}, str(started))
        check("and does not start the gated one",
              "gated" not in started, str(started))


def test_a_named_subset_starts_only_those() -> None:
    """Two of the three real steps bring up three services each rather
    than a whole profile. Reading those as "the profile is covered"
    overstates coverage by nineteen services."""
    scenario("named subset")
    tmp, root = tree(PROFILE,
                     "docker compose -f docker-compose.minimal.yml up -d alpha")
    with tmp:
        started = rep.started_by_ci(root)
        check("only the named service counts", started == {"alpha"},
              str(started))


def test_flags_are_not_services() -> None:
    scenario("flags ignored")
    tmp, root = tree(PROFILE,
                     "docker compose -f docker-compose.minimal.yml up -d "
                     "--build --wait --quiet-pull")
    with tmp:
        started = rep.started_by_ci(root)
        check("no flag became a service name",
              started == {"alpha", "beta"}, str(started))


def test_a_gated_service_named_explicitly_does_count() -> None:
    """`profiles:` means opt-in, not unreachable. If a step asks for it
    by name it has been exercised."""
    scenario("gated but named")
    tmp, root = tree(PROFILE,
                     "docker compose -f docker-compose.minimal.yml up -d gated")
    with tmp:
        check("it counts as started",
              rep.started_by_ci(root) == {"gated"},
              str(rep.started_by_ci(root)))


# ── the split that drives prioritisation ─────────────────────────────

def test_never_run_is_split_by_deployment_default() -> None:
    """A service that boots on a real deployment with nothing having
    exercised it is a different risk from one a deployment only gets by
    asking for it. Collapsing them would make the number useless for
    deciding anything."""
    scenario("split by default")
    tmp, root = tree(PROFILE,
                     "docker compose -f docker-compose.minimal.yml up -d alpha")
    with tmp:
        defaults, optin, built, started = rep.survey(root)
        check("beta is a never-run default", defaults == ["beta"],
              str(defaults))
        check("gated is a never-run opt-in", optin == ["gated"], str(optin))
        check("the denominator counts built services", built == 3, str(built))
        check("and one was started", started == 1, str(started))


def test_a_service_with_no_build_is_not_in_the_denominator() -> None:
    """`redis:7-alpine` is somebody else's code. This measures whether
    *our* images have ever run."""
    scenario("image-only excluded")
    body = PROFILE + "  redis:\n    image: redis:7-alpine\n"
    tmp, root = tree(body, "docker compose -f docker-compose.minimal.yml up -d")
    with tmp:
        _, _, built, _ = rep.survey(root)
        check("redis is not counted", built == 3, str(built))


# ── I-1 ──────────────────────────────────────────────────────────────

def test_an_absent_workflow_reports_nothing_started() -> None:
    """Not a crash, and not silent full coverage either — the honest
    direction for a *coverage* number is to claim less, not more."""
    scenario("absent workflow")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docker-compose.minimal.yml").write_text(PROFILE)
        check("nothing is claimed as started",
              rep.started_by_ci(root) == set(), "")
        defaults, optin, built, started = rep.survey(root)
        check("so every built service is never-run", started == 0, str(started))
        check("and the denominator is still real", built == 3, str(built))


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_number_is_real() -> None:
    scenario("repository measured")
    defaults, optin, built, started = rep.survey()
    check("a real number of images exists", built > 40, str(built))
    check("CI starts some of them", started > 5, str(started))
    check("and not all of them — this is the finding",
          started < built, f"{started}/{built}")
    check("the two categories account for every never-run service",
          len(defaults) + len(optin) == built - started,
          f"{len(defaults)}+{len(optin)} vs {built - started}")


def run_all() -> None:
    test_a_line_continuation_is_not_a_service_name()
    test_a_named_subset_starts_only_those()
    test_flags_are_not_services()
    test_a_gated_service_named_explicitly_does_count()
    test_never_run_is_split_by_deployment_default()
    test_a_service_with_no_build_is_not_in_the_denominator()
    test_an_absent_workflow_reports_nothing_started()
    test_the_repository_number_is_real()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Execution Coverage Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
