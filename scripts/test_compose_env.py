"""Tests for `check_compose_env` — the warning that printed all day.

`docker-compose.minimal.yml` declares `POSTGRES_PASSWORD: ${DB_PASSWORD}`
with no default, which is correct — a default password in a shipped
compose file is what `check_secret_fallbacks` forbids. So CI must supply
one, and the bring-up step did not; the only step that set it was the
sovereign boot, 150 lines further down. postgres refuses to initialise
with an empty superuser password, so every dependent failed with
`dependency failed to start`.

Compose said so on every invocation:

    The "DB_PASSWORD" variable is not set. Defaulting to a blank string.

Read past as noise, all day. A warning nobody reads is worth what no
warning is worth.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_compose_env as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 12
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


def test_a_variable_with_a_default_is_not_required() -> None:
    """`${X:-fallback}` is a deliberate statement that blank is fine."""
    scenario("default not required")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.yml"
        f.write_text("services:\n  a:\n    environment:\n"
                     "      X: ${WITH_DEFAULT:-ok}\n", encoding="utf-8")
        check("not required", gate.required_by(f, []) == set(),
              str(gate.required_by(f, [])))


def test_a_variable_without_a_default_is_required() -> None:
    scenario("no default required")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.yml"
        f.write_text("services:\n  a:\n    environment:\n"
                     "      X: ${NEEDED}\n", encoding="utf-8")
        check("required", gate.required_by(f, []) == {"NEEDED"},
              str(gate.required_by(f, [])))


def test_naming_services_narrows_the_requirement() -> None:
    """`up -d postgres` starts postgres, not the profile. Ignoring the
    service list produced five findings for Grafana, Tailscale and Vault
    variables belonging to services that step never starts."""
    scenario("service scoping")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.yml"
        f.write_text("services:\n"
                     "  postgres:\n    environment:\n      P: ${DB_PASSWORD}\n"
                     "  vault:\n    environment:\n      V: ${VAULT_TOKEN}\n",
                     encoding="utf-8")
        check("postgres alone needs only its own",
              gate.required_by(f, ["postgres"]) == {"DB_PASSWORD"},
              str(gate.required_by(f, ["postgres"])))
        check("vault alone needs only its own",
              gate.required_by(f, ["vault"]) == {"VAULT_TOKEN"},
              str(gate.required_by(f, ["vault"])))
        check("the whole profile needs both",
              gate.required_by(f, []) == {"DB_PASSWORD", "VAULT_TOKEN"},
              str(gate.required_by(f, [])))


def test_dependencies_are_started_too() -> None:
    """`up -d app` starts what app depends on, so those count."""
    scenario("depends_on closure")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.yml"
        f.write_text("services:\n"
                     "  app:\n    depends_on: [db]\n"
                     "  db:\n    environment:\n      P: ${DB_PASSWORD}\n",
                     encoding="utf-8")
        check("the dependency's variable is required",
              gate.required_by(f, ["app"]) == {"DB_PASSWORD"},
              str(gate.required_by(f, ["app"])))


def test_only_defined_services_count_as_service_names() -> None:
    """The bug its own calibration caught. The bring-up line ends with a
    `\\` continuation; treating that as a service name matched nothing,
    scoped the check to nothing, and passed the exact defect this gate
    was written for."""
    scenario("shell tokens are not services")
    defined = {"postgres", "redis"}
    check("a continuation is not a service",
          gate.named_services("-d --build \\", defined) == [], "")
    check("a pipe target is not a service",
          gate.named_services("-d 2>&1 | tee /tmp/x.log", defined) == [], "")
    check("real names are kept",
          gate.named_services("-d postgres redis", defined) ==
          ["postgres", "redis"], "")


def test_recognising_nothing_means_the_whole_profile() -> None:
    """Fail closed: over-report rather than under-report."""
    scenario("unknown scope is whole profile")
    check("no recognised names -> empty list -> whole profile",
          gate.named_services("-d --build", {"a"}) == [], "")


def test_a_step_missing_a_variable_is_reported() -> None:
    scenario("missing variable reported")
    findings, steps, _ = gate.audit()
    check("the repository passes today", findings == [], str(findings))
    check("and bring-up steps were actually found", steps > 0, str(steps))


def test_an_unparseable_workflow_is_a_finding() -> None:
    """I-1. A workflow that will not parse had its steps examined by
    nobody, which is not the same as having no problems."""
    scenario("unparseable workflow")
    original = gate.WORKFLOWS
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "broken.yml").write_text("a: [unclosed\n", encoding="utf-8")
        try:
            gate.WORKFLOWS = root
            findings, _, _ = gate.audit()
        finally:
            gate.WORKFLOWS = original
    check("it is reported", len(findings) == 1, str(findings))
    check("and says unreadable",
          findings and "unreadable" in findings[0], str(findings))


def test_the_real_workflows_are_all_covered() -> None:
    scenario("real workflows covered")
    _, steps, workflows = gate.audit()
    check("every workflow was read", workflows >= 9, str(workflows))
    check("and the bring-up steps were found", steps >= 3, str(steps))


# ── the other direction: set, but never read ─────────────────────────

def test_a_variable_the_profile_never_reads_is_reported() -> None:
    """The defect this direction was added for.

    `MEMU_ALLOW_FAKE_EMBEDDINGS: "true"` was set by four CI bring-up
    steps and named by only one of the three profiles. Compose passes a
    variable into a container solely when the service asks for it, so
    for `full` and `sovereign` it reached nothing — while looking, in
    the workflow, exactly like configuration. memu-core raises at import
    when sentence-transformers cannot load and that flag is not true, so
    the container died before listening and compose reported only
    `is unhealthy`.
    """
    scenario("unread variable reported")
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "c.yml"
        f.write_text("services:\n  a:\n    environment:\n"
                     "      KNOWN: \"${KNOWN:-x}\"\n")
        found = gate.unread_by_compose(f, {"KNOWN": "1", "IGNORED": "1"})
        check("the unread one is named", found == ["IGNORED"], str(found))
        check("and the one it does read is not",
              "KNOWN" not in found, str(found))


def test_a_missing_compose_file_is_a_finding_not_a_pass() -> None:
    """I-1. A profile that is not there was not checked, and reporting
    nothing wrong about a file nobody read is the defect itself."""
    scenario("missing profile refuses")
    found = gate.unread_by_compose(Path("/nonexistent/c.yml"), {"X": "1"})
    check("it refuses rather than passing", found != [], str(found))
    check("and says nothing could be checked",
          any("nothing here could be checked" in f for f in found), str(found))


def test_the_repository_reads_every_variable_its_steps_set() -> None:
    scenario("repository passes both directions")
    findings, steps, _ = gate.audit()
    check("no findings", findings == [], str(findings))
    check("across a real number of bring-up steps", steps >= 4, str(steps))


def run_all() -> None:
    test_a_variable_the_profile_never_reads_is_reported()
    test_a_missing_compose_file_is_a_finding_not_a_pass()
    test_the_repository_reads_every_variable_its_steps_set()
    test_a_variable_with_a_default_is_not_required()
    test_a_variable_without_a_default_is_required()
    test_naming_services_narrows_the_requirement()
    test_dependencies_are_started_too()
    test_only_defined_services_count_as_service_names()
    test_recognising_nothing_means_the_whole_profile()
    test_a_step_missing_a_variable_is_reported()
    test_an_unparseable_workflow_is_a_finding()
    test_the_real_workflows_are_all_covered()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Compose Env Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
