"""Tests for `live_smoke` — the live check that can now say no.

Its predecessor, `scripts/test_core_integration.py`, ended in a bare
`return 0` after catching every exception it could raise. With the whole
stack down it printed eleven failures and exited 0. So the first thing
this file pins is the property that was missing: **it must be able to
fail**, and the failure must come from the probes rather than from a
constant.

Every scenario drives a synthetic compose document and a fake command
runner, so none of them needs Docker, a network, or the repository to be
in any particular state. The one that reads the real tree says so.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ci import live_smoke as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 17
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


def _compose(**services) -> str:
    """A compose document whose services declare healthchecks like the real ones."""
    lines = ["services:"]
    for name, port in services.items():
        real = name.replace("_", "-")
        lines.append(f"  {real}:")
        lines.append("    build: ./x")
        if port is not None:
            lines.append("    healthcheck:")
            lines.append(
                f"      test: [\"CMD-SHELL\", \"python -c \\\"import urllib.request; "
                f"urllib.request.urlopen('http://localhost:{port}/health')\\\"\"]")
    return "\n".join(lines) + "\n"


class FakeRunner:
    """Answers `ps` from a health map and `exec` from a status map."""

    def __init__(self, health: dict, statuses: dict | None = None,
                 ps_code: int = 0):
        self.health = health
        self.statuses = statuses or {}
        self.ps_code = ps_code
        self.execs: list = []

    def __call__(self, argv):
        if "ps" in argv:
            if self.ps_code != 0:
                return self.ps_code, "", "cannot connect to the docker daemon"
            rows = [json.dumps({"Service": s, "Health": h})
                    for s, h in self.health.items()]
            return 0, "\n".join(rows) + "\n", ""
        if "exec" in argv:
            service = argv[argv.index("exec") + 2]
            self.execs.append(service)
            # Whether the call carries credentials is part of its
            # identity now: the dashboard is exercised twice, once
            # without and once with, and a fake that cannot tell them
            # apart cannot model a gateway at all.
            authed = "Authorization" in " ".join(str(a) for a in argv)
            key = (service, "auth" if authed else "noauth")
            status = self.statuses.get(key, self.statuses.get(service, 200))
            if status == "unreachable":
                return 1, "", "ERROR URLError"
            return 0, f"STATUS {status}\n", ""
        return 0, "", ""


def _audit(compose_text: str, runner):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "c.yml").write_text(compose_text, encoding="utf-8")
        original = gate.REPO
        try:
            gate.REPO = root
            # `now` advances past the deadline on the second call, so
            # the wait exercises its real loop without taking two
            # minutes to prove it waits two minutes.
            ticks = iter([0.0, 1e9])
            return gate.audit("c.yml", runner,
                              sleep=lambda _s: None,
                              now=lambda: next(ticks, 1e9))
        finally:
            gate.REPO = original


# ── The property the old file did not have ───────────────────────────

def test_an_unhealthy_service_fails() -> None:
    """The whole reason this file exists."""
    scenario("unhealthy fails")
    text = _compose(**{"heartbeat": 8010, "memu_core": 8001, "dashboard": 8080})
    runner = FakeRunner({"heartbeat": "healthy", "memu-core": "unhealthy",
                         "dashboard": "healthy"})
    failures, probed, _ = _audit(text, runner)
    check("it fails", len(failures) >= 1, str(failures))
    check("and names the service",
          any("memu-core" in f for f in failures), str(failures))
    check("and every service was counted", probed == 3, str(probed))


def test_a_whole_stack_down_fails() -> None:
    """The exact case the predecessor exited 0 on."""
    scenario("whole stack down fails")
    text = _compose(**{"heartbeat": 8010, "memu_core": 8001, "dashboard": 8080})
    failures, probed, exercised = _audit(text, FakeRunner({}))
    check("a stack that started nothing fails", len(failures) >= 3, str(failures))
    check("no exercise was counted as run", exercised == 0, str(exercised))
    check("and the denominator still shows what was expected",
          probed == 3, str(probed))


def test_a_healthy_stack_passes() -> None:
    scenario("healthy stack passes")
    text = _compose(**{"heartbeat": 8010, "memu_core": 8001, "dashboard": 8080})
    runner = FakeRunner(
        {"heartbeat": "healthy", "memu-core": "healthy",
         "dashboard": "healthy"},
        # A gateway that fails closed: it refuses the call with no
        # credentials and answers the one that carries them.
        {("dashboard", "noauth"): 403, ("dashboard", "auth"): 200})
    failures, probed, exercised = _audit(text, runner)
    check("no failures", failures == [], str(failures))
    check("three services probed", probed == 3, str(probed))
    check("four endpoints exercised", exercised == 4, str(exercised))


# ── Failing closed ───────────────────────────────────────────────────

def test_a_missing_profile_is_a_failure() -> None:
    """I-1. A smoke test with no stack to smoke has not passed."""
    scenario("missing profile fails")
    with tempfile.TemporaryDirectory() as tmp:
        original = gate.REPO
        try:
            gate.REPO = Path(tmp)
            failures, probed, _ = gate.audit("absent.yml", FakeRunner({}),
                                             sleep=lambda _s: None,
                                             now=lambda: 0.0)
        finally:
            gate.REPO = original
    check("absence is reported", len(failures) == 1, str(failures))
    check("and nothing is claimed as inspected", probed == 0, str(probed))


def test_a_profile_with_no_healthchecks_is_a_failure() -> None:
    """I-2. Inspecting nothing must not read the same as inspecting all."""
    scenario("zero denominator fails")
    text = _compose(**{"opaque": None})
    failures, probed, _ = _audit(text, FakeRunner({}))
    check("zero inspected is a failure", len(failures) >= 1, str(failures))
    check("and says so", any("nothing was inspected" in f for f in failures),
          str(failures))


def test_a_docker_failure_is_a_failure_not_a_pass() -> None:
    """`docker compose ps` failing means the answer is unknown."""
    scenario("docker failure fails")
    text = _compose(**{"heartbeat": 8010})
    try:
        _audit(text, FakeRunner({}, ps_code=1))
        check("it raised rather than reporting clean", False, "no exception")
    except RuntimeError as exc:
        check("it raised rather than reporting clean", True, str(exc))


def test_a_service_with_no_healthcheck_is_not_read_as_healthy() -> None:
    scenario("no-healthcheck is not healthy")
    text = _compose(**{"heartbeat": 8010})
    runner = FakeRunner({"heartbeat": ""})
    failures, _, _ = _audit(text, runner)
    check("an empty health field is a failure", len(failures) >= 1, str(failures))
    check("reported as no-healthcheck",
          any("no-healthcheck" in f for f in failures), str(failures))


# ── The deleted port map ─────────────────────────────────────────────

def test_ports_come_from_the_healthcheck() -> None:
    """The map is derived from the compose file, not typed here."""
    scenario("ports derived from healthchecks")
    import yaml
    doc = yaml.safe_load(_compose(**{"memu_core": 8001, "dashboard": 8080}))
    ports = gate.health_ports(doc)
    check("both resolve", ports == {"memu-core": 8001, "dashboard": 8080},
          str(ports))


def test_a_healthcheck_without_a_port_is_omitted_not_guessed() -> None:
    """`redis`, `postgres` and `ollama` probe with their own CLIs."""
    scenario("portless healthcheck omitted")
    import yaml
    doc = yaml.safe_load(
        "services:\n"
        "  redis:\n"
        "    image: redis:7-alpine\n"
        "    healthcheck:\n"
        "      test: [\"CMD\", \"redis-cli\", \"ping\"]\n")
    check("no port is invented", gate.health_ports(doc) == {},
          str(gate.health_ports(doc)))


def test_an_exercise_for_an_absent_service_is_reported() -> None:
    """Five of the old file's eleven probes addressed services that were
    not in the profile, and printing a line was all that happened."""
    scenario("exercise drift is reported")
    text = _compose(**{"heartbeat": 8010})       # no memu-core, no dashboard
    runner = FakeRunner({"heartbeat": "healthy"})
    failures, _, _ = _audit(text, runner)
    check("the drift is a failure, not a printed line",
          any("cannot be exercised" in f for f in failures), str(failures))


# ── The real tree ────────────────────────────────────────────────────

def test_the_real_profile_declares_the_exercised_services() -> None:
    """Reads the repository: every service in EXERCISES must exist with a
    health port, or the exercise list has drifted from the profile."""
    scenario("real profile matches the exercise list")
    import yaml
    doc = yaml.safe_load(
        (gate.REPO / "docker-compose.minimal.yml").read_text(encoding="utf-8"))
    ports = gate.health_ports(doc)
    missing = [ex.service for ex in gate.EXERCISES if ex.service not in ports]
    check("every exercised service is in the minimal profile",
          missing == [], str(missing))
    check("and the profile declares a useful number of health ports",
          len(ports) > 10, str(len(ports)))


def test_a_profile_gated_service_is_not_a_failure() -> None:
    """The defect this file's first live run committed against itself.

    It reported seventeen failures and sixteen were of this shape:

        - audio-service: declared in docker-compose.minimal.yml but not
          running — the profile did not start it

    `audio-service` is `profiles: ["sensors"]`. A bare `docker compose
    up` is *meant* not to start it. Eighteen of the minimal profile's
    services are gated that way and this file demanded all of them.

    The systemic finding inverted: the scope was larger than reality, so
    it reported failure over things that were right. That is worse than
    the usual direction — it sends people to break working code, and
    sixteen false alarms are how the one true finding
    (`dashboard: health=starting`) gets lost in the noise."""
    scenario("profile-gated not a failure")
    text = ("services:\n"
            "  core:\n"
            "    healthcheck:\n"
            "      test: [\"CMD-SHELL\", \"curl localhost:8001/health\"]\n"
            "  sensor:\n"
            "    profiles: [\"sensors\"]\n"
            "    healthcheck:\n"
            "      test: [\"CMD-SHELL\", \"curl localhost:8021/health\"]\n")
    runner = FakeRunner({"core": "healthy"})
    failures, probed, _ = _audit(text, runner)
    check("the gated service is not reported",
          not any("sensor" in f for f in failures), str(failures))
    check("and it is not counted as inspected", probed == 1, str(probed))


def test_an_ungated_service_that_is_absent_is_still_a_failure() -> None:
    """The rule must not have been widened into silence. A service with
    no `profiles:` key that the bring-up did not start is still wrong."""
    scenario("ungated absence still fails")
    text = ("services:\n"
            "  core:\n"
            "    healthcheck:\n"
            "      test: [\"CMD-SHELL\", \"curl localhost:8001/health\"]\n"
            "  missing:\n"
            "    healthcheck:\n"
            "      test: [\"CMD-SHELL\", \"curl localhost:8002/health\"]\n")
    failures, probed, _ = _audit(text, FakeRunner({"core": "healthy"}))
    check("the absent ungated service is reported",
          any("missing" in f for f in failures), str(failures))
    check("and the message says it should have been started",
          any("should have started it" in f for f in failures), str(failures))
    check("both are counted", probed == 2, str(probed))


def test_gated_services_are_identified_from_the_compose_file() -> None:
    """Derived from `profiles:`, not from a list of names."""
    scenario("gated set derived")
    doc = {"services": {"a": {"profiles": ["x"]}, "b": {},
                        "c": {"profiles": []}, "d": None}}
    got = gate.gated_services(doc)
    check("only the one with a non-empty profiles key", got == {"a"},
          str(got))


def test_the_gateway_is_proven_in_both_directions() -> None:
    """The dashboard's 503 on the first live run was not a defect — it
    was Wave 1 Track A working. With no `KAI_DASHBOARD_TOKEN`,
    `authenticate()` returns 503 "this gateway fails closed by design".

    But a probe that only ever sees a refusal cannot tell *refusing
    correctly* from *permanently broken* — and the dashboard HAD been
    permanently broken, on missing python-multipart, until this morning.
    So both directions are asserted. I-3, applied to a live service
    rather than to a gate."""
    scenario("gateway both directions")
    unauth = [e for e in gate.EXERCISES
              if e.service == "dashboard" and not e.headers]
    authed = [e for e in gate.EXERCISES
              if e.service == "dashboard" and e.headers]
    check("there is an unauthenticated call", len(unauth) == 1, str(unauth))
    check("and an authenticated one", len(authed) == 1, str(authed))
    check("the unauthenticated one must be refused",
          unauth and unauth[0].expect == (401, 403), str(unauth))
    check("the authenticated one must be let in",
          authed and 200 in authed[0].expect, str(authed))
    check("and it carries a bearer token",
          authed and authed[0].headers.get("Authorization", "").startswith("Bearer "),
          str(authed))


def test_a_gateway_that_answers_without_credentials_is_a_failure() -> None:
    """The finding this pair exists to catch: an endpoint that should
    demand credentials and does not."""
    scenario("open gateway fails")
    text = _compose(**{"heartbeat": 8010, "memu_core": 8001, "dashboard": 8080})
    runner = FakeRunner(
        {"heartbeat": "healthy", "memu-core": "healthy", "dashboard": "healthy"},
        {("dashboard", "noauth"): 200, ("dashboard", "auth"): 200})
    failures, _, _ = _audit(text, runner)
    check("answering 200 with no credentials is reported",
          any("gateway must refuse" in f for f in failures), str(failures))


def test_a_gateway_that_refuses_valid_credentials_is_a_failure() -> None:
    """And the other direction: a gateway nobody can get into."""
    scenario("closed gateway fails")
    text = _compose(**{"heartbeat": 8010, "memu_core": 8001, "dashboard": 8080})
    runner = FakeRunner(
        {"heartbeat": "healthy", "memu-core": "healthy", "dashboard": "healthy"},
        {("dashboard", "noauth"): 403, ("dashboard", "auth"): 403})
    failures, _, _ = _audit(text, runner)
    check("refusing valid credentials is reported",
          any("with credentials" in f for f in failures), str(failures))


def run_all() -> None:
    test_the_gateway_is_proven_in_both_directions()
    test_a_gateway_that_answers_without_credentials_is_a_failure()
    test_a_gateway_that_refuses_valid_credentials_is_a_failure()
    test_a_profile_gated_service_is_not_a_failure()
    test_an_ungated_service_that_is_absent_is_still_a_failure()
    test_gated_services_are_identified_from_the_compose_file()
    test_an_unhealthy_service_fails()
    test_a_whole_stack_down_fails()
    test_a_healthy_stack_passes()
    test_a_missing_profile_is_a_failure()
    test_a_profile_with_no_healthchecks_is_a_failure()
    test_a_docker_failure_is_a_failure_not_a_pass()
    test_a_service_with_no_healthcheck_is_not_read_as_healthy()
    test_ports_come_from_the_healthcheck()
    test_a_healthcheck_without_a_port_is_omitted_not_guessed()
    test_an_exercise_for_an_absent_service_is_reported()
    test_the_real_profile_declares_the_exercised_services()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Live Smoke Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
