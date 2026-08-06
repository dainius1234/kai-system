"""Tests for `compose_probe` — reaching a stack that publishes no ports.

Nine `curl http://localhost:PORT` sites across five steps of
`core-tests.yml` addressed host ports that stopped existing at `e4655bc`
("Edge lockdown — remove all host-port bindings except dashboard
loopback"). `tool-gate`, `memu-core`, `memu-core-introspect`, `agentic`
and `memu-graph` are on networks declared `internal: true`: there is no
port to restore and no address to route to.

The map could not be repaired, only deleted, and these two primitives
are what replaced it — `wait_healthy`, which reads Docker's verdict on
the healthcheck each service already declares, and `exec_http`, which
makes the call from inside the container.

Everything here runs against a fake command runner. `wait_healthy` also
takes its clock and its sleep as arguments, so the timeout path is
asserted in microseconds rather than by waiting five real minutes to
prove it waits five minutes.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ci import compose_probe as probe  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 16
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


class FakeRunner:
    """`ps` answers from a scripted sequence; `exec` from a canned reply."""

    def __init__(self, sequence, exec_reply=(0, "STATUS 200\n{\"ok\":true}", "")):
        # A list of dicts, one per `ps` call; the last repeats forever.
        self.sequence = list(sequence)
        self.exec_reply = exec_reply
        self.ps_calls = 0
        self.exec_calls = []

    def __call__(self, argv):
        if "ps" in argv:
            index = min(self.ps_calls, len(self.sequence) - 1)
            self.ps_calls += 1
            health = self.sequence[index]
            if health is None:
                return 1, "", "cannot connect to the docker daemon"
            rows = [json.dumps({"Service": s, "Health": h})
                    for s, h in health.items()]
            return 0, "\n".join(rows) + "\n", ""
        if "exec" in argv:
            self.exec_calls.append(argv)
            return self.exec_reply
        return 0, "", ""


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.slept = 0.0

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.t += seconds


# ── health ports come from the compose file ──────────────────────────

def test_both_healthcheck_spellings_yield_a_port() -> None:
    """`python -c urlopen(...)` in the minimal profile, `wget -qO-` in
    the sovereign one. The pattern looked for is the address, not the
    tool — matching on `python -c` would have silently skipped every
    sovereign service."""
    scenario("both healthcheck spellings")
    doc = {"services": {
        "memu-core": {"healthcheck": {"test": [
            "CMD-SHELL",
            "python -c \"import urllib.request; "
            "urllib.request.urlopen('http://localhost:8001/health')\""]}},
        "tool-gate": {"healthcheck": {"test": [
            "CMD-SHELL", "wget -qO- http://localhost:8000/health || exit 1"]}},
    }}
    ports = probe.health_ports(doc)
    check("both resolve", ports == {"memu-core": 8001, "tool-gate": 8000},
          str(ports))


def test_a_portless_healthcheck_yields_nothing() -> None:
    """`redis-cli ping`, `pg_isready`, `ollama list` name no port. They
    are still waited on — `wait_healthy` needs no port — but no port is
    invented for them."""
    scenario("portless healthcheck")
    doc = {"services": {
        "redis": {"healthcheck": {"test": ["CMD", "redis-cli", "ping"]}},
        "postgres": {"healthcheck": {"test": [
            "CMD-SHELL", "pg_isready -U keeper -d sovereign"]}},
    }}
    check("no port is guessed", probe.health_ports(doc) == {},
          str(probe.health_ports(doc)))


# ── wait_healthy ─────────────────────────────────────────────────────

def test_it_returns_when_everything_is_healthy() -> None:
    scenario("all healthy returns clean")
    runner = FakeRunner([{"a": "healthy", "b": "healthy"}])
    clock = FakeClock()
    unhealthy = probe.wait_healthy("c.yml", ["a", "b"], timeout=60,
                                   runner=runner, sleep=clock.sleep,
                                   now=clock.now)
    check("nothing outstanding", unhealthy == [], str(unhealthy))
    check("and it did not sleep", clock.slept == 0, str(clock.slept))


def test_it_waits_for_a_service_that_becomes_healthy() -> None:
    scenario("waits then succeeds")
    runner = FakeRunner([
        {"a": "starting"}, {"a": "starting"}, {"a": "healthy"}])
    clock = FakeClock()
    unhealthy = probe.wait_healthy("c.yml", ["a"], timeout=60, interval=2,
                                   runner=runner, sleep=clock.sleep,
                                   now=clock.now)
    check("it eventually passes", unhealthy == [], str(unhealthy))
    check("having polled more than once", runner.ps_calls >= 3,
          str(runner.ps_calls))


def test_it_gives_up_and_says_what_was_wrong() -> None:
    scenario("timeout reports last state")
    runner = FakeRunner([{"a": "unhealthy"}])
    clock = FakeClock()
    unhealthy = probe.wait_healthy("c.yml", ["a"], timeout=10, interval=2,
                                   runner=runner, sleep=clock.sleep,
                                   now=clock.now)
    check("it fails", len(unhealthy) == 1, str(unhealthy))
    check("and reports the last known state",
          "unhealthy" in unhealthy[0], str(unhealthy))
    check("and it did not wait forever", clock.slept <= 12, str(clock.slept))


def test_a_service_that_is_not_running_is_reported_as_such() -> None:
    """Distinct from unhealthy: nothing started it at all."""
    scenario("absent service reported")
    runner = FakeRunner([{"b": "healthy"}])
    clock = FakeClock()
    unhealthy = probe.wait_healthy("c.yml", ["a"], timeout=4, interval=2,
                                   runner=runner, sleep=clock.sleep,
                                   now=clock.now)
    check("it fails", len(unhealthy) == 1, str(unhealthy))
    check("and says it is not running",
          "not running" in unhealthy[0], str(unhealthy))


def test_no_healthcheck_is_never_read_as_healthy() -> None:
    """A container Docker is not checking is one nobody is checking."""
    scenario("no-healthcheck is not healthy")
    runner = FakeRunner([{"a": ""}])
    clock = FakeClock()
    unhealthy = probe.wait_healthy("c.yml", ["a"], timeout=4, interval=2,
                                   runner=runner, sleep=clock.sleep,
                                   now=clock.now)
    check("it fails", len(unhealthy) == 1, str(unhealthy))
    check("and says no-healthcheck",
          "no-healthcheck" in unhealthy[0], str(unhealthy))


def test_a_docker_error_does_not_read_as_healthy() -> None:
    """I-1. If `docker compose ps` fails, the answer is unknown."""
    scenario("docker error is not health")
    runner = FakeRunner([None])
    clock = FakeClock()
    unhealthy = probe.wait_healthy("c.yml", ["a"], timeout=4, interval=2,
                                   runner=runner, sleep=clock.sleep,
                                   now=clock.now)
    check("it fails", len(unhealthy) == 1, str(unhealthy))
    check("and names docker as the reason",
          "docker error" in unhealthy[0], str(unhealthy))


# ── exec_http ────────────────────────────────────────────────────────

def test_a_2xx_is_an_answer() -> None:
    scenario("2xx passes")
    runner = FakeRunner([{}], exec_reply=(0, 'STATUS 200\n{"status":"ok"}', ""))
    ok, detail, body = probe.exec_http("c.yml", "s", 8001, "GET", "/health",
                                       runner=runner)
    check("it passes", ok, detail)
    check("the status is reported", detail == "HTTP 200", detail)
    check("and the body comes back", '"ok"' in body, body)


def test_a_4xx_is_still_an_answer() -> None:
    """The service is up and enforcing something. That is not a dead
    service, and conflating the two would make every authenticated
    endpoint look down."""
    scenario("4xx is an answer")
    runner = FakeRunner([{}], exec_reply=(0, "STATUS 403\n{}", ""))
    ok, detail, _ = probe.exec_http("c.yml", "s", 8001, "GET", "/x",
                                    runner=runner)
    check("it passes", ok, detail)
    check("reported as 403", detail == "HTTP 403", detail)


def test_a_5xx_is_a_failure() -> None:
    scenario("5xx fails")
    runner = FakeRunner([{}], exec_reply=(0, "STATUS 503\n{}", ""))
    ok, detail, _ = probe.exec_http("c.yml", "s", 8001, "GET", "/x",
                                    runner=runner)
    check("it fails", not ok, detail)


def test_no_answer_at_all_is_a_failure() -> None:
    """The container answered nothing recognisable — not a pass."""
    scenario("no status line fails")
    runner = FakeRunner([{}], exec_reply=(0, "some noise\n", ""))
    ok, detail, _ = probe.exec_http("c.yml", "s", 8001, "GET", "/x",
                                    runner=runner)
    check("it fails", not ok, detail)
    check("and says there was no status", "no status line" in detail, detail)


def test_a_failed_exec_is_a_failure() -> None:
    scenario("exec failure fails")
    runner = FakeRunner([{}], exec_reply=(1, "", "no such service: ghost"))
    ok, detail, _ = probe.exec_http("c.yml", "ghost", 8001, "GET", "/x",
                                    runner=runner)
    check("it fails", not ok, detail)
    check("and surfaces docker's message",
          "no such service" in detail, detail)


def test_a_dead_container_is_not_reported_as_no_healthcheck() -> None:
    """The defect this instrument committed against itself.

    `agentic` died at import — `ModuleNotFoundError: No module named
    'system_fsm'` — and this function reported `agentic: no-healthcheck`.
    That is false: agentic declares one. Docker reports an empty
    `Health` for a container that is not running, so the message sent
    the reader to the compose file to look for a healthcheck that was
    already there, while the traceback in the container log went
    unmentioned. A diagnostic that reports something other than what
    happened."""
    scenario("dead container named as dead")

    def runner(cmd):
        return 0, ('{"Service":"agentic","State":"exited","ExitCode":1,'
                   '"Health":""}\n'), ""

    states = probe.compose_health("x.yml", runner)
    check("it is not called no-healthcheck",
          "no-healthcheck" not in states["agentic"], states["agentic"])
    check("it is named as exited", "exited" in states["agentic"],
          states["agentic"])
    check("with its exit code", "exit 1" in states["agentic"],
          states["agentic"])
    check("and points at the container log",
          "container log" in states["agentic"], states["agentic"])


def test_a_restarting_container_is_named_as_restarting() -> None:
    """`document-parser` crash-looped for months while reporting
    nothing useful. A restart loop is not a health state."""
    scenario("restart loop named")

    def runner(cmd):
        return 0, '{"Service":"dp","State":"restarting","Health":""}\n', ""

    states = probe.compose_health("x.yml", runner)
    check("named as restarting", "restarting" in states["dp"], states["dp"])
    check("not as no-healthcheck", "no-healthcheck" not in states["dp"],
          states["dp"])


def test_a_running_container_without_a_healthcheck_still_says_so() -> None:
    """The original rule survives: running + no healthcheck is still
    `no-healthcheck`, not `healthy`."""
    scenario("running no-healthcheck preserved")

    def runner(cmd):
        return 0, '{"Service":"x","State":"running","Health":""}\n', ""

    check("unchanged",
          probe.compose_health("x.yml", runner)["x"] == "no-healthcheck",
          probe.compose_health("x.yml", runner)["x"])


def run_all() -> None:
    test_a_dead_container_is_not_reported_as_no_healthcheck()
    test_a_restarting_container_is_named_as_restarting()
    test_a_running_container_without_a_healthcheck_still_says_so()
    test_both_healthcheck_spellings_yield_a_port()
    test_a_portless_healthcheck_yields_nothing()
    test_it_returns_when_everything_is_healthy()
    test_it_waits_for_a_service_that_becomes_healthy()
    test_it_gives_up_and_says_what_was_wrong()
    test_a_service_that_is_not_running_is_reported_as_such()
    test_no_healthcheck_is_never_read_as_healthy()
    test_a_docker_error_does_not_read_as_healthy()
    test_a_2xx_is_an_answer()
    test_a_4xx_is_still_an_answer()
    test_a_5xx_is_a_failure()
    test_no_answer_at_all_is_a_failure()
    test_a_failed_exec_is_a_failure()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Compose Probe Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
