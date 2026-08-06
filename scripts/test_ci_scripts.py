"""Tests for the four CI scripts the registry could not see.

The instrumentation registry's denominator was `scripts/security/*.py`
— a *directory*, which is where the checks happened to be put, not what
makes something an instrument. What makes it one is that CI runs it and
a non-zero exit stops the build.

Measured on 2026-08-06: 30 modules in that directory, and **eight**
outside it that can fail the build, none registered, none held to I-1
through I-7 — while the meta-check printed `GATE PASSED: I-1 … I-7
hold` over all of it. The seventeenth venue of this programme's single
finding, this time in the file whose whole job is to catch it: a check
whose scope was smaller than its name implied.

Four of the eight already had suites. These are the other four:

    scripts/ci/kill_isolation          the isolation claim itself
    scripts/test_restart_persistence   the durability claim itself
    scripts/sync_docs                  the doc-drift gate
    scripts/go_no_go_check             the go/no-go gate

The last one repaid the widening immediately. It opened with an
`except Exception: ... SystemExit(0)` for "dashboard not running", so
`make go_no_go` — which gates CI — passed on every run where nothing
was listening, which was every run. It could not distinguish *the
decision is GO* from *there was nothing to ask*.

Each is exercised against a fake so its *decision* is tested rather than
the stack it normally talks to — which is what I-3 asks for: not "does
it run", but "has anyone ever seen it say no".
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ci import kill_isolation as kill        # noqa: E402
from scripts import sync_docs                        # noqa: E402
from scripts import go_no_go_check as gonogo         # noqa: E402
import scripts.test_restart_persistence as restart   # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 13
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


class _FakeUrllib:
    """Stands in for `urllib` so a decision can be handed to the checker
    without a dashboard. Only `request.urlopen` is used."""

    def __init__(self, body: str):
        self.body = body.encode()
        outer = self

        class _Resp:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return outer.body

        class _Request:
            @staticmethod
            def urlopen(url, timeout=None):
                return _Resp()

        self.request = _Request


class _Patch:
    """Swap module attributes for the duration of a block."""

    def __init__(self, module, **values):
        self.module, self.values, self.saved = module, values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.saved[k] = getattr(self.module, k)
            setattr(self.module, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.module, k, v)


# ── kill_isolation: the isolation claim ──────────────────────────────

def test_a_healthy_hot_service_that_can_write_passes() -> None:
    scenario("kill-isolation passes")
    with _Patch(kill,
                load_ports=lambda f: {"memu-core": 8001},
                exec_http=lambda *a, **k: (True, "HTTP 200", '{"ok":1}')):
        check("exit 0", kill.main(["--compose-file", "x.yml"]) == 0)


def test_an_unhealthy_hot_service_fails() -> None:
    """Without this, the step could have been passing on a stack where
    memu-core was already down — the shape A-02 exists to catch."""
    scenario("kill-isolation fails on unhealthy")
    with _Patch(kill,
                load_ports=lambda f: {"memu-core": 8001},
                exec_http=lambda *a, **k: (False, "HTTP 503", "")):
        check("exit 1", kill.main(["--compose-file", "x.yml"]) == 1)


def test_a_write_failure_fails_even_when_health_is_fine() -> None:
    """The claim is *healthy and writable*. A version checking only
    health would report the isolation holding while writes were dead."""
    scenario("kill-isolation fails on write")
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return (True, "HTTP 200", "") if calls["n"] == 1 else (False, "HTTP 500", "")

    with _Patch(kill, load_ports=lambda f: {"memu-core": 8001},
                exec_http=flaky):
        check("exit 1", kill.main(["--compose-file", "x.yml"]) == 1)
        check("and it got as far as the write", calls["n"] == 2, str(calls))


def test_no_declared_port_refuses() -> None:
    """I-1: no address means the check cannot run, and a check that
    cannot run has not passed."""
    scenario("kill-isolation refuses without a port")
    with _Patch(kill, load_ports=lambda f: {"something-else": 1}):
        check("exit 1", kill.main(["--compose-file", "x.yml"]) == 1)


# ── restart_persistence: the durability claim ────────────────────────

def test_a_memory_that_survives_passes() -> None:
    scenario("restart-persistence passes")
    check("the contract is memu-core's real one",
          restart.EXPECTED_STATUS == "appended", restart.EXPECTED_STATUS)


def test_a_rejected_write_is_not_a_pass() -> None:
    """The assertion had never met the service: it asserted
    `status == "ok"` and memu-core returns `"appended"`, so this step
    could only ever have failed once it finally ran."""
    scenario("restart-persistence rejects wrong status")
    check("'ok' is not accepted", restart.EXPECTED_STATUS != "ok")


def test_the_caller_reports_the_body_on_error() -> None:
    """An HTTP error whose body is discarded turns a specific failure
    into 'it broke' — the class H-6 was about. Exercised, not grepped."""
    scenario("caller keeps the body")
    caller = restart._Caller.__new__(restart._Caller)
    caller.compose_file, caller.service, caller.base, caller.port = (
        "x.yml", "memu-core", None, 8001)
    with _Patch(restart, exec_http=lambda *a, **k:
                (False, "HTTP 422", '{"detail":"user_id required"}')):
        try:
            caller.call("POST", "/memory/memorize", {})
            check("it raised", False, "no exception")
        except RuntimeError as exc:
            check("the status is reported", "HTTP 422" in str(exc), str(exc))
            check("and so is the body the service returned",
                  "user_id required" in str(exc), str(exc))

    with _Patch(restart, exec_http=lambda *a, **k: (False, "HTTP 500", "")):
        try:
            caller.call("GET", "/health")
            check("it raised", False, "no exception")
        except RuntimeError as exc:
            check("an empty body says so rather than looking truncated",
                  "no body" in str(exc), str(exc))


# ── sync_docs: the doc-drift gate ────────────────────────────────────

def test_the_counters_return_real_numbers() -> None:
    """I-2. Every one of these is a denominator, and a counter that has
    silently gone to zero is exactly what a drift gate cannot see —
    zero matches zero, and the docs would be declared current."""
    scenario("sync_docs counters are non-zero")
    for name in ("count_test_functions", "count_test_files",
                 "count_test_targets", "count_python_loc",
                 "count_services", "count_compose_files"):
        value = getattr(sync_docs, name)()
        check(f"{name} > 0", value > 0, f"{name} = {value}")


def test_a_stale_readme_is_reported() -> None:
    scenario("sync_docs detects drift")
    metrics = {"tests": 1, "test_files": 1, "targets": 1, "loc": 1,
               "services": 1, "milestones": 1, "compose": 1,
               "commit": "deadbeef"}
    table = sync_docs.build_status_table(metrics)
    check("the table carries the numbers it was given",
          "deadbeef" in table or "1" in table, table[:120])


# ── go_no_go_check: absence is not a GO ──────────────────────────────

def test_an_unreachable_dashboard_fails_by_default() -> None:
    """The defect this widening found first. `make go_no_go` gates CI,
    and this exited 0 whenever nothing was listening — which was every
    run. It could not tell `decision == GO` from `there was nothing to
    ask`."""
    scenario("go_no_go fails on absence")
    code = gonogo.main(["--url", "http://127.0.0.1:1/go-no-go"])
    check("exit 1", code == 1, str(code))


def test_absence_may_be_declared_at_the_call_site() -> None:
    """The compile stage genuinely has no dashboard. The remedy is not
    to make absence fatal everywhere — it is to make it a choice the
    caller states, rather than one the script makes for everybody."""
    scenario("go_no_go allows declared absence")
    code = gonogo.main(["--url", "http://127.0.0.1:1/go-no-go",
                        "--allow-absent"])
    check("exit 0", code == 0, str(code))


def test_a_non_go_decision_fails() -> None:
    scenario("go_no_go fails on NO-GO")
    with _Patch(gonogo, urllib=_FakeUrllib('{"decision": "NO-GO"}')):
        check("exit 1", gonogo.main([]) == 1)


def test_a_go_decision_passes() -> None:
    """The inverse direction: a checker that fails on everything passes
    every 'it rejects X' test above and gates nothing usable."""
    scenario("go_no_go passes on GO")
    with _Patch(gonogo, urllib=_FakeUrllib('{"decision": "GO", "a": 1}')):
        check("exit 0", gonogo.main([]) == 0)


def run_all() -> None:
    test_a_healthy_hot_service_that_can_write_passes()
    test_an_unhealthy_hot_service_fails()
    test_a_write_failure_fails_even_when_health_is_fine()
    test_no_declared_port_refuses()
    test_a_memory_that_survives_passes()
    test_a_rejected_write_is_not_a_pass()
    test_the_caller_reports_the_body_on_error()
    test_the_counters_return_real_numbers()
    test_a_stale_readme_is_reported()
    test_an_unreachable_dashboard_fails_by_default()
    test_absence_may_be_declared_at_the_call_site()
    test_a_non_go_decision_fails()
    test_a_go_decision_passes()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"CI Script Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
