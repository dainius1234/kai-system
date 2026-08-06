"""Live end-to-end smoke against a running compose profile — one that can fail.

It replaces `scripts/test_core_integration.py`, which could not.

That file wrapped every probe in a `safe_get`/`safe_post` pair that
caught every exception, printed a line, and returned `None`; the caller
printed "service X not reachable" and `continue`d; and `main()` ended
with a bare `return 0`. **With the entire stack down it printed eleven
failures and exited 0.** It was the one CI step whose job is to prove the
system actually runs, and it was structurally incapable of saying no.

Nobody saw it either way: `core-tests.yml` had been dying at step 7 of 59
for thirty commits, and this is step 49.

It also carried its own map of eleven `http://localhost:PORT` URLs. Two
things were wrong with that beyond the duplication:

  - Five of the eleven name services that are not in the minimal profile
    at all (`executor`, `camera`, `kai-advisor`, `avatar`, `audio`), so
    those probes could only ever have printed a failure and continued.
  - Since `e4655bc` — *"Edge lockdown — remove all host-port bindings
    except dashboard loopback"* — **none** of them are reachable from the
    host. `tool-gate`, `memu-core`, `memu-core-introspect` and `agentic`
    sit on networks declared `internal: true`. There is no port to
    restore and no address to route to; that is the lockdown working.

So the probes move inside the network, which is the only place they can
run, and the port map is deleted rather than corrected:

  - **Health** is read from `docker compose ps`, which reports the
    verdict of the healthcheck each service already declares beside
    itself in the compose file. No port appears in this file.
  - **Exercises** run through `docker compose exec`, inside the
    container, against `localhost` — with the port taken from that
    service's own healthcheck rather than typed here again.

Three rules it now keeps that its predecessor did not:

  I-1  A probe that could not run is a failure. "Unreachable" is not
       "fine", and neither is "the profile has no such service".
  I-2  It prints a denominator. A smoke test that inspected nothing must
       be distinguishable from one that inspected everything, and the
       old one printed the same thing in both cases.
  I-3  The verdict is computed from the probes, not hard-coded. There is
       no `return 0` at the bottom of this file.

Exit 0 = every expected service is healthy and every exercise answered.
Exit 1 = one did not, or nothing was inspected.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.ci.compose_probe import (  # noqa: E402
    REPO, Runner, compose_health, exec_http, health_ports, run as _run)


# (service, method, path, body) — the endpoints that prove the spine is
# wired, not merely listening. Kept short on purpose: this is a smoke
# test, and a long one that nobody reads is how the old file grew to
# eleven probes of which five addressed nothing.

@dataclass(frozen=True)
class Exercise:
    """One endpoint call, and what counts as the right answer."""
    service: str
    method: str
    path: str
    body: dict = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    expect: Optional[Tuple[int, ...]] = None
    why: str = ""


def _dashboard_token() -> str:
    """The token CI gives the dashboard, read at call time."""
    return os.getenv("KAI_DASHBOARD_TOKEN", "")


EXERCISES: Tuple[Exercise, ...] = (
    Exercise("heartbeat", "POST", "/tick"),
    Exercise("memu-core", "POST", "/memory/memorize", {
        "timestamp": "2026-01-01T00:00:00Z",
        "event_type": "ci-live-smoke",
        "result_raw": "live smoke",
        "user_id": "ci",
    }),
    # The inbound identity gateway, proven in both directions.
    #
    # The first live run of this file reported `dashboard GET /go-no-go:
    # HTTP 503` and that was not a defect — it was Wave 1 Track A
    # working: with no `KAI_DASHBOARD_TOKEN`, `authenticate()` returns
    #
    #   503 "… dashboard authentication is misconfigured. This gateway
    #        fails closed by design."
    #
    # A probe that only ever sees 503 cannot tell *refusing correctly*
    # from *permanently broken*, and the dashboard had in fact been
    # permanently broken (no python-multipart) until this morning. So
    # both directions are asserted, which is I-3 applied to a live
    # service instead of to a gate.
    Exercise("dashboard", "GET", "/go-no-go", expect=(401, 403),
             why="no credentials — the gateway must refuse"),
    Exercise("dashboard", "GET", "/go-no-go",
             headers={"Authorization": f"Bearer {_dashboard_token()}"},
             expect=(200, 503),
             why="with credentials — 200 GO or 503 NO_GO, both of which "
                 "mean the gateway let us in and the report was built"),
)


def gated_services(doc: dict) -> set:
    """Services a bare `docker compose up` deliberately does not start.

    A service with a `profiles:` key is opt-in: `docker compose up`
    without `--profile` skips it, by design.

    This function exists because the first live run of this file
    reported **seventeen** failures, and sixteen of them were these:

        - audio-service: declared in docker-compose.minimal.yml but not
          running — the profile did not start it

    `audio-service` is `profiles: ["sensors"]`. It was *correctly* not
    running. Eighteen of the minimal profile's services are gated this
    way, and the smoke test was demanding all of them.

    That is the systemic finding inverted: usually a check's scope is
    smaller than its name and it reports success over what it missed.
    Here the scope was **larger than reality**, and it reported failure
    over things that were right — which is worse, because a survey with
    false positives sends people to break working code, and because
    sixteen false alarms are how the one true finding
    (`dashboard: health=starting`) gets lost.

    Reported as a count rather than silently dropped, so the
    denominator still says what was and was not looked at.
    """
    return {name for name, cfg in (doc.get("services") or {}).items()
            if (cfg or {}).get("profiles")}


def audit(compose_file: str, runner: Runner = _run,
          timeout: float = 120.0,
          sleep: Callable[[float], None] = time.sleep,
          now: Callable[[], float] = time.monotonic
          ) -> Tuple[List[str], int, int]:
    """Return (failures, services probed, exercises run).

    `sleep`/`now` are injectable for the same reason `wait_healthy`
    injects them: a test that proves this waits two minutes must not
    take two minutes to say so.
    """
    import yaml

    path = REPO / compose_file
    if not path.exists():
        # I-1: an absent profile is a failure. A smoke test with no stack
        # to smoke has not passed.
        return ([f"{compose_file}: missing — nothing to verify"], 0, 0)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    all_ports = health_ports(doc)
    gated = gated_services(doc)
    ports = {name: port for name, port in all_ports.items()
             if name not in gated}
    failures: List[str] = []

    # Wait, do not sample. The first live run caught `dashboard:
    # health=starting` — the container had been up for two seconds and
    # its healthcheck had not reported yet. A smoke test that reads the
    # clock once turns "not finished starting" into "broken", which is a
    # different statement about a working system.
    if ports:
        from scripts.ci.compose_probe import wait_healthy
        wait_healthy(compose_file, sorted(ports), timeout=timeout,
                     runner=runner, sleep=sleep, now=now)

    states = compose_health(compose_file, runner)
    probed = 0
    for service in sorted(ports):
        probed += 1
        state = states.get(service)
        if state is None:
            failures.append(f"{service}: declared in {compose_file} with no "
                            f"`profiles:` key, so this bring-up should have "
                            f"started it, and did not")
        elif state != "healthy":
            failures.append(f"{service}: health={state}")

    exercised = 0
    for ex in EXERCISES:
        service, method, endpoint, body = ex.service, ex.method, ex.path, ex.body
        if service not in ports:
            # Not a skip. The exercise list and the profile disagreeing is
            # exactly the drift that left five of the old file's eleven
            # probes addressed to services that were never there.
            failures.append(f"{service}: no health port in {compose_file}, so "
                            f"{method} {endpoint} cannot be exercised")
            continue
        if states.get(service) != "healthy":
            continue        # already reported above; do not double-count
        exercised += 1
        ok, detail, _ = exec_http(compose_file, service, ports[service],
                                  method, endpoint, body, runner,
                                  headers=ex.headers, expect=ex.expect)
        if not ok:
            note = f" ({ex.why})" if ex.why else ""
            failures.append(f"{service} {method} {endpoint}{note}: {detail}")

    if probed == 0:
        failures.append(f"{compose_file}: no service declares a health port "
                        f"outside a `profiles:` gate — nothing was "
                        f"inspected, which is not a pass")
    return failures, probed, exercised


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", default="docker-compose.minimal.yml")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        failures, probed, exercised = audit(args.compose_file)
    except Exception as exc:                       # I-1
        print(f"FAIL: the smoke test could not run: {exc}")
        print("  A check that could not execute is not a check that passed.")
        return 1

    import yaml
    doc = yaml.safe_load(
        (REPO / args.compose_file).read_text(encoding="utf-8")) or {}
    gated = len(gated_services(doc) & set(health_ports(doc)))
    print(f"  inspected: {probed} service(s) with a declared health port, "
          f"{exercised} endpoint exercise(s) ({args.compose_file})")
    if gated:
        # The denominator, stated. These are not skipped quietly: a
        # bare `docker compose up` is meant not to start them, and
        # saying so is the difference between a scope and an omission.
        print(f"  not expected up: {gated} service(s) behind a "
              f"`profiles:` gate this bring-up did not activate")
    print()
    if failures:
        print(f"FAIL: {len(failures)} problem(s):")
        print()
        for line in failures:
            print(f"  - {line}")
        return 1
    print("PASS: every service with a healthcheck is healthy and every "
          "exercised endpoint answered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
