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
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

# (service, method, path, body) — the endpoints that prove the spine is
# wired, not merely listening. Kept short on purpose: this is a smoke
# test, and a long one that nobody reads is how the old file grew to
# eleven probes of which five addressed nothing.
EXERCISES: Tuple[Tuple[str, str, str, dict], ...] = (
    ("heartbeat", "POST", "/tick", {}),
    ("memu-core", "POST", "/memory/memorize", {
        "timestamp": "2026-01-01T00:00:00Z",
        "event_type": "ci-live-smoke",
        "result_raw": "live smoke",
        "user_id": "ci",
    }),
    ("dashboard", "GET", "/go-no-go", {}),
)

Runner = Callable[[Sequence[str]], Tuple[int, str, str]]


def _run(argv: Sequence[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(list(argv), capture_output=True, text=True, timeout=120)
    return proc.returncode, proc.stdout, proc.stderr


def health_ports(doc: dict) -> Dict[str, int]:
    """service -> the port its own healthcheck probes.

    Derived, never typed. A service whose healthcheck names no port
    (`redis`, `postgres`, `ollama` use their own CLI probes) is simply
    absent here rather than guessed at.
    """
    out: Dict[str, int] = {}
    for name, cfg in sorted((doc.get("services") or {}).items()):
        test = ((cfg or {}).get("healthcheck") or {}).get("test")
        if not test:
            continue
        text = " ".join(test) if isinstance(test, list) else str(test)
        match = re.search(r"localhost:(\d+)", text)
        if match:
            out[name] = int(match.group(1))
    return out


def compose_health(compose_file: str, runner: Runner) -> Dict[str, str]:
    """service -> health as Docker itself reports it.

    `docker compose ps` answers with the verdict of the healthcheck the
    compose file already declares, so the port lives in exactly one
    place. A service with no healthcheck reports an empty string, which
    is reported as `no-healthcheck` and never silently read as healthy.
    """
    code, out, err = runner(["docker", "compose", "-f", compose_file,
                             "ps", "--format", "json"])
    if code != 0:
        raise RuntimeError(f"docker compose ps failed ({code}): {err.strip()[:300]}")
    states: Dict[str, str] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows = row if isinstance(row, list) else [row]
        for entry in rows:
            name = entry.get("Service") or entry.get("Name")
            if not name:
                continue
            states[name] = (entry.get("Health") or "").strip() or "no-healthcheck"
    return states


def exec_http(compose_file: str, service: str, port: int, method: str,
              path: str, body: dict, runner: Runner) -> Tuple[bool, str]:
    """Call an endpoint from inside the container, the only place it exists.

    stdlib only, because the service images carry no HTTP client beyond
    what their own healthchecks use — which is `urllib.request`.
    """
    payload = json.dumps(body)
    program = (
        "import json,sys,urllib.request,urllib.error\n"
        f"body={payload!r}\n"
        f"req=urllib.request.Request('http://localhost:{port}{path}',"
        f"method='{method}')\n"
        "req.add_header('Content-Type','application/json')\n"
        f"data=body.encode() if '{method}'!='GET' else None\n"
        "try:\n"
        "    r=urllib.request.urlopen(req,data=data,timeout=15)\n"
        "    print('STATUS', r.status)\n"
        "except urllib.error.HTTPError as e:\n"
        "    print('STATUS', e.code)\n"
        "except Exception as e:\n"
        "    print('ERROR', type(e).__name__, e); sys.exit(1)\n"
    )
    code, out, err = runner(["docker", "compose", "-f", compose_file,
                             "exec", "-T", service, "python", "-c", program])
    text = (out + err).strip()
    if code != 0:
        return False, text[:300] or f"exit {code}"
    match = re.search(r"STATUS (\d+)", out)
    if not match:
        return False, text[:300] or "no status line"
    status = int(match.group(1))
    # 4xx is an answer: the service is up and enforcing something. 5xx and
    # no-answer are not.
    return status < 500, f"HTTP {status}"


def audit(compose_file: str, runner: Runner = _run) -> Tuple[List[str], int, int]:
    """Return (failures, services probed, exercises run)."""
    import yaml

    path = REPO / compose_file
    if not path.exists():
        # I-1: an absent profile is a failure. A smoke test with no stack
        # to smoke has not passed.
        return ([f"{compose_file}: missing — nothing to verify"], 0, 0)
    doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    ports = health_ports(doc)
    failures: List[str] = []

    states = compose_health(compose_file, runner)
    probed = 0
    for service in sorted(ports):
        probed += 1
        state = states.get(service)
        if state is None:
            failures.append(f"{service}: declared in {compose_file} but not "
                            f"running — the profile did not start it")
        elif state != "healthy":
            failures.append(f"{service}: health={state}")

    exercised = 0
    for service, method, endpoint, body in EXERCISES:
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
        ok, detail = exec_http(compose_file, service, ports[service],
                               method, endpoint, body, runner)
        if not ok:
            failures.append(f"{service} {method} {endpoint}: {detail}")

    if probed == 0:
        failures.append(f"{compose_file}: no service declares a health port — "
                        f"nothing was inspected, which is not a pass")
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

    print(f"  inspected: {probed} service(s) with a declared health port, "
          f"{exercised} endpoint exercise(s) ({args.compose_file})")
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
