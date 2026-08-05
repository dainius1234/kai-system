"""Talk to a running compose stack without knowing a single port number.

Every service in this repository declares how to check itself, beside
itself, in the compose file:

    healthcheck:
      test: ["CMD-SHELL", "python -c \\"import urllib.request;
             urllib.request.urlopen('http://localhost:8001/health')\\""]

`core-tests.yml` then kept a **second** copy of that map, as nine
`curl http://localhost:PORT` sites spread across five steps. The copy
drifted, and then it stopped being reachable at all: commit `e4655bc`
— *"Edge lockdown — remove all host-port bindings except dashboard
loopback"* — put `tool-gate`, `memu-core`, `memu-core-introspect`,
`agentic` and `memu-graph` on networks declared `internal: true`.

There is no port to restore and no address to route to. The host cannot
reach those services, by design, and that design is correct. So the copy
cannot be repaired; it can only be deleted, and the probes moved to the
one place the services exist — inside the network.

Two primitives are enough for every step that used to curl:

  `wait_healthy`   polls `docker compose ps` and waits for Docker's own
                   verdict on the healthcheck. No port, no endpoint, no
                   knowledge of what "healthy" means for that service —
                   the service already said.
  `exec_http`      runs one HTTP call inside the container via
                   `docker compose exec`, with the port taken from that
                   service's healthcheck rather than typed again.

Both fail closed. A service that is missing, a service with no
healthcheck, and a `docker` command that errors are all reported —
never read as "fine". The whole reason this file exists is that
"could not reach it" had been reading as "nothing to report" for thirty
commits.

The sovereign boot step already did it the right way for one service,
and only one:

    docker compose exec -T postgres pg_isready -U keeper -d sovereign

That line was correct and never generalised. This is the generalisation.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

Runner = Callable[[Sequence[str]], Tuple[int, str, str]]

#: How long `docker compose exec` gets before it is treated as hung.
EXEC_TIMEOUT = 120


def run(argv: Sequence[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(list(argv), capture_output=True, text=True,
                          timeout=EXEC_TIMEOUT)
    return proc.returncode, proc.stdout, proc.stderr


def health_ports(doc: dict) -> Dict[str, int]:
    """service -> the port its own healthcheck probes.

    Derived, never typed. Both spellings in this tree are matched — the
    `python -c "...urlopen('http://localhost:8001/health')"` form and the
    `wget -qO- http://localhost:8000/health` form — because the pattern
    looked for is the address, not the tool.

    A service whose healthcheck names no port (`redis` runs
    `redis-cli ping`, `postgres` runs `pg_isready`, `ollama` runs
    `ollama list`) is simply absent from the result rather than guessed
    at. Those are still waited on by `wait_healthy`, which needs no port.
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


def compose_health(compose_file: str, runner: Runner = run) -> Dict[str, str]:
    """service -> health as Docker itself reports it.

    An empty `Health` field means the container declares no healthcheck.
    That is reported as `no-healthcheck` and never collapsed into
    `healthy`: a container Docker is not checking is a container nobody
    is checking.
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
        for entry in (row if isinstance(row, list) else [row]):
            name = entry.get("Service") or entry.get("Name")
            if not name:
                continue
            states[name] = (entry.get("Health") or "").strip() or "no-healthcheck"
    return states


def wait_healthy(compose_file: str, services: Sequence[str],
                 timeout: float = 300.0, interval: float = 2.0,
                 runner: Runner = run,
                 sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], float] = time.monotonic) -> List[str]:
    """Wait for each named service to report healthy. Return what did not.

    `now` and `sleep` are injectable so the timeout path can be tested
    without a test that takes five minutes to prove it waits five
    minutes.
    """
    deadline = now() + timeout
    pending = list(services)
    last: Dict[str, str] = {}
    while True:
        docker_error = None
        try:
            states = compose_health(compose_file, runner)
        except RuntimeError as exc:
            # Caught by this module's own tests: the loop below used to
            # overwrite this with "not running", so a broken daemon was
            # reported as a service nobody started — a wrong diagnosis,
            # which sends the next person after the wrong problem.
            docker_error = f"docker error: {exc}"
            states = {}
        still: List[str] = []
        for name in pending:
            state = states.get(name)
            if state == "healthy":
                continue
            last[name] = docker_error or state or "not running"
            still.append(name)
        pending = still
        if not pending or now() >= deadline:
            break
        sleep(interval)
    return [f"{name}: {last.get(name, 'unknown')}" for name in pending]


def exec_http(compose_file: str, service: str, port: int, method: str,
              path: str, body: dict | None = None,
              runner: Runner = run) -> Tuple[bool, str, str]:
    """One HTTP call from inside the container. Returns (ok, detail, body).

    stdlib only, because the service images carry no HTTP client beyond
    what their own healthchecks already use.
    """
    payload = json.dumps(body or {})
    program = (
        "import sys,urllib.request,urllib.error\n"
        f"body={payload!r}\n"
        f"req=urllib.request.Request('http://localhost:{port}{path}',"
        f"method='{method}')\n"
        "req.add_header('Content-Type','application/json')\n"
        f"data=body.encode() if '{method}' not in ('GET','HEAD') else None\n"
        "try:\n"
        "    r=urllib.request.urlopen(req,data=data,timeout=30)\n"
        "    print('STATUS', r.status)\n"
        "    sys.stdout.write(r.read().decode('utf-8','replace'))\n"
        "except urllib.error.HTTPError as e:\n"
        "    print('STATUS', e.code)\n"
        "    sys.stdout.write(e.read().decode('utf-8','replace'))\n"
        "except Exception as e:\n"
        "    print('ERROR', type(e).__name__, e); sys.exit(1)\n"
    )
    code, out, err = runner(["docker", "compose", "-f", compose_file,
                             "exec", "-T", service, "python", "-c", program])
    if code != 0:
        return False, ((out + err).strip()[:300] or f"exit {code}"), ""
    match = re.search(r"STATUS (\d+)", out)
    if not match:
        # Say what was wrong *and* what was seen. Returning the raw output
        # alone left the reader to work out that the problem was the shape
        # of the reply rather than its content.
        seen = (out + err).strip()[:200]
        return False, f"no status line in the reply: {seen or '(no output)'}", ""
    status = int(match.group(1))
    payload_text = out.split("\n", 1)[1] if "\n" in out else ""
    # 4xx is an answer — the service is up and enforcing something.
    # 5xx and no-answer are not.
    return status < 500, f"HTTP {status}", payload_text


def load_ports(compose_file: str) -> Dict[str, int]:
    import yaml
    path = REPO / compose_file
    if not path.exists():
        raise FileNotFoundError(f"{compose_file}: missing — nothing to probe")
    return health_ports(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wait on compose healthchecks")
    parser.add_argument("--compose-file", required=True)
    parser.add_argument("--services", nargs="+", required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(list(argv) if argv is not None else None)

    unhealthy = wait_healthy(args.compose_file, args.services, args.timeout)
    print(f"  waited on: {len(args.services)} service(s) in "
          f"{args.compose_file}, {len(args.services) - len(unhealthy)} healthy")
    if unhealthy:
        print(f"\nFAIL: {len(unhealthy)} service(s) never became healthy "
              f"within {args.timeout:.0f}s:\n")
        for line in unhealthy:
            print(f"  - {line}")
        print("\n  'no-healthcheck' means the container declares none, so "
              "Docker is\n  not checking it and neither is anybody else. "
              "'not running' means\n  the service is not in this stack at all.")
        return 1
    print("PASS: every named service reports healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
