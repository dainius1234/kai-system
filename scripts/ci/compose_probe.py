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

    **`State` is read too, and it wins.** On 2026-08-06 `agentic` died
    at import (`ModuleNotFoundError: No module named 'system_fsm'`) and
    this function reported it as::

        - agentic: no-healthcheck

    which is false. `agentic` declares a healthcheck; it was *dead*, and
    Docker reports an empty `Health` for a container that is not
    running. So the message sent the reader to the compose file to look
    for a healthcheck that was already there, while the real cause — a
    traceback in the container log — went unmentioned.

    That is the defect this whole programme is about, committed by the
    instrument built to find it: **a diagnostic that reports something
    other than what happened.** A container that is `exited` or
    `restarting` now says so, and says it before anything about
    healthchecks.
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
            health = (entry.get("Health") or "").strip()
            state = (entry.get("State") or "").strip().lower()
            if state and state != "running":
                # Dead, restarting or created. Whatever the healthcheck
                # says about it is beside the point, and saying
                # "no-healthcheck" about a corpse points at the wrong
                # file entirely.
                code = entry.get("ExitCode")
                suffix = f" (exit {code})" if code not in (None, "", 0) else ""
                states[name] = f"{state}{suffix} — check its container log"
            else:
                states[name] = health or "no-healthcheck"
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


#: Lines docker compose prints on every invocation regardless of outcome.
#: They are not errors and burying the real one under them is how a
#: signal becomes wallpaper.
_COMPOSE_NOISE = re.compile(
    r'^\s*time="[^"]*"\s+level=warning|^\s*$|the attribute `version` is obsolete')


def _meaningful(text: str, limit: int = 400) -> str:
    """`text` with compose's unconditional warnings removed."""
    kept = [ln for ln in text.splitlines() if not _COMPOSE_NOISE.search(ln)]
    return "\n".join(kept).strip()[:limit]


def exec_http(compose_file: str, service: str, port: int, method: str,
              path: str, body: dict | None = None,
              runner: Runner = run,
              headers: Dict[str, str] | None = None,
              expect: Tuple[int, ...] | None = None) -> Tuple[bool, str, str]:
    """One HTTP call from inside the container. Returns (ok, detail, body).

    stdlib only, because the service images carry no HTTP client beyond
    what their own healthchecks already use.

    `headers` carries credentials. `expect` names the statuses that count
    as success; without it the rule stays "anything under 500".

    `expect` exists because a gateway that fails closed answers 503 to an
    unauthenticated caller, and a probe that only ever sees 503 cannot
    tell *refusing correctly* from *broken*. Asserting the refusal AND
    the acceptance is the same I-3 discipline the gates keep: prove the
    rule can fail, or you have not shown it is a rule.
    """
    payload = json.dumps(body or {})
    program = (
        "import sys,urllib.request,urllib.error\n"
        f"body={payload!r}\n"
        f"req=urllib.request.Request('http://localhost:{port}{path}',"
        f"method='{method}')\n"
        "req.add_header('Content-Type','application/json')\n"
        f"for k, v in {dict(headers or {})!r}.items():\n"
        "    req.add_header(k, v)\n"
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
        # Say the exit code *and* the output, with compose's own
        # boilerplate stripped. This used to return `(out + err)` alone,
        # so a failed exec reported
        #
        #   FAIL: memorize request failed: time="…" level=warning
        #   msg="The \"DB_PASSWORD\" variable is not set…"
        #
        # — three lines of compose wallpaper, no exit code, and no hint
        # that `docker compose exec` itself had failed rather than the
        # service having answered badly. The warning is not the error;
        # printing it as though it were sends the reader after the wrong
        # thing entirely.
        return False, f"docker compose exec failed (exit {code}): " \
                      f"{_meaningful(out + err) or '(no output)'}", ""
    match = re.search(r"STATUS (\d+)", out)
    if not match:
        # Say what was wrong *and* what was seen. Returning the raw output
        # alone left the reader to work out that the problem was the shape
        # of the reply rather than its content.
        seen = (out + err).strip()[:200]
        return False, f"no status line in the reply: {seen or '(no output)'}", ""
    status = int(match.group(1))
    payload_text = out.split("\n", 1)[1] if "\n" in out else ""
    if expect is not None:
        ok = status in expect
        wanted = " or ".join(str(s) for s in expect)
        detail = (f"HTTP {status}" if ok
                  else f"HTTP {status}, expected {wanted}")
        return ok, detail, payload_text
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
        print("\n  'exited'/'restarting' means the container is dead or "
              "looping — read its\n  container log, not the compose file. "
              "'no-healthcheck' means it is\n  running and declares none, "
              "so Docker is not checking it and neither\n  is anybody else. "
              "'not running' means the service is not in this stack\n  at "
              "all.")
        return 1
    print("PASS: every named service reports healthy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
