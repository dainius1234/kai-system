"""Restart-persistence smoke test for memu-core.

Verifies that a memory written before a container restart is still
retrievable after the container restarts — confirming TurboVec index and
SQLite DB survive restart via the mounted volume.

Usage (called from CI after memu-core is running):
    python3 scripts/test_restart_persistence.py \
        --compose-file docker-compose.minimal.yml \
        --service memu-core

The restart is a *host* operation and the HTTP calls are a *container*
operation, and since `e4655bc` those are not the same place. memu-core
sits on `agent-net` and `data-net`, both declared `internal: true`, so
`http://localhost:8001` from the runner reaches nothing — there is no
published port and no routable address. The step had been aimed at a
closed door since that commit, and nobody noticed because the workflow
was dying forty steps earlier.

So `docker compose restart` stays on the host, where it belongs, and
every HTTP call goes through `docker compose exec` into the container,
with the port read from the service's own healthcheck rather than passed
in. `--url` is still accepted for a stack that does publish ports.

Exit 0 on success, 1 on failure.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import urllib.error
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ci.compose_probe import exec_http, load_ports  # noqa: E402


class _Caller:
    """Makes an HTTP call to the service, wherever the service can be reached.

    Two modes, because a stack with published ports and a stack behind
    `internal: true` networks are genuinely different situations and
    pretending otherwise is what broke this script.
    """

    def __init__(self, compose_file: str, service: str, url: str | None):
        self.compose_file = compose_file
        self.service = service
        self.base = url.rstrip("/") if url else None
        self.port = None
        if not self.base:
            ports = load_ports(compose_file)
            if service not in ports:
                raise RuntimeError(
                    f"{service} declares no health port in {compose_file}, so "
                    f"there is no address to call it on. Give it a healthcheck "
                    f"or pass --url.")
            self.port = ports[service]

    def where(self) -> str:
        return self.base or f"{self.service} (via docker compose exec, port {self.port})"

    def call(self, method: str, path: str, body: dict | None = None):
        if self.base:
            data = json.dumps(body or {}).encode() if method != "GET" else None
            req = urllib.request.Request(
                f"{self.base}{path}", data=data, method=method,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        ok, detail, payload = exec_http(
            self.compose_file, self.service, self.port, method, path, body)
        if not ok:
            # The body, not just the status. `exec_http` returns what the
            # service actually said and this used to drop it on the
            # floor, so a failure here read as `HTTP 500` and nothing
            # else — the reader then has no way to tell a crashed
            # handler from a rejected payload without another CI round
            # trip. Same class as the three instrument defects fixed
            # earlier today: reporting less than the instrument knows.
            body_text = (payload or "").strip()
            raise RuntimeError(
                f"{method} {path} -> {detail}"
                + (f"; body: {body_text[:400]}" if body_text
                   else "; the service returned no body"))
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{detail}: response was not JSON ({exc})")


def _query(path: str, params: dict) -> str:
    import urllib.parse
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    return f"{path}?{qs}"


def _wait_healthy(caller: "_Caller", timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if caller.call("GET", "/health").get("status") == "ok":
                return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"{caller.where()} not healthy after {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=None,
                        help="only for a stack that publishes host ports; "
                             "otherwise calls go through docker compose exec")
    parser.add_argument("--compose-file", default="docker-compose.minimal.yml")
    parser.add_argument("--service", default="memu-core")
    args = parser.parse_args()

    try:
        caller = _Caller(args.compose_file, args.service, args.url)
    except Exception as exc:
        print(f"FAIL: cannot address {args.service}: {exc}")
        return 1
    base = caller.where()
    compose_cmd = ["docker", "compose", "-f", args.compose_file]
    marker_event = "ci-restart-persistence-test"
    marker_user = "ci-restart-persist"

    print(f"[1/4] Writing marker memory via {base} ...")
    try:
        result = caller.call(
            "POST", "/memory/memorize",
            {
                "timestamp": "2026-01-01T00:00:00Z",
                "event_type": marker_event,
                "result_raw": "persistence probe — must survive restart",
                "user_id": marker_user,
            },
        )
    except Exception as exc:
        print(f"FAIL: memorize request failed: {exc}")
        return 1

    if result.get("status") != "ok":
        print(f"FAIL: memorize returned non-ok status: {result}")
        return 1
    print(f"     memorize ok: {result}")

    print(f"[2/4] Restarting container '{args.service}' ...")
    try:
        subprocess.run(
            compose_cmd + ["restart", args.service],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"FAIL: docker compose restart failed: {exc.stderr.decode()}")
        return 1

    print(f"[3/4] Waiting for health via {base} ...")
    try:
        _wait_healthy(caller, timeout=90)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("     health ok after restart")

    print(f"[4/4] Retrieving memories for user '{marker_user}' ...")
    try:
        records = caller.call("GET", _query(
            "/memory/retrieve",
            {"query": marker_event, "user_id": marker_user, "top_k": 50}))
    except Exception as exc:
        print(f"FAIL: retrieve request failed: {exc}")
        return 1

    if not isinstance(records, list):
        print(f"FAIL: retrieve returned non-list: {records}")
        return 1

    found = any(
        (r.get("event_type") == marker_event or
         marker_event in str(r.get("content", "")))
        for r in records
    )
    if not found:
        event_types = [r.get("event_type") for r in records[:10]]
        print(f"FAIL: marker record not found after restart. Got event_types: {event_types}")
        return 1

    print(f"PASS: marker memory survived memu-core restart ({len(records)} records returned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
