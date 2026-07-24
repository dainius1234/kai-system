"""Restart-persistence smoke test for memu-core.

Verifies that a memory written before a container restart is still
retrievable after the container restarts — confirming TurboVec index and
SQLite DB survive restart via the mounted volume.

Usage (called from CI after memu-core is running):
    python3 scripts/test_restart_persistence.py \
        --url http://localhost:8001 \
        --compose-file docker-compose.minimal.yml \
        --service memu-core

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


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _get(url: str, params: dict | None = None) -> dict | list:
    if params:
        qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{url}?{qs}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _wait_healthy(base_url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=3) as resp:
                if json.loads(resp.read()).get("status") == "ok":
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"memu-core at {base_url} not healthy after {timeout}s")


def main() -> int:
    import urllib.parse

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001")
    parser.add_argument("--compose-file", default="docker-compose.minimal.yml")
    parser.add_argument("--service", default="memu-core")
    args = parser.parse_args()

    base = args.url.rstrip("/")
    compose_cmd = ["docker", "compose", "-f", args.compose_file]
    marker_event = "ci-restart-persistence-test"
    marker_user = "ci-restart-persist"

    print(f"[1/4] Writing marker memory to {base}/memory/memorize ...")
    try:
        result = _post(
            f"{base}/memory/memorize",
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

    print(f"[3/4] Waiting for {base}/health ...")
    try:
        _wait_healthy(base, timeout=90)
    except RuntimeError as exc:
        print(f"FAIL: {exc}")
        return 1
    print("     health ok after restart")

    print(f"[4/4] Retrieving memories for user '{marker_user}' ...")
    try:
        records = _get(
            f"{base}/memory/retrieve",
            {"query": marker_event, "user_id": marker_user, "top_k": 50},
        )
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
