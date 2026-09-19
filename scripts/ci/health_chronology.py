#!/usr/bin/env python3
"""Health events for a container, with elapsed time COMPUTED from StartedAt.

D183 was banked on a judgement about elapsed time rather than a
calculation, and D184 turned that into a rule: compute the offset, never
eyeball it. Docker records `State.StartedAt` and a `State.Health.Log`
whose entries carry their own `Start`/`End`; the difference is the only
thing that answers "did this happen before or after readiness", and it is
arithmetic, not impression.

Prints, per event, the offset from container start, the exit code and the
first line of the probe's output — so a health log that is failing for a
reason says the reason.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime


def _parse(ts: str) -> datetime | None:
    if not ts or ts.startswith("0001-01-01"):
        return None
    # Docker emits nanosecond precision; datetime takes six digits.
    head, _, tail = ts.partition(".")
    if tail:
        digits = "".join(c for c in tail if c.isdigit())[:6].ljust(6, "0")
        ts = f"{head}.{digits}+00:00"
    else:
        ts = head + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: health_chronology.py <container>", file=sys.stderr)
        return 2
    container = argv[1]
    try:
        raw = subprocess.run(
            ["docker", "inspect", container, "--format", "{{json .State}}"],
            capture_output=True, text=True, timeout=30, check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        # I-1: a container we cannot inspect is not a container that is
        # fine. Say which, and fail.
        print(f"REFUSED: cannot inspect {container}: {exc}")
        return 1

    state = json.loads(raw)
    started = _parse(state.get("StartedAt", ""))
    print(f"  Status        {state.get('Status')}")
    print(f"  StartedAt     {state.get('StartedAt')}")
    print(f"  FinishedAt    {state.get('FinishedAt')}")
    print(f"  ExitCode      {state.get('ExitCode')}")

    health = state.get("Health") or {}
    log = health.get("Log") or []
    print(f"  Health.Status {health.get('Status', '(no healthcheck)')}")
    print(f"  Health.FailingStreak {health.get('FailingStreak')}")
    print(f"  health events recorded: {len(log)}")
    if started is None:
        print("  StartedAt is unparseable — offsets NOT COMPUTED, and no "
              "ordering claim is available from this record.")
        return 0
    if not log:
        print("  No health events. Whether readiness was reached is NOT "
              "established by this record.")
        return 0

    first_healthy = None
    for i, event in enumerate(log):
        start = _parse(event.get("Start", ""))
        end = _parse(event.get("End", ""))
        offset = f"{(start - started).total_seconds():8.1f}s" if start else "       ?"
        took = f"{(end - start).total_seconds():5.1f}s" if start and end else "    ?"
        out = (event.get("Output") or "").strip().splitlines()
        line = out[0][:110] if out else ""
        code = event.get("ExitCode")
        print(f"    [{i}] +{offset} took={took} exit={code}  {line}")
        if code == 0 and first_healthy is None and start:
            first_healthy = (start - started).total_seconds()

    if first_healthy is not None:
        print(f"  FIRST PASSING HEALTH PROBE at +{first_healthy:.1f}s from "
              f"StartedAt — this is the readiness boundary anything else "
              f"is measured against.")
    else:
        print("  NO PASSING HEALTH PROBE in the recorded window. The "
              "readiness boundary was never crossed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
