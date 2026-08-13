#!/usr/bin/env python3
"""KAI-GATE-049 diagnostic probes. Stdlib only — runs INSIDE the image.

Two subcommands, both read-only with respect to the system under test:

    ingest    POST /graph/ingest and report monotonic boundaries. Unlike
              scripts/test_graph_live.py this does NOT stop at 300s,
              because 300s is the number under investigation.
    sample    one observation of the container's internal state: cognee's
              own log file, CPU consumed by the serving process, and its
              open connections to the delegate.

WHY A SEPARATE INGEST PROBE EXISTS
==================================

`test_graph_live.py` gives ingest `timeout=300`. Runs 4 and 6 both
reported ~291s of silence in `extract_graph_and_summarize`, and the
arithmetic says why:

    cognee log file created   16:55:42
    + 300s client budget    = 17:00:42   == exactly when C3 gave up
    extract_graph_and_summarize started 16:55:51 -> 291s of "silence"

**The boundary is OUR client's timeout, not the system's.** The two runs
agree because both used the same 300s budget, not because the system did
the same thing twice. Continuing to measure with that client would keep
reproducing our own number — an instrument reporting on itself, which is
R9's watcher and I-8's rule in a new place.

So this probe waits far longer, and the point is to find out what the
operation actually does, not to make a test pass.

WHY THE CPU SAMPLE MATTERS MORE THAN THE LOG
============================================

`/proc/1/stat` utime+stime says whether the serving process is BURNING
CPU or BLOCKED. Those are different diagnoses with different owners:
sustained CPU growth means the work is genuinely slow; flat CPU with an
established socket to the delegate means it is waiting on `ollama`; flat
CPU with no socket means it is stuck somewhere else entirely. A log line
cannot distinguish those and a timeout certainly cannot.
"""
from __future__ import annotations

import json
import os
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
from typing import Optional

BASE = os.getenv("MEMU_GRAPH_URL", "http://localhost:8061")
SOURCE_ID = "kai-gate-049-stall-probe"


def _hexaddr(field: str) -> tuple[str, int]:
    """Decode a /proc/net/tcp address. Little-endian, per the kernel."""
    host_hex, port_hex = field.split(":")
    packed = struct.pack("<I", int(host_hex, 16))
    return socket.inet_ntoa(packed), int(port_hex, 16)


TCP_STATES = {
    "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV",
    "04": "FIN_WAIT1", "05": "FIN_WAIT2", "06": "TIME_WAIT",
    "07": "CLOSE", "08": "CLOSE_WAIT", "09": "LAST_ACK",
    "0A": "LISTEN", "0B": "CLOSING",
}


def connections() -> list[dict]:
    out = []
    try:
        lines = open("/proc/net/tcp", encoding="utf-8").read().splitlines()[1:]
    except OSError:
        return out
    for line in lines:
        f = line.split()
        if len(f) < 4:
            continue
        try:
            local = _hexaddr(f[1])
            remote = _hexaddr(f[2])
        except (ValueError, struct.error):
            continue
        out.append({"local": f"{local[0]}:{local[1]}",
                    "remote": f"{remote[0]}:{remote[1]}",
                    "state": TCP_STATES.get(f[3].upper(), f[3])})
    return out


def cpu_ticks() -> tuple[int, int]:
    """(utime, stime) of pid 1, in clock ticks. Growth = computing."""
    try:
        stat = open("/proc/1/stat", encoding="utf-8").read()
    except OSError:
        return (-1, -1)
    # The comm field can contain spaces and parentheses; split after it.
    tail = stat[stat.rfind(")") + 2:].split()
    try:
        return int(tail[11]), int(tail[12])
    except (IndexError, ValueError):
        return (-1, -1)


def cognee_log_tail(lines: int = 12) -> list[str]:
    """The newest cognee log FILE, which stdout does not carry.

    The container writes /data/home/.cognee/logs/<ts>.log. Nothing has
    ever read it — every diagnosis so far used `docker logs`, which
    carries a different, smaller stream.
    """
    root = os.path.join(os.environ.get("HOME", "/data/home"), ".cognee", "logs")
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return [f"(no cognee log directory at {root})"]
    if not names:
        return [f"(cognee log directory {root} is empty)"]
    path = os.path.join(root, names[-1])
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            content = fh.read().splitlines()
    except OSError as exc:
        return [f"(cannot read {path}: {exc})"]
    return [f"(file {path}, {len(content)} lines total)"] + content[-lines:]


def sample() -> int:
    u, s = cpu_ticks()
    conns = connections()
    delegate = [c for c in conns if c["remote"].endswith(":11434")]
    print(json.dumps({
        "monotonic": round(time.monotonic(), 3),
        "wall": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid1_utime_ticks": u,
        "pid1_stime_ticks": s,
        "clock_ticks_per_sec": os.sysconf("SC_CLK_TCK"),
        "connections_total": len(conns),
        "delegate_connections": delegate,
    }))
    tail = cognee_log_tail()
    for line in tail:
        print(f"  cognee-log | {line}")
    # I-2. A sample that says nothing about how much it saw reads
    # identically whether /proc was readable or empty — and an empty
    # /proc/net/tcp is exactly what a mis-targeted probe would report.
    print(f"  inspected: {len(conns)} connection(s), {len(tail)} cognee "
          f"log line(s), pid-1 cpu {'READ' if u >= 0 else 'UNREADABLE'}")
    return 0


def ingest(budget: float) -> int:
    """POST and WAIT. Reports where control actually was."""
    payload = json.dumps({
        "text": "Ada Lovelace wrote the first algorithm for the Analytical "
                "Engine, which Charles Babbage designed in London.",
        "source_id": SOURCE_ID,
    }).encode()
    req = urllib.request.Request(
        f"{BASE}/graph/ingest", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    print(f"ENTERED  http-post  monotonic={time.monotonic():.3f} "
          f"wall={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
          f"budget={budget}s")
    sys.stdout.flush()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=budget) as resp:
            body = resp.read().decode("utf-8", "replace")
            elapsed = time.monotonic() - started
            print(f"RETURNED http-post  status={resp.status} "
                  f"elapsed={elapsed:.1f}s")
            print(f"BODY {body[:800]}")
            return 0
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - started
        print(f"RETURNED http-post  status={exc.code} elapsed={elapsed:.1f}s")
        print(f"BODY {exc.read().decode('utf-8', 'replace')[:800]}")
        return 0          # an HTTP error IS a return: control came back
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f"NO-RETURN http-post {type(exc).__name__}: {exc} "
              f"elapsed={elapsed:.1f}s")
        print("  This is the observation window ending, NOT a proven hang: "
              "the server may still be working.")
        return 1


USAGE = "usage: probe_graph_stall.py {ingest <budget-seconds>|sample}"


def parse_argv(argv: list[str]) -> tuple[Optional[str], Optional[float], str]:
    """(action, budget, error) — PURE. No I/O, nothing started.

    Separated from `main` so a command line can be validated WITHOUT a
    container, a stack, or a request. Run 8 of the stall diagnosis
    invoked this file as `python - 900`, so argv[1] was the budget and
    the subcommand was absent; the probe correctly refused, but nothing
    checked the command line until the whole 900s observation had been
    scheduled, brought a stack up, and produced an evidence file with no
    measurement in it. A caller that can ask "would you accept this?"
    costs nothing and answers before the stack exists — I-8, evidence
    from somewhere other than the run it is meant to validate.
    """
    if len(argv) < 2:
        return None, None, "no subcommand given. " + USAGE
    action = argv[1]
    if action not in ("ingest", "sample"):
        return None, None, f"unknown subcommand {action!r}. " + USAGE
    if action == "sample":
        if len(argv) != 2:
            return None, None, "sample takes no arguments. " + USAGE
        return "sample", None, ""
    if len(argv) != 3:
        return None, None, ("ingest needs an explicit budget in seconds — "
                            "there is no default, because the budget is the "
                            "thing under investigation. " + USAGE)
    try:
        budget = float(argv[2])
    except ValueError:
        return None, None, f"ingest budget {argv[2]!r} is not a number. " + USAGE
    if budget <= 0:
        return None, None, f"ingest budget {budget} must be positive. " + USAGE
    return "ingest", budget, ""


def main(argv: list[str]) -> int:
    action, budget, error = parse_argv(argv)
    if error:
        print(error, file=sys.stderr)
        return 2
    if action == "sample":
        return sample()
    assert budget is not None      # parse_argv guarantees it for "ingest"
    return ingest(budget)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
