#!/usr/bin/env python3
"""KAI-GATE-050 probes. Stdlib only — runs INSIDE the image.

    ingest <budget> <source_id>   POST /graph/ingest, report status+body
    cognee-log                    dump the newest cognee log FILE IN FULL

WHY A SECOND PROBE RATHER THAN REUSING THE STALL ONE
====================================================

KAI-GATE-049 asked *where the time went*. This asks *whether the answer
the caller received matches what happened inside*. Different question,
different evidence: the body and the pipeline terminal status, not CPU
and sockets.

WHY `cognee-log` DUMPS THE WHOLE FILE
=====================================

Run 9 sampled cognee's log file every 20s and stopped when the request
returned. The pipeline failed at 19:06:51 — *after* the last sample — so
the terminal status of the failing pipeline was never captured, and the
one line that would say what cognee concluded is missing from the
record. R10: the full artefact is what stops one blind spot being
swapped for another. The file is ~30 lines; there is no reason to
excerpt it.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.getenv("MEMU_GRAPH_URL", "http://localhost:8061")

# The text is deliberately the same shape as KAI-GATE-049's, so a
# difference in outcome cannot be attributed to a different payload.
TEXT = ("Ada Lovelace wrote the first algorithm for the Analytical "
        "Engine, which Charles Babbage designed in London.")


def ingest(budget: float, source_id: str) -> int:
    payload = json.dumps({"text": TEXT, "source_id": source_id}).encode()
    req = urllib.request.Request(
        f"{BASE}/graph/ingest", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")
    print(f"ENTERED  http-post  source_id={source_id} "
          f"wall={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} "
          f"budget={budget}s")
    sys.stdout.flush()
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=budget) as resp:
            elapsed = time.monotonic() - started
            body = resp.read().decode("utf-8", "replace")
            print(f"RETURNED http-post  status={resp.status} "
                  f"elapsed={elapsed:.1f}s")
            print(f"BODY {body}")
            return 0
    except urllib.error.HTTPError as exc:
        elapsed = time.monotonic() - started
        body = exc.read().decode("utf-8", "replace")
        print(f"RETURNED http-post  status={exc.code} elapsed={elapsed:.1f}s")
        print(f"BODY {body}")
        # An HTTP error IS a return: control came back, and a 5xx here is
        # the CORRECT behaviour for a failed pipeline. Exit 0 either way;
        # the verdict belongs to the analyser, not to the probe.
        return 0
    except Exception as exc:
        elapsed = time.monotonic() - started
        print(f"NO-RETURN http-post {type(exc).__name__}: {exc} "
              f"elapsed={elapsed:.1f}s")
        return 1


def cognee_log() -> int:
    root = os.path.join(os.environ.get("HOME", "/data/home"), ".cognee", "logs")
    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        print(f"NO-COGNEE-LOG {root}: {exc}")
        return 1
    if not names:
        print(f"NO-COGNEE-LOG {root} is empty")
        return 1
    path = os.path.join(root, names[-1])
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
    except OSError as exc:
        print(f"NO-COGNEE-LOG cannot read {path}: {exc}")
        return 1
    lines = content.splitlines()
    print(f"COGNEE-LOG {path} lines={len(lines)} bytes={len(content)} "
          f"(FULL FILE, not an excerpt)")
    for line in lines:
        print(line)
    print(f"  inspected: 1 cognee log file, {len(lines)} line(s)")
    return 0


USAGE = ("usage: probe_ingest_contract.py "
         "{ingest <budget-seconds> <source-id>|cognee-log}")


def parse_argv(argv):
    """(action, budget, source_id, error) — PURE. Nothing is started.

    Separated so a caller's command line can be validated with no
    container and no request. Run 8 of KAI-GATE-049 lost a whole 900s
    observation to `python - "$WINDOW"` — the budget became argv[1], the
    subcommand was absent, and nothing checked until the stack was up.
    `scripts/test_ingest_contract.py` asks THIS function about the
    command lines it reads out of the collector, so the expected answer
    comes from the probe rather than from a rule kept beside it (R5).
    """
    if len(argv) < 2:
        return None, None, None, "no subcommand given. " + USAGE
    action = argv[1]
    if action not in ("ingest", "cognee-log"):
        return None, None, None, f"unknown subcommand {action!r}. " + USAGE
    if action == "cognee-log":
        if len(argv) != 2:
            return None, None, None, "cognee-log takes no arguments. " + USAGE
        return "cognee-log", None, None, ""
    if len(argv) != 4:
        return None, None, None, ("ingest needs a budget AND a source_id: the "
                                  "source_id correlates the request with "
                                  "cognee's own log. " + USAGE)
    try:
        budget = float(argv[2])
    except ValueError:
        return None, None, None, f"budget {argv[2]!r} is not a number. " + USAGE
    if budget <= 0:
        return None, None, None, f"budget {budget} must be positive. " + USAGE
    if not argv[3].strip():
        return None, None, None, "source_id must not be empty. " + USAGE
    return "ingest", budget, argv[3], ""


def main(argv: list[str]) -> int:
    action, budget, source_id, error = parse_argv(argv)
    if error:
        print(error, file=sys.stderr)
        return 2
    if action == "cognee-log":
        return cognee_log()
    return ingest(budget, source_id)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
