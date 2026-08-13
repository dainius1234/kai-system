#!/usr/bin/env python3
"""KAI-GATE-049: which stage owns the silence, from the stall stage logs.

The operator's question, and the only one this answers:

    What exact stage owns the ~291s silence after chunking has already
    completed?

    The critical measurement is where the last "entered" marker occurs
    without its matching "returned" marker.

So this pairs cognee's own task markers, then uses the CPU and socket
samples to say WHY the unpaired one is unpaired. A stage list is not a
diagnosis: "still in extract_graph_and_summarize" is compatible with
slow work, a blocked wait, and a deadlock, and those have three
different owners.

    CPU growing, socket to :11434 open   -> genuinely slow LLM work
    CPU flat,    socket to :11434 open   -> waiting on the delegate
    CPU flat,    no socket               -> stuck somewhere else
    CPU growing, no socket               -> local compute, not the LLM

WHAT THIS DELIBERATELY REFUSES TO CONCLUDE
==========================================

If the probe did not return inside the observation window, that is the
WINDOW ending — not a proven hang. Runs 4 and 6 "reproduced" ~291s
because both used the same 300s client budget; treating an
instrument-determined number as a system property is the mistake this
whole unit exists to undo, and it must not be repeated one level up.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# cognee's own markers, from cognee/modules/pipelines/operations/run_tasks_base.py
_STARTED = re.compile(r"(?:Coroutine|Async Generator) task started: `([^`]+)`")
_COMPLETED = re.compile(r"(?:Coroutine|Async Generator) task completed: `([^`]+)`")
_TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)")


def read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def task_markers(text: Optional[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
    """([(task, first-seen timestamp)] entered-without-returned, [all seen])."""
    if text is None:
        return [], []
    entered: Dict[str, str] = {}
    order: List[str] = []
    completed = set()
    for line in text.splitlines():
        ts = _TS.search(line)
        stamp = ts.group(1) if ts else ""
        m = _STARTED.search(line)
        if m and m.group(1) not in entered:
            entered[m.group(1)] = stamp
            order.append(m.group(1))
        m = _COMPLETED.search(line)
        if m:
            completed.add(m.group(1))
    unpaired = [(t, entered[t]) for t in order if t not in completed]
    return unpaired, order


def samples(text: Optional[str]) -> List[dict]:
    out = []
    if text is None:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def cpu_verdict(rows: List[dict]) -> Tuple[str, str]:
    """(verdict, detail) from CPU growth and delegate sockets."""
    if len(rows) < 2:
        return "UNKNOWN", (f"{len(rows)} sample(s) — at least 2 are needed to "
                           f"see growth at all")
    hz = rows[0].get("clock_ticks_per_sec") or 100
    first, last = rows[0], rows[-1]
    ticks = ((last.get("pid1_utime_ticks", 0) + last.get("pid1_stime_ticks", 0))
             - (first.get("pid1_utime_ticks", 0) + first.get("pid1_stime_ticks", 0)))
    span = (last.get("monotonic", 0) - first.get("monotonic", 0)) or 1
    cpu_seconds = ticks / hz
    busy_fraction = cpu_seconds / span
    delegate = any(r.get("delegate_connections") for r in rows)
    growing = busy_fraction > 0.05          # 5% of one core, sustained

    detail = (f"{cpu_seconds:.1f}s CPU over {span:.0f}s wall "
              f"({busy_fraction * 100:.1f}% of one core); "
              f"delegate socket seen in "
              f"{sum(1 for r in rows if r.get('delegate_connections'))}"
              f"/{len(rows)} samples")
    if growing and delegate:
        return "SLOW LLM WORK", detail + " — computing AND talking to the delegate"
    if not growing and delegate:
        return "WAITING ON DELEGATE", detail + " — blocked on ollama, not computing"
    if not growing and not delegate:
        return "STUCK ELSEWHERE", detail + " — not computing and not connected"
    return "LOCAL COMPUTE", detail + " — computing with no delegate connection"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-logs", required=True)
    args = ap.parse_args()
    d = Path(args.stage_logs)

    rc_env = read(d / "rc.env") or ""
    env = dict(line.split("=", 1) for line in rc_env.splitlines() if "=" in line)

    expected = ("ingest.log", "samples.log", "service-logs.log",
                "ollama-after.log", "ollama-baseline.log")
    present = [n for n in expected if (d / n).exists()]
    print(f"  inspected: {len(present)} of {len(expected)} expected stage log(s)")
    missing = [n for n in expected if n not in present]
    if missing:
        print(f"  NOT COLLECTED: {', '.join(missing)}")
    print()

    window = env.get("WINDOW", "?")
    budget = env.get("LIVE_CYCLE_BUDGET", "?")
    ingest_rc = env.get("INGEST_RC", "")
    ingest = read(d / "ingest.log") or ""

    print("  THE PREMISE THIS UNIT CORRECTED")
    print(f"    runs 4 and 6 reproduced ~291s because both used the same "
          f"{budget}s client")
    print(f"    budget, not because the system did the same thing twice. This "
          f"run watched")
    print(f"    for {window}s instead.")
    print()

    m = re.search(r"RETURNED http-post\s+status=(\d+)\s+elapsed=([\d.]+)s", ingest)
    if m:
        print(f"  INGEST RETURNED: status={m.group(1)} after {m.group(2)}s")
        print(f"    Control came back. The operation has a measured duration "
              f"and outcome\n    for the first time.")
    elif "NO-RETURN" in ingest:
        nm = re.search(r"NO-RETURN http-post (\S+): .*elapsed=([\d.]+)s", ingest)
        print(f"  INGEST DID NOT RETURN inside {window}s"
              + (f" ({nm.group(1)} at {nm.group(2)}s)" if nm else ""))
        print("    This is the OBSERVATION WINDOW ending, not a proven hang.")
    else:
        print(f"  INGEST OUTCOME NOT ESTABLISHED (probe exit {ingest_rc or '?'})")
    print()

    unpaired, order = task_markers(read(d / "service-logs.log"))
    print(f"  cognee task markers seen: {len(order)}")
    for name in order:
        print(f"    entered  {name}")
    print()
    if unpaired:
        print("  ENTERED WITHOUT RETURNING — the stage that owns the silence:")
        for name, stamp in unpaired:
            print(f"    {name}   (entered {stamp or 'unstamped'})")
        if any(n == "extract_graph_and_summarize" for n, _ in unpaired):
            print("    NOTE: this task is asyncio.gather(extract_graph_from_data,")
            print("    summarize_text) — TWO concurrent LLM paths under one")
            print("    marker, so cognee's own logging cannot say which.")
    elif order:
        print("  every task that was entered also returned — the silence is "
              "NOT an\n  unpaired cognee task")
    else:
        print("  NO cognee task markers found — nothing to pair")
    print()

    rows = samples(read(d / "samples.log"))
    verdict, detail = cpu_verdict(rows)
    print(f"  samples taken: {len(rows)}")
    print(f"  STATE DURING THE SILENCE: {verdict}")
    print(f"    {detail}")
    print()
    print("  This report names a stage and a state. It does NOT authorise a")
    print("  remedy: slow work, a blocked wait and a deadlock have three")
    print("  different owners, and raising a timeout would answer none of")
    print("  them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
