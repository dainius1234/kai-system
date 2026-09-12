#!/usr/bin/env python3
"""Compact a GitHub Actions API dump into the four fields anyone wants.

Why this exists
---------------

The MCP `list_workflow_runs` call returns ~440,000 characters — the full
repository object is repeated inside every run — and the harness spills
it to a file rather than into the conversation. On 2026-08-07 I wrote
essentially this same parser inline **four times**, each with slightly
different fields, and one of those attempts crashed on a `conclusion`
key that is absent while a run is still in progress.

That is the "list beside the thing" defect wearing work clothes: the
same logic re-derived at each use instead of named once. Fixing the
class rather than the instance applies to my own workflow too (R6).

Usage
-----

    python scripts/ci/summarise_runs.py <saved.json> [--limit N]
    python scripts/ci/summarise_runs.py <saved.json> --jobs

`--jobs` reads a `list_workflow_jobs` dump instead and prints the failing
steps, which is the other thing always wanted and never wanted in full.

Reads a file rather than calling the API on purpose: the API is not
reachable from the authoring environment (the proxy returns 403 and
GH_TOKEN is empty), so the only route is the MCP tool. This takes what
that tool already saved.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path


def load(path: Path) -> dict:
    """The saved tool result, which may have prose wrapped around it."""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise SystemExit(f"{path}: no JSON object found in the file")
    return json.loads(match.group(0))


def minutes(run: dict) -> str:
    try:
        a = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00"))
        return f"{(b - a).total_seconds() / 60:5.1f}m"
    except (KeyError, ValueError):
        return "    ?"


def show_runs(doc: dict, limit: int) -> int:
    runs = doc.get("workflow_runs") or []
    if not runs:
        # I-1: an empty list is a fact about the query, not a clean bill.
        print("  no workflow_runs in this dump — wrong file, or the "
              "filter matched nothing")
        return 0

    print(f"  inspected: {len(runs)} run(s) in the dump, showing "
          f"{min(limit, len(runs))}")
    print()
    for r in runs[:limit]:
        # .get on every field: `conclusion` is absent while in_progress,
        # and the first version of this crashed on exactly that.
        print(f"  #{r.get('run_number'):<5} {str(r.get('id')):<12} "
              f"{(r.get('head_sha') or '')[:8]}  "
              f"{str(r.get('status')):<12} "
              f"{str(r.get('conclusion') or '—'):<10} {minutes(r)}")
    return 0


def show_jobs(doc: dict, limit: int) -> int:
    jobs = (doc.get("jobs") or {}).get("jobs") or doc.get("jobs") or []
    if isinstance(jobs, dict):
        jobs = jobs.get("jobs") or []
    if not jobs:
        print("  no jobs in this dump — wrong file?")
        return 0

    for j in jobs:
        steps = j.get("steps") or []
        bad = [s for s in steps if s.get("conclusion") == "failure"]
        done = [s for s in steps if s.get("conclusion") == "success"]
        print(f"  job {j.get('id')}  {j.get('name')}  "
              f"{j.get('conclusion')}")
        print(f"    inspected: {len(steps)} step(s); {len(done)} success, "
              f"{len(bad)} failure")
        for s in bad[:limit]:
            print(f"    FAILED  step {s.get('number')}: {s.get('name')}")
            print(f"            {s.get('started_at')} -> "
                  f"{s.get('completed_at')}")
        if not bad and j.get("conclusion") == "failure":
            # The job failed but no step is marked failure — that is the
            # schema-invalid-workflow shape, which cost a day once.
            print("    NO STEP IS MARKED FAILED, yet the job failed. That "
                  "is the signature of a\n    workflow GitHub refused to "
                  "schedule — check the file parses and that every\n    "
                  "step has a `run` or a `uses`.")
    return 0


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    flags = {a for a in argv[1:] if a.startswith("--")}
    if not args:
        print(__doc__)
        return 2

    path = Path(args[0])
    if not path.is_file():
        raise SystemExit(f"{path}: not a file")

    limit = 8
    for flag in flags:
        if flag.startswith("--limit="):
            limit = int(flag.split("=", 1)[1])

    doc = load(path)
    return show_jobs(doc, limit) if "--jobs" in flags else show_runs(doc, limit)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
