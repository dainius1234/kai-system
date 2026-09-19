#!/usr/bin/env python3
"""The CI summariser must survive the shapes the API actually returns.

Calibrated against a confirmed defect, not a suspected one: on
2026-08-07 the inline version of this parser crashed with `KeyError:
'conclusion'` because a run still in progress has no such key. That is
assertion one.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.ci.summarise_runs import load, main, minutes  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def run(path: Path, *flags: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(["prog", str(path), *flags])
    return buf.getvalue()


def main_test(tmp: Path) -> None:
    in_progress = {"run_number": 715, "id": 1, "head_sha": "abc12345def",
                   "status": "in_progress",
                   "created_at": "2026-08-07T10:00:00Z",
                   "updated_at": "2026-08-07T10:01:00Z"}
    finished = {"run_number": 714, "id": 2, "head_sha": "beef0000cafe",
                "status": "completed", "conclusion": "failure",
                "created_at": "2026-08-07T09:00:00Z",
                "updated_at": "2026-08-07T09:25:00Z"}

    runs = tmp / "runs.json"
    runs.write_text(json.dumps({"workflow_runs": [in_progress, finished]}))

    out = run(runs)
    check("an in-progress run does not crash the parser",
          "715" in out)          # the KeyError that started this
    check("a missing conclusion renders as a dash", "—" in out)
    check("a real conclusion is shown", "failure" in out)
    check("elapsed minutes are computed", "25.0m" in out)
    check("sha is truncated, not full", "abc12345" in out
          and "abc12345def" not in out)
    check("denominator is reported", "inspected: 2 run(s)" in out)
    check("--limit is honoured", "714" not in run(runs, "--limit=1"))

    # I-1 — an empty result must say so rather than print nothing.
    empty = tmp / "empty.json"
    empty.write_text(json.dumps({"workflow_runs": []}))
    check("empty list says so rather than printing silence",
          "no workflow_runs" in run(empty))

    # The tool result is prose-wrapped in practice; load must cope.
    wrapped = tmp / "wrapped.txt"
    wrapped.write_text('Error: too big. Saved to disk.\n'
                       + json.dumps({"workflow_runs": [finished]})
                       + "\ntrailing prose\n")
    check("JSON is extracted from a prose-wrapped dump",
          (load(wrapped).get("workflow_runs") or [{}])[0].get("id") == 2)

    # ── jobs mode ──
    jobs = tmp / "jobs.json"
    jobs.write_text(json.dumps({"jobs": {"jobs": [{
        "id": 9, "name": "test", "conclusion": "failure", "steps": [
            {"number": 1, "name": "Checkout", "conclusion": "success"},
            {"number": 45, "name": "Build the images",
             "conclusion": "failure",
             "started_at": "x", "completed_at": "y"}]}]}}))
    out_jobs = run(jobs, "--jobs")
    check("jobs mode names the failing step",
          "step 45" in out_jobs and "Build the images" in out_jobs)
    check("jobs mode reports its denominator",
          "inspected: 2 step(s)" in out_jobs)
    check("jobs mode does not list the passing steps",
          "Checkout" not in out_jobs)

    # The shape that once cost a day: job failed, no step marked failed,
    # because GitHub refused to schedule a schema-invalid workflow.
    ghost = tmp / "ghost.json"
    ghost.write_text(json.dumps({"jobs": {"jobs": [
        {"id": 9, "name": "test", "conclusion": "failure", "steps": []}]}}))
    check("a job that failed with no failing step is called out",
          "NO STEP IS MARKED FAILED" in run(ghost, "--jobs"))

    check("minutes() degrades rather than raising on a bad timestamp",
          minutes({"created_at": "nonsense", "updated_at": "x"}).strip() == "?")


if __name__ == "__main__":
    import tempfile
    print("CI summariser tests")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as d:
        main_test(Path(d))
    print("=" * 60)
    print(f"CI summariser tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    sys.exit(1 if FAILED else 0)
