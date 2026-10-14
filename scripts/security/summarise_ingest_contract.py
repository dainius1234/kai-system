#!/usr/bin/env python3
"""KAI-GATE-050: does `/graph/ingest` report success over a failed pipeline?

THE INVARIANT, AS THE OPERATOR STATED IT
========================================

    A boundary may not promote an internal failure into externally
    successful completion. If the API contract is asynchronous/accepted
    semantics, it must say so explicitly and expose a durable operation
    state; otherwise downstream failure must remain failure.

`/graph/ingest` is SYNCHRONOUS by construction — `memu-graph/app.py`
awaits `cognee.cognify(...)` inline and declares a 502 on failure — so
the second clause applies. There is no accepted-semantics defence
available to it.

WHY HTTP 200 IS NOT THE SUCCESS PREDICATE
=========================================

The whole finding is that the 200 is unreliable. Using it to decide
success would be the instrument agreeing with the defect — I-8's rule
that the expected answer must not come from the thing under test.

**The success predicate is cognee's own terminal pipeline status**,
parsed from its log file, and the verdict is the CORRELATION between
that and the HTTP response:

    every pipeline completed  + 2xx  -> CONSISTENT (success reported as success)
    some pipeline failed      + 5xx  -> CONSISTENT (failure reported as failure)
    some pipeline failed      + 2xx  -> SUCCESS-SHAPED FAILURE
    no pipeline status at all        -> UNMEASURED (R11: no subject, no observation)

THE CONTROLS ARE REAL, AND THEY COME FROM THE SAME RUN
======================================================

`/graph/ingest` drives TWO cognee pipelines: `add` (which completed in
run 9) and `cognify` (which failed). So a single request supplies both a
**known-positive** — a pipeline that genuinely completed — and a
**known-negative** — one that genuinely failed, in the same log, from
the same stack. Neither control is synthetic and neither is borrowed
from the assertion being tested.

That matters because on a CPU runner a genuinely successful *cognify*
may never occur, and an instrument that could only be calibrated by
waiting for one would be uncalibratable. Classifying every pipeline run
independently sidesteps that entirely.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Markers derived from cognee/modules/pipelines/operations/
# run_tasks_with_telemetry.py lines 26, 41 and 53 — NOT guessed, and not
# a list maintained beside the thing they describe (R5).
_STARTED = re.compile(r"Pipeline run started: `([^`]+)`")
_COMPLETED = re.compile(r"Pipeline run completed: `([^`]+)`")
_ERRORED = re.compile(r"Pipeline run errored: `([^`]+)`")
# run_tasks.py:147. This is raised and then deliberately NOT re-raised,
# so it can appear with no accompanying `errored` marker at all.
_FAILED = re.compile(r"PipelineRunFailedError: ([^\n\[]+)")

_STATUS = re.compile(r"RETURNED http-post\s+status=(\d+)\s+elapsed=([\d.]+)s")
_BODY = re.compile(r"^BODY (.*)$", re.M)
_ENTERED = re.compile(r"^ENTERED\s+http-post\s+source_id=(\S+)", re.M)


def read(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def pipeline_states(text: Optional[str]) -> Tuple[Dict[str, str], List[str]]:
    """({pipeline: terminal state}, [failure messages]).

    States are COMPLETED / ERRORED / NO-TERMINAL-MARKER. The third is not
    padding: cognee swallows `PipelineRunFailedError` without re-raising
    it, so a pipeline can stop without either terminal marker, and
    calling that "completed" would reproduce the defect inside the
    instrument.
    """
    states: Dict[str, str] = {}
    failures: List[str] = []
    if text is None:
        return states, failures
    for line in text.splitlines():
        m = _STARTED.search(line)
        if m:
            states.setdefault(m.group(1), "NO-TERMINAL-MARKER")
        m = _COMPLETED.search(line)
        if m:
            states[m.group(1)] = "COMPLETED"
        m = _ERRORED.search(line)
        if m:
            states[m.group(1)] = "ERRORED"
        m = _FAILED.search(line)
        if m:
            failures.append(m.group(1).strip())
    return states, failures


def http_result(text: Optional[str]) -> Tuple[Optional[int], str, Optional[str]]:
    """(status, body, source_id) from one probe transcript."""
    if text is None:
        return None, "", None
    sm = _STATUS.search(text)
    bm = _BODY.search(text)
    em = _ENTERED.search(text)
    return (int(sm.group(1)) if sm else None,
            bm.group(1) if bm else "",
            em.group(1) if em else None)


def classify(states: Dict[str, str], failures: List[str],
             status: Optional[int]) -> Tuple[str, str]:
    """(verdict, why). The correlation IS the finding."""
    if status is None:
        return "UNMEASURED", "no HTTP result recorded — nothing was asked, or the probe did not return"
    if not states and not failures:
        return "UNMEASURED", "no cognee pipeline status recorded — the internal outcome is unknown, so the boundary cannot be judged"
    bad = {p: s for p, s in states.items() if s != "COMPLETED"}
    failed = bool(bad) or bool(failures)
    ok_http = 200 <= status < 300
    if failed and ok_http:
        detail = ", ".join(f"{p}={s}" for p, s in bad.items()) or "—"
        return ("SUCCESS-SHAPED FAILURE",
                f"HTTP {status} over a pipeline that did not complete "
                f"({detail}; {len(failures)} failure message(s))")
    if failed and not ok_http:
        return ("CONSISTENT",
                f"HTTP {status} over a failed pipeline — the failure "
                f"propagated, which is the required behaviour")
    if not failed and ok_http:
        return ("CONSISTENT",
                f"HTTP {status} over {len(states)} completed pipeline(s)")
    return ("INVERTED",
            f"HTTP {status} over pipelines that all completed — a failure "
            f"reported where none occurred")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-logs", required=True)
    args = ap.parse_args()
    d = Path(args.stage_logs)

    # Each observation is one clean stack: ingest-N.log + cognee-log-N.log.
    rounds = sorted({int(m.group(1)) for p in d.glob("ingest-*.log")
                     if (m := re.match(r"ingest-(\d+)\.log$", p.name))})
    print(f"  inspected: {len(rounds)} clean-stack observation(s) in {d}")
    if not rounds:
        print("  NOT COLLECTED: no ingest-N.log found. Nothing was measured.")
        print("  KAI-GATE-050 remains UNMEASURED — which is not the same as clean.")
        return 1
    print()

    print("  THE SUCCESS PREDICATE IS COGNEE'S TERMINAL PIPELINE STATUS,")
    print("  NOT THE HTTP CODE. The HTTP code is the thing under suspicion;")
    print("  using it to decide success would be the instrument agreeing")
    print("  with the defect.")
    print()

    verdicts = []
    for n in rounds:
        ingest = read(d / f"ingest-{n}.log")
        cognee = read(d / f"cognee-log-{n}.log")
        status, body, source_id = http_result(ingest)
        states, failures = pipeline_states(cognee)
        verdict, why = classify(states, failures, status)
        verdicts.append(verdict)

        print(f"  OBSERVATION {n} — clean stack, source_id={source_id or '?'}")
        print(f"    cognee pipelines seen: {len(states)}")
        for pipeline, state in states.items():
            mark = "ok " if state == "COMPLETED" else "BAD"
            print(f"      [{mark}] {pipeline}  {state}")
        for f in failures:
            print(f"      FAILURE MESSAGE: {f}")
        print(f"    HTTP: {status if status is not None else 'NO RESULT'}")
        print(f"    BODY: {body[:200] if body else '(none)'}")
        print(f"    VERDICT: {verdict}")
        print(f"      {why}")
        print()

    # I-2 / R4: state the denominator, and never let a partial population
    # read as a clean one.
    laundered = verdicts.count("SUCCESS-SHAPED FAILURE")
    unmeasured = verdicts.count("UNMEASURED")
    print(f"  DENOMINATOR: {len(verdicts)} observation(s); "
          f"{laundered} success-shaped failure(s), "
          f"{verdicts.count('CONSISTENT')} consistent, "
          f"{unmeasured} unmeasured, "
          f"{verdicts.count('INVERTED')} inverted")
    print()

    if unmeasured:
        print("  KAI-GATE-050: UNMEASURED. At least one observation established")
        print("  neither the internal outcome nor the external one. An")
        print("  unmeasured run is not a clean run.")
        return 1
    if laundered:
        print("  KAI-GATE-050: THE INVARIANT IS VIOLATED.")
        print("    A boundary may not promote an internal failure into")
        print("    externally successful completion. /graph/ingest is")
        print("    SYNCHRONOUS — memu-graph/app.py awaits cognify inline and")
        print("    declares a 502 on failure — so accepted-semantics is not")
        print("    available as a defence.")
        print()
        if laundered == len(verdicts):
            print(f"    Reproduced in {laundered}/{len(verdicts)} independent")
            print("    clean stacks.")
        else:
            print(f"    Observed in {laundered} of {len(verdicts)} clean stacks —")
            print("    NOT deterministic on this evidence.")
        print()
        print("    This gate reports the correlation. It does NOT authorise a")
        print("    remedy: raising a timeout, changing the model, or editing")
        print("    the endpoint are three different owners, and one of them")
        print("    would make the acceptance test greener while the operation")
        print("    still failed.")
        return 1

    print("  KAI-GATE-050: no success-shaped failure in this population.")
    print("    Absence in this population is not closure (Rule 7).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
