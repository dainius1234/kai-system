#!/usr/bin/env python3
"""Item 8's six verdicts, on two axes that may not launder one another.

WHY TWO AXES
============

Frozen design R2 requires it, and the reason is that this run does two
jobs at once:

* **Axis 1 — the HuggingFace/network contingency.** Does the retry loop
  survive a transient failure and refuse after a persistent one?
* **Axis 2 — the collectors' first qualification against a real Docker
  daemon.** Every one of the existing collector's 67 calibration
  assertions used an injected fake; this is the first time either
  collector meets a daemon.

> *"A collector fault leaves Axis 1's result standing and leaves item 10's
> provenance unmoved; a clean binding cannot turn a failed contingency
> into a success."*

Reporting them in one column would let a provenance failure read as a
contingency failure, or — worse — let a clean image binding decorate a
branch that measured nothing.

WHAT IT REFUSES TO DO
=====================

R11: if the results file is absent or short, this says so and does not
compose a verdict from the branches that happen to exist. Six rows were
promised; fewer than six is a fact about the run, not a smaller
experiment.

It never repairs, re-runs or re-weights anything. D247 §5 and D289: no
re-draws, and an UNMEASURED branch is banked as it occurred.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

EXPECTED = [(i, b) for i in ("memu-core", "memu-graph")
            for b in ("B1", "B2", "B3")]

PASS = "PASS"
WRONG = "WRONG_FAILURE"
UNMEASURED = "UNMEASURED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_file():
        print("ITEM 8 UNMEASURED — EXPERIMENT INSTRUMENT FAILURE")
        print(f"  unmet prerequisite: {path} does not exist. No branch "
              f"result was recorded, so there is nothing to report and "
              f"nothing to conclude about the contingency.")
        return 4

    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not rows:
        print("ITEM 8 UNMEASURED — EXPERIMENT INSTRUMENT FAILURE")
        print(f"  unmet prerequisite: {path} is empty. Six branches were "
              f"precommitted and none reported.")
        return 4

    by_key = {(r.get("image"), r.get("branch")): r for r in rows}

    print("ITEM 8 — HUGGINGFACE/NETWORK CONTINGENCY")
    print("=" * 72)
    print()
    print("AXIS 1 — the contingency")
    print("-" * 72)
    for image, branch in EXPECTED:
        r = by_key.get((image, branch))
        if r is None:
            print(f"  {image:<12} {branch}  NOT REPORTED")
            continue
        print(f"  {image:<12} {branch}  {r['verdict']:<14} "
              f"attempts={r.get('attempts_observed', '?')} "
              f"elapsed={r.get('elapsed_seconds', '?')}s")
        if r.get("note"):
            print(f"  {'':<12}     {r['note']}")

    print()
    print("AXIS 2 — image provenance (separate; cannot move Axis 1)")
    print("-" * 72)
    for image, branch in EXPECTED:
        r = by_key.get((image, branch))
        if r is None:
            continue
        bind = r.get("executed_binding", "see identity artifacts")
        print(f"  {image:<12} {branch}  {r.get('image_state', 'UNRECORDED'):<30} "
              f"{bind}")

    counts = {v: sum(1 for r in rows if r.get("verdict") == v)
              for v in (PASS, WRONG, UNMEASURED)}
    print()
    print(f"  inspected: {len(rows)} branch result(s) of {len(EXPECTED)} "
          f"precommitted")
    for v in (PASS, WRONG, UNMEASURED):
        print(f"    {v:<14} {counts[v]}")

    print()
    print("  Reading rules, frozen before these results existed:")
    print("   * B2 measures recovery from an INJECTED FETCH-COMMAND failure.")
    print("     It does NOT measure recovery from a real network outage.")
    print("   * UNMEASURED is never an adverse result about the contingency.")
    print("   * WRONG_FAILURE is never a PASS and never a FAIL of it.")
    print("   * No re-draws. An UNMEASURED branch stays UNMEASURED and Item 8")
    print("     is incomplete for that subject. (D247 §5, D289)")

    if len(rows) != len(EXPECTED):
        print()
        print(f"INCOMPLETE: {len(rows)} of {len(EXPECTED)} branches reported. "
              f"The denominator is six and it is not adjusted downward.")
        return 4
    if counts[PASS] == len(EXPECTED):
        print()
        print("ALL SIX PASS: both contingencies survive a transient failure "
              "and refuse after a persistent one.")
        return 0
    print()
    print(f"NOT ALL SIX PASS: {counts[PASS]}/{len(EXPECTED)}. Item 8 is not "
          f"satisfied. Every outcome above is banked as it occurred.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
