#!/usr/bin/env python3
"""Item 8's six results, on two axes that may not launder one another.

WHY TWO AXES, AND A THIRD COLUMN
================================

This run does two jobs at once:

* **Axis 1 — the HuggingFace/network contingency.** Does the retry loop
  recover from a failing fetch and refuse after persistent denial?
* **Axis 2 — image provenance**, and the collectors' first qualification
  against a real Docker daemon.

Frozen R2: *"A collector fault leaves Axis 1's result standing and leaves
item 10's provenance unmoved; a clean binding cannot turn a failed
contingency into a success."*

The first implementation of this pair had **one** verdict field. A failed
`.Image` binding rewrote it to UNMEASURED, and this file then printed
that field under "AXIS 1". So an image-provenance fault silently became a
contingency measurement — precisely the laundering R2 forbids. Caught in
adversarial review before any build existed.

Three columns now, and they are computed independently:

    axis1_verdict          PASS / WRONG_FAILURE / UNMEASURED
    axis2_provenance       BOUND / MISMATCH / UNRECORDED /
                           IMAGE_NOT_PRODUCED_BY_DESIGN
    qualified_for_closure  true only when BOTH are sound

WHY A ROW COUNT IS NOT A DENOMINATOR
====================================

The first implementation keyed rows by `(image, branch)` into a dict —
which silently collapses duplicates — and then checked only
`len(rows) == 6`. Six rows containing a duplicate and a missing branch
would have satisfied it while one of the six precommitted subjects had
never been measured at all.

**A denominator is the set of precommitted subjects, not a number of
lines.** This requires exactly the six expected keys, each once, no
extras, before any conclusion is drawn — and reports the mismatch
precisely when it is not so.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

# The frozen denominator, in the frozen order.
EXPECTED = [(i, b) for i in ("memu-core", "memu-graph")
            for b in ("B1", "B2", "B3")]

PASS = "PASS"
WRONG = "WRONG_FAILURE"
UNMEASURED = "UNMEASURED"
SOUND_A2 = {"BOUND", "IMAGE_NOT_PRODUCED_BY_DESIGN"}


def qualifies(r: dict) -> tuple[bool, str]:
    """Closure qualification is DERIVED HERE, not trusted from the runner.

    The runner produces observations and per-axis classifications. It
    does not certify the composite claim — an observation producer that
    also certifies the conclusion drawn from it is a second authority for
    the same statement, and rule 26 says no consequential mechanism
    self-approves. So this recomputes it from the row's evidence, and a
    runner that shipped a `qualified_for_closure` field would be
    contradicted rather than believed.
    """
    if r.get("axis1_verdict") != PASS:
        return False, f"Axis 1 is {r.get('axis1_verdict')}"
    a2 = r.get("axis2_provenance")
    if a2 not in SOUND_A2:
        return False, f"Axis 2 is {a2}"
    if r.get("branch") == "B3":
        return True, "refused by design, no image to bind"
    # Positive branches need the iidfile corroboration R2 requires.
    # ABSENT is not "no objection": it is the corroboration missing.
    corr = r.get("iidfile_corroboration")
    if corr != "CORROBORATED":
        return False, f"iidfile corroboration is {corr}"
    return True, "Axis 1 PASS, bound, iidfile corroborated"


def refuse(reason: str, detail: str = "") -> int:
    print("ITEM 8 UNMEASURED — EXPERIMENT INSTRUMENT FAILURE")
    print(f"  unmet prerequisite: {reason}")
    if detail:
        print(f"  {detail}")
    print("  No conclusion is drawn about the contingency from a partial "
          "or malformed result set.")
    return 4


def validate_keys(rows: list[dict]) -> tuple[bool, list[str]]:
    """Exactly the six precommitted subjects, each exactly once."""
    seen = [(r.get("image"), r.get("branch")) for r in rows]
    problems: list[str] = []
    for key in EXPECTED:
        n = seen.count(key)
        if n == 0:
            problems.append(f"MISSING: {key[0]}/{key[1]} was never reported")
        elif n > 1:
            problems.append(f"DUPLICATE: {key[0]}/{key[1]} reported {n} times")
    for key in sorted(set(seen) - set(EXPECTED)):
        problems.append(f"UNEXPECTED: {key[0]}/{key[1]} is not a "
                        f"precommitted subject")
    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_file():
        return refuse(f"{path} does not exist",
                      "No branch result was recorded, so there is nothing "
                      "to report.")
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines()
                if l.strip()]
    except json.JSONDecodeError as e:
        return refuse(f"{path} is not readable as JSONL: {e}")
    if not rows:
        return refuse(f"{path} is empty",
                      "Six branches were precommitted and none reported.")

    ok, problems = validate_keys(rows)

    print("ITEM 8 — HUGGINGFACE/NETWORK CONTINGENCY")
    print("=" * 74)
    print()
    print("AXIS 1 — the contingency (computed with NO identity input)")
    print("-" * 74)
    for image, branch in EXPECTED:
        matches = [r for r in rows if (r.get("image"), r.get("branch"))
                   == (image, branch)]
        if not matches:
            print(f"  {image:<12} {branch}  NOT REPORTED")
            continue
        for r in matches:
            print(f"  {image:<12} {branch}  {r.get('axis1_verdict', '?'):<14}"
                  f" retries={r.get('genuine_retries_observed', '?')}"
                  f" elapsed={r.get('elapsed_seconds', '?')}s")
            if r.get("note"):
                print(f"  {'':<12}     {r['note']}")

    print()
    print("AXIS 2 — provenance (separate; may block closure, never Axis 1)")
    print("-" * 74)
    for image, branch in EXPECTED:
        for r in [r for r in rows if (r.get("image"), r.get("branch"))
                  == (image, branch)]:
            q, why = qualifies(r)
            print(f"  {image:<12} {branch}  "
                  f"{r.get('axis2_provenance', 'UNRECORDED'):<30}"
                  f" iidfile={r.get('iidfile_corroboration', 'n/a'):<14}"
                  f" qualifies={'yes' if q else 'NO'}")
            if not q:
                print(f"  {'':<12}     {why}")

    a1 = {v: sum(1 for r in rows if r.get("axis1_verdict") == v)
          for v in (PASS, WRONG, UNMEASURED)}
    a2_sound = sum(1 for r in rows if r.get("axis2_provenance") in SOUND_A2)
    quals = {(r.get("image"), r.get("branch")): qualifies(r) for r in rows}
    qualified = sum(1 for v in quals.values() if v[0])

    # A runner that certifies its own composite claim is contradicted,
    # not trusted. Nothing currently emits this field; if something does,
    # a disagreement is a finding.
    for r in rows:
        if "qualified_for_closure" in r:
            got, why = qualifies(r)
            if bool(r["qualified_for_closure"]) != got:
                print(f"  DISAGREEMENT: {r.get('image')}/{r.get('branch')} "
                      f"row claims qualified={r['qualified_for_closure']}, "
                      f"derived {got} ({why})")

    print()
    print(f"  inspected: {len(rows)} result row(s) against "
          f"{len(EXPECTED)} precommitted subject(s)")
    print(f"    AXIS 1   PASS {a1[PASS]}  WRONG_FAILURE {a1[WRONG]}  "
          f"UNMEASURED {a1[UNMEASURED]}")
    print(f"    AXIS 2   sound {a2_sound} of {len(rows)}")
    print(f"    QUALIFIED FOR CLOSURE  {qualified} of {len(EXPECTED)}")

    print()
    print("  Reading rules, frozen before these results existed:")
    print("   * B2 measures recovery from an INJECTED FETCH-COMMAND failure.")
    print("     It does NOT measure recovery from a real network outage.")
    print("   * UNMEASURED is never an adverse result about the contingency.")
    print("   * WRONG_FAILURE is never a PASS and never a FAIL of it.")
    print("   * An Axis-2 fault blocks closure and leaves Axis 1 standing.")
    print("   * Closure qualification is DERIVED here from the evidence,")
    print("     never taken from the producer of it (rule 26).")
    print("   * No re-draws. An UNMEASURED branch stays UNMEASURED and Item 8")
    print("     is incomplete for that subject. (D247 §5, D289)")

    if not ok:
        print()
        for p in problems:
            print(f"FAIL: {p}")
        print()
        print("The denominator is the six precommitted subjects, not a count "
              "of lines. It is not adjusted downward, and a duplicate does "
              "not substitute for a missing subject.")
        return 4

    if a1[PASS] == len(EXPECTED) and qualified == len(EXPECTED):
        print()
        print("ALL SIX QUALIFY: both contingencies recover from an injected "
              "failure and refuse after persistent denial, and every branch "
              "carries sound provenance.")
        return 0

    print()
    if a1[PASS] == len(EXPECTED):
        print(f"AXIS 1 COMPLETE, PROVENANCE INCOMPLETE: 6/6 contingency PASS "
              f"but only {qualified}/6 qualify for closure. The contingency "
              f"result stands; item 10's provenance does not move for the "
              f"branches whose Axis 2 is unsound.")
    else:
        print(f"NOT ALL SIX PASS: Axis 1 {a1[PASS]}/6. Item 8 is not "
              f"satisfied. Every outcome above is banked as it occurred.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
