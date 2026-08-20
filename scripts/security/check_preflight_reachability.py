#!/usr/bin/env python3
"""The measurement path must not be able to reach the experiment.

WHY GREPPING THE YAML WAS THE WRONG ALTITUDE
============================================

`item8-preflight.yml` exists so the operator can authorise a MEASUREMENT
without authorising the six-build experiment. I argued it was
structurally incapable of reaching the experiment, and my evidence was:

    grep -rn 'run_item8_experiment' .github/workflows/item8-preflight.yml
    → no output

The workflow ran `scripts/test_item8_verdicts.py`, which opens with
`RUNNER = .../run_item8_experiment.sh` and executes it. So the path was
one file away, and the absence I pointed at was the absence of a *name*
in *one* file — not the absence of a *path* through the graph.

**A reachability claim needs a reachability check.** This computes the
transitive closure of what the preflight workflow can execute, and
refuses if the forbidden set is inside it.

WHAT IS FORBIDDEN, AND WHY EACH
===============================

    run_item8_experiment.sh     spends the frozen denominator
    derive_item8_dockerfile.py  produces the subjects it would spend it on
    summarise_item8.py          turns subject evidence into a closure claim

None of the three has any business on a path whose whole purpose is to
answer one question about the daemon before anybody is allowed to build
anything.

HOW THE CLOSURE IS BUILT
========================

From the workflow's own `run:` blocks — not from a list kept beside it
(R5) — every referenced repository script becomes a root. Each root is
then scanned for references to other repository scripts, and so on until
nothing new appears. The scan is deliberately crude in the safe
direction: it matches a script's *basename anywhere in the text*, so a
mention in a comment counts as a reference. A reachability check that
under-reports is worse than one that over-reports, and this one is built
to over-report.

Exit 0 = the forbidden set is not reachable, and the closure is printed.
Exit 1 = it is reachable, with the path that reaches it.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

WORKFLOW = ".github/workflows/item8-preflight.yml"

# Derived from what each one DOES, stated in the docstring above.
FORBIDDEN = (
    "run_item8_experiment.sh",
    "derive_item8_dockerfile.py",
    "summarise_item8.py",
)

_SCRIPT = re.compile(r"scripts/[A-Za-z0-9_/]+\.(?:py|sh)")


def scripts_in(text: str) -> set[str]:
    return set(_SCRIPT.findall(text))


def closure(roots: set[str]) -> tuple[set[str], dict[str, str]]:
    """Everything reachable, and who first reached each thing."""
    seen: set[str] = set()
    via: dict[str, str] = {}
    frontier = list(roots)
    for r in roots:
        via[r] = WORKFLOW
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        f = REPO / cur
        if not f.is_file():
            continue
        for nxt in scripts_in(f.read_text(errors="replace")):
            if nxt not in seen:
                via.setdefault(nxt, cur)
                frontier.append(nxt)
        # A basename mentioned anywhere counts, even without a path --
        # deliberately over-reporting, because the failure this exists
        # to stop was an under-report.
        text = f.read_text(errors="replace")
        for name in FORBIDDEN:
            if name in text:
                hit = next((s for s in seen | set(frontier)
                            if s.endswith(name)), f"scripts/**/{name}")
                via.setdefault(hit, cur)
                if hit not in seen:
                    frontier.append(hit)
    return seen, via


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workflow", default=WORKFLOW)
    ap.add_argument("--expect-reachable", action="append", default=[],
                    help="calibration: a path that MUST be in the closure, "
                         "so a check that reports 'nothing reachable' "
                         "cannot pass by finding nothing")
    args = ap.parse_args()

    wf = REPO / args.workflow
    if not wf.is_file():
        print(f"REFUSED: {args.workflow} does not exist. A reachability "
              f"claim about a workflow that is not there is not a claim.")
        return 1
    roots = scripts_in(wf.read_text())
    if not roots:
        print(f"REFUSED: {args.workflow} references no repository script at "
              f"all. Either it does nothing, or this check is not reading "
              f"it — and a closure of zero is exactly what a broken "
              f"extractor produces (R11).")
        return 1

    reached, via = closure(roots)

    print("PREFLIGHT REACHABILITY")
    print("=" * 68)
    print(f"  workflow : {args.workflow}")
    print(f"  roots    : {len(roots)}")
    for r in sorted(roots):
        print(f"      {r}")
    print(f"  reachable: {len(reached)}")
    for r in sorted(reached - roots):
        print(f"      {r}   (via {via.get(r, '?')})")
    print()

    breaches = []
    for name in FORBIDDEN:
        for r in sorted(reached):
            if r.endswith(name):
                breaches.append((name, r, via.get(r, "?")))
    print(f"  inspected: {len(reached)} reachable script(s) against "
          f"{len(FORBIDDEN)} forbidden target(s)")
    print()

    missing = [e for e in args.expect_reachable if e not in reached]
    if missing:
        print(f"FAIL: the closure does not contain {', '.join(missing)}, "
              f"which calibration says it must. A reachability check that "
              f"finds nothing passes every forbidden test for the wrong "
              f"reason.")
        return 1

    if breaches:
        for name, path, who in breaches:
            print(f"FAIL: {path} is REACHABLE from {args.workflow} via "
                  f"{who}")
        print()
        print("The standalone measurement exists so the operator can "
              "authorise measuring WITHOUT authorising the six builds. A "
              "path from it to the experiment's machinery makes that "
              "distinction a matter of trust rather than of structure, and "
              "the previous claim of incapability was made by grepping one "
              "file for a name. (D301)")
        return 1

    print("PASS: no path from the measurement to the experiment's runner, "
          "deriver or claim engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
