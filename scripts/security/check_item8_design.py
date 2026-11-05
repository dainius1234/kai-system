#!/usr/bin/env python3
"""The frozen Item-8 design must be intact, or nothing is built.

WHY THIS EXISTS
===============

D288 froze the Item-8 canonical design (D285's region) at

    0055ead8f51d8758bcd6f05b9b1fff84dd9509e91e79c79b6a2500ab78488796

and the freeze carries one clause that makes it a control rather than a
document:

> *"Before any build, the workflow recomputes this design's fingerprint
> and REFUSES if it differs."*

Without that, a frozen design is a sentence in a file, and an amended
experiment running under a frozen experiment's authority is the worst
failure available to us — it would look exactly like the real thing and
be invisible in the record. D275 §2 made the same argument for Stage 2
and this is that argument executed.

WHAT "UNCHANGED" MEANS HERE
===========================

The expected digest is a **literal in this file**, not a value read from
the document it checks. A check that reads its expectation from its own
subject verifies nothing (I-8): the two must come from different places,
and here the second place is this constant, reviewed and committed
separately from the region it pins.

The region is located between two explicit markers rather than by line
number, because line numbers in an append-only log move on every entry.
The markers are assembled from parts at import time for one reason
recorded in D284: writing either marker whole would put a second copy of
it in any file that quotes this module's source, and a region whose
boundaries are ambiguous is not a region.

WHAT IT DOES NOT DO
===================

It does not check that the *implementation* matches the design — that is
a human review, and D285 was frozen precisely so that review has a fixed
target. This proves only that the target has not moved.

Exit 0 = the frozen design is byte-identical to what was frozen.
Exit 1 = it has moved, the decisions file is unreadable, or the region
         cannot be located unambiguously. In every one of those cases
         **no build may proceed.**
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
DECISIONS = REPO / "kai-pm" / "DECISIONS.md"

# D288's frozen value. A literal here, deliberately not read from the
# document, so the expectation and the subject come from different places.
FROZEN_R2 = "0055ead8f51d8758bcd6f05b9b1fff84dd9509e91e79c79b6a2500ab78488796"
FROZEN_BYTES = 7776

# Superseded and DEAD (D286). Recognised so that a tree still carrying the
# old region gets a diagnosis rather than a bare mismatch.
SUPERSEDED = {
    "b8ba2ae363d827b33e8d10c54a44789f35c22f0ad14f04b306897fa416e8ff98":
        "R1 (D283/D284) — superseded by R2 and explicitly dead",
}

_STEM = "### CANONICAL ITEM-8 DESIGN R2 "
BEGIN = _STEM + "— BEGIN"
END = _STEM + "— END"


def region(text: str) -> tuple[str | None, str]:
    """The frozen region, or a refusal explaining why there isn't one."""
    nb, ne = text.count(BEGIN), text.count(END)
    if nb != 1 or ne != 1:
        return None, (f"the region markers appear {nb} and {ne} time(s); "
                      f"exactly one of each is required. A fingerprint over "
                      f"an ambiguous region pins nothing")
    body = text.split(BEGIN, 1)[1].split(END, 1)[0]
    return BEGIN + body, ""


def digest(region_text: str) -> tuple[str, int]:
    """D284's published recipe, and nothing else."""
    norm = "\n".join(" ".join(l.split())
                     for l in region_text.splitlines()).strip()
    raw = norm.encode()
    return hashlib.sha256(raw).hexdigest(), len(raw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=str(DECISIONS))
    ap.add_argument("--expect", default=FROZEN_R2,
                    help="override the expected digest; calibration only")
    ap.add_argument("--quiet", action="store_true",
                    help="print the computed digest only")
    args = ap.parse_args()

    path = pathlib.Path(args.file)
    if not path.is_file():
        print(f"REFUSED: {path} does not exist. A frozen design that cannot "
              f"be read cannot be shown to be unchanged, and 'probably fine' "
              f"is not a freeze.")
        return 1

    found, err = region(path.read_text())
    if err:
        print(f"REFUSED: {err}.")
        return 1

    got, size = digest(found)
    if args.quiet:
        print(got)
        return 0

    print("ITEM-8 FROZEN DESIGN — INTEGRITY BEFORE ANY BUILD")
    print("=" * 68)
    print(f"  expected : {args.expect}")
    print(f"  computed : {got}")
    print(f"  bytes    : {size}  (frozen at {FROZEN_BYTES})")
    print()
    print(f"  inspected: 1 canonical region across 1 frozen design")

    if got == args.expect:
        print()
        print("PASS: the frozen Item-8 design is byte-identical to what "
              "D288 froze.")
        return 0

    print()
    if got in SUPERSEDED:
        print(f"FAIL: this tree carries {SUPERSEDED[got]}.")
        print("      A superseded design must never run under a frozen")
        print("      design's authority.")
    else:
        print("FAIL: the frozen Item-8 design has MOVED.")
        print("      An amended experiment running under a frozen design's")
        print("      authority is the worst failure available to us, and it")
        print("      would be invisible without this check.")
    print()
    print("NO BUILD MAY PROCEED. Amending a frozen design requires a new")
    print("pre-registration, a new fingerprint and a new operator act —")
    print("never an edit. (D288)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
