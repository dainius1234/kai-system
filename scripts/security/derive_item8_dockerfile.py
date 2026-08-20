#!/usr/bin/env python3
"""Derive Item-8's experimental Dockerfiles. Never edit the shipped ones.

WHAT THIS IS FOR
================

Frozen design R2 (D285, `0055ead8…8796`) needs three variants of each of
two Dockerfiles. It also forbids, in the same breath, modifying the
shipped files:

> *"Experimental Dockerfiles are derived mechanically from the real ones.
> The shipped Dockerfiles are never modified by this experiment. The
> derivation asserts mutation cardinality — exactly the counts above, no
> more, no fewer; a zero-match silent edit is a failure (rule 18)."*

So this reads a shipped Dockerfile and writes a derived one. It never
writes to the source, and it refuses rather than guessing whenever the
source does not look the way the design says it looks.

THE MUTATION COUNTS ARE THE POINT
=================================

Rule 18: an edit expected to change one target must prove exactly one
intended target changed, and a zero-match silent edit is a failure. That
rule was earned by a string replacement whose anchor no longer matched,
applied without an assertion, silently doing nothing — which is exactly
what would happen here if `memu-core/Dockerfile`'s retry loop were
reworded and this script kept "working".

    B1  0 treatment mutations   (control)
    B2  1 treatment mutation    (first-attempt fetch failure)
    B3  1 treatment mutation    (--network=none on the HF RUN only)

**The pinned syntax line is scaffolding, not a treatment.** It is added
to all three branches identically, so it cannot be a variable between
them, and it is excluded from the counts above. That distinction is
frozen in R2 and is not this script's to reinterpret.

WHY THE ANCHOR IS THE `RUN for attempt` LINE
============================================

Both contingencies open with the same shape — a five-attempt `for` loop
whose body is the fetch. That line is the boundary of the instruction the
experiment acts on, it is derived from the file rather than from a line
number kept beside it (R5), and if it is absent the honest answer is a
refusal, not a best guess at which `RUN` was meant.

WHY B3 CANNOT USE A BUILD-LEVEL FLAG
====================================

`docker build --network=none` denies network to EVERY `RUN`, which kills
`pip install` at `memu-graph/Dockerfile:5` and `:8` long before the build
reaches the HF loop. The branch would fail for the wrong reason and read
as a pass. R2 forbids it; this script implements the per-instruction form
and nothing else.

Exit 0 = the derived file was written and every assertion held.
Exit 1 = refused. The source did not match what the design describes, or
         the mutation count was not exactly what the branch requires.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

# Frozen in R2. Pinned by digest because `docker/dockerfile:1` is a
# moving pointer -- measured 2026-08-18, it resolved to a digest that is
# none of the version tags -- and a one-shot pre-registered experiment
# may not float its own toolchain.
SYNTAX_LINE = ("# syntax=docker/dockerfile:1.9.0@sha256:"
               "fe40cf4e92cd0c467be2cfc30657a680ae2398318afd50b0c80585784c604f28")

# The subjects, and the treatment count each branch is allowed.
BRANCHES = {"B1": 0, "B2": 1, "B3": 1}
IMAGES = ("memu-core", "memu-graph")

# The HF-fetching instruction opens with this, in both Dockerfiles.
#
# UNCHANGED, deliberately. This anchor runs against SHIPPED sources, and
# its strictness is what makes the mutation cardinality meaningful: if
# it also matched a flagged RUN it could match a file this experiment had
# already derived, and a derivation of a derivation would satisfy the
# same assertions while being a different thing.
_RETRY_OPEN = re.compile(r"^RUN for attempt in 1 2 3 4 5; do \\\s*$", re.M)

# ── THE SAME INSTRUCTION, LOCATED IN A *DERIVED* FILE ────────────────
#
# B3's derived Dockerfile carries `RUN --network=none for attempt …`, so
# the shipped-source anchor above does not match it -- correctly, since
# that anchor's job is to find something to derive FROM.
#
# The claim engine needs the opposite direction: given an archived
# derived Dockerfile, which instruction is the subject? That is this,
# and it is here rather than in the summariser so there is exactly ONE
# statement in the tree of what Item 8's target instruction looks like.
# (D298)
_TARGET_OPEN = re.compile(
    r"^RUN (?:--\S+ )*for attempt in 1 2 3 4 5; do \\\s*$", re.M)


def find_target_run(text: str) -> str | None:
    """The whole target RUN of a DERIVED Dockerfile, verbatim."""
    m = _TARGET_OPEN.search(text)
    if not m:
        return None
    start = m.start()
    idx = start
    for line in text[start:].splitlines(keepends=True):
        idx += len(line)
        if not line.rstrip("\n").endswith("\\"):
            break
    return text[start:idx]

# NO INSTRUMENTATION MARKERS. An earlier repair added
# `ITEM8-MARK ATTEMPT=$attempt` to all three branches, because the
# verdict layer was grepping a rendered build log and needed a token
# that could not be forged by the log echoing the instruction.
#
# `--progress=rawjson` makes that unnecessary: BuildKit attributes
# RUNTIME OUTPUT to a vertex separately from the vertex's INSTRUCTION
# TEXT, so the Dockerfiles' own retry lines are sufficient evidence and
# the subject carries no instrumentation at all. The scaffolding is
# removed, and with it the question of whether it counted against the
# frozen mutation cardinality. (D293)

# B2's marker carries NO INTERPOLATED VALUE, and that is deliberate.
#
# An earlier version emitted `ITEM8-B2-INJECTED-ATTEMPT=\$attempt`,
# intending the shell to expand the loop variable at runtime. It does
# not: inside a double-quoted string a backslash before `$` SUPPRESSES
# parameter expansion, so the real container printed the literal text
# `$attempt`. The calibration's fake docker, meanwhile, manufactured
# `=1` -- so the fixture proved a behaviour the shipped derivation did
# not implement, and the fake was semantically BETTER than the real
# command path. Measured against /bin/sh, not reasoned about. (D294)
#
# The number is not needed. The shim's own control flow guarantees the
# injected branch is the FIRST iteration: the sentinel file cannot exist
# before it is created. So the marker is a constant, single-quoted so no
# shell touches it, and the criterion is "exactly one occurrence".
# Another interpolation argument deleted rather than won.
#
# B2's shim. `attempt` is the shell loop variable; on the FIRST iteration
# the sentinel is absent, so we create it and return failure without ever
# running the real command. Every later iteration finds it and runs the
# genuine fetch. Written as a file test rather than a numeric comparison
# on ${attempt} because Docker substitutes ${attempt} -- which is not a
# build arg -- to the empty string before the shell sees it, a defect
# already documented at memu-graph/Dockerfile:105-109.
_B2_SHIM = ("if [ ! -f /tmp/item8-b2-first-attempt-consumed ]; then \\\n"
            "        touch /tmp/item8-b2-first-attempt-consumed; \\\n"
            "        echo 'ITEM8-B2-INJECTED-FIRST-ATTEMPT'; \\\n"
            "        false; \\\n"
            "      else \\\n"
            "        {REAL}; \\\n"
            "      fi")


def refuse(msg: str) -> int:
    print(f"REFUSED: {msg}")
    return 1


def find_retry_run(text: str) -> tuple[int, int] | None:
    """(start, end) character offsets of the whole HF-fetching RUN."""
    m = _RETRY_OPEN.search(text)
    if not m:
        return None
    start = m.start()
    # The instruction ends at the first line that does not continue.
    idx = start
    lines = []
    for line in text[start:].splitlines(keepends=True):
        lines.append(line)
        idx += len(line)
        if not line.rstrip("\n").endswith("\\"):
            break
    return start, idx


def derive(src: str, branch: str) -> tuple[str, int, str]:
    """Return (derived text, treatment mutations applied, error)."""
    span = find_retry_run(src)
    if span is None:
        return "", 0, ("no `RUN for attempt in 1 2 3 4 5; do \\` instruction "
                       "found. The design names that loop as the subject; a "
                       "derivation that cannot find it must not guess which "
                       "RUN was meant")
    a, b = span
    run_text = src[a:b]
    mutations = 0

    if branch == "B3":
        # Per-instruction denial. NOT build-level: that would deny network
        # to pip install too, and the branch would fail for the wrong
        # reason while looking like a pass.
        run_text = run_text.replace("RUN for attempt",
                                    "RUN --network=none for attempt", 1)
        mutations = 1
    elif branch == "B2":
        body_open = "; do \\\n"
        i = run_text.find(body_open)
        if i < 0:
            return "", 0, "the retry loop's body could not be located"
        head = run_text[:i + len(body_open)]
        rest = run_text[i + len(body_open):]
        # The real command is everything up to the `&& exit 0;` that ends
        # the success path.
        marker = "&& exit 0; \\\n"
        j = rest.find(marker)
        if j < 0:
            return "", 0, ("the retry loop has no `&& exit 0;` success "
                           "path; the shim has nothing to wrap")
        real = rest[:j].rstrip()
        if real.endswith("\\"):
            real = real[:-1].rstrip()
        shim = _B2_SHIM.replace("{REAL}", real.strip())
        run_text = head + "      " + shim + " " + marker + rest[j + len(marker):]
        mutations = 1

    out = src[:a] + run_text + src[b:]
    # Scaffolding, applied to every branch identically, excluded from the
    # treatment count by the frozen design.
    out = SYNTAX_LINE + "\n" + out
    return out, mutations, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True, choices=IMAGES)
    ap.add_argument("--branch", required=True, choices=sorted(BRANCHES))
    ap.add_argument("--source", help="override the shipped Dockerfile path")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src_path = pathlib.Path(args.source) if args.source else \
        REPO / args.image / "Dockerfile"
    if not src_path.is_file():
        return refuse(f"{src_path} does not exist. There is nothing to "
                      f"derive from, and an invented Dockerfile is not a "
                      f"derivation")

    src = src_path.read_text()
    out_path = pathlib.Path(args.out)
    if out_path.resolve() == src_path.resolve():
        return refuse("--out is the shipped Dockerfile. R2 forbids modifying "
                      "it, and a derivation that overwrites its own source "
                      "destroys the thing under test")

    derived, applied, err = derive(src, args.branch)
    if err:
        return refuse(err)

    expected = BRANCHES[args.branch]
    if applied != expected:
        return refuse(f"{args.branch} requires exactly {expected} treatment "
                      f"mutation(s); {applied} were applied. Rule 18: an "
                      f"edit that changed the wrong number of targets is a "
                      f"failure, and a zero-match silent edit is the worst "
                      f"of them")
    if args.branch != "B1" and derived == SYNTAX_LINE + "\n" + src:
        return refuse(f"{args.branch} produced a file identical to the "
                      f"source plus scaffolding. The treatment did not "
                      f"land, whatever the counter says")
    if not derived.startswith(SYNTAX_LINE):
        return refuse("the pinned frontend line is missing from the derived "
                      "file; `RUN --network` needs syntax >= 1.3 and a "
                      "floating frontend is forbidden")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(derived)
    sha = hashlib.sha256(derived.encode()).hexdigest()

    print("ITEM-8 DERIVED DOCKERFILE")
    print("=" * 68)
    print(f"  image    : {args.image}")
    print(f"  branch   : {args.branch}")
    print(f"  source   : {src_path.relative_to(REPO) if src_path.is_relative_to(REPO) else src_path}")
    print(f"  out      : {out_path}")
    print(f"  sha256   : {sha}")
    print(f"  frontend : pinned, {SYNTAX_LINE.split('=', 1)[1]}")
    print()
    print(f"  inspected: 1 shipped Dockerfile, {applied} treatment "
          f"mutation(s) of {expected} required")
    print()
    print("PASS: derived without touching the shipped Dockerfile.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
