#!/usr/bin/env python3
"""No Item-8 build starts without an authority envelope that validates.

TWO HOLES THIS CLOSES
=====================

**1. An alternate trigger.** The workflow carries `workflow_dispatch:`
alongside its sentinel push. Stage 1 documents why dispatch is inert
today — GitHub registers workflows from the DEFAULT branch, and this file
lives only on a feature branch. **That is a property of the platform's
current state, not of our design.** If the branch ever merges, dispatch
becomes live and the sentinel stops being the only way in. Relying on a
circumstance is not failing closed, so the guard below runs regardless of
how the job was triggered and refuses when the envelope is absent.

**2. Review implementation A, execute implementation B.** A sentinel that
is merely an empty file authorises *the act*, not *the artefact*. Nothing
would stop the tree changing between the review that approved an
implementation and the push that runs it — and the run would look
perfectly authorised.

So `kai-pm/ITEM8_GO` is an ENVELOPE, not a flag:

    frozen_r2       = <the frozen canonical-design digest>
    approved_commit = <the implementation commit that was reviewed>
    approved_tree   = <its tree>

and this proves three things before any build:

    a. the envelope's `frozen_r2` equals the design digest computed from
       the file right now -- so the envelope cannot authorise a design
       that has since moved;
    b. `approved_commit` is an ancestor of HEAD -- so we are running
       forward from what was reviewed, not from some other history;
    c. HEAD differs from `approved_commit` by **the envelope alone**.
       Anything else in that diff means the executed tree is not the
       reviewed tree.

(c) is the load-bearing one. It is the difference between "somebody was
allowed to run this" and "this exact artefact was allowed to run".

WHAT IT DOES NOT DO
===================

It does not judge the implementation — that is the adversarial review,
and its conclusion is expressed by which commit gets named in the
envelope. This only proves the thing about to run is the thing that was
named.

Exit 0 = authorised, and the executed tree is the approved tree.
Exit 1 = refused. No build may start.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
SENTINEL = REPO / "kai-pm" / "ITEM8_GO"
SENTINEL_REL = "kai-pm/ITEM8_GO"


def git(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr).strip()


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def refuse(msg: str) -> int:
    print(f"REFUSED: {msg}")
    print()
    print("NO ITEM-8 BUILD MAY START. Execution authority is an envelope "
          "that must validate, not a file that must exist. (D291)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sentinel", default=str(SENTINEL))
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--allow-no-ci", action="store_true",
                    help="calibration only: exercise the git-side bindings "
                         "without CI environment variables present")
    args = ap.parse_args()

    path = pathlib.Path(args.sentinel)
    if not path.is_file():
        return refuse(f"{SENTINEL_REL} does not exist. Item 8's execution "
                      f"authority is this envelope; without it nothing is "
                      f"authorised, however the job was triggered")

    env = parse(path.read_text())
    missing = [k for k in ("frozen_r2", "approved_commit", "approved_tree")
               if not env.get(k)]
    if missing:
        return refuse(f"{SENTINEL_REL} is missing {', '.join(missing)}. An "
                      f"envelope that does not name what it authorises "
                      f"authorises nothing")

    # ONE SHOT, FIRST. These controls depend on nothing but the
    # environment, so they cost nothing and give a clearer
    # diagnosis than a git-state refusal that happens to fire
    # earlier. Ordering was found by calibration: the fixtures
    # for these controls could not reach them.
    # FAIL CLOSED ON ABSENCE. The first version accepted a missing
    # variable, so an environment that simply did not set them passed
    # every one-shot control -- absence read as consent. `--allow-no-ci`
    # exists so the calibration can exercise the git-side bindings, and
    # is never used by the workflow.
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT")
    event = os.environ.get("GITHUB_EVENT_NAME")
    if not args.allow_no_ci:
        if attempt is None:
            return refuse("GITHUB_RUN_ATTEMPT is not set. One-shot execution "
                          "cannot be established, and an unestablished "
                          "control is not a satisfied one")
        if event is None:
            return refuse("GITHUB_EVENT_NAME is not set. The trigger cannot "
                          "be established, so sentinel authority cannot be "
                          "distinguished from any other entry path")
    if attempt is not None and attempt != "1":
        return refuse(f"GITHUB_RUN_ATTEMPT={attempt}. A re-run executes the "
                      f"same authorised commit a second time, which is a "
                      f"replacement execution of a frozen no-redraw "
                      f"experiment. A second run needs a second authority")
    if event is not None and event != "push":
        return refuse(f"GITHUB_EVENT_NAME={event}. Only the sentinel push "
                      f"carries execution authority; a manual dispatch does "
                      f"not, whatever the platform permits")


    # (a) the design it authorises must be the design that exists now
    sys.path.insert(0, str(REPO / "scripts" / "security"))
    import check_item8_design as design  # noqa: E402

    found, err = design.region((REPO / "kai-pm" / "DECISIONS.md").read_text())
    if err:
        return refuse(f"the frozen design could not be located: {err}")
    got, size = design.digest(found)
    if env["frozen_r2"] != got:
        return refuse(f"the envelope authorises design {env['frozen_r2']} but "
                      f"the tree now holds {got}. An envelope cannot "
                      f"authorise a design that has since moved")
    if got != design.FROZEN_R2:
        return refuse(f"the design digest {got} is not the frozen value "
                      f"{design.FROZEN_R2}")

    # (b) we must be running forward from what was reviewed
    rc, _ = git("merge-base", "--is-ancestor", env["approved_commit"],
                args.head)
    if rc != 0:
        return refuse(f"approved_commit {env['approved_commit'][:12]} is not "
                      f"an ancestor of {args.head}. The executed history is "
                      f"not a continuation of the reviewed one")

    rc, tree = git("rev-parse", f"{env['approved_commit']}^{{tree}}")
    if rc != 0:
        return refuse(f"approved_commit {env['approved_commit'][:12]} cannot "
                      f"be resolved in this repository")
    if tree != env["approved_tree"]:
        return refuse(f"approved_tree {env['approved_tree'][:12]} does not "
                      f"match the tree of approved_commit ({tree[:12]}). The "
                      f"envelope is internally inconsistent")

    # (c) THE LOAD-BEARING ONE: the executed tree IS the reviewed tree,
    #     plus the ADDITION of the envelope and nothing else.
    rc, out = git("diff", "--name-status", env["approved_commit"], args.head)
    if rc != 0:
        return refuse(f"could not diff {env['approved_commit'][:12]}..{args.head}")
    rows = [l.split("\t") for l in out.splitlines() if l.strip()]
    extra = [r for r in rows if r[-1] != SENTINEL_REL]
    if extra:
        return refuse(
            f"the executed tree is NOT the reviewed tree. Beyond "
            f"{SENTINEL_REL}, {len(extra)} path(s) changed since "
            f"{env['approved_commit'][:12]}:\n  "
            + "\n  ".join(" ".join(r) for r in extra[:20])
            + ("\n  ..." if len(extra) > 20 else "")
            + "\n\nReview approved one artefact; this would run another")
    # ADDED, not merely "the only path that changed". A later edit to an
    # existing sentinel would satisfy the weaker test and authorise a
    # SECOND six-build denominator under the first authorisation.
    if not rows or rows[0][0] != "A":
        status = rows[0][0] if rows else "no change at all"
        return refuse(f"{SENTINEL_REL} shows diff status '{status}', not 'A'. "
                      f"Execution authority is the ADDITION of the envelope; "
                      f"editing an existing one would re-authorise a frozen "
                      f"no-redraw experiment")

    # (d) ONE SHOT. Re-running a workflow reuses the same commit and ref
    #     and would otherwise satisfy every check above.
    # (e) DIRECT CHILD. Ancestry alone permits arbitrary intervening
    #     commits whose changes were later reverted, which would pass (c).
    rc, parents = git("rev-list", "--parents", "-n", "1", args.head)
    if rc != 0 or not parents:
        # A query that failed cannot establish the relationship, and the
        # first version SKIPPED the check in that case -- an instrument
        # failure silently satisfying the control it was meant to apply.
        return refuse(f"could not resolve the parents of {args.head}; the "
                      f"direct-child relationship is unestablished, and "
                      f"unestablished is not satisfied")
    bits = parents.split()
    if len(bits) != 2:
        return refuse(f"{args.head} has {max(len(bits) - 1, 0)} parent(s); "
                      f"execution authority requires exactly one, so the "
                      f"reviewed commit is unambiguous")
    if bits[1] != env["approved_commit"]:
        return refuse(f"{args.head}'s parent is {bits[1][:12]}, not "
                      f"approved_commit {env['approved_commit'][:12]}. "
                      f"Intervening history is not reviewed history")

    print("ITEM-8 EXECUTION AUTHORITY")
    print("=" * 68)
    print(f"  envelope        : {SENTINEL_REL}")
    print(f"  frozen design   : {got}  ({size} bytes)")
    print(f"  approved commit : {env['approved_commit']}")
    print(f"  approved tree   : {env['approved_tree']}")
    print(f"  changed since   : ADD {SENTINEL_REL}, nothing else")
    print(f"  run attempt     : {os.environ.get('GITHUB_RUN_ATTEMPT', 'n/a')}")
    print(f"  event           : {os.environ.get('GITHUB_EVENT_NAME', 'n/a')}")
    print()
    print(f"  inspected: 1 authority envelope across 3 binding(s) "
          f"(design, direct-child ancestry, tree identity, "
          f"one-shot)")
    print()
    print("PASS: the artefact about to run is the artefact that was "
          "reviewed, under the design that was frozen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
