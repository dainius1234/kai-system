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

    # (c) THE LOAD-BEARING ONE: only the envelope may have changed
    rc, out = git("diff", "--name-only", env["approved_commit"], args.head)
    if rc != 0:
        return refuse(f"could not diff {env['approved_commit'][:12]}..{args.head}")
    changed = [p for p in out.splitlines() if p.strip()]
    extra = [p for p in changed if p != SENTINEL_REL]
    if extra:
        return refuse(
            f"the executed tree is NOT the reviewed tree. Beyond "
            f"{SENTINEL_REL}, {len(extra)} path(s) changed since "
            f"{env['approved_commit'][:12]}:\n  " + "\n  ".join(extra[:20])
            + ("\n  ..." if len(extra) > 20 else "")
            + "\n\nReview approved one artefact; this would run another")

    print("ITEM-8 EXECUTION AUTHORITY")
    print("=" * 68)
    print(f"  envelope        : {SENTINEL_REL}")
    print(f"  frozen design   : {got}  ({size} bytes)")
    print(f"  approved commit : {env['approved_commit']}")
    print(f"  approved tree   : {env['approved_tree']}")
    print(f"  changed since   : {len(changed)} path(s), "
          f"{'only the envelope' if changed == [SENTINEL_REL] else 'NONE'}")
    print()
    print(f"  inspected: 1 authority envelope across 3 binding(s) "
          f"(design, ancestry, tree identity)")
    print()
    print("PASS: the artefact about to run is the artefact that was "
          "reviewed, under the design that was frozen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
