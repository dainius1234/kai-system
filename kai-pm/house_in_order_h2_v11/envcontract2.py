#!/usr/bin/env python3
"""ENVIRONMENT-SUBJECT CAPABILITY CONTRACT — v1.1.

Carried forward from HOUSE_H2 v1.0 with its logic UNCHANGED: it was the
control that caught the depth-1 blindness, and it works. Copied rather
than imported so the v1.1 package is self-contained.

Original header follows.


Kai's wording: a subject fact may only be emitted when the measurement
environment has POSITIVELY DEMONSTRATED the capability required to
observe that fact.

Earned by D338: the H1 subject checkout is depth-1, so `rev-list --count`
returned 1 for all 272 documents. Nothing failed. Every row looked
plausible. The axis was blind, uniformly and silently.

Missing capability -> UNMEASURED / ENVIRONMENT_CAPABILITY_MISSING.
NEVER substitute a plausible value.
"""
from __future__ import annotations
import subprocess

def _git(repo, *a):
    return subprocess.run(["git",*a],cwd=repo,capture_output=True,text=True)

def probe(tree_repo, history_repo, subject_sha):
    """Return (contract_rows, capabilities_ok)."""
    rows=[]

    r=_git(tree_repo,"rev-parse","HEAD")
    rows.append(dict(capability="EXACT_TREE_CHECKOUT",
        required_state=f"HEAD == {subject_sha[:12]}",
        observed_state=r.stdout.strip()[:12],
        evidence="git rev-parse HEAD in the tree source",
        disposition="OK" if r.stdout.strip()==subject_sha else "MISSING"))

    # The `is-shallow` flag is a PROXY, and a misleading one here: this
    # repository is shallow-marked yet reports README.md = 122, because
    # its boundary IS the 2026-08-05 import and no earlier per-file
    # history exists to lose. A proxy would have failed a source that can
    # do the job, and passed a deep source pointed at the wrong subject.
    # The capability is therefore tested DIRECTLY below, and the boundary
    # is recorded as a DECLARED LIMITATION rather than hidden or waived.
    r=_git(history_repo,"rev-parse","--is-shallow-repository")
    shallow=r.stdout.strip()
    b=_git(history_repo,"log","--reverse","--format=%H %ad","--date=short")
    oldest=(b.stdout.splitlines() or ["?"])[0]
    rows.append(dict(capability="HISTORY_BOUNDARY",
        required_state="boundary recorded, not assumed absent",
        observed_state=f"shallow={shallow}; oldest reachable {oldest[:52]}",
        evidence="git rev-parse --is-shallow-repository; git log --reverse",
        disposition="DECLARED_LIMITATION" if shallow=="true" else "OK"))

    # the history source must actually CONTAIN the subject commit
    r=_git(history_repo,"cat-file","-e",f"{subject_sha}^{{commit}}")
    rows.append(dict(capability="SUBJECT_REACHABLE_IN_HISTORY",
        required_state="subject commit object present",
        observed_state="present" if r.returncode==0 else "absent",
        evidence=f"git cat-file -e {subject_sha[:12]}^{{commit}}",
        disposition="OK" if r.returncode==0 else "MISSING"))

    # positive demonstration: a known-multi-commit path must exceed 1
    r=_git(history_repo,"rev-list","--count",subject_sha,"--","README.md")
    try: n=int(r.stdout.strip() or 0)
    except ValueError: n=0
    rows.append(dict(capability="HISTORY_SOURCE_NON_DEGENERATE",
        required_state="a known-maintained path reports > 1 commit",
        observed_state=f"README.md = {n}",
        evidence=f"git rev-list --count {subject_sha[:12]} -- README.md",
        disposition="OK" if n>1 else "MISSING"))

    return rows, all(x["disposition"] in ("OK","DECLARED_LIMITATION")
                     for x in rows)


AXIS_REQUIREMENTS = {
    "LIFECYCLE": ("HISTORY_SOURCE_NON_DEGENERATE",
                  "SUBJECT_REACHABLE_IN_HISTORY"),
    "FUNCTION": ("EXACT_TREE_CHECKOUT",),
    "AUTHORITY": ("EXACT_TREE_CHECKOUT",),
    "GENERATION": ("EXACT_TREE_CHECKOUT",),
    "VALIDITY": ("EXACT_TREE_CHECKOUT",),
    "SCOPE": ("EXACT_TREE_CHECKOUT",),
}

def axis_blocked(axis, rows):
    """Capabilities this axis needs that are NOT demonstrated."""
    have={r["capability"] for r in rows
          if r["disposition"] in ("OK","DECLARED_LIMITATION")}
    return [c for c in AXIS_REQUIREMENTS.get(axis,()) if c not in have]
