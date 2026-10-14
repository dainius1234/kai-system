#!/usr/bin/env python3
"""CALIBRATION — SUBJECT BINDING AGAINST A MOVING SYMBOLIC REF.

This is the permanent regression case for an execution-proven defect.

THE DEFECT. main() resolved the supplied ref to a commit and recorded
that identity, then passed the SYMBOLIC REF down into materialise(),
which dereferenced it AGAIN for `ls-tree` and for `git archive`. If the
ref moved in between, BOTH sides of the expect/got reconciliation saw
the new commit, so they agreed, the run reported "reconciles: True", and
the result was stamped with the OLD commit while containing the NEW
commit's content.

That is the worst shape we know: a silent MISBINDING that presents as a
clean, fully populated, internally consistent table. The reconciliation
control could not see it, because both of its inputs moved together.

THE INVARIANT. Resolve once to an immutable object id; use only that id
for tree derivation, ls-tree, archive, reconciliation and stamping. A
symbolic ref is a POINTER. Only an object id is an IDENTITY.

THE FIXTURE IS DISCRIMINATING, NOT DECORATIVE. The pre-repair code FAILS
case 2 and 3 below (demonstrated on a synthetic repository before the
repair, with movement forced at a controlled boundary). Case 1 fails if
anyone removes the guard. Movement is never produced by a sleep or a
race: it is forced at an exact, deterministic point.

A NOTE ON SELF-OBSERVATION (R9). Case 3 drives run_census.main(), and
main() calls qualify(), and qualify() runs THIS module. Left alone that
recurses forever -- an instrument whose own execution is part of what it
measures. The qualification step is stubbed for the duration of case 3
and restored afterwards; it is downstream of subject binding and plays
no part in what this fixture tests.
"""
from __future__ import annotations
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import caltrace as ct
import qualify as Q
import run_census as RC

_REAL_RUN = subprocess.run


def _git(root, *args):
    return _REAL_RUN(["git", *args], cwd=root, capture_output=True, text=True)


def build(root: pathlib.Path):
    """A repo with two commits whose CONTENT is trivially distinguishable:
    commit A has one document, commit B has three. Branch `main` is left
    pointing at A."""
    root.mkdir(parents=True, exist_ok=True)
    _git(root, "init", "-q", "-b", "main")
    (root / "docs").mkdir()
    (root / "docs" / "a.md").write_text("# a\n")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=c@x", "-c", "user.name=c",
         "commit", "-q", "-m", "A")
    A = _git(root, "rev-parse", "HEAD").stdout.strip()
    (root / "docs" / "b.md").write_text("# b\n")
    (root / "docs" / "c.md").write_text("# c\n")
    _git(root, "add", "-A")
    _git(root, "-c", "user.email=c@x", "-c", "user.name=c",
         "commit", "-q", "-m", "B")
    B = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "update-ref", "refs/heads/main", A)
    return A, B


def case1_guard_rejects_a_pointer(repo):
    """materialise() must REFUSE a symbolic ref outright."""
    with tempfile.TemporaryDirectory() as d:
        try:
            RC.materialise(repo, "main", pathlib.Path(d) / "s")
            ct.check("case1 materialise() REFUSES a symbolic ref",
                     False, "it accepted 'main' instead of aborting")
        except SystemExit as e:
            ct.check("case1 materialise() REFUSES a symbolic ref",
                     "immutable" in str(e).lower(), str(e))


def case2_movement_after_resolution_has_no_effect(repo, A, B):
    """Resolve -> move the branch -> materialise the RESOLVED id."""
    commit = _git(repo, "rev-parse", "main^{commit}").stdout.strip()
    ct.check("case2 resolution yields A", commit == A, f"{commit} vs {A}")
    _git(repo, "update-ref", "refs/heads/main", B)      # the world moves
    try:
        with tempfile.TemporaryDirectory() as d:
            sub = pathlib.Path(d) / "s"
            expect = RC.materialise(repo, commit, sub)
            docs = [p for p in expect if p.endswith(".md")]
            ct.check("case2 materialisation follows the RESOLVED id, "
                     "not the moved branch", len(docs) == 1, str(expect))
    finally:
        _git(repo, "update-ref", "refs/heads/main", A)


def case3_end_to_end_stamp_matches_content(repo, A, B):
    """Drive the real main() and move the branch at the exact moment
    materialisation first touches git. Stamp and content must both be A."""
    state = {"moved": False}

    def wrapper(cmd, *a, **k):
        c = [str(x) for x in cmd]
        if (not state["moved"] and len(c) > 1 and c[0] == "git"
                and c[1] == "ls-tree" and str(k.get("cwd")) == str(repo)):
            _REAL_RUN(["git", "update-ref", "refs/heads/main", B],
                      cwd=repo, capture_output=True)
            state["moved"] = True
        return _REAL_RUN(cmd, *a, **k)

    real_qualify, real_report = Q.qualify, Q.report
    out = pathlib.Path(tempfile.mkdtemp()) / "r.json"
    argv = sys.argv
    try:
        # R9: break the self-observation loop for the duration.
        Q.qualify = lambda *a, **k: ([], [], (0, 0, []))
        Q.report = lambda *a, **k: ""
        subprocess.run = wrapper
        RC.subprocess.run = wrapper
        sys.argv = ["run_census.py", "--repo", str(repo), "--ref", "main",
                    "--out", str(out)]
        RC.main()
    finally:
        sys.argv = argv
        subprocess.run = _REAL_RUN
        RC.subprocess.run = _REAL_RUN
        Q.qualify, Q.report = real_qualify, real_report
        _REAL_RUN(["git", "update-ref", "refs/heads/main", A], cwd=repo,
                  capture_output=True)

    ct.check("case3 the branch actually moved mid-run (fixture is live)",
             state["moved"], "the boundary was never reached")
    r = json.loads(out.read_text())
    treeA = _git(repo, "rev-parse", f"{A}^{{tree}}").stdout.strip()
    ct.check("case3 STAMPED commit is A", r["subject"]["commit"] == A,
             f"{r['subject']['commit']} vs {A}")
    ct.check("case3 STAMPED tree is tree(A)", r["subject"]["tree"] == treeA,
             f"{r['subject']['tree']} vs {treeA}")
    ct.check("case3 MEASURED content is A (1 document, not B's 3)",
             r["documents"] == 1, f"documents={r['documents']}")
    ct.check("case3 B had ZERO influence on the result",
             r["subject"]["commit"] == A and r["documents"] == 1
             and r["subject"]["tree"] == treeA, json.dumps(r["subject"]))
    ct.check("case3 the invocation ref is recorded as mutable",
             r["subject"]["invocation_ref"] == "main"
             and r["subject"]["immutable_ref_as_invoked"] is False,
             json.dumps(r["subject"]))


def run():
    with tempfile.TemporaryDirectory() as td:
        repo = pathlib.Path(td) / "repo"
        A, B = build(repo)
        case1_guard_rejects_a_pointer(repo)
        case2_movement_after_resolution_has_no_effect(repo, A, B)
        case3_end_to_end_stamp_matches_content(repo, A, B)


if __name__ == "__main__":
    ct.reset()
    run()
    print(f"cal_subject_binding: {ct.PASSED} passed, {ct.FAILED} failed")
    for f in ct.FAILURES:
        print("  FAIL", f)
    sys.exit(1 if ct.FAILED else 0)
