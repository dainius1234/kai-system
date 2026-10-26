#!/usr/bin/env python3
"""CENSUS v1.1 RUNNER — portable, subject-bound, denominator-reconciling.

PORTABILITY (D340 defect, carried into the v1.1 scope). Nothing here is
hard-coded: the repository, the subject ref and the output path are all
parameters, and the package resolves its own directory from __file__.
H2's pass_a.py advertised generic parameters in RUN.md while hard-coding
an absolute repository path and a session /tmp path; that is what made
the package unreproducible on any other machine.

SUBJECT BINDING. v1.0 listed files from git but read their CONTENT from
the working tree, so a dirty checkout silently mixed two subjects. Here
the subject is MATERIALISED with `git archive` and verified against
`git ls-tree` before anything is measured (R11: no subject, no
observation).

THE RESOLVE-ONCE INVARIANT. The supplied ref is resolved to an
IMMUTABLE COMMIT ID EXACTLY ONCE, and every later step -- tree
derivation, ls-tree, archive, reconciliation and stamping -- uses that
commit id. Nothing dereferences the symbolic ref again.

This is not a precaution, it is a repair. The previous version passed
the symbolic ref down into materialise(), which re-dereferenced it for
ls-tree and archive AFTER main() had recorded the identity. A branch
moving in between made both sides of the reconciliation see the NEW
commit, so they agreed, "reconciles: True" was reported, and the result
was stamped with the OLD commit. Demonstrated by execution on a
synthetic repository with the movement forced at a controlled boundary.
`materialise()` now REFUSES anything that is not a 40-hex object id, so
the defect cannot be reintroduced by a future caller.

    python3 run_census.py --repo /path/to/repo --ref <sha> --out out.json
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys
import tempfile

# A subject identity must be an immutable object id. A symbolic ref is
# a POINTER, and a pointer can move between the moment it is read and
# the moment it is used.
IMMUTABLE_OID = re.compile(r"[0-9a-f]{40}")

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import applicability as AP   # noqa: E402
import claims as C          # noqa: E402
import docgraph as G        # noqa: E402
import opscan as O          # noqa: E402
import qualify as Q         # noqa: E402
import repairs as RP        # noqa: E402

INSTRUMENT = "HOUSE-IN-ORDER-CENSUS-INSTRUMENT v1.1"


def _run(cmd, cwd=None, **kw):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, **kw)


def materialise(repo: pathlib.Path, commit: str, dest: pathlib.Path):
    """Extract the exact tree at `commit` into `dest` and prove it matches.

    `commit` MUST be an immutable 40-hex object id, and the guard below
    enforces it rather than trusting the caller.

    THE DEFECT THIS PREVENTS. Earlier this function took the SYMBOLIC REF
    and re-dereferenced it for both `ls-tree` and `git archive`, after
    main() had already resolved and recorded the identity. If the ref
    moved in between, ls-tree and archive both saw the NEW commit, so
    `expect` and `got` agreed perfectly and the run reported
    "reconciles: True" while the result was stamped with the OLD commit.
    A silent subject MISBINDING that presents as a clean, fully
    populated table -- the exact shape this programme exists to remove.
    Proven by execution against a synthetic repository, not argued.

    A symbolic ref is a pointer. Only an object id is an identity.

    Returns the expected tracked-file list from the ORIGINAL repository,
    which is independent evidence of what the subject contains (I-8).
    """
    if not IMMUTABLE_OID.fullmatch(commit or ""):
        raise SystemExit(
            f"R11 ABORT: materialise() requires an immutable 40-hex commit "
            f"id; got {commit!r}. A symbolic ref may move between "
            f"resolution and materialisation.")
    expect = sorted(p for p in _run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=repo).stdout.splitlines() if p)
    if not expect:
        raise SystemExit(f"R11 ABORT: commit {commit!r} lists no files in {repo}")

    # binary stdout: the archive is bytes, so it cannot go through the
    # text-mode helper above.
    tar = subprocess.run(["git", "archive", "--format=tar", commit],
                         cwd=repo, stdout=subprocess.PIPE)
    dest.mkdir(parents=True, exist_ok=True)
    x = subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout)
    if x.returncode != 0:
        raise SystemExit(f"R11 ABORT: could not materialise {commit}")

    _run(["git", "init", "-q"], cwd=dest)
    # --force: a file may be BOTH tracked at the ref and gitignored, and
    # dropping it would shrink the denominator invisibly.
    _run(["git", "add", "-A", "--force"], cwd=dest)
    _run(["git", "-c", "user.email=c@x", "-c", "user.name=c",
          "commit", "-q", "-m", "subject"], cwd=dest)

    got = sorted(p for p in _run(["git", "ls-tree", "-r", "--name-only",
                                  "HEAD"], cwd=dest).stdout.splitlines() if p)
    if got != expect:
        missing = sorted(set(expect) - set(got))[:10]
        extra = sorted(set(got) - set(expect))[:10]
        raise SystemExit(
            f"R11 ABORT: materialised subject does not match {commit}. "
            f"expected {len(expect)} files, got {len(got)}. "
            f"missing={missing} extra={extra}")
    return expect


def census(subject: pathlib.Path):
    """The measurement. Every denominator is emitted and reconciled."""
    docs = G.tracked_md(subject)
    if not docs:
        raise SystemExit("R11 ABORT: subject contains no tracked documents")
    tracked_all = set(O.tracked(subject))

    edges = G.build_graph(subject, docs)
    ops, acc = O.collect(subject, docs)
    C.classify(ops, docs, tracked_all)

    n, tally, s = C.account(ops)
    if not (n == s == acc["admitted_candidate_operations"]):
        raise SystemExit(f"P12 ABORT: {n} ops, {s} dispositions, {acc}")

    counts = collections.Counter()
    for o in ops:
        counts[f"claims.DISPOSITIONS::{o.disposition}"] += 1
        counts[f"opscan.OP_MODES::{o.mode}"] += 1
        counts[f"claims.PATH_SYNTAX::"
               f"{C.path_syntax(o.frags, o.dynamic)}"] += 1
    for r, k in acc["rejected_non_operations"].items():
        counts[f"opscan.REJECTION_REASONS::{r}"] += k
    for _s, _d, kind, _r, ctx in edges:
        counts[f"docgraph.KINDS::{kind}"] += 1
        counts[f"docgraph.EDGE_CONTEXTS::{ctx}"] += 1

    claims_out = {}
    claim_tally = collections.Counter()
    for d in docs:
        claim, srcs, buckets, wit, _scope = C.scoped_claim(ops, d, tracked_all)
        claims_out[d] = {"claim": claim, "sources": srcs, "buckets": buckets,
                         "witnesses": wit}
        claim_tally[claim] += 1
        counts[f"claims.CLAIMS::{claim}"] += 1
        for k, v in buckets.items():
            counts[f"claims.TARGET_DISPOSITIONS::{k}"] += v
        for k, v in wit.items():
            counts[f"claims.EXCLUSION_WITNESSES::{k}"] += v

    # Every declared value that never occurred is reported as an EXPLICIT
    # ZERO. An absent row reads like "not applicable"; a zero states it.
    for _mn, _p, mod in Q.instrument_modules():
        name = _p.stem
        for alpha, values in mod.ALPHABETS.items():
            for v in values:
                counts.setdefault(f"{name}.{alpha}::{v}", 0)

    return {
        "documents": len(docs),
        "edges": len(edges),
        "denominator_reconciliation": {
            **acc,
            "sum_dispositions": s,
            "reconciles": n == s == acc["admitted_candidate_operations"],
        },
        "disposition_tally": tally,
        "edge_kind_tally": dict(G.kind_tally(edges)),
        "edge_context_tally": dict(G.context_tally(edges)),
        "claim_tally": dict(claim_tally),
        "claims": claims_out,
        "subject_counts": dict(counts),
        "repair_evidence": RP.measure(ops, docs, tracked_all),
    }


def main():
    ap = argparse.ArgumentParser(description=INSTRUMENT)
    ap.add_argument("--repo", default=".", help="repository to analyse")
    ap.add_argument("--ref", default="HEAD", help="subject commit/ref")
    ap.add_argument("--out", default=None, help="write JSON result here")
    ap.add_argument("--label", default=None, help="human label for the run")
    a = ap.parse_args()

    repo = pathlib.Path(a.repo).resolve()
    # THE SUBJECT IS RESOLVED EXACTLY ONCE. From here on nothing
    # dereferences the symbolic ref again -- the tree is derived FROM
    # the resolved commit, and the commit is what is materialised,
    # reconciled and stamped.
    commit = _run(["git", "rev-parse", f"{a.ref}^{{commit}}"],
                  cwd=repo).stdout.strip()
    if not IMMUTABLE_OID.fullmatch(commit or ""):
        raise SystemExit(
            f"R11 ABORT: cannot resolve ref {a.ref!r} to an immutable "
            f"commit in {repo} (got {commit!r})")
    tree = _run(["git", "rev-parse", f"{commit}^{{tree}}"],
                cwd=repo).stdout.strip()
    if not IMMUTABLE_OID.fullmatch(tree or ""):
        raise SystemExit(f"R11 ABORT: cannot derive tree from {commit}")

    with tempfile.TemporaryDirectory() as td:
        subject = pathlib.Path(td) / "subject"
        expect = materialise(repo, commit, subject)
        res = census(subject)

    res["instrument"] = INSTRUMENT
    res["subject"] = {"repo": str(repo), "invocation_ref": a.ref,
                      "commit": commit, "tree": tree,
                      "immutable_ref_as_invoked":
                          bool(IMMUTABLE_OID.fullmatch(a.ref or "")),
                      "tracked_files": len(expect),
                      "label": a.label or a.ref}

    rows, findings, cal = Q.qualify(res["subject_counts"],
                                    subject_label=commit)
    res["qualification"] = {
        "rows": rows,
        "findings": [{"alphabet": x, "value": y, "finding": z}
                     for x, y, z in findings],
        "calibration": {"passed": cal[0], "failed": cal[1],
                        "failures": cal[2]},
    }

    # ── KAI'S D342 FREEZE CONDITION ──────────────────────────────────
    # The applicability restriction must not be separable from the
    # numbers it restricts. The record is embedded in the census AND
    # written as a standalone artefact carrying identical canonical
    # bytes, so the binding is checkable from either side.
    record, blob, sha = AP.build(INSTRUMENT, commit, tree, rows,
                                 res.pop("repair_evidence"))
    if not AP.verify(record, blob, sha):
        raise SystemExit("ABORT: applicability record failed self-binding")

    side = None
    if a.out:
        p = pathlib.Path(a.out)
        side = p.with_name(
            p.name.replace("census-", "applicability-", 1)
            if p.name.startswith("census-") else p.stem + ".applicability.json")
        side.write_bytes(blob)

    res["applicability"] = {
        "artefact": side.name if side else None,
        "sha256": sha,
        "binding_rule": record["binding_rule"],
        "record": record,
    }

    print(Q.report(rows, findings, cal, subject_label=commit))
    print()
    print(f"SUBJECT {commit}  tree {tree}")
    print(f"  documents {res['documents']}   edges {res['edges']}")
    dr = res["denominator_reconciliation"]
    print(f"  raw_candidate_matches         {dr['raw_candidate_matches']}")
    for k, v in sorted(dr["rejected_non_operations"].items()):
        print(f"      rejected {k:<26} {v}")
    print(f"  rejected_non_operations       {dr['rejected_total']}")
    print(f"  admitted_candidate_operations {dr['admitted_candidate_operations']}")
    print(f"  sum(dispositions)             {dr['sum_dispositions']}")
    print(f"  reconciles                    {dr['reconciles']}")
    print("  claims:")
    for k, v in sorted(res["claim_tally"].items()):
        print(f"      {k:<36} {v}")

    rec = res["applicability"]["record"]
    print(f"\n  APPLICABILITY RECORD  sha256 {sha[:16]}…  "
          f"artefact {rec['declared_states']} states, "
          f"{rec['usable_on_this_subject']} usable, "
          f"{rec['restricted_on_this_subject']} restricted")
    for alpha, val, _why in AP.restricted(rec):
        print(f"      RESTRICTED  {alpha}::{val}")
    print("      " + rec["binding_rule"])
    for r in rec["repair_evidence"]["repairs"]:
        print(f"      REPAIR {r['rule_id']}: {r['RULE_STATUS']}, "
              f"CURRENT_SUBJECT_EFFECT={r['CURRENT_SUBJECT_EFFECT']} "
              f"({r['operations_affected_on_this_subject']} operations)")

    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        print(f"\nwritten: {a.out}")
        if side:
            print(f"written: {side}  (sha256 {sha})")
    return 1 if (findings or cal[1]) else 0


if __name__ == "__main__":
    sys.exit(main())
