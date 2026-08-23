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
the ref is MATERIALISED with `git archive`, so the analysed bytes are
exactly the bytes of that commit -- and the materialisation is verified
against `git ls-tree` of the original repository before anything is
measured (R11: no subject, no observation).

    python3 run_census.py --repo /path/to/repo --ref <sha> --out out.json
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import subprocess
import sys
import tempfile

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


def materialise(repo: pathlib.Path, ref: str, dest: pathlib.Path):
    """Extract the exact tree at `ref` into `dest` and prove it matches.

    Returns the expected tracked-file list from the ORIGINAL repository,
    which is independent evidence of what the subject contains (I-8).
    """
    expect = sorted(p for p in _run(
        ["git", "ls-tree", "-r", "--name-only", ref],
        cwd=repo).stdout.splitlines() if p)
    if not expect:
        raise SystemExit(f"R11 ABORT: ref {ref!r} lists no files in {repo}")

    # binary stdout: the archive is bytes, so it cannot go through the
    # text-mode helper above.
    tar = subprocess.run(["git", "archive", "--format=tar", ref],
                         cwd=repo, stdout=subprocess.PIPE)
    dest.mkdir(parents=True, exist_ok=True)
    x = subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout)
    if x.returncode != 0:
        raise SystemExit(f"R11 ABORT: could not materialise {ref}")

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
            f"R11 ABORT: materialised subject does not match {ref}. "
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
    head = _run(["git", "rev-parse", a.ref], cwd=repo).stdout.strip()
    tree = _run(["git", "rev-parse", f"{a.ref}^{{tree}}"], cwd=repo).stdout.strip()
    if not head:
        raise SystemExit(f"R11 ABORT: cannot resolve ref {a.ref!r} in {repo}")

    with tempfile.TemporaryDirectory() as td:
        subject = pathlib.Path(td) / "subject"
        expect = materialise(repo, a.ref, subject)
        res = census(subject)

    res["instrument"] = INSTRUMENT
    res["subject"] = {"repo": str(repo), "ref": a.ref, "commit": head,
                      "tree": tree, "tracked_files": len(expect),
                      "label": a.label or a.ref}

    rows, findings, cal = Q.qualify(res["subject_counts"],
                                    subject_label=head)
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
    record, blob, sha = AP.build(INSTRUMENT, head, tree, rows,
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

    print(Q.report(rows, findings, cal, subject_label=head))
    print()
    print(f"SUBJECT {head}  tree {tree}")
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
