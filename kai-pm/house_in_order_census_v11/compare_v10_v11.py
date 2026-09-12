#!/usr/bin/env python3
"""v1.0 <-> v1.1 RECONCILIATION ON ONE IDENTICAL SUBJECT.

Kai's D341 sequence, step 4: "run v1.1 against the same original H1
subject ... reconcile v1.0 <-> v1.1 row/count/claim deltas on that
identical subject."

Both instruments are pointed at the SAME materialised tree in the same
process, so any delta is attributable to the instrument and to nothing
else -- not to a dirty checkout, not to new commits, not to a different
machine. v1.0 is imported from its frozen package and is NOT modified.

Every delta must be EXPLAINED, not merely reported. An unexplained delta
is a finding.
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import claims as C          # noqa: E402
import docgraph as G        # noqa: E402
import opscan as O          # noqa: E402
import repairs as RP        # noqa: E402
import run_census as RC     # noqa: E402


def load_v10(frozen: pathlib.Path):
    sys.path.insert(0, str(frozen))
    import doccensus2 as dc
    import genlink3 as g3
    return dc, g3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--frozen-v10", required=True,
                    help="path to the frozen Census v1.0 package")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    repo = pathlib.Path(a.repo).resolve()
    frozen = pathlib.Path(a.frozen_v10).resolve()
    dc, g3 = load_v10(frozen)

    # RESOLVE ONCE to an immutable commit, then use only that id. This
    # evidence object is subject-dependent, so it carries commit AND
    # tree explicitly rather than leaving the tree to be derived later.
    commit = RC._run(["git", "rev-parse", f"{a.ref}^{{commit}}"],
                     cwd=repo).stdout.strip()
    if not RC.IMMUTABLE_OID.fullmatch(commit or ""):
        raise SystemExit(f"R11 ABORT: cannot resolve {a.ref!r} to an "
                         f"immutable commit in {repo}")
    tree = RC._run(["git", "rev-parse", f"{commit}^{{tree}}"],
                   cwd=repo).stdout.strip()

    with tempfile.TemporaryDirectory() as td:
        subject = pathlib.Path(td) / "subject"
        RC.materialise(repo, commit, subject)

        # ── v1.0, frozen, unmodified ─────────────────────────────────
        d10 = dc.tracked_md(subject)
        e10 = dc.build_graph(subject, d10)
        o10 = g3.collect(subject, d10)
        ta10 = set(g3.tracked(subject))
        n10, t10, _s10 = g3.account(o10)
        c10 = {}
        for d in d10:
            v, _src, b, _w = g3.writer_claim_scoped(o10, d, ta10)
            c10[d] = (v, b)

        # ── v1.1 ─────────────────────────────────────────────────────
        d11 = G.tracked_md(subject)
        e11 = G.build_graph(subject, d11)
        o11, acc11 = O.collect(subject, d11)
        ta11 = set(O.tracked(subject))
        C.classify(o11, d11, ta11)
        n11, t11, _s11 = C.account(o11)
        c11 = {}
        for d in d11:
            v, _src, b, _w, _sc = C.scoped_claim(o11, d, ta11)
            c11[d] = (v, b)
        # Kai's D342 ruling: preserve the exact reclassified instances in
        # the RECONCILIATION evidence, not only in the census.
        repair_ev = RP.measure(o11, d11, ta11)

    out = []
    def p(s=""):
        out.append(s)
        print(s)

    p(f"v1.0 <-> v1.1 RECONCILIATION — subject {commit}")
    p(f"  tree {tree}   (invoked as {a.ref!r})")
    p(f"  population       docs {len(d10)} -> {len(d11)}   "
      f"{'IDENTICAL' if d10 == d11 else 'DIFFERS'}")
    p(f"  edges            {len(e10)} -> {len(e11)}   "
      f"delta {len(e11) - len(e10)}")
    p(f"  operations       {n10} -> {n11}   delta {n11 - n10}")
    p(f"    v1.1 raw {acc11['raw_candidate_matches']} = rejected "
      f"{acc11['rejected_total']} + admitted "
      f"{acc11['admitted_candidate_operations']}")
    p("    v1.0 emitted NO admission denominator: every raw match became")
    p("    an operation, which is the defect this reconciles.")

    p("\n  DISPOSITION DELTAS")
    keys = sorted(set(t10) | set(t11))
    for k in keys:
        p(f"    {k:<32} {t10.get(k, 0):>6} -> {t11.get(k, 0):>6}   "
          f"{t11.get(k, 0) - t10.get(k, 0):+d}")

    p("\n  CLAIM DELTAS")
    tally10 = collections.Counter(v for v, _b in c10.values())
    tally11 = collections.Counter(v for v, _b in c11.values())
    for k in sorted(set(tally10) | set(tally11)):
        p(f"    {k:<36} {tally10.get(k, 0):>4} -> {tally11.get(k, 0):>4}")
    p("    NOTE: v1.0 'NO_WRITER' and v1.1 "
      "'NO_WRITER_WITHIN_ANALYZED_SCOPE' are the")
    p("    SAME STATE RENAMED (Kai D341). Closure rules are unchanged.")

    RENAME = {"NO_WRITER": "NO_WRITER_WITHIN_ANALYZED_SCOPE"}
    changed = [(d, c10[d][0], c11[d][0]) for d in d10
               if d in c11 and RENAME.get(c10[d][0], c10[d][0]) != c11[d][0]]
    p(f"\n  DOCUMENTS WHOSE CLAIM CHANGED: {len(changed)}")
    for d, a0, a1 in changed[:20]:
        p(f"    {d}  {a0} -> {a1}")

    # Bucket movement explains the operation-count change per document.
    moved = [(d, c10[d][1].get("COULD_REACH_T", 0),
              c11[d][1].get("COULD_REACH_T", 0)) for d in d10 if d in c11]
    widen = [m for m in moved if m[2] > m[1]]
    narrow = [m for m in moved if m[2] < m[1]]
    p(f"\n  OPEN-BUCKET MOVEMENT (COULD_REACH_T per document)")
    p(f"    narrowed (more constructively excluded) : {len(narrow)}")
    p(f"    widened  (fewer excluded)               : {len(widen)}")
    if widen:
        w = min(widen, key=lambda m: m[1] - m[2])
        p(f"    largest widening: {w[0]}  {w[1]} -> {w[2]}")
        p("    WIDENING IS EXPECTED AND CORRECT: v1.0 excluded absolute and")
        p("    URI-shaped targets on SYNTAX. Those exclusions were unsound,")
        p("    so removing them REOPENS operations that were wrongly closed.")
    if narrow:
        nmin = min(narrow, key=lambda m: m[2] - m[1])
        p(f"    largest narrowing: {nmin[0]}  {nmin[1]} -> {nmin[2]}")

    res = {"subject": commit, "subject_commit": commit,
           "subject_tree": tree, "invocation_ref": a.ref,
           "immutable_ref_as_invoked":
               bool(RC.IMMUTABLE_OID.fullmatch(a.ref or "")),
           "docs_v10": len(d10), "docs_v11": len(d11),
           "edges_v10": len(e10), "edges_v11": len(e11),
           "ops_v10": n10, "ops_v11": n11, "admission": acc11,
           "dispositions_v10": t10, "dispositions_v11": t11,
           "claims_v10": dict(tally10), "claims_v11": dict(tally11),
           "claim_changes": [{"doc": d, "v10": x, "v11": y}
                             for d, x, y in changed],
           "narrowed": len(narrow), "widened": len(widen),
           "repair_evidence": repair_ev}

    recl = next(r for r in repair_ev["repairs"]
                if r["rule_id"] == "READ_TARGET_RECLASSIFIED_CONSERVATIVELY")
    p(f"\n  READ_TARGET_RECLASSIFIED_CONSERVATIVELY: "
      f"{len(recl['reclassified_operations'])} operations")
    p("    Relevance PROVEN (.md), exact tracked target UNPROVEN. The")
    p("    operations are preserved; the withdrawn part is the unproven")
    p("    claim about WHICH document each one touches.")
    for r in recl["reclassified_operations"]:
        p(f"    {r['src']}:{r['line']} mode={r['mode']} "
          f"{r['fragments']} — v1.0 bound it to "
          f"{r['v10_would_have_bound_to']}, v1.1 says {r['v11_disposition']}")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1, default=str))
        p(f"\nwritten: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
