#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — FINAL BLIND HOLDOUT. Selection rule frozen in D367 9.

    key = sha256("H2FINAL-D367:"
                 + "86a1399e6e31477ba67cd38c12d22627a8b4d6ef"   # D366
                 + ":" + FINAL_CANDIDATE_AGGREGATE
                 + ":" + path)
    sort ascending, select the first 40

The candidate aggregate did not exist when the rule was frozen, so the
sample could not be known during implementation -- while the rule itself
was committed before a line of repair code was written. If a candidate
fails and code changes, the new identity deterministically yields a NEW
sample, and previously revealed rows become regression evidence only.

NOT SELF-ADJUDICATED. Kai adjudicates all 40 across all six axes plus
consequential evidence facts. THIS SCRIPT COMPUTES NO AGREEMENT FIGURE,
and any figure Orion computed would carry no admission weight (D367 10).
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

D366_COMMIT = "86a1399e6e31477ba67cd38c12d22627a8b4d6ef"
SALT = "H2FINAL-D367:"
SIZE = 40


class HoldoutInputError(AssertionError):
    """I1-A / I1-B refused. Raised, never returned."""


def select(paths, candidate_aggregate):
    return sorted(paths, key=lambda p: hashlib.sha256(
        f"{SALT}{D366_COMMIT}:{candidate_aggregate}:{p}".encode()
    ).hexdigest())[:SIZE]


# ── I1-A — the seed is the VALIDATED STAGE-A IDENTITY ─────────────────
#
# v1.2 computed  aggregate = sha256(read_bytes(MANIFEST.sha256))  and used
# it as CANDIDATE_AGGREGATE. That manifest CONTAINS EXECUTION OUTPUT -- ten
# entries, nine source modules plus h2v12-classification.json -- so the
# blind sample was seeded by the candidate's own result. Mutate the
# evidence, and the sample that audits the evidence moves.
#
# D381 §18 supersedes the definition for this lineage:
#     FINAL_CANDIDATE_AGGREGATE = canonical H2_STAGE_A_V2 stage_a_identity
# D367 §9's selection equation is UNCHANGED. Only what is substituted into
# it changes.
#
# THE AUTHORITY BOUNDARY (D381 §10). stage_identity DETERMINES whether a
# descriptor is valid for production; holdout CONSUMES ONLY the validated
# identity. The V1+PRODUCTION refusal, the V2 governance contract and the
# calibration zero-weight rule are NOT duplicated here, and must not be.
def final_candidate_aggregate(descriptor_path):
    """The validated Stage-A identity, or REFUSE. Never a manifest digest."""
    import stage_identity as SI
    p = pathlib.Path(descriptor_path)
    if not p.is_file():
        raise HoldoutInputError(
            f"REFUSE: no Stage-A descriptor at {descriptor_path}. The blind "
            f"selection seed is the validated Stage-A identity; there is no "
            f"fallback to a manifest, a package digest or result bytes.")
    desc = json.loads(p.read_bytes().decode("utf-8"))
    if desc.get("mode") != "PRODUCTION":
        raise HoldoutInputError(
            f"REFUSE: Stage-A descriptor mode is {desc.get('mode')!r}. A "
            f"CALIBRATION identity carries ZERO holdout weight (D381 §13) "
            f"and may never seed the blind 40.")
    return SI.stage_a_identity(desc)          # validates, then derives


# ── I1-B — the universe is the FROZEN SUBJECT TREE ────────────────────
def reconcile(tree_paths, output_paths):
    """The candidate output decides WHETHER reconciliation passes. It must
    never decide WHAT POPULATION IS SELECTED. That distinction is the whole
    I1 repair, so this compares MULTISETS and refuses on any divergence."""
    t, o = collections.Counter(tree_paths), collections.Counter(output_paths)
    if t == o:
        return True
    missing = sorted((t - o).elements())
    extra = sorted((o - t).elements())
    dup = sorted(p for p, n in o.items() if n > 1)
    raise HoldoutInputError(
        f"REFUSE BEFORE SELECTION: candidate output does not reconcile with "
        f"the frozen subject tree. tree-only={missing[:5]} "
        f"output-only={extra[:5]} duplicated={dup[:5]} "
        f"(tree {sum(t.values())} vs output {sum(o.values())})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--stage-a", required=True, dest="stage_a",
                    help="the PRODUCTION Stage-A descriptor. Its VALIDATED "
                         "identity is FINAL_CANDIDATE_AGGREGATE (D381 18). "
                         "MANIFEST.sha256 is no longer accepted: it contains "
                         "execution output, so it seeded the blind sample "
                         "from the candidate's own result (I1-A).")
    ap.add_argument("--tree-paths", required=True, dest="tree_paths",
                    help="the immutable frozen subject tree path population "
                         "(I1-B). Selection is from THIS, never from output.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = json.load(open(a.result))
    aggregate = final_candidate_aggregate(a.stage_a)          # I1-A
    tree_paths = [l.strip() for l in
                  pathlib.Path(a.tree_paths).read_text().splitlines()
                  if l.strip()]
    output_paths = [r["path"] for r in res["rows"]]
    reconcile(tree_paths, output_paths)                       # I1-B, may REFUSE
    by_path = {r["path"]: r for r in res["rows"]}
    chosen = select(sorted(tree_paths), aggregate)            # from the TREE

    payload = {
        "holdout": "H2FINAL-D367", "size": SIZE,
        "selection_rule": f'sha256("{SALT}" + "{D366_COMMIT}" + ":" + '
                          f'CANDIDATE_AGGREGATE + ":" + path), ascending, '
                          f'first {SIZE}',
        "candidate_aggregate": aggregate,
        "d366_commit": D366_COMMIT,
        "subject": res["subject"], "subject_tree": res["subject_tree"],
        "population": res["population"],
        "adjudication": "KAI. Orion computes no agreement figure and this "
                        "package carries none. Orion self-adjudication has "
                        "ZERO final admission weight (D367 10).",
        "evaluation_rule": {
            "incorrect non-abstention verdict": "BLOCKER",
            "false evidence fact": "BLOCKER",
            "unsupported scope widening": "BLOCKER",
            "forbidden or undeclared state emitted": "BLOCKER",
            "determining witness absent or silently truncated": "BLOCKER",
            "genuinely ambiguous source evidence": "UNRESOLVED",
            "UNKNOWN": "ABSTENTION — never negative evidence",
            "earnable positive emitted as UNKNOWN":
                "OVER_ABSTENTION / coverage finding, not automatically a "
                "safety blocker",
        },
        "rows": [by_path[p] for p in chosen],
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"BLIND HOLDOUT — {len(chosen)} of {res['population']} documents")
    print(f"  candidate aggregate {aggregate}")
    print(f"  rule frozen in D367 before any repair code existed")
    print(f"  written to {a.out}")
    print("  NOT SELF-ADJUDICATED. No agreement figure computed.")


if __name__ == "__main__":
    main()
