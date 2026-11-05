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
import hashlib
import json
import pathlib

D366_COMMIT = "86a1399e6e31477ba67cd38c12d22627a8b4d6ef"
SALT = "H2FINAL-D367:"
SIZE = 40


def select(paths, candidate_aggregate):
    return sorted(paths, key=lambda p: hashlib.sha256(
        f"{SALT}{D366_COMMIT}:{candidate_aggregate}:{p}".encode()
    ).hexdigest())[:SIZE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--manifest", required=True,
                    help="the candidate MANIFEST.sha256; its sha256 IS the "
                         "aggregate that determines the sample")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = json.load(open(a.result))
    aggregate = hashlib.sha256(
        pathlib.Path(a.manifest).read_bytes()).hexdigest()
    by_path = {r["path"]: r for r in res["rows"]}
    chosen = select(sorted(by_path), aggregate)

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
