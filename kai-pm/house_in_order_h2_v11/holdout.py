#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — BLIND HOLDOUT SELECTION.

The rule was FIXED IN PRECOMMIT.md §4 before any v1.1 code existed
(contract sha256
fa1069103a721cf5911641cbe6447360069eb9f2a3873a4296531ae280f4258e):

    key   = sha256("H2V11:" + path)
    order = ascending by key
    take  = first 24 of the 272

Deterministic, unseeded by any result, and different from D340's set by
construction — the salt differs, so the two selections cannot coincide
by design or by accident.

D339 recorded why the rule matters more than the sample looking fair:
non-curation is established by a selection rule committed BEFORE
selection, never by the sample happening to look proportionate.

I DO NOT SELF-ADJUDICATE THIS HOLDOUT. The rows are emitted for
independent blind adjudication. Any agreement figure computed by me is
descriptive and carries no acceptance weight.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib

SALT = "H2V11:"
TAKE = 24


def select(paths):
    return [p for _k, p in sorted(
        (hashlib.sha256((SALT + p).encode()).hexdigest(), p) for p in paths
    )][:TAKE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.load(open(a.result))
    rows = {r["path"]: r for r in res["rows"]}
    chosen = select(rows.keys())

    axes = ("LIFECYCLE", "FUNCTION", "AUTHORITY", "GENERATION", "VALIDITY",
            "SCOPE")
    out = {
        "selection_rule": f'sha256("{SALT}" + path) ascending, first {TAKE}',
        "precommit_sha256":
            "fa1069103a721cf5911641cbe6447360069eb9f2a3873a4296531ae280f4258e",
        "population": len(rows),
        "selected": len(chosen),
        "adjudication": "INDEPENDENT AND BLIND. Not self-adjudicated.",
        "rows": [{
            "path": p,
            **{ax: rows[p][ax]["value"] for ax in axes},
            "EVIDENCE_FACTS": {f: rows[p]["EVIDENCE_FACTS"][f]["present"]
                               for f in ("MAINTENANCE_OBSERVED",
                                         "SELF_ASSERTS_CURRENT",
                                         "CONSUMED_AT_SUBJECT")},
        } for p in chosen],
    }
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"BLIND HOLDOUT — {len(chosen)} of {len(rows)} by precommitted rule")
    print(f"  rule: {out['selection_rule']}")
    hdr = f"  {'path':<52}{'LIFECYCLE':<12}{'FUNCTION':<14}{'VALIDITY':<15}"
    print(hdr)
    for r in out["rows"]:
        print(f"  {r['path'][:50]:<52}{r['LIFECYCLE']:<12}"
              f"{r['FUNCTION']:<14}{r['VALIDITY']:<15}")
    print(f"\n  written: {a.out}")
    print("  FOR INDEPENDENT BLIND ADJUDICATION — no self-scoring.")


if __name__ == "__main__":
    main()
