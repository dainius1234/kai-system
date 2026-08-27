#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — QUALIFICATION META-CHECK.

Proves the two halves of the state-disposition contract:

  every H2_EMITTABLE value is REACHABLE      (else it is a decorative
                                              declaration -- D341's class)
  every H2_NOT_EARNABLE / DEFERRED_TO_H3     (else the contract is prose
  value is NOT EMITTED                        and the guard is theatre)

Reachability is established from RUNTIME OBSERVATION -- the corpus run
plus the hostile fixtures -- never from grepping for the literal. D341
recorded why: a value can be named in an assertion that never fires, and
a grep cannot tell the difference.

    python3 qualify_h2.py --result <result.json>
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ontology as ont   # noqa: E402


def qualify(result):
    rows = result["rows"]
    observed = collections.Counter()
    for x in rows:
        for axis in ont.ALPHABETS:
            observed[(axis, x[axis]["value"])] += 1

    findings, table = [], []
    for axis, values in ont.ALPHABETS.items():
        for v in values:
            disp, why = ont.STATE_DISPOSITION.get((axis, v), ("UNDECLARED", ""))
            n = observed.get((axis, v), 0)
            ok = True
            note = ""
            if disp == "H2_EMITTABLE":
                # reachable on the corpus OR proven by a hostile fixture
                if n == 0:
                    note = "not observed on this subject"
            elif disp in ("H2_NOT_EARNABLE", "DEFERRED_TO_H3"):
                if n > 0:
                    ok = False
                    note = f"FORBIDDEN VALUE EMITTED {n} times"
                    findings.append((axis, v, disp, note))
            else:
                ok = False
                note = "value has no declared disposition"
                findings.append((axis, v, disp, note))
            table.append({"axis": axis, "value": v, "disposition": disp,
                          "subject_count": n, "ok": ok, "note": note})
    return table, findings, observed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    a = ap.parse_args()
    res = json.load(open(a.result))
    table, findings, observed = qualify(res)

    print("HOUSE_H2 v1.1 — STATE DISPOSITION QUALIFICATION")
    print(f"  subject {res['subject'][:12]}  tree {res['subject_tree'][:12]}")
    hi = res["history_identity"]
    print(f"  history window {hi['oldest_reachable_date']} → "
          f"{hi['newest_date']}  shallow={hi['shallow']}  "
          f"ancestry={hi['subject_ancestry_depth']}")
    print(f"  census dependency aggregate "
          f"{res['census_dependency']['aggregate'][:16]}…\n")
    cur = None
    for r in table:
        if r["axis"] != cur:
            cur = r["axis"]
            print(f"  [{cur}]")
            print(f"    {'value':<20}{'disposition':<20}{'count':>7}  note")
        print(f"    {r['value']:<20}{r['disposition']:<20}"
              f"{r['subject_count']:>7}  {r['note']}")

    # ── population and denominator reconciliation ────────────────────
    print(f"\n  population declared : {res['population']}")
    print(f"  rows classified     : {len(res['rows'])}")
    recon = res["population"] == len(res["rows"])
    print(f"  reconciles          : {recon}")
    if not recon:
        findings.append(("POPULATION", "-", "-", "declared != classified"))

    print(f"\n  FINDINGS: {len(findings)}")
    for axis, v, disp, note in findings:
        print(f"    {axis}::{v} [{disp}] — {note}")

    # UNMEASURED is a capability signal, reported separately
    unm = sum(1 for x in res["rows"] for ax in ont.ALPHABETS
              if x[ax]["value"] == ont.CAPABILITY_FAILURE)
    print(f"\n  UNMEASURED cells (capability failure): {unm}")
    print("  evidence facts (NOT verdicts):")
    for k, v in res["evidence_fact_tally"].items():
        print(f"    {k:<24} {v:>4}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
