#!/usr/bin/env python3
"""HOUSE_H2 v1.1 RUNNER — portable, subject-bound, contract-emitting.

Every source is a parameter. Nothing is hard-coded.

    python3 run_h2_v11.py --subject-repo <checkout> --history-repo <full>
                          --subject <sha> --census-package <path>
                          --passa <passA.json> --out <result.json>

Emits the machine-readable H2->H3 ADMISSION CONTRACT alongside the
classification. The contract supplies evidence; it does not approve
itself. Admission is a governing decision, never a test result.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify2 as cl      # noqa: E402
import envcontract2 as ec   # noqa: E402
import evidence as ev       # noqa: E402
import ontology as ont      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="HOUSE_H2 v1.1")
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--history-repo", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--passa", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tree_src = pathlib.Path(a.subject_repo)
    hist_src = pathlib.Path(a.history_repo)
    packet = json.load(open(a.passa))
    rows = packet["rows"]

    if packet["subject"] != a.subject:
        raise SystemExit(f"R11 ABORT: Pass A subject {packet['subject'][:12]} "
                         f"!= requested {a.subject[:12]}")

    # ── declared-value completeness: a value the contract forgot ─────
    missing = ont.undeclared()
    if missing:
        raise SystemExit(f"CONTRACT ABORT: values with no disposition: {missing}")

    cap_rows, cap_ok = ec.probe(tree_src, hist_src, a.subject)
    blocked = {ax: ec.axis_blocked(ax, cap_rows) for ax in ont.ALPHABETS}
    print("ENVIRONMENT-SUBJECT CAPABILITY CONTRACT")
    for r in cap_rows:
        print(f"  {r['disposition']:<20} {r['capability']:<32} "
              f"{r['observed_state'][:52]}")
    print(f"  contract satisfied: {cap_ok}")

    fam_ok, fam_why = cl.family_rule_proven(rows)
    print(f"\nFAMILY RULE (CODE_AUDIT_BATCH_*): proven={fam_ok} — {fam_why}")

    out = []
    for r in rows:
        text = (tree_src / r["path"]).read_text(errors="ignore")
        verdicts = cl.classify(r, text, blocked, fam_ok, fam_why)
        facts = ev.facts(r, text)
        out.append({**r, **verdicts, "EVIDENCE_FACTS": facts})
    if len(out) != len(rows):
        raise SystemExit("POPULATION ABORT: classified != Pass A population")

    # ── tallies ──────────────────────────────────────────────────────
    print(f"\nCLASSIFIED {len(out)} == Pass A population {len(rows)} "
          f"== declared population {packet['population']}")
    axis_tally = {}
    for axis in ont.ALPHABETS:
        c = collections.Counter(x[axis]["value"] for x in out)
        axis_tally[axis] = dict(c)
        print(f"\n  {axis}")
        for k, v in c.most_common():
            print(f"    {k:<18} {v:>4}")

    fact_tally = {f: sum(1 for x in out if x["EVIDENCE_FACTS"][f]["present"])
                  for f in ont.EVIDENCE_FACTS}
    print("\n  EVIDENCE FACTS (not verdicts)")
    for k, v in fact_tally.items():
        print(f"    {k:<24} {v:>4}")

    # ── emitted-value audit against the declared dispositions ────────
    emitted = {(ax, x[ax]["value"]) for x in out for ax in ont.ALPHABETS}
    violations = [(ax, v) for ax, v in emitted
                  if v != ont.CAPABILITY_FAILURE and not ont.emittable(ax, v)]
    if violations:
        raise SystemExit(f"CONTRACT ABORT: forbidden values emitted: {violations}")

    census_manifest = pathlib.Path(a.census_package) / "MANIFEST.sha256"
    result = {
        "instrument": "HOUSE-IN-ORDER-H2-CLASSIFIER v1.1",
        "subject": a.subject,
        "subject_tree": packet["subject_tree"],
        "history_identity": packet["history_identity"],
        "census_dependency": packet["census_dependency"],
        "population": len(out),
        "capability_contract": cap_rows,
        "capability_satisfied": cap_ok,
        "axis_tally": axis_tally,
        "evidence_fact_tally": fact_tally,
        "admission_contract": {
            "classifier": "HOUSE-IN-ORDER-H2-CLASSIFIER v1.1",
            "subject_commit": a.subject,
            "subject_tree": packet["subject_tree"],
            "history_identity": packet["history_identity"],
            "census_dependency": packet["census_dependency"],
            "state_dispositions": ont.disposition_rows(),
            "evidence_facts_are_not_verdicts": list(ont.EVIDENCE_FACTS),
            "population": len(out),
            "restrictions": [
                "LIFECYCLE=ACTIVE is H2_NOT_EARNABLE: no independently "
                "qualified positive rule exists at H2 (D360 §5).",
                "AUTHORITY states are DEFERRED_TO_H3; H2 verifies no claim.",
                "GENERATION verdicts are H2_NOT_EARNABLE while the write "
                "search space is open and region scope undetermined.",
                "SCOPE region overrides are H2_NOT_EARNABLE.",
                "UNKNOWN and UNMEASURED are abstentions. Per D340 §7 and "
                "D358 they may NEVER be used as negative evidence or as "
                "an exclusion criterion by any consumer.",
                "History-derived facts are bound to the declared "
                "observation window and are void outside it.",
            ],
            "self_approval": "NONE. This contract supplies evidence. "
                             "Admission of HOUSE_H2 to HOUSE_H3 is a "
                             "governing decision, not a test result.",
        },
        "rows": out,
    }
    pathlib.Path(a.out).write_text(json.dumps(result, indent=1))
    blob = json.dumps(result["admission_contract"], indent=1,
                      sort_keys=True).encode()
    print(f"\n  admission contract sha256: {hashlib.sha256(blob).hexdigest()}")
    print(f"  written: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
