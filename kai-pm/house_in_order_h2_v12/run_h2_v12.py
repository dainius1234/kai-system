#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — RUNNER. Assembles evidence facts and verdicts.

Emits a machine-readable admission contract that approves NOTHING about
itself. Admission is a governing decision, never a test result.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify as cl                                          # noqa: E402
import ontology as ont                                         # noqa: E402
import passa                                                   # noqa: E402
import subjectbind as sb                                       # noqa: E402

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}
# A CURRENCY label claims the document's own up-to-dateness, which git
# can refute. A version or planning date does not make that claim, so a
# later commit does not contradict it.
CURRENCY = re.compile(r"last updated|last reviewed|^reviewed|last-updated",
                      re.I)


def parse_date(tok):
    t = tok.strip().replace(",", "")
    for pat, order in (
            (r"(20\d\d)-(\d\d)-(\d\d)", "ymd"),
            (r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(20\d\d)", "dmy"),
            (r"([A-Za-z]+)\.?\s+(\d{1,2})\s+(20\d\d)", "mdy")):
        m = re.fullmatch(pat, t)
        if not m:
            continue
        if order == "ymd":
            return datetime.date(*map(int, m.groups()))
        if order == "dmy":
            return datetime.date(int(m.group(3)),
                                 MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
        return datetime.date(int(m.group(3)),
                             MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
    return None


def contradiction_of(row):
    """D3: a CURRENCY self-claim the history refutes.

    `last` was already in the Pass A row and v1.1 never consulted it.
    Only currency labels are testable this way -- a version date may
    legitimately precede a later edit, so it is not a contradiction.
    """
    for w in row["witnesses"].get("DATE", []):
        if w["applicability_scope"] != "WHOLE_FILE":
            continue
        if not CURRENCY.search(w["local_context"]):
            continue
        claimed = parse_date(w["witness_value"])
        if not claimed or not row.get("last"):
            continue
        try:
            gl = datetime.date(*map(int, row["last"].split("-")))
        except Exception:
            continue
        if (gl - claimed).days > 0:
            return {"claimed": str(claimed), "git_last": row["last"],
                    "drift_days": (gl - claimed).days,
                    "selector": w["source_selector"],
                    "context": w["local_context"][:120]}
    return None


def evidence_facts(row, claims, contradiction):
    """FACTS. None of these is a verdict (D360 5)."""
    f = {}
    f["MAINTENANCE_OBSERVED"] = row["commits_in_window"] > 1
    f["CONSUMED_AT_SUBJECT"] = bool(row["readers"])
    f["CITES_COMMIT"] = bool(row["witnesses"].get("COMMIT"))
    f["CITES_RUN"] = bool(row["witnesses"].get("RUN_ID"))
    f["CARRIES_DATE_STAMP"] = bool(row["witnesses"].get("DATE"))
    f["BINDING_CONTRADICTION"] = contradiction is not None
    ac = sb.authority_claim(claims)
    f["SELF_ASSERTS_AUTHORITY"] = ac == "SELF_ASSERTS_AUTHORITY"
    f["SELF_ASSERTS_NON_AUTHORITY"] = ac == "SELF_ASSERTS_NON_AUTHORITY"
    return f, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--passa", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pa = json.load(open(a.passa))
    sr = pathlib.Path(a.subject_repo)
    head = passa.git(sr, "rev-parse", "HEAD").stdout.strip()
    if head != pa["subject"]:
        raise SystemExit(f"R11 ABORT: subject repo HEAD {head[:12]} != "
                         f"Pass A subject {pa['subject'][:12]}")

    rows, facts_tally = [], collections.Counter()
    nominal = collections.Counter()
    for row in pa["rows"]:
        text = (sr / row["path"]).read_text(errors="ignore")
        claims, stats = sb.bind_claims(row["path"], text)
        contradiction = contradiction_of(row)
        facts, ac = evidence_facts(row, claims, contradiction)
        row["authority_claim"] = ac
        out = cl.classify(row, text, contradiction)
        out["evidence_facts"] = facts
        out["authority_claim"] = ac
        # D14/D15: the DETERMINING rows are always carried, with counts
        det = sb.determining_claims(claims)
        out["authority_evidence"] = {
            "determining": det, "total": stats["total"],
            "shown": len(det),
            "truncated": len(det) < stats["total"] and False,
            "note": "SELF-bound determining rows are carried in full; "
                    "non-determining claims are counted, not dropped",
            "counts": stats}
        if contradiction:
            out["binding_contradiction"] = contradiction
        for k, v in facts.items():
            if v:
                facts_tally[k] += 1
        if out["FUNCTION"].get("observed", "").startswith("NOMINAL_FUNCTION="):
            nominal[out["FUNCTION"]["observed"].split("=", 1)[1]] += 1
        rows.append(out)

    assert len(rows) == pa["population"], "population mismatch"
    tallies = {ax: dict(collections.Counter(r[ax]["value"] for r in rows))
               for ax in ont.ALPHABETS}
    payload = {
        "instrument": "HOUSE_H2_CLASSIFIER_v1.2",
        "subject": pa["subject"], "subject_tree": pa["subject_tree"],
        "history_identity": pa["history_identity"],
        "census_dependency": pa["census_dependency"],
        "population": pa["population"], "rows": rows,
        "axis_tallies": tallies,
        "evidence_fact_tally": dict(facts_tally),
        "nominal_function_tally": dict(nominal),
        "admission_contract": {
            "self_approval": "NONE",
            "status": "CANDIDATE. NOT FROZEN. NOT ADMITTED.",
            "note": "Admission is a governing decision, never a test "
                    "result. This package supplies evidence and approves "
                    "nothing about itself.",
            "dispositions": ont.disposition_rows(),
        },
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))

    print(f"HOUSE_H2 v1.2 — {len(rows)} rows == population {pa['population']}")
    print(f"  subject {pa['subject'][:12]} tree {pa['subject_tree'][:12]}\n")
    for ax in ont.ALPHABETS:
        t = tallies[ax]
        pos = {k: v for k, v in t.items() if k not in ("UNKNOWN", "UNMEASURED")}
        print(f"  {ax:<12} positives {sum(pos.values()):>4}  "
              f"UNKNOWN {t.get('UNKNOWN', 0):>4}   {pos or ''}")
    print("\n  evidence facts (NOT verdicts):")
    for k in ont.EVIDENCE_FACTS:
        if k in facts_tally:
            print(f"    {k:<28}{facts_tally[k]:>5}")
    print(f"\n  NOMINAL_FUNCTION (self-description, earns no verdict): "
          f"{sum(nominal.values())}")
    print(f"    {dict(nominal)}")
    ac = hashlib.sha256(json.dumps(payload["admission_contract"],
                                   sort_keys=True).encode()).hexdigest()
    print(f"\n  admission contract sha256 {ac}")
    print("  self_approval: NONE")


if __name__ == "__main__":
    main()
