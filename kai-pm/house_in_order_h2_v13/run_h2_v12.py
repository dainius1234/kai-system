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
            # F6 (Kai): `context` is PRESENTATION, not determining
            # evidence. It is named as an excerpt, and the evidence that
            # actually determined the contradiction travels beside it as
            # a complete 9-field witness -- which is also what lets E1
            # bind this fact to a compliant trace.
            return {"claimed": str(claimed), "git_last": row["last"],
                    "drift_days": (gl - claimed).days,
                    "selector": w["source_selector"],
                    "context_excerpt": w["local_context"][:120],
                    "context_excerpt_is_partial":
                        len(w["local_context"]) > 120,
                    "determining_witness": dict(w)}
    return None


def _witness_trace(row, family):
    """The first emitted witness of `family`, already 9-field compliant."""
    ws = (row["witnesses"] or {}).get(family) or []
    if not ws:
        return None
    w = dict(ws[0])
    w["evidence_total"] = len(ws)
    w["evidence_shown"] = 1
    w["truncated"] = len(ws) > 1
    return w


def _history_trace(row, subject, count):
    """MAINTENANCE_OBSERVED. Genuinely history-derived, so the selector is
    the exact reproducible history operation -- frozen subject, path and
    all -- not the bare `history:<path>` label v1.3a used, which named no
    operation and reproduced nothing.
    """
    sel = f"git:rev-list --count {subject} -- {row['path']}"
    return {"witness_type": "COMMIT_COUNT_IN_WINDOW", "witness_value": str(count),
            "source_path": row["path"], "source_selector": sel,
            "local_context": f"{sel} => {count}; last commit "
                             f"{row.get('last') or 'unknown'}",
            "applicability_scope": "WHOLE_FILE",
            "evidence_total": 1, "evidence_shown": 1, "truncated": False,
            "polarity": "POSITIVE", "certainty": "VERIFIED",
            "temporal": "AT_COMMIT", "subject": "SELF"}


def _reader_trace(row, subject_repo):
    """CONSUMED_AT_SUBJECT. The determining evidence is a STATIC READER
    REFERENCE produced by the Census opscan -- NOT history. v1.3a labelled
    it `history:` , which named the wrong evidence class entirely.

    The evidence lives in the READING document, so `source_path` is the
    reader and `local_context` is that document's actual source line. A
    fact with several reader references carries the D367 5
    evidence_total / evidence_shown semantics rather than dropping any.
    """
    ops = row.get("reader_ops") or []
    if not ops:
        return None                      # A6-ii will abstain, correctly
    o = sorted(ops, key=lambda x: (x["src"], x["line"]))[0]
    try:
        line = (pathlib.Path(subject_repo) / o["src"]).read_text(
            errors="ignore").splitlines()[o["line"] - 1].strip()
    except Exception:
        return None                      # no locator -> no certification
    # The Census stores `expr` as an AST DUMP -- "Name(id='CHANGELOG',
    # ctx=Load())" -- which is a description of the node, not a token in
    # the source. D367 5 wants "the EXACT token or value matched", so the
    # trace carries the identifier the dump names, and only if that
    # identifier is genuinely present in the reading line. If it is not,
    # we abstain rather than certify a value the source does not contain.
    ident = next((g for g in re.findall(r"id='([^']+)'", o["expr"] or "")
                  if g in line), None)
    if not ident:
        return None
    return {"witness_type": "STATIC_READER_REFERENCE",
            "witness_value": ident, "source_path": o["src"],
            "source_selector": f"opscan:{o['src']}:L{o['line']}",
            "local_context": line, "applicability_scope": "SPAN",
            "reader_ast_expr": o["expr"], "reader_mode": o["mode"],
            "evidence_total": len(ops), "evidence_shown": 1,
            "truncated": len(ops) > 1,
            "polarity": "POSITIVE", "certainty": "OBSERVED",
            "temporal": "AT_COMMIT", "subject": f"OTHER:{row['path']}"}


def _claim_trace(row, determining, total):
    """A 9-field trace for a SELF-bound authority fact.

    The determining rows are CLAIM records, not witnesses -- a selector
    and a sentence, but not the nine fields -- so E1 could not bind them
    and A6-ii would have ABSTAINED four SELF_ASSERTS_AUTHORITY and one
    SELF_ASSERTS_NON_AUTHORITY rather than tracing them. That would be E1
    deleting facts instead of binding them, and it would move AUTHORITY,
    which belongs to step 5. The claim is promoted to a trace instead.

    `applicability_scope` is SPAN, not WHOLE_FILE: the evidence is the
    sentence. WHOLE_FILE would be an unearned widening and a scope
    decision, which is M3's at step 2.
    """
    c = determining[0]
    return {"witness_type": "SELF_AUTHORITY_CLAIM",
            "witness_value": c.get("term") or "",
            "source_path": row["path"], "source_selector": c["selector"],
            "local_context": c["text"], "applicability_scope": "SPAN",
            "evidence_total": total, "evidence_shown": len(determining),
            "truncated": len(determining) < total,
            "polarity": c["polarity"], "certainty": "OBSERVED",
            "temporal": "AT_COMMIT", "subject": "SELF"}


NINE_FIELDS = ("witness_type", "witness_value", "source_path",
               "source_selector", "local_context", "applicability_scope",
               "evidence_total", "evidence_shown", "truncated")


# E1 FAMILY GATE (Kai). value-in-context is necessary, not sufficient:
# a generated sentence containing a generated number would pass it. The
# trace's EVIDENCE CLASS must match the producer that made the fact
# positive.
TRACE_CLASS = {
    "MAINTENANCE_OBSERVED":       ("COMMIT_COUNT_IN_WINDOW", "git:rev-list"),
    "CONSUMED_AT_SUBJECT":        ("STATIC_READER_REFERENCE", "opscan:"),
    "CITES_COMMIT":               ("COMMIT", "L"),
    "CITES_RUN":                  ("RUN_ID", "L"),
    "CARRIES_DATE_STAMP":         ("DATE_STAMP", "L"),
    "BINDING_CONTRADICTION":      ("DATE_STAMP", "L"),
    "SELF_ASSERTS_AUTHORITY":     ("SELF_AUTHORITY_CLAIM", "L"),
    "SELF_ASSERTS_NON_AUTHORITY": ("SELF_AUTHORITY_CLAIM", "L"),
}


def _class_ok(name, tr):
    want = TRACE_CLASS.get(name)
    if not want:
        return False
    kind, sel = want
    return tr.get("witness_type") == kind and \
        str(tr.get("source_selector", "")).startswith(sel)


def _compliant(t):
    """E1 / D367 5. Present is not enough -- the trace must be SEMANTICALLY
    TRUTHFUL: all nine fields, and the context must actually contain the
    value it claims to evidence.
    """
    if not t or any(t.get(k) in (None, "") for k in NINE_FIELDS):
        return False
    return str(t["witness_value"]) in str(t["local_context"])


def evidence_facts(row, claims, contradiction, determining=(),
                   subject="", subject_repo="."):
    """FACTS, each bound to the trace that DETERMINED it. None is a
    verdict (D360 5).

    E1. v1.2 emitted these as bare booleans. D367 5 requires every
    POSITIVE evidence fact to carry a source-bound witness sufficient for
    independent adjudication, and 81 of 316 positives carried none that
    was bound to the fact itself: 71 MAINTENANCE_OBSERVED and 5
    CONSUMED_AT_SUBJECT had no witness at all, and 5
    BINDING_CONTRADICTION carried a 5-field contradiction record rather
    than a 9-field witness.

    A6-i: the producer that SETS the boolean carries the witness that set
    it. A6-ii: a positive with no compliant trace is NOT emitted as
    positive -- it abstains. A related trace living elsewhere in the
    package does not qualify; the binding is to the fact.
    """
    cand, traces = {}, {}
    cand["MAINTENANCE_OBSERVED"] = (
        row["commits_in_window"] > 1,
        lambda: _history_trace(row, subject, row["commits_in_window"]))
    cand["CONSUMED_AT_SUBJECT"] = (
        bool(row["readers"]),
        lambda: _reader_trace(row, subject_repo))
    cand["CITES_COMMIT"] = (bool(row["witnesses"].get("COMMIT")),
                            lambda: _witness_trace(row, "COMMIT"))
    cand["CITES_RUN"] = (bool(row["witnesses"].get("RUN_ID")),
                         lambda: _witness_trace(row, "RUN_ID"))
    cand["CARRIES_DATE_STAMP"] = (bool(row["witnesses"].get("DATE")),
                                  lambda: _witness_trace(row, "DATE"))
    cand["BINDING_CONTRADICTION"] = (
        contradiction is not None,
        lambda: dict(contradiction["determining_witness"])
        if contradiction else None)
    ac = sb.authority_claim(claims)
    for name, want in (("SELF_ASSERTS_AUTHORITY", "SELF_ASSERTS_AUTHORITY"),
                       ("SELF_ASSERTS_NON_AUTHORITY",
                        "SELF_ASSERTS_NON_AUTHORITY")):
        cand[name] = (ac == want,
                      (lambda d=determining, n=len(claims): _claim_trace(
                          row, d, n) if d else None))

    f, abstained = {}, []
    for name, (positive, mk) in cand.items():
        if not positive:
            f[name] = False
            continue
        t = mk()
        if _compliant(t) and _class_ok(name, t):
            f[name] = True
            traces[name] = t
        else:                       # A6-ii: no compliant trace, no positive
            f[name] = False
            abstained.append(name)
    return f, ac, traces, abstained


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
        # D14/D15: the DETERMINING rows are always carried, with counts.
        # E1 needs them BEFORE the facts, because a SELF-authority fact
        # must bind to the row that determined it.
        det = sb.determining_claims(claims)
        facts, ac, fact_traces, abstained = evidence_facts(
            row, claims, contradiction, det, pa["subject"], sr)
        row["authority_claim"] = ac
        out = cl.classify(row, text, contradiction)
        out["evidence_facts"] = facts
        # E1: every POSITIVE fact carries the trace that determined it.
        out["evidence_fact_traces"] = fact_traces
        if abstained:
            out["evidence_facts_abstained_no_compliant_trace"] = abstained
        out["authority_claim"] = ac
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
