#!/usr/bin/env python3
"""READ-ONLY VALIDITY-BINDING AUDIT — evidence instrument, NOT a classifier.

Banked under D364 as EVIDENCE ONLY. This script measures; it repairs
nothing, and it is not part of the HOUSE_H2 package. It must never be
imported by a classifier.

WHAT IT ANSWERS, for every non-abstention VALIDITY verdict HOUSE_H2 v1.1
emits: does the evidence that earned the verdict prove a WHOLE-DOCUMENT
validity state, or does it merely record that the document contains that
evidence somewhere?

WHY IT IS BANKED. The measurement is history-sensitive. The active
repository is SHALLOW (ancestry 280 at the subject, oldest 2026-08-05)
and CANNOT reproduce the date-drift results, which need the full history
(ancestry 986, oldest 2025-06-18). Re-derivation requires a fresh
non-shallow clone; without this file the method would have to be
reconstructed from prose.

Every source is a PARAMETER. v1.0's pass_a.py hard-coded a home
directory while its RUN.md advertised generic inputs; that is what made
it unreproducible elsewhere (D340 5).

    python3 validity_binding_audit.py \
        --subject-repo <exact checkout at the subject> \
        --history-repo <NON-SHALLOW clone containing the subject> \
        --subject <40-hex> \
        --package <house_in_order_h2_v11 dir> \
        --out audit.json

TWO INDEPENDENT CHECKS ARE THE POINT (I-8). Neither asks the pattern
that produced a witness whether the witness is real:

  1. does a commit-shaped witness RESOLVE as a commit in the declared
     history source?  ('ed25519' does not. Nor does a Docker digest
     fragment, nor an 11-digit workflow run id.)
  2. does a self-claimed date SURVIVE the history?  ('Last updated: X'
     is an assertion by the document; git says when it last changed.)

CAVEAT CARRIED IN THE OUTPUT: "does not resolve" is NOT proof a token is
not a commit somewhere else. It IS proof the classifier never checked.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import json
import pathlib
import re
import subprocess
import sys

HEADING = re.compile(r"^#{1,6} ", re.M)
IMMUTABLE_OID = re.compile(r"[0-9a-f]{40}")

# A binding statement names THE DOCUMENT as its subject. Kai's ruling:
# the requirement is SEMANTIC, not a Markdown-header layout rule -- so
# this matches the labelled-field FORM wherever it occurs, and the
# adjudication of each hit is reported with its context for review. It
# is a nomination, never a silent verdict.
BINDING_LABEL = re.compile(
    r"(audited snapshot|acquisition commit|validated checkpoint|"
    r"findings-bearing[^:]*|subject|measured at|snapshot)\s*:", re.I)
CURRENCY_LABEL = re.compile(
    r"(last updated|last reviewed|reviewed|last-updated|date)\s*:", re.I)

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=str(repo),
                          capture_output=True, text=True)


def parse_date(tok):
    t = tok.strip().replace(",", "")
    m = re.fullmatch(r"(20\d\d)-(\d\d)-(\d\d)", t)
    if m:
        return datetime.date(*map(int, m.groups()))
    m = re.fullmatch(r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(20\d\d)", t)
    if m:
        return datetime.date(int(m.group(3)),
                             MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
    m = re.fullmatch(r"([A-Za-z]+)\.?\s+(\d{1,2})\s+(20\d\d)", t)
    if m:
        return datetime.date(int(m.group(3)),
                             MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
    return None


def audit(subject_repo, history_repo, subject, package, result):
    """One row per non-abstention VALIDITY verdict. No verdict is repaired."""
    sys.path.insert(0, str(package))
    import passa  # the SUBJECT of the audit, read for its own patterns

    pats = {"EXACT_SNAPSHOT": passa.SHA, "RUN_ARTEFACT": passa.RUN,
            "TIME_BOUND": passa.DATE, "CURRENT_TREE": passa.PRESENT}
    rows = []
    for x in result["rows"]:
        verdict = x["VALIDITY"]["value"]
        if verdict not in pats:
            continue                      # UNKNOWN / UNMEASURED: abstentions
        p = x["path"]
        txt = (pathlib.Path(subject_repo) / p).read_text(errors="ignore")
        m = pats[verdict].search(txt[:6000])
        if m is None:
            rows.append({"path": p, "verdict": verdict,
                         "class": "AMBIGUOUS",
                         "why": "no witness recoverable from the head window"})
            continue
        off = m.start()
        ls = txt.rfind("\n", 0, off) + 1
        le = txt.find("\n", off)
        ctx = txt[ls:le if le > 0 else len(txt)].strip()
        row = {
            "path": p, "verdict": verdict, "witness": m.group(0),
            "line": txt[:off].count("\n") + 1,
            "total_lines": txt.count("\n") + 1,
            "context": ctx,
            "pct_after": round(100 * (len(txt) - off) / max(len(txt), 1)),
            "sections_after": len(HEADING.findall(txt[off:])),
            "binding_label_on_witness_line": bool(BINDING_LABEL.search(ctx)),
        }

        # ── check 1: does a commit-shaped witness resolve? ──────────────
        if verdict == "EXACT_SNAPSHOT":
            tok = m.group(0)
            row["preceded_by_sha256_prefix"] = bool(
                re.search(r"sha256:\s*$", txt[max(0, off - 12):off]))
            row["resolves_as_commit"] = git(
                history_repo, "cat-file", "-e",
                f"{tok}^{{commit}}").returncode == 0

        # ── check 2: does a self-claimed date survive the history? ──────
        if verdict == "TIME_BOUND":
            claimed = parse_date(m.group(0))
            last = git(history_repo, "log", "-1", "--format=%ad",
                       "--date=short", subject, "--", p).stdout.strip()
            row["currency_label_on_witness_line"] = bool(
                CURRENCY_LABEL.search(ctx))
            if claimed and last:
                gl = datetime.date(*map(int, last.split("-")))
                row.update(claimed_date=str(claimed), git_last=last,
                           drift_days=(gl - claimed).days,
                           binding_contradiction=(gl - claimed).days > 0)

        rows.append(row | classify(row))
    return rows


def classify(row):
    """Adjudication under Kai's five classes. NOMINATION, not authority --
    every row ships its witness and context so an independent reviewer
    can overturn it."""
    v = row["verdict"]
    if v == "EXACT_SNAPSHOT":
        if row.get("preceded_by_sha256_prefix"):
            return {"class": "FALSE_POSITIVE",
                    "why": "witness is a sha256 DIGEST, not a commit"}
        if not row.get("resolves_as_commit"):
            return {"class": "FALSE_POSITIVE",
                    "why": "witness does not resolve as a commit in the "
                           "declared history source"}
        if row["binding_label_on_witness_line"]:
            return {"class": "WHOLE_FILE_BINDING_PROVEN",
                    "why": "document-level binding statement"}
        return {"class": "REGION_OR_CITATION_ONLY",
                "why": "commit cited for one local statement"}
    if v == "CURRENT_TREE":
        return {"class": "FALSE_POSITIVE",
                "why": "present-tense token describes a component or another "
                       "document, not this document's currency"}
    if v == "TIME_BOUND":
        if not row.get("currency_label_on_witness_line") or row["line"] > 10:
            return {"class": "REGION_OR_CITATION_ONLY",
                    "why": "date is a heading/table/prose token, not a "
                           "document-level stamp"}
        return {"class": "SELF_CLAIM_ONLY",
                "why": "document-level date stamp — an assertion by the "
                       "document, not a demonstrated binding"}
    if v == "RUN_ARTEFACT":
        return {"class": "AMBIGUOUS", "why": "population empty on this subject"}
    return {"class": "AMBIGUOUS", "why": "unhandled verdict"}


def main():
    ap = argparse.ArgumentParser(description="read-only VALIDITY audit")
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--history-repo", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--result", default=None,
                    help="h2v11-classification.json (default: in --package)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # R11: prove the subject and the history source BEFORE measuring.
    if not IMMUTABLE_OID.fullmatch(a.subject):
        raise SystemExit("R11 ABORT: --subject must be an immutable 40-hex id")
    head = git(a.subject_repo, "rev-parse", "HEAD").stdout.strip()
    if head != a.subject:
        raise SystemExit(f"R11 ABORT: subject repo HEAD {head[:12]} != "
                         f"subject {a.subject[:12]}")
    if git(a.history_repo, "cat-file", "-e",
           f"{a.subject}^{{commit}}").returncode != 0:
        raise SystemExit("R11 ABORT: subject absent from the history source")
    shallow = git(a.history_repo, "rev-parse",
                  "--is-shallow-repository").stdout.strip()
    if shallow != "false":
        raise SystemExit(
            "R11 ABORT: history source is SHALLOW. The date-drift check "
            "compares a self-claimed date against the last commit touching "
            "the document; a truncated window silently produces a WRONG "
            "answer rather than no answer. Refusing to measure.")

    pkg = pathlib.Path(a.package)
    result = json.load(open(a.result or pkg / "h2v11-classification.json"))
    if result["subject"] != a.subject:
        raise SystemExit(f"R11 ABORT: result subject {result['subject'][:12]} "
                         f"!= --subject {a.subject[:12]}")

    rows = audit(a.subject_repo, a.history_repo, a.subject, pkg, result)
    first = git(a.history_repo, "log", "--reverse", "--format=%H %ad",
                "--date=short").stdout.splitlines()[0].split()
    payload = {
        "audit": "VALIDITY_BINDING", "banked_under": "D364 (evidence only)",
        "subject": a.subject, "subject_tree": result["subject_tree"],
        "population_total": result["population"],
        "population_audited": len(rows),
        "history_identity": {
            "origin": git(a.history_repo, "remote", "get-url",
                          "origin").stdout.strip(),
            "shallow": shallow,
            "ancestry_at_subject": int(git(
                a.history_repo, "rev-list", "--count",
                a.subject).stdout.strip() or 0),
            "oldest_commit": first[0], "oldest_date": first[1],
        },
        "caveat": "A witness that does not resolve is not PROVEN not to be a "
                  "commit elsewhere. It is proven that the classifier never "
                  "checked.",
        "rows": rows,
    }
    tally = collections.Counter(r["class"] for r in rows)
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        by[r["verdict"]][r["class"]] += 1
    payload["tally"] = dict(tally)
    payload["tally_by_verdict"] = {k: dict(v) for k, v in by.items()}
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))

    print(f"VALIDITY-BINDING AUDIT — subject {a.subject[:12]} "
          f"tree {result['subject_tree'][:12]}")
    print(f"  history: ancestry {payload['history_identity']['ancestry_at_subject']}"
          f"  oldest {payload['history_identity']['oldest_date']}"
          f"  shallow={shallow}")
    print(f"  population {result['population']}, "
          f"non-abstention VALIDITY {len(rows)}\n")
    order = ["WHOLE_FILE_BINDING_PROVEN", "SELF_CLAIM_ONLY",
             "REGION_OR_CITATION_ONLY", "AMBIGUOUS", "FALSE_POSITIVE"]
    print(f"  {'emitted state':<16}{'N':>3} " +
          "".join(f"{c[:9]:>11}" for c in order))
    for v in ("RUN_ARTEFACT", "EXACT_SNAPSHOT", "TIME_BOUND", "CURRENT_TREE"):
        n = sum(by[v].values())
        print(f"  {v:<16}{n:>3} " + "".join(f"{by[v][c]:>11}" for c in order))
    print(f"  {'TOTAL':<16}{len(rows):>3} " +
          "".join(f"{tally[c]:>11}" for c in order))
    proven = tally["WHOLE_FILE_BINDING_PROVEN"]
    print(f"\n  WHOLE_FILE_BINDING_PROVEN : {proven} of {len(rows)}")
    print(f"  NOT PROVEN AT CLAIMED SCOPE: {len(rows) - proven} of {len(rows)}")
    print(f"  adjudicated FALSE_POSITIVE : {tally['FALSE_POSITIVE']}")
    bc = [r for r in rows if r.get("binding_contradiction")]
    print(f"\n  BINDING_CONTRADICTION (self-claimed date the history refutes)"
          f": {len(bc)}")
    for r in sorted(bc, key=lambda r: -r["drift_days"])[:5]:
        print(f"    +{r['drift_days']:>3}d  claims {r['claimed_date']}  "
              f"git {r['git_last']}  {r['path']}")
    print("\n  NO VERDICT REPAIRED. NO CLASSIFIER CHANGED. EVIDENCE ONLY.")


if __name__ == "__main__":
    main()
