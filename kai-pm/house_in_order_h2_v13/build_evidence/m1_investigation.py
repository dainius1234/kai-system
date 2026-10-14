#!/usr/bin/env python3
"""STEP 3 / M1 — ROOT-CAUSE INVESTIGATION. MEASUREMENT ONLY.

THIS FILE IMPLEMENTS NO REPAIR. It changes no candidate code, defines no
new axis, whitelists no path and no document name. It reads the accepted
lineage's own output and the frozen subject tree, and reports what is
there. `classify.validity` is NOT called, re-implemented or corrected
here.

THE GOVERNING DEFECT, as ruled by Kai. A review or inspection event is
being promoted into VALIDITY = TIME_BOUND. A review event proves that
someone reviewed the document; it does not, by itself, establish the
document's validity period, currency interval or effective state.

THE SEMANTIC BUCKETS BELOW ARE A REPORTING PARTITION AND DECIDE NOTHING.
They exist so the population can be counted against Kai's invariant
without me quietly inventing the invariant's boundary. Three of the five
buckets are explicitly referred back to Kai as OPEN QUESTIONS rather than
answered here (C, D, E). Anything that falls in none of them is reported
as F_UNCLASSIFIED and travels -- it is not dropped (R17).

MECHANISM, NEVER A LITERAL. The affected population is counted by SOURCE
PREDICATE, not by matching a date string. Counting the literal
"Reviewed: 27 July 2026" gives a DIFFERENT and WRONG answer, and the
report prints both so the gap is visible. That specific error is banked:
it is the D368 / R13 incident.

    python3 m1_investigation.py --subject-repo R --tree T \\
        --passa P --classification C --out F
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import hashlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import passa as P                                              # noqa: E402

# ── the reporting partition. DECLARED, closed-world, decides nothing ──
# A: the source language makes the DOCUMENT'S STATE temporal.
STATE = {"last updated", "updated", "snapshot", "audited snapshot",
         "measured at", "validated checkpoint", "acquisition commit",
         "version", "subject", "findings-bearing snapshot"}
# B: an inspection EVENT. Kai's governing defect class.
REVIEW = {"reviewed", "last reviewed", "review date"}
# C: an ORIGIN / lifecycle event. OPEN QUESTION for Kai -- "Created" is
#    no more a validity interval than "Reviewed" is, but Kai's ruling
#    named only review/inspection. Reported, not decided.
ORIGIN = {"created", "generated", "opened", "sent", "written", "agreed",
          "prepared", "started", "finalised", "finalized",
          "report completed", "log started", "register started",
          "planning date"}
# E: a bare `Date:` label. OPEN QUESTION -- it names a date and no state.
BARE = {"date"}
# D: routes that carry no label at all. OPEN QUESTION, one bucket each.
NON_PREDICATE_ROUTES = ("H1_LINE", "SELF_SUBJECT", "BARE_DATELINE",
                        "CONTEXTUAL_STATUS")


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


class Corpus:
    def __init__(self, repo, tree):
        self.repo, self.tree, self._c = repo, tree, {}

    def text(self, path):
        if path not in self._c:
            try:
                self._c[path] = _git(self.repo, "show",
                                     f"{self.tree}:{path}").decode("utf-8")
            except UnicodeDecodeError as e:
                raise SystemExit(f"R11 ABORT: {path} is not UTF-8 ({e})")
        return self._c[path]

    def head(self, path):
        return self.text(path)[:P.HEAD_BYTES]


def route_and_label(head, start):
    """WHICH M3 route promoted this witness, and the source predicate.

    This MIRRORS `_scope_of`'s own order so the reported route is the one
    that actually fired. It re-decides nothing: the scope itself is read
    from the accepted candidate's output, never recomputed here.
    """
    ls = head.rfind("\n", 0, start) + 1
    le = head.find("\n", start)
    line = head[ls:le if le >= 0 else len(head)]
    before = head[ls:start]
    if P.TABLE_ROW.match(line):
        return "TABLE_ROW", "", line
    if ls >= P._preamble_end(head):
        return "UNDER_H2", "", line
    m1 = P.H1.search(head)
    if m1 and m1.start() == ls:
        return "H1_LINE", "", line
    if P.SELF_SUBJECT.search(line):
        return "SELF_SUBJECT", "", line
    lab = P._label_of(before)
    if lab is not None:
        if P.DOC_BINDING.match(lab):
            return "INTRINSIC_LABEL", lab, line
        if P._contextual_document_metadata(head, ls, lab, line):
            return "CONTEXTUAL_STATUS", lab, line
        return "LABEL_NOT_BINDING", lab, line
    if P.ROOT_LIFECYCLE.search(before):
        return "ROOT_LIFECYCLE", "", line
    if P.BARE_DATELINE.match(before):
        return "BARE_DATELINE", "", line
    return "NONE", "", line


def bucket(route, lab):
    if route in NON_PREDICATE_ROUTES:
        return "D_NON_PREDICATE"
    if route == "ROOT_LIFECYCLE":
        return "C_ORIGIN_EVENT"
    if lab in STATE:
        return "A_DOCUMENT_STATE"
    if lab in REVIEW:
        return "B_REVIEW_EVENT"
    if lab in ORIGIN:
        return "C_ORIGIN_EVENT"
    if lab in BARE:
        return "E_BARE_DATE_LABEL"
    return "F_UNCLASSIFIED"


def offset_of(text, witness):
    """The witness's byte offset, from its own selector and value.

    Pass A records `source_selector` as L<line>. The value is located
    within that line. Fail closed rather than guessing an offset: a
    silently wrong offset would report the wrong route.
    """
    line_no = int(witness["source_selector"].lstrip("L"))
    off = 0
    for i, line in enumerate(text.split("\n"), 1):
        if i == line_no:
            j = line.find(witness["witness_value"])
            if j < 0:
                raise SystemExit(
                    f"R11 ABORT: {witness['witness_value']!r} not on "
                    f"{witness['source_path']} L{line_no}.")
            return off + j
        off += len(line) + 1
    raise SystemExit(f"R11 ABORT: {witness['source_path']} has no L{line_no}.")


def analyse(passa_rows, class_rows, corpus):
    witnesses = {r["path"]: r["witnesses"] for r in passa_rows}
    whole = {}
    for path, w_by_kind in witnesses.items():
        head = corpus.head(path)
        rows = []
        for w in w_by_kind.get("DATE", []):
            if w["applicability_scope"] != "WHOLE_FILE":
                continue
            rt, lb, line = route_and_label(head, offset_of(corpus.text(path), w))
            rows.append({"w": w, "route": rt, "label": lb, "line": line,
                         "bucket": bucket(rt, lb)})
        whole[path] = rows

    tb = [r for r in class_rows if r["VALIDITY"]["value"] == "TIME_BOUND"]
    det = {}
    for r in tb:
        dw = r["VALIDITY"]["witness"]
        match = [x for x in whole[r["path"]]
                 if x["w"]["source_selector"] == dw["source_selector"]
                 and x["w"]["witness_value"] == dw["witness_value"]]
        if not match:
            raise SystemExit(f"R11 ABORT: determining witness for "
                             f"{r['path']} not found among its WHOLE_FILE "
                             f"DATE witnesses.")
        det[r["path"]] = match[0]
    return whole, tb, det


def render(pa, cl, corpus, whole, tb, det, digests):
    o = []
    A = o.append
    rows = cl["rows"]
    A("STEP 3 / M1 — ROOT-CAUSE INVESTIGATION. MEASUREMENT ONLY, NO REPAIR.")
    A("PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT. NOT AN ANSWER KEY.")
    A(f"subject      {cl['subject']}")
    A(f"subject tree {cl['subject_tree']}")
    A(f"admission contract  self_approval="
      f"{cl['admission_contract']['self_approval']} "
      f"status={cl['admission_contract']['status']}")
    A("")
    A("1. THE PROMOTION PATH, exactly")
    A("   classify.py  validity(row, contradiction)  L139-142")
    A("       w = _binding_witness(row, \"DATE\")")
    A("       if w is not None:")
    A("           return E.claim(w, \"TIME_BOUND\", scope=\"WHOLE_FILE\",")
    A("                          rationale=\"document-level date binding\")")
    A("   classify.py  _binding_witness(row, *kinds)  L98-104 — the FIRST")
    A("   witness of the named kind whose applicability_scope is already")
    A("   WHOLE_FILE, in Pass A emission order (source order).")
    A("")
    A("2. WHAT ENTERS THE PATH")
    A("   passa.scan L509-518 emits every DATE token starting inside the")
    A("   head window as witness_type=DATE_STAMP. The PREDICATE that")
    A("   earned document scope is NOT in witness_type -- every date is")
    A("   DATE_STAMP -- but it IS carried verbatim in `local_context`,")
    A("   which since the F5 repair holds the complete logical source")
    A("   line. The evidence needed to tell a review event from a state")
    A("   binding is therefore ALREADY PRESENT and simply never read.")
    A("")
    A("3. VALIDITY AS MEASURED UNDER THE ACCEPTED LINEAGE")
    A(f"   population rows {cl['population']}   "
      f"VALIDITY {cl['axis_tallies']['VALIDITY']}")
    A("")
    A("4. DETERMINING-WITNESS BUCKET over the TIME_BOUND rows")
    by = collections.Counter(v["bucket"] for v in det.values())
    for k in sorted(by):
        A(f"     {k:<22}{by[k]:>5}")
    A("")
    A("5. DETERMINING ROUTE + SOURCE PREDICATE, every combination")
    lab = collections.Counter((v["route"], v["label"]) for v in det.values())
    for (rt, lb), c in lab.most_common():
        A(f"     {rt:<20}{lb!r:<22}{c:>5}")
    A("")
    A("6. THE AFFECTED POPULATION")
    aff, rescued = [], []
    for p, v in det.items():
        if v["bucket"] != "B_REVIEW_EVENT":
            continue
        others = [x for x in whole[p]
                  if x is not v and x["bucket"] == "A_DOCUMENT_STATE"]
        (rescued if others else aff).append((p, v, others))
    A(f"     review event determines TIME_BOUND            {len(aff) + len(rescued)}")
    A(f"     of those, NO independent document-state binding {len(aff)}"
      f"   <- AFFECTED")
    A(f"     of those, an independent A-class binding exists {len(rescued)}"
      f"   <- would still promote under Kai's invariant")
    noother = sum(1 for p, v, _ in aff if len(whole[p]) == 1)
    A(f"     affected rows carrying NO OTHER document-scoped DATE witness"
      f" of any bucket: {noother} of {len(aff)}")
    A("")
    A("7. LITERAL vs MECHANISM — the D368 / R13 trap, shown not asserted")
    carriers = []
    for r in rows:
        head = corpus.head(r["path"])
        for i, line in enumerate(head.split("\n"), 1):
            if P._label_of(line) in REVIEW:
                carriers.append((r["path"], i, line.strip()))
                break
    lit = sum(1 for _, _, l in carriers if "Reviewed: 27 July 2026" in l)
    A(f"     documents carrying a root review-class PREDICATE : {len(carriers)}")
    A(f"     documents matching the LITERAL 'Reviewed: 27 July 2026'"
      f" : {lit}")
    A(f"     distinct review predicates observed: "
      f"{sorted({P._label_of(l) for _, _, l in carriers})}")
    A("     Counting the literal would report the wrong population. The")
    A("     affected figure above is a predicate count, not a string count.")
    A("")
    A("8. PROTECTED POSITIVES, RE-MEASURED. HISTORICAL 10 NOT ASSUMED.")
    A(f"     strict A only                          {by['A_DOCUMENT_STATE']}")
    A(f"     A + E (bare 'Date:' label)             "
      f"{by['A_DOCUMENT_STATE'] + by['E_BARE_DATE_LABEL']}")
    A(f"     everything not B                       "
      f"{len(tb) - by['B_REVIEW_EVENT']}")
    A("     The boundary between these three is KAI'S RULING, not mine.")
    A("")
    A("9. EVERY NON-B DETERMINING ROW — the candidate protected set")
    for p in sorted(det):
        v = det[p]
        if v["bucket"] == "B_REVIEW_EVENT":
            continue
        A(f"     {v['bucket']:<20}{v['route']:<18}{v['label']!r:<16}{p}"
          f"  {v['w']['source_selector']}")
        A(f"       {v['line'][:110]}")
    A("")
    A(f"10. EVERY AFFECTED ROW ({len(aff)}), with its raw source line")
    for p, v, _ in sorted(aff):
        A(f"     {p}\t{v['w']['source_selector']}\t{v['label']}")
        A(f"       {v['line']}")
    A("")
    A("11. DOCUMENTS WHERE AN A-CLASS BINDING IS NOT THE DETERMINING ONE")
    A("    (determining-witness selection is source order, not semantics)")
    n = 0
    for p, v in sorted(det.items()):
        others = [x for x in whole[p]
                  if x is not v and x["bucket"] == "A_DOCUMENT_STATE"]
        if others:
            n += 1
            A(f"     {p}  determining={v['bucket']}/{v['label']!r} "
              f"{v['w']['source_selector']}  also A: "
              f"{[(o['label'], o['w']['source_selector']) for o in others]}")
    A(f"     count: {n}")
    A("")
    A("12. SCOPE COLLATERAL — the SAME helper feeds a different axis")
    wit = {r["path"]: r["witnesses"] for r in pa["rows"]}
    scope_wf = [r for r in rows if r["SCOPE"]["value"] == "WHOLE_FILE"]
    affset = {p for p, _, _ in aff}
    coll = 0
    for r in scope_wf:
        if r["path"] not in affset:
            continue
        other = [x for k in ("COMMIT", "RUN_ID", "SUPERSEDED_BY")
                 for x in wit[r["path"]].get(k, [])
                 if x["applicability_scope"] == "WHOLE_FILE"]
        if not other:
            coll += 1
    A(f"     SCOPE=WHOLE_FILE rows                         {len(scope_wf)}")
    A(f"     of those, affected AND with no other binding  {coll}")
    A("     scope() consumes _binding_witness too. A repair placed in the")
    A("     SHARED helper would move these SCOPE verdicts as well. A root")
    A("     'Reviewed:' date IS document-scoped, so SCOPE=WHOLE_FILE looks")
    A("     CORRECT and must not move. This constrains where a repair may")
    A("     be placed. Reported as a constraint, not acted on.")
    A("")
    A("13. REPRODUCTION — the run artefacts are NOT banked, deliberately")
    A("    Banking the candidate's full verdict aggregate in the repository")
    A("    would place a self-referential input where a later Stage A")
    A("    precommit could read it. Kai's architecture forbids a")
    A("    self-referential aggregate, so only this derived report and the")
    A("    instrument are banked. The inputs are HASH-PINNED and")
    A("    deterministically regenerable -- two independent runs produced")
    A("    byte-identical output.")
    for k, v in digests.items():
        A(f"      sha256({k:<15}) {v}")
    A("      python3 passa.py --subject-repo <SUBJECT> \\")
    A("          --history-repo <SUBJECT> --subject "
      f"{cl['subject']} \\")
    A("          --census-package kai-pm/house_in_order_census_v11 "
      "--out passA.json")
    A("      python3 run_h2_v12.py --subject-repo <SUBJECT> "
      "--passa passA.json \\")
    A("          --out classification.json")
    A("      python3 build_evidence/m1_investigation.py --subject-repo "
      "<SUBJECT> \\")
    A(f"          --tree {cl['subject_tree']} \\")
    A("          --passa passA.json --classification classification.json "
      "--out F")
    A("")
    A("14. UNCLASSIFIED — everything the partition did not cover")
    unc = [(p, v) for p, v in det.items() if v["bucket"] == "F_UNCLASSIFIED"]
    A(f"     {len(unc)}")
    for p, v in sorted(unc):
        A(f"       {p} {v['route']} {v['label']!r} :: {v['line'][:90]}")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--passa", required=True)
    ap.add_argument("--classification", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pa = json.loads(pathlib.Path(a.passa).read_text())
    cl = json.loads(pathlib.Path(a.classification).read_text())
    if pa["subject_tree"] != a.tree or cl["subject_tree"] != a.tree:
        raise SystemExit(f"R11 ABORT: run artefacts are not bound to "
                         f"{a.tree[:12]}.")
    digests = {label: hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
               for label, path in (("passA", a.passa),
                                   ("classification", a.classification))}
    corpus = Corpus(a.subject_repo, a.tree)
    whole, tb, det = analyse(pa["rows"], cl["rows"], corpus)
    text = render(pa, cl, corpus, whole, tb, det, digests)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
