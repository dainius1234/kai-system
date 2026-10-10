#!/usr/bin/env python3
"""THE E2 EVALUATOR. Produces E2_RESULT.txt.

BANKED WITH THE IMPLEMENTATION, BEFORE THE CORPUS RUN. Written without
having seen any E2 corpus output, run ONCE afterwards, and the
implementation is not altered on the strength of what it reports.

HOW OLD AND NEW ARE MADE COMPARABLE. The package is copied to a temporary
directory and the TWO files E2 touched -- run_h2_v12.py and ontology.py --
are restored from the predecessor git object. Every other module is
asserted byte-identical. The full runner is then executed in BOTH trees
against the SAME Pass A input, so any difference is attributable to E2
and to nothing else.

Pass A is produced once and shared, because passa.py is untouched. The
evaluator verifies that too.

    python3 eval_e2.py --subject-repo R --tree T --census-package C
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

E2_PREDECESSOR = "a766493"
OLD_FACT = "CONSUMED_AT_SUBJECT"
NEW_FACT = "STATIC_REFERENCE_AT_SUBJECT"
CHANGED = ("run_h2_v12.py", "ontology.py")
UNCHANGED = ("passa.py", "classify.py", "envelope.py", "subjectbind.py",
             "qualify.py", "holdout.py")
PKG_PATH = "kai-pm/house_in_order_h2_v13"
# The M1 canonical result, as accepted by Kai at c4960ff. Stated so the
# invariance check compares against the RULED figures, not against
# whatever this run happens to produce.
M1_VALIDITY = {"UNKNOWN": 260, "TIME_BOUND": 7, "EXACT_SNAPSHOT": 5}
M3_SCOPE = (201, 291)


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(map(str, cmd))}\n"
                         f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout


def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def predecessor_pkg(repo, ref, work):
    dst = work / "pred"
    shutil.copytree(PKG, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "build_evidence"))
    for name in CHANGED:
        (dst / name).write_bytes(_git(repo, "show", f"{ref}:{PKG_PATH}/{name}"))
        if digest(dst / name) == digest(PKG / name):
            raise SystemExit(f"R11 ABORT: {name} is identical to the "
                             f"predecessor; nothing is being compared.")
    for name in UNCHANGED:
        if (PKG / name).exists() and digest(dst / name) != digest(PKG / name):
            raise SystemExit(f"R11 ABORT: {name} differs between the trees.")
    return dst


def render(old, new, pa, digests, changed_files):
    o = []
    A = o.append
    orows = {r["path"]: r for r in old["rows"]}
    nrows = {r["path"]: r for r in new["rows"]}
    A("E2 — RESULT. PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT.")
    A("NOT AN ANSWER KEY. NO TARGET COUNT WAS USED.")
    A(f"subject      {new['subject']}")
    A(f"subject tree {new['subject_tree']}")
    A(f"predecessor  {E2_PREDECESSOR}, run_h2_v12.py + ontology.py from the "
      f"git object")
    A(f"production files changed: {', '.join(changed_files)}")
    A("only those two differ between the runs; asserted, not assumed")
    A("")

    A("1. CONSERVATION")
    A(f"   rows in {pa['population']} · predecessor {len(orows)} · "
      f"E2 {len(nrows)} · identical path set {set(orows) == set(nrows)}")
    wf = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "WHOLE_FILE")
    sp = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "SPAN")
    A(f"   Pass A witnesses {wf + sp} · M3 scope {wf}/{sp} "
      f"(ruled {M3_SCOPE[0]}/{M3_SCOPE[1]}: {(wf, sp) == M3_SCOPE})")
    A("")

    A("2. THE FACT DELTA")
    ot = old["evidence_fact_tally"]
    nt = new["evidence_fact_tally"]
    A(f"   predecessor {OLD_FACT:<30}{ot.get(OLD_FACT)}")
    A(f"   E2          {NEW_FACT:<30}{nt.get(NEW_FACT)}")
    A(f"   OLD name present in the E2 tally : {OLD_FACT in nt}")
    A(f"   NEW name present in the old tally: {NEW_FACT in ot}")
    op = sorted(p for p, r in orows.items() if r["evidence_facts"].get(OLD_FACT))
    np_ = sorted(p for p, r in nrows.items()
                 if r["evidence_facts"].get(NEW_FACT))
    A(f"   positive identities identical    : {op == np_}")
    A("   the identities:")
    for p in np_:
        t = nrows[p]["evidence_fact_traces"][NEW_FACT]
        A(f"     {p}")
        A(f"       {t['source_selector']}  value={t['witness_value']!r}  "
          f"total={t['evidence_total']} shown={t['evidence_shown']} "
          f"truncated={t['truncated']}")
    A("")

    A("3. TRACE IDENTITY across the whole corpus")
    moved = [p for p in np_
             if orows[p]["evidence_fact_traces"][OLD_FACT]
             != nrows[p]["evidence_fact_traces"][NEW_FACT]]
    A(f"   rows whose trace content changed : {len(moved)}   (must be 0)")
    for p in moved:
        A(f"     <<< {p}")
    A("")

    A("4. THE OLD NAME MUST NOT SURVIVE ANYWHERE IN THE OUTPUT")
    blob = json.dumps(new)
    A(f"   {OLD_FACT!r} anywhere in the E2 record : {OLD_FACT in blob}")
    for word in ("consumed", "executable", "executed", "runtime", "reachab",
                 "invoc"):
        A(f"   {word!r:<14} anywhere in the E2 record : {word in blob.lower()}")
    A("")

    A("5. EVERY OTHER EVIDENCE FACT — must be unchanged")
    for k in sorted(set(ot) | set(nt)):
        if k in (OLD_FACT, NEW_FACT):
            continue
        same = ot.get(k) == nt.get(k)
        A(f"   {k:<32}{ot.get(k, 0):>5} -> {nt.get(k, 0):<5}"
          f"{'' if same else '   <<< MOVED'}")
    A(f"   NOMINAL_FUNCTION tally unchanged: "
      f"{old['nominal_function_tally'] == new['nominal_function_tally']}")
    A(f"   A6-ii abstention lists unchanged: "
      f"{[r.get('evidence_facts_abstained_no_compliant_trace') for r in old['rows']] == [r.get('evidence_facts_abstained_no_compliant_trace') for r in new['rows']]}")
    A("")

    A("6. ALL SIX AXES — must be unchanged, and M1/M3 must hold")
    for ax in sorted(old["axis_tallies"]):
        same = old["axis_tallies"][ax] == new["axis_tallies"][ax]
        A(f"   {ax:<12}{'UNCHANGED' if same else '<<< MOVED':<12}"
          f"{new['axis_tallies'][ax]}")
    A(f"   VALIDITY == the M1 ruled result  : "
      f"{new['axis_tallies']['VALIDITY'] == M1_VALIDITY}")
    A("")

    A("7. PER-ROW VERDICT DIFF, all six axes")
    diffs = [(p, ax) for p in nrows for ax in old["axis_tallies"]
             if orows[p][ax]["value"] != nrows[p][ax]["value"]]
    A(f"   rows x axes differing            : {len(diffs)}   (must be 0)")
    for p, ax in diffs:
        A(f"     <<< {p} {ax} {orows[p][ax]['value']} -> "
          f"{nrows[p][ax]['value']}")
    A("")

    A("8. READER IDENTITIES — unchanged by construction, verified anyway")
    rd = {r["path"]: (r["readers"], r["reader_ops"]) for r in pa["rows"]}
    A(f"   rows with readers                : "
      f"{sum(1 for v in rd.values() if v[0])}")
    A(f"   rows with reader_ops             : "
      f"{sum(1 for v in rd.values() if v[1])}")
    A("   passa.py is byte-identical to the predecessor, so these come "
      "from the same Pass A for both runs.")
    A("")

    A("9. INPUT DIGESTS")
    for k, v in digests.items():
        A(f"   {k:<28}{v}")
    A("")
    A("10. WHAT THIS IS NOT")
    A("   Not a correctness reference, not an answer key, not an")
    A("   adjudication. A delta is not automatically correct and no delta")
    A("   has been repaired. The five was used as a locator only, never")
    A("   as a target.")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--old-ref", default=E2_PREDECESSOR)
    ap.add_argument("--out", default=str(HERE / "E2_RESULT.txt"))
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="e2_eval_"))
    pred = predecessor_pkg(a.repo, a.old_ref, work)
    subject = _git(a.subject_repo, "rev-parse", "HEAD").decode().strip()

    passa_out = work / "passA.json"
    run([sys.executable, str(PKG / "passa.py"),
         "--subject-repo", a.subject_repo, "--history-repo", a.subject_repo,
         "--subject", subject, "--census-package", a.census_package,
         "--out", str(passa_out)])
    pa = json.loads(passa_out.read_text())
    if pa["subject_tree"] != a.tree:
        raise SystemExit(f"R11 ABORT: Pass A is bound to "
                         f"{pa['subject_tree'][:12]}, not {a.tree[:12]}.")

    outs = {}
    for label, pkg in (("predecessor", pred), ("e2", PKG)):
        dst = work / f"class_{label}.json"
        run([sys.executable, str(pkg / "run_h2_v12.py"),
             "--subject-repo", a.subject_repo, "--passa", str(passa_out),
             "--out", str(dst)])
        outs[label] = json.loads(dst.read_text())

    digests = {"passA.json": digest(passa_out),
               "run_h2_v12.py (E2)": digest(PKG / "run_h2_v12.py"),
               "run_h2_v12.py (pred)": digest(pred / "run_h2_v12.py"),
               "ontology.py (E2)": digest(PKG / "ontology.py"),
               "ontology.py (pred)": digest(pred / "ontology.py"),
               "passa.py (both)": digest(PKG / "passa.py"),
               "classify.py (both)": digest(PKG / "classify.py")}
    text = render(outs["predecessor"], outs["e2"], pa, digests, CHANGED)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(text)


if __name__ == "__main__":
    main()
