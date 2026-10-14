#!/usr/bin/env python3
"""THE M1 EVALUATOR. Produces STEP3_M1_RESULT.txt.

BANKED WITH THE IMPLEMENTATION, BEFORE THE CORPUS RUN. It is written
without having seen the repaired mechanism's corpus output, run ONCE
afterwards, and the implementation is not altered on the strength of what
it reports.

HOW OLD AND NEW ARE MADE COMPARABLE. The package is copied to a temporary
directory and `classify.py` there is REPLACED BY THE GIT OBJECT of the
predecessor. The full runner is then executed in BOTH trees against the
SAME Pass A input. Every other module -- passa.py, envelope.py,
ontology.py, subjectbind.py, run_h2_v12.py -- is byte-identical between
the two runs, so any difference in the output is attributable to
classify.py and to nothing else. The evaluator ASSERTS that identity
rather than assuming it.

Pass A is produced once and shared, because passa.py is untouched; the
evaluator verifies that too.

    python3 eval_m1.py --subject-repo R --tree T --census-package C \\
        --out F
"""
from __future__ import annotations
import argparse
import collections
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
import passa as PA                                             # noqa: E402
import classify as NEW                                         # noqa: E402

M1_PREDECESSOR = "b0da564"
CLASSIFY_PATH = "kai-pm/house_in_order_h2_v13/classify.py"
SHARED = ("passa.py", "envelope.py", "ontology.py", "subjectbind.py",
          "run_h2_v12.py")


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


def predecessor_tree(repo, ref, work):
    """A package identical to the candidate except for classify.py."""
    dst = work / "predecessor"
    dst.mkdir()
    for name in SHARED + ("classify.py",):
        shutil.copy2(PKG / name, dst / name)
    (dst / "classify.py").write_bytes(_git(repo, "show", f"{ref}:{CLASSIFY_PATH}"))
    for name in SHARED:
        if digest(dst / name) != digest(PKG / name):
            raise SystemExit(f"R11 ABORT: {name} differs between the two "
                             f"trees. Only classify.py may differ.")
    if digest(dst / "classify.py") == digest(PKG / "classify.py"):
        raise SystemExit("R11 ABORT: predecessor classify.py is identical to "
                         "the candidate. Nothing is being compared.")
    return dst


class Corpus:
    def __init__(self, repo, tree):
        self.repo, self.tree, self._c = repo, tree, {}

    def text(self, path):
        if path not in self._c:
            self._c[path] = _git(self.repo, "show",
                                 f"{self.tree}:{path}").decode("utf-8")
        return self._c[path]

    def raw_line(self, path, selector):
        try:
            n = int(str(selector).lstrip("L"))
        except ValueError:
            return "(no line selector)"
        lines = self.text(path).split("\n")
        return lines[n - 1] if 0 < n <= len(lines) else "(line out of range)"


def det_of(axis):
    w = axis.get("witness")
    if not w:
        return "—", None
    return f"{w['source_selector']} {w['witness_value']!r}", w


def render(old, new, corpus, pa, digests):
    o = []
    A = o.append
    orows = {r["path"]: r for r in old["rows"]}
    nrows = {r["path"]: r for r in new["rows"]}
    A("STEP 3 / M1 — RESULT. PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT.")
    A("NOT AN ANSWER KEY. NO TARGET COUNT WAS USED.")
    A(f"subject      {new['subject']}")
    A(f"subject tree {new['subject_tree']}")
    A(f"predecessor  {M1_PREDECESSOR} classify.py, from the git object")
    A("only classify.py differs between the two runs; asserted, not assumed")
    A("")

    A("1. CONSERVATION")
    A(f"   rows in {pa['population']} · predecessor rows {len(orows)} · "
      f"M1 rows {len(nrows)}")
    A(f"   identical path set                 {set(orows) == set(nrows)}")
    wf = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "WHOLE_FILE")
    sp = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "SPAN")
    A(f"   Pass A witnesses {wf + sp} · accepted M3 scope {wf}/{sp}"
      f"   (must read 201/291)")
    A("")

    A("2. FULL VALIDITY DISTRIBUTION")
    A(f"   predecessor  {old['axis_tallies']['VALIDITY']}")
    A(f"   M1           {new['axis_tallies']['VALIDITY']}")
    A("")

    A("3. EVERY OTHER AXIS — must be unchanged")
    for ax in sorted(old["axis_tallies"]):
        if ax == "VALIDITY":
            continue
        same = old["axis_tallies"][ax] == new["axis_tallies"][ax]
        A(f"   {ax:<12}{'UNCHANGED' if same else '<<< MOVED':<12}"
          f"{new['axis_tallies'][ax]}")
    A("")

    A("4. SCOPE COMPARISON, per row")
    moved = [p for p in nrows
             if orows[p]["SCOPE"]["value"] != nrows[p]["SCOPE"]["value"]]
    mw = [p for p in nrows
          if det_of(orows[p]["SCOPE"])[0] != det_of(nrows[p]["SCOPE"])[0]]
    A(f"   SCOPE verdicts changed            {len(moved)}   (must be 0)")
    A(f"   SCOPE determining witness changed {len(mw)}   (must be 0)")
    for p in moved + [x for x in mw if x not in moved]:
        A(f"     <<< {p}  {orows[p]['SCOPE']['value']} -> "
          f"{nrows[p]['SCOPE']['value']}")
    A("")

    A("5. EVIDENCE-FACT COMPARISON")
    ofacts, nfacts = old["evidence_fact_tally"], new["evidence_fact_tally"]
    for k in sorted(set(ofacts) | set(nfacts)):
        same = ofacts.get(k) == nfacts.get(k)
        A(f"   {k:<32}{ofacts.get(k, 0):>5} -> {nfacts.get(k, 0):<5}"
          f"{'' if same else '   <<< MOVED'}")
    A(f"   NOMINAL_FUNCTION tally unchanged: "
      f"{old['nominal_function_tally'] == new['nominal_function_tally']}")
    A("")

    changed = sorted(p for p in nrows
                     if orows[p]["VALIDITY"]["value"]
                     != nrows[p]["VALIDITY"]["value"])
    rescued = sorted(p for p in nrows
                     if orows[p]["VALIDITY"]["value"] == "TIME_BOUND"
                     and nrows[p]["VALIDITY"]["value"] == "TIME_BOUND"
                     and det_of(orows[p]["VALIDITY"])[0]
                     != det_of(nrows[p]["VALIDITY"])[0])
    A(f"6. CHANGED VALIDITY ROWS: {len(changed)}")
    A(f"   RESCUED — verdict held, determining witness moved to a "
      f"qualified one: {len(rescued)}")
    A("")
    A("7. EVERY RESCUED ROW, with both determining witnesses")
    for p in rescued:
        ob, ow = det_of(orows[p]["VALIDITY"])
        nb, nw = det_of(nrows[p]["VALIDITY"])
        A(f"   {p}   {nrows[p]['VALIDITY']['value']}")
        A(f"     before  {ob}   {corpus.raw_line(p, ow['source_selector'])}")
        A(f"     after   {nb}   {corpus.raw_line(p, nw['source_selector'])}")
        A(f"     reason  {nrows[p]['VALIDITY']['rationale']}")
    A("")
    A(f"8. EVERY CHANGED ROW ({len(changed)}), with its raw source reason")
    by_pred = collections.Counter()
    for p in changed:
        ob, ow = det_of(orows[p]["VALIDITY"])
        line = corpus.raw_line(p, ow["source_selector"]) if ow else "(none)"
        pred = NEW._predicate_of(ow) if ow else None
        by_pred[pred] += 1
        A(f"   {p}")
        A(f"     {orows[p]['VALIDITY']['value']} -> "
          f"{nrows[p]['VALIDITY']['value']}")
        A(f"     determining before : {ob}")
        A(f"     determining after  : {det_of(nrows[p]['VALIDITY'])[0]}")
        A(f"     source predicate   : {pred!r}")
        A(f"     raw source line    : {line}")
        A(f"     new observation    : {nrows[p]['VALIDITY'].get('observed')}")
    A("")
    A("9. CHANGED ROWS BY SOURCE PREDICATE — the mechanism, not a literal")
    for pred, n in by_pred.most_common():
        A(f"   {pred!r:<26}{n:>5}")
    A("")
    A("10. INPUT DIGESTS")
    for k, v in digests.items():
        A(f"   {k:<28}{v}")
    A("")
    A("11. WHAT THIS IS NOT")
    A("   Not a correctness reference, not an answer key, not an")
    A("   adjudication. A delta is not automatically correct. No delta")
    A("   below has been repaired. No target count was used: neither the")
    A("   historical 10 nor 5, 7, 38 or any other aggregate.")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--old-ref", default=M1_PREDECESSOR)
    ap.add_argument("--out", default=str(HERE / "STEP3_M1_RESULT.txt"))
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="m1_eval_"))
    old_pkg = predecessor_tree(a.repo, a.old_ref, work)

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
    for label, pkg in (("predecessor", old_pkg), ("m1", PKG)):
        dst = work / f"class_{label}.json"
        run([sys.executable, str(pkg / "run_h2_v12.py"),
             "--subject-repo", a.subject_repo, "--passa", str(passa_out),
             "--out", str(dst)])
        outs[label] = json.loads(dst.read_text())

    digests = {"passA.json": digest(passa_out),
               "classify.py (M1)": digest(PKG / "classify.py"),
               "classify.py (pred)": digest(old_pkg / "classify.py"),
               "passa.py (both)": digest(PKG / "passa.py")}
    corpus = Corpus(a.subject_repo, a.tree)
    text = render(outs["predecessor"], outs["m1"], corpus, pa, digests)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
