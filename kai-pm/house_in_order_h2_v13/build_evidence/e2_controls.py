#!/usr/bin/env python3
"""E2 CONTROLS. Banked WITH the implementation, before any corpus run.

THE PREDECESSOR IS EXECUTED, NOT PARAPHRASED. The package is copied to a
temporary directory and BOTH files E2 touched -- run_h2_v12.py and
ontology.py -- are restored from the git object. Every other module is
asserted byte-identical, so a difference can only come from E2.

THE PREDECESSOR RUNS IN A SUBPROCESS. Loading two versions of the same
module into one interpreter would let sys.modules hand the old runner the
NEW ontology, and the control would compare a chimera. The subprocess has
its own module table.

CONTROLS ARE UNIT-LEVEL, ON NAMED ROWS. `evidence_facts` is called
directly. No corpus aggregate is computed here: the single corpus
evaluation happens after this implementation is banked.

SECTION 6 IS A CONSTRUCTED KNOWN-POSITIVE / KNOWN-NEGATIVE PAIR (I-8).
A throwaway git repository is built with two documents and two readers --
one referencing its document by a fixed path, one by a path the analysis
cannot resolve. The second document IS read and the fact is still False.
That is what makes "False is an abstention, not a negative" a
demonstration rather than a sentence in a docstring.

    python3 e2_controls.py --subject-repo R --tree T --census-package C
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
import ontology as ont                                         # noqa: E402
import passa as PA                                             # noqa: E402
import run_h2_v12 as R                                         # noqa: E402
import subjectbind as sb                                       # noqa: E402

E2_PREDECESSOR = "a766493"
OLD_FACT = "CONSUMED_AT_SUBJECT"
NEW_FACT = "STATIC_REFERENCE_AT_SUBJECT"
CHANGED = ("run_h2_v12.py", "ontology.py")
UNCHANGED = ("passa.py", "classify.py", "envelope.py", "subjectbind.py",
             "qualify.py", "holdout.py")
PKG_PATH = "kai-pm/house_in_order_h2_v13"
FAILED = []


def check(name, got, want, extra=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'OK  ' if ok else '<<< '}{name:<64}{str(got)[:40]:<42}"
          f"{'' if ok else 'expected ' + str(want)[:40]}")
    if extra:
        print(f"        {extra}")
    return ok


def _git(repo, *args):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
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
            raise SystemExit(f"R11 ABORT: {name} differs between the trees. "
                             f"Only {CHANGED} may differ.")
    return dst


PROBE = r"""
import json, sys, pathlib
sys.path.insert(0, sys.argv[1])
import run_h2_v12 as R, subjectbind as sb
row = json.load(sys.stdin)
text = (pathlib.Path(sys.argv[2]) / row["path"]).read_text(errors="ignore")
claims, _ = sb.bind_claims(row["path"], text)
f, _ac, tr, ab = R.evidence_facts(row, claims, None,
                                  sb.determining_claims(claims),
                                  sys.argv[3], sys.argv[2])
json.dump({"facts": f, "traces": tr, "abstained": ab,
           "trace_class_keys": sorted(R.TRACE_CLASS)}, sys.stdout)
"""


def probe(pkg, row, subject_repo, subject):
    r = subprocess.run([sys.executable, "-c", PROBE, str(pkg), subject_repo,
                        subject], input=json.dumps(row), capture_output=True,
                       text=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: probe failed in {pkg}\n{r.stderr[-600:]}")
    return json.loads(r.stdout)


def here(row, subject_repo, subject):
    text = (pathlib.Path(subject_repo) / row["path"]).read_text(errors="ignore")
    claims, _ = sb.bind_claims(row["path"], text)
    f, _ac, tr, ab = R.evidence_facts(row, claims, None,
                                      sb.determining_claims(claims),
                                      subject, subject_repo)
    return {"facts": f, "traces": tr, "abstained": ab}


# ── section 6: the constructed abstention demonstration ───────────────
FIXED_READER = '''import pathlib
DOC = pathlib.Path("fixed.md")
def go():
    return DOC.read_text()
'''
DYNAMIC_READER = '''import os, pathlib
def go():
    root = pathlib.Path(os.environ["SOME_ROOT"])
    return (root / "dynamic.md").read_text()
'''


def abstention_demo(census):
    """A document that IS referenced, whose fact is still False.

    Built as a real repository and put through the real census, so the
    dispositions are the analysis's own answer and not my assertion.
    """
    sys.path.insert(0, str(census))
    import claims as C, docgraph as G, opscan as O          # noqa: E402
    work = pathlib.Path(tempfile.mkdtemp(prefix="e2_demo_"))
    (work / "fixed.md").write_text("# fixed\n")
    (work / "dynamic.md").write_text("# dynamic\n")
    (work / "fixed_reader.py").write_text(FIXED_READER)
    (work / "dynamic_reader.py").write_text(DYNAMIC_READER)
    for cmd in (["init", "-q"], ["add", "-A"],
                ["-c", "user.email=e@x", "-c", "user.name=e",
                 "commit", "-qm", "demo"]):
        subprocess.run(["git", "-C", str(work), *cmd], check=True,
                       capture_output=True)
    docs = G.tracked_md(str(work))
    ops, _acc = O.collect(work, docs)
    C.classify(ops, docs, set(O.tracked(str(work))))
    by_doc = {}
    for o in ops:
        by_doc.setdefault(o.target or "(unresolved)", []).append(o.disposition)
    readers = {d: sorted({o.src for o in ops
                          if o.target == d and o.disposition == "RESOLVED_READ"})
               for d in docs}
    shutil.rmtree(work, ignore_errors=True)
    return docs, by_doc, readers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--old-ref", default=E2_PREDECESSOR)
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="e2_ctl_"))
    pred = predecessor_pkg(a.repo, a.old_ref, work)
    subject = _git(a.subject_repo, "rev-parse", "HEAD").decode().strip()

    print(f"PREDECESSOR   {a.old_ref}, run_h2_v12.py + ontology.py from the "
          f"git object, EXECUTED in a subprocess")
    print(f"CANDIDATE     working tree {PKG.name}")
    print(f"UNCHANGED     {', '.join(UNCHANGED)} asserted byte-identical")

    # rows for the five source-confirmed positives, built by frozen Pass A
    rows = {}
    for p in ("CHANGELOG.md", "README.md", "SESSION_BACKLOG.md",
              "docs/PROJECT_BACKLOG.md",
              "kai-pm/INSTRUMENTATION_ARCHITECTURE.md"):
        txt = _git(a.subject_repo, "show", f"{a.tree}:{p}").decode()
        w = PA.scan(p, txt, a.subject_repo, subject)
        # readers/reader_ops come from the census, so take them from a
        # real Pass A build rather than inventing them
        rows[p] = {"path": p, "witnesses": {k: [x.asdict() for x in v]
                                            for k, v in w.items()},
                   "commits_in_window": 0, "last": "", "readers": [],
                   "reader_ops": []}

    print("\n0. THE FIVE POSITIVE ROWS COME FROM A REAL PASS A BUILD")
    pa_rows, _tracked = PA.build(a.subject_repo, a.subject_repo, subject,
                                 pathlib.Path(a.census_package))
    byp = {r["path"]: r for r in pa_rows}
    positives = sorted(p for p, r in byp.items() if r["readers"])
    check("rows carrying a static reader reference", len(positives), 5,
          f"{positives}")

    # 1 ────────────────────────────────────────────────────────────────
    print("\n1. FAIL-OLD / PASS-NEW on every one of the five")
    for p in positives:
        old = probe(pred, byp[p], a.subject_repo, subject)
        new = here(byp[p], a.subject_repo, subject)
        n = p.split("/")[-1]
        check(f"{n:<40} predecessor emits OLD",
              old["facts"].get(OLD_FACT), True)
        check(f"{n:<40} predecessor has no NEW key",
              NEW_FACT in old["facts"], False)
        check(f"{n:<40} candidate emits NEW",
              new["facts"].get(NEW_FACT), True)
        check(f"{n:<40} candidate has no OLD key",
              OLD_FACT in new["facts"], False)

    # 2 ────────────────────────────────────────────────────────────────
    print("\n2. TRACE IDENTITY — the evidence must not move, only the name")
    for p in positives:
        old = probe(pred, byp[p], a.subject_repo, subject)["traces"][OLD_FACT]
        new = here(byp[p], a.subject_repo, subject)["traces"][NEW_FACT]
        check(f"{p.split('/')[-1]:<40} trace byte-identical", new, old,
              f"witness_type={new['witness_type']} "
              f"selector={new['source_selector']} "
              f"total={new['evidence_total']} shown={new['evidence_shown']} "
              f"truncated={new['truncated']}")

    # 3 ────────────────────────────────────────────────────────────────
    print("\n3. TRACE_CLASS BINDING, and the CAN-FAIL proof")
    check("TRACE_CLASS carries the NEW key", NEW_FACT in R.TRACE_CLASS, True,
          f"{R.TRACE_CLASS.get(NEW_FACT)}")
    check("TRACE_CLASS no longer carries the OLD key",
          OLD_FACT in R.TRACE_CLASS, False)
    check("EVIDENCE_FACTS carries the NEW name",
          NEW_FACT in ont.EVIDENCE_FACTS, True)
    check("EVIDENCE_FACTS no longer carries the OLD name",
          OLD_FACT in ont.EVIDENCE_FACTS, False)
    row = byp[positives[0]]
    saved = dict(R.TRACE_CLASS)
    R.TRACE_CLASS.pop(NEW_FACT)
    try:
        broken = here(row, a.subject_repo, subject)
    finally:
        R.TRACE_CLASS.clear()
        R.TRACE_CLASS.update(saved)
    check("deliberate TRACE_CLASS mismatch -> A6-ii abstention",
          (broken["facts"][NEW_FACT], broken["abstained"]),
          (False, [NEW_FACT]),
          "it does NOT raise; it abstains, which is why this proof exists")
    R.TRACE_CLASS[NEW_FACT] = ("WRONG_WITNESS_TYPE", "opscan:")
    try:
        wrong = here(row, a.subject_repo, subject)
    finally:
        R.TRACE_CLASS.clear()
        R.TRACE_CLASS.update(saved)
    check("wrong witness_type in TRACE_CLASS -> A6-ii abstention",
          (wrong["facts"][NEW_FACT], wrong["abstained"]),
          (False, [NEW_FACT]))
    check("restored", here(row, a.subject_repo, subject)["facts"][NEW_FACT],
          True)

    # 4 ────────────────────────────────────────────────────────────────
    print("\n4. NO RUNTIME OR REACHABILITY MEANING SURVIVES IN THE OUTPUT")
    tr = here(row, a.subject_repo, subject)["traces"][NEW_FACT]
    blob = json.dumps({"facts": here(row, a.subject_repo, subject)["facts"],
                       "trace": tr})
    for word in ("CONSUMED", "consumed", "executable", "executed", "runtime",
                 "reachab", "invoc"):
        check(f"emitted record contains {word!r}", word in blob, False)
    check("witness_type names the evidence class", tr["witness_type"],
          "STATIC_READER_REFERENCE")
    check("selector names the producing instrument",
          tr["source_selector"].startswith("opscan:"), True)
    check("the fact name itself contains no runtime verb",
          any(v in NEW_FACT.lower() for v in ("consum", "exec", "run", "invoke",
                                              "reach")), False)

    # 5 ────────────────────────────────────────────────────────────────
    print("\n5. THE ANALYSIS SCOPE IS PINNED BY EXISTING PROVENANCE")
    C = pathlib.Path(a.census_package)
    man = {l.split(None, 1)[1].strip().lstrip("*"): l.split(None, 1)[0]
           for l in (C / "MANIFEST.sha256").read_text().splitlines()
           if l.strip()}
    check("opscan.py is listed in MANIFEST.sha256", "opscan.py" in man, True)
    check("its listed hash matches the file on disk",
          man.get("opscan.py"), digest(C / "opscan.py"))
    src = (C / "opscan.py").read_text()
    for const in ("EXCLUDE_DIRS", "SRC_SUFFIX", "def source_population"):
        check(f"{const} is defined inside that pinned file", const in src, True)
    check("no new schema field was added to the trace",
          sorted(tr), sorted(probe(pred, row, a.subject_repo,
                                   subject)["traces"][OLD_FACT]))

    # 6 ────────────────────────────────────────────────────────────────
    print("\n6. FALSE IS AN ABSTENTION — CONSTRUCTED DEMONSTRATION")
    docs, by_doc, readers = abstention_demo(a.census_package)
    check("demo documents built", sorted(docs), ["dynamic.md", "fixed.md"])
    check("fixed.md has a resolvable static reference",
          bool(readers.get("fixed.md")), True, f"{readers.get('fixed.md')}")
    check("dynamic.md has NO resolvable static reference",
          bool(readers.get("dynamic.md")), False,
          f"dispositions seen: {by_doc}")
    check("...yet dynamic.md IS read by dynamic_reader.py -- so a False "
          "fact cannot mean 'not read'",
          "dynamic.md" in DYNAMIC_READER, True)

    # 7 ────────────────────────────────────────────────────────────────
    print("\n7. NOTHING ELSE MAY HAVE MOVED")
    for name in UNCHANGED:
        if (PKG / name).exists():
            check(f"{name:<40} byte-identical to predecessor",
                  digest(PKG / name), digest(pred / name))
    old_all = probe(pred, row, a.subject_repo, subject)["facts"]
    new_all = here(row, a.subject_repo, subject)["facts"]
    check("every OTHER evidence fact unchanged",
          {k: v for k, v in new_all.items() if k != NEW_FACT},
          {k: v for k, v in old_all.items() if k != OLD_FACT})
    check("fact count unchanged", len(new_all), len(old_all))
    check("EVIDENCE_FACTS length unchanged", len(ont.EVIDENCE_FACTS), 10)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nCONTROLS: "
          f"{'ALL PASS' if not FAILED else 'FAILED — ' + '; '.join(FAILED)}")
    raise SystemExit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
