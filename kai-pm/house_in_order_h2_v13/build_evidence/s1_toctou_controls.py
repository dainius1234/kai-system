#!/usr/bin/env python3
"""S1 TOCTOU HARDENING CONTROLS — consumption-time source verification.

Banked WITH the implementation, before any canonical measurement.

THE PREDECESSOR IS EXECUTED. The whole package -- H2 and census -- is
restored from ba258b8 into a temporary tree and run as a subprocess, so
the old code can never see the new constants.

TIMING IS DETERMINISTIC, NOT LUCKY. Mutations fire from a read hook keyed
on a counted trigger read, which is the instant a concurrent writer would
have to hit. The hook changes only WHEN the external write lands, never
what the analyser does. A race that fires on scheduling luck would prove
nothing either way.

THE THREE CONSUMPTION PATHS ARE TARGETED SEPARATELY, because a control
that only exercises the first one leaves the other two unproven and they
are in different modules.

    python3 s1_toctou_controls.py --subject-repo R --census-package C
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
REPO_ROOT = PKG.parent.parent
sys.path.insert(0, str(PKG))
import passa as PA                                             # noqa: E402

PRED = "ba258b8"
H2 = "kai-pm/house_in_order_h2_v13"
CENSUS = "kai-pm/house_in_order_census_v11"
INJECT_MD = "# Beta\n\n**Last updated:** 2099-11-22\n"
INJECT_PY = 'import pathlib\nDOC = pathlib.Path("docs/gamma.md")\n'
FAILED = []


def check(name, got, want, extra=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'OK  ' if ok else '<<< '}{name:<62}{str(got)[:36]:<38}"
          f"{'' if ok else 'expected ' + str(want)[:30]}")
    if extra:
        print(f"        {extra}")
    return ok


def sh(cwd, *args, ok=True):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if ok and r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout.decode(errors="replace")


def predecessor_tree(work):
    """H2 AND census as they were at ba258b8 -- both matter, because the
    census read boundary is part of what changed."""
    dst = work / "pred"
    dst.mkdir()
    for rel in (H2, CENSUS):
        out = dst / pathlib.Path(rel).name
        out.mkdir(parents=True)
        names = sh(REPO_ROOT, "git", "ls-tree", "--name-only",
                   f"{PRED}:{rel}").split()
        for n in names:
            if not n.endswith(".py"):
                continue
            blob = subprocess.run(["git", "-C", str(REPO_ROOT), "show",
                                   f"{PRED}:{rel}/{n}"], capture_output=True)
            (out / n).write_bytes(blob.stdout)
    return dst / "house_in_order_h2_v13", dst / "house_in_order_census_v11"


DOC = "# Alpha\n\n**Last updated:** 2026-01-01\n"
DOC2 = "# Beta\n\n**Last updated:** 2026-01-02\n"
DOC3 = "# Gamma\n\n**Last updated:** 2026-01-03\n"
READER = ('import pathlib\nDOC = pathlib.Path("docs/alpha.md")\n'
          'def go():\n    return DOC.read_text()\n')


def base(work, name):
    root = work / name
    (root / "docs").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "docs" / "alpha.md").write_text(DOC)
    (root / "docs" / "beta.md").write_text(DOC2)
    (root / "docs" / "gamma.md").write_text(DOC3)
    (root / "scripts" / "r.py").write_text(READER)
    sh(root, "git", "init", "-q", ".")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=t@x", "-c", "user.name=t",
       "commit", "-qm", "s")
    return root, sh(root, "git", "rev-parse", "HEAD").strip()


# argv: pkg census root subject trigger n target payload restore
PROBE = r"""
import hashlib, json, pathlib, sys
pkg, census, root, subject = sys.argv[1:5]
trigger, n_trigger, target, payload, restore = sys.argv[5:10]
sys.path.insert(0, pkg)
tgt = pathlib.Path(root) / target
original = tgt.read_bytes()
seen = {"n": 0}
state = {"injected": False, "restored": False}
log = []

def note(self, data):
    caller = sys._getframe(2).f_globals.get("__name__", "?")
    log.append({"reader": caller, "file": self.name,
                "sha": hashlib.sha256(data).hexdigest()[:16]})
    if self.name == trigger:
        seen["n"] += 1
        if seen["n"] == int(n_trigger) and not state["injected"]:
            tgt.write_bytes(payload.encode())
            state["injected"] = True
    elif state["injected"] and not state["restored"] \
            and self.name == pathlib.Path(target).name and restore == "yes":
        tgt.write_bytes(original)
        state["restored"] = True

rt, rb = pathlib.Path.read_text, pathlib.Path.read_bytes
def h_text(self, *a, **k):
    out = rt(self, *a, **k)
    note(self, out.encode())
    return out
def h_bytes(self, *a, **k):
    out = rb(self, *a, **k)
    note(self, out)
    return out
pathlib.Path.read_text, pathlib.Path.read_bytes = h_text, h_bytes

import passa as PA
res = {"raised": None, "rows": 0}
try:
    rows, tracked = PA.build(root, root, subject, pathlib.Path(census))
    res["rows"] = len(rows)
    res["shas"] = {r["path"]: r["sha256"] for r in rows}
except SystemExit as e:
    res["raised"] = str(e)
except Exception as e:
    res["raised"] = f"{type(e).__name__}: {e}"
pathlib.Path.read_text, pathlib.Path.read_bytes = rt, rb
res["injected"] = state["injected"]
res["restored"] = state["restored"]
res["log"] = log
res["worktree_dirty_after"] = bool(__import__("subprocess").run(
    ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all"],
    capture_output=True).stdout.strip())
json.dump(res, sys.stdout)
"""


def probe(pkg, census, root, subject, trigger, n, target, payload, restore):
    r = subprocess.run([sys.executable, "-c", PROBE, str(pkg), str(census),
                        str(root), subject, trigger, str(n), target, payload,
                        "yes" if restore else "no"],
                       capture_output=True, text=True)
    if not r.stdout:
        raise SystemExit(f"R11 ABORT: probe produced nothing: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def consumed_foreign(res, target_name, payload):
    """Did any reader consume the injected bytes?"""
    want = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return [e for e in res["log"]
            if e["file"] == target_name and e["sha"] == want]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--census-package", required=True)
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="s1h_"))
    p_h2, p_cen = predecessor_tree(work)
    print(f"PREDECESSOR   {PRED}: H2 and census restored from git objects")
    print(f"CANDIDATE     working tree")
    print(f"VERIFIER      sha256(consumed bytes) vs sha256(frozen blob bytes)")
    print(f"              NOT a git object id comparison")

    # 1 ────────────────────────────────────────────────────────────────
    print("\n1. FAIL-OLD against the accepted predecessor ba258b8")
    root, subj = base(work, "old_a")
    old_a = probe(p_h2, p_cen, root, subj, "alpha.md", 1, "docs/beta.md",
                  INJECT_MD, restore=False)
    check("A  mutation fired", old_a["injected"], True)
    check("A  predecessor COMPLETED the measurement",
          old_a["raised"] is None, True, f"{old_a['rows']} rows")
    check("A  foreign bytes were consumed",
          len(consumed_foreign(old_a, "beta.md", INJECT_MD)) > 0, True,
          f"readers: {[e['reader'] for e in consumed_foreign(old_a, 'beta.md', INJECT_MD)]}")
    root, subj = base(work, "old_b")
    old_b = probe(p_h2, p_cen, root, subj, "alpha.md", 1, "docs/beta.md",
                  INJECT_MD, restore=True)
    check("B  change-and-restore fired", (old_b["injected"],
                                          old_b["restored"]), (True, True))
    check("B  predecessor COMPLETED the measurement",
          old_b["raised"] is None, True)
    check("B  foreign bytes consumed anyway",
          len(consumed_foreign(old_b, "beta.md", INJECT_MD)) > 0, True)
    check("B  worktree CLEAN afterwards, so a post-check sees nothing",
          old_b["worktree_dirty_after"], False)

    # 2 ────────────────────────────────────────────────────────────────
    print("\n2. PASS-NEW — each consumption path targeted separately")
    cen = a.census_package
    for label, trigger, n, target, payload, restore in (
            ("docgraph document read", "alpha.md", 1, "docs/beta.md",
             INJECT_MD, False),
            ("passa document read", "beta.md", 1, "docs/beta.md",
             INJECT_MD, False),
            ("opscan source read", "alpha.md", 1, "scripts/r.py",
             INJECT_PY, False),
            ("docgraph, change-and-restore", "alpha.md", 1, "docs/beta.md",
             INJECT_MD, True),
            ("passa, change-and-restore", "beta.md", 1, "docs/beta.md",
             INJECT_MD, True),
            ("opscan, change-and-restore", "alpha.md", 1, "scripts/r.py",
             INJECT_PY, True)):
        root, subj = base(work, f"new_{label.replace(' ', '_').replace(',', '')}")
        res = probe(PKG, cen, root, subj, trigger, n, target, payload, restore)
        name = pathlib.Path(target).name
        check(f"{label:<32} mutation fired", res["injected"], True)
        check(f"{label:<32} REFUSED", res["raised"] is not None, True,
              str(res["raised"])[:118])
        check(f"{label:<32} names the diverging path",
              target in str(res["raised"]), True)
        check(f"{label:<32} no rows produced", res["rows"], 0)
        check(f"{label:<32} names CONSUMED BYTES DIVERGE",
              "CONSUMED BYTES DIVERGE" in str(res["raised"]), True)

    # 3 ────────────────────────────────────────────────────────────────
    print("\n3. CAN-FAIL — the verifier must FIRE, not merely pass")
    root, subj = base(work, "canfail")
    clean = probe(PKG, cen, root, subj, "no_such_file", 99, "docs/beta.md",
                  INJECT_MD, restore=False)
    check("an unmutated run of the SAME probe completes",
          clean["raised"] is None, True, f"{clean['rows']} rows")
    check("  so the refusals above are caused by the mutation, not by the "
          "probe", clean["injected"], False)

    # 4 ────────────────────────────────────────────────────────────────
    print("\n4. SEMANTIC NON-INTERFERENCE — census behaviour with no verifier")
    sys.path.insert(0, str(p_cen))
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    oldG = load(p_cen / "docgraph.py", "oldG")
    oldO = load(p_cen / "opscan.py", "oldO")
    oldC = load(p_cen / "claims.py", "oldC")
    sys.path.insert(0, cen)
    newG = load(pathlib.Path(cen) / "docgraph.py", "newG")
    newO = load(pathlib.Path(cen) / "opscan.py", "newO")
    newC = load(pathlib.Path(cen) / "claims.py", "newC")
    sr = a.subject_repo
    od, nd = oldG.tracked_md(sr), newG.tracked_md(sr)
    check("tracked_md population identical", od, nd, f"{len(nd)} documents")
    check("source_population identical",
          oldO.source_population(pathlib.Path(sr)),
          newO.source_population(pathlib.Path(sr)),
          f"{len(newO.source_population(pathlib.Path(sr)))} source files")
    oe = oldG.build_graph(sr, od)
    ne = newG.build_graph(sr, nd)                     # no verifier passed
    check("docgraph edges identical with no verifier", oe, ne,
          f"{len(ne)} edges")
    oo, oacc = oldO.collect(pathlib.Path(sr), od)
    no, nacc = newO.collect(pathlib.Path(sr), nd)     # no verifier passed
    check("opscan admission accounting identical", oacc, nacc)
    oldC.classify(oo, od, set(oldO.tracked(sr)))
    newC.classify(no, nd, set(newO.tracked(sr)))
    check("opscan/claims dispositions identical",
          sorted((o.src, o.line, o.mode, o.disposition, o.target)
                 for o in oo),
          sorted((o.src, o.line, o.mode, o.disposition, o.target)
                 for o in no), f"{len(no)} operations")
    check("reader/writer semantics identical",
          sorted((o.src, o.target) for o in oo
                 if o.disposition == "RESOLVED_READ"),
          sorted((o.src, o.target) for o in no
                 if o.disposition == "RESOLVED_READ"))
    check("claims.py byte-identical (untouched)",
          hashlib.sha256((pathlib.Path(cen) / "claims.py").read_bytes()
                         ).hexdigest(),
          hashlib.sha256((p_cen / "claims.py").read_bytes()).hexdigest())
    for n in ("classify.py", "envelope.py", "ontology.py", "subjectbind.py",
              "run_h2_v12.py", "holdout.py", "qualify.py"):
        check(f"H2 {n:<18} byte-identical",
              hashlib.sha256((PKG / n).read_bytes()).hexdigest(),
              hashlib.sha256((p_h2 / n).read_bytes()).hexdigest())

    # 5 ────────────────────────────────────────────────────────────────
    print("\n5. READER-POPULATION GUARD")
    tracked = set(sh(sr, "git", "ls-tree", "-r", "--name-only",
                     "HEAD").split())
    pop = set(nd) | set(newO.source_population(pathlib.Path(sr)))
    check("every consumed path is inside the tracked frozen population",
          pop <= tracked, True, f"{len(pop)} of {len(tracked)} tracked paths")
    root, subj = base(work, "guardrun")
    g = probe(PKG, cen, root, subj, "no_such_file", 99, "docs/beta.md",
              INJECT_MD, restore=False)
    readers = sorted({e["reader"] for e in g["log"]})
    old_readers = sorted({e["reader"] for e in old_a["log"]})
    check("BEFORE: bytes were consumed by THREE modules directly",
          old_readers, ["docgraph", "opscan", "passa"],
          "the predecessor's three own read sites, each unverified. My "
          "first expectation here named only two because I had read the "
          "beta.md-specific list rather than the whole reader set")
    check("AFTER: every consumption funnels through the ONE verified reader",
          readers, ["passa"],
          "docgraph and opscan now obtain bytes through read_source, which "
          "lives in passa.py and cannot return unverified bytes")
    check("  so ANY other module appearing here is an uncovered new reader",
          [r for r in readers if r != "passa"], [],
          "that is the guard: a future filesystem reader outside the "
          "verified path shows up as a foreign name and fails this control")

    # 6 ────────────────────────────────────────────────────────────────
    print("\n6. CENSUS PACKAGE INTEGRITY")
    manifest = pathlib.Path(cen) / "MANIFEST.sha256"
    bad = [ln.split("  ", 1)[1] for ln in manifest.read_text().splitlines()
           if ln.strip() and hashlib.sha256(
               (pathlib.Path(cen) / ln.split("  ", 1)[1]).read_bytes()
           ).hexdigest() != ln.split("  ", 1)[0]]
    check("MANIFEST.sha256 matches every file", bad, [])
    check("aggregate CHANGED, as it must after an authorised edit",
          hashlib.sha256(manifest.read_bytes()).hexdigest()
          == "eb7aad7c1a565cb25fcf6a7e250133e95d210f3e8ceb8765489046e3d945fa0e",
          False,
          f"new aggregate {hashlib.sha256(manifest.read_bytes()).hexdigest()}")

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nCONTROLS: "
          f"{'ALL PASS' if not FAILED else 'FAILED — ' + '; '.join(FAILED)}")
    raise SystemExit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
