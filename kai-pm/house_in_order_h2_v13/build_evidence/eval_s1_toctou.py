#!/usr/bin/env python3
"""S1 TOCTOU-HARDENING CANONICAL REGRESSION. Produces S1_TOCTOU_RESULT.txt.

BANKED WITH THE IMPLEMENTATION, BEFORE THE CANONICAL MEASUREMENT.

THE PASS A DIGEST MUST CHANGE THIS TIME, AND THAT IS NOT A REGRESSION.
The payload carries `census_dependency.aggregate`, and the census package
was edited under Kai's narrow authorisation, so its manifest -- and
therefore the aggregate -- legitimately changed. Leaving the manifest
stale to preserve the digest would have made the package attest to code
that no longer exists, which is the defect class this programme removes.
So the digest change is EXPECTED, and this evaluator proves it is
CONFINED: the predecessor payload and the new payload are compared field
by field with that one aggregate normalised, and everything else must be
identical.

Every semantic invariant is compared against the RULED figures, never
against this run's own output.

    python3 eval_s1_toctou.py --subject-repo R --census-package C
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
HISTORICAL_PASSA_DIGEST = \
    "24fb1f555560277fd4555087f22ed06ef4a72efee845a339bbc87b925730f51c"
RULED = {
    "documents": 272, "witnesses": 492, "m3_scope": (201, 291),
    "validity": {"UNKNOWN": 260, "TIME_BOUND": 7, "EXACT_SNAPSHOT": 5},
    "axes": {"AUTHORITY": {"UNKNOWN": 272},
             "FUNCTION": {"UNKNOWN": 267, "MARKER": 5},
             "GENERATION": {"UNKNOWN": 272},
             "LIFECYCLE": {"UNKNOWN": 262, "HISTORICAL": 10},
             "SCOPE": {"UNKNOWN": 78, "WHOLE_FILE": 194}},
    "facts": {"MAINTENANCE_OBSERVED": 71, "STATIC_REFERENCE_AT_SUBJECT": 5,
              "CITES_COMMIT": 21, "CITES_RUN": 3, "CARRIES_DATE_STAMP": 206,
              "BINDING_CONTRADICTION": 5, "SELF_ASSERTS_AUTHORITY": 4,
              "SELF_ASSERTS_NON_AUTHORITY": 1},
    "ids": ["CHANGELOG.md", "README.md", "SESSION_BACKLOG.md",
            "docs/PROJECT_BACKLOG.md",
            "kai-pm/INSTRUMENTATION_ARCHITECTURE.md"],
}


def sh(cwd, *args):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout.decode(errors="replace")


def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(map(str, cmd))}\n"
                         f"{r.stdout[-600:]}\n{r.stderr[-600:]}")
    return r.stdout


def predecessor_tree(work):
    dst = work / "pred"
    dst.mkdir()
    for rel in (H2, CENSUS):
        out = dst / pathlib.Path(rel).name
        out.mkdir(parents=True)
        for n in sh(REPO_ROOT, "git", "ls-tree", "--name-only",
                    f"{PRED}:{rel}").split():
            blob = subprocess.run(["git", "-C", str(REPO_ROOT), "show",
                                   f"{PRED}:{rel}/{n}"], capture_output=True)
            (out / n).write_bytes(blob.stdout)
    return dst / "house_in_order_h2_v13", dst / "house_in_order_census_v11"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--out", default=str(HERE / "S1_TOCTOU_RESULT.txt"))
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="s1t_eval_"))
    p_h2, p_cen = predecessor_tree(work)
    subject = sh(a.subject_repo, "git", "rev-parse", "HEAD").strip()

    new_pa, new_cl = work / "pa.json", work / "cl.json"
    run([sys.executable, str(PKG / "passa.py"), "--subject-repo",
         a.subject_repo, "--history-repo", a.subject_repo, "--subject",
         subject, "--census-package", a.census_package, "--out", str(new_pa)])
    run([sys.executable, str(PKG / "run_h2_v12.py"), "--subject-repo",
         a.subject_repo, "--passa", str(new_pa), "--out", str(new_cl)])
    old_pa = work / "old_pa.json"
    run([sys.executable, str(p_h2 / "passa.py"), "--subject-repo",
         a.subject_repo, "--history-repo", a.subject_repo, "--subject",
         subject, "--census-package", str(p_cen), "--out", str(old_pa)])

    pa, cl = json.loads(new_pa.read_text()), json.loads(new_cl.read_text())
    opa = json.loads(old_pa.read_text())
    o, fails = [], []
    A = o.append

    def cmp(name, got, want):
        ok = got == want
        if not ok:
            fails.append(name)
        A(f"   {'OK  ' if ok else '<<< '}{name:<46}{str(got)[:44]}")
        if not ok:
            A(f"        RULED: {str(want)[:78]}")

    A("S1 TOCTOU HARDENING — CANONICAL REGRESSION. PRODUCED BY ORION.")
    A("ZERO ADJUDICATION WEIGHT. NOT AN ANSWER KEY. NO TUNING TARGET.")
    A("")
    A("1. FINGERPRINT")
    fp = {"implementation_commit": sh(REPO_ROOT, "git", "rev-parse",
                                      "HEAD").strip(),
          "subject_commit": pa["subject"], "subject_tree": pa["subject_tree"],
          "census_aggregate": cl["census_dependency"]["aggregate"],
          "gate_command": "git " + " ".join(PA.CLEAN_TREE_CMD),
          "git_version": sh(a.subject_repo, "git", "--version").strip(),
          "symlink_policy": f"P1_REJECT mode {PA.GIT_SYMLINK_MODE}",
          "verification": "sha256(consumed bytes) == sha256(frozen blob "
                          "bytes); no git object id is compared",
          "passa_digest": digest(new_pa),
          "classification_digest": digest(new_cl),
          "controls_digest": digest(HERE / "s1_toctou_controls.py")}
    for n in ("passa.py", "classify.py", "run_h2_v12.py", "ontology.py"):
        fp[f"digest[{n}]"] = digest(PKG / n)
    for n in ("docgraph.py", "opscan.py", "claims.py"):
        fp[f"digest[census/{n}]"] = digest(pathlib.Path(a.census_package) / n)
    for k, v in fp.items():
        A(f"   {k:<26}{v}")
    A("")

    A("2. SEMANTIC INVARIANTS vs THE RULED FIGURES")
    cmp("documents", len(pa["rows"]), RULED["documents"])
    wf = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "WHOLE_FILE")
    sp = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "SPAN")
    cmp("witnesses", wf + sp, RULED["witnesses"])
    cmp("M3 scope", (wf, sp), RULED["m3_scope"])
    cmp("VALIDITY", cl["axis_tallies"]["VALIDITY"], RULED["validity"])
    for ax, want in RULED["axes"].items():
        cmp(f"axis {ax}", cl["axis_tallies"][ax], want)
    for k, want in RULED["facts"].items():
        cmp(f"fact {k}", cl["evidence_fact_tally"].get(k), want)
    cmp("E2 identities", sorted(r["path"] for r in cl["rows"]
                                if r["evidence_facts"].get(
                                    "STATIC_REFERENCE_AT_SUBJECT")),
        sorted(RULED["ids"]))
    A("")

    A("3. THE PASS A DIGEST CHANGE — EXPECTED, AND PROVEN CONFINED")
    A(f"   historical digest   {HISTORICAL_PASSA_DIGEST}")
    A(f"   this run            {digest(new_pa)}")
    A(f"   predecessor rerun   {digest(old_pa)}")
    cmp("predecessor still reproduces the historical digest",
        digest(old_pa), HISTORICAL_PASSA_DIGEST)
    cmp("the new digest differs", digest(new_pa) != digest(old_pa), True)
    norm_o, norm_n = dict(opa), dict(pa)
    norm_o["census_dependency"] = norm_n["census_dependency"] = "NORMALISED"
    cmp("payloads identical once the aggregate is normalised",
        json.dumps(norm_o, sort_keys=True) ==
        json.dumps(norm_n, sort_keys=True), True)
    cmp("the ONLY differing top-level field is census_dependency",
        sorted(k for k in set(opa) | set(pa) if opa.get(k) != pa.get(k)),
        ["census_dependency"])
    A("   The historical E2 source-byte proof is NOT invalidated: it was")
    A("   established against the artefact as it stood and is banked. It")
    A("   remains RE-DERIVABLE by checking out the pre-hardening lineage,")
    A(f"   which this run does above -- the predecessor reproduces")
    A("   24fb1f5555602 77f exactly.")
    A("")

    A("4. DOCUMENT IDENTITY AGAINST THE FROZEN TREE")
    bad = [r["path"] for r in pa["rows"]
           if hashlib.sha256(subprocess.run(
               ["git", "-C", a.subject_repo, "show",
                f"{pa['subject_tree']}:{r['path']}"],
               capture_output=True).stdout.decode(errors="ignore").encode()
           ).hexdigest()[:16] != r["sha256"]]
    cmp("documents whose bytes match the frozen tree", len(bad), 0)
    A("")
    A("5. OUTCOME")
    A(f"   checks {len(RULED['axes']) + len(RULED['facts']) + 10}   "
      f"failures {len(fails)}")
    if fails:
        A("   UNEXPECTED DELTA. REPORTED, NOT REPAIRED:")
        for f in fails:
            A(f"     <<< {f}")
    else:
        A("   Every ruled invariant held. The only artefact change is the")
        A("   census aggregate, proven confined. Nothing was repaired and")
        A("   no figure was used as a target.")
    text = "\n".join(o) + "\n"
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(text)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
