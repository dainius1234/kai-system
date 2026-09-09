#!/usr/bin/env python3
"""THE S1 CANONICAL REGRESSION. Produces S1_RESULT.txt.

BANKED WITH THE IMPLEMENTATION, BEFORE THE CANONICAL MEASUREMENT. Written
without having seen any post-S1 corpus output, run ONCE afterwards, and
the implementation is not altered on the strength of what it reports.

WHAT IT ASSERTS. The regression invariants Kai listed are compared
against the RULED figures, not against whatever this run happens to
produce -- a self-comparison would agree with itself. Any unexpected
delta is reported and nothing is repaired.

IT ALSO RECORDS THE FINGERPRINT Kai required: implementation commit,
production-file digests, subject commit and tree, census aggregate and
code identity, the exact gate command and git version, the controls
digest and the result digest.

    python3 eval_s1.py --subject-repo R --tree T --census-package C
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import passa as PA                                             # noqa: E402

# THE RULED FIGURES. Stated here so the comparison is against the
# adjudicated record, never against this run's own output.
RULED = {
    "documents": 272,
    "witnesses": 492,
    "m3_scope": (201, 291),
    "validity": {"UNKNOWN": 260, "TIME_BOUND": 7, "EXACT_SNAPSHOT": 5},
    "axes": {"AUTHORITY": {"UNKNOWN": 272},
             "FUNCTION": {"UNKNOWN": 267, "MARKER": 5},
             "GENERATION": {"UNKNOWN": 272},
             "LIFECYCLE": {"UNKNOWN": 262, "HISTORICAL": 10},
             "SCOPE": {"UNKNOWN": 78, "WHOLE_FILE": 194}},
    "facts": {"MAINTENANCE_OBSERVED": 71, "STATIC_REFERENCE_AT_SUBJECT": 5,
              "CITES_COMMIT": 21, "CITES_RUN": 3,
              "CARRIES_DATE_STAMP": 206, "BINDING_CONTRADICTION": 5,
              "SELF_ASSERTS_AUTHORITY": 4, "SELF_ASSERTS_NON_AUTHORITY": 1},
    "static_reference_identities": [
        "CHANGELOG.md", "README.md", "SESSION_BACKLOG.md",
        "docs/PROJECT_BACKLOG.md", "kai-pm/INSTRUMENTATION_ARCHITECTURE.md"],
    "passa_digest":
        "24fb1f555560277fd4555087f22ed06ef4a72efee845a339bbc87b925730f51c",
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--out", default=str(HERE / "S1_RESULT.txt"))
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="s1_eval_"))
    subject = sh(a.subject_repo, "git", "rev-parse", "HEAD").strip()
    passa_out = work / "passA.json"
    run([sys.executable, str(PKG / "passa.py"),
         "--subject-repo", a.subject_repo, "--history-repo", a.subject_repo,
         "--subject", subject, "--census-package", a.census_package,
         "--out", str(passa_out)])
    cls_out = work / "classification.json"
    run([sys.executable, str(PKG / "run_h2_v12.py"),
         "--subject-repo", a.subject_repo, "--passa", str(passa_out),
         "--out", str(cls_out)])
    pa = json.loads(passa_out.read_text())
    cl = json.loads(cls_out.read_text())

    o, fails = [], []
    A = o.append

    def cmp(name, got, want):
        ok = got == want
        if not ok:
            fails.append(name)
        A(f"   {'OK  ' if ok else '<<< '}{name:<44}{str(got)[:46]}")
        if not ok:
            A(f"        RULED: {str(want)[:80]}")

    A("S1 — CANONICAL REGRESSION RESULT. PRODUCED BY ORION.")
    A("ZERO ADJUDICATION WEIGHT. NOT AN ANSWER KEY. NO TUNING TARGET.")
    A("Compared against the RULED figures, not against this run's output.")
    A("")
    A("1. FINGERPRINT")
    fp = {
        "implementation_commit": sh(a.repo, "git", "rev-parse", "HEAD").strip(),
        "subject_commit": pa["subject"], "subject_tree": pa["subject_tree"],
        "census_aggregate": cl["census_dependency"]["aggregate"],
        "gate_command": "git " + " ".join(PA.CLEAN_TREE_CMD),
        "git_version": sh(a.subject_repo, "git", "--version").strip(),
        "symlink_policy": f"P1_REJECT mode {PA.GIT_SYMLINK_MODE}",
        "passa_digest": digest(passa_out),
        "classification_digest": digest(cls_out),
        "controls_digest": digest(HERE / "s1_controls.py"),
    }
    for n in ("passa.py", "classify.py", "run_h2_v12.py", "ontology.py",
              "envelope.py", "subjectbind.py"):
        fp[f"digest[{n}]"] = digest(PKG / n)
    for k, v in fp.items():
        A(f"   {k:<26}{v}")
    A("")

    A("2. REGRESSION INVARIANTS vs THE RULED FIGURES")
    cmp("documents", len(pa["rows"]), RULED["documents"])
    wf = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "WHOLE_FILE")
    sp = sum(1 for r in pa["rows"] for v in r["witnesses"].values()
             for w in v if w["applicability_scope"] == "SPAN")
    cmp("witnesses", wf + sp, RULED["witnesses"])
    cmp("M3 scope WHOLE_FILE/SPAN", (wf, sp), RULED["m3_scope"])
    cmp("VALIDITY", cl["axis_tallies"]["VALIDITY"], RULED["validity"])
    for ax, want in RULED["axes"].items():
        cmp(f"axis {ax}", cl["axis_tallies"][ax], want)
    for k, want in RULED["facts"].items():
        cmp(f"fact {k}", cl["evidence_fact_tally"].get(k), want)
    ids = sorted(r["path"] for r in cl["rows"]
                 if r["evidence_facts"].get("STATIC_REFERENCE_AT_SUBJECT"))
    cmp("E2 static-reference identities", ids,
        sorted(RULED["static_reference_identities"]))
    cmp("passA payload digest", digest(passa_out), RULED["passa_digest"])
    A("")
    A("3. DOCUMENT IDENTITY AGAINST THE FROZEN TREE")
    bad = []
    for r in pa["rows"]:
        blob = subprocess.run(["git", "-C", a.subject_repo, "show",
                               f"{pa['subject_tree']}:{r['path']}"],
                              capture_output=True).stdout
        if hashlib.sha256(blob.decode(errors="ignore").encode()
                          ).hexdigest()[:16] != r["sha256"]:
            bad.append(r["path"])
    cmp("documents whose bytes match the frozen tree", len(bad), 0)
    A("")
    A("4. THE GATE RAN")
    rec = PA._source_binding_gate(a.subject_repo, subject)
    A(f"   {json.dumps(rec, indent=3)}")
    A("")
    A("5. OUTCOME")
    A(f"   invariants checked {2 + len(RULED['axes']) + len(RULED['facts']) + 5}"
      f"   failures {len(fails)}")
    if fails:
        A("   UNEXPECTED DELTA. REPORTED, NOT REPAIRED:")
        for f in fails:
            A(f"     <<< {f}")
    else:
        A("   Every ruled invariant held. Nothing was repaired, and no")
        A("   figure was used as a target.")
    text = "\n".join(o) + "\n"
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
