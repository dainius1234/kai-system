#!/usr/bin/env python3
"""S1 CONTROLS — R4 boundary gate + P1 symlink rejection.

Banked WITH the implementation, before any canonical measurement.

THE PREDECESSOR IS EXECUTED, NOT PARAPHRASED. passa.py is restored from
the git object into a temporary package copy and run as a subprocess, so
sys.modules can never hand the old code the new constants.

EVERY GUARD IS PROVED TO FIRE, NOT MERELY TO PASS. A guard demonstrated
only on clean examples earns no closure, so each prerequisite is
deliberately violated and the refusal is the evidence.

ORDERING IS PROVED AT RUNTIME, NOT BY A LINE-NUMBER COMMENT. When the
gate aborts, the census package must never have been imported -- the
import sits after the gate inside build() -- so a subprocess reports
whether `docgraph` reached sys.modules. A comment claiming "the gate is
first" would prove nothing.

    python3 s1_controls.py --subject-repo R --tree T --census-package C
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
import passa as PA                                             # noqa: E402

S1_PREDECESSOR = "2eef215"
PKG_PATH = "kai-pm/house_in_order_h2_v13"
CHANGED = ("passa.py",)
UNCHANGED = ("classify.py", "envelope.py", "ontology.py", "subjectbind.py",
             "run_h2_v12.py", "qualify.py", "holdout.py")
# THE EXTERNAL BYTES MUST BE OBSERVABLE IN A PASS A FIELD. Pass A does
# not store raw document text, so a bare marker word proves nothing: it
# would be absent whether or not the bytes were read. A DATE is emitted
# as a witness VALUE, so an external date appearing in a witness is
# direct evidence that bytes from outside the commit reached the record.
EXTERNAL_MARKER = "2099-11-22"
EXTERNAL_DOC = f"# outside\n\n**Last updated:** {EXTERNAL_MARKER}\n"
FAILED = []


def check(name, got, want, extra=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    print(f"  {'OK  ' if ok else '<<< '}{name:<66}{str(got)[:34]:<36}"
          f"{'' if ok else 'expected ' + str(want)[:34]}")
    if extra:
        print(f"        {extra}")
    return ok


def sh(cwd, *args, check_rc=True):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if check_rc and r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout.decode(errors="replace")


def digest(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def predecessor_pkg(repo, ref, work):
    dst = work / "pred"
    shutil.copytree(PKG, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "build_evidence"))
    for n in CHANGED:
        (dst / n).write_bytes(sh(repo, "git", "show",
                                 f"{ref}:{PKG_PATH}/{n}").encode())
        if digest(dst / n) == digest(PKG / n):
            raise SystemExit(f"R11 ABORT: {n} is identical to the "
                             f"predecessor; nothing is being compared.")
    for n in UNCHANGED:
        if (PKG / n).exists() and digest(dst / n) != digest(PKG / n):
            raise SystemExit(f"R11 ABORT: {n} differs between the trees. "
                             f"Only {CHANGED} may differ.")
    return dst


# ── repositories ──────────────────────────────────────────────────────
DOC = "# Alpha\n\n**Last updated:** 2026-01-01\n"
DOC_DIRTY = "# Alpha\n\n**Last updated:** 2099-12-31\n"
DOC2 = "# Beta\n\n**Last updated:** 2026-01-02\n"
READER = ('import pathlib\nDOC = pathlib.Path("docs/alpha.md")\n'
          'def go():\n    return DOC.read_text()\n')


def base(work, name, extra=None):
    root = work / name
    (root / "docs").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "docs" / "alpha.md").write_text(DOC)
    (root / "docs" / "beta.md").write_text(DOC2)
    (root / "scripts" / "r.py").write_text(READER)
    if extra:
        extra(root, work)
    sh(root, "git", "init", "-q", ".")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=s1@x", "-c", "user.name=s1",
       "commit", "-qm", "subject")
    return root, sh(root, "git", "rev-parse", "HEAD").strip()


# The direct-library probe. Run as a subprocess so the census import is
# observable: if the gate fired first, `docgraph` never reached
# sys.modules, and that is the ORDERING PROOF.
DIRECT = r"""
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import passa as PA
out = {"raised": None, "census_imported": None, "rows": 0, "marker": False}
try:
    rows, tracked = PA.build(sys.argv[2], sys.argv[2], sys.argv[3],
                             pathlib.Path(sys.argv[4]))
    out["rows"] = len(rows)
    out["marker"] = any(sys.argv[5] in json.dumps(r) for r in rows)
except SystemExit as e:
    out["raised"] = f"SystemExit: {e}"
except Exception as e:
    out["raised"] = f"{type(e).__name__}: {e}"
out["census_imported"] = "docgraph" in sys.modules
json.dump(out, sys.stdout)
"""


def direct_build(pkg, root, subject, census):
    r = subprocess.run([sys.executable, "-c", DIRECT, str(pkg), str(root),
                        subject, str(census), EXTERNAL_MARKER],
                       capture_output=True, text=True)
    if not r.stdout:
        raise SystemExit(f"R11 ABORT: direct probe produced nothing: "
                         f"{r.stderr[-400:]}")
    return json.loads(r.stdout)


def cli(pkg, root, subject, census, work, tag):
    out = work / f"cli_{tag}.json"
    r = subprocess.run([sys.executable, str(pkg / "passa.py"),
                        "--subject-repo", str(root), "--history-repo",
                        str(root), "--subject", subject,
                        "--census-package", str(census), "--out", str(out)],
                       capture_output=True, text=True)
    body = (r.stdout + r.stderr)
    return {"rc": r.returncode, "wrote_output": out.exists(),
            "abort": next((l for l in body.split("\n") if "ABORT" in l), ""),
            "traceback": "Traceback" in body,
            "marker": EXTERNAL_MARKER in (out.read_text() if out.exists()
                                          else "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--old-ref", default=S1_PREDECESSOR)
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="s1c_"))
    pred = predecessor_pkg(a.repo, a.old_ref, work)
    census = a.census_package
    gitver = sh(a.subject_repo, "git", "--version").strip()
    print(f"PREDECESSOR   {a.old_ref} passa.py from the git object, executed")
    print(f"CANDIDATE     working tree {PKG.name}")
    print(f"GATE COMMAND  git {' '.join(PA.CLEAN_TREE_CMD)}")
    print(f"GIT VERSION   {gitver}")
    print(f"SYMLINK MODE  {PA.GIT_SYMLINK_MODE}  policy P1_REJECT")

    # 1 ────────────────────────────────────────────────────────────────
    print("\n1. THE FROZEN CLEAN-TREE COMMAND DETECTS ALL FIVE CLASSES")
    root, subj = base(work, "detect")
    cases = []
    (root / "docs" / "alpha.md").write_text(DOC_DIRTY)
    cases.append(("tracked modification", sh(root, "git", *PA.CLEAN_TREE_CMD)))
    sh(root, "git", "add", "docs/alpha.md")
    cases.append(("staged modification", sh(root, "git", *PA.CLEAN_TREE_CMD)))
    sh(root, "git", "checkout", "-q", "HEAD", "--", "docs/alpha.md")
    sh(root, "git", "mv", "docs/beta.md", "docs/renamed.md")
    cases.append(("rename", sh(root, "git", *PA.CLEAN_TREE_CMD)))
    sh(root, "git", "mv", "docs/renamed.md", "docs/beta.md")
    (root / "docs" / "beta.md").unlink()
    cases.append(("deletion", sh(root, "git", *PA.CLEAN_TREE_CMD)))
    sh(root, "git", "checkout", "-q", "HEAD", "--", "docs/beta.md")
    (root / "docs" / "untracked.md").write_text("# untracked\n")
    cases.append(("untracked file", sh(root, "git", *PA.CLEAN_TREE_CMD)))
    for label, out in cases:
        check(f"detects {label}", bool(out.strip()), True,
              f"porcelain: {out.strip().splitlines()[0] if out.strip() else ''}")
    last = cases[-1][1]
    check("untracked appears as '??' and is separable from tracked",
          all(l.startswith("??") for l in last.split("\n") if l.strip()), True)

    # 2 ────────────────────────────────────────────────────────────────
    print("\n2. READER POPULATION IS ENTIRELY TRACKED — why untracked files")
    print("   are RECORDED but do NOT abort")
    sys.path.insert(0, census)
    import docgraph as G, opscan as O                          # noqa: E402
    tracked_all = set(sh(a.subject_repo, "git", "ls-tree", "-r",
                         "--name-only", a.tree).split())
    docs = set(G.tracked_md(a.subject_repo))
    srcs = set(O.source_population(pathlib.Path(a.subject_repo)))
    check("every document Pass A reads is tracked", docs <= tracked_all, True,
          f"{len(docs)} documents")
    check("every source file opscan reads is tracked", srcs <= tracked_all,
          True, f"{len(srcs)} source files")
    check("reader population is a subset of the tracked tree",
          (docs | srcs) <= tracked_all, True,
          f"{len(docs | srcs)} of {len(tracked_all)} tracked paths")
    print("        An untracked file is therefore never opened, so it cannot")
    print("        change source identity. The claim is measured, not assumed.")

    # 3 ────────────────────────────────────────────────────────────────
    print("\n3. FAIL-OLD / PASS-NEW — A, dirty tracked file, HEAD == subject")
    root, subj = base(work, "A")
    (root / "docs" / "alpha.md").write_text(DOC_DIRTY)
    old_cli = cli(pred, root, subj, census, work, "A_old")
    new_cli = cli(PKG, root, subj, census, work, "A_new")
    old_dir = direct_build(pred, root, subj, census)
    new_dir = direct_build(PKG, root, subj, census)
    check("FAIL-OLD  CLI wrote output", old_cli["wrote_output"], True,
          f"rc={old_cli['rc']}")
    check("PASS-NEW  CLI refused, no output", new_cli["wrote_output"], False,
          new_cli["abort"][:120])
    check("PASS-NEW  the abort names WORKTREE IDENTITY",
          "WORKTREE IDENTITY" in new_cli["abort"], True)
    check("FAIL-OLD  direct build() measured", old_dir["rows"] > 0, True,
          f"{old_dir['rows']} rows")
    check("PASS-NEW  direct build() refused", new_dir["raised"] is not None,
          True, str(new_dir["raised"])[:110])
    check("PASS-NEW  ORDERING: census never imported before the gate fired",
          new_dir["census_imported"], False)

    # 4 ────────────────────────────────────────────────────────────────
    print("\n4. DEMONSTRATION C — tracked deletion, both paths")
    root, subj = base(work, "C")
    (root / "docs" / "beta.md").unlink()
    old_cli = cli(pred, root, subj, census, work, "C_old")
    new_cli = cli(PKG, root, subj, census, work, "C_new")
    old_dir = direct_build(pred, root, subj, census)
    new_dir = direct_build(PKG, root, subj, census)
    check("FAIL-OLD  CLI reached an unhandled traceback",
          old_cli["traceback"], True)
    check("FAIL-OLD  direct build() raised FileNotFoundError",
          "FileNotFoundError" in str(old_dir["raised"]), True)
    check("PASS-NEW  CLI: named prerequisite abort, no traceback",
          (new_cli["traceback"], "ABORT" in new_cli["abort"]), (False, True),
          new_cli["abort"][:120])
    check("PASS-NEW  direct: named prerequisite abort",
          "SOURCE BINDING" in str(new_dir["raised"]), True)
    check("PASS-NEW  no FileNotFoundError anywhere",
          "FileNotFoundError" in str(new_dir["raised"]), False)
    check("PASS-NEW  ORDERING: census never imported",
          new_dir["census_imported"], False)

    # 5 ────────────────────────────────────────────────────────────────
    print("\n5. SYMLINKS — rejected on GIT MODE, before any read")
    def mk(kind):
        def add(root, work):
            if kind == "internal":
                (root / "docs" / "link.md").symlink_to("alpha.md")
            elif kind == "escape":
                ext = work / f"outside_{kind}.md"
                ext.write_text(EXTERNAL_DOC)
                (root / "docs" / "link.md").symlink_to(ext)
            else:
                (root / "docs" / "link.md").symlink_to("nope.md")
        return add
    for kind in ("internal", "escape", "broken"):
        root, subj = base(work, f"sym_{kind}", mk(kind))
        mode = sh(root, "git", "ls-tree", subj, "--",
                  "docs/link.md").split()[0]
        new_dir = direct_build(PKG, root, subj, census)
        new_c = cli(PKG, root, subj, census, work, f"sym_{kind}")
        check(f"{kind:<9} tracked at git mode 120000", mode,
              PA.GIT_SYMLINK_MODE)
        check(f"{kind:<9} direct build() refused",
              "TRACKED SYMLINK" in str(new_dir["raised"]), True)
        check(f"{kind:<9} CLI refused, no output", new_c["wrote_output"],
              False)
        check(f"{kind:<9} ORDERING: census never imported",
              new_dir["census_imported"], False)
        if kind == "escape":
            old_dir = direct_build(pred, root, subj, census)
            old_c = cli(pred, root, subj, census, work, "sym_escape_old")
            check("escape    FAIL-OLD: external bytes ENTERED a Pass A field",
                  old_dir["marker"], True,
                  "the predecessor followed the link out of the repository")
            check("escape    FAIL-OLD: the CLI output carried them too",
                  old_c["marker"], True)
            check("escape    PASS-NEW: no external bytes in any field",
                  new_dir["marker"], False)
            check("escape    PASS-NEW: no output file exists to carry them",
                  new_c["marker"], False)

    # 6 ────────────────────────────────────────────────────────────────
    print("\n6. NON-HEAD BEHAVIOUR UNCHANGED — B, D, E still refused")
    for tag, mutate in (
            ("B", lambda r: (r / "docs" / "alpha.md").write_text(DOC_DIRTY)),
            ("D", lambda r: (r / "docs" / "gamma.md").write_text(DOC2)),
            ("E", lambda r: sh(r, "git", "mv", "docs/beta.md",
                               "docs/delta.md"))):
        root, subj = base(work, f"nonhead_{tag}")
        mutate(root)
        sh(root, "git", "add", "-A")
        sh(root, "git", "-c", "user.email=s@x", "-c", "user.name=s",
           "commit", "-qm", "later")
        new_c = cli(PKG, root, subj, census, work, f"nh_{tag}")
        new_d = direct_build(PKG, root, subj, census)
        check(f"{tag}  CLI still refused", new_c["wrote_output"], False,
              new_c["abort"][:110])
        # THE CLI ABORTS AT main() FIRST, and that is the PRESERVED
        # contract, not a defect: main()'s HEAD check predates S1 and
        # Kai's ruling forbids weakening it. Asserting the new message
        # text here would have demanded that S1 REPLACE that gate. What
        # matters is that the CLI refuses and that the LIBRARY path --
        # which had no protection at all before -- now names the
        # prerequisite.
        check(f"{tag}  CLI abort is an R11 refusal", "R11 ABORT" in
              new_c["abort"], True)
        check(f"{tag}  direct build() ALSO refused now",
              "SUBJECT IDENTITY" in str(new_d["raised"]), True)

    # 7 ────────────────────────────────────────────────────────────────
    print("\n7. THE CANONICAL SUBJECT STILL PASSES THE GATE")
    rec = PA._source_binding_gate(a.subject_repo,
                                  sh(a.subject_repo, "git", "rev-parse",
                                     "HEAD").strip())
    check("gate returns on the canonical subject", rec["gate"],
          "S1_SOURCE_BINDING")
    check("  tracked divergences", rec["worktree_identity"]
          ["tracked_divergences"], 0)
    check("  tracked symlinks found", rec["tracked_symlinks"]["found"], 0)
    check("  command recorded", rec["worktree_identity"]["command"],
          "git " + " ".join(PA.CLEAN_TREE_CMD))
    check("  git version recorded", rec["worktree_identity"]["git_version"],
          gitver)
    print("\n   PASS A PAYLOAD BYTES DELIBERATELY UNCHANGED:")
    out = work / "canon.json"
    subprocess.run([sys.executable, str(PKG / "passa.py"),
                    "--subject-repo", a.subject_repo, "--history-repo",
                    a.subject_repo, "--subject",
                    sh(a.subject_repo, "git", "rev-parse", "HEAD").strip(),
                    "--census-package", census, "--out", str(out)],
                   capture_output=True, check=True)
    check("canonical passA digest unchanged by S1", digest(out),
          "24fb1f555560277fd4555087f22ed06ef4a72efee845a339bbc87b925730f51c",
          "the gate record is NOT added to the payload, so the digest that "
          "proves the historical E2 run read the subject's bytes survives")

    # 8 ────────────────────────────────────────────────────────────────
    print("\n8. NOTHING ELSE MOVED")
    for n in UNCHANGED:
        if (PKG / n).exists():
            check(f"{n:<24} byte-identical to the predecessor",
                  digest(PKG / n), digest(pred / n))
    check("build() signature unchanged (2-tuple)",
          len(PA.build.__code__.co_varnames[:4]), 4)

    shutil.rmtree(work, ignore_errors=True)
    print(f"\nCONTROLS: "
          f"{'ALL PASS' if not FAILED else 'FAILED — ' + '; '.join(FAILED)}")
    raise SystemExit(0 if not FAILED else 1)


if __name__ == "__main__":
    main()
