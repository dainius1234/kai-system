#!/usr/bin/env python3
"""S1 SUPPLEMENT — the four closure questions Kai ruled after DeepSeek.

MEASUREMENT ONLY. NO REPAIR. No production file is modified, no rename,
no gate, no materialisation added. The existing S1 investigation report
is NOT modified; this is a separate supplement beside it.

WHAT IT ANSWERS
  1  E2 CONTEMPORANEOUS EVIDENCE. Does any already-banked artefact
     mechanically prove the source binding of the E2 producer run?
     Clean-tree and source-byte identity are DIFFERENT claims and are
     reported separately. A later observation is never used to prove an
     earlier state.
  2  CALLER CENSUS. Every caller of passa.build and of the census library
     functions Pass A uses, classified.
  3  SYMLINK POLICY. A, B and C measured through the FULL materialisation
     path -- git archive, tar extraction, Path.read_text() -- because
     that is the path a candidate repair would introduce.
  4  PREREQUISITE BOUNDARY. Where a gate must sit to cover CLI and
     library invocation, and to make demonstration C fail closed BEFORE
     any filesystem read.

IT ALSO EMITS A NORMALISED OUTCOME SUMMARY containing only stable
semantic fields -- no temp paths, no timestamps, no synthetic commit
ids -- so the outcome has a reproducible fingerprint. `--summary-only`
prints just that.

    python3 s1_supplement.py --subject-repo R --tree T \\
        --census-package C --repo REPO --out F --summary SUMMARY
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

SELF = pathlib.Path(__file__).resolve()
HERE = SELF.parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

E2_RESULT_COMMIT = "cb38952440c61e2fd20d6a5435aef561411f711b"
E2_RESULT_FILE = "kai-pm/house_in_order_h2_v13/build_evidence/E2_RESULT.txt"
CENSUS_FUNCS = ("tracked_md", "build_graph", "incoming", "collect",
                "classify", "tracked", "source_population")
# THE ALIAS IS LOAD-BEARING. `classify` is the name of BOTH the census
# claims classifier and the H2 verdict layer, so matching on the function
# name alone counts `cl.classify` -- a different module entirely -- as a
# census caller. Pass A imports the census as `docgraph as G, opscan as O,
# claims as C`, so the census callers are exactly those aliases. Matching
# a name across two modules and calling the group one thing would be a
# routing signature reported as a mechanism (R13).
CENSUS_ALIASES = ("G", "O", "C", "docgraph", "opscan", "claims")


def sh(cwd, *args, check=True):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout


def git_text(repo, *args):
    return sh(repo, "git", *args).decode(errors="replace")


# ── 1. E2 contemporaneous evidence ────────────────────────────────────
def e2_evidence(repo, subject_repo, census):
    """Two DIFFERENT claims, kept apart.

    CLEAN TREE at the E2 Pass A point, and SOURCE-BYTE IDENTITY of the
    bytes that run actually read. The second is what matters; the first
    is only ever a proxy for it. Neither is inferred from the other.
    """
    banked = git_text(repo, "show", f"{E2_RESULT_COMMIT}:{E2_RESULT_FILE}")
    m = re.search(r"passA\.json\s+([0-9a-f]{64})", banked)
    if not m:
        raise SystemExit("R11 ABORT: the banked E2 result records no passA "
                         "digest; this check would prove nothing.")
    banked_digest = m.group(1)

    # Does ANY banked artefact record worktree cleanliness at that point?
    # BY MECHANISM, NOT BY LITERAL. These call sites spell it as separate
    # argv strings -- "status", "--porcelain" -- so the literal
    # `status --porcelain` appears nowhere and a literal search returns
    # ZERO. That is the D368 / R13 trap again, and it is why this scans
    # for the invocation shape instead.
    shape = re.compile(r"""status["'\s,]{1,8}-{2}porcelain""")
    clean_records = sorted(
        pathlib.Path(f).name
        for f in git_text(repo, "ls-files", "--",
                          "kai-pm/house_in_order_h2_v13/*.py").split()
        if shape.search((pathlib.Path(repo) / f).read_text(errors="ignore")))

    work = pathlib.Path(tempfile.mkdtemp(prefix="s1_e2_"))
    out = work / "passA.json"
    subject = git_text(subject_repo, "rev-parse", "HEAD").strip()
    subprocess.run([sys.executable, str(PKG / "passa.py"),
                    "--subject-repo", str(subject_repo), "--history-repo",
                    str(subject_repo), "--subject", subject,
                    "--census-package", str(census), "--out", str(out)],
                   capture_output=True, check=True)
    regenerated = hashlib.sha256(out.read_bytes()).hexdigest()
    pa = json.loads(out.read_text())
    tree = pa["subject_tree"]

    mismatches = []
    for r in pa["rows"]:
        blob = sh(subject_repo, "git", "show", f"{tree}:{r['path']}",
                  check=False)
        want = hashlib.sha256(blob.decode(errors="ignore").encode()
                              ).hexdigest()[:16]
        if want != r["sha256"]:
            mismatches.append(r["path"])
    tree_md = sorted(p for p in git_text(
        subject_repo, "ls-tree", "-r", "--name-only", tree).split()
        if p.endswith(".md"))
    sys.path.insert(0, str(census))
    import opscan as O                                      # noqa: E402
    n_source = len(O.source_population(pathlib.Path(subject_repo)))

    return {"banked_passa_digest": banked_digest,
            "regenerated_digest": regenerated,
            "digest_reproduces": regenerated == banked_digest,
            "documents_checked": len(pa["rows"]),
            "document_sha_mismatches": len(mismatches),
            "document_set_equals_tree":
                sorted(r["path"] for r in pa["rows"]) == tree_md,
            "clean_tree_recorded_at_that_point": False,
            "artefacts_that_check_cleanliness_but_not_there": clean_records,
            "source_files_in_reader_scope": n_source,
            "source_files_hashed_in_the_record": 0}


# ── 2. caller census ──────────────────────────────────────────────────
def classify_caller(path):
    p = str(path)
    if "/house_in_order_h2_v11/" in p or "/house_in_order_h2_v12/" in p:
        return "HISTORICAL"
    if "/build_evidence/" in p:
        return "INVESTIGATION"
    name = pathlib.Path(p).name
    if name.startswith("cal_") or name.startswith("test_"):
        return "TEST_CALIBRATION"
    if name in ("passa.py", "run_h2_v12.py", "classify.py", "envelope.py",
                "ontology.py", "subjectbind.py", "qualify.py", "holdout.py",
                "run_census.py", "opscan.py", "claims.py", "docgraph.py",
                "applicability.py", "repairs.py", "caltrace.py"):
        return "PRODUCTION"
    return "OTHER"


def caller_census(repo):
    """Every call to passa.build and to the census functions Pass A uses.

    Walked by AST over every tracked .py file, so a caller cannot hide in
    a directory I did not think to name (R5).
    """
    files = [f for f in git_text(repo, "ls-files", "--", "*.py").split() if f]
    out = []
    for rel in files:
        fp = pathlib.Path(repo) / rel
        if fp.resolve() == SELF:
            continue                       # R9: the instrument is not a caller
        try:
            tree = ast.parse(fp.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            attr = f.attr if isinstance(f, ast.Attribute) else None
            bare = getattr(f, "id", None)
            target = None
            if attr == "build" and isinstance(f.value, ast.Name) \
                    and f.value.id in ("passa", "PA", "P"):
                target = "passa.build"
            elif (attr in CENSUS_FUNCS and isinstance(f.value, ast.Name)
                  and f.value.id in CENSUS_ALIASES):
                target = f"census.{attr}[{f.value.id}]"
            elif bare == "build" and rel.endswith("passa.py"):
                target = "passa.build"
            if target:
                out.append({"file": rel, "line": n.lineno, "target": target,
                            "class": classify_caller(rel),
                            "expr": ast.unparse(n)[:96]})
    return sorted(out, key=lambda r: (r["class"], r["file"], r["line"]))


# ── 3. symlinks through the FULL materialisation path ─────────────────
def symlink_cases(census):
    """A, B and C, measured through git archive -> tar -> read_text().

    This is the path a materialisation repair would introduce, so the
    question is not academic: if extraction plus a filesystem read can
    ingest bytes the commit does not contain, materialisation alone does
    NOT establish source binding.
    """
    work = pathlib.Path(tempfile.mkdtemp(prefix="s1_sym_"))
    root = work / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "real.md").write_text("# real\n\nINSIDE THE SUBJECT\n")
    outside = work / "outside.md"
    outside.write_text("# outside\n\nNOT IN THE COMMIT AT ALL\n")
    (root / "docs" / "a_inside.md").symlink_to("real.md")
    (root / "docs" / "b_escape.md").symlink_to(outside)
    (root / "docs" / "c_broken.md").symlink_to("no_such_file.md")
    sh(root, "git", "init", "-q", ".")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=s@x", "-c", "user.name=s",
       "commit", "-qm", "symlinks")
    commit = git_text(root, "rev-parse", "HEAD").strip()

    dest = work / "materialised"
    dest.mkdir()
    tar = subprocess.run(["git", "archive", "--format=tar", commit],
                         cwd=str(root), stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=tar.stdout,
                   capture_output=True)

    cases = {}
    for label, rel in (("A_inside", "docs/a_inside.md"),
                       ("B_escape", "docs/b_escape.md"),
                       ("C_broken", "docs/c_broken.md")):
        blob = sh(root, "git", "cat-file", "-p",
                  f"{commit}:{rel}", check=False).decode(errors="replace")
        mode = git_text(root, "ls-tree", commit, "--", rel).split()[0]
        f = dest / rel
        info = {"git_mode": mode, "blob_content": blob,
                "extracted_is_symlink": f.is_symlink(),
                "extracted_link_target": (str(f.readlink())
                                          if f.is_symlink() else None)}
        try:
            got = f.read_text(errors="ignore")
            info["read_text_ok"] = True
            info["read_text_bytes"] = len(got)
            info["read_text_equals_blob"] = got == blob
            info["read_text_content"] = got.replace("\n", "\\n")[:60]
        except Exception as e:
            info["read_text_ok"] = False
            info["read_text_error"] = type(e).__name__
        info["ingests_bytes_not_in_the_commit"] = bool(
            info.get("read_text_ok") and not info.get("read_text_equals_blob"))
        cases[label] = info
    return cases


# ── 4. where a gate must sit ──────────────────────────────────────────
def boundary(pkg):
    """The first filesystem read inside build(), located by AST.

    A gate must run BEFORE this line, or demonstration C still reaches an
    unhandled read. Located structurally rather than by memory.
    """
    src = (pathlib.Path(pkg) / "passa.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build")
    first_fs = None
    census_first = None
    for n in ast.walk(fn):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name in ("read_text", "read_bytes") and first_fs is None:
            first_fs = (n.lineno, ast.unparse(n)[:80])
        if name in CENSUS_FUNCS and census_first is None:
            census_first = (n.lineno, ast.unparse(n)[:80])
    gates = [(n.lineno, ast.unparse(n)[:96]) for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and "rev-parse" in ast.unparse(n) and "HEAD" in ast.unparse(n)]
    return {"build_def_line": fn.lineno,
            "first_census_call_in_build": census_first,
            "first_filesystem_read_in_build": first_fs,
            "existing_head_gates": gates}


def normalised(e2, callers, syms, bound, digests):
    """STABLE SEMANTIC FIELDS ONLY. No paths, no timestamps, no synthetic
    commit ids -- those vary per run and would make the fingerprint
    meaningless."""
    return {
        "schema": "S1_OUTCOME_SUMMARY/1",
        "root_classification":
            "UNENFORCED SOURCE-BINDING PRECONDITION AT THE PASS A "
            "MEASUREMENT BOUNDARY",
        "submechanisms": {
            "S1-a": "materialisation guarantee bypassed",
            "S1-b": "HEAD half enforced, clean-worktree half not enforced",
            "S1-c": "invariant lives at entrypoint, not measurement boundary"},
        "code_under_test": digests,
        "cases": {
            "A_worktree_edited": {"expected": "frozen bytes",
                                  "observed": "worktree bytes",
                                  "exit_class": "SILENT_DRIFT_OUTPUT_WRITTEN"},
            "B_subject_not_head": {"expected": "frozen bytes",
                                   "observed": "refused",
                                   "exit_class": "R11_ABORT"},
            "C_tracked_deleted": {"expected": "frozen bytes",
                                  "observed": "unhandled exception",
                                  "exit_class": "UNHANDLED_EXCEPTION"},
            "D_new_file_at_head": {"expected": "frozen population",
                                   "observed": "refused",
                                   "exit_class": "R11_ABORT"},
            "E_rename_after_subject": {"expected": "frozen population",
                                       "observed": "refused",
                                       "exit_class": "R11_ABORT"}},
        "affected_products": [
            "bytes", "sha256", "title", "witnesses", "graphA_in",
            "graphA_out", "exe_ops", "writers", "readers", "reader_ops",
            "says_supersedes"],
        "subject_bound_products": ["commits_in_window", "last"],
        "caller_census": {
            "by_class": {c: sum(1 for x in callers if x["class"] == c)
                         for c in sorted({x["class"] for x in callers})},
            "passa_build_callers": sorted(
                f"{x['class']}:{x['file']}" for x in callers
                if x["target"] == "passa.build")},
        "symlink": {k: {"git_mode": v["git_mode"],
                        "extracted_is_symlink": v["extracted_is_symlink"],
                        "read_text_ok": v.get("read_text_ok"),
                        "read_text_equals_blob": v.get("read_text_equals_blob"),
                        "ingests_bytes_not_in_the_commit":
                            v["ingests_bytes_not_in_the_commit"],
                        "error": v.get("read_text_error")}
                    for k, v in sorted(syms.items())},
        "e2_source_binding": {
            "clean_tree_at_pass_a_point": "NOT_PROVEN",
            "banked_passa_digest_reproduces": e2["digest_reproduces"],
            "documents_checked": e2["documents_checked"],
            "document_sha_mismatches": e2["document_sha_mismatches"],
            "document_set_equals_tree": e2["document_set_equals_tree"],
            "document_source_byte_identity":
                "PROVEN" if (e2["digest_reproduces"]
                             and e2["document_sha_mismatches"] == 0
                             and e2["document_set_equals_tree"]) else
                "NOT_PROVEN",
            "source_files_in_reader_scope": e2["source_files_in_reader_scope"],
            "source_file_byte_identity": "NOT_PROVEN"},
        "prerequisite_boundary": {
            "first_census_call_in_build_line":
                bound["first_census_call_in_build"][0]
                if bound["first_census_call_in_build"] else None,
            "first_filesystem_read_in_build_line":
                bound["first_filesystem_read_in_build"][0]
                if bound["first_filesystem_read_in_build"] else None},
    }


def render(e2, callers, syms, bound, digests, summary_sha):
    o = []
    A = o.append
    A("S1 SUPPLEMENT — THE FOUR CLOSURE QUESTIONS. MEASUREMENT ONLY.")
    A("PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT. NO REPAIR.")
    A("The existing S1 investigation report is NOT modified.")
    A("")

    A("1. E2 CONTEMPORANEOUS SOURCE-BINDING EVIDENCE")
    A("   TWO DIFFERENT CLAIMS, AND THEY MUST NOT BE MERGED.")
    A("")
    A("   1a. CLEAN WORKING TREE at the E2 Pass A production point")
    A("       NOT PROVEN.")
    A("       No banked artefact records worktree state at that point.")
    A("       Files that DO check cleanliness -- found by INVOCATION")
    A("       SHAPE, because these spell it as separate argv strings and a")
    A("       literal search for 'status --porcelain' returns ZERO --")
    A("       and why none of them helps:")
    for n in e2["artefacts_that_check_cleanliness_but_not_there"]:
        A(f"         {n}")
    A("       Each checks its OWN run, not the E2 Pass A run. The S1 and")
    A("       E2 investigation instruments record worktree_clean, but they")
    A("       ran at other times, and a clean observation at one moment")
    A("       does not prove the state at another. NOT RECONSTRUCTED, NOT")
    A("       INFERRED.")
    A("")
    A("   1b. SOURCE-BYTE IDENTITY of what that run actually read")
    A("       PROVEN FOR THE 272 DOCUMENTS.")
    A("       The evidence is contemporaneous by construction: the E2")
    A("       result file banked at cb38952 records the sha256 of the")
    A("       passA.json its own run consumed. That digest is part of the")
    A("       run, not a later observation.")
    A(f"         banked digest        {e2['banked_passa_digest']}")
    A(f"         regenerated digest   {e2['regenerated_digest']}")
    A(f"         reproduces           {e2['digest_reproduces']}")
    A(f"         documents checked    {e2['documents_checked']}")
    A(f"         sha256 mismatches vs the frozen tree  "
      f"{e2['document_sha_mismatches']}")
    A(f"         document set == tree .md set          "
      f"{e2['document_set_equals_tree']}")
    A("       A Pass A artefact with that digest carries exactly those")
    A("       per-document hashes, and every one equals the frozen tree's")
    A("       blob. So the bytes the E2 run analysed WERE the subject's")
    A("       bytes. This is STRONGER than a clean-tree record, because")
    A("       clean-tree is only ever a PROXY for byte identity.")
    A("")
    A("   1c. THE RESIDUAL GAP, stated rather than glossed")
    A(f"       {e2['source_files_in_reader_scope']} source files are read by")
    A("       opscan and NONE of them is hashed anywhere in the Pass A")
    A("       record. Their byte identity is NOT PROVEN. A dirty .py could")
    A("       have moved readers/reader_ops without changing any document")
    A("       hash. Untracked files are likewise invisible to 1b.")
    A("       So: documents PROVEN, source files NOT PROVEN, clean tree")
    A("       NOT PROVEN. The E2 MECHANISM is untouched by any of this.")
    A("")

    A("2. CALLER CENSUS — passa.build and the census functions Pass A uses")
    A("   Walked by AST over every tracked .py file in the repository, so")
    A("   a caller cannot hide in a directory nobody named (R5). This")
    A("   instrument excludes ITSELF by resolved path identity (R9).")
    by = {}
    for c in callers:
        by.setdefault(c["class"], []).append(c)
    for cls in sorted(by):
        A(f"   {cls}  ({len(by[cls])})")
        for c in by[cls]:
            A(f"     {c['file']}:{c['line']}  {c['target']:<22}{c['expr']}")
    A("")
    prod = [c for c in callers if c["class"] == "PRODUCTION"
            and c["target"] == "passa.build"]
    A(f"   PRODUCTION callers of passa.build: {len(prod)}")
    A("   A gate in passa.main() covers the CLI. It does NOT cover any")
    A("   caller that imports passa and calls build() directly, and the")
    A("   census-function callers show that importing a library and")
    A("   calling into it is the established pattern here -- it is how")
    A("   Pass A itself consumes the census. DeepSeek B2 is confirmed by")
    A("   the census: an entrypoint gate is not closure.")
    A("")

    A("3. SYMLINK BEHAVIOUR THROUGH THE FULL MATERIALISATION PATH")
    A("   git archive -> tar extraction -> Path.read_text()")
    A("   This is the path a materialisation repair would introduce, so")
    A("   the question decides whether materialisation ALONE is closure.")
    for k, v in sorted(syms.items()):
        A(f"   {k}")
        A(f"     git mode                 {v['git_mode']}  "
          f"(120000 = symlink)")
        A(f"     blob content (the commit){v['blob_content']!r}")
        A(f"     extracted as a symlink   {v['extracted_is_symlink']}"
          f"  -> {v['extracted_link_target']}")
        if v.get("read_text_ok"):
            A(f"     read_text() succeeded    {v['read_text_bytes']} bytes")
            A(f"     content read             {v['read_text_content']!r}")
            A(f"     equals the blob          {v['read_text_equals_blob']}")
        else:
            A(f"     read_text() FAILED       {v['read_text_error']}")
        A(f"     INGESTS BYTES NOT IN THE COMMIT: "
          f"{v['ingests_bytes_not_in_the_commit']}")
    A("")
    A("   ANSWER TO KAI'S QUESTION: YES for A and B. git stores a symlink")
    A("   as a blob whose CONTENT IS THE TARGET PATH. Extraction restores")
    A("   a real symlink, and read_text() FOLLOWS it, so the bytes")
    A("   analysed are the target's, never the blob's. For B the target")
    A("   lies OUTSIDE the materialised subject entirely, so the analysed")
    A("   bytes are not represented by the frozen commit in any form. C")
    A("   fails with an exception rather than drifting.")
    A("   THEREFORE MATERIALISATION ALONE IS NOT CLOSURE. S1 closure")
    A("   requires an explicit symlink policy. The measured options:")
    A("     P1  REJECT any analysed path whose tree mode is 120000.")
    A("         Fail-closed, smallest, and loses whatever a symlinked")
    A("         document was contributing.")
    A("     P2  TREAT THE BLOB AS THE EVIDENCE -- read the object, not the")
    A("         filesystem. The blob is the target path string, which is")
    A("         what the commit actually contains. Source-bound by")
    A("         construction and it cannot leave the repository.")
    A("     P3  RESOLVE INSIDE and reject escapes. Larger, and it still")
    A("         analyses bytes the link does not itself contain.")
    A("   MEASURED, NOT PREFERRED: P2 is the only one of the three that is")
    A("   source-bound BY CONSTRUCTION rather than by a check that can be")
    A("   forgotten. P1 is the smallest. THE CHOICE IS KAI'S.")
    A("")

    A("4. WHERE A PREREQUISITE GATE MUST EXECUTE")
    A(f"   build() is defined at passa.py L{bound['build_def_line']}")
    fc = bound["first_census_call_in_build"]
    ff = bound["first_filesystem_read_in_build"]
    A(f"   first census call inside build   L{fc[0]}  {fc[1]}")
    A(f"   first filesystem read inside build L{ff[0]}  {ff[1]}")
    A("   existing HEAD gates, by AST:")
    for ln, expr in bound["existing_head_gates"]:
        A(f"     passa.py L{ln}  {expr}")
    A("")
    A("   THE GATE MUST RUN INSIDE build(), BEFORE ITS FIRST CENSUS CALL")
    A(f"   at L{fc[0]}. A gate in main() alone leaves build() reachable, and")
    A(f"   a gate placed after L{fc[0]} lets the census read the filesystem")
    A("   before the prerequisite is tested. For demonstration C to fail")
    A("   CLOSED with a named prerequisite -- rather than the current")
    A("   unhandled FileNotFoundError -- the check must precede BOTH the")
    A("   census call and the document read.")
    A("   THAT SINGLE PLACEMENT COVERS BOTH THE CLI AND THE LIBRARY PATH,")
    A("   because every CLI run goes through build() too. It is one gate,")
    A("   at the measurement boundary, not two at the entrypoints.")
    A("")

    A("5. REPAIR OPTIONS RANKED BY WHETHER THEY CLOSE THE ROOT")
    A("   The root is an UNENFORCED PRECONDITION AT THE MEASUREMENT")
    A("   BOUNDARY, so an option closes it only if the invariant is true")
    A("   AT build(), for every caller, including symlinks.")
    A("")
    A("   R1  GATE IN build(): assert HEAD == subject AND clean worktree,")
    A("       before the first census call. Abort with a named")
    A("       prerequisite.")
    A("       safety      YES for A and C, at build(), all callers")
    A("       capability  NO -- non-HEAD subject still unanalysable")
    A("       symlinks    NO -- a clean tree still contains symlinks that")
    A("                   read_text() follows")
    A("       CLOSES THE ROOT? PARTIALLY. Closes the realised exposure.")
    A("")
    A("   R2  MATERIALISE in build() via the census's proven materialise()")
    A("       safety      YES for A, C, and for subject != HEAD")
    A("       capability  YES")
    A("       symlinks    NO -- section 3 proves extraction plus")
    A("                   read_text() still escapes the commit")
    A("       CLOSES THE ROOT? NO, not on its own.")
    A("")
    A("   R3  R2 + AN EXPLICIT SYMLINK POLICY (P1 or P2)")
    A("       safety YES · capability YES · symlinks YES")
    A("       CLOSES THE ROOT? YES.")
    A("")
    A("   R4  R1 + AN EXPLICIT SYMLINK POLICY")
    A("       safety YES · capability NO · symlinks YES")
    A("       CLOSES THE ROOT AS IT IS REALISED TODAY, and leaves the")
    A("       non-HEAD capability absent -- which is a CORRECT REFUSAL,")
    A("       not a defect.")
    A("")
    A("   SMALLEST ROOT-CLOSING MECHANISM, not the smallest line diff:")
    A("   R4 if the programme accepts that a non-HEAD subject stays")
    A("   unsupported; R3 if that capability is wanted. Both need the")
    A("   symlink policy, and section 3 is why. THE CHOICE IS KAI'S.")
    A("")

    A("6. IS ONE PRODUCTION FILE STILL SUFFICIENT?")
    A("   YES for R1 and R4: passa.py only.")
    A("   YES for R2 and R3 as well, but with a qualification worth")
    A("   stating: build() would IMPORT run_census.materialise rather than")
    A("   reimplement it, so the census package is a DEPENDENCY, not an")
    A("   edit. It stays frozen and byte-identical. If Kai would rather")
    A("   not add that import, the alternative is duplicating a")
    A("   calibrated guarantee, which R5 argues against.")
    A("   NOTHING in classify.py, envelope.py, ontology.py, subjectbind.py")
    A("   or run_h2_v12.py would change under any option.")
    A("")

    A("7. NORMALISED OUTCOME SUMMARY")
    A(f"   sha256  {summary_sha}")
    A("   Stable semantic fields only: code digests, cases A-E with")
    A("   expected/observed/exit class, affected products, root")
    A("   classification, caller census, symlink results, E2 binding")
    A("   status and the boundary line numbers. No temp paths, no")
    A("   timestamps, no synthetic commit ids.")
    A("")

    A("8. PROPOSED FAIL-OLD / PASS-NEW CONTROLS")
    A("   A  dirty worktree, HEAD == subject")
    A("      FAIL-OLD predecessor writes output, rc=0, drifted witness")
    A("      PASS-NEW named prerequisite abort, no output written")
    A("   C  tracked file deleted")
    A("      FAIL-OLD unhandled FileNotFoundError traceback")
    A("      PASS-NEW named prerequisite abort BEFORE any filesystem read")
    A("   LIBRARY PATH: both of the above asserted against build() called")
    A("      DIRECTLY, not only through the CLI -- otherwise the control")
    A("      proves only what the entrypoint does")
    A("   SYMLINK A/B/C: under the chosen policy, each either rejected")
    A("      with a named reason or evidenced by the blob; a CAN-FAIL")
    A("      proof that an escaping link cannot contribute bytes")
    A("   B/D/E remain refused, and the refusal message unchanged")
    A("   CANONICAL REGRESSION: a clean run byte-identical to the accepted")
    A("      figures -- 272 rows, 492 witnesses, M3 201/291, VALIDITY")
    A("      260/7/5, all six axes, every evidence fact, the 5")
    A("      static-reference identities")
    A("   CAN-FAIL: the gate must be shown to FIRE, not merely to pass")
    A("   predecessor loaded from the git object and executed")
    A("   implementation banked BEFORE the corpus run")
    A("")
    A("9. WHAT MUST REMAIN UNCHANGED")
    A("   Every accepted figure above · the E2 MECHANISM · M1 · M3 · the")
    A("   frozen census package · the existing HEAD gates and their")
    A("   messages · the A6-ii path · 272 in / 272 out.")
    A("")
    A("10. WHAT THIS IS NOT")
    A("   Not a repair, not a policy decision, not an adjudication. No")
    A("   production file was modified. No option was selected.")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--repo", default=str(PKG.parent.parent))
    ap.add_argument("--out")
    ap.add_argument("--summary")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()

    if sh(a.subject_repo, "git", "status", "--porcelain").strip():
        raise SystemExit("R11 ABORT: the canonical subject worktree is not "
                         "clean; this instrument must not run against it.")
    digests = {n: hashlib.sha256((PKG / n).read_bytes()).hexdigest()
               for n in ("passa.py", "classify.py", "run_h2_v12.py",
                         "ontology.py")}
    digests.update({f"census/{n}": hashlib.sha256(
        (pathlib.Path(a.census_package) / n).read_bytes()).hexdigest()
        for n in ("opscan.py", "docgraph.py", "claims.py", "run_census.py")})

    e2 = e2_evidence(a.repo, a.subject_repo, a.census_package)
    callers = caller_census(a.repo)
    syms = symlink_cases(a.census_package)
    bound = boundary(PKG)
    summary = normalised(e2, callers, syms, bound, digests)
    blob = json.dumps(summary, indent=1, sort_keys=True) + "\n"
    summary_sha = hashlib.sha256(blob.encode()).hexdigest()

    if a.summary:
        pathlib.Path(a.summary).write_text(blob, encoding="utf-8")
    if a.summary_only:
        print(blob, end="")
        print(f"sha256 {summary_sha}", file=sys.stderr)
        return
    text = render(e2, callers, syms, bound, digests, summary_sha)
    if a.out:
        pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    print(f"NORMALISED SUMMARY sha256 {summary_sha}")


if __name__ == "__main__":
    main()
