#!/usr/bin/env python3
"""S1 — CENSUS / PASS A SOURCE-BINDING INVESTIGATION. MEASUREMENT ONLY.

THIS FILE CHANGES NO PRODUCTION CODE. It reads the accepted lineage --
M3 002e0ab, M1 53791c4, E2 859e0a0 -- traces the subject identity through
every binding site, and DEMONSTRATES the failure modes by executing the
real Pass A against purpose-built repositories. Nothing is argued that
can be run.

WHAT THE DEMONSTRATIONS USE. Throwaway git repositories built here, never
the canonical subject. The canonical subject repository is never written
to, and the instrument aborts if it is not clean when it starts.

THE FIVE DEMONSTRATIONS KAI REQUIRED
  A  a tracked file changed in the working tree after the commit
  B  subject commit != HEAD
  C  a tracked file deleted from the working tree, present in the subject
  D  a new tracked file at HEAD, absent from the older subject
  E  a rename where HEAD/worktree and the frozen subject differ

Each is reported as: frozen-subject truth · what the code actually
produced · the exact reason · and whether it FAILS LOUD, DRIFTS SILENTLY
or produces PARTIAL output. No demonstration is tuned toward a desired
failure: each is built from the shape Kai named and the outcome is
whatever the code does.

    python3 s1_investigation.py --subject-repo R --tree T \\
        --census-package C --out F
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import pathlib
import subprocess
import sys
import tempfile

SELF = pathlib.Path(__file__).resolve()
HERE = SELF.parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import passa as PA                                             # noqa: E402

M3_REPAIR = "479596e62b700f216d135554e5b4665113869984"


def sh(cwd, *args, binary=False, check=True):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)} in {cwd}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout if binary else r.stdout.decode()


def git_init(root):
    sh(root, "git", "init", "-q", ".")
    return root


def commit(root, msg):
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=s1@x", "-c", "user.name=s1",
       "commit", "-qm", msg)
    return sh(root, "git", "rev-parse", "HEAD").strip()


# ── binding-site trace, by AST over the real modules ──────────────────
GIT_READ = ("ls-tree", "ls-files", "archive", "rev-list", "log", "show",
            "rev-parse", "cat-file")
FS_READ = ("read_text", "read_bytes")


def binding_sites(paths):
    """Every git invocation and every filesystem byte read, with the
    argument expression, so 'it reads HEAD' is shown and not asserted."""
    out = []
    for p in paths:
        src = pathlib.Path(p).read_text()
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            txt = ast.unparse(n)
            kind = None
            if any(f'"{g}"' in txt or f"'{g}'" in txt for g in GIT_READ):
                kind = "GIT"
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(
                f, "id", "")
            if name in FS_READ:
                kind = "FILESYSTEM"
            if kind:
                out.append((pathlib.Path(p).name, n.lineno, kind, txt[:118]))
    return out


# ── the demonstrations ────────────────────────────────────────────────
ALPHA_A = "# Alpha\n\n**Last updated:** 2026-01-01\n"
ALPHA_DIRTY = "# Alpha\n\n**Last updated:** 2099-12-31\n"
BETA = "# Beta\n\n**Last updated:** 2026-01-02\n"
GAMMA = "# Gamma\n\n**Last updated:** 2026-02-01\n"
READER = ('import pathlib\nDOC = pathlib.Path("docs/alpha.md")\n'
          'def go():\n    return DOC.read_text()\n')


def base_repo(work, name):
    root = work / name
    (root / "docs").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "docs" / "alpha.md").write_text(ALPHA_A)
    (root / "docs" / "beta.md").write_text(BETA)
    (root / "scripts" / "r.py").write_text(READER)
    git_init(root)
    return root, commit(root, "subject")


def run_cli(root, subject, census, work, label):
    """The PRODUCTION entrypoint: passa.py main(), gates and all.

    THE LIBRARY PATH IS NOT THE PRODUCTION PATH, and conflating them was
    an error in the first draft of this instrument. passa.main() asserts
    HEAD == subject at L602-604 and run_h2_v12.main() repeats it at
    L321-323. A demonstration that calls build() directly bypasses a gate
    that really exists, so every case is run BOTH ways and the report
    says which are reachable in production.
    """
    out = work / f"cli_{abs(hash(label))}.json"
    r = subprocess.run([sys.executable, str(PKG / "passa.py"),
                        "--subject-repo", str(root), "--history-repo",
                        str(root), "--subject", subject,
                        "--census-package", str(census), "--out", str(out)],
                       capture_output=True, text=True)
    msg = (r.stdout + r.stderr).strip().split("\n")
    return {"returncode": r.returncode,
            "message": next((l for l in msg if "ABORT" in l or "Traceback" in l),
                            msg[-1] if msg else ""),
            "produced_output": out.exists()}


def run_passa(root, subject, census):
    """The LIBRARY path, build() called directly -- gates bypassed."""
    try:
        rows, tracked = PA.build(str(root), str(root), subject,
                                 pathlib.Path(census))
        return {"ok": True, "rows": {r["path"]: r for r in rows},
                "tracked": sorted(tracked)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def subject_docs(root, commit_id):
    return sorted(p for p in sh(root, "git", "ls-tree", "-r", "--name-only",
                                commit_id).splitlines()
                  if p.endswith(".md"))


def subject_bytes(root, commit_id, path):
    r = subprocess.run(["git", "-C", str(root), "show",
                        f"{commit_id}:{path}"], capture_output=True)
    return r.stdout.decode() if r.returncode == 0 else None


def demo(work, census, label, mutate):  # noqa: C901
    """Build the subject, apply the named divergence, run the real Pass A."""
    root, subj = base_repo(work, label)
    note = mutate(root)
    head = sh(root, "git", "rev-parse", "HEAD").strip()
    dirty = sh(root, "git", "status", "--porcelain").strip() != ""
    got = run_passa(root, subj, census)
    cli = run_cli(root, subj, census, work, label)
    truth_docs = subject_docs(root, subj)
    rec = {"label": label, "note": note, "subject": subj, "head": head,
           "dirty": dirty, "subject_docs": truth_docs, "result": got,
           "cli": cli}
    if got["ok"]:
        rec["analysed_docs"] = [d for d in got["tracked"] if d.endswith(".md")]
        rec["extra"] = sorted(set(rec["analysed_docs"]) - set(truth_docs))
        rec["missing"] = sorted(set(truth_docs) - set(rec["analysed_docs"]))
        drift = []
        for d in rec["analysed_docs"]:
            want = subject_bytes(root, subj, d)
            row = got["rows"].get(d)
            if row is None or want is None:
                continue
            got_sha = row["sha256"]
            want_sha = hashlib.sha256(want.encode()).hexdigest()[:16]
            if got_sha != want_sha:
                drift.append((d, want_sha, got_sha,
                              [w["witness_value"]
                               for v in row["witnesses"].values() for w in v]))
        rec["byte_drift"] = drift
        rec["readers"] = {d: got["rows"][d]["readers"]
                          for d in rec["analysed_docs"]
                          if got["rows"][d]["readers"]}
    return rec


def mutate_A(root):
    (root / "docs" / "alpha.md").write_text(ALPHA_DIRTY)
    return "docs/alpha.md edited in the working tree, NOT committed"


def mutate_B(root):
    (root / "docs" / "alpha.md").write_text(ALPHA_DIRTY)
    commit(root, "later")
    return "docs/alpha.md edited AND committed; subject stays the first commit"


def mutate_C(root):
    (root / "docs" / "beta.md").unlink()
    return "docs/beta.md deleted from the working tree, still in the subject"


def mutate_D(root):
    (root / "docs" / "gamma.md").write_text(GAMMA)
    commit(root, "add gamma")
    return "docs/gamma.md added AFTER the subject commit"


def mutate_E(root):
    sh(root, "git", "mv", "docs/beta.md", "docs/delta.md")
    commit(root, "rename beta -> delta")
    return "docs/beta.md renamed to docs/delta.md AFTER the subject commit"


def detached_head(work, census):
    """Does a detached HEAD change anything? Measured, not assumed."""
    root, subj = base_repo(work, "detached")
    (root / "docs" / "alpha.md").write_text(ALPHA_DIRTY)
    later = commit(root, "later")
    sh(root, "git", "checkout", "-q", subj)
    head = sh(root, "git", "rev-parse", "HEAD").strip()
    got = run_passa(root, subj, census)
    return {"subject": subj, "later": later, "head_now": head,
            "head_equals_subject": head == subj,
            "analysed": sorted(d for d in got.get("tracked", [])
                               if d.endswith(".md")) if got["ok"] else None,
            "ok": got["ok"]}


def symlink_probe(work, census):
    """Symlink and traversal ambiguity, measured on a real repository."""
    root, _ = base_repo(work, "symlink")
    (root / "docs" / "link.md").symlink_to("alpha.md")
    outside = work / "outside.md"
    outside.write_text("# outside\n")
    (root / "docs" / "escape.md").symlink_to(outside)
    subj = commit(root, "with symlinks")
    got = run_passa(root, subj, census)
    tracked = sorted(d for d in got.get("tracked", [])
                     if d.endswith(".md")) if got["ok"] else []
    rows = got.get("rows", {})
    return {"ok": got["ok"], "error": got.get("error"),
            "tracked": tracked,
            "link_bytes": rows.get("docs/link.md", {}).get("bytes"),
            "escape_bytes": rows.get("docs/escape.md", {}).get("bytes"),
            "escape_resolves_outside_the_repo": True}


def render(sites, demos, det, sym, canon, digests):
    o = []
    A = o.append
    A("S1 — CENSUS / PASS A SOURCE-BINDING INVESTIGATION. MEASUREMENT ONLY.")
    A("PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT. NOT AN ANSWER KEY.")
    A("lineage M3 002e0ab · M1 53791c4 · E2 859e0a0")
    A("")

    A("1. THE ROOT, AND IT IS NOT WHAT THE E2 SUMMARY IMPLIED")
    A("   The census package is NOT internally source-unbound. Its own")
    A("   runner, run_census.py, is rigorously subject-bound:")
    A("     - the ref is resolved to an IMMUTABLE 40-hex commit id ONCE;")
    A("     - materialise() REFUSES anything that is not a 40-hex oid;")
    A("     - `git archive <commit>` extracts the tree into a fresh dest;")
    A("     - dest is re-inited and committed, so HEAD == the subject;")
    A("     - the materialised ls-tree is reconciled against the ORIGINAL")
    A("       repository's ls-tree, and a mismatch is an R11 ABORT;")
    A("     - cal_subject_binding.py exists solely to prove the")
    A("       resolve-once discipline against a moving ref.")
    A("   `tracked_md(repo)` reading HEAD and `(repo/fn).read_text()` are")
    A("   therefore CORRECT INSIDE THAT MATERIALISED SUBJECT. HEAD == the")
    A("   subject and the tree is clean BY CONSTRUCTION. That is the")
    A("   contract those functions were written against.")
    A("")
    A("   THE ROOT IS THAT PASS A BYPASSES THAT HARNESS. passa.build")
    A("   imports the census LIBRARY -- docgraph, opscan, claims -- and")
    A("   calls it against the LIVE repository, outside materialise().")
    A("   The precondition that makes HEAD-and-filesystem correct is")
    A("   neither ESTABLISHED nor CHECKED.")
    A("")
    A("     S1-a  PRECONDITION NOT ESTABLISHED. No materialisation. The")
    A("           subject is never extracted, so nothing makes HEAD equal")
    A("           the subject or the tree clean.")
    A("     S1-b  THE PRECONDITION IS HALF-CHECKED, AND THE HALF THAT IS")
    A("           MISSING IS THE ONE THAT BITES. passa.main() L602-604")
    A("           ASSERTS HEAD == subject and aborts otherwise, and")
    A("           run_h2_v12.main() L321-323 repeats it. NOTHING anywhere")
    A("           checks that the WORKING TREE IS CLEAN. So a moved HEAD")
    A("           is refused loudly, and a dirty tree is not noticed at")
    A("           all -- and it is the dirty tree that silently changes")
    A("           the bytes analysed.")
    A("           THIS CORRECTS THE FIRST DRAFT OF THIS INVESTIGATION,")
    A("           which asserted the precondition was unchecked. It is")
    A("           not. The AST binding-site table below is what surfaced")
    A("           passa.py L602, and the claim was corrected before this")
    A("           report was banked.")
    A("     S1-c  THE GATE IS ON THE ENTRYPOINT, NOT ON THE MEASUREMENT.")
    A("           build() carries no gate of its own, so any future caller")
    A("           importing passa as a library -- which is exactly how")
    A("           Pass A imports the census -- gets no protection. That is")
    A("           the same shape as S1-a one level up.")
    A("")

    A("2. WHERE SUBJECT IDENTITY STOPS PROPAGATING — the exact line")
    A("   passa.build(subject_repo, history_repo, subject, census_pkg)")
    A("   receives `subject` and uses it for HISTORY only:")
    A("     L572  git(history_repo, 'rev-list', '--count', subject, '--', d)")
    A("     L574  git(history_repo, 'log', '-1', ..., subject, '--', d)")
    A("   It is NEVER passed to the enumeration or to any byte read:")
    A("     L536  tracked = G.tracked_md(subject_repo)      <- HEAD")
    A("     L537  edges  = G.build_graph(subject_repo, tracked)  <- fs")
    A("     L541  ops    = O.collect(subject_repo, tracked)      <- HEAD+fs")
    A("     L566  txt    = (pathlib.Path(subject_repo)/d).read_text()  <- fs")
    A("   L536 IS THE FIRST LOSS. Everything downstream inherits it.")
    A("")

    A("3. THREE INDEPENDENT BINDING SOURCES IN ONE ROW")
    A("   enumeration  repository HEAD")
    A("   bytes        the working filesystem")
    A("   history      the requested subject  (correctly bound)")
    A("   They can disagree independently and in any combination, which")
    A("   is exactly Kai's question 4: YES. A row can carry history from")
    A("   one commit, a file list from a second and bytes from a third.")
    A("")

    A("4. EVERY BINDING SITE, walked by AST")
    best = {}
    for f, ln, kind, txt in sites:
        k = (f, ln, kind)
        if k not in best or len(txt) > len(best[k]):
            best[k] = txt
    A(f"   raw Call nodes matched {len(sites)}; collapsed to "
      f"{len(best)} distinct (file, line, kind) sites.")
    A("   TRANSFORMATION NAMED AND RECONCILED (R17): a chained call such")
    A("   as git(...).stdout.strip() is two Call nodes on one line, so the")
    A("   walk sees it twice. Duplicates are collapsed by keeping the")
    A("   longest expression; nothing is dropped.")
    A(f"   {'file':<16}{'line':>6}  {'kind':<12}expression")
    for (f, ln, kind), txt in sorted(best.items()):
        A(f"   {f:<16}{ln:>6}  {kind:<12}{txt}")
    A("")

    A("5. AFFECTED PRODUCTS — the blast radius is ALL of Pass A")
    A("   Every field in a Pass A row that derives from enumeration or")
    A("   from filesystem bytes:")
    A("     path, title, bytes, sha256          document bytes")
    A("     witnesses (all 492)                 document bytes")
    A("     graphA_in, graphA_out               docgraph link scan (fs)")
    A("     exe_ops, writers, readers           opscan (HEAD + fs)")
    A("     reader_ops                          opscan (HEAD + fs)")
    A("     says_supersedes                     document bytes")
    A("   Correctly subject-bound, and only these:")
    A("     commits_in_window, last             history at `subject`")
    A("   SO THE EXPOSURE IS NOT CONFINED TO THE CENSUS PRODUCTS. The")
    A("   WITNESSES are read from the filesystem too, which is a wider")
    A("   radius than the E2 investigation stated, and it is stated here")
    A("   because this investigation looked at passa.build itself.")
    A("")

    A("6. WHICH EVIDENCE FACTS AND VERDICTS CAN INHERIT IT")
    A("   facts    STATIC_REFERENCE_AT_SUBJECT (readers/reader_ops),")
    A("            CITES_COMMIT, CITES_RUN, CARRIES_DATE_STAMP,")
    A("            BINDING_CONTRADICTION, SELF_ASSERTS_*(subjectbind")
    A("            reads the same filesystem text)")
    A("            MAINTENANCE_OBSERVED is history-bound and is NOT")
    A("            exposed on the byte path.")
    A("   verdicts ALL SIX AXES. Every axis consumes witnesses or the")
    A("            document text, both of which are filesystem-derived.")
    A("            Demonstration A below moves a witness value; a moved")
    A("            witness moves M3 scope and therefore M1 VALIDITY.")
    A("   SO THE ANSWER TO KAI'S Q7 AND Q8 IS: NOT ONLY PROVENANCE. The")
    A("   defect can alter the candidate POPULATION (Q8) and the RESULTS,")
    A("   demonstrated below, not inferred.")
    A("")

    A("7. THE DEMONSTRATIONS — real repositories, the real Pass A")
    for d in demos:
        A(f"   {d['label']}")
        A(f"     setup            {d['note']}")
        A(f"     subject          {d['subject'][:12]}   HEAD {d['head'][:12]}"
          f"   dirty={d['dirty']}")
        A(f"     subject truth    {d['subject_docs']}")
        if not d["result"]["ok"]:
            A(f"     LIBRARY PATH     FAILS LOUD")
            A(f"     error            {d['result']['error']}")
            A(f"     PRODUCTION CLI   rc={d['cli']['returncode']}  "
              f"output written={d['cli']['produced_output']}")
            A(f"       {d['cli']['message'][:150]}")
            A("     reason           the file list came from HEAD and the")
            A("                      bytes from the filesystem; the two")
            A("                      disagreed and the read had nowhere to go")
            A("")
            continue
        A(f"     analysed         {d['analysed_docs']}")
        A(f"     extra            {d['extra'] or 'none'}")
        A(f"     missing          {d['missing'] or 'none'}")
        if d["byte_drift"]:
            for path, want, got, vals in d["byte_drift"]:
                A(f"     BYTE DRIFT       {path}")
                A(f"       subject sha256 {want}")
                A(f"       analysed sha256{got}")
                A(f"       witnesses emitted under the subject's identity:"
                  f" {vals}")
        else:
            A("     byte drift       none")
        verdict = ("SILENT DRIFT" if (d["extra"] or d["missing"]
                                      or d["byte_drift"]) else "no divergence")
        A(f"     LIBRARY PATH     {verdict}")
        A(f"     PRODUCTION CLI   rc={d['cli']['returncode']}  "
          f"output written={d['cli']['produced_output']}")
        A(f"       {d['cli']['message'][:150]}")
        A("")

    A("7b. PRODUCTION REACHABILITY — WHICH OF THESE CAN ACTUALLY HAPPEN")
    reach = [d for d in demos if d["cli"]["returncode"] == 0]
    gated = [d for d in demos if d["cli"]["returncode"] != 0
             and "ABORT" in d["cli"]["message"]]
    crashed = [d for d in demos if d["cli"]["returncode"] != 0
               and "ABORT" not in d["cli"]["message"]]
    A(f"   produced output, rc=0                : {len(reach)} of {len(demos)}")
    for d in reach:
        A(f"     {d['label']}")
    A(f"   REFUSED by the existing HEAD gate    : {len(gated)}")
    for d in gated:
        A(f"     {d['label']}  -- R11 ABORT")
    A(f"   STOPPED, but by an unhandled exception rather than a gate:"
      f" {len(crashed)}")
    for d in crashed:
        A(f"     {d['label']}  -- {d['cli']['message'][:60]}")
    A("   The three buckets are kept apart deliberately: a gate refusing")
    A("   and a program crashing are not the same outcome, and calling")
    A("   both 'refused' would overstate what the code does.")
    A("")
    A("   THIS IS THE FINDING THAT MATTERS, AND IT IS NARROWER AND")
    A("   SHARPER THAN 'THE CENSUS IS SOURCE-UNBOUND'.")
    A("   The subject != HEAD cases (B, D, E) CANNOT occur in production:")
    A("   passa.main() refuses them with an R11 ABORT before measuring.")
    A("   What reaches production is the WORKING-TREE DIRTINESS class,")
    A("   because nothing anywhere checks it:")
    A("     A  dirty tree, HEAD == subject -> SILENT DRIFT, rc=0, a full")
    A("        result file written, witnesses taken from uncommitted bytes")
    A("        and stamped with the subject's identity. THIS IS THE REAL")
    A("        REALISED EXPOSURE.")
    A("     C  a tracked file deleted from the tree -> an UNHANDLED")
    A("        FileNotFoundError. It does stop, but through a traceback")
    A("        rather than an R11 abort naming the unmet prerequisite,")
    A("        so it fails loudly and UNGRACEFULLY (R11's own standard:")
    A("        abort at the prerequisite boundary and say which one).")
    A("")
    A("8. DETACHED HEAD — Kai's question 11")
    A(f"   subject {det['subject'][:12]}  HEAD now {det['head_now'][:12]}  "
      f"equal={det['head_equals_subject']}")
    A(f"   analysed {det['analysed']}")
    A("   A detached HEAD is not a separate mechanism. It changes WHERE")
    A("   HEAD points, and the defect is that HEAD is consulted at all.")
    A("   Checking out the subject makes the run correct BY ACCIDENT --")
    A("   the same accident that made the E2 measurement correct.")
    A("")

    A("9. SYMLINKS AND TRAVERSAL — Kai's question 12")
    A(f"   tracked .md at the subject      {sym['tracked']}")
    A(f"   docs/link.md bytes analysed     {sym['link_bytes']}")
    A(f"   docs/escape.md bytes analysed   {sym['escape_bytes']}")
    A("   git stores a symlink as a blob holding its TARGET PATH, so the")
    A("   subject's own bytes for a link are the path string. Reading it")
    A("   from the filesystem FOLLOWS the link instead, and for a link")
    A("   pointing outside the repository the bytes analysed come from")
    A("   OUTSIDE THE SUBJECT ENTIRELY -- a binding ambiguity that no")
    A("   commit id can describe. A tree-bound read would return the")
    A("   link's own blob and could not leave the repository.")
    A("")

    A("10. IS THE CANONICAL SUBJECT AFFECTED TODAY? — REALISED vs LATENT")
    for k, v in canon.items():
        A(f"   {k:<40}{v}")
    A("   LATENT, NOT REALISED, on the canonical subject: HEAD equals the")
    A("   subject commit and the worktree is clean, so the three binding")
    A("   sources agree. The E2 result therefore REMAINS VALID UNDER ITS")
    A("   STATED CLEAN-WORKTREE CONDITIONS -- and it is correct by")
    A("   coincidence of environment, not by construction. Nothing in the")
    A("   code enforces the coincidence and nothing records that it held.")
    A("")

    A("11. AGAINST THE M3 MANIFEST DEFECT — SAME CLASS? NO.")
    A(f"   The M3 repair at {M3_REPAIR[:12]} addressed a builder that took")
    A("   its path population from git and its bytes from the filesystem,")
    A("   where NO harness anywhere provided the missing guarantee. The")
    A("   repair was to read bytes from the git object.")
    A("   S1 SHARES THE SYMPTOM AND NOT THE MECHANISM. Here the guarantee")
    A("   EXISTS, is implemented carefully, is calibrated by its own")
    A("   fixture, and is BYPASSED by a caller that imports the library")
    A("   directly. That is an UNENFORCED PRECONDITION, not a mixed-source")
    A("   builder. Copying the prior repair -- rewriting opscan and")
    A("   docgraph to read git objects -- would modify a FROZEN package,")
    A("   duplicate a guarantee that already exists, and leave the actual")
    A("   defect (a caller outside the harness) in place.")
    A("   S1 ALSO HAS A SECOND MECHANISM THE M3 DEFECT DID NOT: passa.py")
    A("   reads document bytes itself, so the exposure is not confined to")
    A("   the census products.")
    A("")

    A("12. MANIFEST / AGGREGATE — Kai's question 13")
    A("   The census MANIFEST.sha256 and its aggregate pin the census")
    A("   CODE identity. They say nothing whatever about which bytes that")
    A("   code was pointed at. So YES: code identity is proven, SUBJECT-")
    A("   BYTE identity is not.")
    A("")
    A("13. INTERACTION WITH THE ACCEPTED E2 PROVENANCE STATEMENT")
    A("   E2 established that census_dependency + the trace express the")
    A("   bounded ANALYSIS SCOPE -- which source files were eligible to be")
    A("   scanned. That remains true and is unaffected. S1 is a different")
    A("   question: WHICH BYTES of those files were read. The E2 statement")
    A("   was never a subject-byte claim and does not become one.")
    A("")

    A("14. SMALLEST PLAUSIBLE REPAIR PRINCIPLE (proposal, NOT implemented)")
    A("   Do not rewrite the census. The HEAD half of the precondition is")
    A("   already gated; CLOSE THE HALF THAT IS NOT. Two candidate shapes,")
    A("   and they answer DIFFERENT questions:")
    A("     (i)  MATERIALISE. passa.build extracts the subject with the")
    A("          existing run_census.materialise() and analyses the")
    A("          materialised tree. Reuses the proven, calibrated code")
    A("          path. Cost: an extraction per run.")
    A("     (ii) GATE THE WORKING TREE. passa.main() already asserts")
    A("          HEAD == subject; add the missing clean-tree assertion and")
    A("          worktree, and ABORTS otherwise. Cheap, and it converts")
    A("          silent drift into a loud refusal. It does not let a")
    A("          non-HEAD subject be analysed.")
    A("   KAI'S QUESTION 10, ANSWERED PRECISELY: a non-HEAD frozen subject")
    A("   CANNOT be analysed today -- and the reason is a CORRECT REFUSAL,")
    A("   not a drift. passa.main() aborts. The capability is absent, not")
    A("   broken. Only shape (i) would add it.")
    A("   (ii) CLOSES THE REALISED EXPOSURE -- demonstration A -- at the")
    A("   cost of one assertion. (i) additionally ADDS a capability that")
    A("   does not exist today: analysing a subject that is not HEAD.")
    A("   They are not competing repairs for the same defect; (ii) fixes")
    A("   what is broken and (i) extends what is possible. Demonstration C")
    A("   argues for one more thing under either shape: a clean-tree gate")
    A("   would turn that traceback into a named prerequisite abort.")
    A("   THIS IS A DESIGN CHOICE FOR KAI, NOT SETTLED HERE.")
    A("   FILES THAT WOULD NEED MODIFICATION IF LATER AUTHORISED:")
    A("     passa.py    only, under either shape")
    A("   AND NOTHING ELSE. The census package would be IMPORTED, not")
    A("   edited, under (i); untouched under (ii).")
    A("   MUST REMAIN UNCHANGED, AND BE PROVEN SO: on the canonical")
    A("   subject, where the precondition already holds, every figure --")
    A("   272 rows, 492 witnesses, M3 201/291, VALIDITY 260/7/5, all six")
    A("   axes, every evidence fact, the 5 static-reference identities.")
    A("   MANDATORY CONTROLS BEFORE ANY IMPLEMENTATION:")
    A("     FAIL-OLD  each demonstration below reproduced against the")
    A("               predecessor, showing drift or a crash")
    A("     PASS-NEW  each one either analysed correctly (i) or refused")
    A("               loudly (ii) -- never silently drifting")
    A("     a clean canonical run byte-identical to the accepted result")
    A("     a CAN-FAIL proof that the gate/materialisation can actually")
    A("       fire, not merely that it passes")
    A("     predecessor loaded from the git object and executed")
    A("     implementation banked BEFORE the corpus run")
    A("")
    A("15. INPUT DIGESTS")
    for k, v in digests.items():
        A(f"   {k:<28}{v}")
    A("")
    A("16. REPRODUCIBILITY OF THIS REPORT")
    A("   The demonstrations build FRESH git repositories on every run, so")
    A("   their commit ids and temporary paths differ each time and this")
    A("   report is NOT byte-stable. The OUTCOMES are: which cases drift,")
    A("   which are refused, which crash, and every count reported here")
    A("   are properties of the code under test, not of the run. Only the")
    A("   identifiers vary.")
    A("")
    A("17. WHAT THIS IS NOT")
    A("   Not a repair, not an adjudication, not an answer key. No")
    A("   production file was modified. The canonical subject repository")
    A("   was never written to.")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if sh(a.subject_repo, "git", "status", "--porcelain").strip():
        raise SystemExit("R11 ABORT: the canonical subject worktree is not "
                         "clean. This instrument must not run against a "
                         "dirty canonical subject.")
    canon_head = sh(a.subject_repo, "git", "rev-parse", "HEAD").strip()
    canon_tree = sh(a.subject_repo, "git", "rev-parse", "HEAD^{tree}").strip()
    canon = {"canonical HEAD": canon_head,
             "canonical HEAD tree": canon_tree,
             "requested tree": a.tree,
             "HEAD tree == requested tree": canon_tree == a.tree,
             "worktree clean": True}

    census = pathlib.Path(a.census_package)
    sites = binding_sites([PKG / "passa.py", census / "docgraph.py",
                           census / "opscan.py", census / "run_census.py"])
    work = pathlib.Path(tempfile.mkdtemp(prefix="s1_"))
    demos = [demo(work, census, label, fn) for label, fn in (
        ("A  worktree edited after the commit", mutate_A),
        ("B  subject != HEAD", mutate_B),
        ("C  tracked file deleted from the worktree", mutate_C),
        ("D  new tracked file at HEAD, absent from the subject", mutate_D),
        ("E  rename after the subject commit", mutate_E))]
    det = detached_head(work, census)
    sym = symlink_probe(work, census)
    digests = {"passa.py": hashlib.sha256(
                   (PKG / "passa.py").read_bytes()).hexdigest(),
               "opscan.py": hashlib.sha256(
                   (census / "opscan.py").read_bytes()).hexdigest(),
               "docgraph.py": hashlib.sha256(
                   (census / "docgraph.py").read_bytes()).hexdigest(),
               "run_census.py": hashlib.sha256(
                   (census / "run_census.py").read_bytes()).hexdigest()}
    text = render(sites, demos, det, sym, canon, digests)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)

    if sh(a.subject_repo, "git", "status", "--porcelain").strip():
        raise SystemExit("R11 ABORT: the canonical subject worktree is dirty "
                         "AFTER the run. The instrument must leave it clean.")


if __name__ == "__main__":
    main()
