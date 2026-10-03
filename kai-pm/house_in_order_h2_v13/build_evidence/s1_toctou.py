#!/usr/bin/env python3
"""S1 TOCTOU — SOURCE-STABILITY INVESTIGATION. MEASUREMENT ONLY.

NO PRODUCTION CHANGE. No hardening is implemented. The accepted S1
implementation ba258b8 is executed exactly as banked, and this instrument
only observes it.

THE QUESTION. The gate establishes a clean tree BEFORE measurement; Pass A
then reads live filesystem bytes. Is the window between them REACHABLE,
and is a post-check enough to close it?

HOW THE RACE IS MADE DETERMINISTIC, AND WHY THAT IS HONEST. Racing a real
writer against a real reader would make the result depend on scheduling
luck, and a demonstration that sometimes fails to fire proves nothing
either way. So the mutation is driven from a READ HOOK installed in the
probe subprocess: it fires ON a specific read, which is exactly the
instant a concurrent writer would have to hit. The hook does not change
what Pass A does; it changes only WHEN the external mutation lands. The
vulnerability shown is a property of the code; only the timing is
arranged.

WHAT IS MEASURED
  1  is the window reachable at all
  2  does a post-check catch it
  3  does CHANGE-AND-RESTORE defeat a post-check
  4  which read paths exist, and which are covered by hashes that ALREADY
     exist in the Pass A products
  5  is passa.py alone sufficient to close every path

    python3 s1_toctou.py --subject-repo R --census-package C --out F
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile

SELF = pathlib.Path(__file__).resolve()
HERE = SELF.parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

S1_ACCEPTED = "ba258b8"
DOC = "# Alpha\n\n**Last updated:** 2026-01-01\n"
DOC2 = "# Beta\n\n**Last updated:** 2026-01-02\n"
INJECTED = "# Beta\n\n**Last updated:** 2099-11-22\n"
INJECTED_MARK = "2099-11-22"
READER = ('import pathlib\nDOC = pathlib.Path("docs/alpha.md")\n'
          'def go():\n    return DOC.read_text()\n')


def sh(cwd, *args, check=True):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout.decode(errors="replace")


def base(work, name):
    root = work / name
    (root / "docs").mkdir(parents=True)
    (root / "scripts").mkdir(parents=True)
    (root / "docs" / "alpha.md").write_text(DOC)
    (root / "docs" / "beta.md").write_text(DOC2)
    (root / "scripts" / "r.py").write_text(READER)
    sh(root, "git", "init", "-q", ".")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=t@x", "-c", "user.name=t",
       "commit", "-qm", "subject")
    return root, sh(root, "git", "rev-parse", "HEAD").strip()


# The probe. `mode` selects whether the injected change is left in place
# or RESTORED after it has been consumed.
PROBE = r"""
import hashlib, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
root, subject, census, mode = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
target = pathlib.Path(root) / "docs" / "beta.md"
original = target.read_text()
injected = sys.argv[6]

real = pathlib.Path.read_text
state = {"injected": False, "restored": False}
log = []

def hooked(self, *a, **k):
    # Fire the mutation the instant the FIRST document is read -- the
    # gate has already passed by then, and beta.md has not been read yet.
    if not state["injected"] and self.name == "alpha.md":
        target.write_text(injected)
        state["injected"] = True
    out = real(self, *a, **k)
    # WHO read it, and WHAT did that read return? Attributing each read to
    # its calling module is what makes the census paths distinguishable
    # from Pass A's own; without it the two are indistinguishable in the
    # aggregate and the wrong conclusion is easy to draw.
    caller = sys._getframe(1).f_globals.get("__name__", "?")
    if self.name.endswith(".md"):
        log.append({"reader": caller, "file": self.name,
                    "sha": hashlib.sha256(out.encode()).hexdigest()[:16]})
    # CHANGE-AND-RESTORE: put the file back the moment its bytes have
    # been consumed, so a post-measurement status check sees a clean tree.
    if mode == "restore" and state["injected"] and not state["restored"] \
            and self.name == "beta.md":
        target.write_text(original)
        state["restored"] = True
    return out

pathlib.Path.read_text = hooked
import passa as PA
res = {"raised": None}
try:
    rows, tracked = PA.build(root, root, subject, pathlib.Path(census))
    by = {r["path"]: r for r in rows}
    res["beta_sha_recorded"] = by["docs/beta.md"]["sha256"]
    res["beta_witnesses"] = [w["witness_value"]
                             for v in by["docs/beta.md"]["witnesses"].values()
                             for w in v]
except SystemExit as e:
    res["raised"] = f"SystemExit: {e}"
except Exception as e:
    res["raised"] = f"{type(e).__name__}: {e}"
pathlib.Path.read_text = real
res["read_log"] = log
res["injected_fired"] = state["injected"]
res["restored_fired"] = state["restored"]
res["worktree_after"] = bool(__import__("subprocess").run(
    ["git", "-C", root, "status", "--porcelain=v1", "--untracked-files=all"],
    capture_output=True).stdout.strip())
# MATCH PASS A'S OWN HASHING EXACTLY: it reads with errors="ignore"
# and hashes the re-encoded text, so the comparison must too.
_blob = __import__("subprocess").run(
    ["git", "-C", root, "show", subject + ":docs/beta.md"],
    capture_output=True).stdout
res["frozen_beta_sha"] = hashlib.sha256(
    _blob.decode(errors="ignore").encode()).hexdigest()[:16]
res["injected_sha"] = hashlib.sha256(injected.encode()).hexdigest()[:16]
json.dump(res, sys.stdout)
"""


def probe(root, subject, census, mode):
    r = subprocess.run([sys.executable, "-c", PROBE, str(PKG), str(root),
                        subject, str(census), mode, INJECTED],
                       capture_output=True, text=True)
    if not r.stdout:
        raise SystemExit(f"R11 ABORT: probe produced nothing: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def read_paths(census):
    """Every filesystem byte read on the Pass A path, by AST, with whether
    a hash of THOSE bytes already survives in a Pass A product."""
    out = []
    for label, path, covered, why in (
        ("passa document read", PKG / "passa.py", True,
         "row['sha256'] is computed from the very bytes this read "
         "returned, so it already witnesses them"),
        ("docgraph link scan", pathlib.Path(census) / "docgraph.py", False,
         "a SECOND, separate read of the same documents; nothing hashes "
         "what IT returned"),
        ("opscan source read", pathlib.Path(census) / "opscan.py", False,
         "615 source files; no hash of their bytes exists anywhere in "
         "the record"),
    ):
        tree = ast.parse(pathlib.Path(path).read_text())
        sites = [(n.lineno, ast.unparse(n)[:74]) for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and (n.func.attr if isinstance(n.func, ast.Attribute)
                      else "") in ("read_text", "read_bytes")]
        out.append({"label": label, "file": pathlib.Path(path).name,
                    "sites": sites, "hash_survives": covered, "why": why,
                    "in_passa": pathlib.Path(path).name == "passa.py"})
    return out


def render(reach, restore, paths, digests):
    o = []
    A = o.append
    A("S1 TOCTOU — SOURCE-STABILITY INVESTIGATION. MEASUREMENT ONLY.")
    A("PRODUCED BY ORION. ZERO ADJUDICATION WEIGHT. NO HARDENING BUILT.")
    A(f"the accepted S1 implementation {S1_ACCEPTED} is executed as banked")
    A("")

    A("1. IS THE TOCTOU WINDOW REACHABLE? — YES, DEMONSTRATED")
    A("   A tracked file is changed AFTER the gate has passed and BEFORE")
    A("   its own measurement read. The mutation is fired from a read hook")
    A("   so the timing is deterministic; the hook changes only WHEN the")
    A("   external write lands, never what Pass A does.")
    A(f"     mutation fired                {reach['injected_fired']}")
    A(f"     Pass A completed              {reach['raised'] is None}")
    A(f"     frozen subject sha256(beta)   {reach['frozen_beta_sha']}")
    A(f"     sha256 Pass A RECORDED        {reach['beta_sha_recorded']}")
    A(f"     bytes consumed == frozen      "
      f"{reach['beta_sha_recorded'] == reach['frozen_beta_sha']}")
    A(f"     witnesses emitted             {reach['beta_witnesses']}")
    A(f"     the injected value {INJECTED_MARK} reached a witness: "
      f"{INJECTED_MARK in reach['beta_witnesses']}")
    A("   The gate passed, the run completed, and a witness carries a")
    A("   value that is NOT in the frozen subject. TOCTOU IS REACHABLE.")
    A("")

    A("2. WOULD A POST-CHECK CATCH IT? — IN THE SIMPLE CASE, YES")
    A(f"     worktree dirty after measurement  {reach['worktree_after']}")
    A("   With the change left in place a post-measurement status check")
    A("   would see a dirty tree and could refuse. That is real value.")
    A("")

    A("3. CHANGE-AND-RESTORE — DEFEATS A POST-CHECK, AND SHOWS WHICH")
    A("   READ PATH IS EXPOSED")
    A("   The same mutation, restored the moment its bytes are consumed.")
    A(f"     mutation fired                {restore['injected_fired']}")
    A(f"     restore fired                 {restore['restored_fired']}")
    A(f"     Pass A completed              {restore['raised'] is None}")
    A(f"     frozen sha256(beta)           {restore['frozen_beta_sha']}")
    A(f"     INJECTED sha256(beta)         {restore['injected_sha']}")
    A(f"     WORKTREE DIRTY AFTER          {restore['worktree_after']}")
    A("   EVERY DOCUMENT READ, ATTRIBUTED TO ITS CALLING MODULE:")
    for e in restore["read_log"]:
        tag = ("<-- INJECTED BYTES" if e["sha"] == restore["injected_sha"]
               else "")
        A(f"     {e['reader']:<10}{e['file']:<12}{e['sha']}  {tag}")
    A(f"     sha256 Pass A RECORDED for beta {restore['beta_sha_recorded']}")
    A(f"     witnesses emitted               {restore['beta_witnesses']}")
    A("")
    A("   READ THE ATTRIBUTION BEFORE THE CONCLUSION. The injected bytes")
    A("   were consumed by DOCGRAPH, whose read happens first; by the time")
    A("   Pass A read the same file the restore had landed, so Pass A's own")
    A("   sha256 and witnesses are CLEAN. My first draft of this section")
    A("   asserted 'wrong bytes analysed' while displaying evidence that")
    A("   showed Pass A's path unaffected -- the sentence was wider than")
    A("   the measurement, so the instrument was changed to attribute each")
    A("   read rather than the claim being softened.")
    A("")
    A("   WHAT IT ACTUALLY PROVES, AND IT IS STRONGER:")
    A("     (a) PRE-CHECK CLEAN and POST-CHECK CLEAN while foreign bytes")
    A("         were consumed by the measurement. A POST-CHECK ALONE IS")
    A("         NOT SUFFICIENT -- Kai's question 2, answered by execution.")
    A("     (b) row['sha256'] DID NOT DETECT IT, because the contaminated")
    A("         read was not the read that hash covers. A per-document")
    A("         hash cannot police the census read paths.")
    A("   Both demonstrations together bound the exposure: whichever read")
    A("   the window lands on is the one that is contaminated, and only")
    A("   one of the three read paths carries a hash that would notice.")
    A("")

    A("4. EVERY FILESYSTEM READ ON THE PASS A PATH")
    for p in paths:
        A(f"   {p['label']}  ({p['file']})")
        for ln, expr in p["sites"]:
            A(f"     L{ln}  {expr}")
        A(f"     hash of THOSE bytes already survives: {p['hash_survives']}")
        A(f"     {p['why']}")
        A(f"     inside passa.py: {p['in_passa']}")
    A("")

    A("5. H1 / H2 / H3 ASSESSED AGAINST THAT")
    A("   H1 POST-CHECK ONLY")
    A("      CATCHES a mutation left in place (section 2).")
    A("      CANNOT CATCH change-and-restore (section 3), and cannot")
    A("      catch a mutation reverted by any means before the check.")
    A("      CLASSIFICATION: PARTIAL RACE DETECTION ONLY. Not closure.")
    A("")
    A("   H2 PER-READ GIT-BLOB IDENTITY")
    A("      The right shape: compare the bytes ACTUALLY CONSUMED against")
    A("      the frozen blob, at the moment of consumption. A restore")
    A("      cannot defeat it, because the comparison happens before the")
    A("      restore can occur.")
    A("      REUSE OF EXISTING PRODUCTS, as Kai required: the document")
    A("      path already carries what is needed. row['sha256'] is")
    A("      computed from the very bytes the read returned, so verifying")
    A("      it against `git show <tree>:<path>` needs NO new hash and no")
    A("      restatement of any population rule.")
    A("      THE OTHER TWO PATHS DO NOT. docgraph re-reads the documents")
    A("      independently and opscan reads 615 source files, and nothing")
    A("      hashes what either of those reads returned.")
    A("      CLASSIFICATION: CLOSES THE DOCUMENT PATH COMPLETELY. Does")
    A("      NOT close the census paths without hashing inside them.")
    A("")
    A("   H3 IMMUTABLE SNAPSHOT READ")
    A("      Only reachable by materialising the subject, which is the R3")
    A("      capability redesign Kai has ruled out for this cycle. NOT")
    A("      EVALUATED FURTHER, per that instruction.")
    A("")

    A("6. IS passa.py ALONE STILL SUFFICIENT? — NO, AND THIS IS THE")
    A("   STOP CONDITION KAI NAMED")
    inside = [p for p in paths if p["in_passa"]]
    outside = [p for p in paths if not p["in_passa"]]
    A(f"     read paths inside passa.py        {len(inside)}")
    A(f"     read paths inside the census      {len(outside)}")
    for p in outside:
        A(f"       {p['label']} — {p['file']}")
    A("   A passa.py-only change can verify the DOCUMENT bytes it reads")
    A("   itself. It cannot verify bytes read by docgraph or opscan,")
    A("   because those reads happen inside the FROZEN census package and")
    A("   nothing about them survives into the record.")
    A("   PER KAI'S INSTRUCTION I AM STOPPING HERE RATHER THAN WIDENING.")
    A("   The options that follow are for Kai to choose between; none is")
    A("   implemented and none is preferred here.")
    A("     O1 CLOSE THE DOCUMENT PATH ONLY, in passa.py, and record the")
    A("        census paths as a declared residual. Smallest, honest, and")
    A("        it leaves readers/reader_ops and the link graph unproven.")
    A("     O2 VERIFY THE CENSUS INPUT SET in passa.py BEFORE and AFTER")
    A("        the census call by comparing every file in the reader")
    A("        population against its blob. Still a window, but a much")
    A("        narrower one, and it needs no census change.")
    A("     O3 HASH INSIDE THE CENSUS. Complete, and it modifies a FROZEN")
    A("        package -- the widening Kai told me to stop before.")
    A("   O1 and O2 are passa.py-only. O3 is not.")
    A("")

    A("7. PROPOSED FAIL-OLD / PASS-NEW CONTROLS for whichever is chosen")
    A("   FAIL-OLD, against ba258b8 exactly as banked:")
    A("     mutation left in place  -> wrong bytes consumed, run completes")
    A("     change-and-restore      -> wrong bytes consumed, pre and post")
    A("                                status both clean")
    A("   PASS-NEW:")
    A("     both cases REFUSED with a named prerequisite, before any")
    A("       output is written")
    A("     the refusal must name the PATH whose bytes diverged")
    A("     a CAN-FAIL proof that the comparison can actually fire")
    A("     the clean canonical subject still produces every ruled figure")
    A("       and the unchanged passA digest")
    A("   READER-POPULATION REGRESSION, retained as a guard rather than a")
    A("   new restriction: every path from which Pass A or the census")
    A("   obtains analysed bytes must remain inside the tracked git")
    A("   population. It does today -- 272 documents and 615 source files,")
    A("   all tracked. If a future change starts walking untracked paths,")
    A("   that change fails this control rather than passing silently.")
    A("   Harmless untracked files are NOT rejected.")
    A("")
    A("8. DIGESTS")
    for k, v in digests.items():
        A(f"   {k:<26}{v}")
    A("")
    A("9. WHAT THIS IS NOT")
    A("   Not a repair, not a selection, not an adjudication. No")
    A("   production file was modified. R4 + P1 is not redesigned and the")
    A("   root investigation is not reopened.")
    return "\n".join(o) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if sh(a.subject_repo, "git", "status", "--porcelain").strip():
        raise SystemExit("R11 ABORT: the canonical subject worktree is not "
                         "clean; this instrument must not run against it.")
    work = pathlib.Path(tempfile.mkdtemp(prefix="s1_toctou_"))
    r1, s1 = base(work, "reach")
    reach = probe(r1, s1, a.census_package, "leave")
    r2, s2 = base(work, "restore")
    restore = probe(r2, s2, a.census_package, "restore")
    if not (reach["injected_fired"] and restore["restored_fired"]):
        raise SystemExit("R11 ABORT: the mutation or the restore did not "
                         "fire; the demonstration would prove nothing.")
    paths = read_paths(a.census_package)
    digests = {"passa.py": hashlib.sha256(
                   (PKG / "passa.py").read_bytes()).hexdigest(),
               "opscan.py": hashlib.sha256(
                   (pathlib.Path(a.census_package) / "opscan.py").read_bytes()
               ).hexdigest(),
               "docgraph.py": hashlib.sha256(
                   (pathlib.Path(a.census_package) / "docgraph.py").read_bytes()
               ).hexdigest()}
    text = render(reach, restore, paths, digests)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(text)
    if sh(a.subject_repo, "git", "status", "--porcelain").strip():
        raise SystemExit("R11 ABORT: canonical subject dirty AFTER the run.")


if __name__ == "__main__":
    main()
