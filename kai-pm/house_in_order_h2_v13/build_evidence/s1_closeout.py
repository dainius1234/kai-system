#!/usr/bin/env python3
"""S1 EVIDENCE CLOSE-OUT. EVIDENCE ONLY. NO PRODUCTION CHANGE.

The O3 production mechanism is ACCEPTED and is not touched by this
instrument. Nothing here edits, redesigns or re-runs the corpus; it
closes the eight evidence questions Kai listed and reports what it
measures, including anything that does not close.

WHY EACH SECTION IS SHAPED THE WAY IT IS

1  PREDECESSOR CANONICAL-PATH DIGEST (DeepSeek B1). The O3 result's one
   red check was diagnosed as a harness artefact: `census_dependency`
   `.package` is recorded as `str(a.census_package)` -- the caller's own
   argument string -- and the evaluator pointed the predecessor at a
   TEMP directory. The diagnosis is credible and it is still only a
   diagnosis, so it is tested rather than asserted.

   The pre-O3 lineage is extracted with `git archive`, which writes
   nothing into `.git` and needs no worktree registration, at the
   HISTORICAL CANONICAL RELATIVE LOCATIONS -- `kai-pm/...census_v11`
   beside `kai-pm/...h2_v13` -- and the predecessor is executed from
   there.

   THE RECORDED PATH STRING IS AN UNRECORDED PARAMETER, so it is
   ENUMERATED, not guessed and not fitted. The candidate set is fixed in
   this file BEFORE the run, drawn from the spelling actually banked in
   the v1.2 payload (`../house_in_order_census_v11`) and the obvious
   spellings of the same location. EVERY candidate's digest is reported
   whether or not it matches. The digest is sha256 of the payload bytes:
   a match cannot be manufactured.

   The substitution method -- take the produced payload, replace only
   that one field, re-serialise exactly as passa.py does -- is
   CALIBRATED against a real run first (I-8 known-positive), and the
   claim that the field is the ONLY degree of freedom is MEASURED by
   running the predecessor twice under different spellings and diffing
   the payloads, not assumed.

2  FROZEN GIT-OBJECT MECHANISM (B2). Asserted against the production
   SOURCE by AST, not by quoting a comment. A docstring saying "from git
   object storage" is not evidence that the code does it.

3  READER-GUARD CAN-FAIL (B3). The banked guard asserts the reader set
   is exactly {passa}. A guard demonstrated only where it passes proves
   nothing, so a probe-only bypass reader is introduced into a TEMPORARY
   COPY of the package and the guard must FAIL on it. Production is not
   modified; the copy is deleted.

4  ENUMERATION BINDING. Whether the census populations are still derived
   from git, measured from source and at runtime, including a repo-wide
   search for the one untracked-enumerating function in the package.

5  H2 PRODUCTION CALLER PROOF. Runtime capture of the `read_source`
   argument the census actually receives when production Pass A runs --
   an AST check alone would not prove the call executes.

6  STANDALONE None CONTRACT. Every call site in the repository is
   enumerated and classified. Superseded lineages are reported as what
   they are and are NOT claimed as covered.

7  DIGEST LINEAGE. Recomputed from git objects, not recalled.

8  SYMLINK + VERIFIED READER. P1 must refuse BEFORE the verifier can
   consume the path -- proved by observing that no frozen-blob batch and
   no census import happened, not by reading line numbers.

    python3 s1_closeout.py --subject-repo R --census-package C
"""
from __future__ import annotations
import argparse
import ast
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
H2_REL = "kai-pm/house_in_order_h2_v13"
CENSUS_REL = "kai-pm/house_in_order_census_v11"
HISTORICAL_PASSA_DIGEST = \
    "24fb1f555560277fd4555087f22ed06ef4a72efee845a339bbc87b925730f51c"
HISTORICAL_AGGREGATE = \
    "eb7aad7c1a565cb25fcf6a7e250133e95d210f3e8ceb8765489046e3d945fa0e"
O3_PASSA_DIGEST_PREFIX = "755f8b39a21aa586"
O3_AGGREGATE_PREFIX = "29064d65"

# Fixed BEFORE the run. The first is the historical canonical relative
# location; the third is the spelling actually banked in the v1.2
# payload. Every one of these is reported.
CANDIDATE_PACKAGE_STRINGS = [
    CENSUS_REL,
    f"./{CENSUS_REL}",
    "../house_in_order_census_v11",
    f"{CENSUS_REL}/",
    str(REPO_ROOT / CENSUS_REL),
    f"{REPO_ROOT}/{CENSUS_REL}/",
]

OUT, FAILED = [], []


def A(s=""):
    OUT.append(s)


def check(name, got, want, extra=""):
    ok = got == want
    if not ok:
        FAILED.append(name)
    A(f"   {'OK  ' if ok else '<<< '}{name:<58}{str(got)[:44]}")
    if not ok:
        A(f"        EXPECTED: {str(want)[:96]}")
    if extra:
        A(f"        {extra}")
    return ok


def sh(cwd, *args):
    r = subprocess.run(args, cwd=str(cwd), capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(args)}: "
                         f"{r.stderr.decode(errors='replace')[:300]}")
    return r.stdout.decode(errors="replace")


def blob(ref_path):
    r = subprocess.run(["git", "-C", str(REPO_ROOT), "show", ref_path],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git show {ref_path} failed")
    return r.stdout


def sha(b):
    return hashlib.sha256(b).hexdigest()


def digest_file(p):
    return sha(pathlib.Path(p).read_bytes())


def run_ok(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=None if cwd is None else str(cwd),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: {' '.join(map(str, cmd))}\n"
                         f"{r.stdout[-500:]}\n{r.stderr[-500:]}")
    return r.stdout


# ── fixtures ──────────────────────────────────────────────────────────
DOC = "# Alpha\n\n**Last updated:** 2026-01-01\n"
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
        extra(root)
    sh(root, "git", "init", "-q", ".")
    sh(root, "git", "add", "-A")
    sh(root, "git", "-c", "user.email=s1@x", "-c", "user.name=s1",
       "commit", "-qm", "subject")
    return root, sh(root, "git", "rev-parse", "HEAD").strip()


# ── 1. predecessor canonical-path digest ──────────────────────────────
def extract_lineage(work):
    """The pre-O3 lineage at its historical canonical relative layout.

    `git archive` reads objects and writes only into the destination --
    no worktree is registered, nothing in .git is touched, and the
    current checkout is not disturbed.
    """
    dst = work / "pre_o3"
    dst.mkdir()
    tar = work / "pre_o3.tar"
    with open(tar, "wb") as fh:
        r = subprocess.run(["git", "-C", str(REPO_ROOT), "archive",
                            "--format=tar", PRED, H2_REL, CENSUS_REL],
                           stdout=fh, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git archive {PRED} failed: "
                         f"{r.stderr.decode(errors='replace')[:200]}")
    subprocess.run(["tar", "-xf", str(tar)], cwd=str(dst), check=True)
    return dst


def section1(a, work):
    A("1. PREDECESSOR CANONICAL-PATH DIGEST  (DeepSeek B1)")
    A("   The one red check in the O3 result. Tested, not asserted.")
    A("")
    dst = extract_lineage(work)
    h2, cen = dst / H2_REL, dst / CENSUS_REL
    A(f"   lineage        {PRED} extracted with git archive, no worktree "
      f"registered")
    A(f"   layout         {H2_REL}")
    A(f"                  {CENSUS_REL}   (historical canonical relative "
      f"location)")
    A("")
    A("   1a. THE EXTRACTED LINEAGE IS THE HISTORICAL ONE")
    check("extracted passa.py == the ba258b8 blob",
          digest_file(h2 / "passa.py"), sha(blob(f"{PRED}:{H2_REL}/passa.py")))
    check("extracted docgraph.py == the ba258b8 blob",
          digest_file(cen / "docgraph.py"),
          sha(blob(f"{PRED}:{CENSUS_REL}/docgraph.py")))
    check("extracted opscan.py == the ba258b8 blob",
          digest_file(cen / "opscan.py"),
          sha(blob(f"{PRED}:{CENSUS_REL}/opscan.py")))
    agg = digest_file(cen / "MANIFEST.sha256")
    check("extracted census aggregate == the historical aggregate",
          agg, HISTORICAL_AGGREGATE)
    check("extracted passa.py DIFFERS from production (something is "
          "being compared)",
          digest_file(h2 / "passa.py") != digest_file(PKG / "passa.py"), True)
    A("")

    subject = sh(a.subject_repo, "git", "rev-parse", "HEAD").strip()
    A("   1b. REAL RUNS OF THE PREDECESSOR FROM THE EXTRACTED LINEAGE")
    A("   Two spellings of the SAME location, both executed for real, so")
    A("   the degrees of freedom are measured rather than assumed.")
    runs = {}
    for label, cwd, arg in (
            ("repo-root relative", dst, CENSUS_REL),
            ("package relative", h2, "../house_in_order_census_v11")):
        out = work / f"pred_{label.split()[0]}.json"
        run_ok([sys.executable, str(h2 / "passa.py"),
                "--subject-repo", a.subject_repo,
                "--history-repo", a.subject_repo, "--subject", subject,
                "--census-package", arg, "--out", str(out)], cwd=cwd)
        runs[arg] = (out, digest_file(out))
        A(f"   {label:<22}--census-package {arg:<34}"
          f"{digest_file(out)[:16]}")
    A("")
    (o1, d1), (o2, d2) = runs[CENSUS_REL], runs["../house_in_order_census_v11"]
    p1, p2 = json.loads(o1.read_text()), json.loads(o2.read_text())
    diff = sorted(k for k in set(p1) | set(p2) if p1.get(k) != p2.get(k))
    check("the ONLY field that differs between the two runs",
          diff, ["census_dependency"])
    check("  and inside it, only `package`",
          sorted(k for k in p1["census_dependency"]
                 if p1["census_dependency"][k] != p2["census_dependency"][k]),
          ["package"])
    check("both runs carry the historical census aggregate",
          (p1["census_dependency"]["aggregate"],
           p2["census_dependency"]["aggregate"]),
          (HISTORICAL_AGGREGATE, HISTORICAL_AGGREGATE))
    A("")

    A("   1c. THE SUBSTITUTION METHOD, CALIBRATED BEFORE IT IS USED (I-8)")

    def substitute(payload, s):
        p = json.loads(json.dumps(payload))
        p["census_dependency"]["package"] = s
        return sha(json.dumps(p, indent=1).encode())

    check("known-positive: substituting the string a real run ACTUALLY "
          "used", substitute(p1, CENSUS_REL), d1,
          "so the re-serialisation reproduces passa.py's own bytes exactly")
    check("known-negative: a different string gives a different digest",
          substitute(p1, "SOMETHING/ELSE") != d1, True)
    check("cross-check: substituting run 1 into run 2's spelling gives "
          "run 2", substitute(p1, "../house_in_order_census_v11"), d2)
    A("")

    A("   1d. EVERY CANDIDATE SPELLING, ALL REPORTED")
    A(f"   target (historical) {HISTORICAL_PASSA_DIGEST}")
    hits = []
    for s in CANDIDATE_PACKAGE_STRINGS:
        d = substitute(p1, s)
        hit = d == HISTORICAL_PASSA_DIGEST
        if hit:
            hits.append(s)
        A(f"   {'>>> MATCH' if hit else '         '}  "
          f"{s:<52}{d[:32]}")
    A("")
    check("exactly one candidate spelling reproduces the historical "
          "digest", len(hits), 1,
          f"reproducing spelling: {hits[0] if hits else 'NONE'}")
    if hits:
        real = hits[0] in runs
        A("   WHAT IS ESTABLISHED, IN THE EXACT WORDS THE EVIDENCE EARNS")
        A(f"   The pre-O3 lineage {PRED}, extracted from git objects at the")
        A("   historical canonical relative layout and EXECUTED for real")
        A("   against the frozen subject, produces a payload that is")
        A("   byte-identical to the historical artefact")
        A(f"   {HISTORICAL_PASSA_DIGEST}")
        A(f"   when `census_dependency.package` carries `{hits[0]}`.")
        A("   Every other byte of the payload -- all 272 rows, every")
        A("   witness, the history identity, the census aggregate -- is")
        A("   produced by execution, not by substitution.")
        A("")
        A(f"   reached by a real end-to-end run: {'YES' if real else 'NO'}")
        if not real:
            A("   HOW THAT ONE FIELD WAS SET, STATED PLAINLY. The matching")
            A("   spelling is the CANONICAL ABSOLUTE path, and that path")
            A("   now holds the O3 census. Running the predecessor with it")
            A("   would read the O3 manifest and record the O3 aggregate,")
            A("   and placing historical content there would mean editing")
            A("   production, which is prohibited. So that single field was")
            A("   set by the substitution method of 1c -- CALIBRATED first")
            A("   against a real run's own string (known-positive), against")
            A("   a wrong string (known-negative), and across the two real")
            A("   runs. 1b MEASURED, by executing the predecessor twice,")
            A("   that this field is the only degree of freedom.")
            A("   CLAIM SCOPE (R17): the predecessor lineage reproduces the")
            A("   historical digest exactly, conditional on that recorded")
            A("   path string. NOT claimed: that a single end-to-end")
            A("   invocation reproduced it with nothing substituted.")
            A("   A sha256 preimage cannot be manufactured, and the")
            A("   candidate set was fixed in this file before the run.")
    else:
        A("   NO CANDIDATE REPRODUCED THE HISTORICAL DIGEST.")
        A("   Per Kai's instruction this is a STOP. Nothing is tuned, no")
        A("   further candidate is invented, and the close-out does not")
        A("   claim reproduction.")
    A("")
    return hits, dst


# ── 2. frozen git-object mechanism ────────────────────────────────────
def section2():
    A("2. THE EXACT FROZEN GIT-OBJECT MECHANISM  (DeepSeek B2)")
    A("   Asserted against the production SOURCE by AST. A docstring")
    A("   claiming git object storage is not evidence that it is used.")
    A("")
    src = (PKG / "passa.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)}
    fb, mvr = fns["_frozen_blobs"], fns["make_verified_reader"]
    fb_src = ast.unparse(fb)
    inner = [n for n in ast.walk(mvr)
             if isinstance(n, ast.FunctionDef) and n.name == "read_source"][0]
    rs_src = ast.unparse(inner)

    def literals(node):
        return [n.value for n in ast.walk(node)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]

    check("_frozen_blobs invokes `git cat-file --batch`",
          all(t in literals(fb) for t in ("cat-file", "--batch")), True,
          "argv literals, matched in the call -- not a comment")
    check("_frozen_blobs enumerates from `git ls-tree -r` at the subject",
          all(t in literals(fb) for t in ("ls-tree", "-r", "--name-only")),
          True)
    check("_frozen_blobs performs NO filesystem read",
          [n for n in ast.walk(fb) if isinstance(n, ast.Attribute)
           and n.attr in ("read_text", "read_bytes", "open", "iterdir",
                          "glob", "rglob")], [],
          "the frozen reference cannot be a live path: there is no read")
    check("_frozen_blobs references no filesystem path constructor",
          "pathlib.Path" in fb_src, False)
    check("the verifier hashes the CONSUMED bytes",
          "hashlib.sha256(data).hexdigest()" in rs_src, True)
    check("the verifier hashes the FROZEN BLOB BYTES the same way",
          "hashlib.sha256(blob).hexdigest()" in rs_src, True)
    check("the comparison is hash vs hash",
          any(isinstance(n, ast.Compare)
              and ast.unparse(n) in ("got != want", "want != got")
              for n in ast.walk(inner)), True)
    check("no git object id is obtained anywhere in the verifier",
          [s for s in literals(mvr)
           if s in ("rev-parse", "hash-object", "cat-file")], [],
          "`git cat-file --batch` is called once, in _frozen_blobs, and "
          "its BLOB CONTENT is what is compared -- never its object id")
    check("the reader reads before it verifies, and returns only after",
          [ast.unparse(n)[:40] for n in inner.body
           if isinstance(n, ast.Return)], ["return data.decode(errors='ignore')"],
          "READ -> VERIFY THOSE BYTES -> USE: the object hashed is the "
          "object returned, so change-and-restore cannot substitute")
    A("")
    A("   RECORDED, as required:")
    A("     · frozen content comes from GIT OBJECT STORAGE, via")
    A("       `git cat-file --batch` fed from `git ls-tree -r` at the")
    A("       frozen subject commit;")
    A("     · the comparison is SHA-256(consumed bytes) against")
    A("       SHA-256(frozen git blob bytes), both computed by hashlib in")
    A("       the same expression pair;")
    A("     · NO live filesystem path is used as the frozen reference --")
    A("       _frozen_blobs contains no read of any kind;")
    A("     · NO ordinary file SHA-256 is compared against a git object")
    A("       id. Object ids are never obtained in the verifier.")
    A("")


# ── 3. reader-guard can-fail ──────────────────────────────────────────
GUARD_PROBE = r"""
import hashlib, json, pathlib, sys
pkg, census, root, subject = sys.argv[1:5]
sys.path.insert(0, pkg)
log = []
rt, rb = pathlib.Path.read_text, pathlib.Path.read_bytes
def note(self, data):
    caller = sys._getframe(2).f_globals.get("__name__", "?")
    log.append({"reader": caller, "file": self.name})
def h_text(self, *a, **k):
    out = rt(self, *a, **k); note(self, out.encode()); return out
def h_bytes(self, *a, **k):
    out = rb(self, *a, **k); note(self, out); return out
pathlib.Path.read_text, pathlib.Path.read_bytes = h_text, h_bytes
import passa as PA
res = {"raised": None, "rows": 0}
try:
    rows, tracked = PA.build(root, root, subject, pathlib.Path(census))
    res["rows"] = len(rows)
except SystemExit as e:
    res["raised"] = str(e)
except Exception as e:
    res["raised"] = f"{type(e).__name__}: {e}"
pathlib.Path.read_text, pathlib.Path.read_bytes = rt, rb
res["readers"] = sorted({e["reader"] for e in log
                         if e["file"].endswith((".md", ".py"))})
json.dump(res, sys.stdout)
"""

BYPASS_MODULE = '''"""PROBE ONLY. Never in production. A direct filesystem reader
outside the authorised verified-reader path, introduced solely to prove
the reader guard can detect one."""
import pathlib


def peek(repo, rel):
    return (pathlib.Path(repo) / rel).read_text(errors="ignore")
'''


def guard_verdict(readers):
    """THE GUARD, stated once and applied to both cases."""
    return sorted(r for r in readers if r != "passa")


def section3(a, work):
    A("3. READER-GUARD CAN-FAIL  (DeepSeek B3)")
    A("   The guard: every filesystem consumption on the Pass A path must")
    A("   be attributed to `passa`, because that is where the verified")
    A("   reader lives. Any other module is an uncovered bypass.")
    A("   PROBE ONLY -- the bypass reader is added to a TEMPORARY COPY.")
    A("")

    def probe(pkg, census, root, subject):
        r = subprocess.run([sys.executable, "-c", GUARD_PROBE, str(pkg),
                            str(census), str(root), subject],
                           capture_output=True, text=True)
        if not r.stdout:
            raise SystemExit(f"R11 ABORT: guard probe produced nothing: "
                             f"{r.stderr[-400:]}")
        return json.loads(r.stdout)

    root, subj = base(work, "guard_ok")
    good = probe(PKG, a.census_package, root, subj)
    check("KNOWN-NEGATIVE: production run completes", good["raised"], None,
          f"{good['rows']} rows")
    check("KNOWN-NEGATIVE: reader set", good["readers"], ["passa"])
    check("KNOWN-NEGATIVE: the guard PASSES", guard_verdict(good["readers"]),
          [])
    A("")

    # A temporary copy of the package with ONE probe-only bypass reader.
    tmp_pkg = work / "bypass_pkg"
    shutil.copytree(PKG, tmp_pkg, ignore=shutil.ignore_patterns(
        "__pycache__", "build_evidence"))
    (tmp_pkg / "bypass_reader.py").write_text(BYPASS_MODULE)
    src = (tmp_pkg / "passa.py").read_text()
    anchor = "    tracked = G.tracked_md(subject_repo)"
    if anchor not in src:
        raise SystemExit("R11 ABORT: could not locate the injection anchor "
                         "in the temporary copy; the bypass would not be "
                         "exercised and the control would prove nothing.")
    src = src.replace(anchor, anchor + (
        "\n    import bypass_reader as _BP        # PROBE ONLY"
        "\n    _BP.peek(subject_repo, tracked[0])"), 1)
    (tmp_pkg / "passa.py").write_text(src)
    check("the temporary copy DIFFERS from production",
          digest_file(tmp_pkg / "passa.py") != digest_file(PKG / "passa.py"),
          True)
    check("production passa.py is untouched by this section",
          digest_file(PKG / "passa.py"), PROD_PASSA_DIGEST)

    root, subj = base(work, "guard_fail")
    bad = probe(tmp_pkg, a.census_package, root, subj)
    check("KNOWN-POSITIVE: the bypassing run still completes",
          bad["raised"], None,
          "the bypass reads unverified bytes and nothing stops it -- "
          "which is exactly why the guard has to detect it")
    check("KNOWN-POSITIVE: the bypass appears in the reader set",
          bad["readers"], ["bypass_reader", "passa"])
    check("KNOWN-POSITIVE: the guard FAILS, naming the bypass",
          guard_verdict(bad["readers"]), ["bypass_reader"],
          "the guard is not vacuous: it detects a future direct reader "
          "introduced outside the verified path")
    shutil.rmtree(tmp_pkg, ignore_errors=True)
    A("")


# ── 4. enumeration binding ────────────────────────────────────────────
WALKERS = ("rglob", "glob", "walk", "iterdir", "listdir", "scandir")


def section4(a, work, pre_o3):
    A("4. ENUMERATION BINDING — ARE THE CENSUS POPULATIONS STILL GIT-DERIVED")
    A("   Enumeration is NOT altered by this instrument. It is measured.")
    A("")
    cen = pathlib.Path(a.census_package)
    for mod, fname, want_argv in (
            ("docgraph", "tracked_md", ("ls-tree", "-r", "--name-only")),
            ("opscan", "tracked", ("ls-tree", "-r", "--name-only")),
            ("opscan", "source_population", None)):
        tree = ast.parse((cen / f"{mod}.py").read_text())
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == fname][0]
        lits = [n.value for n in ast.walk(fn)
                if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        walk = sorted({n.attr for n in ast.walk(fn)
                       if isinstance(n, ast.Attribute) and n.attr in WALKERS})
        check(f"{mod}.{fname} walks no filesystem", walk, [])
        if want_argv:
            check(f"{mod}.{fname} enumerates via git {' '.join(want_argv)}",
                  all(t in lits for t in want_argv), True)
    A("")
    A("   4a. THE WHOLE PACKAGE, NOT ONLY THOSE THREE FUNCTIONS")
    A("   MY FIRST PREDICATE HERE WAS WRONG AND THE CORRECTION STAYS")
    A("   VISIBLE. It matched the attribute NAME `walk` and reported the")
    A("   hits as filesystem enumeration. Eleven of them are `ast.walk`")
    A("   and `HERE.glob` -- AST traversal and the package reading its")
    A("   OWN directory. That is R13/doctrine 37 exactly: same name is a")
    A("   routing signature, not the same mechanism. The predicate below")
    A("   classifies by RECEIVER, and every hit is reported whatever its")
    A("   class.")
    hits, buckets = [], {"AST TRAVERSAL": [], "THE PACKAGE'S OWN DIRECTORY":
                         [], "FILESYSTEM, SUBJECT-REACHING": []}
    for p in sorted(cen.glob("*.py")):
        tree = ast.parse(p.read_text())
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Attribute) and n.attr in WALKERS):
                continue
            recv = ast.unparse(n.value)
            kind = ("AST TRAVERSAL" if recv == "ast"
                    else "THE PACKAGE'S OWN DIRECTORY" if recv == "HERE"
                    else "FILESYSTEM, SUBJECT-REACHING")
            buckets[kind].append(f"{p.name}:{n.lineno} {ast.unparse(n)}")
            hits.append((kind, p.name, n.lineno, ast.unparse(n)))
    A(f"   raw hits on the name predicate: {len(hits)} — ALL of them:")
    for kind, lines in buckets.items():
        A(f"     {kind} ({len(lines)})")
        for l in lines:
            A(f"       {l}")
    check("call sites that could enumerate SUBJECT filesystem content",
          buckets["FILESYSTEM, SUBJECT-REACHING"], [],
          "`HERE` is the census package's own directory (qualify.py:46), "
          "so its globs derive the CALIBRATION population from the tree "
          "(R5) and touch no subject content; qualify.py is not on the "
          "Pass A path at all, which imports docgraph, opscan and claims")
    A("")
    A("   4b. THE ONE UNTRACKED-ENUMERATING FUNCTION, AND WHETHER IT RUNS")
    calls = subprocess.run(["git", "-C", str(REPO_ROOT), "grep", "-n",
                            "untracked_md"], capture_output=True, text=True)
    lines = [l for l in calls.stdout.splitlines() if l.strip()]
    A(f"   `docgraph.untracked_md` uses `git ls-files --others`. Every")
    A(f"   occurrence in the repository ({len(lines)}), reported in full:")
    for l in lines:
        A(f"     {l[:112]}")
    callers = [l for l in lines if "def untracked_md" not in l]
    check("call sites of untracked_md anywhere in the repository",
          callers, [],
          "it is defined and never invoked, so it contributes to no "
          "population on any path")
    A("")
    A("   4c. RUNTIME — THE POPULATIONS AT THE SUBJECT ARE INSIDE THE TREE")
    sys.path.insert(0, str(cen))
    import importlib.util

    def load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    G = load(cen / "docgraph.py", "co_G")
    O = load(cen / "opscan.py", "co_O")
    docs = G.tracked_md(a.subject_repo)
    srcs = O.source_population(pathlib.Path(a.subject_repo))
    tracked = set(sh(a.subject_repo, "git", "ls-tree", "-r", "--name-only",
                     "HEAD").split())
    check("docgraph document population", len(docs), 272)
    check("opscan source population", len(srcs), 615)
    check("every enumerated path is inside the tracked frozen tree",
          sorted((set(docs) | set(srcs)) - tracked), [],
          f"{len(set(docs) | set(srcs))} of {len(tracked)} tracked paths")
    A("")
    A("   4d. UNCHANGED FROM THE PRE-O3 LINEAGE")
    pG = load(pre_o3 / CENSUS_REL / "docgraph.py", "pre_G")
    pO = load(pre_o3 / CENSUS_REL / "opscan.py", "pre_O")
    check("docgraph population identical to pre-O3",
          pG.tracked_md(a.subject_repo), docs)
    check("opscan population identical to pre-O3",
          pO.source_population(pathlib.Path(a.subject_repo)), srcs)
    A("")
    A("   CONCLUSION: both populations remain GIT-DERIVED from")
    A("   `git ls-tree -r` at the frozen tree. Neither walks arbitrary")
    A("   untracked filesystem content. No STOP condition arises.")
    A("")


# ── 5 & 6. caller proofs ──────────────────────────────────────────────
CALLER_PROBE = r"""
import json, pathlib, sys
pkg, census, root, subject = sys.argv[1:5]
sys.path.insert(0, pkg); sys.path.insert(0, census)
import docgraph as G, opscan as O
seen = {}
_bg, _co = G.build_graph, O.collect
def bg(repo, docs, read_source=None):
    seen["build_graph"] = read_source is not None
    seen["build_graph_name"] = getattr(read_source, "__qualname__", None)
    return _bg(repo, docs, read_source=read_source)
def co(repo, docs, read_source=None):
    seen["collect"] = read_source is not None
    seen["collect_name"] = getattr(read_source, "__qualname__", None)
    return _co(repo, docs, read_source=read_source)
G.build_graph, O.collect = bg, co
import passa as PA
out = {"raised": None, "rows": 0}
try:
    rows, tracked = PA.build(root, root, subject, pathlib.Path(census))
    out["rows"] = len(rows)
except SystemExit as e:
    out["raised"] = str(e)
except Exception as e:
    out["raised"] = f"{type(e).__name__}: {e}"
out.update(seen)
json.dump(out, sys.stdout)
"""


def section5(a, work):
    A("5. H2 PRODUCTION CALLER PROOF — THE VERIFIED READER IS SUPPLIED")
    A("   Captured at RUNTIME from the census side. An AST check alone")
    A("   would show a keyword in the text, not that the call executes.")
    A("")
    root, subj = base(work, "caller")
    r = subprocess.run([sys.executable, "-c", CALLER_PROBE, str(PKG),
                        str(a.census_package), str(root), subj],
                       capture_output=True, text=True)
    if not r.stdout:
        raise SystemExit(f"R11 ABORT: caller probe produced nothing: "
                         f"{r.stderr[-400:]}")
    res = json.loads(r.stdout)
    check("production Pass A completed", res["raised"], None,
          f"{res['rows']} rows")
    check("docgraph.build_graph RECEIVED a reader", res.get("build_graph"),
          True)
    check("  and it is the verifier from passa.py",
          res.get("build_graph_name"), "make_verified_reader.<locals>.read_source")
    check("opscan.collect RECEIVED a reader", res.get("collect"), True)
    check("  and it is the same verifier", res.get("collect_name"),
          "make_verified_reader.<locals>.read_source")
    A("")
    A("   5a. AND THE SOURCE AGREES — every census call site in the")
    A("       CURRENT H2 PRODUCTION PACKAGE passes read_source")
    A("       (build_evidence instruments are NOT production and are")
    A("        excluded here; they are listed in full in section 6)")
    sites = [s for s in call_sites(PKG) if "build_evidence" not in s["where"]]
    for s in sites:
        A(f"     {s['where']:<52}read_source={'YES' if s['rs'] else 'NO'}")
    check("current-production call sites without read_source",
          [s["where"] for s in sites if not s["rs"]], [])
    A("")


def call_sites(root):
    """Every build_graph / collect call under `root`, by AST."""
    out = []
    for p in sorted(pathlib.Path(root).rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        try:
            tree = ast.parse(p.read_text(errors="ignore"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            name = (n.func.attr if isinstance(n.func, ast.Attribute)
                    else getattr(n.func, "id", ""))
            if name not in ("build_graph", "collect"):
                continue
            if isinstance(n.func, ast.Attribute) and name == "collect" and \
                    ast.unparse(n.func.value) not in ("O", "opscan", "oldO",
                                                      "newO"):
                continue
            rel = str(p.relative_to(REPO_ROOT)) if REPO_ROOT in p.parents \
                else str(p)
            out.append({"where": f"{rel}:{n.lineno}", "rs": any(
                k.arg == "read_source" for k in n.keywords),
                "expr": ast.unparse(n)[:70]})
    return out


def section6(a, work):
    A("6. THE STANDALONE `read_source=None` CONTRACT")
    A("   The default is NOT removed. What is proved is that the callers")
    A("   which rely on it operate under their own frozen-subject")
    A("   contract, and that current H2 production is not among them.")
    A("")
    sites = call_sites(REPO_ROOT / "kai-pm")
    A(f"   Every build_graph/collect call site under kai-pm/ "
      f"({len(sites)}), reported in full:")
    for s in sites:
        A(f"     {'reader ' if s['rs'] else '  None '} {s['where']:<58}"
          f"{s['expr'][:44]}")
    A("")
    cur = [s for s in sites if s["where"].startswith(H2_REL + "/")
           and "build_evidence" not in s["where"]]
    check("call sites in the CURRENT production package", len(cur), 2)
    check("all of them supply a reader", [s["where"] for s in cur
                                          if not s["rs"]], [])
    none_sites = sorted({s["where"] for s in sites if not s["rs"]})
    A("   Call sites relying on the None default, named rather than")
    A("   summarised. They fall in three groups: the standalone census")
    A("   runner and its calibrators, the SUPERSEDED v1.1/v1.2 H2")
    A("   lineages, and this programme's own control instruments, which")
    A("   call without a reader deliberately to measure non-interference.")
    for w in none_sites:
        A(f"     {w}")
    A("")
    A("   6a. THE STANDALONE RUNNER'S OWN CONTRACT")
    rc = (pathlib.Path(a.census_package) / "run_census.py")
    tree = ast.parse(rc.read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    mat = ast.unparse(fns["materialise"])
    main = ast.unparse(fns["main"])
    check("run_census materialises the subject with `git archive`",
          "'archive'" in mat, True)
    check("  into a fresh directory, not the working tree",
          "mkdtemp" in main or "TemporaryDirectory" in main, True)
    check("  refusing anything that is not an immutable 40-hex object id",
          "40-hex" in mat, True)
    check("  and verifying the result against `git ls-tree`",
          "'ls-tree'" in mat, True)
    check("census() is called on the MATERIALISED path, not on `repo`",
          "census(subject)" in main, True)
    A("   So the None default is exercised only against a tree that git")
    A("   itself just wrote from the frozen commit and that was verified")
    A("   before measurement. That is a different -- and older --")
    A("   discharge of the same obligation, not an unguarded read.")
    A("")
    A("   6b. SCOPE OF THIS CLAIM (R17)")
    A("   MEASURED: every build_graph/collect call site under kai-pm/, by")
    A("   AST, plus the runtime capture in section 5.")
    A("   CLAIMED: no CURRENT H2 production path (house_in_order_h2_v13,")
    A("   excluding build_evidence instruments) reaches a census read with")
    A("   read_source=None.")
    A("   NOT CLAIMED: that the superseded v1.1/v1.2 H2 lineages listed")
    A("   above are covered. They are not. They are prior packages that")
    A("   remain in the repository and predate this mechanism entirely.")
    A("")


# ── 7. digest lineage ─────────────────────────────────────────────────
def section7(a, pre_o3):
    A("7. DIGEST LINEAGE — TWO IMMUTABLE ARTEFACTS, TWO CODE STATES")
    A("")
    pre_agg = digest_file(pre_o3 / CENSUS_REL / "MANIFEST.sha256")
    now_agg = digest_file(pathlib.Path(a.census_package) / "MANIFEST.sha256")
    check("pre-O3 census aggregate, recomputed from the git objects",
          pre_agg, HISTORICAL_AGGREGATE)
    check("O3 census aggregate, recomputed from the working tree",
          now_agg[:8], O3_AGGREGATE_PREFIX)
    check("the two aggregates differ", pre_agg != now_agg, True)
    A("")
    A("   PRE-O3 LINEAGE")
    A(f"     code            {PRED} (S1 R4+P1, accepted)")
    A(f"     census aggregate{'':<2}{HISTORICAL_AGGREGATE}")
    A(f"     Pass A digest   {HISTORICAL_PASSA_DIGEST}")
    A("   O3 LINEAGE")
    A(f"     code            bd1cbb4 (S1 O3, accepted)")
    A(f"     census aggregate{'':<2}{now_agg}")
    A(f"     Pass A digest   {O3_PASSA_DIGEST_PREFIX}…")
    A("")
    A("   THE NEW RESULT DOES NOT OVERWRITE, CORRECT OR RETROACTIVELY")
    A("   ALTER THE HISTORICAL ARTEFACT. Each digest is the fingerprint of")
    A("   a payload produced by a different code state against the same")
    A("   frozen subject. Both remain immutable lineage evidence. The")
    A("   historical E2 source-byte proof stands on the pre-O3 artefact")
    A("   and is re-derivable from the git objects, which section 1 does.")
    A("")


# ── 8. symlink + verified reader ──────────────────────────────────────
SYMLINK_PROBE = r"""
import json, pathlib, subprocess, sys
pkg, census, root, subject = sys.argv[1:5]
sys.path.insert(0, pkg)
import passa as PA
calls = {"frozen_blobs": 0, "reader_made": 0}
_fb, _mv = PA._frozen_blobs, PA.make_verified_reader
def fb(*a, **k):
    calls["frozen_blobs"] += 1
    return _fb(*a, **k)
def mv(*a, **k):
    calls["reader_made"] += 1
    return _mv(*a, **k)
PA._frozen_blobs, PA.make_verified_reader = fb, mv
out = {"raised": None, "rows": 0}
try:
    rows, tracked = PA.build(root, root, subject, pathlib.Path(census))
    out["rows"] = len(rows)
except SystemExit as e:
    out["raised"] = str(e)
except Exception as e:
    out["raised"] = f"{type(e).__name__}: {e}"
out.update(calls)
out["census_imported"] = "docgraph" in sys.modules
json.dump(out, sys.stdout)
"""


def section8(a, work):
    A("8. SYMLINK (P1) + VERIFIED READER REGRESSION")
    A("   P1 must refuse BEFORE the verifier can consume the path.")
    A("   Ordering is proved by observing that the frozen-blob batch never")
    A("   ran and the census was never imported -- not from line numbers.")
    A("")

    def probe(root, subj):
        r = subprocess.run([sys.executable, "-c", SYMLINK_PROBE, str(PKG),
                            str(a.census_package), str(root), subj],
                           capture_output=True, text=True)
        if not r.stdout:
            raise SystemExit(f"R11 ABORT: symlink probe produced nothing: "
                             f"{r.stderr[-400:]}")
        return json.loads(r.stdout)

    def add_symlink(root):
        (root / "docs" / "link.md").symlink_to("alpha.md")

    root, subj = base(work, "symlink", extra=add_symlink)
    mode = sh(root, "git", "ls-files", "-s", "docs/link.md").split()[0]
    check("the fixture really contains a TRACKED symlink", mode,
          PA.GIT_SYMLINK_MODE)
    res = probe(root, subj)
    check("Pass A REFUSED", res["raised"] is not None, True,
          str(res["raised"])[:110])
    check("  naming the symlink prerequisite",
          "TRACKED SYMLINK" in str(res["raised"]), True)
    check("  naming the offending path", "docs/link.md" in str(res["raised"]),
          True)
    check("no rows were produced", res["rows"], 0)
    check("the frozen-blob batch NEVER RAN", res["frozen_blobs"], 0,
          "so the refusal precedes the verifier, at runtime")
    check("no verified reader was ever constructed", res["reader_made"], 0)
    check("the census was never imported", res["census_imported"], False)
    A("")
    A("   KNOWN-NEGATIVE — the same fixture without the symlink")
    root2, subj2 = base(work, "symlink_neg")
    ok = probe(root2, subj2)
    check("completes", ok["raised"], None, f"{ok['rows']} rows")
    check("and THERE the verifier is constructed", ok["reader_made"], 1,
          "so section 8's zeros are caused by the refusal, not by the probe")
    A("")


PROD_PASSA_DIGEST = ""


def main():
    global PROD_PASSA_DIGEST
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--out", default=str(HERE / "S1_CLOSEOUT.txt"))
    a = ap.parse_args()
    PROD_PASSA_DIGEST = digest_file(PKG / "passa.py")

    work = pathlib.Path(tempfile.mkdtemp(prefix="s1_closeout_"))
    A("S1 EVIDENCE CLOSE-OUT. PRODUCED BY ORION.")
    A("ZERO ADJUDICATION WEIGHT. NOT AN ANSWER KEY. NO TUNING TARGET.")
    A("EVIDENCE ONLY — no production file is modified by this instrument.")
    A("")
    A("0. FINGERPRINT")
    for k, v in (
        ("repository HEAD", sh(REPO_ROOT, "git", "rev-parse", "HEAD").strip()),
        ("O3 implementation", "bd1cbb4b76790339c7da59829b4019621c09f17f"),
        ("O3 result", "864993292e667cd98c9c3bb1b59125378b113ce1"),
        ("pre-O3 lineage", PRED),
        ("subject commit", sh(a.subject_repo, "git", "rev-parse",
                              "HEAD").strip()),
        ("git version", sh(a.subject_repo, "git", "--version").strip()),
        ("gate command", "git " + " ".join(PA.CLEAN_TREE_CMD)),
        ("symlink policy", f"P1_REJECT mode {PA.GIT_SYMLINK_MODE}"),
        ("digest[passa.py]", PROD_PASSA_DIGEST),
        ("digest[census/docgraph.py]",
         digest_file(pathlib.Path(a.census_package) / "docgraph.py")),
        ("digest[census/opscan.py]",
         digest_file(pathlib.Path(a.census_package) / "opscan.py")),
        ("digest[this instrument]", digest_file(__file__)),
    ):
        A(f"   {k:<28}{v}")
    A("")

    hits, pre_o3 = section1(a, work)
    if not hits:
        A("STOPPING HERE. Kai's instruction on B1 is explicit: if the")
        A("historical digest does not reproduce, STOP. Sections 2-8 are")
        A("NOT RUN, and nothing about them is claimed.")
        finish(a, work)
    section2()
    section3(a, work)
    section4(a, work, pre_o3)
    section5(a, work)
    section6(a, work)
    section7(a, pre_o3)
    section8(a, work)
    finish(a, work)


def finish(a, work):
    A("OUTCOME")
    A(f"   checks {len([l for l in OUT if l.startswith(('   OK  ', '   <<< '))])}"
      f"   failures {len(FAILED)}")
    if FAILED:
        A("   REPORTED, NOT REPAIRED:")
        for f in FAILED:
            A(f"     <<< {f}")
    else:
        A("   Every close-out check held. No production file was modified,")
        A("   no evaluator was tuned and no figure was used as a target.")
    text = "\n".join(OUT) + "\n"
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(text)
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
