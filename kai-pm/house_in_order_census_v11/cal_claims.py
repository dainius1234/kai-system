#!/usr/bin/env python3
"""CALIBRATION — dispositions, path syntax, exclusion witnesses, claims.

Expected answers derive from how each fixture is CONSTRUCTED, never from
the module under test (I-8).

The REMOTE_URI and ABSOLUTE corrections are calibrated as BOUNDARY
PAIRS, because a repair that only proves the corrected case can silently
destroy the property it was protecting. For each, the suite proves BOTH:

  the corrected side  -- URI-shaped / absolute syntax alone does NOT
                         exclude, so the operation stays COULD_REACH_T;
  the preserved side  -- the SAME family of target still IS excluded
                         when real evidence (a differing filename)
                         supports it.
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import caltrace as ct
import claims as C
import opscan as O

A_DISP = "claims.DISPOSITIONS"
A_SYN = "claims.PATH_SYNTAX"
A_TD = "claims.TARGET_DISPOSITIONS"
A_W = "claims.EXCLUSION_WITNESSES"
A_C = "claims.CLAIMS"


def mkop(frags, mode="W", dynamic=None, target=None):
    o = O.Op("fixture.sh", mode, "fixture")
    o.frags = list(frags)
    o.dynamic = (any(c in f for f in frags for c in "${}*")
                 if dynamic is None else dynamic)
    o.target = target
    return o


def td(frags, target, mode="W", dynamic=None, tracked=(), resolved=None):
    """`target` is the claim subject T; `resolved` is the op's own
    already-resolved target, which is a different thing."""
    d, w, detail = C.target_disposition(
        mkop(frags, mode, dynamic, resolved), target, set(tracked))
    ct.observe(A_TD, d)
    if w:
        ct.observe(A_W, w)
    return d, w, detail


# ── 1. PATH SYNTAX IS AN OBSERVATION ─────────────────────────────────
def syntax_cases():
    cases = [
        (["docs/a.md"], False, "REPO_RELATIVE"),
        (["/etc/passwd"], False, "ABSOLUTE"),
        (["https://example.com/a.md"], False, "URI_SYNTAX"),
        (["$OUT/a.md"], False, "SHELL_VARIABLE"),
        ([], True, "DYNAMIC_UNKNOWN"),
    ]
    for frags, dyn, want in cases:
        got = C.path_syntax(frags, dyn)
        ct.observe(A_SYN, got)
        ct.assert_value(f"path_syntax({frags}) == {want}", got == want,
                        A_SYN, want, f"got {got}")


# ── 2. THE TWO CORRECTED WITNESSES, AS BOUNDARY PAIRS ────────────────
def boundary_pairs():
    T = "docs/foo.md"

    # --- URI FAMILY -------------------------------------------------
    # corrected side: "://" alone must NOT exclude. Reasoned about as
    # the path string it is, this literal HAS the target as a suffix,
    # so no constructive witness exists. v1.0 excluded it on sight.
    d, w, _ = td(["https://example.com/docs/foo.md"], T)
    ct.assert_value(
        "URI syntax alone does NOT exclude (v1.0 excluded this)",
        d == "COULD_REACH_T", A_TD, "COULD_REACH_T", f"got {d}/{w}")
    # preserved side: same family, real evidence -> still excluded.
    d, w, _ = td(["https://example.com/bar.md"], T)
    ct.assert_value(
        "URI-shaped target with a DIFFERENT filename is still excluded",
        d == "EXCLUDED_FROM_T" and w == "FIXED_COMPLETE_PATH_DIFFERS",
        A_W, "FIXED_COMPLETE_PATH_DIFFERS", f"got {d}/{w}")

    # --- ABSOLUTE FAMILY (R6: same defect, different character) ------
    # corrected side: an absolute path can point INTO the repository.
    d, w, _ = td(["/home/user/repo/docs/foo.md"], T)
    ct.assert_value(
        "absolute path naming the target is NOT excluded (v1.0 excluded it)",
        d == "COULD_REACH_T", A_TD, "COULD_REACH_T", f"got {d}/{w}")
    # preserved side: /dev/null must still be excluded, and the witness
    # must not describe a device node as a repository directory (D332).
    d, w, detail = td(["/dev/null"], "README.md")
    ct.assert_value(
        "/dev/null is still excluded from a root-level target",
        d == "EXCLUDED_FROM_T" and w == "FIXED_COMPLETE_PATH_DIFFERS",
        A_W, "FIXED_COMPLETE_PATH_DIFFERS", f"got {d}/{w}")
    ct.check("exclusion witness does not call a device node a repo directory",
             "directory" not in (detail or ""), str(detail))

    # The concrete v1.0 defect, named: a hard-coded absolute path into
    # the repository. H2's pass_a.py contains exactly this shape.
    d, w, _ = td(["/home/user/kai-system/data/SOUL.md"], "data/SOUL.md")
    ct.check("absolute path INTO the repository is not falsely excluded",
             d == "COULD_REACH_T", f"got {d}/{w}")
    # ... and the same absolute prefix naming a DIFFERENT repo path is
    # still excluded, so the repair did not blind the witness.
    d, w, _ = td(["/home/user/kai-system/other/SOUL.md"], "data/SOUL.md")
    ct.check("same absolute prefix, different repo path, still excluded",
             d == "EXCLUDED_FROM_T", f"got {d}/{w}")


# ── 3. EVERY TARGET DISPOSITION AND WITNESS ──────────────────────────
def target_dispositions():
    T = "docs/foo.md"

    d, _w, _ = td(["docs/foo.md"], T, resolved="docs/foo.md")
    ct.assert_value("resolved exact target -> REACHES_T",
                    d == "REACHES_T", A_TD, "REACHES_T", f"got {d}")

    d, w, _ = td(["docs/bar.md"], T)
    ct.assert_value("different fixed filename -> COMPLETE_PATH witness",
                    d == "EXCLUDED_FROM_T"
                    and w == "FIXED_COMPLETE_PATH_DIFFERS",
                    A_W, "FIXED_COMPLETE_PATH_DIFFERS", f"got {d}/{w}")
    # The DISPOSITION itself needs an assertion of its own: crediting
    # only the witness left EXCLUDED_FROM_T reached-but-undiscriminated,
    # which is what the leg-3 gate caught on its first run.
    ct.assert_value("a constructively excluded write -> EXCLUDED_FROM_T",
                    d == "EXCLUDED_FROM_T", A_TD, "EXCLUDED_FROM_T",
                    f"got {d}")

    d, w, _ = td(["other/foo.md"], T)
    ct.assert_value("same filename, disjoint directory -> DIRECTORY witness",
                    d == "EXCLUDED_FROM_T"
                    and w == "FIXED_DIRECTORY_DISJOINT",
                    A_W, "FIXED_DIRECTORY_DISJOINT", f"got {d}/{w}")

    # A dynamic PREFIX cannot change a fixed final component.
    d, w, _ = td(["$OUT/bar.md"], T)
    ct.assert_value("dynamic prefix, fixed differing basename -> BASENAME",
                    d == "EXCLUDED_FROM_T"
                    and w == "FIXED_BASENAME_DIFFERS",
                    A_W, "FIXED_BASENAME_DIFFERS", f"got {d}/{w}")

    # ... but it CAN produce the target when the basename matches.
    d, w, _ = td(["$OUT/foo.md"], T)
    ct.assert_value("dynamic prefix, MATCHING basename -> COULD_REACH_T",
                    d == "COULD_REACH_T", A_TD, "COULD_REACH_T", f"got {d}/{w}")

    # A fully dynamic expression is never constructively excludable.
    d, w, _ = td(["$DEST"], T)
    ct.check("fully dynamic -> COULD_REACH_T", d == "COULD_REACH_T",
             f"got {d}/{w}")


# ── 4. DISPOSITIONS OVER A REAL FIXTURE REPOSITORY ───────────────────
FIXTURE_PY = '''
import pathlib, json
pathlib.Path("docs/a.md").write_text("x")          # RESOLVED_WRITE
pathlib.Path("docs/b.md").read_text()              # RESOLVED_READ
open("docs/a.md", "r+")                            # READ_AND_WRITE
pathlib.Path("out.json").write_text("{}")          # NON_DOCUMENT
pathlib.Path("nope/missing.md").write_text("x")    # UNRESOLVED_TARGET
import os
pathlib.Path(os.environ["DEST"]).write_text("x")   # UNRESOLVED_RELEVANCE
import tempfile
tmp = tempfile.mkdtemp()
pathlib.Path(str(tmp) + "/SOUL.md").write_text("x")  # D332 shape
(ROOT / "docs" / "b.md").read_text()                 # dynamic prefix read
'''


def mkrepo(root, files):
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    subprocess.run(["git", "init", "-q"], cwd=root)
    subprocess.run(["git", "add", "-A"], cwd=root)
    subprocess.run(["git", "-c", "user.email=c@x", "-c", "user.name=c",
                    "commit", "-q", "-m", "f"], cwd=root)


def dispositions():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        mkrepo(root, {"docs/a.md": "# a\n", "docs/b.md": "# b\n",
                      "SOUL.md": "# soul\n", "gen.py": FIXTURE_PY})
        docs = [p for p in O.tracked(root) if p.endswith(".md")]
        ops, acc = O.collect(root, docs)
        C.classify(ops, docs, set(O.tracked(root)))

        # RELEVANCE vs TARGET must not be conflated (MISBINDING).
        soul = [o for o in ops if o.frags and o.frags[-1].endswith("SOUL.md")]
        ct.check("D332 shape: a tmp-dir SOUL.md write is NOT attributed to "
                 "the tracked SOUL.md",
                 bool(soul) and all(o.target != "SOUL.md" for o in soul),
                 str([(o.frags, o.disposition, o.target) for o in soul]))
        ct.check("D332 shape: relevance is still PROVEN, so it is "
                 "UNRESOLVED_TARGET and not UNRESOLVED_RELEVANCE",
                 bool(soul) and all(o.disposition == "UNRESOLVED_TARGET"
                                    for o in soul),
                 str([(o.frags, o.disposition) for o in soul]))
        dynread = [o for o in ops if o.mode == "R" and o.frags
                   and o.frags[-1] == "b.md"]
        ct.check("dynamic-prefix read keeps its relevance "
                 "(UNRESOLVED_TARGET, never erased)",
                 bool(dynread) and all(o.disposition == "UNRESOLVED_TARGET"
                                       for o in dynread),
                 str([(o.frags, o.disposition) for o in dynread]))
        for o in ops:
            ct.observe(A_DISP, o.disposition)
        got = {o.disposition for o in ops}
        for want in C.ALPHABETS["DISPOSITIONS"]:
            ct.assert_value(f"fixture emits {want}", want in got,
                            A_DISP, want, str(sorted(got)))
        n, tally, s = C.account(ops)
        ct.check("P12 every operation has exactly one disposition",
                 n == s and n == acc["admitted_candidate_operations"],
                 f"{n} vs {s} vs {acc}")


# ── 5. EVERY CLAIM VALUE, INCLUDING THE SCOPED NEGATIVE ──────────────
def claim_values():
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        # only writer in the tree targets docs/a.md
        mkrepo(root, {"docs/a.md": "# a\n", "docs/b.md": "# b\n",
                      "gen.py": 'import pathlib\n'
                                'pathlib.Path("docs/a.md").write_text("x")\n'})
        docs = [p for p in O.tracked(root) if p.endswith(".md")]
        ta = set(O.tracked(root))
        ops, _acc = O.collect(root, docs)
        C.classify(ops, docs, ta)

        c, srcs, b, _w, scope = C.scoped_claim(ops, "docs/a.md", ta)
        ct.observe(A_C, c)
        ct.assert_value("proven writer -> PROVEN_WRITE_RELATION",
                        c == "PROVEN_WRITE_RELATION" and srcs == ["gen.py"],
                        A_C, "PROVEN_WRITE_RELATION", f"{c} {srcs} {b}")

        # docs/b.md: the ONLY write is constructively excluded ->
        # the scoped negative is emitted. Closure rules unchanged.
        c, _s, b, w, scope = C.scoped_claim(ops, "docs/b.md", ta)
        ct.observe(A_C, c)
        ct.assert_value(
            "all writes positively excluded -> NO_WRITER_WITHIN_ANALYZED_SCOPE",
            c == "NO_WRITER_WITHIN_ANALYZED_SCOPE"
            and b["COULD_REACH_T"] == 0,
            A_C, "NO_WRITER_WITHIN_ANALYZED_SCOPE", f"{c} {b} {w}")
        ct.check("the scoped negative carries its analysis scope",
                 isinstance(scope, dict) and "source_population" in scope
                 and "excluded_by_construction" in scope, str(scope))

        # one unexcludable dynamic write reopens the space.
        mkrepo2 = dict()
        with tempfile.TemporaryDirectory() as d2:
            r2 = pathlib.Path(d2)
            mkrepo(r2, {"docs/a.md": "# a\n", "docs/b.md": "# b\n",
                        "gen.py": 'import pathlib, os\n'
                                  'pathlib.Path(os.environ["D"]).write_text("x")\n'})
            docs2 = [p for p in O.tracked(r2) if p.endswith(".md")]
            ta2 = set(O.tracked(r2))
            ops2, _a2 = O.collect(r2, docs2)
            C.classify(ops2, docs2, ta2)
            c2, _s2, b2, _w2, _sc = C.scoped_claim(ops2, "docs/b.md", ta2)
            ct.observe(A_C, c2)
            ct.assert_value(
                "a single unexcludable dynamic write -> NO_PROVEN_WRITER",
                c2 == "NO_PROVEN_WRITER" and b2["COULD_REACH_T"] >= 1,
                A_C, "NO_PROVEN_WRITER", f"{c2} {b2}")
            ct.check("KNOWN-NEGATIVE: the scoped negative is NOT emitted "
                     "when the space is open",
                     c2 != "NO_WRITER_WITHIN_ANALYZED_SCOPE", str(c2))
        del mkrepo2


def run():
    syntax_cases()
    boundary_pairs()
    target_dispositions()
    dispositions()
    claim_values()


if __name__ == "__main__":
    ct.reset()
    run()
    print(f"cal_claims: {ct.PASSED} passed, {ct.FAILED} failed")
    for f in ct.FAILURES:
        print("  FAIL", f)
    sys.exit(1 if ct.FAILED else 0)
