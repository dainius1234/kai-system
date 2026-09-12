#!/usr/bin/env python3
"""CALIBRATION — document reference graph, kinds and edge contexts.

Generated crossings plus metamorphic assertions. Expected answers derive
from how each fixture is CONSTRUCTED (I-8).

P10 is the property under test: context is RECORDED but never allowed to
change an edge's kind or drop it.
"""
from __future__ import annotations
import itertools
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import caltrace as ct
import docgraph as G

A_KIND = "docgraph.KINDS"
A_CTX = "docgraph.EDGE_CONTEXTS"


def mkrepo(root, files, ignored=None):
    for rel, body in list(files.items()) + list((ignored or {}).items()):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    if ignored:
        (root / ".gitignore").write_text(
            "\n".join(sorted({r.split("/")[0] for r in ignored})) + "\n")
    subprocess.run(["git", "init", "-q"], cwd=root)
    subprocess.run(["git", "add", "-A"], cwd=root)
    subprocess.run(["git", "-c", "user.email=c@x", "-c", "user.name=c",
                    "commit", "-q", "-m", "f"], cwd=root)


def wrap(ref, context):
    """Construction defines the expected context label."""
    return {
        "PROSE_PATH":  f"see {ref} for detail\n",
        "INLINE_CODE": f"run `{ref}` now\n",
        "FENCED_CODE": f"```\n{ref}\n```\n",
        "TABLE":       f"| a | b |\n|---|---|\n| x | {ref} |\n",
        "QUOTE":       f"> quoting {ref} here\n",
    }[context]


SYNTAX = ["markdown_link", "explicit_path", "bare_basename"]
AMBIG = ["unique", "duplicate"]
CONTEXT = ["PROSE_PATH", "INLINE_CODE", "FENCED_CODE", "TABLE", "QUOTE"]
EXISTS = [True, False]


def crossings():
    n = 0
    for syn, amb, ctx, ex in itertools.product(SYNTAX, AMBIG, CONTEXT, EXISTS):
        n += 1
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            files = {"src.md": "# src\n"}
            if amb == "unique":
                target, base = "t/only.md", "only.md"
                if ex:
                    files[target] = "# only\n"
            else:
                target, base = "t/README.md", "README.md"
                files["u/README.md"] = "# other\n"
                if ex:
                    files[target] = "# t readme\n"
            ref = {"markdown_link": f"[x]({target})",
                   "explicit_path": target,
                   "bare_basename": base}[syn]
            files["src.md"] = "# src\n" + wrap(ref, ctx)
            mkrepo(root, files)
            tracked = G.tracked_md(root)
            edges = [e for e in G.build_graph(root, tracked)
                     if e[0] == "src.md"]
            for e in edges:
                ct.observe(A_KIND, e[2])
                ct.observe(A_CTX, e[4])
            label = f"[{syn}/{amb}/{ctx}/exists={ex}]"

            if syn == "markdown_link":
                want = "MARKDOWN_LINK" if ex else "BROKEN_LINK"
                got = [e for e in edges if e[2] == want]
                ct.assert_value(f"{label} link -> {want}", bool(got),
                                A_KIND, want, str(edges))
                ct.assert_value(f"{label} link context is MARKDOWN_LINK",
                                bool(got) and all(e[4] == "MARKDOWN_LINK"
                                                  for e in got),
                                A_CTX, "MARKDOWN_LINK", str(got))
                if not ex:
                    ct.check(f"{label} broken link not retargeted",
                             all(e[1] is None for e in got), str(got))
            elif syn == "explicit_path":
                got = [e for e in edges if e[2] == "EXPLICIT_PATH_REFERENCE"]
                if ex:
                    ct.assert_value(f"{label} explicit path -> edge",
                                    len(got) == 1, A_KIND,
                                    "EXPLICIT_PATH_REFERENCE", str(edges))
                    ct.check(f"{label} resolves to the exact target",
                             bool(got) and got[0][1] == target, str(got))
                    ct.assert_value(f"{label} CONTEXT PRESERVED, not semantic",
                                    bool(got) and got[0][4] == ctx,
                                    A_CTX, ctx,
                                    f"want {ctx} got {got[0][4] if got else None}")
                else:
                    ct.check(f"{label} absent target -> no explicit edge",
                             not got, str(got))
            else:
                if amb == "duplicate" and ex:
                    got = [e for e in edges if e[2] == "BASENAME_AMBIGUOUS"]
                    ct.assert_value(f"{label} duplicate basename -> AMBIGUOUS",
                                    bool(got), A_KIND, "BASENAME_AMBIGUOUS",
                                    str(edges))
                    ct.check(f"{label} ambiguous attributed to nothing",
                             all(e[1] is None for e in got), str(got))
                    ct.assert_value(f"{label} ambiguous context preserved",
                                    all(e[4] == ctx for e in got),
                                    A_CTX, ctx, str(got))
                elif amb == "unique" and ex:
                    got = [e for e in edges if e[2] == "BASENAME_UNIQUE"]
                    ct.assert_value(f"{label} unique basename -> UNIQUE",
                                    bool(got), A_KIND, "BASENAME_UNIQUE",
                                    str(edges))
                    ct.assert_value(f"{label} unique basename ctx preserved",
                                    all(e[4] == ctx for e in got),
                                    A_CTX, ctx, str(got))
    return n


def pairs_for(files, ignored=None):
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        mkrepo(root, files, ignored)
        tracked = G.tracked_md(root)
        e = G.build_graph(root, tracked)
        return G.distinct_pairs(e), e, tracked


def metamorphic():
    base = {"src.md": "# s\nsee t/only.md here\n", "t/only.md": "# o\n"}
    p1, _, _ = pairs_for(base)

    two = dict(base); two["src.md"] += "also [x](t/only.md)\n"
    p2, e2, _ = pairs_for(two)
    ct.check("M1 second evidence kind does not increment distinct pairs",
             p1 == p2, f"{p1} vs {p2}")
    ct.check("M1b both evidence kinds still recorded",
             {k for _s, _d, k, _r, _c in e2} >= {"MARKDOWN_LINK",
                                                 "EXPLICIT_PATH_REFERENCE"},
             str(G.kind_tally(e2)))

    conv = {"src.md": "# s\nsee [t](t/only.md) here\n", "t/only.md": "# o\n"}
    p3, _, _ = pairs_for(conv)
    ct.check("M2 explicit->link preserves pair", p1 == p3, f"{p1} vs {p3}")

    steal = dict(base); steal["z/only.md"] = "# decoy\n"
    p4, _e4, _ = pairs_for(steal)
    ct.check("M3 duplicate basename does not steal exact-path reference",
             ("src.md", "t/only.md") in p4
             and ("src.md", "z/only.md") not in p4, str(sorted(p4)))

    gone = {"src.md": "# s\n[x](t/only.md)\n", "other.md": "# o\n"}
    _p5, e5, _ = pairs_for(gone)
    brk = [e for e in e5 if e[2] == "BROKEN_LINK"]
    ct.check("M4 deleted target -> BROKEN", bool(brk), str(e5))
    ct.check("M4b broken link not retargeted",
             all(e[1] is None for e in brk))

    _p6, _e6, t6 = pairs_for(base)
    _p7, _e7, t7 = pairs_for(base, ignored={"cache/junk.md": "# junk\n"})
    ct.check("M5 ignored .md does not change tracked denominator",
             t6 == t7, f"{t6} vs {t7}")

    rev = dict(reversed(list(base.items())))
    p8, _, _ = pairs_for(rev)
    ct.check("M6 enumeration order does not change pairs", p1 == p8,
             f"{p1} vs {p8}")

    dup = dict(base); dup["src.md"] += "see t/only.md again\nand t/only.md\n"
    p9, e9, _ = pairs_for(dup)
    ct.check("M7 duplicated mention does not change distinct pairs",
             p1 == p9, f"{p1} vs {p9}")
    ct.check("M7b raw evidence count DOES rise (views differ visibly)",
             G.kind_tally(e9)["EXPLICIT_PATH_REFERENCE"] > 1,
             str(G.kind_tally(e9)))


def alphabet_totality():
    """D341 F2: the EMITTED edge-context alphabet must be exactly what
    the module declares -- no dead value, no undeclared value."""
    seen = set()
    txt = "a\n> q\n| t |\n```\nf\n```\n`c` x\nplain x\n"
    lines, lctx = txt.splitlines(), G.line_contexts(txt)
    for pos in range(len(txt)):
        for is_link in (True, False):
            seen.add(G.edge_context(txt, lines, lctx, pos, is_link))
    ct.check("no edge context outside the declared alphabet",
             seen <= set(G.ALPHABETS["EDGE_CONTEXTS"]), str(sorted(seen)))
    ct.check("the internal OTHER line class never escapes as an edge class",
             "OTHER" not in seen and "OTHER" in G.LINE_CONTEXTS,
             str(sorted(seen)))


def run():
    n = crossings()
    metamorphic()
    alphabet_totality()
    return n


if __name__ == "__main__":
    ct.reset()
    n = run()
    print(f"cal_docgraph: {n} crossings; {ct.PASSED} passed, "
          f"{ct.FAILED} failed")
    for f in ct.FAILURES[:12]:
        print("  FAIL", f)
    sys.exit(1 if ct.FAILED else 0)
