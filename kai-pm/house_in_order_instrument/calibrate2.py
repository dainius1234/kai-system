#!/usr/bin/env python3
"""P10 CALIBRATION — generated crossings + metamorphic assertions.

Expected answers derive from how each fixture is CONSTRUCTED, never from
the collector (I-8).
"""
import itertools
import pathlib
import subprocess
import sys
import tempfile
import collections

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import doccensus2 as dc

P = F = 0
FAILS = []
def check(name, cond, detail=""):
    global P, F
    if cond:
        P += 1
    else:
        F += 1
        FAILS.append(f"{name} :: {detail}")


def mkrepo(root, files, ignored=None):
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    for rel, body in (ignored or {}).items():
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
    """Place `ref` in a given textual context. Construction defines the
    expected context label."""
    return {
        "PROSE_PATH":  f"see {ref} for detail\n",
        "INLINE_CODE": f"run `{ref}` now\n",
        "FENCED_CODE": f"```\n{ref}\n```\n",
        "TABLE":       f"| a | b |\n|---|---|\n| x | {ref} |\n",
        "QUOTE":       f"> quoting {ref} here\n",
    }[context]


# ── 1. GENERATED CROSSINGS ───────────────────────────────────────────
SYNTAX   = ["markdown_link", "explicit_path", "bare_basename"]
AMBIG    = ["unique", "duplicate"]
CONTEXT  = ["PROSE_PATH", "INLINE_CODE", "FENCED_CODE", "TABLE", "QUOTE"]
EXISTS   = [True, False]

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
                files["u/README.md"] = "# other\n"     # duplicate basename
                if ex:
                    files[target] = "# t readme\n"
            ref = {"markdown_link": f"[x]({target})",
                   "explicit_path": target,
                   "bare_basename": base}[syn]
            files["src.md"] = "# src\n" + wrap(ref, ctx)
            mkrepo(root, files)
            tracked = dc.tracked_md(root)
            edges = [e for e in dc.build_graph(root, tracked)
                     if e[0] == "src.md"]
            label = f"[{syn}/{amb}/{ctx}/exists={ex}]"

            if syn == "markdown_link":
                # context is irrelevant: a link is a link
                want = "MARKDOWN_LINK" if ex else "BROKEN_LINK"
                got = [e for e in edges if e[2] == want]
                check(f"{label} link -> {want}", bool(got), str(edges))
                check(f"{label} link context recorded as MARKDOWN_LINK",
                      all(e[4] == "MARKDOWN_LINK" for e in got), str(got))
                if not ex:
                    check(f"{label} broken link not retargeted",
                          all(e[1] is None for e in got), str(got))
            elif syn == "explicit_path":
                got = [e for e in edges if e[2] == "EXPLICIT_PATH_REFERENCE"]
                if ex:
                    check(f"{label} explicit path -> edge", len(got) == 1,
                          str(edges))
                    check(f"{label} resolves to the exact target",
                          got and got[0][1] == target, str(got))
                    check(f"{label} CONTEXT PRESERVED, not semantic",
                          got and got[0][4] == ctx,
                          f"want {ctx} got {got[0][4] if got else None}")
                else:
                    check(f"{label} absent target -> no explicit edge",
                          not got, str(got))
            else:  # bare_basename
                if amb == "duplicate" and ex:
                    got = [e for e in edges if e[2] == "BASENAME_AMBIGUOUS"]
                    check(f"{label} duplicate basename -> AMBIGUOUS",
                          bool(got), str(edges))
                    check(f"{label} ambiguous attributed to nothing",
                          all(e[1] is None for e in got), str(got))
                    check(f"{label} ambiguous context preserved",
                          all(e[4] == ctx for e in got), str(got))
                elif amb == "unique" and ex:
                    got = [e for e in edges if e[2] == "BASENAME_UNIQUE"]
                    check(f"{label} unique basename -> BASENAME_UNIQUE",
                          bool(got), str(edges))
                    check(f"{label} unique basename context preserved",
                          all(e[4] == ctx for e in got), str(got))
    return n


# ── 2. METAMORPHIC ASSERTIONS (Kai's seven) ──────────────────────────
def pairs_for(files, ignored=None):
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        mkrepo(root, files, ignored)
        tracked = dc.tracked_md(root)
        e = dc.build_graph(root, tracked)
        return dc.distinct_pairs(e), e, tracked


def metamorphic():
    base = {"src.md": "# s\nsee t/only.md here\n", "t/only.md": "# o\n"}

    p1, _, _ = pairs_for(base)
    # M1 second evidence kind must not increment the pair count
    two = dict(base); two["src.md"] += "also [x](t/only.md)\n"
    p2, e2, _ = pairs_for(two)
    check("M1 second evidence kind does not increment distinct pairs",
          p1 == p2, f"{p1} vs {p2}")
    check("M1b both evidence kinds still recorded",
          {k for _s, _d, k, _r, _c in e2} >= {"MARKDOWN_LINK",
                                              "EXPLICIT_PATH_REFERENCE"},
          str(dc.kind_tally(e2)))

    # M2 converting explicit path to a link preserves the pair
    conv = {"src.md": "# s\nsee [t](t/only.md) here\n", "t/only.md": "# o\n"}
    p3, _, _ = pairs_for(conv)
    check("M2 explicit->link preserves source->target pair", p1 == p3,
          f"{p1} vs {p3}")

    # M3 unrelated duplicate basename cannot steal an exact-path reference
    steal = dict(base); steal["z/only.md"] = "# decoy\n"
    p4, e4, _ = pairs_for(steal)
    check("M3 duplicate basename does not steal exact-path reference",
          ("src.md", "t/only.md") in p4 and ("src.md", "z/only.md") not in p4,
          str(sorted(p4)))

    # M4 deleted target becomes BROKEN, never retargeted
    gone = {"src.md": "# s\n[x](t/only.md)\n", "other.md": "# o\n"}
    _p5, e5, _ = pairs_for(gone)
    brk = [e for e in e5 if e[2] == "BROKEN_LINK"]
    check("M4 deleted target -> BROKEN", bool(brk), str(e5))
    check("M4b broken link not retargeted", all(e[1] is None for e in brk))

    # M5 ignored markdown cannot alter the tracked denominator
    _p6, _e6, t6 = pairs_for(base)
    _p7, _e7, t7 = pairs_for(base, ignored={"cache/junk.md": "# junk\n"})
    check("M5 ignored .md does not change tracked denominator",
          t6 == t7, f"{t6} vs {t7}")

    # M6 enumeration order cannot alter the verdict
    rev = dict(reversed(list(base.items())))
    p8, _, _ = pairs_for(rev)
    check("M6 enumeration order does not change pairs", p1 == p8,
          f"{p1} vs {p8}")

    # M7 duplicated mention cannot alter distinct-pair count
    dup = dict(base); dup["src.md"] += "see t/only.md again\nand t/only.md\n"
    p9, e9, _ = pairs_for(dup)
    check("M7 duplicated mention does not change distinct pairs", p1 == p9,
          f"{p1} vs {p9}")
    check("M7b but raw evidence count DOES rise (views differ visibly)",
          dc.kind_tally(e9)["EXPLICIT_PATH_REFERENCE"] > 1,
          str(dc.kind_tally(e9)))


if __name__ == "__main__":
    n = crossings()
    metamorphic()
    print(f"P10 CALIBRATION — {n} generated crossings + 7 metamorphic groups")
    print(f"  assertions passed : {P}")
    print(f"  assertions failed : {F}")
    for f in FAILS[:12]:
        print(f"  FAIL {f}")
    sys.exit(1 if F else 0)
