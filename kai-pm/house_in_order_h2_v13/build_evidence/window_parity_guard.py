#!/usr/bin/env python3
"""MEASUREMENT-WINDOW PARITY GUARD.

Kai, cycle-5 closure §7: my cycle-3 and cycle-4 evaluators passed FULL
TEXT to `_scope_of` while production `scan()` passes `head[:HEAD_BYTES]`.
Measured across all 492 records the two agreed 492/492, so no reported
result was affected -- but **zero observed effect is not a reason to
leave the divergence unguarded**, and the agreement was a fact about this
corpus rather than about the code.

THE DIVERGENCE IS REAL AND OBSERVABLE. It is not `_preamble_end`: a
witness starting before HEAD_BYTES sits at root under either window. It
is the LABEL-UNIQUENESS DENOMINATOR, which iterates the lines of whichever
window it was given. A document carrying `**Date:**` once inside the head
window and again beyond it resolves WHOLE_FILE over `head` and SPAN over
the full text. That is the property this guard protects.

THE UNIVERSE IS DISCOVERED, NOT LISTED. The first version of this file
began from a hand-written `SCOPE_CALLERS` tuple of filenames. That is a
list kept beside the thing it governs -- R5, the exact defect class this
guard exists to catch, reproduced inside the guard. A new module calling
`_scope_of` would have escaped it silently. The universe is now walked.

The only exclusion is structural and is this file: the guard's own probe
must call `_scope_of` with BOTH windows in order to prove they differ.
The instrument is not one of the callers it governs.

This is MEASUREMENT HARDENING. It does not touch `_scope_of`, does not
change M3 scope logic, and requires no corpus run.

    python3 window_parity_guard.py            # run the guard
    python3 window_parity_guard.py --can-fail # prove it can fail, 2 ways
"""
from __future__ import annotations
import argparse
import ast
import pathlib
import sys

GUARD = pathlib.Path(__file__).resolve()
PKG = GUARD.parent.parent
sys.path.insert(0, str(PKG))
import passa as P                                              # noqa: E402

REQUIRED_WINDOW = "head"


def discover(root):
    """Every `_scope_of` call in the governed source tree.

    Walked, never listed. Returns (relative path, lineno, first-argument
    source expression) so a violation names itself.
    """
    found = []
    for py in sorted(pathlib.Path(root).rglob("*.py")):
        if py.resolve() == GUARD:
            # THE ONLY EXCLUSION, and it is by resolved PATH IDENTITY, not
            # by directory name or category. The guard's own leg-B probe
            # must call _scope_of with BOTH windows in order to prove they
            # differ; the instrument is not one of the callers it governs.
            # There is deliberately no EXCLUDE_DIRS list: fixtures and
            # tests inside the governed tree ARE scanned, because a
            # divergent call in a fixture is still a divergent call. A
            # skip for "__pycache__" was removed as dead code -- rglob
            # matches *.py and that directory holds only *.pyc -- so no
            # directory-name exclusion survives anywhere in this file.
            continue
        try:
            tree = ast.parse(py.read_text())
        except SyntaxError:
            found.append((py.relative_to(root), 0, "<UNPARSEABLE>"))
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call) or not n.args:
                continue
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(
                f, "id", "")
            if name == "_scope_of":
                found.append((py.relative_to(root), n.lineno,
                              ast.unparse(n.args[0])))
    return found


def leg_a(root):
    """STATIC. Every discovered call site passes the production window."""
    sites = discover(root)
    rows = [(p, ln, arg, arg == REQUIRED_WINDOW) for p, ln, arg in sites]
    return bool(rows) and all(r[3] for r in rows), rows


def leg_b():
    """BEHAVIOURAL. A probe on which the two windows genuinely disagree,
    so leg A is never merely asserting that two identical things are
    identical. A control that cannot fail proves nothing (I-8).
    """
    doc = ("# T\n\n**Date:** 2026-01-02\n\n" + "filler line\n" * 700 +
           "**Date:** 2026-03-04\n\n## S\n")
    i = doc.index("2026-01-02")
    head_scope = P._scope_of(doc[:P.HEAD_BYTES], i, "DATE")
    full_scope = P._scope_of(doc, i, "DATE")
    return head_scope != full_scope, head_scope, full_scope


def report(root):
    a_ok, rows = leg_a(root)
    b_ok, hs, fs = leg_b()
    print(f"LEG A — every _scope_of call DISCOVERED under {root} must pass "
          f"{REQUIRED_WINDOW!r}")
    for p, ln, arg, good in rows:
        print(f"  {str(p):<44} L{ln:<5} arg={arg!r:<22} "
              f"{'OK' if good else '<<< DIVERGENT'}")
    print(f"  {len(rows)} call sites discovered by walking the tree, "
          f"no filename list")
    print(f"  leg A: {'PASS' if a_ok else 'FAIL'}\n")
    print("LEG B — the probe must DISCRIMINATE, or leg A guards nothing")
    print(f"  scope over head[:{P.HEAD_BYTES}] : {hs}")
    print(f"  scope over full text        : {fs}")
    print(f"  leg B: {'PASS' if b_ok else 'FAIL'}")
    return a_ok and b_ok


def _can_fail(root):
    print("CAN-FAIL PROOF — the guard is byte-identical throughout\n")
    ok = True

    # A. EXISTING CALLER DRIFT.
    victim = root / "build_evidence" / "make_record_manifest.py"
    original = victim.read_text()
    try:
        victim.write_text(original.replace("P._scope_of(head, s, det)",
                                           "P._scope_of(text, s, det)"))
        a_ok, rows = leg_a(root)
        bad = [r for r in rows if not r[3]]
        print(f"  A. existing caller mutated head -> text : "
              f"{'FAIL' if not a_ok else 'PASS'}  "
              f"{'OK, guard fires' if not a_ok else '<<< VACUOUS'}")
        for p, ln, arg, _ in bad:
            print(f"       caught: {p} L{ln} arg={arg!r}")
        ok &= not a_ok
    finally:
        victim.write_text(original)

    # B. NEW CALLER DRIFT -- a module the guard has never heard of.
    rogue = root / "_parity_rogue_tmp.py"
    try:
        rogue.write_text("import passa\n"
                         "def f(text, s):\n"
                         "    return passa._scope_of(text, s, 'DATE')\n")
        a_ok, rows = leg_a(root)
        bad = [r for r in rows if not r[3]]
        print(f"  B. NEW module added to the tree          : "
              f"{'FAIL' if not a_ok else 'PASS'}  "
              f"{'OK, discovered automatically' if not a_ok else '<<< ESCAPED'}")
        for p, ln, arg, _ in bad:
            print(f"       caught: {p} L{ln} arg={arg!r}")
        ok &= not a_ok
    finally:
        rogue.unlink(missing_ok=True)

    b_ok, hs, fs = leg_b()
    print(f"  leg-B probe discriminates ({hs} vs {fs})  : "
          f"{'OK' if b_ok else '<<< VACUOUS'}")
    print(f"\n  restored tree still passes: {report(root)}")
    return ok and b_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can-fail", action="store_true")
    a = ap.parse_args()
    if a.can_fail:
        ok = _can_fail(PKG)
        print(f"\nCAN-FAIL: {'PROVEN, both modes' if ok else 'NOT PROVEN'}")
        raise SystemExit(0 if ok else 1)
    ok = report(PKG)
    print(f"\nWINDOW PARITY: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
