#!/usr/bin/env python3
"""MEASUREMENT-WINDOW PARITY GUARD.

Kai, cycle-5 closure §7: my cycle-3 and cycle-4 evaluators passed FULL
TEXT to `_scope_of` while production `scan()` passes `head[:HEAD_BYTES]`.
Measured across all 492 records the two agreed 492/492, so no reported
result was affected -- but **zero observed effect is not a reason to
leave the divergence unguarded**, and the agreement was a fact about this
corpus rather than about the code.

THE DIVERGENCE IS REAL AND OBSERVABLE. It is not `_preamble_end`: a
witness starting before HEAD_BYTES is at root under either window. It is
the LABEL-UNIQUENESS DENOMINATOR, which iterates the lines of whichever
window it was given. A document carrying `**Date:**` once inside the head
window and again beyond it resolves WHOLE_FILE over `head` and SPAN over
the full text. That is what this guard protects.

This is MEASUREMENT HARDENING. It does not touch `_scope_of`, does not
change M3 scope logic, and requires no corpus run.

    python3 window_parity_guard.py            # run the guard
    python3 window_parity_guard.py --can-fail # prove each leg can fail
"""
from __future__ import annotations
import argparse
import ast
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
import passa as P                                              # noqa: E402

# Every path that computes an applicability scope must hand `_scope_of`
# the SAME source window. Declared here so a new caller cannot be added
# silently without either appearing in this list or failing leg A.
SCOPE_CALLERS = ("passa.py", "build_evidence/make_record_manifest.py")
REQUIRED_WINDOW = "head"


def _call_sites(src):
    """(lineno, first-argument source expression) for each _scope_of call."""
    out = []
    for n in ast.walk(ast.parse(src)):
        if not isinstance(n, ast.Call):
            continue
        f = n.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        if name == "_scope_of" and n.args:
            out.append((n.lineno, ast.unparse(n.args[0])))
    return out


def leg_a(sources):
    """STATIC. Every _scope_of call site passes the production window."""
    rows, ok = [], True
    for label, src in sources.items():
        sites = _call_sites(src)
        if not sites:
            rows.append((label, "-", "NO CALL SITES", False))
            ok = False
            continue
        for lineno, arg in sites:
            good = arg == REQUIRED_WINDOW
            ok &= good
            rows.append((label, lineno, arg, good))
    return ok, rows


def leg_b():
    """BEHAVIOURAL. A probe on which the two windows genuinely disagree,
    so the guard is never merely asserting that two identical things are
    identical (I-8: a control that cannot fail proves nothing).
    """
    doc = ("# T\n\n**Date:** 2026-01-02\n\n" + "filler line\n" * 700 +
           "**Date:** 2026-03-04\n\n## S\n")
    i = doc.index("2026-01-02")
    head_scope = P._scope_of(doc[:P.HEAD_BYTES], i, "DATE")
    full_scope = P._scope_of(doc, i, "DATE")
    discriminates = head_scope != full_scope
    return discriminates, head_scope, full_scope


def run(sources, verbose=True):
    a_ok, rows = leg_a(sources)
    b_ok, hs, fs = leg_b()
    if verbose:
        print("LEG A — every _scope_of call site passes "
              f"{REQUIRED_WINDOW!r}, the production window")
        for label, lineno, arg, good in rows:
            print(f"  {label:<42} L{lineno:<5} arg={arg!r:<10} "
                  f"{'OK' if good else '<<< DIVERGENT'}")
        print(f"  leg A: {'PASS' if a_ok else 'FAIL'}")
        print()
        print("LEG B — the probe must DISCRIMINATE, or leg A guards nothing")
        print(f"  scope over head[:{P.HEAD_BYTES}] : {hs}")
        print(f"  scope over full text        : {fs}")
        print(f"  windows disagree on the probe: {b_ok}")
        print(f"  leg B: {'PASS' if b_ok else 'FAIL'}")
    return a_ok and b_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--can-fail", action="store_true",
                    help="prove each leg fails when its property is broken")
    a = ap.parse_args()

    live = {label: (PKG / label).read_text() for label in SCOPE_CALLERS}

    if not a.can_fail:
        ok = run(live)
        print(f"\nWINDOW PARITY: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    print("CAN-FAIL PROOF — each leg is broken deliberately and must FAIL\n")
    broken = dict(live)
    broken["build_evidence/make_record_manifest.py"] = live[
        "build_evidence/make_record_manifest.py"].replace(
        "P._scope_of(head, s, det)", "P._scope_of(text, s, det)")
    a_ok, rows = leg_a(broken)
    bad = [r for r in rows if not r[3]]
    print(f"  leg A with one caller switched to full text -> "
          f"{'FAIL' if not a_ok else 'PASS'}   "
          f"{'OK, the guard fires' if not a_ok else '<<< GUARD IS VACUOUS'}")
    for label, lineno, arg, _ in bad:
        print(f"      caught: {label} L{lineno} arg={arg!r}")
    b_ok, hs, fs = leg_b()
    print(f"  leg B probe discriminates ({hs} vs {fs}) -> "
          f"{'OK' if b_ok else '<<< PROBE IS VACUOUS'}")
    print("\n  A guard whose legs cannot fail is the I-8 defect. Both can.")
    raise SystemExit(0 if (not a_ok) and b_ok else 1)


if __name__ == "__main__":
    main()
