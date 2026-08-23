#!/usr/bin/env python3
"""FOUR-LEG INSTRUMENT QUALIFICATION — the meta-check Census v1.0 lacked.

D341 found the same defect three times in three artefacts: a DECLARED
ALPHABET LARGER THAN THE EMITTED ONE. genlink.POSSIBLE_WRITER, the
doccensus2 OTHER edge context, and H2's REFERENCE were one defect
wearing three costumes, each found by hand. This is the gate that finds
the fourth before we do (R6: fix the class, not the instance).

Kai's D341 ruling defines the four legs:

  1 IMPLEMENTATION_EMITTABLE       a real code path assigns the value
  2 FIXTURE_REACHABLE              a fixture actually reached it at runtime
  3 CALIBRATION_DISCRIMINATING     a passing assertion is ABOUT it
  4 SUBJECT_POPULATION_APPLICABILITY  how often it occurs on the exact
                                   real subject

LEG 4 IS MANDATORY REPORTING, NOT A PASS/FAIL TEST. Kai: "A legitimate
state may simply be absent from a particular corpus." A tiny repository
with no writers does not make PROVEN_WRITE_RELATION defective. Turning a
corpus accident into an ontology is exactly the P10 inversion.

But it carries a binding consequence:

  A DOWNSTREAM CLAIM ABOUT A GIVEN SUBJECT MAY NOT RELY ON A STATE WHOSE
  SUBJECT-POPULATION APPLICABILITY ON THAT SUBJECT IS ZERO.

That is what catches D341 F4 -- NO_WRITER passed legs 1-3 and was still
unusable on the real tree.

DENOMINATOR DERIVATION (R5). The alphabets are discovered by scanning
the package directory for modules exporting ALPHABETS. There is no
hand-written list of values beside the thing being checked; such a list
is a defect waiting to be found.

I-8. The emission scan EXCLUDES the ALPHABETS declaration itself and
every docstring. A detector whose population includes its own
declaration is checking itself against itself.
"""
from __future__ import annotations
import ast
import importlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import caltrace as ct  # noqa: E402

CAL_MODULES = ("cal_docgraph", "cal_opscan", "cal_claims")


def instrument_modules():
    """Derived from the tree, never hand-listed (R5)."""
    out = []
    for p in sorted(HERE.glob("*.py")):
        if p.stem.startswith("cal_") or p.stem in ("qualify", "caltrace",
                                                   "run_census"):
            continue
        try:
            m = importlib.import_module(p.stem)
        except Exception:
            continue
        if hasattr(m, "ALPHABETS"):
            out.append((p.stem, p, m))
    return out


def _excluded_constant_ids(tree):
    """Constants that are DECLARATIONS, not emissions (I-8)."""
    skip = set()
    for node in ast.walk(tree):
        # the ALPHABETS declaration itself
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("ALPHABETS",
                                                        "LINE_CONTEXTS"):
                    for c in ast.walk(node.value):
                        if isinstance(c, ast.Constant):
                            skip.add(id(c))
        # docstrings
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) \
                    and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                skip.add(id(body[0].value))
    return skip


def emission_sites(path: pathlib.Path, value: str):
    """Line numbers where the module assigns/returns this literal."""
    tree = ast.parse(path.read_text())
    skip = _excluded_constant_ids(tree)
    return sorted({getattr(n, "lineno", 0) for n in ast.walk(tree)
                   if isinstance(n, ast.Constant)
                   and isinstance(n.value, str)
                   and n.value == value and id(n) not in skip})


def run_calibration():
    ct.reset()
    for name in CAL_MODULES:
        m = importlib.import_module(name)
        m.run()
    return ct.PASSED, ct.FAILED, list(ct.FAILURES)


def qualify(subject_counts=None, subject_label="(none)"):
    """Returns (rows, findings). subject_counts maps 'alphabet::value'
    to an observed count on the real subject (leg 4)."""
    subject_counts = subject_counts or {}
    passed, failed, failures = run_calibration()

    rows, findings = [], []
    for modname, path, mod in instrument_modules():
        for alpha, values in mod.ALPHABETS.items():
            key = f"{modname}.{alpha}"
            for v in values:
                sites = emission_sites(path, v)
                l1 = bool(sites)
                l2 = ct.OBSERVED.get((key, v), 0)
                l3 = ct.ASSERTED.get((key, v), 0)
                l4 = subject_counts.get(f"{key}::{v}")
                rows.append({"alphabet": key, "value": v,
                             "L1_emittable": l1, "L1_sites": sites,
                             "L2_fixture_reached": l2,
                             "L3_calibration_asserted": l3,
                             "L4_subject_count": l4})
                if not l1:
                    findings.append((key, v, "NO EMISSION PATH — declared "
                                     "but never assigned by any code path"))
                elif not l2:
                    findings.append((key, v, "NO FIXTURE REACHES IT — "
                                     "emittable but never observed at runtime"))
                elif not l3:
                    findings.append((key, v, "NOT DISCRIMINATED — reached "
                                     "but no passing assertion is about it"))
    return rows, findings, (passed, failed, failures)


def report(rows, findings, cal, subject_label="(none)"):
    passed, failed, failures = cal
    out = []
    out.append("FOUR-LEG INSTRUMENT QUALIFICATION — Census v1.1")
    out.append(f"  calibration assertions : {passed} passed, {failed} failed")
    for f in failures:
        out.append(f"    FAIL {f}")
    out.append(f"  leg-4 subject          : {subject_label}")
    out.append("")
    hdr = (f"  {'value':<44}{'L1':>4}{'L2':>7}{'L3':>7}{'L4':>9}")
    cur = None
    for r in rows:
        if r["alphabet"] != cur:
            cur = r["alphabet"]
            out.append(f"\n  [{cur}]")
            out.append(hdr)
        l4 = "-" if r["L4_subject_count"] is None else r["L4_subject_count"]
        out.append(f"  {r['value']:<44}"
                   f"{'ok' if r['L1_emittable'] else 'NO':>4}"
                   f"{r['L2_fixture_reached']:>7}"
                   f"{r['L3_calibration_asserted']:>7}"
                   f"{str(l4):>9}")
    out.append("")
    out.append(f"  declared values : {len(rows)}   (denominator derived "
               f"from the tree, not a list beside it)")
    out.append(f"  FINDINGS        : {len(findings)}")
    for a, v, w in findings:
        out.append(f"    {a} :: {v} :: {w}")

    # Leg 4 is reported, never failed -- but its consequence is binding.
    zero = [r for r in rows if r["L4_subject_count"] == 0]
    if zero:
        out.append("")
        out.append("  LEG-4 ZERO-APPLICABILITY (not a defect; a RESTRICTION "
                   "on downstream claims about this subject):")
        for r in zero:
            out.append(f"    {r['alphabet']} :: {r['value']} — "
                       f"NOT AVAILABLE as current-subject evidence")
    return "\n".join(out)


if __name__ == "__main__":
    rows, findings, cal = qualify()
    print(report(rows, findings, cal))
    sys.exit(1 if (findings or cal[1]) else 0)
