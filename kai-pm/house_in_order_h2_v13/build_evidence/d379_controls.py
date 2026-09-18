#!/usr/bin/env python3
"""D379 HOSTILE CONTROLS — governed by D379 as corrected by D380.

EVIDENCE CLASS: PRODUCER MEASUREMENT - SIGHTED - ZERO ADMISSION WEIGHT.
Orion's own assessment carries no final admission weight.

Every control is EXECUTED, never merely asserted (R2). Controls whose
subject is a shipped process entry point assert the ACTUAL SUBPROCESS
RETURN CODE. Controls whose subject is a classification decision function
call that function directly, because the function IS the subject; calling
a helper and inferring a process exit status is what let an earlier defect
through and is not done here.

SECTION REGISTRY. Sections not yet implemented FAIL LOUDLY. A control file
that reports green over a section it does not cover is the exact defect
class this programme exists to find, so absence is never silence here.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

V = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V))

import classify                                              # noqa: E402
import passa                                                 # noqa: E402
from envelope import Witness                                 # noqa: E402

PASSED, FAILED, FAILURES = 0, 0, []
# Sections of the D379 hostile matrix. Implemented sections run; the rest
# fail as NOT_IMPLEMENTED so this file can never report a green tranche.
SECTIONS = ["M2", "D14", "Q1a", "Q1b", "86", "SB", "I1A", "I1B",
            "DEP", "STAGE_A", "STDLIB"]
IMPLEMENTED = {"M2"}


def check(name: str, condition: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
    return condition


def subject_digests():
    print("SUBJECT — the exact bytes measured")
    for n in ("passa.py", "classify.py", "envelope.py", "run_h2_v12.py",
              "qualify.py", "holdout.py", "cal_fixtures.py", "ontology.py",
              "subjectbind.py"):
        p = V / n
        if p.exists():
            print(f"  {n:<16} {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print()


# ── M2 / A3-i — LIFECYCLE SUBJECT BINDING ─────────────────────────────
#
# THE PRECOMMITTED D379 PREDICATE, fixed before this measurement ran:
#
#   LIMB I   the three false audited-snapshot COMMIT routes must NOT
#            classify HISTORICAL
#   LIMB II  a genuine document-own-lifecycle COMMIT route must STILL
#            classify HISTORICAL
#
#   BOTH satisfied   -> no M2 semantic edit; discrimination DEMONSTRATED
#   EITHER fails     -> the bounded A3-i correction is the ONLY authorised
#                       semantic mutation
#
# Both limbs are required. LIMB I alone is satisfiable by A3-ii, which was
# rejected for removing genuine cases along with the defective ones.
# LIMB II is what proves DISCRIMINATION rather than SUPPRESSION.
#
# No git history is consulted and no Pass A is run: the decision path
#   _scope_of -> witness.applicability_scope -> _binding_witness -> lifecycle
# is a pure function of the document text and the witness, so the measurement
# needs neither a history source nor a candidate.

COMMIT = "2d830f25d569baa5ce955dd8d17e8f0744239876"

FALSE_ROUTE_TEXT = (
    "# Synthetic Code Audit Report\n"
    "\n"
    f"**Audited snapshot:** default branch through findings commit `{COMMIT}`\n"
    "\n"
    "## Findings\n"
    "\nProse about the audited tree.\n")

GENUINE_ROUTE_TEXT = (
    "# Synthetic Code Audit Snapshot\n"
    "\n"
    f"**Acquisition commit:** `{COMMIT}`\n"
    "\n"
    "## Contents\n"
    "\nThis document was taken at the commit above.\n")


def _commit_witness(text, path):
    """Build the COMMIT witness exactly as passa.scan() builds one."""
    m = passa.HEX.search(text)
    assert m is not None and m.group(0) == COMMIT
    assert passa._eligible(m)
    head = text[:passa.HEAD_BYTES]
    return Witness(
        witness_type="COMMIT", witness_value=m.group(0), source_path=path,
        source_selector=passa._selector(text, m.start()),
        local_context=passa._context(text, m.start(), m.end()),
        applicability_scope=passa._scope_of(head, m.start(), "HEX"),
        evidence_total=1, evidence_shown=1, truncated=False,
        polarity="POSITIVE", certainty="VERIFIED")


def _lifecycle_of(text, path):
    w = _commit_witness(text, path)
    row = {"path": path, "witnesses": {"COMMIT": [w.asdict()]}}
    snap = classify._binding_witness(row, "COMMIT")
    res = classify.lifecycle(path=path, superseded_by=None,
                             snapshot_witness=snap.asdict() if snap else None,
                             blocked=None)
    return res, w, snap


def section_M2():
    print("M2 / A3-i — LIFECYCLE SUBJECT BINDING")
    print("  invariant: the subject of a LIFECYCLE verdict is THE DOCUMENT.")
    print("  Evidence about an artefact the document DESCRIBES is evidence")
    print("  about that artefact.\n")

    f_res, f_w, f_snap = _lifecycle_of(
        FALSE_ROUTE_TEXT, "kai-pm/SYNTHETIC_CODE_AUDIT_FALSE_ROUTE.md")
    g_res, g_w, g_snap = _lifecycle_of(
        GENUINE_ROUTE_TEXT, "kai-pm/SYNTHETIC_CODE_AUDIT_GENUINE_ROUTE.md")

    for label, res, w, snap in (
            ("M2-1 LIMB I  false audited-snapshot route", f_res, f_w, f_snap),
            ("M2-2 LIMB II genuine own-lifecycle route", g_res, g_w, g_snap)):
        d = w.asdict()
        print(f"  {label}")
        print(f"      applicability_scope  {d['applicability_scope']}")
        print(f"      witness subject      {d.get('subject')}")
        print(f"      _binding_witness     "
              f"{'returned a witness' if snap else 'None'}")
        print(f"      LIFECYCLE            {res['value']}")
        print(f"      rationale            {str(res.get('rationale'))[:70]}")

    limb1 = check("M2-1 LIMB I: the false audited-snapshot COMMIT route does "
                  "NOT classify HISTORICAL",
                  f_res["value"] != "HISTORICAL",
                  f"classified {f_res['value']} — the commit is the snapshot "
                  f"the document AUDITS, i.e. evidence about a DIFFERENT "
                  f"subject, and it determined the document's own lifecycle")
    limb2 = check("M2-2 LIMB II: the genuine document-own-lifecycle COMMIT "
                  "route STILL classifies HISTORICAL",
                  g_res["value"] == "HISTORICAL",
                  f"classified {g_res['value']} — suppression, not "
                  f"discrimination")

    print()
    print(f"  LIMB I  {'PASS' if limb1 else 'FAIL'}"
          f"     LIMB II {'PASS' if limb2 else 'FAIL'}")
    if limb1 and limb2:
        print("  M2 PREDICATE SATISFIED — no M2 semantic edit is authorised.")
        print("  M2 discrimination DEMONSTRATED.")
    else:
        print("  M2 PREDICATE NOT SATISFIED — the bounded A3-i correction is")
        print("  the ONLY authorised semantic mutation. No adjacent cleanup,")
        print("  no vocabulary redesign, no unrelated Pass-A semantics.")
    print()
    # The discrimination question stated as one measurement, not as prose:
    check("M2-3 the two routes are DISTINGUISHED from one another",
          f_res["value"] != g_res["value"],
          f"both routes classified {f_res['value']} with the same rationale — "
          f"the binding predicate did not discriminate between evidence about "
          f"the document and evidence about an artefact it describes")


def main() -> int:
    print("=" * 70)
    print("D379 HOSTILE CONTROLS — PRODUCER MEASUREMENT, SIGHTED,")
    print("                        ZERO ADMISSION WEIGHT")
    print("=" * 70)
    print()
    subject_digests()

    section_M2()

    print("-" * 70)
    print("SECTION COVERAGE")
    for s in SECTIONS:
        state = "IMPLEMENTED" if s in IMPLEMENTED else "NOT_IMPLEMENTED"
        print(f"  {s:<10} {state}")
        if s not in IMPLEMENTED:
            check(f"section {s} is implemented", False,
                  "NOT_IMPLEMENTED — this control file does not yet cover "
                  "this section of the D379 hostile matrix")
    print()
    print("=" * 70)
    print(f"{PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"  FAIL {f}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
