#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — DECLARED ONTOLOGY AND STATE DISPOSITIONS.

D360 §5 is the contract this file encodes: EVIDENCE FACTS ARE NOT
VERDICTS. HOUSE_H2 v1.0 awarded ACTIVE from `commits > 1` or from a
present-tense self-claim. Neither proves current lifecycle:

  * commits > 1 proves OBSERVED MAINTENANCE within the measured window.
    A deprecated document can be edited; a historical one can receive
    cleanup; a file can carry many commits and govern nothing.
  * a present-tense claim proves only that THE DOCUMENT ASSERTS
    currentness -- not that the assertion is true. 29 of v1.0's 43
    ACTIVE verdicts, 67%, rested on this alone.
  * an executable read proves CONSUMED_AT_SUBJECT. Stale, deprecated and
    accidentally retained artefacts are read by code too.

So the three become EVIDENCE FACTS carried in their own field, and
LIFECYCLE=ACTIVE is declared H2_NOT_EARNABLE until a separately
justified and calibrated positive rule exists.

WHY DISPOSITIONS ARE DATA AND NOT COMMENTS. D341 found three declared
values that no code path could emit, in three artefacts, each discovered
by hand. A comment saying "deliberate" is indistinguishable from an
unimplemented branch. Every declared value therefore carries exactly one
machine-readable disposition, and `qualify_h2.py` proves it.
"""
from __future__ import annotations

# ── the six axes, exactly as AUTHORITY_ONTOLOGY.md declares them ──────
ALPHABETS = {
    "LIFECYCLE": ("ACTIVE", "HISTORICAL", "SUPERSEDED", "UNKNOWN"),
    "FUNCTION": ("GOVERNANCE", "STATUS", "PLAN", "EVIDENCE", "REFERENCE",
                 "RUNTIME_INPUT", "TEMPLATE", "MARKER", "USER_GUIDE",
                 "OTHER", "UNKNOWN"),
    "AUTHORITY": ("AUTHORITATIVE", "VERIFIED_DERIVED", "ADVISORY",
                  "NON_AUTHORITY", "UNKNOWN"),
    "GENERATION": ("MANUAL", "PARTIAL_DERIVED", "FULL_DERIVED", "UNKNOWN"),
    "VALIDITY": ("CURRENT_TREE", "EXACT_SNAPSHOT", "RUN_ARTEFACT",
                 "TIME_BOUND", "UNKNOWN"),
    "SCOPE": ("WHOLE_FILE", "HEADING", "TABLE", "MANAGED_REGION"),
}

# UNMEASURED is not an ontology value. It is the capability-failure
# signal required by the environment-subject contract, and it may appear
# on any axis whose capability was not demonstrated.
CAPABILITY_FAILURE = "UNMEASURED"

# ── evidence facts — NEVER verdicts (D360 §5) ────────────────────────
EVIDENCE_FACTS = ("MAINTENANCE_OBSERVED", "SELF_ASSERTS_CURRENT",
                  "CONSUMED_AT_SUBJECT")

DISPOSITIONS = ("H2_EMITTABLE", "H2_NOT_EARNABLE", "DEFERRED_TO_H3")

STATE_DISPOSITION = {
    # LIFECYCLE ------------------------------------------------------
    ("LIFECYCLE", "SUPERSEDED"): ("H2_EMITTABLE",
        "an explicitly named successor document"),
    ("LIFECYCLE", "HISTORICAL"): ("H2_EMITTABLE",
        "a properly bound snapshot or dated-artefact witness"),
    ("LIFECYCLE", "ACTIVE"): ("H2_NOT_EARNABLE",
        "D360 §5: maintenance, self-assertion and consumption are "
        "evidence facts, not verdicts. No independently qualified "
        "positive rule for ACTIVE exists at H2."),
    ("LIFECYCLE", "UNKNOWN"): ("H2_EMITTABLE", "abstention"),
    # FUNCTION -------------------------------------------------------
    **{("FUNCTION", v): ("H2_EMITTABLE", "path nominates, witness earns")
       for v in ALPHABETS["FUNCTION"]},
    # AUTHORITY ------------------------------------------------------
    **{("AUTHORITY", v): ("DEFERRED_TO_H3",
        "H2 verifies no claim; authority is qualified at H3")
       for v in ("AUTHORITATIVE", "VERIFIED_DERIVED", "ADVISORY",
                 "NON_AUTHORITY")},
    ("AUTHORITY", "UNKNOWN"): ("H2_EMITTABLE", "abstention"),
    # GENERATION -----------------------------------------------------
    ("GENERATION", "MANUAL"): ("H2_NOT_EARNABLE",
        "requires a CLOSED search space yielding NO_WRITER; Census "
        "leaves the space open on this subject"),
    ("GENERATION", "PARTIAL_DERIVED"): ("H2_NOT_EARNABLE",
        "requires established write SCOPE; region determination is not "
        "implemented at H2"),
    ("GENERATION", "FULL_DERIVED"): ("H2_NOT_EARNABLE",
        "requires a writer proven to own WHOLE_FILE"),
    ("GENERATION", "UNKNOWN"): ("H2_EMITTABLE", "abstention"),
    # VALIDITY -------------------------------------------------------
    **{("VALIDITY", v): ("H2_EMITTABLE", "binding evidence in the text")
       for v in ALPHABETS["VALIDITY"]},
    # SCOPE ----------------------------------------------------------
    ("SCOPE", "WHOLE_FILE"): ("H2_EMITTABLE", "file-level default"),
    **{("SCOPE", v): ("H2_NOT_EARNABLE",
        "region override determination is not implemented at H2")
       for v in ("HEADING", "TABLE", "MANAGED_REGION")},
}


def emittable(axis, value):
    return STATE_DISPOSITION.get((axis, value), (None, None))[0] == "H2_EMITTABLE"


def disposition_rows():
    """Flat, machine-readable table for the admission contract."""
    out = []
    for axis, values in ALPHABETS.items():
        for v in values:
            d, why = STATE_DISPOSITION.get((axis, v), ("UNDECLARED", None))
            out.append({"axis": axis, "value": v, "disposition": d,
                        "rationale": why})
    return out


def undeclared():
    """Any declared value with no disposition. Must always be empty --
    a value the contract forgot is the defect this file exists to stop."""
    return [(a, v) for a, vs in ALPHABETS.items() for v in vs
            if (a, v) not in STATE_DISPOSITION]
