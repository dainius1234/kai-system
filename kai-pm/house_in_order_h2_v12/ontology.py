#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — DECLARED ONTOLOGY, DISPOSITIONS, AND THE INVARIANT.

Carries forward D360 5 unchanged -- EVIDENCE FACTS ARE NOT VERDICTS --
and adds the corrective amendment authorised in D367 3.

THE CORRECTION (D16). The governing text,
`kai-pm/house_in_order_instrument/AUTHORITY_ONTOLOGY.md:44`, states that
UNKNOWN is "First-class on EVERY axis, independently." v1.1's SCOPE
alphabet omitted it -- the only axis of six -- so `emittable('SCOPE',
'UNKNOWN')` was False and injecting the abstention raised CONTRACT
VIOLATION. SCOPE was the one axis STRUCTURALLY FORBIDDEN to say "I do
not know", which is why 50 VALIDITY verdicts had nowhere honest to go.

This is a CORRECTIVE AMENDMENT, not new semantics: the executable
ontology is brought into conformance with an invariant already ratified
in the governing text. It does not invent a region-scoped VALIDITY state.

THE SECOND CORRECTION (D6). `SCOPE=WHOLE_FILE` was dispositioned
"file-level default". A default is not a measurement. It is now earnable
ONLY from a witness, and its rationale says so.

WHY THE INVARIANT IS A FUNCTION AND NOT A COMMENT (D17). v1.1's
qualification iterated ALPHABETS, so a value MISSING from an alphabet
lay outside its denominator entirely: removing UNKNOWN from VALIDITY
left the gate at 0 findings while 216 documents emitted it. A check
whose universe is the list it is checking cannot see an omission from
that list. `ontology_invariants()` therefore derives its expectation from
the AXIS SET rather than from any alphabet, and `qualify_h2.py` proves it
can fail by removing a value.
"""
from __future__ import annotations

ABSTENTION = "UNKNOWN"

# ── the six axes ──────────────────────────────────────────────────────
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
    # D16 REPAIRED: UNKNOWN restored, matching the governing invariant.
    "SCOPE": ("WHOLE_FILE", "HEADING", "TABLE", "MANAGED_REGION", "UNKNOWN"),
}

CAPABILITY_FAILURE = "UNMEASURED"

# ── evidence facts — NEVER verdicts (D360 5) ─────────────────────────
# v1.2 adds the two the audits earned: a contradiction between a
# document's self-claim and the history, and a nominal (self-declared)
# function. Both are FACTS. Neither is a verdict.
EVIDENCE_FACTS = ("MAINTENANCE_OBSERVED", "SELF_ASSERTS_CURRENT",
                  "CONSUMED_AT_SUBJECT", "CITES_COMMIT", "CITES_RUN",
                  "CARRIES_DATE_STAMP", "BINDING_CONTRADICTION",
                  "NOMINAL_FUNCTION", "SELF_ASSERTS_AUTHORITY",
                  "SELF_ASSERTS_NON_AUTHORITY")

DISPOSITIONS = ("H2_EMITTABLE", "H2_NOT_EARNABLE", "DEFERRED_TO_H3")

STATE_DISPOSITION = {
    # LIFECYCLE ------------------------------------------------------
    ("LIFECYCLE", "SUPERSEDED"): ("H2_EMITTABLE",
        "an explicitly named successor document"),
    ("LIFECYCLE", "HISTORICAL"): ("H2_EMITTABLE",
        "a properly bound snapshot or dated-artefact witness"),
    ("LIFECYCLE", "ACTIVE"): ("H2_NOT_EARNABLE",
        "D360 5: maintenance, self-assertion and consumption are "
        "evidence facts, not verdicts. No independently qualified "
        "positive rule for ACTIVE exists at H2."),
    ("LIFECYCLE", "UNKNOWN"): ("H2_EMITTABLE", "abstention"),
    # FUNCTION -------------------------------------------------------
    # D367 6 / correction E: path and title share provenance, so they
    # earn a NOMINAL_FUNCTION evidence fact, not a proven role. The
    # values stay emittable because MARKER is earnable objectively and
    # the rest may be earned by a qualified rule; what changed is that
    # self-description alone no longer earns them.
    **{("FUNCTION", v): ("H2_EMITTABLE",
        "an objective witness, or a qualified rule; self-description "
        "alone yields NOMINAL_FUNCTION and an abstention")
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
    **{("VALIDITY", v): ("H2_EMITTABLE",
        "document-level binding + verified witness kind + whole-document "
        "applicability + no unresolved contradiction")
       for v in ("CURRENT_TREE", "EXACT_SNAPSHOT", "RUN_ARTEFACT",
                 "TIME_BOUND")},
    ("VALIDITY", "UNKNOWN"): ("H2_EMITTABLE", "abstention"),
    # SCOPE ----------------------------------------------------------
    # D6 REPAIRED: no longer a default. Earned from a witness or absent.
    ("SCOPE", "WHOLE_FILE"): ("H2_EMITTABLE",
        "EARNED from a document-level binding witness. NOT a default: "
        "a value emitted when nothing was measured is a claim."),
    **{("SCOPE", v): ("H2_NOT_EARNABLE",
        "region override determination is not implemented at H2")
       for v in ("HEADING", "TABLE", "MANAGED_REGION")},
    ("SCOPE", "UNKNOWN"): ("H2_EMITTABLE", "abstention"),
}


def emittable(axis, value):
    return STATE_DISPOSITION.get((axis, value), (None, None))[0] == "H2_EMITTABLE"


def disposition_rows():
    out = []
    for axis, values in ALPHABETS.items():
        for v in values:
            d, why = STATE_DISPOSITION.get((axis, v), ("UNDECLARED", None))
            out.append({"axis": axis, "value": v, "disposition": d,
                        "rationale": why})
    return out


def undeclared():
    """Any declared value with no disposition. Must always be empty."""
    return [(a, v) for a, vs in ALPHABETS.items() for v in vs
            if (a, v) not in STATE_DISPOSITION]


def ontology_invariants():
    """THE GOVERNING INVARIANT, CHECKED FROM THE AXIS SET (D17).

    Findings are derived by iterating AXES -- deliberately NOT by
    iterating each axis's alphabet, because that is precisely the loop
    that could not see UNKNOWN missing from SCOPE. The universe of this
    check must not be the list under test.

    Returns [] when conforming.
    """
    findings = []
    for axis in ALPHABETS:                       # <- the AXIS SET
        if ABSTENTION not in ALPHABETS[axis]:
            findings.append(
                (axis, f"{ABSTENTION} absent from the alphabet; the "
                       f"governing ontology requires it on EVERY axis"))
            continue
        disp = STATE_DISPOSITION.get((axis, ABSTENTION))
        if disp is None:
            findings.append((axis, f"{ABSTENTION} has no disposition"))
        elif disp[0] != "H2_EMITTABLE":
            findings.append(
                (axis, f"{ABSTENTION} is {disp[0]}, so the axis cannot "
                       f"abstain"))
    return findings


def values_outside_alphabet(rows):
    """A value EMITTED but unknown to the ontology (D17, other half).

    `undeclared()` catches a declared value with no disposition. This
    catches the inverse -- output the ontology has never heard of --
    which is the direction that let 216 UNKNOWN rows pass a green gate.
    Derived from the OUTPUT, not from the alphabet.
    """
    bad = []
    for r in rows:
        for axis in ALPHABETS:
            v = r[axis]["value"]
            if v == CAPABILITY_FAILURE:
                continue
            if v not in ALPHABETS[axis]:
                bad.append((r.get("path"), axis, v))
    return bad
