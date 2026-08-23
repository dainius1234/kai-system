#!/usr/bin/env python3
"""SUBJECT APPLICABILITY RECORD — the restriction travels with the
measurement.

KAI'S D342 FREEZE CONDITION. Leg-4 restrictions may not live only inside
`qualify.py` or a narrative report. A future tool can read
`census-worldA.json` and never read the qualification report, and would
then use a state that has zero applicability on that subject as though
it were evidence.

So every census carries an applicability record, and the record is bound
to the census by SHA-256 in both directions:

  * the full record is embedded as a TOP-LEVEL BLOCK in the census, so
    it cannot be separated from the numbers by copying the file;
  * the identical canonical bytes are also written as a standalone
    artefact and named in the manifest, so the binding is checkable
    without parsing the census.

DOWNSTREAM_USABLE_ON_THIS_SUBJECT is the field a consumer must read. It
is FALSE whenever a state is qualified in the abstract but did not occur
on this exact subject -- which is D341 F4's situation and is the reason
this file exists.

NOT A PASS/FAIL GATE. Kai: "A legitimate state may simply be absent from
a particular corpus." Zero applicability is a RESTRICTION, never a
defect, and the reason field says so explicitly rather than leaving a
reader to infer it.
"""
from __future__ import annotations
import hashlib
import json

RESTRICTION_ZERO = (
    "ZERO_SUBJECT_POPULATION: the state is qualified (emittable, "
    "fixture-reachable, calibration-discriminating) but occurred zero "
    "times on this exact subject. It is NOT a defect and NOT evidence "
    "of absence in the world; it may not support a subject-specific "
    "downstream claim about this subject.")
RESTRICTION_UNQUALIFIED = (
    "INSTRUMENT_NOT_QUALIFIED_FOR_THIS_STATE: the state failed a "
    "qualification leg, so nothing it produces is admissible.")
RESTRICTION_UNMEASURED = (
    "NOT_MEASURED_ON_THIS_SUBJECT: no subject population count was "
    "recorded, so applicability is UNKNOWN and may not be assumed.")


def build(instrument_version, subject_commit, subject_tree, rows,
          repair_evidence=None):
    """Returns (record, canonical_bytes, sha256_hex)."""
    states = []
    for r in rows:
        l1 = bool(r["L1_emittable"])
        l2 = int(r["L2_fixture_reached"])
        l3 = int(r["L3_calibration_asserted"])
        l4 = r["L4_subject_count"]
        qualified = l1 and l2 > 0 and l3 > 0

        if not qualified:
            usable, reason = False, RESTRICTION_UNQUALIFIED
        elif l4 is None:
            usable, reason = False, RESTRICTION_UNMEASURED
        elif l4 == 0:
            usable, reason = False, RESTRICTION_ZERO
        else:
            usable, reason = True, None

        states.append({
            "alphabet": r["alphabet"],
            "value": r["value"],
            "L1_IMPLEMENTATION_EMITTABLE": l1,
            "L2_FIXTURE_REACHABLE": l2,
            "L3_CALIBRATION_DISCRIMINATING": l3,
            "L4_SUBJECT_POPULATION_COUNT": l4,
            "DOWNSTREAM_USABLE_ON_THIS_SUBJECT": usable,
            "DOWNSTREAM_RESTRICTION_REASON": reason,
        })

    record = {
        "record_type": "SUBJECT_APPLICABILITY_RECORD",
        "instrument_version": instrument_version,
        "subject_commit": subject_commit,
        "subject_tree": subject_tree,
        "declared_states": len(states),
        "usable_on_this_subject": sum(
            1 for s in states if s["DOWNSTREAM_USABLE_ON_THIS_SUBJECT"]),
        "restricted_on_this_subject": sum(
            1 for s in states if not s["DOWNSTREAM_USABLE_ON_THIS_SUBJECT"]),
        "binding_rule":
            "A downstream claim about this subject MAY NOT rely on any "
            "state whose DOWNSTREAM_USABLE_ON_THIS_SUBJECT is false.",
        "states": states,
    }
    if repair_evidence is not None:
        record["repair_evidence"] = repair_evidence

    blob = canonical(record)
    return record, blob, hashlib.sha256(blob).hexdigest()


def canonical(record) -> bytes:
    """One byte-form, so the census and the sidecar cannot disagree."""
    return json.dumps(record, indent=1, sort_keys=True,
                      default=str).encode() + b"\n"


def verify(record, blob, sha) -> bool:
    return canonical(record) == blob and hashlib.sha256(blob).hexdigest() == sha


def restricted(record):
    return [(s["alphabet"], s["value"], s["DOWNSTREAM_RESTRICTION_REASON"])
            for s in record["states"]
            if not s["DOWNSTREAM_USABLE_ON_THIS_SUBJECT"]]
