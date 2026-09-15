#!/usr/bin/env python3
"""D368 HOLDOUT COVERAGE MEASUREMENT — the exact computation behind the
figures D370 cited for rule 43.

Banked under the D370 CORRECTION because D370 used those figures as
earned evidence while the DERIVED_CLAIM card lived only in a chat
message. Chat is not repository canon, and rule 33 was already banked
before D370 was written.

READ-ONLY. Measures an existing frozen artefact. Adjudicates nothing,
repairs nothing, and is not part of any candidate.

    python3 d368_holdout_coverage_card.py \
        --holdout kai-pm/house_in_order_h2_v12/h2v12-holdout.json

THE GROUPING KEY IS A CONSTRUCTION, NOT A CAUSE. It is
`witness_type / binding label / emitted value`. Rows sharing a key
TRAVELLED THE SAME CLASSIFIER PATH. That is a ROUTING SIGNATURE (rule
37), not an adjudicated causal mechanism, and the count must never be
reported as a mechanism count.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import re

AXES = ("LIFECYCLE", "FUNCTION", "AUTHORITY", "GENERATION", "VALIDITY", "SCOPE")
ABSTENTIONS = ("UNKNOWN", "UNMEASURED")

# The binding label is the leading `Label:` of the witness's own context
# line. Same expression Pass A uses to recognise a structured binding.
LABEL = re.compile(
    r"\s*[>\-*|]?\s*[*_`]{0,2}\s*([A-Za-z][A-Za-z ()-]{2,24}?)\s*[*_`]{0,2}\s*:")


def routing_signature(cell):
    """The constructed key. None for abstentions."""
    if cell["value"] in ABSTENTIONS:
        return None
    w = cell.get("witness")
    if not w:
        return f"{cell['value']}:no-witness"
    m = LABEL.match(w.get("local_context", ""))
    label = m.group(1).strip().lower() if m else "(inline)"
    return f"{w['witness_type']}/{label}->{cell['value']}"


def family_key(path):
    """DOCUMENT FAMILY IS ALSO A CONSTRUCTION, and a cruder one than the
    routing signature: the leading underscore-delimited token of the
    basename, or the top directory. It is a filename heuristic, NOT an
    adjudicated document class, and its count carries the same rule 37
    caveat."""
    base = path.split("/")[-1]
    return base.split("_")[0] if "_" in base else path.split("/")[0]


def measure(rows):
    per_axis, all_sig, families = {}, collections.Counter(), set()
    for r in rows:
        families.add(family_key(r["path"]))
    for ax in AXES:
        sigs = [routing_signature(r[ax]) for r in rows]
        live = [s for s in sigs if s]
        c = collections.Counter(live)
        all_sig.update(c)
        per_axis[ax] = {
            "non_abstention_cells": len(live),
            "distinct_routing_signatures": len(c),
            "largest_signature": (c.most_common(1)[0][0] if c else None),
            "largest_share": (c.most_common(1)[0][1] if c else 0),
            "false_positive_detection_opportunity": len(live) > 0,
        }
    return per_axis, all_sig, families


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    hp = pathlib.Path(a.holdout)
    raw = hp.read_bytes()
    h = json.loads(raw)
    rows = h["rows"]
    per_axis, all_sig, families = measure(rows)
    total_cells = len(rows) * len(AXES)
    live_cells = sum(all_sig.values())
    top_sig, top_n = all_sig.most_common(1)[0]

    card = {
        "DERIVED_CLAIM": {
            "claim": "coverage profile of the frozen D368 40-row holdout: "
                     "non-abstention cells, distinct routing signatures per "
                     "axis, and which axes carry a false-positive detection "
                     "opportunity",
            "claim_class": "coverage / structural",
            "producer": "Orion",
            "subject_commit": h["subject"],
            "subject_tree": h["subject_tree"],
            "source_artefact": str(hp),
            "source_artefact_sha256": hashlib.sha256(raw).hexdigest(),
            "candidate_aggregate": h["candidate_aggregate"],
            "instrument": "kai-pm/d368_holdout_coverage_card.py",
            "instrument_sha256": hashlib.sha256(
                pathlib.Path(__file__).read_bytes()).hexdigest(),
            "query_or_command":
                "per-cell key = witness_type + '/' + leading `Label:` of the "
                "witness local_context + '->' + emitted value; abstentions "
                "(UNKNOWN, UNMEASURED) excluded; counted over 6 axes x 40 rows",
            "denominator_or_search_universe": f"{total_cells} axis cells "
                                              f"({len(rows)} rows x {len(AXES)} axes)",
            "unit": "CELLS and ROUTING SIGNATURES (not mechanisms, not "
                    "literal matches)",
            "raw_result": {
                "rows": len(rows),
                "total_cells": total_cells,
                "non_abstention_cells": live_cells,
                "distinct_routing_signatures": len(all_sig),
                "distinct_document_families": len(families),
                "largest_routing_signature": top_sig,
                "largest_routing_signature_cells": top_n,
                "per_axis": per_axis,
                "signature_tally": dict(all_sig.most_common()),
            },
            "derived_result":
                f"{live_cells} of {total_cells} cells non-abstention; "
                f"{sum(v for k, v in all_sig.items() if k.startswith('DATE_STAMP/reviewed'))}"
                f" of {live_cells} share the DATE_STAMP/reviewed signature "
                f"across two axes; "
                f"{sum(1 for ax in AXES if not per_axis[ax]['false_positive_detection_opportunity'])}"
                f" of {len(AXES)} axes carry NO false-positive detection "
                f"opportunity in this sample",
            "interpretation":
                "the sample's capacity to expose an INCORRECT NON-ABSTENTION "
                "VERDICT is confined to the axes with non-abstention cells",
            "limitations": [
                "the routing signature is the PRODUCER'S CONSTRUCTION, not an "
                "adjudicated causal equivalence class (rule 37). Rows sharing "
                "a key travelled the same classifier path; nothing here shows "
                "one cause produced them",
                "'document family' is a cruder construction still — a "
                "filename heuristic — and carries the same caveat",
                "axes with zero non-abstention cells are NOT untestable: "
                "OVER-ABSTENTION on them remains adjudicable from source",
                "says nothing about whether any verdict is correct",
                "no effective sample size is computed; there is no "
                "statistical model to support one",
            ],
            "independence_status":
                "SINGLE-PRODUCER REPRODUCIBLE EVIDENCE. Not independent "
                "corroboration. Under rule 33 as amended, this claim does not "
                "itself determine admission, closure, authority or "
                "irreversible scope, so an independent leg is not mandatory",
            "rerun":
                "python3 kai-pm/d368_holdout_coverage_card.py "
                "--holdout kai-pm/house_in_order_h2_v12/h2v12-holdout.json",
        }
    }
    text = json.dumps(card, indent=1)
    if a.out:
        pathlib.Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
