#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — FINAL BLIND HOLDOUT. Selection rule frozen in D367 9.

    key = sha256("H2FINAL-D367:"
                 + "86a1399e6e31477ba67cd38c12d22627a8b4d6ef"   # D366
                 + ":" + FINAL_CANDIDATE_AGGREGATE
                 + ":" + path)
    sort ascending, select the first 40

The candidate aggregate did not exist when the rule was frozen, so the
sample could not be known during implementation -- while the rule itself
was committed before a line of repair code was written. If a candidate
fails and code changes, the new identity deterministically yields a NEW
sample, and previously revealed rows become regression evidence only.

NOT SELF-ADJUDICATED. Kai adjudicates all 40 across all six axes plus
consequential evidence facts. THIS SCRIPT COMPUTES NO AGREEMENT FIGURE,
and any figure Orion computed would carry no admission weight (D367 10).
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

D366_COMMIT = "86a1399e6e31477ba67cd38c12d22627a8b4d6ef"
SALT = "H2FINAL-D367:"
SIZE = 40


class HoldoutInputError(AssertionError):
    """I1-A / I1-B refused. Raised, never returned."""


def select(paths, candidate_aggregate):
    return sorted(paths, key=lambda p: hashlib.sha256(
        f"{SALT}{D366_COMMIT}:{candidate_aggregate}:{p}".encode()
    ).hexdigest())[:SIZE]


# ── I1-A — the seed is the VALIDATED STAGE-A IDENTITY ─────────────────
#
# v1.2 computed  aggregate = sha256(read_bytes(MANIFEST.sha256))  and used
# it as CANDIDATE_AGGREGATE. That manifest CONTAINS EXECUTION OUTPUT -- ten
# entries, nine source modules plus h2v12-classification.json -- so the
# blind sample was seeded by the candidate's own result. Mutate the
# evidence, and the sample that audits the evidence moves.
#
# D381 §18 supersedes the definition for this lineage:
#     FINAL_CANDIDATE_AGGREGATE = canonical H2_STAGE_A_V2 stage_a_identity
# D367 §9's selection equation is UNCHANGED. Only what is substituted into
# it changes.
#
# THE AUTHORITY BOUNDARY (D381 §10). stage_identity DETERMINES whether a
# descriptor is valid for production; holdout CONSUMES ONLY the validated
# identity. The V1+PRODUCTION refusal, the V2 governance contract and the
# calibration zero-weight rule are NOT duplicated here, and must not be.
def final_candidate_aggregate(descriptor_path):
    """The validated Stage-A identity, or REFUSE. Never a manifest digest."""
    import stage_identity as SI
    p = pathlib.Path(descriptor_path)
    if not p.is_file():
        raise HoldoutInputError(
            f"REFUSE: no Stage-A descriptor at {descriptor_path}. The blind "
            f"selection seed is the validated Stage-A identity; there is no "
            f"fallback to a manifest, a package digest or result bytes.")
    try:
        desc = SI.parse_descriptor_bytes(SI._read_regular_once(p))
    except SI.StageIdentityError as e:
        raise HoldoutInputError(str(e)) from None
    if desc.get("mode") != "PRODUCTION":
        raise HoldoutInputError(
            f"REFUSE: Stage-A descriptor mode is {desc.get('mode')!r}. A "
            f"CALIBRATION identity carries ZERO holdout weight (D381 §13) "
            f"and may never seed the blind 40.")
    return SI.stage_a_identity(desc)          # validates, then derives


# ── I1-B — the universe is the FROZEN SUBJECT TREE ────────────────────
def _validated_unique(paths, side):
    """v4.1 C1 path order, then uniqueness, for ONE population, alone.

    Each path must already be canonical (D380 §6.10; an NFD path REFUSES at
    the NFC step and never reaches duplicate comparison). A duplicate on
    this side REFUSES here, BEFORE any comparison with the other side, so a
    duplicate present on BOTH sides can never cancel out (F12)."""
    import stage_identity as SI
    seen = set()
    for p in paths:
        try:
            q = SI._norm_path(p)
        except SI.StageIdentityError as e:
            raise HoldoutInputError(f"REFUSE BEFORE SELECTION ({side}): {e}") from None
        if q in seen:
            raise HoldoutInputError(
                f"REFUSE BEFORE SELECTION: duplicate {side} path {q!r} "
                f"(F12 — refused independently, before reconciliation)")
        seen.add(q)
    return seen


def plan_selection(tree_paths, output_paths, aggregate):
    """THE single selection decision (v4.5 §17, v4.1 C5, Kai Q8).

    1-4 validate canonical form and uniqueness of the TREE population;
    5-6 the same, independently, for the OUTPUT population; 7-8 exact sets
    and exact cardinalities; 9 only then select — from the TREE. Candidate
    output decides only WHETHER reconciliation succeeds, never the
    population or the seed. The aggregate is supplied, never derived here.
    """
    t = _validated_unique(tree_paths, "tree")
    o = _validated_unique(output_paths, "output")
    if t != o or len(tree_paths) != len(output_paths):
        raise HoldoutInputError(
            f"REFUSE BEFORE SELECTION: candidate output does not reconcile with "
            f"the frozen subject tree. tree-only={sorted(t - o)[:5]} "
            f"output-only={sorted(o - t)[:5]} "
            f"(tree {len(tree_paths)} vs output {len(output_paths)})")
    return select(sorted(tree_paths), aggregate)


def reconcile(tree_paths, output_paths):
    """Kept as the pre-selection predicate only; it now delegates to the same
    independent-uniqueness rule, so it cannot accept what plan_selection
    refuses."""
    _validated_unique(tree_paths, "tree")
    _validated_unique(output_paths, "output")
    if sorted(tree_paths) != sorted(output_paths):
        raise HoldoutInputError("REFUSE BEFORE SELECTION: populations differ")
    return True


def read_tree_paths(path, desc):
    """v4.1 C5 step 5: the EXACT --tree-paths bytes. No strip, no blank-skip.
    The bytes must be exactly the D380 §6.6 construction, and their identity
    and count must equal the Stage-A tree_paths block."""
    import stage_identity as SI
    try:
        data = SI._read_regular_once(path)
        text = data.decode("utf-8")
    except (SI.StageIdentityError, UnicodeDecodeError) as e:
        raise HoldoutInputError(f"REFUSE: --tree-paths unreadable or not UTF-8: {e}") from None
    if not text.endswith("\n"):
        raise HoldoutInputError("REFUSE: --tree-paths lacks the final LF (D380 §6.6)")
    paths = text[:-1].split("\n")
    _validated_unique(paths, "tree")
    canon = "".join(p + "\n" for p in sorted(paths, key=lambda p: p.encode("utf-8")))
    if canon.encode("utf-8") != data:
        raise HoldoutInputError(
            "REFUSE: --tree-paths bytes are not the exact D380 §6.6 construction "
            "(order, CRLF, whitespace or blank line)")
    if SI.sha256_hex(data) != desc["tree_paths"]["tree_paths_identity"] or \
            len(paths) != desc["tree_paths"]["population"]:
        raise HoldoutInputError(
            "REFUSE: --tree-paths identity/count does not equal the Stage-A "
            "tree_paths block")
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--stage-a", required=True, dest="stage_a",
                    help="the PRODUCTION Stage-A descriptor. Its VALIDATED "
                         "identity is FINAL_CANDIDATE_AGGREGATE (D381 18).")
    ap.add_argument("--stage-b", required=True, dest="stage_b",
                    help="the ORIGINAL classification Stage-B binding")
    ap.add_argument("--expected-binding-sha256", required=True,
                    dest="binding_sha256",
                    help="the independently held anchor for --stage-b")
    ap.add_argument("--tree-paths", required=True, dest="tree_paths",
                    help="the immutable frozen subject tree path population "
                         "(I1-B). Selection is from THIS, never from output.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import stage_identity as SI
    aggregate = final_candidate_aggregate(a.stage_a)          # 1. I1-A
    desc = SI.parse_descriptor_bytes(SI._read_regular_once(a.stage_a))
    try:                                                      # 2-4. one read
        _rb, res, _binding = SI.consume_bound_artifact(
            a.result, a.stage_b, a.binding_sha256, stage_a_desc=desc,
            producer_component="CLASSIFICATION")
    except SI.StageIdentityError as e:
        raise HoldoutInputError(f"REFUSE: result does not verify: {e}") from None
    if res["subject"] != desc["subject"]["commit"]:
        raise HoldoutInputError("REFUSE: result subject is not the Stage-A subject")
    tree_paths = read_tree_paths(a.tree_paths, desc)          # 5.
    output_paths = [r["path"] for r in res["rows"]]
    chosen = plan_selection(tree_paths, output_paths, aggregate)   # 6.
    by_path = {r["path"]: r for r in res["rows"]}

    payload = {
        "holdout": "H2FINAL-D367", "size": SIZE,
        "selection_rule": f'sha256("{SALT}" + "{D366_COMMIT}" + ":" + '
                          f'CANDIDATE_AGGREGATE + ":" + path), ascending, '
                          f'first {SIZE}',
        "candidate_aggregate": aggregate,
        "d366_commit": D366_COMMIT,
        "subject": res["subject"], "subject_tree": res["subject_tree"],
        "population": res["population"],
        "adjudication": "KAI. Orion computes no agreement figure and this "
                        "package carries none. Orion self-adjudication has "
                        "ZERO final admission weight (D367 10).",
        "evaluation_rule": {
            "incorrect non-abstention verdict": "BLOCKER",
            "false evidence fact": "BLOCKER",
            "unsupported scope widening": "BLOCKER",
            "forbidden or undeclared state emitted": "BLOCKER",
            "determining witness absent or silently truncated": "BLOCKER",
            "genuinely ambiguous source evidence": "UNRESOLVED",
            "UNKNOWN": "ABSTENTION — never negative evidence",
            "earnable positive emitted as UNKNOWN":
                "OVER_ABSTENTION / coverage finding, not automatically a "
                "safety blocker",
        },
        "rows": [by_path[p] for p in chosen],
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"BLIND HOLDOUT — {len(chosen)} of {res['population']} documents")
    print(f"  candidate aggregate {aggregate}")
    print(f"  rule frozen in D367 before any repair code existed")
    print(f"  written to {a.out}")
    print("  NOT SELF-ADJUDICATED. No agreement figure computed.")


if __name__ == "__main__":
    main()
