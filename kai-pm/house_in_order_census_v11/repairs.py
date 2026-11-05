#!/usr/bin/env python3
"""REPAIR-IMPACT EVIDENCE — kept ORTHOGONAL to the analyser's semantics.

KAI'S D342 CORRECTION. A repair whose current-subject effect is nil must
NOT become another ontology value; that mixes two separate things again.
Semantics live in `claims.ALPHABETS`. Repair impact lives here, in a
structured record:

    RULE_STATUS            = CORRECTED_AND_QUALIFIED
    CURRENT_SUBJECT_EFFECT = NONE | OPERATIONS_RECLASSIFIED

meaning: the old rule was unsound, the corrected rule is proven by
calibration, and here is what it did -- or did not do -- to THIS subject.

EVERY FIGURE BELOW IS MEASURED ON THE SUBJECT BEING ANALYSED. None is a
constant kept beside the code. A hand-maintained impact claim is exactly
the "list beside the thing" defect R5 forbids, and it would rot the
moment the tree changed.
"""
from __future__ import annotations
import pathlib

import claims as C


def _suffix_match(literal: str, docs) -> bool:
    """Could this absolute/URI literal denote a tracked document?"""
    for d in docs:
        if literal == d or literal.endswith("/" + d):
            return True
    return False


def measure(ops, docs, tracked_all):
    """Returns the repair-evidence block for this exact subject."""
    docset = set(docs)
    fixed_ops = []
    for o in ops:
        lex_dyn = any(("$" in f or "{" in f or "*" in f) for f in o.frags)
        fixed_ops.append((o, not lex_dyn and not o.dynamic))

    # ── R1: URI syntax is not remote semantics ───────────────────────
    uri = [o for o in ops
           if C.path_syntax(o.frags, o.dynamic) == "URI_SYNTAX"]
    uri_would_have_been_excluded = len(uri)
    uri_could_denote_a_document = sum(
        1 for o in uri if _suffix_match("/".join(o.frags), docs))

    # ── R2: absolute syntax is not "outside the repository" (R6) ─────
    absolute = [o for o in ops
                if C.path_syntax(o.frags, o.dynamic) == "ABSOLUTE"]
    abs_could_denote_a_document = sum(
        1 for o in absolute if _suffix_match("/".join(o.frags), docs))

    # ── R3: read/write target reclassified conservatively ────────────
    # Exactly the operations v1.0's suffix resolution WOULD have bound
    # to a tracked document, but whose prefix is not resolvable.
    reclassified = []
    for o, fully_fixed in fixed_ops:
        if fully_fixed or not o.frags:
            continue
        if not any(f.endswith(".md") for f in o.frags):
            continue
        if C._resolve_exact(o.frags, docset) is not None:
            reclassified.append({
                "src": o.src, "line": o.line, "mode": o.mode,
                "fragments": list(o.frags),
                "v10_would_have_bound_to": C._resolve_exact(o.frags, docset),
                "v11_disposition": o.disposition})

    def block(rule_id, was, now, affected, detail, extra=None):
        b = {
            "rule_id": rule_id,
            "RULE_STATUS": "CORRECTED_AND_QUALIFIED",
            "unsound_rule_in_v1_0": was,
            "corrected_rule_in_v1_1": now,
            "operations_affected_on_this_subject": affected,
            "CURRENT_SUBJECT_EFFECT": "NONE" if affected == 0
                                      else "OPERATIONS_RECLASSIFIED",
            "how_measured": detail,
            "claim_level_effect_established_by": "compare-v10-v11.json",
        }
        if extra:
            b.update(extra)
        return b

    return {
        "note": "Repair impact is orthogonal to the analyser's semantic "
                "alphabets. No repair introduces an ontology value.",
        "repairs": [
            block(
                "URI_SYNTAX_NOT_REMOTE_SEMANTICS",
                "any target containing '://' was EXCLUDED_FROM_T as a "
                "'remote URI, not a repository path'",
                "URI syntax is a recorded observation only; the literal "
                "is reasoned about as the path string it is",
                uri_could_denote_a_document,
                "counted operations whose path syntax is URI_SYNTAX and "
                "whose literal has a tracked document as a path suffix, "
                "i.e. those v1.0 excluded but v1.1 must leave open",
                {"uri_syntax_operations_on_this_subject":
                 uri_would_have_been_excluded}),
            block(
                "ABSOLUTE_PATH_NOT_OUTSIDE_REPOSITORY",
                "any target beginning '/' was EXCLUDED_FROM_T as an "
                "'absolute system path, outside the repository target "
                "domain'",
                "an absolute literal excludes a target only when that "
                "target is not a path SUFFIX of it",
                abs_could_denote_a_document,
                "counted absolute-syntax operations whose literal has a "
                "tracked document as a path suffix, i.e. those v1.0 "
                "falsely excluded",
                {"absolute_operations_on_this_subject": len(absolute)}),
            block(
                "READ_TARGET_RECLASSIFIED_CONSERVATIVELY",
                "a dynamic-prefix path whose literal suffix matched a "
                "tracked file was RESOLVED to that file",
                "document relevance is proven by the fixed '.md' "
                "component, but the exact tracked target is NOT proven, "
                "so the disposition is UNRESOLVED_TARGET",
                len(reclassified),
                "counted operations with proven .md relevance, an "
                "unresolvable prefix, and a literal suffix that v1.0's "
                "exact-resolution step would have bound to a document",
                {"reclassified_operations": reclassified,
                 "semantic_note":
                     "These are NOT read relations that were erased. "
                     "Each operation is preserved in full; what was "
                     "withdrawn is the unproven claim about WHICH "
                     "tracked document it touches. Relevance proven, "
                     "exact target unproven."}),
        ],
    }
