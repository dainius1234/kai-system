#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — EVIDENCE ENVELOPES AND THE CONSERVATION RULE.

Implements D367 4 (scope/applicability semantics) and 5 (the nine-field
witness trace). It ALSO gives Kai's root-cause ruling a mechanism:

    NO UNPROVEN PROMOTION. A downstream claim may not widen the subject,
    scope, polarity, evidence kind, temporal applicability or certainty
    of its supporting evidence without an explicit qualified promotion.
    If the promotion is absent, ABSTAIN.

This is not an addition to the frozen acceptance bar. D367 already
requires that every evidence fact, claim and verdict carry its own
applicability scope and a witness sufficient for independent
adjudication; the envelope is HOW that is delivered rather than
remembered. Seventeen defect classes were one shape -- an observation
recorded at one scope, reported at a wider one -- and remembering not to
do it failed seventeen times.

WHY A LATTICE AND NOT A CHECKLIST. Each dimension is ordered, so
"wider" is COMPUTED rather than judged. A rule that requires judgement
at the call site is a rule that will be applied to the case in front of
the author and to nothing else -- which is exactly how R5 was breached
four times in one instrument.
"""
from __future__ import annotations
import dataclasses
import json
import re
import unicodedata

# ── the ordered dimensions ────────────────────────────────────────────
# Each tuple is ordered NARROW -> WIDE. Moving right is a PROMOTION and
# requires an explicit, named justification. Moving left is always safe.

SCOPE_ORDER = ("SPAN", "SECTION", "WHOLE_FILE")
CERTAINTY_ORDER = ("ASSERTED", "OBSERVED", "VERIFIED")
TEMPORAL_ORDER = ("AT_COMMIT", "WINDOW", "UNBOUND")

# Polarity and subject do not form a lattice: any CHANGE is a violation,
# because there is no sense in which NEGATIVE is a wider POSITIVE.
POLARITIES = ("POSITIVE", "NEGATIVE", "NEUTRAL")


class PromotionError(AssertionError):
    """A claim tried to exceed its evidence. Raised, never returned --
    a violation that can be ignored by a caller is not a control."""


class SubjectError(PromotionError):
    """The subject grammar refused. D381 3.

    A subclass, so every existing `except PromotionError` still catches a
    subject violation -- a fail-closed rule that a caller could step over
    by catching the narrower parent would not be fail-closed.
    """


# ── D381 3: THE CLOSED SUBJECT GRAMMAR ────────────────────────────────
#
# INC-2026-09-18-31: HOUSE_H2 conflated DOCUMENT-LEVEL APPLICABILITY with
# SEMANTIC SELF SUBJECT. Every witness inherited `subject="SELF"` from a
# dataclass default, so the subject dimension was CONSTANT in production
# and no corpus row could discriminate a correct consumer from a broken
# one. The default is gone (see REQUIRED below) and the grammar is closed.
#
# THE FOUR CANONICAL FORMS, AND NO OTHERS:
#
#     SELF
#     AMBIGUOUS
#     OTHER:DOCUMENT:<normalised repo-relative path>
#     OTHER:GIT_COMMIT:<full lower-case 40-hex commit>
#
# `OTHER:<path>` is the TRANSITIONAL legacy spelling of the DOCUMENT form
# and is deliberately NARROWER than the canonical payload: it forbids a
# colon, so a malformed typed form can never reach it. New producers emit
# the canonical typed form.

SUBJECT_SELF = "SELF"
SUBJECT_AMBIGUOUS = "AMBIGUOUS"
DOCUMENT_PREFIX = "OTHER:DOCUMENT:"
GIT_COMMIT_PREFIX = "OTHER:GIT_COMMIT:"
LEGACY_PREFIX = "OTHER:"

FULL_COMMIT = re.compile(r"\A[0-9a-f]{40}\Z")


def valid_repo_relative_path(p) -> bool:
    """D380 6.10: relative - POSIX '/' - no leading '/' - no '.' or '..'
    segment - no backslash alias - valid UTF-8 - Unicode NFC."""
    if not isinstance(p, str) or not p:
        return False
    try:
        p.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if p != unicodedata.normalize("NFC", p):
        return False
    if "\\" in p or p.startswith("/"):
        return False
    return all(seg not in ("", ".", "..") for seg in p.split("/"))


def parse_subject(subject):
    """The D381 3.1 NORMATIVE PARSER. Returns (kind, payload) or raises.

    A CLOSED ORDERED MATCH, NOT A HEURISTIC. A generic "uppercase token"
    test was expressly rejected: it would reserve arbitrary uppercase
    repository path names as though they were type tags.

    RULES 3 AND 4 ARE TERMINAL ON THEIR PREFIX. That is the whole defect
    being closed here: a malformed `OTHER:GIT_COMMIT:not-a-commit` must
    REFUSE, and must never fall through to legacy parsing where it would
    be silently accepted as a document path.
    """
    if not isinstance(subject, str):
        raise SubjectError(f"subject must be a string, got {type(subject).__name__}")
    # 1, 2 -- the two bare forms
    if subject == SUBJECT_SELF:
        return ("SELF", None)
    if subject == SUBJECT_AMBIGUOUS:
        return ("AMBIGUOUS", None)
    # 3 -- typed DOCUMENT, TERMINAL
    if subject.startswith(DOCUMENT_PREFIX):
        payload = subject[len(DOCUMENT_PREFIX):]
        if not valid_repo_relative_path(payload):
            raise SubjectError(
                f"malformed OTHER:DOCUMENT payload {payload!r} — REFUSED. "
                f"A typed prefix is TERMINAL: it does not fall through to "
                f"legacy parsing (D381 3.1 rule 3).")
        return ("DOCUMENT", payload)
    # 4 -- typed GIT_COMMIT, TERMINAL
    if subject.startswith(GIT_COMMIT_PREFIX):
        payload = subject[len(GIT_COMMIT_PREFIX):]
        if not FULL_COMMIT.match(payload):
            raise SubjectError(
                f"malformed OTHER:GIT_COMMIT payload {payload!r} — REFUSED. "
                f"A full lower-case 40-hex commit is required, and a typed "
                f"prefix is TERMINAL (D381 3.1 rule 4).")
        return ("GIT_COMMIT", payload)
    # 5 -- the transitional legacy DOCUMENT alias, deliberately narrower
    if subject.startswith(LEGACY_PREFIX):
        payload = subject[len(LEGACY_PREFIX):]
        if not payload or ":" in payload or not valid_repo_relative_path(payload):
            raise SubjectError(
                f"legacy OTHER: payload {payload!r} — REFUSED. The "
                f"transitional alias admits a non-empty repo-relative path "
                f"containing NO ':' (D381 3.1 rule 5). A path containing a "
                f"colon is representable only as OTHER:DOCUMENT:<path>.")
        return ("DOCUMENT", payload)
    # 6 -- everything else
    raise SubjectError(
        f"unknown subject {subject!r} — REFUSED. The grammar is CLOSED: "
        f"SELF, AMBIGUOUS, OTHER:DOCUMENT:<path>, OTHER:GIT_COMMIT:<40hex>, "
        f"or the transitional OTHER:<path> (D381 3.1 rule 6).")


def subject_is_self(subject) -> bool:
    """Validated SELF. The ONLY way an axis asks the question.

    D381 5.5: consumers GATE on the producer-bound subject and never
    reinterpret it. This reads the field; it does not re-derive it from a
    label, a local_context, a path or an applicability scope.
    """
    return parse_subject(subject)[0] == "SELF"


class _Required:
    """Sentinel: `subject` has NO default and must be stated explicitly."""
    __slots__ = ()

    def __repr__(self):
        return "<REQUIRED>"


REQUIRED = _Required()


@dataclasses.dataclass(frozen=True)
class Witness:
    """The nine mandatory fields of D367 5, plus the envelope dimensions.

    `value` is the EXACT matched token, never a description. v1.1
    recorded the static string "cites a commit sha" for every VALIDITY
    cell; an adjudicator could not audit the cell from the package and
    had to open the source document. So could I.
    """
    witness_type: str          # what kind of evidence this is
    witness_value: str         # the EXACT token or value matched
    source_path: str           # the document it came from
    source_selector: str       # stable selector: "L<line>" or "L<a>-L<b>"
    local_context: str         # surrounding text, enough to judge it
    applicability_scope: str   # SPAN | SECTION | WHOLE_FILE
    evidence_total: int        # how many candidate rows existed
    evidence_shown: int        # how many are carried here
    truncated: bool            # explicit, never inferred
    # envelope dimensions beyond scope
    polarity: str = "NEUTRAL"
    certainty: str = "OBSERVED"
    temporal: str = "AT_COMMIT"
    # NO DEFAULT. D381 3 / INC-31: `subject = "SELF"` here is precisely how
    # 167 witnesses inherited a semantic claim nobody made about them. The
    # sentinel keeps the field order stable while making omission FAIL.
    subject: str = REQUIRED

    def __post_init__(self):
        if self.subject is REQUIRED:
            raise SubjectError(
                f"subject NOT STATED for {self.witness_type} witness at "
                f"{self.source_path} {self.source_selector}. There is no "
                f"default: a witness's semantic subject is determined once, "
                f"at the governed producer boundary (D381 5.5).")
        parse_subject(self.subject)          # fail-closed, D381 3.1
        if self.applicability_scope not in SCOPE_ORDER:
            raise PromotionError(f"unknown scope {self.applicability_scope!r}")
        if self.polarity not in POLARITIES:
            raise PromotionError(f"unknown polarity {self.polarity!r}")
        if self.certainty not in CERTAINTY_ORDER:
            raise PromotionError(f"unknown certainty {self.certainty!r}")
        if self.temporal not in TEMPORAL_ORDER:
            raise PromotionError(f"unknown temporal {self.temporal!r}")
        if self.evidence_shown > self.evidence_total:
            raise PromotionError("evidence_shown exceeds evidence_total")
        # R10: an excerpt that does not announce its own partiality reads
        # like the whole thing. The flag is DERIVED, so it cannot drift.
        if self.truncated != (self.evidence_shown < self.evidence_total):
            raise PromotionError(
                f"truncated={self.truncated} contradicts "
                f"{self.evidence_shown}/{self.evidence_total}")

    def asdict(self):
        return dataclasses.asdict(self)


# ── declared promotions ───────────────────────────────────────────────
# The ONLY legal widenings. Each names the justification that must hold.
# A promotion absent from this table cannot be performed at all, so a
# new one is a visible, reviewable change rather than a call-site choice.

DECLARED_PROMOTIONS = {
    ("SPAN", "WHOLE_FILE"):
        "DOCUMENT_LEVEL_BINDING: the witness is a structured statement "
        "whose subject is explicitly the document as a whole",
    ("SECTION", "WHOLE_FILE"):
        "DOCUMENT_LEVEL_BINDING: as above",
    ("OBSERVED", "VERIFIED"):
        "INDEPENDENT_CONFIRMATION: the observation was checked against a "
        "source that could have refuted it",
}


def _rank(order, v):
    return order.index(v)


def conserve(witness: Witness, *, claim_scope: str, claim_polarity: str,
             claim_certainty: str = None, claim_temporal: str = None,
             claim_subject: str = None, promotion: str = None):
    """Return None if the claim is within its evidence; raise otherwise.

    `promotion` names the qualified justification. It is checked against
    DECLARED_PROMOTIONS -- a caller cannot invent one by passing a
    plausible string, because the (from, to) pair must be declared AND
    the justification must match the declaration.
    """
    claim_certainty = claim_certainty or witness.certainty
    claim_temporal = claim_temporal or witness.temporal
    claim_subject = claim_subject or witness.subject

    # subject and polarity: any change is a violation, never a promotion
    if claim_subject != witness.subject:
        raise PromotionError(
            f"SUBJECT CHANGED: witness binds {witness.subject!r}, claim "
            f"asserts {claim_subject!r}. This is the MISBINDING axis "
            f"({witness.source_path} {witness.source_selector}).")
    if claim_polarity != witness.polarity and witness.polarity != "NEUTRAL":
        raise PromotionError(
            f"POLARITY CHANGED: witness is {witness.polarity}, claim is "
            f"{claim_polarity} ({witness.source_path} "
            f"{witness.source_selector}: {witness.witness_value!r})")

    for order, w, c, name in (
            (SCOPE_ORDER, witness.applicability_scope, claim_scope, "SCOPE"),
            (CERTAINTY_ORDER, witness.certainty, claim_certainty, "CERTAINTY"),
            (TEMPORAL_ORDER, witness.temporal, claim_temporal, "TEMPORAL")):
        if _rank(order, c) <= _rank(order, w):
            continue                      # narrowing or equal: always safe
        key = (w, c)
        if key not in DECLARED_PROMOTIONS:
            raise PromotionError(
                f"UNDECLARED PROMOTION on {name}: {w} -> {c} is not a "
                f"declared widening ({witness.source_path} "
                f"{witness.source_selector})")
        if promotion != DECLARED_PROMOTIONS[key]:
            raise PromotionError(
                f"UNPROVEN PROMOTION on {name}: {w} -> {c} requires "
                f"{DECLARED_PROMOTIONS[key][:40]}…, and none was supplied "
                f"({witness.source_path} {witness.source_selector})")
    return None


def claim(witness: Witness, value: str, *, scope=None, polarity=None,
          certainty=None, temporal=None, subject=None, promotion=None,
          rationale=""):
    """Build a verdict/evidence cell from a witness, conserving the
    envelope. On violation the caller gets an exception, not a value --
    there is deliberately no 'return the claim anyway' path."""
    scope = scope or witness.applicability_scope
    polarity = polarity or witness.polarity
    conserve(witness, claim_scope=scope, claim_polarity=polarity,
             claim_certainty=certainty, claim_temporal=temporal,
             claim_subject=subject, promotion=promotion)
    cell = {"value": value, "rationale": rationale, "witness": witness.asdict(),
            "claim_scope": scope, "claim_polarity": polarity}
    if promotion:
        cell["promotion"] = promotion
    return cell


def abstain(axis: str, needed: str, observed: str, why: str):
    """The only legal output when a promotion is absent.

    An abstention is NOT negative evidence and may never be consumed as
    an exclusion criterion (D340 7 / D358). It carries what WOULD have
    been required so the gap is legible instead of silent."""
    return {"value": "UNKNOWN", "abstention": True,
            "would_require": needed, "observed": observed,
            "rationale": why, "witness": None}


def unmeasured(axis: str, prerequisite: str):
    """R11: no subject, no observation. Distinct from UNKNOWN -- this is
    a capability failure, not an epistemic one."""
    return {"value": "UNMEASURED", "abstention": True,
            "failed_prerequisite": prerequisite, "witness": None,
            "rationale": f"{axis} not measurable: {prerequisite}"}


def sidecar_ref(digest: str, count: int):
    """D367 5: oversized evidence may live in a hash-bound sidecar.
    Silent truncation is forbidden; a reference is not silence."""
    return {"sidecar_sha256": digest, "rows": count}


if __name__ == "__main__":
    print(json.dumps({"scope_order": SCOPE_ORDER,
                      "certainty_order": CERTAINTY_ORDER,
                      "temporal_order": TEMPORAL_ORDER,
                      "declared_promotions": {f"{a}->{b}": v for (a, b), v
                                              in DECLARED_PROMOTIONS.items()}},
                     indent=1))
