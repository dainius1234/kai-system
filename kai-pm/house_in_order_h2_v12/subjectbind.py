#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — CLAIM SUBJECT BINDING AND POLARITY.

Repairs D12 and D13. Emits EVIDENCE FACTS ONLY; AUTHORITY verdicts stay
DEFERRED_TO_H3 and are never formed here.

D12 -- SUBJECT MISBINDING. v1.1's SELF_PROSE admitted a bare
sentence-initial `it` and bound it to SELF with NO ANTECEDENT
RESOLUTION. In README.md the source reads:

    **Status lives in one place:** [`kai-pm/UH_PROGRESS_TRACKER.md`](…).
    It is the source of truth for UH work …

so `It` refers to UH_PROGRESS_TRACKER.md, named in the previous
sentence. A document ATTRIBUTING authority to another was recorded as
CLAIMING it -- the MISBINDING axis by name. A pronoun now binds to SELF
only when no nearer antecedent exists; if a document is named in the
preceding window it binds to OTHER:<path>, and if that is ambiguous it
binds to AMBIGUOUS and earns nothing.

D13 -- POLARITY INVERSION. `\\bauthoritative\\b` matches inside
`non-authoritative`, because a hyphen is a word boundary, while
AUTH_NEG required the separate word `not`. So:

    'This document is non-authoritative.'  ->  SELF_ASSERTS_AUTHORITY

A document explicitly disclaiming authority was recorded as claiming it.
No SELF-bound corpus row triggered it, so it was latent -- and latent in
the dangerous direction. Negation is now detected FIRST and covers
attached negative prefixes as well as separate negative words.
"""
from __future__ import annotations
import re

# ── polarity ─────────────────────────────────────────────────────────
AUTH_TERM = re.compile(r"authoritative|single source of truth|"
                       r"source of truth|source of programme truth", re.I)
# NEGATION, tested FIRST. Two families, because v1.1 had only the second:
#   attached prefix : non-authoritative, un-authoritative, nonauthoritative
#   separate word   : not authoritative, never the source of truth
NEG_PREFIX = re.compile(r"\b(?:non|un)-?(?=authoritative)", re.I)
NEG_WORD = re.compile(r"\b(?:not|never|no longer|isn't|is not|aren't)\b"
                      r"[^.;]{0,40}?(?=authoritative|source of)", re.I)


def polarity_of(sentence):
    """POSITIVE / NEGATIVE / None. Negation wins, and is tested first."""
    if not AUTH_TERM.search(sentence):
        return None
    if NEG_PREFIX.search(sentence) or NEG_WORD.search(sentence):
        return "NEGATIVE"
    return "POSITIVE"


# ── subject binding ──────────────────────────────────────────────────
MDPATH = re.compile(r"`?([A-Za-z0-9_./-]+\.md)`?")
# A CONTROLLED FIELD binds to SELF: the label's subject is the document.
SELF_FIELD = re.compile(r"^\s{0,3}[>\-*|]?\s*[*_`]{0,2}\s*"
                        r"(?:status|authority)\s*[*_`]{0,2}\s*:", re.I)
# EXPLICIT self-reference binds to SELF regardless of nearby paths.
SELF_EXPLICIT = re.compile(r"^\s*(?:this (?:document|file|register|entry|page))\b",
                           re.I)
# A BARE PRONOUN binds to SELF only if nothing nearer is named.
BARE_PRONOUN = re.compile(r"^\s*it\b", re.I)

ANTECEDENT_WINDOW = 400          # characters of preceding text consulted


def _sentences(text):
    """(start, text) pairs. Split on line and sentence boundaries so a
    selector can be computed for each."""
    out, pos = [], 0
    for chunk in re.split(r"(?<=[.!?])\s+|\n", text):
        i = text.find(chunk, pos)
        if i < 0:
            i = pos
        if chunk.strip():
            out.append((i, chunk.strip()))
        pos = i + len(chunk)
    return out


def bind_subject(text, start, sentence, self_path):
    """Which document does this sentence's claim describe?

    Returns (subject, why). SELF only when earned.
    """
    if SELF_FIELD.match(sentence):
        return "SELF", "controlled field; the label's subject is the document"
    if SELF_EXPLICIT.match(sentence):
        return "SELF", "explicit self-reference"

    named = MDPATH.findall(sentence)
    if named:
        others = [p for p in named if p != self_path]
        if len(set(others)) == 1:
            return f"OTHER:{others[0]}", "the sentence names another document"
        if others:
            return "AMBIGUOUS", f"the sentence names {len(set(others))} documents"
        return "SELF", "the sentence names this document"

    if BARE_PRONOUN.match(sentence):
        # D12: resolve the antecedent before assuming SELF.
        window = text[max(0, start - ANTECEDENT_WINDOW):start]
        prior = MDPATH.findall(window)
        prior = [p for p in prior if p != self_path]
        if prior:
            return (f"OTHER:{prior[-1]}",
                    f"bare pronoun; nearest antecedent is {prior[-1]}, named "
                    f"in the preceding text -- NOT this document")
        return "SELF", "bare pronoun with no nearer antecedent"

    return "AMBIGUOUS", "no resolvable subject"


def bind_claims(path, text, head_bytes=None):
    """All authority claims with subject and polarity. Evidence only.

    Returns (claims, stats) where each claim is a dict carrying its own
    selector and context, so the artefact can adjudicate its own cell
    (D14/D15) instead of recording a static description.

    SCANS THE WHOLE DOCUMENT BY DEFAULT, and that is not incidental.
    The first v1.2 draft imposed a 6000-byte window here by analogy with
    Pass A. v1.1 had no such window on this extractor -- it received the
    full text -- and the window silently destroyed two claims Kai had
    adjudicated CORRECT: PHASE1_READINESS.md's self-declaration at byte
    18,877 and DECISIONS.md's at byte 287,981. Both fell to
    NO_SELF_CLAIM.

    A repair that only proves the corrected case can destroy the property
    it was protecting. That is the whole reason a repair must also prove
    a same-family counterexample, and it is why the five rows Kai
    adjudicated correct are asserted unchanged in cal_fixtures.
    """
    head = text if head_bytes is None else text[:head_bytes]
    claims = []
    for start, sent in _sentences(head):
        pol = polarity_of(sent)
        if pol is None:
            continue
        subject, why = bind_subject(head, start, sent, path)
        claims.append({
            "polarity": pol, "subject": subject, "subject_reason": why,
            "selector": f"L{head[:start].count(chr(10)) + 1}",
            "text": sent[:200],
        })
    stats = {"total": len(claims),
             "self_bound": sum(1 for c in claims if c["subject"] == "SELF"),
             "other_bound": sum(1 for c in claims
                                if c["subject"].startswith("OTHER:")),
             "ambiguous": sum(1 for c in claims
                              if c["subject"] == "AMBIGUOUS")}
    return claims, stats


def authority_claim(claims):
    """SELF-bound claims only. NEVER an authority state -- an evidence
    fact. AUTHORITY verdicts remain DEFERRED_TO_H3."""
    s = {c["polarity"] for c in claims if c["subject"] == "SELF"}
    if {"POSITIVE", "NEGATIVE"} <= s:
        return "CONFLICTING_SELF_CLAIMS"
    if "POSITIVE" in s:
        return "SELF_ASSERTS_AUTHORITY"
    if "NEGATIVE" in s:
        return "SELF_ASSERTS_NON_AUTHORITY"
    return "NO_SELF_CLAIM"


def determining_claims(claims):
    """The SELF-bound rows that DETERMINED the fact.

    D14: v1.1 truncated evidence to [:6] with no declaration, and for
    DECISIONS.md -- 43 claims, exactly 1 SELF-bound -- the artefact
    carried ZERO SELF rows. The conclusion contradicted its own visible
    evidence and the decisive witness was unrecoverable. The determining
    rows are now ALWAYS carried, whatever the total.
    """
    return [c for c in claims if c["subject"] == "SELF"]
