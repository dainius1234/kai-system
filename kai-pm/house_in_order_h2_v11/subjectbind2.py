#!/usr/bin/env python3
"""CLAIM SUBJECT BINDING — v1.1. Carried forward from HOUSE_H2 v1.0
UNCHANGED in logic: it was calibrated there and its boundary pairs
(13/13b) hold. Copied rather than imported so the v1.1 package is
self-contained and portable.

Original header follows.

CLAIM SUBJECT BINDING — deliberately small, abstention-first.

Kai's ruling: this is a CLAIM EXTRACTOR, not an English-understanding
engine. No pronoun resolution, no grammatical NLP. If the subject is not
STRUCTURALLY clear, abstain.

Earned by D338: 'chronology only' in CODE_AUDIT_MASTER describes the
registers it SUPERSEDES, not itself -- a fact I recorded at D326 and
then built a classifier that ignored.

PRIORITY (first match wins):
  1 structured metadata / controlled field      -> SELF
  2 explicit SELF subject                        -> SELF
  3 explicit OTHER named title/path              -> OTHER
  4 structurally explicit heading scope          -> that scope
  5 otherwise                                    -> AMBIGUOUS_SUBJECT
"""
from __future__ import annotations
import pathlib, re

SUBJECTS = ("SELF", "OTHER", "AMBIGUOUS_SUBJECT", "QUOTED_NOT_DECLARATION")

# `\bfinal\b` REMOVED. 98 of 103 self-claims fired on it, 91 from
# "Status: CONFIRMED — pending final consolidation", which says
# consolidation is PENDING -- the opposite of an authority claim. One
# fired AUTHORITY_POSITIVE on "It is not the final repository total."
# "Final" describes a stage, not authority.
AUTH_POS = (r"single source of truth", r"\bauthoritative\b",
            r"source of truth")
AUTH_NEG = (r"not authoritative", r"not a source of programme truth",
            r"not programme authority", r"chronology only",
            r"\bnot\b[^.]{0,20}\bauthoritative\b")

# SPLIT deliberately. Only a CONTROLLED FIELD survives inside a
# blockquote; self-referring PROSE inside a quote is still a quotation.
# Boundary pair 13/13b caught the over-broad first attempt, in which
# "> This document is the single source of truth." became a declaration.
SELF_FIELD = re.compile(r"^\s*\*{0,2}(?:status|authority)\*{0,2}\s*:", re.I)
SELF_PROSE = re.compile(
    r"^\s*(?:this (?:document|file|register|entry|page)|it)\b", re.I)
SELF_MARK = re.compile(SELF_FIELD.pattern + "|" + SELF_PROSE.pattern, re.I)
MDPATH = re.compile(r"`?([A-Za-z0-9_./-]+\.md)`?")


def _sentences(text: str):
    """(sentence, line_index) with blockquote/fence context preserved."""
    out, fence = [], False
    for i, ln in enumerate(text.splitlines()):
        s = ln.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence; continue
        if fence:
            continue
        quoted = s.startswith(">")
        body = s.lstrip("> ").strip()
        # NOT ':' -- a colon BINDS a controlled field to its value
        # ("Status: SINGLE SOURCE OF TRUTH"). Splitting there severed the
        # field from the claim, so the subject marker was never seen and
        # a SELF claim abstained as AMBIGUOUS. Tokenisation silently
        # changing a semantic outcome.
        for part in re.split(r"(?<=[.;])\s+", body):
            if part:
                out.append((part, i, quoted))
    return out


def bind_claims(path: str, text: str, front_matter: dict | None = None):
    """Return (claims, ambiguous_count).

    claims: list of (polarity, subject, evidence) where polarity is
    'AUTHORITY_POSITIVE' or 'AUTHORITY_NEGATIVE'.
    """
    self_name = pathlib.PurePosixPath(path).name
    claims, ambiguous = [], 0

    # 1 — structured metadata is SELF by construction
    for k, v in (front_matter or {}).items():
        if k.lower() in ("authority", "status") and isinstance(v, str):
            pol = ("AUTHORITY_NEGATIVE"
                   if any(re.search(p, v, re.I) for p in AUTH_NEG)
                   else "AUTHORITY_POSITIVE"
                   if any(re.search(p, v, re.I) for p in AUTH_POS) else None)
            if pol:
                claims.append((pol, "SELF", f"front-matter {k}: {v}"))

    for sent, _i, quoted in _sentences(text):
        neg = any(re.search(p, sent, re.I) for p in AUTH_NEG)
        pos = (not neg) and any(re.search(p, sent, re.I) for p in AUTH_POS)
        if not (neg or pos):
            continue
        pol = "AUTHORITY_NEGATIVE" if neg else "AUTHORITY_POSITIVE"

        # A quotation is not a declaration -- UNLESS the quoted content
        # is a CONTROLLED FIELD about this document. Markdown blockquotes
        # are routinely used as CALLOUT BANNERS:
        #   > **STATUS: RECOVERY PROTOCOL — NOT A SOURCE OF PROGRAMME TRUTH**
        # That is a self-declaration formatted as a callout, and treating
        # it as someone else's words silently discarded a document's own
        # disclaimer. Structural rule, not semantics: controlled field
        # inside a quote = declaration; free prose inside a quote = quote.
        if quoted and not SELF_FIELD.search(sent):
            claims.append((pol, "QUOTED_NOT_DECLARATION", sent[:90]))
            continue

        # 3 — an explicitly named OTHER document owns the claim
        named = [m.group(1) for m in MDPATH.finditer(sent)]
        others = [n for n in named
                  if pathlib.PurePosixPath(n).name != self_name]
        if others:
            claims.append((pol, "OTHER", f"names {others[0]}: {sent[:70]}"))
            continue
        if named:                      # names ITSELF explicitly
            claims.append((pol, "SELF", f"names itself: {sent[:70]}"))
            continue

        # 2 — explicit SELF marker
        if SELF_MARK.search(sent):
            claims.append((pol, "SELF", sent[:90]))
            continue

        # 5 — abstain
        ambiguous += 1
        claims.append((pol, "AMBIGUOUS_SUBJECT", sent[:90]))
    return claims, ambiguous


def authority_claim(claims):
    """SELF-bound claims only. Never an authority STATE."""
    s = {p for p, subj, _ in claims if subj == "SELF"}
    if {"AUTHORITY_POSITIVE", "AUTHORITY_NEGATIVE"} <= s:
        return "CONFLICTING_SELF_CLAIMS"
    if "AUTHORITY_POSITIVE" in s:
        return "SELF_ASSERTS_AUTHORITY"
    if "AUTHORITY_NEGATIVE" in s:
        return "SELF_ASSERTS_NON_AUTHORITY"
    return "NO_SELF_CLAIM"
