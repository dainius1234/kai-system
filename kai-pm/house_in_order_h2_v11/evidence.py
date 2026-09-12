#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — EVIDENCE FACTS. Structurally separate from verdicts.

D360 §5. These three facts are what v1.0 was silently converting into
LIFECYCLE verdicts. They are now emitted in their own field and CANNOT
reach a verdict, because `classify2.py` never reads them when deciding
LIFECYCLE. The separation is structural, not a comment.

  MAINTENANCE_OBSERVED  commits > 1 at the subject IN THE DECLARED
                        WINDOW. Says the document was changed. Says
                        nothing about whether it governs anything now.
  SELF_ASSERTS_CURRENT  the document claims currentness ABOUT ITSELF.
                        Says what it asserts. Not that it is true.
  CONSUMED_AT_SUBJECT   an executable operation reads it at the subject.
                        Strong third-party evidence of use -- and still
                        not lifecycle: stale, deprecated and
                        accidentally retained artefacts are read too.

THE PROSE RULE (D360 §4). No unbound token may promote anything. v1.0's
`DECL` was a case-insensitive substring match that pulled 'current',
'active', 'stale' AND 'historical' out of the same documents -- the
D339 `\bfinal\b` defect at corpus scale. Here a currentness claim
requires SUBJECT, PREDICATE AND POLARITY to be structurally
distinguishable, and the same quoted-vs-declaration rule that
subjectbind's boundary pair earned.
"""
from __future__ import annotations
import re

import subjectbind2 as sb

# POSITIVE currentness predicates. Bound forms only -- never a bare
# token. "current phase", not "current".
CURRENT_POS = (
    r"\bis (?:the )?current\b", r"\bcurrently\b", r"\bcurrent (?:phase|"
    r"focus|state|status|master|authority)\b", r"\bstatus\s*:\s*active\b",
    r"\bstatus\s*:\s*current\b", r"\bin force\b", r"\bstill (?:in force|"
    r"current|active)\b",
)
# NEGATIVE polarity must be tested FIRST: "no longer current" contains
# "current", and a polarity-blind matcher would read it as the opposite
# of what it says.
CURRENT_NEG = (
    r"\bno longer\b", r"\bnot current\b", r"\bsuperseded\b",
    r"\bdeprecated\b", r"\bobsolete\b", r"\bstale\b", r"\bhistorical\b",
    r"\barchived\b", r"\bwithdrawn\b",
)


def currentness_claims(path: str, text: str):
    """(claims, ambiguous) — subject-bound currentness assertions.

    Reuses subjectbind2's sentence segmentation and its structural
    quoted-vs-declaration rule: a CONTROLLED FIELD inside a blockquote is
    a self-declaration (callout banners are how this repository writes
    them); free prose inside a quote is someone else's words.
    """
    claims, ambiguous = [], 0
    self_name = path.rsplit("/", 1)[-1]
    for sent, _i, quoted in sb._sentences(text):
        neg = any(re.search(p, sent, re.I) for p in CURRENT_NEG)
        pos = (not neg) and any(re.search(p, sent, re.I) for p in CURRENT_POS)
        if not (neg or pos):
            continue
        pol = "CURRENT_NEGATIVE" if neg else "CURRENT_POSITIVE"

        if quoted and not sb.SELF_FIELD.search(sent):
            claims.append((pol, "QUOTED_NOT_DECLARATION", sent[:90]))
            continue
        named = [m.group(1) for m in sb.MDPATH.finditer(sent)]
        others = [n for n in named if n.rsplit("/", 1)[-1] != self_name]
        if others:
            claims.append((pol, "OTHER", f"names {others[0]}: {sent[:70]}"))
            continue
        if named:
            claims.append((pol, "SELF", f"names itself: {sent[:70]}"))
            continue
        if sb.SELF_MARK.search(sent):
            claims.append((pol, "SELF", sent[:90]))
            continue
        ambiguous += 1
        claims.append((pol, "AMBIGUOUS_SUBJECT", sent[:90]))
    return claims, ambiguous


def facts(row, text):
    """The three evidence facts, each with its witness. NEVER a verdict."""
    out = {}

    n = row["commits_in_window"]
    out["MAINTENANCE_OBSERVED"] = (
        {"present": True, "witness": f"{n} commits at the subject within "
                                     f"the declared history window"}
        if n > 1 else
        {"present": False, "witness": f"commits_in_window={n}: one "
                                      f"observation is not evidence of "
                                      f"maintenance, nor of its absence"})

    cl, amb = currentness_claims(row["path"], text)
    selfpos = {p for p, s, _e in cl if s == "SELF"}
    if "CURRENT_POSITIVE" in selfpos and "CURRENT_NEGATIVE" in selfpos:
        out["SELF_ASSERTS_CURRENT"] = {
            "present": False, "witness": "CONFLICTING_SELF_CLAIMS: the "
            "document asserts both current and not-current about itself"}
    elif "CURRENT_POSITIVE" in selfpos:
        ev = next(e for p, s, e in cl
                  if s == "SELF" and p == "CURRENT_POSITIVE")
        out["SELF_ASSERTS_CURRENT"] = {"present": True, "witness": ev[:90]}
    else:
        out["SELF_ASSERTS_CURRENT"] = {
            "present": False,
            "witness": "no subject-bound positive currentness claim"}
    out["currentness_claim_evidence"] = [f"{p}|{s}|{e[:60]}"
                                         for p, s, e in cl][:6]
    out["ambiguous_currentness_claims"] = amb

    out["CONSUMED_AT_SUBJECT"] = (
        {"present": True, "witness": f"read by {row['readers']}"}
        if row["readers"] else
        {"present": False, "witness": "no proven executable reader"})

    out["_NOT_A_VERDICT"] = ("D360 §5: these are evidence facts. None of "
                             "them earns LIFECYCLE=ACTIVE.")
    return out
