#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — RUNNER. Assembles evidence facts and verdicts.

Emits a machine-readable admission contract that approves NOTHING about
itself. Admission is a governing decision, never a test result.
"""
from __future__ import annotations
import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import classify as cl                                          # noqa: E402
import ontology as ont                                         # noqa: E402
import passa                                                   # noqa: E402
import subjectbind as sb                                       # noqa: E402

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}
# A CURRENCY label claims the document's own up-to-dateness, which git
# can refute. A version or planning date does not make that claim, so a
# later commit does not contradict it.
CURRENCY = re.compile(r"last updated|last reviewed|^reviewed|last-updated",
                      re.I)


def parse_date(tok):
    t = tok.strip().replace(",", "")
    for pat, order in (
            (r"(20\d\d)-(\d\d)-(\d\d)", "ymd"),
            (r"(\d{1,2})\s+([A-Za-z]+)\.?\s+(20\d\d)", "dmy"),
            (r"([A-Za-z]+)\.?\s+(\d{1,2})\s+(20\d\d)", "mdy")):
        m = re.fullmatch(pat, t)
        if not m:
            continue
        if order == "ymd":
            return datetime.date(*map(int, m.groups()))
        if order == "dmy":
            return datetime.date(int(m.group(3)),
                                 MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
        return datetime.date(int(m.group(3)),
                             MONTHS[m.group(1)[:3].lower()], int(m.group(2)))
    return None


def contradiction_of(row):
    """D3: a CURRENCY self-claim the history refutes.

    `last` was already in the Pass A row and v1.1 never consulted it.
    Only currency labels are testable this way -- a version date may
    legitimately precede a later edit, so it is not a contradiction.
    """
    for w in row["witnesses"].get("DATE", []):
        if w["applicability_scope"] != "WHOLE_FILE":
            continue
        if not CURRENCY.search(w["local_context"]):
            continue
        claimed = parse_date(w["witness_value"])
        if not claimed or not row.get("last"):
            continue
        try:
            gl = datetime.date(*map(int, row["last"].split("-")))
        except Exception:
            continue
        if (gl - claimed).days > 0:
            # F6 (Kai): `context` is PRESENTATION, not determining
            # evidence. It is named as an excerpt, and the evidence that
            # actually determined the contradiction travels beside it as
            # a complete 9-field witness -- which is also what lets E1
            # bind this fact to a compliant trace.
            return {"claimed": str(claimed), "git_last": row["last"],
                    "drift_days": (gl - claimed).days,
                    "selector": w["source_selector"],
                    "context_excerpt": w["local_context"][:120],
                    "context_excerpt_is_partial":
                        len(w["local_context"]) > 120,
                    "determining_witness": dict(w)}
    return None


def _witness_trace(row, family):
    """The first emitted witness of `family`, already 9-field compliant."""
    ws = (row["witnesses"] or {}).get(family) or []
    if not ws:
        return None
    w = dict(ws[0])
    w["evidence_total"] = len(ws)
    w["evidence_shown"] = 1
    w["truncated"] = len(ws) > 1
    return w


def _history_trace(row, subject, count):
    """MAINTENANCE_OBSERVED. Genuinely history-derived, so the selector is
    the exact reproducible history operation -- frozen subject, path and
    all -- not the bare `history:<path>` label v1.3a used, which named no
    operation and reproduced nothing.
    """
    sel = f"git:rev-list --count {subject} -- {row['path']}"
    return {"witness_type": "COMMIT_COUNT_IN_WINDOW", "witness_value": str(count),
            "source_path": row["path"], "source_selector": sel,
            "local_context": f"{sel} => {count}; last commit "
                             f"{row.get('last') or 'unknown'}",
            "applicability_scope": "WHOLE_FILE",
            "evidence_total": 1, "evidence_shown": 1, "truncated": False,
            "polarity": "POSITIVE", "certainty": "VERIFIED",
            "temporal": "AT_COMMIT", "subject": "SELF"}


# ── KAI-B4-SB-01: CLASSIFICATION READS THE BOUND SUBJECT'S BYTES ─────
# v1.2 read every subject document with Path.read_text(): WORKING-TREE
# bytes, never checked against the frozen commit, and decoded with
# universal-newline translation (a lone CR became LF), so classification
# could measure text the subject does not contain while stamping it with
# the subject's identity. There is no second source-identity system here:
# the Pass-A gate (passa._source_binding_gate) runs before measurement and
# every subject byte arrives through passa.make_verified_reader -- READ ->
# VERIFY THOSE BYTES AGAINST THE FROZEN BLOB -> USE -- which decodes the
# exact bytes with no newline translation. There is no fallback reader.
class SourceBindingError(SystemExit):
    """A subject byte could not be bound to the frozen commit. REFUSE."""


def _read_bound(read_source, subject_repo, rel):
    """The verified bytes of `rel`, decoded, or REFUSE. Never Path.read_text."""
    if read_source is None:
        raise SourceBindingError(
            f"REFUSE: {rel}: no verified subject reader was supplied; "
            f"classification never reads subject bytes any other way "
            f"(KAI-B4-SB-01)")
    try:
        return read_source(subject_repo, rel)
    except OSError as e:
        raise SourceBindingError(
            f"R11 ABORT [SOURCE BINDING / UNREADABLE SOURCE]: {rel}: "
            f"{type(e).__name__}: {e.strerror or e}. A subject file vanished "
            f"or became unreadable during measurement. Refusing to "
            f"measure.") from None


def _reader_trace(row, subject_repo, read_source):
    """STATIC_REFERENCE_AT_SUBJECT. The determining evidence is a STATIC
    READER REFERENCE produced by the Census opscan -- NOT history. v1.3a
    labelled it `history:`, which named the wrong evidence class entirely.

    E2. THE FACT WAS CALLED `CONSUMED_AT_SUBJECT`, WHICH ASSERTED MORE
    THAN THE EVIDENCE. The determining evidence is an `ast.walk` over a
    reading file's syntax tree; nothing anywhere in the chain executes,
    imports or observes anything. The renamed fact means exactly one
    thing:

        A RESOLVABLE STATIC REFERENCE WAS FOUND WITHIN THE BOUNDED
        CENSUS ANALYSIS.

    It does NOT assert runtime consumption, execution, reachability or
    invocation. A resolvable call site is recorded whether or not the
    enclosing function is ever called or its guard ever passes.

    AND ITS FALSE IS AN ABSTENTION, NOT A NEGATIVE. False means no
    resolvable static reference was found in the analysed scope. It does
    NOT mean the document is unread: most candidate operations resolve to
    neither a proven target nor a proven irrelevance, and those are
    unresolved, not absent.

    THE ANALYSED SCOPE IS PINNED, NOT RESTATED. `census_dependency` in
    the record carries the census package and the sha256 of its
    MANIFEST.sha256, which pins `opscan.py`, which is where
    `source_population`, `SRC_SUFFIX` and `EXCLUDE_DIRS` define the
    universe. The `opscan:` selector prefix names the instrument that
    produced the reference. No new schema field is added, because the
    scope is already recoverable from hash-pinned artefacts.

    The evidence lives in the READING document, so `source_path` is the
    reader and `local_context` is that document's actual source line. A
    fact with several reader references carries the D367 5
    evidence_total / evidence_shown semantics rather than dropping any.
    """
    ops = row.get("reader_ops") or []
    if not ops:
        return None                      # A6-ii will abstain, correctly
    o = sorted(ops, key=lambda x: (x["src"], x["line"]))[0]
    # KAI-B4-SB-01: the READING document's bytes come through the same
    # verified reader as the classified document, so they are the frozen
    # blob's bytes or the run REFUSES. The read sits OUTSIDE the try: a
    # binding failure or an unreadable source is never an abstention.
    lines = _read_bound(read_source, subject_repo, o["src"]).splitlines()
    try:
        line = lines[o["line"] - 1].strip()
    except Exception:
        return None                      # no locator -> no certification
    # The Census stores `expr` as an AST DUMP -- "Name(id='CHANGELOG',
    # ctx=Load())" -- which is a description of the node, not a token in
    # the source. D367 5 wants "the EXACT token or value matched", so the
    # trace carries the identifier the dump names, and only if that
    # identifier is genuinely present in the reading line. If it is not,
    # we abstain rather than certify a value the source does not contain.
    ident = next((g for g in re.findall(r"id='([^']+)'", o["expr"] or "")
                  if g in line), None)
    if not ident:
        return None
    return {"witness_type": "STATIC_READER_REFERENCE",
            "witness_value": ident, "source_path": o["src"],
            "source_selector": f"opscan:{o['src']}:L{o['line']}",
            "local_context": line, "applicability_scope": "SPAN",
            "reader_ast_expr": o["expr"], "reader_mode": o["mode"],
            "evidence_total": len(ops), "evidence_shown": 1,
            "truncated": len(ops) > 1,
            "polarity": "POSITIVE", "certainty": "OBSERVED",
            "temporal": "AT_COMMIT", "subject": f"OTHER:{row['path']}"}


def _claim_trace(row, determining, total):
    """A 9-field trace for a SELF-bound authority fact.

    The determining rows are CLAIM records, not witnesses -- a selector
    and a sentence, but not the nine fields -- so E1 could not bind them
    and A6-ii would have ABSTAINED four SELF_ASSERTS_AUTHORITY and one
    SELF_ASSERTS_NON_AUTHORITY rather than tracing them. That would be E1
    deleting facts instead of binding them, and it would move AUTHORITY,
    which belongs to step 5. The claim is promoted to a trace instead.

    `applicability_scope` is SPAN, not WHOLE_FILE: the evidence is the
    sentence. WHOLE_FILE would be an unearned widening and a scope
    decision, which is M3's at step 2.
    """
    c = determining[0]
    return {"witness_type": "SELF_AUTHORITY_CLAIM",
            "witness_value": c.get("term") or "",
            "source_path": row["path"], "source_selector": c["selector"],
            "local_context": c["text"], "applicability_scope": "SPAN",
            "evidence_total": total, "evidence_shown": len(determining),
            "truncated": len(determining) < total,
            "polarity": c["polarity"], "certainty": "OBSERVED",
            "temporal": "AT_COMMIT", "subject": "SELF"}


NINE_FIELDS = ("witness_type", "witness_value", "source_path",
               "source_selector", "local_context", "applicability_scope",
               "evidence_total", "evidence_shown", "truncated")


# E1 FAMILY GATE (Kai). value-in-context is necessary, not sufficient:
# a generated sentence containing a generated number would pass it. The
# trace's EVIDENCE CLASS must match the producer that made the fact
# positive.
TRACE_CLASS = {
    "MAINTENANCE_OBSERVED":       ("COMMIT_COUNT_IN_WINDOW", "git:rev-list"),
    "STATIC_REFERENCE_AT_SUBJECT": ("STATIC_READER_REFERENCE", "opscan:"),
    "CITES_COMMIT":               ("COMMIT", "L"),
    "CITES_RUN":                  ("RUN_ID", "L"),
    "CARRIES_DATE_STAMP":         ("DATE_STAMP", "L"),
    "BINDING_CONTRADICTION":      ("DATE_STAMP", "L"),
    "SELF_ASSERTS_AUTHORITY":     ("SELF_AUTHORITY_CLAIM", "L"),
    "SELF_ASSERTS_NON_AUTHORITY": ("SELF_AUTHORITY_CLAIM", "L"),
    # Kai P1-P4, 2026-10-02: the two governed classes the runner omitted.
    "NOMINAL_FUNCTION":           ("NOMINAL_FUNCTION_TERM", "L"),
    "SELF_ASSERTS_CURRENT":       ("SELF_CURRENTNESS_CLAIM", "L"),
}


# ── TWO INDEPENDENT SUBREPAIRS of one schema-completeness defect ──────
# ontology.EVIDENCE_FACTS governs ten classes; this runner emitted eight,
# and the qualifier correctly reported FACT_CLASS_ABSENT (Kai, DS-B4-01).
#
#   NF   NOMINAL_FUNCTION: WIRING of an already-governed observation,
#        classify.function()'s "NOMINAL_FUNCTION=<role> from
#        self-description". No detector is added.
#   SAC  SELF_ASSERTS_CURRENT: RESTORATION of the historical governed
#        producer, v1.1 evidence.currentness_claims (D361, 438007e).
#
# FACT DISPOSITION, both subrepairs (Kai, DS-B4-04/09):
#   detector or input unavailable            -> REFUSE
#   complete measurement, nothing established -> False ("positive fact
#                                               not established", never
#                                               "proved false")
#   positive candidate, determining trace
#   missing or non-compliant                  -> REFUSE (never False)

# ── SAC — SELF_ASSERTS_CURRENT, historical restoration (Kai P2(a)) ────
#
# NORMATIVE IDENTITY. CURRENT_POS / CURRENT_NEG below are literal copies
# of the tuples in the IMMUTABLE commit 438007e,
# kai-pm/house_in_order_h2_v11/evidence.py. That commit, not this file,
# is their authority; the governed controls compare these tuples with it
# mechanically and REFUSE on any drift. v1.1 is deliberately NOT imported
# at runtime (that would add an old package to the governed population).
# Subject ownership is the D12-repaired subjectbind.bind_subject (full
# repo-relative path; a basename alone is not SELF -- Kai, B1). The v1.1
# boundary is restored here because subjectbind._sentences does not
# carry it: fenced code is not evidence; quoted free prose is not a
# declaration; a quoted CONTROLLED FIELD (governed sb.SELF_FIELD) may be
# one. Negative polarity wins. v1.1's auxiliary diagnostics are NOT
# restored (Kai, B2).

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


def _segments(text):
    """v1.1 subjectbind2._sentences (438007e), restored with absolute
    offsets: (start, sentence, line_no, quoted).

    SEMANTICS ARE v1.1's, and a governed control compares the projection
    (sentence, line_no - 1, quoted) with the historical function over a
    hostile corpus: lines are v1.1's `str.splitlines()` lines; ``` and ~~~
    toggle a fence and fenced lines are dropped; a blockquote marker is
    stripped but remembered; the split is `(?<=[.;])\\s+` (never ':').

    OFFSETS ARE DERIVED, NEVER SEARCHED (Kai, DS-B4-V2-08): the line start
    is the running sum of the splitlines(keepends=True) lengths; the body
    start adds exactly the characters each v1.1 strip step removed; each
    sentence start is the end of the previous separator match. The control
    asserts text[start:start+len(sentence)] == sentence for every segment.
    """
    out, fence, line_start = [], False, 0
    for i, raw in enumerate(text.splitlines(keepends=True)):
        ln = raw.splitlines()[0] if raw.splitlines() else ""
        here, line_start = line_start, line_start + len(raw)
        s = ln.lstrip()
        if s.startswith("```") or s.startswith("~~~"):
            fence = not fence
            continue
        if fence:
            continue
        quoted = s.startswith(">")
        s2 = s.lstrip("> ")
        body = s2.strip()
        lead = (len(ln) - len(s)) + (len(s) - len(s2)) + (len(s2) - len(s2.lstrip()))
        base = here + lead
        pos = 0
        for sep in list(re.finditer(r"(?<=[.;])\s+", body)) + [None]:
            end = sep.start() if sep else len(body)
            part = body[pos:end]
            if part:
                out.append((base + pos, part, i + 1, quoted))
            if sep:
                pos = sep.end()
    return out


def currentness_claims(path, text):
    """Every polarity-bearing currentness sentence with its OWNER.
    Returns [(polarity, subject, line_no, sentence, phrase, sentence_start,
    phrase_start)], both starts ABSOLUTE offsets into `text`."""
    claims = []
    for start, sent, line_no, quoted in _segments(text):
        neg = next((m for m in (re.search(p, sent, re.I) for p in CURRENT_NEG)
                    if m), None)
        pos = None if neg else next(
            (m for m in (re.search(p, sent, re.I) for p in CURRENT_POS) if m),
            None)
        if not (neg or pos):
            continue
        pol = "CURRENT_NEGATIVE" if neg else "CURRENT_POSITIVE"
        if quoted and not sb.SELF_FIELD.match(sent):
            subject = "QUOTED_NOT_DECLARATION"
        else:
            subject, _why = sb.bind_subject(text, start, sent, path)
        hit = neg or pos
        claims.append((pol, subject, line_no, sent, hit.group(0),
                       start, start + hit.start()))
    return claims


def _currentness_fact(row, text):
    """(positive, trace-maker). SELF positive and no SELF negative; a
    SELF negative anywhere wins (v1.1 conflict rule)."""
    cl_ = currentness_claims(row["path"], text)
    self_pos = [c for c in cl_ if c[1] == "SELF" and c[0] == "CURRENT_POSITIVE"]
    self_neg = [c for c in cl_ if c[1] == "SELF" and c[0] == "CURRENT_NEGATIVE"]
    positive = bool(self_pos) and not self_neg

    def mk():
        # Kai KAI-B4-V3-10: one locator vocabulary across the instrument.
        # The selector and context come from the canonical passa._selector
        # / passa._context at the phrase's ABSOLUTE offset; the splitlines
        # ordinal stays internal to the historical segmentation. The
        # canonical context is the complete logical (LF) line, which always
        # contains the whole v1.1 sentence (v1.1 never splits across LF).
        _pol, _subj, _ln, sent, phrase, s_start, p_start = self_pos[0]
        return {"witness_type": "SELF_CURRENTNESS_CLAIM",
                "witness_value": phrase, "source_path": row["path"],
                "source_selector": passa._selector(text, p_start),
                "local_context": passa._context(text, s_start,
                                                s_start + len(sent)),
                # the sentence is the evidence: SPAN, never widened because
                # its semantic subject is the document (Kai P2)
                "applicability_scope": "SPAN",
                "evidence_total": len(self_pos), "evidence_shown": 1,
                "truncated": len(self_pos) > 1,
                "polarity": "POSITIVE", "certainty": "OBSERVED",
                "temporal": "AT_COMMIT", "subject": "SELF"}
    return positive, mk


# ── NF — NOMINAL_FUNCTION, wiring of the classify.function() observation
class FactDispositionError(SystemExit):
    """A governed fact could not be dispositioned honestly: input
    unavailable, a malformed governed observation, or a positive whose
    exact determining trace cannot be built. REFUSE, never False."""


class NominalTraceError(FactDispositionError):
    """The NOMINAL_FUNCTION-specific fact-disposition REFUSE (Kai, V2-09:
    a real subclass, not a second name for the same class)."""

# The EXACT governed observation grammar emitted by classify.function()
# (classify.py, single-role branch). Nothing looser is accepted
# (Kai, DS-B4-08).
NOMINAL_OBSERVATION = re.compile(
    r"NOMINAL_FUNCTION=(?P<role>[A-Z][A-Z_]*) from self-description")


def _nominal_fact(row, text, function_cell):
    """(positive, trace-maker) from classify.function()'s own observation
    `NOMINAL_FUNCTION=<role> from self-description`. The token is located
    with the SAME governed cl.FUNCTION_TERMS / cl.PURPOSE / cl.term_match
    and the same title-then-purpose order classify uses; no second
    vocabulary exists here."""
    obs = (function_cell or {}).get("observed") or ""
    if not obs.startswith("NOMINAL_FUNCTION="):
        return False, None              # complete measurement, not established
    om = NOMINAL_OBSERVATION.fullmatch(obs)
    if om is None:
        raise NominalTraceError(
            f"REFUSE: {row['path']}: malformed NOMINAL_FUNCTION observation "
            f"{obs!r}; the governed grammar is "
            f"'NOMINAL_FUNCTION=<ROLE> from self-description'")
    role = om.group("role")
    if role not in cl.FUNCTION_TERMS:
        raise NominalTraceError(
            f"REFUSE: {row['path']}: NOMINAL_FUNCTION role {role!r} is not a "
            f"governed FUNCTION_TERMS key")
    term = cl.FUNCTION_TERMS[role]
    # The TWO governed channels classify.function() unions, in its order
    # (title, then PURPOSE); each is evaluated, so the evidence population
    # is counted, not assumed (Kai, KAI-B4-V2-10). evidence_total for NF is
    # the number of qualifying governed SOURCE CHANNELS supporting the
    # emitted fact -- not a count of distinct role values (Kai, V3-03).
    # Each channel carries the ABSOLUTE offset of its exact token; the
    # selector and context are derived from it with the canonical
    # passa._selector / passa._context (Kai, KAI-B4-V3-10).
    channels = []
    title = row.get("title") or ""
    if title and cl.term_match(term, title):
        # Pass A's title rule, on Pass A's own line model: the FIRST
        # str.splitlines() line starting with '#', title =
        # ln.lstrip("#").strip()[:120]. If the source cannot reconstruct
        # the bound Pass-A title, the row and the source disagree: REFUSE,
        # never fall through to PURPOSE (V2-04).
        first, off = None, 0
        for raw in text.splitlines(keepends=True):
            ln = raw.splitlines()[0] if raw.splitlines() else ""
            if ln.startswith("#"):
                first = (off, ln)
                break
            off += len(raw)
        if first is None or first[1].lstrip("#").strip()[:120] != title:
            raise NominalTraceError(
                f"REFUSE: {row['path']}: the Pass-A title {title!r} carries the "
                f"nominal term but the source's first heading does not "
                f"reconstruct it")
        line_start, ln = first
        after_hash = ln.lstrip("#")
        title_at = line_start + (len(ln) - len(after_hash)) + \
            (len(after_hash) - len(after_hash.lstrip()))
        m = cl.term_match(term, title)
        channels.append((title_at + m.start(), m.group(0)))
    pm = cl.PURPOSE.search(text[:6000])
    m = pm and cl.term_match(term, pm.group("body"))
    if m:
        channels.append((pm.start("body") + m.start(), m.group(0)))
    if not channels or any(text[at:at + len(tok)] != tok
                           for at, tok in channels):
        raise NominalTraceError(
            f"REFUSE: {row['path']}: classify observed NOMINAL_FUNCTION={role} "
            f"but no exact determining source token could be located")
    hit = channels[0]                   # deterministic: classify's order

    def mk():
        at, token = hit
        return {"witness_type": "NOMINAL_FUNCTION_TERM",
                "witness_value": token, "source_path": row["path"],
                "source_selector": passa._selector(text, at),
                "local_context": passa._context(text, at, at + len(token)),
                "applicability_scope": "SPAN",
                "evidence_total": len(channels), "evidence_shown": 1,
                "truncated": len(channels) > 1,
                "polarity": "POSITIVE", "certainty": "OBSERVED",
                "temporal": "AT_COMMIT", "subject": "SELF"}
    return True, mk


def _class_ok(name, tr):
    want = TRACE_CLASS.get(name)
    if not want:
        return False
    kind, sel = want
    return tr.get("witness_type") == kind and \
        str(tr.get("source_selector", "")).startswith(sel)


def _compliant(t):
    """E1 / D367 5. Present is not enough -- the trace must be SEMANTICALLY
    TRUTHFUL: all nine fields, and the context must actually contain the
    value it claims to evidence.
    """
    if not t or any(t.get(k) in (None, "") for k in NINE_FIELDS):
        return False
    return str(t["witness_value"]) in str(t["local_context"])


# The two classes repaired here. The existing eight keep their A6-ii
# behaviour unchanged; that is outside this repair (Kai: bounded tranche).
REFUSE_ON_UNTRACEABLE_POSITIVE = ("NOMINAL_FUNCTION", "SELF_ASSERTS_CURRENT")


def evidence_facts(row, claims, contradiction, determining=(),
                   subject="", subject_repo=".", *, text=None,
                   function_cell=None, read_source=None):
    """FACTS, each bound to the trace that DETERMINED it. None is a
    verdict (D360 5).

    E1. v1.2 emitted these as bare booleans. D367 5 requires every
    POSITIVE evidence fact to carry a source-bound witness sufficient for
    independent adjudication, and 81 of 316 positives carried none that
    was bound to the fact itself: 71 MAINTENANCE_OBSERVED and 5
    STATIC_REFERENCE_AT_SUBJECT -- then still named CONSUMED_AT_SUBJECT,
    which E2 corrected -- had no witness at all, and 5
    BINDING_CONTRADICTION carried a 5-field contradiction record rather
    than a 9-field witness.

    A6-i: the producer that SETS the boolean carries the witness that set
    it. A6-ii: a positive with no compliant trace is NOT emitted as
    positive -- it abstains. A related trace living elsewhere in the
    package does not qualify; the binding is to the fact.
    """
    cand, traces = {}, {}
    cand["MAINTENANCE_OBSERVED"] = (
        row["commits_in_window"] > 1,
        lambda: _history_trace(row, subject, row["commits_in_window"]))
    # E2: the fact states its EVIDENCE CLASS. `readers` is unchanged --
    # the field is internal and never surfaced as a fact name -- and the
    # predicate, the trace and the population are untouched. Only the
    # claim the name makes is corrected.
    cand["STATIC_REFERENCE_AT_SUBJECT"] = (
        bool(row["readers"]),
        lambda: _reader_trace(row, subject_repo, read_source))
    cand["CITES_COMMIT"] = (bool(row["witnesses"].get("COMMIT")),
                            lambda: _witness_trace(row, "COMMIT"))
    cand["CITES_RUN"] = (bool(row["witnesses"].get("RUN_ID")),
                         lambda: _witness_trace(row, "RUN_ID"))
    cand["CARRIES_DATE_STAMP"] = (bool(row["witnesses"].get("DATE")),
                                  lambda: _witness_trace(row, "DATE"))
    cand["BINDING_CONTRADICTION"] = (
        contradiction is not None,
        lambda: dict(contradiction["determining_witness"])
        if contradiction else None)
    ac = sb.authority_claim(claims)
    for name, want in (("SELF_ASSERTS_AUTHORITY", "SELF_ASSERTS_AUTHORITY"),
                       ("SELF_ASSERTS_NON_AUTHORITY",
                        "SELF_ASSERTS_NON_AUTHORITY")):
        cand[name] = (ac == want,
                      (lambda d=determining, n=len(claims): _claim_trace(
                          row, d, n) if d else None))

    # Kai P3: every governed class is MEASURED on every row. No default:
    # a caller that cannot supply the inputs does not get a False.
    if text is None or function_cell is None:
        raise FactDispositionError("REFUSE: evidence_facts needs the document text and "
                         "the governed FUNCTION cell (Kai P3: no unmeasured "
                         "False)")
    cand["NOMINAL_FUNCTION"] = _nominal_fact(row, text, function_cell)
    cand["SELF_ASSERTS_CURRENT"] = _currentness_fact(row, text)
    if len(ont.EVIDENCE_FACTS) != len(set(ont.EVIDENCE_FACTS)):
        raise FactDispositionError(
            f"REFUSE: ontology.EVIDENCE_FACTS names a governed class more than "
            f"once; the evidence-fact population cannot be reconciled "
            f"(Kai, V3-07)")
    missing = [n for n in ont.EVIDENCE_FACTS if n not in cand]
    extra = [n for n in cand if n not in ont.EVIDENCE_FACTS]
    if missing or extra:
        raise FactDispositionError(f"REFUSE: evidence-fact producer population != the "
                         f"governed schema; missing={missing} extra={extra}")

    f, abstained = {}, []
    for name, (positive, mk) in cand.items():
        if not positive:
            f[name] = False
            continue
        t = mk()
        if _compliant(t) and _class_ok(name, t):
            f[name] = True
            traces[name] = t
        elif name in REFUSE_ON_UNTRACEABLE_POSITIVE:
            # Kai DS-B4-04: a positive candidate without a compliant
            # determining trace is never demoted to False for these two.
            raise FactDispositionError(
                f"REFUSE: {row['path']}: positive {name} candidate has no "
                f"compliant determining trace")
        else:                       # A6-ii: no compliant trace, no positive
            f[name] = False
            abstained.append(name)
    return f, ac, traces, abstained


def _load_stage_a(stage_a_path):
    """Strict Stage-A load (v4.1 C1, D380 §6.10) and DEP-3 runtime check,
    BEFORE any work: the classifier verifies ITS OWN executing runtime."""
    import stage_identity as SI
    try:
        desc = SI.parse_descriptor_bytes(SI._read_regular_once(stage_a_path))
        observed_runtime = SI.verify_runtime_identity(desc)
    except SI.StageIdentityError as e:
        raise SystemExit(f"REFUSE: {e}")
    return desc, observed_runtime


def _consume_pass_a(desc, a):
    """v4.5 §15.2 — classification reads the exact Pass-A bytes ONCE, hashes
    them, parses THE SAME bytes, and validates them against the ORIGINAL
    Pass-A Stage-B binding whose digest must equal the independently held
    anchor. Then the Pass-A provenance is verified SLOT BY SLOT, including
    the Census member digests against the manifest bytes bound by the
    Stage-A aggregate; any slot left unverified REFUSES (v4.1 C4)."""
    import stage_identity as SI
    _pa_path = pathlib.Path(a.passa)
    if not _pa_path.is_file():
        raise SystemExit(
            f"R11 ABORT: no Pass-A artefact at {a.passa}. The subject of "
            f"classification does not exist, so nothing downstream of it "
            f"may be measured. Refusing before any work.")
    try:
        pa_bytes, pa, _binding = SI.consume_bound_artifact(
            a.passa, a.passa_stage_b, a.passa_binding_sha256,
            stage_a_desc=desc, producer_component="PASS_A")
        man = SI._read_regular_once(pathlib.Path(a.census_package) /
                                    SI.CENSUS_MANIFEST)
        pprov = pa.get("producer_provenance")
        if pprov is None:
            raise SI.StageIdentityError(
                "the Pass-A input carries no in-band producer_provenance "
                "(D379 §4). Missing provenance: REFUSE, no silent inheritance.")
        _ident, _h2, unverified = SI.verify_provenance(
            pprov, desc, census_manifest_bytes=man)
        SI.require_complete(unverified, "classification consuming Pass A")
    except SI.StageIdentityError as e:
        raise SystemExit(f"REFUSE: Pass-A input does not verify: {e}")
    return pa_bytes, pa, pprov


def _classification_provenance(desc, observed_runtime, sr, pa_bytes, pprov):
    """D379 §4 CLASSIFICATION in-band provenance + input_binding, from
    OBSERVED values and the exact consumed bytes (v4.5 §16.1/§16.3)."""
    import stage_identity as SI
    ident = SI.stage_a_identity(desc)
    try:
        # D379 Q1a-3: this producer verifies ITS OWN executing bytes.
        members = SI.check_population(desc, "classification")
        commit = passa.git(sr, "rev-parse", "HEAD").stdout.strip()
        tree = passa.git(sr, "rev-parse", f"{commit}^{{tree}}").stdout.strip()
        _n, tp_ident, _p = SI.derive_tree_paths(sr, tree)
    except SI.StageIdentityError as e:
        raise SystemExit(str(e))
    if (commit, tree, tp_ident) != (desc["subject"]["commit"],
                                    desc["subject"]["tree"],
                                    desc["tree_paths"]["tree_paths_identity"]):
        raise SystemExit("REFUSE: the observed subject checkout is not the "
                         "Stage-A subject (commit, tree, tree_paths)")
    return {
        "stage_a_identity": ident,
        "stage_a_descriptor_digest": SI.stage_a_descriptor_digest(desc),
        "producer_component": "CLASSIFICATION",
        "producer_population": [{"class": c, "identity": i, "sha256": d}
                                for c, i, d in members],
        "producer_denominator": len(members),
        "runtime_identity": observed_runtime,
        "subject_commit": commit,
        "subject_tree": tree,
        "tree_paths_identity": tp_ident,
        "input_binding": {
            "pass_a_artifact_sha256": SI.sha256_hex(pa_bytes),
            "pass_a_stage_a_identity": pprov["stage_a_identity"],
            "pass_a_producer_provenance_digest": SI.provenance_digest(pprov),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--passa", required=True)
    ap.add_argument("--passa-stage-b", required=True, dest="passa_stage_b",
                    help="the ORIGINAL Pass-A Stage-B binding (v4.5 §15.2)")
    ap.add_argument("--expected-passa-binding-sha256", required=True,
                    dest="passa_binding_sha256",
                    help="the independently held anchor for that binding")
    ap.add_argument("--census-package", required=True, dest="census_package",
                    help="the governed Census package whose manifest bytes the "
                         "Stage-A aggregate binds (Census member slots)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage-a", required=True, dest="stage_a",
                    help="the Stage-A descriptor this producer consumes and "
                         "verifies, and against which the Pass-A input "
                         "binding is checked (D379 §2/§4).")
    a = ap.parse_args()

    desc, observed_runtime = _load_stage_a(a.stage_a)
    pa_bytes, pa, pprov = _consume_pass_a(desc, a)
    sr = pathlib.Path(a.subject_repo)
    head = passa.git(sr, "rev-parse", "HEAD").stdout.strip()
    if head != pa["subject"] or head != desc["subject"]["commit"]:
        raise SystemExit(f"R11 ABORT: subject repo HEAD {head[:12]} != "
                         f"Pass A / Stage-A subject")
    # KAI-B4-SB-01: the Pass-A source-binding gate (subject identity,
    # tracked divergence, tracked symlinks) runs BEFORE any subject byte is
    # measured, and ONE verified reader for the frozen subject supplies
    # every subject byte this producer consumes: the classified document
    # here and the reading document inside _reader_trace.
    passa._source_binding_gate(sr, pa["subject"])
    read_source = passa.make_verified_reader(sr, pa["subject"])
    _cls_prov = _classification_provenance(desc, observed_runtime, sr,
                                           pa_bytes, pprov)

    rows, facts_tally = [], collections.Counter()
    nominal = collections.Counter()
    for row in pa["rows"]:
        text = _read_bound(read_source, sr, row["path"])
        claims, stats = sb.bind_claims(row["path"], text)
        contradiction = contradiction_of(row)
        # D14/D15: the DETERMINING rows are always carried, with counts.
        # E1 needs them BEFORE the facts, because a SELF-authority fact
        # must bind to the row that determined it.
        det = sb.determining_claims(claims)
        # The governed observation the NOMINAL_FUNCTION fact consumes.
        # classify.function() is a pure calculation over (row, text) and
        # fixed constants -- no cache, mutation or IO -- so cl.classify()
        # below recomputes the identical cell. The equality check after
        # classify is a COHERENCE check, not a proof of correctness; it is
        # kept because authority_claim must be set before classify runs
        # (Kai, DS-B4-V2-03).
        fn_cell = cl.function(row, text)
        facts, ac, fact_traces, abstained = evidence_facts(
            row, claims, contradiction, det, pa["subject"], sr,
            text=text, function_cell=fn_cell, read_source=read_source)
        row["authority_claim"] = ac
        out = cl.classify(row, text, contradiction)
        if out["FUNCTION"] != fn_cell:
            raise SystemExit(f"REFUSE: {row['path']}: classify's FUNCTION cell "
                             f"differs from the observation the fact used")
        out["evidence_facts"] = facts
        # E1: every POSITIVE fact carries the trace that determined it.
        out["evidence_fact_traces"] = fact_traces
        if abstained:
            out["evidence_facts_abstained_no_compliant_trace"] = abstained
        out["authority_claim"] = ac
        out["authority_evidence"] = {
            "determining": det, "total": stats["total"],
            "shown": len(det),
            "truncated": len(det) < stats["total"] and False,
            "note": "SELF-bound determining rows are carried in full; "
                    "non-determining claims are counted, not dropped",
            "counts": stats}
        if contradiction:
            out["binding_contradiction"] = contradiction
        for k, v in facts.items():
            if v:
                facts_tally[k] += 1
        if out["FUNCTION"].get("observed", "").startswith("NOMINAL_FUNCTION="):
            nominal[out["FUNCTION"]["observed"].split("=", 1)[1]] += 1
        rows.append(out)

    assert len(rows) == pa["population"], "population mismatch"
    tallies = {ax: dict(collections.Counter(r[ax]["value"] for r in rows))
               for ax in ont.ALPHABETS}
    payload = {
        # D379 §4 — IN-BAND, with the exact input binding. NEVER a digest
        # of these very bytes.
        "producer_provenance": _cls_prov,
        "instrument": "HOUSE_H2_CLASSIFIER_v1.2",
        "subject": pa["subject"], "subject_tree": pa["subject_tree"],
        "history_identity": pa["history_identity"],
        "census_dependency": pa["census_dependency"],
        "population": pa["population"], "rows": rows,
        "axis_tallies": tallies,
        "evidence_fact_tally": dict(facts_tally),
        "nominal_function_tally": dict(nominal),
        "admission_contract": {
            "self_approval": "NONE",
            "status": "CANDIDATE. NOT FROZEN. NOT ADMITTED.",
            "note": "Admission is a governing decision, never a test "
                    "result. This package supplies evidence and approves "
                    "nothing about itself.",
            "dispositions": ont.disposition_rows(),
        },
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))

    print(f"HOUSE_H2 v1.2 — {len(rows)} rows == population {pa['population']}")
    print(f"  subject {pa['subject'][:12]} tree {pa['subject_tree'][:12]}\n")
    for ax in ont.ALPHABETS:
        t = tallies[ax]
        pos = {k: v for k, v in t.items() if k not in ("UNKNOWN", "UNMEASURED")}
        print(f"  {ax:<12} positives {sum(pos.values()):>4}  "
              f"UNKNOWN {t.get('UNKNOWN', 0):>4}   {pos or ''}")
    print("\n  evidence facts (NOT verdicts):")
    for k in ont.EVIDENCE_FACTS:
        if k in facts_tally:
            print(f"    {k:<28}{facts_tally[k]:>5}")
    print(f"\n  NOMINAL_FUNCTION (self-description, earns no verdict): "
          f"{sum(nominal.values())}")
    print(f"    {dict(nominal)}")
    ac = hashlib.sha256(json.dumps(payload["admission_contract"],
                                   sort_keys=True).encode()).hexdigest()
    print(f"\n  admission contract sha256 {ac}")
    print("  self_approval: NONE")


if __name__ == "__main__":
    main()
