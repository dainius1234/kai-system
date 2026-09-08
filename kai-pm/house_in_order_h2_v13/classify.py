#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — VERDICT LAYER. Every claim conserved against its witness.

No axis forms a verdict from a boolean. Each consumes WITNESSES and
passes them through `envelope.claim()`, which refuses to widen subject,
scope, polarity, certainty or temporal applicability without a declared
promotion. Where the promotion is absent the axis ABSTAINS.

WHAT IS REPAIRED HERE

  D1  a commit cited for one sentence became a whole-file snapshot
      binding. The witness now carries SPAN, and SPAN -> WHOLE_FILE is
      refused unless a DOCUMENT_LEVEL_BINDING witness exists.
  D3  a currency self-claim the history contradicts yields
      BINDING_CONTRADICTION and cannot earn TIME_BOUND.
  D5  the raw present-tense flag is gone as a verdict input.
  D6  SCOPE is EARNED from a binding witness or it is UNKNOWN. v1.1
      emitted WHOLE_FILE on 272 of 272 with no `def scope()` at all.
  D7  a row whose own witness says REGION DETERMINATION REQUIRED can no
      longer emit WHOLE_FILE anyway.
  D8  term matching respects word boundaries and distinguishes a plural
      of the term from a different word ('Audit' vs 'Auditor',
      'Plan' vs 'Plane').
  D9  the OTHER negative is computed over a SUBSTANTIVE purpose capture,
      never a trigger phrase like 'This file'.
  D10 the purpose branch fires or it is a defect, and qualification
      reports its firing count.
  D11 candidate order never decides: two corroborated nominations are
      reported as ambiguity, not resolved by list position.

  M1  a review or inspection event was promoted into TIME_BOUND. M3
      answers what a witness APPLIES TO; VALIDITY must separately answer
      whether it ESTABLISHES the document's temporal state. Those are
      different predicates, and a document-scoped DATE is necessary but
      not sufficient. VALIDITY now requires a positively identified
      document-state binding, and it inspects EVERY document-scoped date
      rather than the first one Pass A emitted -- source order was
      deciding which date determined the verdict.

CORRECTION E, AND IT COSTS 144 ROWS. Path and title are created as part
of the same document by the same author. `PATH says audit + TITLE says
audit` is not two independent proofs of function -- it is one source
counted twice, and this programme already banked that rule when three
"independent confirmations" turned out to be three copies by one author.
Self-description therefore earns the EVIDENCE FACT `NOMINAL_FUNCTION`
and the verdict ABSTAINS. Only objective witnesses earn FUNCTION.
"""
from __future__ import annotations
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import envelope as E                                          # noqa: E402
import ontology as ont                                        # noqa: E402
import passa as PA                                            # noqa: E402
from envelope import Witness                                  # noqa: E402

DOC_BINDING = E.DECLARED_PROMOTIONS[("SPAN", "WHOLE_FILE")]

# ── M1: VALIDITY state-binding predicates. DECLARED CLOSED-WORLD ──────
# M3 answers ONE question: what does this witness apply to? VALIDITY must
# separately answer a DIFFERENT one: does this witness establish the
# document's temporal validity / current-state binding? A WHOLE_FILE DATE
# is NECESSARY for a positive date-based VALIDITY verdict and is NOT
# SUFFICIENT.
#
# THE SET IS THE DEMONSTRATED ONE, NOT A VOCABULARY. Every predicate here
# is observed in the subject as a document-scoped DATE whose source
# language positively states the document's own state or version point. A
# census of ALL 195 document-scoped DATE witnesses in the subject found no
# other state-stating predicate present, so this is not an under-supply
# against source. A future predicate must EARN equivalent semantics from
# source; resemblance to one of these is not admission.
STATE_PREDICATES = {
    "last updated": "the document states its own currency point",
    "updated": "the document states its own currency point",
    "version": "the document states its own version/state point",
}
# NOT HERE, AND EACH FOR A RULED REASON:
#   reviewed / last reviewed / review date  an inspection EVENT (class B)
#   created / generated / opened / started / prepared / planning date,
#   and the root lifecycle datelines sent / written / closed
#                                           an ORIGIN event (class C)
#   date                                    names a date, not a state (E)
#   H1 dates, bare datelines, contextual Status
#                                           no predicate at all (class D)
# All of these may still be document-scoped, may still earn SCOPE, and
# still emit CARRIES_DATE_STAMP. They do not establish VALIDITY.

# ── FUNCTION vocabulary ───────────────────────────────────────────────
# Terms are matched with WORD BOUNDARIES and an explicit plural
# allowance. v1.1 had bare substrings, so 'Audit' matched 'Auditor' and
# 'Plan' matched 'Plane' -- different words, not inflections.
FUNCTION_TERMS = {
    "GOVERNANCE": r"decision|doctrine|operating rule|protocol|governance|charter",
    "STATUS": r"status|progress|dashboard|report card|tracker",
    "PLAN": r"plan|roadmap|backlog|sequence|proposal",
    "EVIDENCE": r"audit|census|review|post-?mortem|measurement|research|"
                r"finding|analysis|report",
    "REFERENCE": r"reference|index|navigation|catalogue|glossary|"
                 r"specification|spec|architecture|design",
    "RUNTIME_INPUT": r"registry|agent|soul|teammate|skill|prompt",
    "USER_GUIDE": r"setup|guide|how to|runbook|installation|demo|playbook|"
                  r"checklist",
    "TEMPLATE": r"template|boilerplate|scaffold",
    "MARKER": r"marker|placeholder",
}


def term_match(term, text):
    """Word-boundary match allowing ONLY a plural 's' (D8).

    'Audit' matches 'Audits' but not 'Auditor'. 'Plan' matches 'Plans'
    but not 'Plane' or 'Planning'.
    """
    return re.search(rf"\b(?:{term})s?\b", text, re.I)


# A SUBSTANTIVE purpose statement: the trigger AND what follows it, so
# the closed-vocabulary negative is computed over content (D9). v1.1
# captured only the trigger -- 'This file', 'records' -- against which no
# function term could ever match, so OTHER was satisfied by construction.
PURPOSE = re.compile(
    r"^\s{0,3}[>\-*|]?\s*(?:this (?:document|file|register)\s+\w+|"
    r"(?:defines|records|holds|tracks))\s+(?P<body>[^.\n]{12,160})",
    re.I | re.M)


def _first_witness(row, *kinds):
    for k in kinds:
        for w in row["witnesses"].get(k, []):
            return Witness(**w)
    return None


def _predicate_of(w):
    """The SOURCE PREDICATE that labels this witness, or None.

    THE SCANNER'S OWN RECOGNISER, NOT A SECOND LIST. `passa._label_of` is
    the recogniser that granted this witness its document scope in the
    first place, so the same grammar decides which predicate granted it.
    A private extractor here would be a list kept beside the thing it
    governs (R5) and could admit a label M3 never saw.

    Position matters: `**Reviewed:** X · **Last updated:** Y` carries two
    predicates on one line, and Y's predicate is the second. The witness
    is located inside its own complete `local_context` line, and the label
    is read from the text BEFORE it -- the same geometry `_scope_of` used.
    If the value occurs more than once and the readings disagree, the
    position is ambiguous and this FAILS CLOSED to None rather than
    picking one.
    """
    line, value = w["local_context"], w["witness_value"]
    labels, i = set(), line.find(value)
    while i >= 0:
        labels.add(PA._label_of(line[:i]))
        i = line.find(value, i + 1)
    return labels.pop() if len(labels) == 1 else None


def _state_binding_witnesses(row):
    """EVERY document-scoped DATE witness that positively binds document
    state. ALL of them, never the first one found.

    M1's second root defect was that `_binding_witness(row, "DATE")`
    returns the FIRST WHOLE_FILE DATE in source order, so a `Created:` or
    a `Status:` line above a `Last updated:` line decided the verdict by
    position. Source order is not a semantic selection rule. This
    collects the qualified set and lets the caller test it whole.
    """
    return [(Witness(**w), lab) for w in row["witnesses"].get("DATE", [])
            if w["applicability_scope"] == "WHOLE_FILE"
            and (lab := _predicate_of(w)) in STATE_PREDICATES]


def _trace_witness(qualified):
    """One qualified witness, for the TRACE only.

    THIS IS NOT PRECEDENCE BETWEEN PREDICATES. The verdict is already
    fixed by the qualified set being non-empty; every member supports the
    same TIME_BOUND. The key is the witness's own source position, which
    is a property of the DOCUMENT and not of the list, so permuting the
    witness list cannot change either the verdict or the trace. No
    semantic ranking is invented, because source evidence does not
    require one.
    """
    def key(item):
        w, _lab = item
        try:
            line = int(w.source_selector.lstrip("L"))
        except ValueError:
            line = 1 << 30
        return (line, w.witness_value)
    return min(qualified, key=key)


def _binding_witness(row, *kinds):
    """A witness whose own applicability is already WHOLE_FILE.

    UNCHANGED BY M1, DELIBERATELY. `scope()` consumes this and its
    semantics are different: a root `Reviewed:` date IS applicable to the
    document as a whole, so SCOPE=WHOLE_FILE is CORRECT there and must not
    move. Repairing VALIDITY inside this shared helper would have moved
    146 SCOPE verdicts as collateral. The M1 repair is VALIDITY-specific
    and lives above.
    """
    for k in kinds:
        for w in row["witnesses"].get(k, []):
            if w["applicability_scope"] == "WHOLE_FILE":
                return Witness(**w)
    return None


# ── SCOPE (D6, D7) ────────────────────────────────────────────────────
def scope(row):
    w = _binding_witness(row, "COMMIT", "RUN_ID", "DATE", "SUPERSEDED_BY")
    if w is None:
        return E.abstain(
            "SCOPE", "a document-level binding witness",
            "no witness whose applicability is the document as a whole",
            "SCOPE is EARNED or absent. v1.1 emitted WHOLE_FILE as a "
            "'file-level default' on 272 of 272 rows; a value emitted "
            "when nothing was measured is a claim, not a measurement.")
    return E.claim(w, "WHOLE_FILE", scope="WHOLE_FILE",
                   rationale="a structured binding whose subject is the "
                             "document as a whole")


# ── VALIDITY (D1-D5) ──────────────────────────────────────────────────
def validity(row, contradiction):
    if contradiction is not None:
        return E.abstain(
            "VALIDITY", "a currency claim the history does not contradict",
            f"BINDING_CONTRADICTION: claims {contradiction['claimed']}, "
            f"history says {contradiction['git_last']} "
            f"(+{contradiction['drift_days']}d)",
            "an assertion the evidence refutes is an assertion plus a "
            "contradiction, not a verdict")

    for kinds, value in (("RUN_ID",), "RUN_ARTEFACT"), (("COMMIT",), "EXACT_SNAPSHOT"):
        w = _binding_witness(row, *kinds)
        if w is not None:
            return E.claim(w, value, scope="WHOLE_FILE",
                           rationale=f"document-level binding, witness kind "
                                     f"VERIFIED as {w.witness_type}")
    # M1. A document-scoped DATE is NECESSARY and NOT SUFFICIENT. The
    # verdict is decided by the qualified SET, never by which member of
    # it Pass A emitted first.
    qualified = _state_binding_witnesses(row)
    if qualified:
        w, lab = _trace_witness(qualified)
        return E.claim(w, "TIME_BOUND", scope="WHOLE_FILE",
                       rationale=f"document-state binding: {lab!r} — "
                                 f"{STATE_PREDICATES[lab]}")

    doc_dates = [(w["source_selector"], _predicate_of(w))
                 for w in row["witnesses"].get("DATE", [])
                 if w["applicability_scope"] == "WHOLE_FILE"]
    if doc_dates:
        # The M1 abstention, and it is a DIFFERENT observation from the
        # one below: dates DO bind this document, and none of them says
        # anything about its state. Both are reported; neither is
        # collapsed into the other.
        return E.abstain(
            "VALIDITY", "a witness that positively binds the document's "
                        "state, currency or version point",
            f"document-scoped dates present, none state-binding: "
            f"{[(s, p) for s, p in doc_dates]}",
            "a review, origin, lifecycle or bare date applies to the "
            "document without establishing its temporal validity. "
            "Someone reviewed it; that is not a currency interval.")

    seen = sorted(k for k, v in row["witnesses"].items() if v)
    return E.abstain(
        "VALIDITY", "a document-level binding witness of a verified kind",
        f"witness kinds present: {seen or 'none'}",
        "witnesses exist but bind a span, not the document. A commit "
        "cited for one sentence is CITES_COMMIT, not a whole-file "
        "snapshot binding.")


# ── FUNCTION (D8-D11, correction E) ───────────────────────────────────
def function(row, text):
    path, title = row["path"], row.get("title") or ""

    # objective witness: size and path role. This is the ONLY family that
    # earns FUNCTION at H2 -- it does not consult self-description.
    if row["bytes"] < 200 and path.endswith("README.md"):
        w = Witness(witness_type="SIZE_AND_ROLE",
                    witness_value=f"{row['bytes']} bytes",
                    source_path=path, source_selector="L1",
                    local_context=title[:120] or "(no title)",
                    applicability_scope="WHOLE_FILE",
                    evidence_total=1, evidence_shown=1, truncated=False,
                    polarity="POSITIVE", certainty="VERIFIED")
        return E.claim(w, "MARKER", rationale="objective: byte count and "
                                              "path role, not self-description")

    # self-description: an EVIDENCE FACT, never a verdict (correction E)
    pm = PURPOSE.search(text[:6000])
    body = pm.group("body") if pm else ""
    hits = sorted({v for v, t in FUNCTION_TERMS.items()
                   if term_match(t, title) or (body and term_match(t, body))})
    if not hits:
        return E.abstain(
            "FUNCTION", "an objective witness of functional role",
            f"no nominal function term in title or purpose "
            f"(purpose captured: {body[:40]!r})" if pm else
            "no nominal function term and no purpose statement",
            "H2 has no qualified positive rule for FUNCTION beyond the "
            "objective MARKER case")
    if len(hits) > 1:
        # D11: order must never decide. Ambiguity is reported.
        return E.abstain(
            "FUNCTION", "a single corroborated functional role",
            f"nominal terms for {hits} both present",
            "two roles are nominated and no evidence discriminates them; "
            "v1.1 resolved this by PATH_NOMINATION order, so a different "
            "ordering emitted a different verdict on identical evidence")
    return E.abstain(
        "FUNCTION", "an objective witness of functional role",
        f"NOMINAL_FUNCTION={hits[0]} from self-description",
        "path and title share provenance: one source counted twice, not "
        "two independent proofs (D367 correction E)")


# ── LIFECYCLE (carried forward from D363, scalar and keyword-only) ────
AUDIT_PATH = re.compile(r"CODE_AUDIT|AUDIT", re.I)
DATED_ARTEFACT = re.compile(r"_20\d\d-\d\d-\d\d|POSTMORTEM_|REALITY_CHECK_")


def lifecycle(*, path, superseded_by, snapshot_witness, blocked):
    """SCALAR AND KEYWORD-ONLY, carried forward from D363 unchanged.

    There is no row-like parameter, so no unauthorised Pass A field can
    enter. Two earlier attempts at this boundary were both overstated by
    me; the claim this earns, and no more: under this interface,
    LIFECYCLE has no parameter through which maintenance,
    self-currentness or consumption can arrive.

    LIFECYCLE WAS NOT IN THE D367 REPAIR SCOPE -- it was qualified under
    D361-D363 -- but its INPUTS changed underneath it: `has_sha` no
    longer exists, having been the flag that fired on 'ed25519', a unix
    timestamp, three run ids and a Docker digest. So v1.1's two
    HISTORICAL rules are ported onto the repaired evidence rather than
    dropped. Silently losing 13 verdicts would have been a scope change
    I had no authority to make.

    The snapshot rule is now STRICTLY STRONGER than v1.1's: it requires a
    commit witness VERIFIED by resolution AND bound at document scope,
    where v1.1 accepted any hex-shaped token anywhere in 6000 bytes.
    """
    if blocked:
        return E.unmeasured("LIFECYCLE", blocked)
    if superseded_by is not None:
        w = Witness(**superseded_by)
        return E.claim(w, "SUPERSEDED",
                       rationale="an explicitly named successor document")
    if snapshot_witness is not None and AUDIT_PATH.search(path):
        w = Witness(**snapshot_witness)
        return E.claim(w, "HISTORICAL",
                       rationale="an audit artefact bound to a VERIFIED "
                                 "snapshot commit at document scope")
    if DATED_ARTEFACT.search(path):
        w = Witness(witness_type="DATED_ARTEFACT_PATH", witness_value=path,
                    source_path=path, source_selector="path",
                    local_context=path, applicability_scope="WHOLE_FILE",
                    evidence_total=1, evidence_shown=1, truncated=False,
                    polarity="POSITIVE", certainty="OBSERVED")
        return E.claim(w, "HISTORICAL",
                       rationale="the path encodes a dated artefact whose "
                                 "scope is the whole document")
    return E.abstain(
        "LIFECYCLE", "a named successor, or a bound snapshot/dated-artefact "
                     "witness",
        f"superseded_by=None, document-scoped snapshot="
        f"{snapshot_witness is not None}",
        "ACTIVE is H2_NOT_EARNABLE: maintenance, self-assertion and "
        "consumption are evidence facts, not verdicts (D360 5)")


def classify(row, text, contradiction, blocked_by_axis=None):
    blocked_by_axis = blocked_by_axis or {}
    sup = (row["witnesses"].get("SUPERSEDED_BY") or [None])[0]
    out = {"path": row["path"]}
    snap = _binding_witness(row, "COMMIT")
    out["LIFECYCLE"] = lifecycle(
        path=row["path"], superseded_by=sup,
        snapshot_witness=snap.asdict() if snap else None,
        blocked=blocked_by_axis.get("LIFECYCLE"))
    out["FUNCTION"] = function(row, text)
    out["VALIDITY"] = validity(row, contradiction)
    out["SCOPE"] = scope(row)
    out["AUTHORITY"] = E.abstain(
        "AUTHORITY", "qualification of the document's claims against "
                     "programme authority",
        f"authority_claim={row.get('authority_claim')}",
        "H2 verifies no claim; AUTHORITY states are DEFERRED_TO_H3")
    out["GENERATION"] = E.abstain(
        "GENERATION", "a closed search space yielding NO_WRITER, or a "
                      "writer with established scope",
        f"writers={row['writers'] or 'none proven'}",
        "MANUAL/PARTIAL_DERIVED/FULL_DERIVED are H2_NOT_EARNABLE: the "
        "write search space is open and region scope is undetermined")

    # contract self-check: no verdict may be a value the contract forbids
    for axis in ont.ALPHABETS:
        v = out[axis]["value"]
        if v == ont.CAPABILITY_FAILURE:
            continue
        if not ont.emittable(axis, v):
            raise AssertionError(
                f"CONTRACT VIOLATION: emitted {axis}={v}, disposition "
                f"{ont.STATE_DISPOSITION.get((axis, v))}")
    return out
