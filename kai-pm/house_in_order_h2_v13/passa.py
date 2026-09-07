#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — PASS A. WITNESSES, NOT BOOLEANS. NO VERDICTS.

v1.1's Pass A emitted `has_sha=True`. That boolean threw away the token,
its position, its kind and its scope -- everything a verdict would need
in order not to over-claim -- and the verdict layer then had no choice
but to guess. Six of the seventeen registered defects live in that one
design decision.

v1.2 emits `Witness` objects carrying the nine fields of D367 5. The
verdict layer cannot widen them, because `envelope.conserve()` will not
let it.

WHAT IS REPAIRED HERE

  D2  witness kind assumed from shape. `\\b[0-9a-f]{7,40}\\b` is not a
      commit-shaped test: it matched 'ed25519' (an algorithm name), a
      unix timestamp, three workflow run ids and a fragment of
      `sha256:b5e68a3…` (a Docker digest). Kind is now DISCRIMINATED --
      by prefix, by composition, and by RESOLUTION against the declared
      history source -- never assumed from a character class.

  D3  self-claims never checked. `last` was already in the row and was
      never consulted. A currency claim the history contradicts now
      produces BINDING_CONTRADICTION and cannot earn a verdict.

  D4  the RUN detector could not fire. Its class `[ :#]*` admitted
      space, colon and hash but not `*` or a backtick, so
      `**Last run:** 31570714150` missed and the SHA pattern took the
      digits first. RUN_ARTEFACT's population of 0 was a regex artefact
      reported as a measurement. Discrimination is now by KIND, so
      fixing precedence alone -- the instance -- is not what happened.

  D5  the raw `present_tense` flag earned CURRENT_TREE on 7 documents,
      0 of which carried the qualified SELF_ASSERTS_CURRENT fact. The
      raw flag no longer exists as a verdict input at all.

SCOPE IS DETERMINED, NOT ASSUMED. A witness sitting in an inline
sentence is SPAN. Only an explicit structured binding whose subject is
the document as a whole yields WHOLE_FILE. The binding predicates are a
DECLARED CLOSED-WORLD SET (D367 6 / Kai's M2 ruling): declared as such,
carrying a per-entry rationale, calibrated with known-positive and
known-negative fixtures, and printing its own denominator. A closed-world
list that announces itself is not the R5 defect; a list kept beside the
thing and passed off as derived is.

CYCLE 6 adds ONE further route, and it is a CONTEXTUAL one: a predicate
that is not intrinsically document-binding may still take the document
as its subject when the surrounding structure establishes it. That set
is declared separately, in CONTEXTUAL_PREDICATES, and holds exactly one
entry. Its conditions are structural and all required; see the block
comment there for the defect it repairs.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from envelope import Witness                                   # noqa: E402

HEAD_BYTES = 6000

# ── token recognisers. NONE of these decides a kind on its own ────────
HEX = re.compile(r"\b[0-9a-f]{7,40}\b")
DIGEST_PREFIX = re.compile(r"(?:sha256|sha512|sha1)\s*:\s*$", re.I)
DECIMAL_RUN = re.compile(r"\b\d{8,}\b")
# D4 REPAIRED. v1.1 admitted only [ :#], so markdown emphasis and code
# spans defeated it. Enumerating the punctuation that may intervene is
# how that defect was built, so the gap is defined NEGATIVELY instead:
# anything up to 10 characters CONTAINING NO LETTERS. Emphasis, colons,
# backticks, brackets and an ordinal ("Deployed run 1 (`…") all pass;
# "the run took 3 seconds" does not, because the gap has letters in it.
RUN_NEAR = re.compile(r"\brun\b(?:\s*(?:id|number))?[^A-Za-z]{0,10}$", re.I)
RUN_URL = re.compile(r"actions/runs/\d+", re.I)

DATE = re.compile(
    r"\b20\d\d-\d\d-\d\d\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\.?,?\s+20\d\d\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},?\s+20\d\d\b", re.I)

SUPBY = re.compile(r"superseded by\s+`?([A-Za-z0-9_./-]+\.md)`?", re.I)
SUPES = re.compile(r"\bsupersedes\b", re.I)

# ── the DECLARED CLOSED-WORLD binding predicates ──────────────────────
# A label whose grammatical subject is THE DOCUMENT ITSELF. Declared,
# not derived -- there is no tree to derive it from -- so it is declared
# closed-world, carries a rationale per entry, and cal_fixtures.py holds
# a known-negative for each family it must NOT match.
BINDING_PREDICATES = {
    r"audited snapshot": "the document states the snapshot it audits",
    r"acquisition commit": "the document states the commit it was taken at",
    r"validated checkpoint": "the document states its validated point",
    r"findings-bearing[^:]*snapshot": "the document states its findings base",
    r"subject": "the document names its measurement subject",
    r"measured at": "the document states its measurement point",
    r"snapshot": "the document states its snapshot",
    r"last updated": "the document states its own currency",
    r"last reviewed": "the document states its own review point",
    r"reviewed": "the document states its own review point",
    r"planning date": "the document states its own authoring date",
    r"date": "the document states its own date",
    r"version": "the document states its own version point",
    # CYCLE 4. Document-lifecycle predicates, source-confirmed in the
    # accepted regression set. Same declared closed-world standing and
    # the same form of rationale as the entries above.
    r"created": "the document states its own creation point",
    r"generated": "the document states its own generation point",
    r"opened": "the document states when it was opened",
    r"updated": "the document states its own currency",
    r"finali[sz]ed": "the document states its own completion point",
    r"report completed": "the document states its own completion point",
    r"(?:log|register) started": "the document states when its record began",
    r"started": "the document states when its record began",
    r"prepared": "the document states its own preparation point",
    r"review date": "the document states its own review point",
    r"sent": "the document states when it was sent",
    r"written": "the document states when it was written",
    r"agreed": "the document states when it was agreed",
}
# The POSITIVE semantic authority for INPUT 3 at document root. Cycle 3
# declared these predicates document-binding and then rejected several of
# them through ARTEFACT_LABEL because their VALUES are artefacts --
# "audited snapshot" was declared binding and refused for containing the
# word "snapshot". That contradiction was the cycle-3 root cause.
# ARTEFACT_LABEL is removed. The question is whether the field defines the
# document's own provenance or scope, not whether the label names an
# artefact.
DOC_BINDING = re.compile(r"^(?:" + "|".join(BINDING_PREDICATES) + r")$", re.I)
# A structured binding is LABEL, optional markdown emphasis, then a
# colon, at the START OF A LINE. The requirement is structural-semantic
# (a labelled field whose subject is the document), not "must be in the
# first N lines" -- a header-layout rule would be a different defect
# wearing the same clothes.
BINDING_LINE = re.compile(
    r"^\s{0,3}[>\-*|]?\s*[*_`]{0,2}\s*(" +
    "|".join(BINDING_PREDICATES) + r")\s*[*_`]{0,2}\s*:", re.I)


# ── M3 scope derivation: the four authorised inputs ───────────────────
# Rev4: WHOLE_FILE requires BOTH a structural document-level binding AND
# subject = the document as a whole. Detector is NOT an input.

TABLE_ROW = re.compile(r"^\s{0,3}\|")
H1 = re.compile(r"^#[^#]", re.M)
SECTION = re.compile(r"^#{2,}\s", re.M)
# A bounded parenthetical qualifier is part of the label, not a new
# predicate: "Last updated (UTC):" is the same field as "Last updated:".
LABEL_LINE = re.compile(r"^\s{0,3}>?\s*(?:[-*+]\s+)?[*_`]{0,2}\s*"
                        r"([A-Za-z][A-Za-z0-9 ._/-]{0,40}?"
                        r"(?:\s*\([^)]{0,20}\))?)"
                        r"\s*[*_`]{0,2}\s*:[*_`]{0,2}\s")
QUALIFIER = re.compile(r"\s*\([^)]*\)")
SELF_SUBJECT = re.compile(
    r"\bthis\s+(?:document|file|register|tracker|log|report|plan|brief|"
    r"note|index|census|audit|spec|specification|record|policy|guide)\b",
    re.I)
# A root LIFECYCLE DATELINE carries the same meaning as a labelled field
# without the colon: "Sent 2026-08-07", "Written 2026-08-07", "Agreed
# with the operator on 2026-08-07", "Closed 2026-08-12". Bounded like
# RUN_NEAR: the verb must sit within 40 characters before the witness and
# in the same clause, so a date in ordinary prose does not qualify.
ROOT_LIFECYCLE = re.compile(
    r"\b(?:sent|written|agreed|closed|prepared|published|issued|completed|"
    r"finali[sz]ed|recorded|measured|started|opened|created|generated|"
    r"drafted|updated|reviewed)\b[^.;]{0,40}$", re.I)
# A bare leading date IS the document's dateline when it opens the line
# at root.
# A bare leading date IS the document's dateline when it opens a root
# line. CYCLE 5: this route is now restricted to the DATE detector.
# Cycle 4 required only that nothing precede the witness on its line, so
# a wrapped continuation of a run list --
#   "31568526480 / `189500b`."
# in EMBEDDING_BACKEND_STATE L16 -- promoted a RUN ID as though it were a
# dateline. The defect was DETECTOR-CLASS LEAKAGE, not position: a
# genuine dateline may legitimately follow other root metadata, so the
# repair is to the detector class and not to the line's ordinal.
BARE_DATELINE = re.compile(r"^\s{0,3}>?\s*[*_`]{0,2}\s*$")
DATELINE_DETECTORS = ("DATE",)

# ── CYCLE 6. CONTEXT-DEPENDENT DOCUMENT METADATA ──────────────────────
# THE ROOT DEFECT REPAIRED HERE. Cycle 5 read a generic label as a
# TERMINAL VETO -- `if not DOC_BINDING.match(lab): return "SPAN"` -- so
# once a label failed the intrinsic test no further evidence could be
# considered, however strong. That is a category error: the absence of
# INTRINSIC document-binding evidence is not PROOF OF LOCAL SCOPE. The
# veto is now a failed conjunct, and the remaining authorised evidence is
# examined. That is the repair; the class below is what the repair lets
# through, and nothing else is admitted by it.
#
# `Status` is neither intrinsically document-binding nor intrinsically
# local. Under a task heading it is the task's status. In a document's
# own root metadata block, beside a field that has ALREADY earned
# document binding on its own terms, its subject is the document.
#
# IT IS DELIBERATELY NOT IN BINDING_PREDICATES. Adding it there would
# promote every root `Status:` on label authority alone -- the same
# defect in the opposite direction. `status` is the ONLY contextual
# predicate; Owner, Purpose and Branch are NOT admitted.
CONTEXTUAL_PREDICATES = {
    "status": "document-level state, and ONLY inside the root metadata "
              "block of a document that has independently earned an "
              "intrinsic document-binding field",
}
# A root metadata line begins the line itself, with nothing but markdown
# emphasis before the label. A bullet, a blockquote marker or leading
# indentation introduces an ENTRY, and the entry is then a nearer subject
# than the document (condition 4).
ROOT_METADATA_LINE = re.compile(r"^[*_`]{0,2}[A-Za-z]")
# Condition 5. The statement must describe the DOCUMENT's state, not an
# artefact or a person it mentions. A handle, a link, a quoted
# identifier, a named file or an issue reference in the value IS a
# nearer subject. This test fails CLOSED -- toward SPAN, the pre-cycle-6
# answer -- because an over-narrow contextual class costs a missed
# promotion, while an over-broad one manufactures WHOLE_FILE.
EMBEDDED_SUBJECT = re.compile(
    r"@[A-Za-z0-9_-]+"
    r"|\[[^\]]*\]\([^)]*\)"
    r"|`[^`]*`"
    r"|\b[\w./-]+\.(?:md|py|ya?ml|json|tsv|txt|sh|toml)\b"
    r"|#\d+", re.I)


def _preamble_end(text):
    m = SECTION.search(text)
    return m.start() if m else len(text)


def _label_of(before):
    """The normalised predicate of a labelled field, or None."""
    m = LABEL_LINE.match(before)
    if not m:
        return None
    return QUALIFIER.sub("", m.group(1)).strip().lower()


def _label_hits(text, lab):
    """INPUT 2's denominator: how many non-table lines carry this label."""
    return sum(1 for ln in text.split("\n")
               if not TABLE_ROW.match(ln) and _label_of(ln) == lab)


def _block_of(text, ls):
    """The contiguous non-blank block holding the line at offset `ls`.

    Returns (lines, index of that line within them). Contiguity is the
    structural expression of 'the same root metadata block': a blank line
    ends the block, so a field further down the preamble is not a
    neighbour.
    """
    lines = text.split("\n")
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1
    i = max(k for k, s in enumerate(starts) if s <= ls)
    a = i
    while a > 0 and lines[a - 1].strip():
        a -= 1
    b = i
    while b + 1 < len(lines) and lines[b + 1].strip():
        b += 1
    return lines[a:b + 1], i - a


def _earned_binding_anchor(text, block, skip):
    """A block member that earns document binding on its OWN terms.

    Condition 3, and the word doing the work is INDEPENDENTLY: the
    anchor must be an intrinsic BINDING_PREDICATES field that would
    itself return WHOLE_FILE -- including INPUT 2's uniqueness test. A
    repeated per-entry stamp is not an anchor, so anchor existence alone
    cannot be borrowed by an arbitrary neighbour.
    """
    for k, ln in enumerate(block):
        if k == skip:
            continue
        lab = _label_of(ln)
        if lab is None or not DOC_BINDING.match(lab):
            continue
        if _label_hits(text, lab) <= 1:
            return lab, ln
    return None


def _contextual_document_metadata(text, ls, lab, line):
    """The five structural conditions, ALL required. Any one absent and
    the answer stays SPAN.

    1  root / pre-H2          -- established by the caller before this runs
    2  not a table row        -- established by the caller before this runs
    3  same contiguous root metadata block as an independently earned
       intrinsic document-binding field
    4  no nearer section, entry or register subject
    5  the statement describes document-level state
    """
    if lab not in CONTEXTUAL_PREDICATES:
        return False
    if not ROOT_METADATA_LINE.match(line):          # 4: it is an entry
        return False
    block, idx = _block_of(text, ls)
    # 4 again: prose, a heading, a table row or a list entry sharing the
    # block means the block is not a root metadata block at all.
    if any(_label_of(ln) is None or not ROOT_METADATA_LINE.match(ln)
           for ln in block):
        return False
    if _earned_binding_anchor(text, block, idx) is None:    # 3
        return False
    _, _, value = line.partition(":")
    return not EMBEDDED_SUBJECT.search(value)               # 5


def _scope_of(text, start, detector=None):
    """WHOLE_FILE iff BOTH conjuncts of the Rev4 rule hold, derived
    separately. Uniqueness (INPUT 2) never promotes on its own.

    `detector` gates the bare-dateline route only (cycle 5). It is NOT a
    scope input anywhere else: Rev4 excludes it, and every other route
    here decides on structure and subject alone.
    """
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", start)
    line = text[ls:le if le >= 0 else len(text)]
    before = text[ls:start]

    # INPUT 4. A table row's subject is the row.
    if TABLE_ROW.match(line):
        return "SPAN"
    # INPUT 1. Inside an H2+ section the section is a nearer subject.
    if ls >= _preamble_end(text):
        return "SPAN"

    # ---- document root ----
    m1 = H1.search(text)
    if m1 and m1.start() == ls:                 # the document names itself
        return "WHOLE_FILE"
    if SELF_SUBJECT.search(line):               # INPUT 3, explicit
        return "WHOLE_FILE"

    lab = _label_of(before)
    if lab is not None:
        # INPUT 3. A root label promotes ONLY on positive document-binding
        # authority. A generic unique label proves nothing about subject.
        if DOC_BINDING.match(lab):
            # INPUT 2, and only now: one document-level field, or a
            # repeated per-entry stamp?
            return "WHOLE_FILE" if _label_hits(text, lab) <= 1 else "SPAN"
        # CYCLE 6. Failing the INTRINSIC test is a failed conjunct, NOT a
        # terminal veto. Absence of intrinsic document-binding evidence is
        # not proof of local scope, so the remaining authorised evidence is
        # examined -- and for a labelled field that evidence is exactly one
        # class, the contextual root metadata class above.
        #
        # It deliberately does NOT fall through to the label-free routes
        # below. ROOT_LIFECYCLE and BARE_DATELINE read the text BEFORE the
        # witness as an unlabelled dateline; a labelled field has already
        # declared its own predicate, and letting it reach those routes
        # would widen the repair past the class Kai demonstrated.
        if _contextual_document_metadata(text, ls, lab, line):
            return "WHOLE_FILE"
        return "SPAN"

    # INPUT 3 without a colon: a root lifecycle dateline.
    if ROOT_LIFECYCLE.search(before):
        return "WHOLE_FILE"
    if BARE_DATELINE.match(before) and detector in DATELINE_DETECTORS:
        return "WHOLE_FILE"

    return "SPAN"


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=str(repo),
                          capture_output=True, text=True)


def _selector(text, start, end=None):
    line = text[:start].count("\n") + 1
    if end is None:
        return f"L{line}"
    last = text[:end].count("\n") + 1
    return f"L{line}" if last == line else f"L{line}-L{last}"


def _context(text, start, end):
    """The COMPLETE logical source line(s) the witness sits in. No cap.

    D367 5: "local_context -- surrounding text sufficient to judge the
    match", and "SILENT TRUNCATION IS FORBIDDEN. The evidence actually
    responsible for the emitted cell must always be recoverable ...
    without guessing which source fragment mattered."

    v1.2 took `line[:200]` -- the first 200 characters OF THE LINE,
    wherever the match sat. A long line with a late match shipped a
    context that did not contain the witness at all: 40 clipped contexts,
    of which 10 (every one a TECH_WATCH DATE row) lacked their own
    `witness_value`, all declaring `truncated=False`.

    A centred 200-character budget was the first repair. It proved
    PRESENCE but not COMPLETENESS, so it left silent truncation in place
    at the line's edges. THE CAP IS NOW GONE.

    There is deliberately no tenth field and no reuse of `truncated`.
    `truncated` is bound by envelope.py to `evidence_shown <
    evidence_total` -- a row-count property -- and context clipping is a
    different truncation wearing the same word. Carrying the complete
    line dissolves the collision instead of encoding it. Where a source
    region is genuinely too large to carry inline, D367 5's authorised
    alternative is a stable selector plus a hash-bound sidecar -- never a
    silent clip.
    """
    ls = text.rfind("\n", 0, start) + 1
    le = text.find("\n", end)
    return text[ls:le if le >= 0 else len(text)].strip()


def classify_token_kind(text, m, history_repo, subject):
    """D2: DISCRIMINATE the kind. Never assume it from the character class.

    Order is by DISCRIMINATING EVIDENCE, not by rule precedence -- fixing
    RUN-before-SHA would have been the instance, not the class.
    """
    tok = m.group(0)
    before = text[max(0, m.start() - 24):m.start()]
    window = text[max(0, m.start() - 24):m.end()]

    if DIGEST_PREFIX.search(before):
        return "DIGEST_FRAGMENT", False        # sha256:b5e68a3… is not a commit
    if RUN_NEAR.search(before) or RUN_URL.search(window):
        return "RUN_ID", False
    if tok.isdigit():
        return "DECIMAL_TOKEN", False          # 1700000000 is a timestamp
    # the only positive commit test: does it RESOLVE in the declared
    # history source? 'ed25519' does not, and neither does a digest.
    resolved = git(history_repo, "cat-file", "-e",
                   f"{tok}^{{commit}}").returncode == 0
    if resolved:
        return "COMMIT", True
    return "HEX_SHAPED_UNRESOLVED", False


def _eligible(m):
    """D14 / A5-ii, START-BOUND. The boundary decides WHICH TOKENS ARE
    ELIGIBLE; it must never decide WHAT AN ELIGIBLE TOKEN IS.

    v1.2 matched against `text[:HEAD_BYTES]`, so a token straddling the
    boundary was recognised in its cut form: 76dbba4... was emitted as a
    29-character prefix with `truncated=False` (D14-A), and a token cut
    below the recogniser's 7-character minimum vanished entirely (D14-B).
    The character index was deciding what a token *was*.

    Recognition now runs against the COMPLETE source and admission is by
    the frozen start predicate alone. `token_end` MAY exceed HEAD_BYTES.
    A token whose START is at or beyond the boundary is NOT admitted
    merely because the scanner can now see it -- that would be A5-i,
    which was rejected at a measured 491 -> ~1973 records.

    This is not a larger HEAD_BYTES and not a guessed extension margin.
    """
    return m.start() < HEAD_BYTES


def scan(path, text, history_repo, subject):
    """Every witness whose token STARTS in the head window, carried whole.
    NO verdict is formed here.

    `head` survives for ONE purpose: _scope_of's uniqueness universe. Its
    denominator is deliberately left unchanged here, because moving it is
    an M3 decision (step 2) and step 1 may not shift applicability scope.
    """
    head = text[:HEAD_BYTES]
    out = collections.defaultdict(list)

    for m in HEX.finditer(text):
        if not _eligible(m):
            continue
        kind, is_commit = classify_token_kind(text, m, history_repo, subject)
        out["COMMIT" if is_commit else kind].append(Witness(
            witness_type=kind, witness_value=m.group(0), source_path=path,
            source_selector=_selector(text, m.start()),
            local_context=_context(text, m.start(), m.end()),
            applicability_scope=_scope_of(head, m.start(), "HEX"),
            evidence_total=1, evidence_shown=1, truncated=False,
            polarity="POSITIVE", certainty="VERIFIED" if is_commit else "OBSERVED"))

    for m in DECIMAL_RUN.finditer(text):
        if not _eligible(m):
            continue
        before = text[max(0, m.start() - 24):m.start()]
        window = text[max(0, m.start() - 24):m.end()]
        if not (RUN_NEAR.search(before) or RUN_URL.search(window)):
            continue
        out["RUN_ID"].append(Witness(
            witness_type="RUN_ID", witness_value=m.group(0), source_path=path,
            source_selector=_selector(text, m.start()),
            local_context=_context(text, m.start(), m.end()),
            applicability_scope=_scope_of(head, m.start(), "DECIMAL_RUN"),
            evidence_total=1, evidence_shown=1, truncated=False,
            polarity="POSITIVE", certainty="OBSERVED"))

    for m in DATE.finditer(text):
        if not _eligible(m):
            continue
        out["DATE"].append(Witness(
            witness_type="DATE_STAMP", witness_value=m.group(0),
            source_path=path, source_selector=_selector(text, m.start()),
            local_context=_context(text, m.start(), m.end()),
            applicability_scope=_scope_of(head, m.start(), "DATE"),
            evidence_total=1, evidence_shown=1, truncated=False,
            polarity="POSITIVE", certainty="OBSERVED"))

    m = next((x for x in SUPBY.finditer(text) if _eligible(x)), None)
    if m:
        out["SUPERSEDED_BY"].append(Witness(
            witness_type="NAMED_SUCCESSOR", witness_value=m.group(1),
            source_path=path, source_selector=_selector(text, m.start()),
            local_context=_context(text, m.start(), m.end()),
            applicability_scope="WHOLE_FILE",   # supersession is file-level
            evidence_total=1, evidence_shown=1, truncated=False,
            polarity="POSITIVE", certainty="OBSERVED"))
    return dict(out)


def build(subject_repo, history_repo, subject, census_pkg):
    sys.path.insert(0, str(census_pkg))
    import docgraph as G, opscan as O, claims as C     # frozen Census v1.1

    tracked = G.tracked_md(subject_repo)
    edges = G.build_graph(subject_repo, tracked)
    inc = G.incoming(edges)
    out_deg = collections.Counter(s for s, d, k, _r, _c in edges if d)

    ops = O.collect(subject_repo, tracked)[0]
    C.classify(ops, tracked, set(O.tracked(subject_repo)))
    exe, writers, readers = (collections.Counter(),
                             collections.defaultdict(set),
                             collections.defaultdict(set))
    # E1: the census Op carries src AND line AND expr. v1.2 kept only the
    # source path, so a CONSUMED_AT_SUBJECT fact had no locator anywhere
    # in the package and could not be traced to the reference that made
    # it true. The locator is retained; `readers` keeps its old shape so
    # no existing consumer changes.
    reader_ops = collections.defaultdict(list)
    for o in ops:
        if o.target and o.disposition in ("RESOLVED_READ", "RESOLVED_WRITE",
                                          "READ_AND_WRITE"):
            exe[o.target] += 1
            (writers if o.disposition != "RESOLVED_READ"
             else readers)[o.target].add(o.src)
            if o.disposition == "RESOLVED_READ":
                reader_ops[o.target].append(
                    {"src": o.src, "line": int(o.line or 0),
                     "mode": o.mode, "expr": o.expr,
                     "disposition": o.disposition})

    rows = []
    for d in tracked:
        txt = (pathlib.Path(subject_repo) / d).read_text(errors="ignore")
        title = ""
        for ln in txt.splitlines():
            if ln.startswith("#"):
                title = ln.lstrip("#").strip()[:120]
                break
        n = git(history_repo, "rev-list", "--count", subject, "--",
                d).stdout.strip()
        last = git(history_repo, "log", "-1", "--format=%ad", "--date=short",
                   subject, "--", d).stdout.strip()
        rows.append({
            "path": d, "title": title, "bytes": len(txt.encode()),
            "sha256": hashlib.sha256(txt.encode()).hexdigest()[:16],
            "commits_in_window": int(n or 0), "last": last,
            "graphA_in": inc.get(d, 0), "graphA_out": out_deg.get(d, 0),
            "exe_ops": exe.get(d, 0), "writers": sorted(writers.get(d, ())),
            "readers": sorted(readers.get(d, ())),
            "reader_ops": reader_ops.get(d, []),
            "says_supersedes": bool(SUPES.search(txt[:HEAD_BYTES])),
            "witnesses": {k: [w.asdict() for w in v] for k, v in
                          scan(d, txt, history_repo, subject).items()},
        })
    assert len(rows) == len(tracked), "PASS A population mismatch"
    return rows, tracked


def main():
    ap = argparse.ArgumentParser(description="HOUSE_H2 v1.2 Pass A")
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--history-repo", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sr, hr = pathlib.Path(a.subject_repo), pathlib.Path(a.history_repo)
    head = git(sr, "rev-parse", "HEAD").stdout.strip()
    if head != a.subject:
        raise SystemExit(f"R11 ABORT: subject repo HEAD {head[:12]} != "
                         f"subject {a.subject[:12]}")
    if git(hr, "cat-file", "-e", f"{a.subject}^{{commit}}").returncode != 0:
        raise SystemExit("R11 ABORT: subject commit absent from history source")
    if git(hr, "rev-parse", "--is-shallow-repository").stdout.strip() != "false":
        raise SystemExit(
            "R11 ABORT: history source is SHALLOW. It does not fail on these "
            "queries -- it returns its graft boundary as a plausible date. "
            "Refusing to measure.")

    rows, tracked = build(sr, hr, a.subject, pathlib.Path(a.census_package))
    cm = pathlib.Path(a.census_package) / "MANIFEST.sha256"
    payload = {
        "subject": a.subject,
        "subject_tree": git(sr, "rev-parse", f"{a.subject}^{{tree}}").stdout.strip(),
        "history_identity": {
            "shallow": "false",
            "oldest_reachable_commit": git(hr, "log", "--reverse",
                "--format=%H").stdout.split("\n")[0],
            "oldest_reachable_date": git(hr, "log", "--reverse",
                "--format=%ad", "--date=short").stdout.split("\n")[0],
            "newest_date": git(hr, "log", "-1", "--format=%ad",
                               "--date=short").stdout.strip(),
            "subject_ancestry_depth": int(git(hr, "rev-list", "--count",
                                              a.subject).stdout.strip() or 0),
        },
        "census_dependency": {"package": str(a.census_package),
                              "aggregate": hashlib.sha256(
                                  cm.read_bytes()).hexdigest()},
        "population": len(tracked), "rows": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))

    kinds = collections.Counter(k for r in rows for k in r["witnesses"]
                                for _ in r["witnesses"][k])
    print(f"PASS A v1.2 COMPLETE — {len(rows)} rows == population {len(tracked)}")
    print("  WITNESS KINDS DISCRIMINATED (D2/D4), not assumed from shape:")
    for k, n in kinds.most_common():
        print(f"    {k:<26}{n:>5}")
    wf = sum(1 for r in rows for v in r["witnesses"].values() for w in v
             if w["applicability_scope"] == "WHOLE_FILE")
    sp = sum(1 for r in rows for v in r["witnesses"].values() for w in v
             if w["applicability_scope"] == "SPAN")
    print(f"  scope determined: WHOLE_FILE {wf} · SPAN {sp}")
    print(f"  binding predicates declared closed-world: "
          f"{len(BINDING_PREDICATES)}")
    print(f"  contextual predicates declared closed-world: "
          f"{len(CONTEXTUAL_PREDICATES)} "
          f"({', '.join(sorted(CONTEXTUAL_PREDICATES))})")
    print("  NO VERDICT ASSIGNED IN PASS A.")


if __name__ == "__main__":
    main()
