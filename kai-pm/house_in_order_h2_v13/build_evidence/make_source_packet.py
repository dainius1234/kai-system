#!/usr/bin/env python3
"""MECHANICAL SOURCE PACKET for M3 adjudication. SOURCE FACTS ONLY.

Prepares, for each of the 492 banked identities, the raw source evidence
Kai needs to adjudicate the record himself. It carries NO judgement of
any kind: no scope, no classification, no comparison against the
candidate, no historical figure.

Survey observations for the Check Engineer. Not the signed check.

TWO DESIGN DECISIONS DECLARED RATHER THAN BURIED
-----------------------------------------------
1. `_scope_of` IS NEVER CALLED, and neither is any classifier in the
   candidate. There is no judgement computed and then withheld; none is
   computed at all.

2. THE LABEL EXTRACTOR IS DELIBERATELY BROADER THAN THE CANDIDATE'S.
   `passa.LABEL_LINE` encodes what the CANDIDATE recognises as a
   labelled field. Reusing it would silently pre-filter the packet to
   the candidate's own grammar: any field the candidate fails to see
   would be absent from the evidence, and the adjudicator could not
   discover a recognition gap that the packet had already hidden. This
   file therefore uses its own, broader, independent geometry --
   anything up to 60 characters before a colon at line start. It
   OVER-supplies. Over-supplying raw source is recoverable; silently
   under-supplying it is not.

WHAT "MECHANICAL GEOMETRY, NOT SEMANTIC JUDGEMENT" MEANS HERE
   heading ancestry  -> the raw heading LINES, never "ROOT" or "SECTION"
   table row         -> the raw line, plus whether it begins with a pipe
   labelled field    -> the raw label text and its repetition count,
                        never "document binding"
   self-reference    -> the enclosing paragraph verbatim, never a boolean

Every field is derived from the frozen git tree. Content is read with
`git show <tree>:<path>` and fails closed on an unresolvable tree, an
unresolvable path, or non-strict UTF-8.

    python3 make_source_packet.py --subject-repo R --tree T \\
        --identities STEP2_M3_IDENTITY_MANIFEST.tsv --out F
    python3 make_source_packet.py ... --prove
"""
from __future__ import annotations
import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent

# INDEPENDENT of passa.LABEL_LINE, and deliberately broader. See note 2.
LABEL_GEOMETRY = re.compile(
    r"^\s{0,3}>?\s*(?:[-*+]\s+)?[*_`]{0,2}\s*([^:\n]{1,60}?)\s*[*_`]{0,2}\s*:")
QUALIFIER = re.compile(r"\s*\([^)]*\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")

FIELD_ORDER = ("path", "start", "end", "detector", "source_selector",
               "matched_text", "line_number", "line_raw",
               "line_begins_with_pipe", "paragraph_raw",
               "heading_ancestry", "first_level2plus_heading_line",
               "label_raw", "label_normalised",
               "label_normalised_count_in_document")

FORBIDDEN = ("WHOLE_FILE", "SPAN", "applicability", "expected_scope",
             "promotion", "demotion", "correct", "incorrect", "PASS", "FAIL")


def _git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout if binary else r.stdout.decode()


def read_from_tree(repo, tree, path):
    blob = _git(repo, "show", f"{tree}:{path}", binary=True)
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(f"R11 ABORT: {path} at {tree[:12]} is not strictly "
                         f"UTF-8 ({e}).")


def normalise_label(raw):
    return QUALIFIER.sub("", raw).strip().lower()


def label_counts(text):
    """Every normalised label in the document, with its occurrence count."""
    counts = {}
    for ln in text.splitlines():
        m = LABEL_GEOMETRY.match(ln)
        if m:
            k = normalise_label(m.group(1))
            counts[k] = counts.get(k, 0) + 1
    return counts


def doc_facts(text):
    lines = text.splitlines()
    starts, off = [], 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1
    headings = []
    for i, ln in enumerate(lines):
        m = HEADING.match(ln)
        if m:
            headings.append((i, len(m.group(1)), ln))
    first_h2 = next((i for i, lvl, _ in headings if lvl >= 2), None)
    return lines, starts, headings, first_h2, label_counts(text)


def packet_for(path, text, start, end, detector, selector, facts):
    lines, starts, headings, first_h2, counts = facts
    li = max(i for i, s in enumerate(starts) if s <= start)
    line = lines[li]

    # RAW heading ancestry: the nearest preceding heading at each level.
    anc, seen = [], set()
    for i, lvl, raw in reversed([h for h in headings if h[0] < li]):
        if lvl in seen or (anc and lvl >= anc[-1]["level"]):
            continue
        seen.add(lvl)
        anc.append({"line_number": i + 1, "level": lvl, "raw": raw})
    anc.reverse()

    # RAW enclosing paragraph: the consecutive non-blank block.
    a = li
    while a > 0 and lines[a - 1].strip():
        a -= 1
    b = li
    while b + 1 < len(lines) and lines[b + 1].strip():
        b += 1
    para = "\n".join(lines[a:b + 1])

    m = LABEL_GEOMETRY.match(line[:max(0, start - starts[li]) + 1]) or \
        LABEL_GEOMETRY.match(line)
    lab = m.group(1).strip() if m else None
    norm = normalise_label(lab) if lab is not None else None

    return {"path": path, "start": start, "end": end, "detector": detector,
            "source_selector": selector,
            "matched_text": text[start:end],
            "line_number": li + 1, "line_raw": line,
            "line_begins_with_pipe": line.lstrip().startswith("|"),
            "paragraph_raw": para,
            "heading_ancestry": anc,
            "first_level2plus_heading_line":
                (first_h2 + 1) if first_h2 is not None else None,
            "label_raw": lab, "label_normalised": norm,
            "label_normalised_count_in_document":
                counts.get(norm) if norm is not None else None}


def build(repo, tree, identities_path):
    rows = []
    for ln in pathlib.Path(identities_path).read_text(
            encoding="utf-8").splitlines()[1:]:
        p, s, e, det, sel = ln.split("\t")
        rows.append((p, int(s), int(e), det, sel))
    cache, out = {}, []
    for p, s, e, det, sel in sorted(rows):
        if p not in cache:
            cache[p] = (read_from_tree(repo, tree, p),)
            cache[p] = (cache[p][0], doc_facts(cache[p][0]))
        text, facts = cache[p]
        out.append(packet_for(p, text, s, e, det, sel, facts))
    return out


def serialise(records):
    lines = []
    for r in records:
        ordered = {k: r[k] for k in FIELD_ORDER}
        lines.append(json.dumps(ordered, ensure_ascii=False, sort_keys=False))
    text = "\n".join(lines) + "\n"
    keys = set()
    for r in records:
        keys |= set(r)
    for bad in FORBIDDEN:
        if any(bad.lower() in k.lower() for k in keys):
            raise SystemExit(f"R11 ABORT: forbidden field name {bad!r}.")
    return text


def _prove(repo, tree, ids):
    ok = True
    a = hashlib.sha256(serialise(build(repo, tree, ids)).encode()).hexdigest()
    b = hashlib.sha256(serialise(build(repo, tree, ids)).encode()).hexdigest()
    print(f"  deterministic regeneration : "
          f"{'IDENTICAL' if a == b else '<<< NON-DETERMINISTIC'}")
    ok &= a == b
    victim = pathlib.Path(repo) / "README.md"
    original = victim.read_bytes()
    try:
        victim.write_bytes(b"# DIRTY WORKTREE PROBE 2099-12-31\n\n" + original)
        d = hashlib.sha256(serialise(build(repo, tree, ids)).encode()
                           ).hexdigest()
    finally:
        victim.write_bytes(original)
    clean = _git(repo, "status", "--porcelain").strip() == ""
    print(f"  dirty-worktree can-fail    : "
          f"{'UNCHANGED, tree-bound' if d == a else '<<< LEAKED'}"
          f"   worktree restored clean: {clean}")
    ok &= d == a and clean
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--identities", required=True)
    ap.add_argument("--out")
    ap.add_argument("--prove", action="store_true")
    a = ap.parse_args()
    if a.prove:
        raise SystemExit(0 if _prove(a.subject_repo, a.tree, a.identities)
                         else 1)
    if not a.out:
        raise SystemExit("--out is required unless --prove")
    recs = build(a.subject_repo, a.tree, a.identities)
    text = serialise(recs)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(f"records           {len(recs)}")
    print(f"fields            {' '.join(FIELD_ORDER)}")
    print(f"content source    git object {a.tree[:12]}, never the filesystem")
    print(f"judgement fields  NONE - no classifier in the candidate is called")
    print(f"sha256(packet)    "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
