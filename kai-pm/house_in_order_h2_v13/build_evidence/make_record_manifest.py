#!/usr/bin/env python3
"""Canonical 492-record source-occurrence manifest for STEP 2 / M3.

Kai step-2 calibration ruling: the frozen totals are not sufficient for
per-record conservation, so the individual record identities are banked
BEFORE any M3 code change. Kai then derives the expected scope set from
this manifest independently and publishes only its hash; the builder
never sees the answer key.

THIS FILE IS BUILD EVIDENCE, NOT A STAGE A MEMBER. Its bytes are derived
from executing the candidate, so under I1-A it belongs to Stage B.

Emits: path, start, end, detector, source_selector,
current_applicability_scope. `detector` is the RECOGNISER, not the
witness_type: HEX and DECIMAL_RUN can match the same span (F9-A), so the
recogniser is load-bearing in the identity.

Canonical form: deterministic sort, UTF-8, TAB delimited, backslash
escaping, no timestamps, no machine-specific paths, no insertion-order
dependence.
"""
from __future__ import annotations
import argparse
import hashlib
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import passa as P                                              # noqa: E402

FIELDS = ("path", "start", "end", "detector", "source_selector",
          "current_applicability_scope")


def esc(v):
    return (str(v).replace("\\", "\\\\").replace("\t", "\\t")
            .replace("\n", "\\n").replace("\r", "\\r"))


def occurrences(repo, tree):
    """Every emitted source occurrence, mirroring scan()'s own gating."""
    names = subprocess.run(
        ["git", "-C", repo, "ls-tree", "-r", "--name-only", tree],
        capture_output=True, text=True, check=True).stdout.splitlines()
    rows = []
    for d in sorted(n for n in names if n.endswith(".md")):
        text = (pathlib.Path(repo) / d).read_text(errors="ignore")
        head = text[:P.HEAD_BYTES]

        def add(det, m, start=None, end=None):
            s = m.start() if start is None else start
            e = m.end() if end is None else end
            rows.append({"path": d, "start": s, "end": e, "detector": det,
                         "source_selector": P._selector(text, s),
                         "current_applicability_scope": P._scope_of(head, s)})

        for det, rx in (("HEX", P.HEX), ("DATE", P.DATE)):
            for m in rx.finditer(text):
                if P._eligible(m):
                    add(det, m)
        for m in P.DECIMAL_RUN.finditer(text):
            if not P._eligible(m):
                continue
            before = text[max(0, m.start() - 24):m.start()]
            window = text[max(0, m.start() - 24):m.end()]
            if P.RUN_NEAR.search(before) or P.RUN_URL.search(window):
                add("DECIMAL_RUN", m)
        sup = next((x for x in P.SUPBY.finditer(text) if P._eligible(x)), None)
        if sup:
            rows.append({"path": d, "start": sup.start(), "end": sup.end(),
                         "detector": "SUPBY",
                         "source_selector": P._selector(text, sup.start()),
                         # supersession is file-level by construction
                         "current_applicability_scope": "WHOLE_FILE"})
    return sorted(rows, key=lambda r: (r["path"], r["start"], r["end"],
                                       r["detector"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = occurrences(a.subject_repo, a.tree)
    keys = [(r["path"], r["start"], r["end"], r["detector"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"R11 ABORT: path+start+end+detector is NOT unique "
                         f"({len(keys)} rows, {len(set(keys))} keys). An "
                         f"explicit ordinal is required; do not add one "
                         f"silently.")
    body = "\n".join("\t".join(esc(r[f]) for f in FIELDS) for r in rows)
    text = "\t".join(FIELDS) + "\n" + body + "\n"
    pathlib.Path(a.out).write_text(text, encoding="utf-8")

    wf = sum(1 for r in rows
             if r["current_applicability_scope"] == "WHOLE_FILE")
    print(f"rows              {len(rows)}")
    print(f"WHOLE_FILE        {wf}")
    print(f"SPAN              {len(rows) - wf}")
    print(f"key unique        yes (path+start+end+detector), no ordinal added")
    print(f"sha256(manifest)  "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
