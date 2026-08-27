#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — PASS A. Evidence packet. NO VERDICTS ASSIGNED.

Every source is a PARAMETER. v1.0's pass_a.py hard-coded
FULL=/home/user/kai-system and a session scratchpad path while its
RUN.md advertised generic inputs; that is what made the package
unreproducible anywhere else (D340 §5).

CENSUS DEPENDENCY — A DECISION RECORDED, NOT ASSUMED. v1.0 consumed
Census v1.0, which D341 later proved carried four defects including an
execution-proven symbolic-ref subject misbinding. v1.1 consumes the
FROZEN, QUALIFIED Census v1.1 instead, supplied by path and recorded by
aggregate. CONSEQUENCE, STATED SO IT IS NOT MISATTRIBUTED: any delta
between v1.0 and v1.1 H2 output may originate in the Census change as
well as in the classifier change. The two are not separable by
comparison alone.

DATE RECOGNITION IS REPAIRED HERE. v1.0's pattern required FULL month
names, so `Version: 1.0 - 2 Mar 2026` was missed and the document was
asserted CURRENT_TREE -- "claims current state, unbound" -- for a
date-bound specification. One wrong cell, in the direction that invites
H3 to check a March document against today's tree.
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

# ── date recognition: ISO, full month, ABBREVIATED month, both orders ─
DATE = re.compile(
    r"\b20\d\d-\d\d-\d\d\b"
    r"|\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"[a-z]*\.?,?\s+20\d\d\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+"
    r"\d{1,2},?\s+20\d\d\b",
    re.I)
SHA = re.compile(r"\b[0-9a-f]{7,40}\b")
RUN = re.compile(r"\brun[_ ]?(?:id)?[ :#]*\d{8,}\b|actions/runs/\d+", re.I)
# a successor must be a PATH-LIKE token: "superseded by the 4,580-finding
# reconciliation" once captured "the" as a named successor (D338).
SUPBY = re.compile(r"superseded by\s+`?([A-Za-z0-9_./-]+\.md)`?", re.I)
SUPES = re.compile(r"\bsupersedes\b", re.I)
PRESENT = re.compile(r"\bcurrent phase\b|\bcurrent focus\b|\bcurrently\b"
                     r"|\blast updated\b|\bnext\b:", re.I)


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=str(repo),
                          capture_output=True, text=True).stdout


def history_identity(history_repo, subject):
    """The observation window, recorded WITH the numbers it produces."""
    shallow = git(history_repo, "rev-parse",
                  "--is-shallow-repository").strip()
    first = git(history_repo, "log", "--reverse", "--format=%H %ad",
                "--date=short").splitlines()
    oldest = first[0] if first else ""
    depth = git(history_repo, "rev-list", "--count", subject).strip()
    return {
        "shallow": shallow,
        "oldest_reachable_commit": oldest.split(" ")[0] if oldest else None,
        "oldest_reachable_date": oldest.split(" ")[1] if " " in oldest else None,
        "newest_date": git(history_repo, "log", "-1", "--format=%ad",
                           "--date=short").strip(),
        "subject_ancestry_depth": int(depth or 0),
    }


def build(subject_repo, history_repo, subject, census_pkg):
    sys.path.insert(0, str(census_pkg))
    import docgraph as G, opscan as O, claims as C   # frozen Census v1.1

    tracked = G.tracked_md(subject_repo)
    edges = G.build_graph(subject_repo, tracked)
    inc = G.incoming(edges)
    out_deg = collections.Counter(s for s, d, k, _r, _c in edges if d)

    ops = O.collect(subject_repo, tracked)[0]
    C.classify(ops, tracked, set(O.tracked(subject_repo)))
    exe = collections.Counter()
    writers = collections.defaultdict(set)
    readers = collections.defaultdict(set)
    for o in ops:
        if o.target and o.disposition in ("RESOLVED_READ", "RESOLVED_WRITE",
                                          "READ_AND_WRITE"):
            exe[o.target] += 1
            (writers if o.disposition != "RESOLVED_READ"
             else readers)[o.target].add(o.src)

    rows = []
    for d in tracked:
        txt = (pathlib.Path(subject_repo) / d).read_text(errors="ignore")
        head = txt[:6000]
        title = ""
        for ln in txt.splitlines():
            if ln.startswith("#"):
                title = ln.lstrip("#").strip()[:80]
                break
        n = git(history_repo, "rev-list", "--count", subject, "--", d).strip()
        rows.append(dict(
            path=d, title=title, bytes=len(txt.encode()),
            sha256=hashlib.sha256(txt.encode()).hexdigest()[:16],
            commits_in_window=int(n or 0),
            last=git(history_repo, "log", "-1", "--format=%ad",
                     "--date=short", subject, "--", d).strip(),
            graphA_in=inc.get(d, 0), graphA_out=out_deg.get(d, 0),
            exe_ops=exe.get(d, 0),
            writers=sorted(writers.get(d, ())),
            readers=sorted(readers.get(d, ())),
            has_sha=bool(SHA.search(head)), has_run=bool(RUN.search(head)),
            has_date=bool(DATE.search(head)),
            superseded_by=(SUPBY.search(head).group(1)
                           if SUPBY.search(head) else None),
            says_supersedes=bool(SUPES.search(head)),
            present_tense=bool(PRESENT.search(head)),
        ))
    assert len(rows) == len(tracked), "PASS A population mismatch"
    return rows, tracked


def main():
    ap = argparse.ArgumentParser(description="HOUSE_H2 v1.1 Pass A")
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--history-repo", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--census-package", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sr, hr = pathlib.Path(a.subject_repo), pathlib.Path(a.history_repo)
    # R11: prove the subject before measuring anything.
    head = git(sr, "rev-parse", "HEAD").strip()
    if head != a.subject:
        raise SystemExit(f"R11 ABORT: subject repo HEAD {head[:12]} != "
                         f"subject {a.subject[:12]}")
    if subprocess.run(["git", "cat-file", "-e", f"{a.subject}^{{commit}}"],
                      cwd=str(hr)).returncode != 0:
        raise SystemExit("R11 ABORT: subject commit absent from history source")

    rows, tracked = build(sr, hr, a.subject, pathlib.Path(a.census_package))
    hist = history_identity(hr, a.subject)
    census_manifest = (pathlib.Path(a.census_package) / "MANIFEST.sha256")
    payload = {
        "subject": a.subject,
        "subject_tree": git(sr, "rev-parse", f"{a.subject}^{{tree}}").strip(),
        "history_identity": hist,
        "census_dependency": {
            "package": str(a.census_package),
            "aggregate": hashlib.sha256(
                census_manifest.read_bytes()).hexdigest(),
        },
        "population": len(tracked),
        "rows": rows,
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))
    print(f"PASS A COMPLETE — {len(rows)} evidence packets == population "
          f"{len(tracked)}")
    print(f"  history window : {hist['oldest_reachable_date']} → "
          f"{hist['newest_date']}  shallow={hist['shallow']}  "
          f"ancestry={hist['subject_ancestry_depth']}")
    print(f"  commits>1 in window : "
          f"{sum(1 for r in rows if r['commits_in_window'] > 1)}")
    print(f"  proven readers      : {sum(1 for r in rows if r['readers'])}")
    print(f"  present-tense claim : "
          f"{sum(1 for r in rows if r['present_tense'])}")
    print(f"  dated               : {sum(1 for r in rows if r['has_date'])}")
    print("  NO VERDICT ASSIGNED IN PASS A.")


if __name__ == "__main__":
    main()
