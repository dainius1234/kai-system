#!/usr/bin/env python3
"""The doctrine must be mechanically comparable, or copies of it drift.

THE FAILURE THAT EARNED THIS
============================

Two records of the same 27 rules diverged. The PM thread's continuity
copy was missing **rule 4** — *"LOOKUP → VERIFY SUBJECT → USE
IDENTIFIER: never use a remembered run id, SHA, artifact or subject
where an authoritative lookup exists"* — and reached 27 by splitting
rule 26 into two. Every rule from position 4 onward was off by one, so
"Rule 17" named different rules in each record.

The casualty was **the anti-drift rule itself**, in the record kept by
the party that fetches identifiers this environment cannot reach.

Nothing detected it. Nothing compares the two records. It surfaced only
because the list happened to be pasted back and read against the file by
a human. (D272)

Before that, the programme's binding order of work — 048 → A-4 →
Assurance/Kingsman — was found to exist nowhere in the tree at all.
(D270 §2)

Two instances is a pattern: **material that governs the work but does
not live in the work drifts silently, and nobody is at fault, because
nobody can see it.**

WHAT THIS ENFORCES
==================

Rule 28: *governing material must be checkable from the work it
governs.* The decidable half of that is:

1. the rule set is **mechanically extractable** — numbered 1..N,
   contiguous, no gaps, no duplicates;
2. every rule has **provenance** — it appears in the "where each rule
   was earned" table, because the doctrine's own standard is that every
   rule was earned by a specific failure;
3. the whole set has a **published fingerprint**, so any external copy
   can be reconciled by comparison rather than by reading.

The fingerprint is the point. A gap check catches a dropped rule inside
this file; only a fingerprint catches a dropped rule in somebody else's
copy of it — and that is the failure that actually happened.

WHAT THIS DOES NOT DO
=====================

It cannot reach the external record. Nothing here proves the PM thread's
copy is correct; it produces the value that copy must reproduce. The
reconciliation is a human act performed with a machine-checkable target,
which is strictly better than a human act performed against prose.

Exit 0 = the doctrine is internally consistent and its fingerprint is
printed.  Exit 1 = a gap, a duplicate, a rule without provenance, or an
unreadable file.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
DOCTRINE = REPO / "kai-pm" / "ENGINEERING_DOCTRINE.md"

# A rule opens with `N. **text**`, and the bold text may wrap across
# lines. Matching only single-line openers is how a first draft of this
# very check "found" 23 of 27 rules and reported four phantom gaps --
# a scope smaller than the check's name, in the check written to catch
# exactly that. Kept as a comment because it nearly shipped.
_RULE = re.compile(r"^\s*(\d+)\.\s+\*\*(.+?)\*\*", re.M | re.S)
_PROVENANCE_ROW = re.compile(r"^\|\s*([\d,\s]+)\s*\|", re.M)

# The rules live between these two headings. Scoping to them is not
# tidiness -- the first draft scanned the WHOLE file and reported rule 7
# as duplicated, because section 0's proactive-duty step 7 is also a
# bold numbered item. A population that included a non-rule, in the
# checker written to catch populations that include the wrong things.
# Third instance this week. The boundary is derived from the document's
# own structure rather than by filtering ids, because filtering ids
# would have hidden it.
_RULES_START = "## The rules"
_RULES_END = "Where each rule was earned"


def rules_section(text: str) -> str:
    """Only the region that actually contains rules."""
    if _RULES_START not in text:
        return ""
    body = text.split(_RULES_START, 1)[1]
    return body.split(_RULES_END, 1)[0]


def rules(text: str) -> dict[int, str]:
    """Every numbered rule, id -> its bold statement, whitespace-normalised."""
    out: dict[int, list[str]] = {}
    for m in _RULE.finditer(rules_section(text)):
        n = int(m.group(1))
        statement = " ".join(m.group(2).split())
        out.setdefault(n, []).append(statement)
    # A duplicate id is a finding, not something to silently overwrite.
    return {n: v[0] if len(v) == 1 else " || ".join(v) for n, v in out.items()}


def duplicates(text: str) -> list[int]:
    seen: dict[int, int] = {}
    for m in _RULE.finditer(rules_section(text)):
        n = int(m.group(1))
        seen[n] = seen.get(n, 0) + 1
    return sorted(n for n, c in seen.items() if c > 1)


def provenance(text: str) -> set[int]:
    """Rule ids appearing in the `where each rule was earned` table."""
    tail = text.split("Where each rule was earned", 1)
    if len(tail) < 2:
        return set()
    covered: set[int] = set()
    for m in _PROVENANCE_ROW.finditer(tail[1]):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                covered.add(int(part))
    return covered


def fingerprint(found: dict[int, str]) -> tuple[str, list[tuple[int, str]]]:
    """A stable identity over the ordered rule set, plus per-rule digests.

    Over (number, normalised statement) so that a reworded rule, a
    dropped rule, a renumbering and a split all move the value. Prose
    elsewhere in the file deliberately does not.
    """
    per = [(n, hashlib.sha256(f"{n}\x1f{found[n]}".encode()).hexdigest()[:12])
           for n in sorted(found)]
    whole = hashlib.sha256(
        "\x1e".join(f"{n}\x1f{found[n]}" for n in sorted(found)).encode()
    ).hexdigest()
    return whole, per


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", default=str(DOCTRINE))
    ap.add_argument("--quiet", action="store_true",
                    help="print the fingerprint only, for reconciliation")
    args = ap.parse_args()

    path = pathlib.Path(args.file)
    if not path.is_file():
        print(f"REFUSED: {path} does not exist. A doctrine that cannot be "
              f"read cannot be compared.")
        return 1
    text = path.read_text()
    found = rules(text)
    if not rules_section(text).strip():
        print(f"REFUSED: {path} has no '{_RULES_START}' section, so the "
              f"population this check traverses cannot be located. A "
              f"fingerprint over an unknown region is worthless.")
        return 1
    if not found:
        print(f"REFUSED: no numbered rules found in {path}. Either the file "
              f"changed shape or this parser did; either way the fingerprint "
              f"would be meaningless.")
        return 1

    whole, per = fingerprint(found)
    if args.quiet:
        print(whole)
        return 0

    print("ENGINEERING DOCTRINE — INTEGRITY AND FINGERPRINT")
    print("=" * 68)
    # `--file` may legitimately point outside the repository -- a fixture,
    # or an external copy being reconciled. relative_to() raises for those,
    # and crashing while REPORTING is rule 6's defect in the reporter.
    try:
        shown = path.relative_to(REPO)
    except ValueError:
        shown = path
    print(f"  file    : {shown}")
    print(f"  rules   : {len(found)}  (ids {min(found)}..{max(found)})")
    print()

    problems: list[str] = []

    expected = set(range(1, max(found) + 1))
    gaps = sorted(expected - set(found))
    if gaps:
        problems.append(f"GAP: rule id(s) {gaps} are missing. A gap is a "
                        f"dropped rule until proven otherwise")
    dupes = duplicates(text)
    if dupes:
        problems.append(f"DUPLICATE: rule id(s) {dupes} appear more than "
                        f"once. A split or a paste, either way ambiguous")

    covered = provenance(text)
    orphans = sorted(set(found) - covered)
    if orphans:
        problems.append(
            f"NO PROVENANCE: rule(s) {orphans} do not appear in the "
            f"'where each rule was earned' table. This file's own standard "
            f"is that every rule was earned by a specific failure; a rule "
            f"with no recorded failure is an opinion")

    for n in sorted(found):
        mark = " " if n in covered else "!"
        digest = dict(per)[n]
        print(f"  {mark}{n:>3}. {digest}  {found[n][:56]}")
    print()
    print(f"  DOCTRINE FINGERPRINT: {whole}")
    print()
    print("  Any external copy of this doctrine must reproduce that value.")
    print("  If it does not, the copy has drifted and the FILE is canonical:")
    print("  it is the artefact both parties can open. (D272, rule 28)")
    print()
    print(f"  inspected: {len(found)} rule(s) across {len(covered)} "
          f"provenance entry(s)")

    if problems:
        print()
        for p in problems:
            print(f"FAIL: {p}")
        return 1
    print("PASS: contiguous, unduplicated, every rule has provenance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
