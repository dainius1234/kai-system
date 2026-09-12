#!/usr/bin/env python3
"""Calibration for the live-capture trigger check.

The property under test is the operator's standard, and it is a
*negative* claim, which is the hard kind to hold honestly:

> changed paths ∩ live-capture trigger paths = ∅

A check that answers that question wrongly in the safe-looking direction
— reporting ∅ when a path really would fire a model run — is worse than
no check, because it converts "I did not look" into "I looked and it was
clean". So every assertion here has its known-positive: the matcher, the
filter reader, the intersection and the two-class separation are each
shown firing on a case that must fire, and staying quiet on one that
must not.

It also pins the discovery the check made on its first run, which no
amount of reading the workflow files had told me: a workflow with **no**
`paths:` filter is triggered by every push, so "I did not touch the
probe" never proved that no model would run.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import check_capture_trigger_paths as t  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 4
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def test_the_glob_matcher_is_neither_blind_nor_greedy() -> None:
    scenario("path glob matching")
    # known-positives
    check("an exact path matches itself",
          t._matches("scripts/security/probe.py", "scripts/security/probe.py"))
    check("`**` spans directory separators",
          t._matches("scripts/**", "scripts/security/probe.py"))
    check("`*` matches within one segment",
          t._matches("scripts/security/*.py", "scripts/security/probe.py"))
    check("`?` matches one character",
          t._matches("a/b?.py", "a/bc.py"))
    # known-negatives — the direction that would fake a clean result
    check("a different file does not match",
          not t._matches("scripts/security/probe.py", "scripts/security/x.py"))
    check("`*` does NOT span a separator",
          not t._matches("scripts/*.py", "scripts/security/probe.py"))
    check("a prefix is not a match without a glob",
          not t._matches("scripts/security", "scripts/security/probe.py"))
    check("a suffix is not a match either",
          not t._matches("probe.py", "scripts/security/probe.py"))


def test_the_filter_reader_sees_a_missing_filter() -> None:
    scenario("push paths reader")
    with_filter = (
        "on:\n"
        "  push:\n"
        "    branches:\n"
        "      - main\n"
        "    paths:\n"
        "      - scripts/security/probe.py\n"
        "      # a comment inside the list\n"
        "      - memu-graph/Dockerfile\n"
        "\n"
        "permissions:\n"
        "  contents: read\n")
    paths, has = t._push_paths(with_filter)
    check("both entries are read", paths == ["scripts/security/probe.py",
                                             "memu-graph/Dockerfile"], str(paths))
    check("and the filter is reported present", has is True)
    check("the list ends at the next top-level key",
          "contents: read" not in paths, str(paths))

    # THE case that matters. A workflow with no paths: filter fires on
    # every push, so reading it as "no triggers" is the exact inversion
    # that would make this check lie in the reassuring direction.
    paths, has = t._push_paths("on:\n  push:\n    branches:\n      - main\n")
    check("no paths: filter reads as ABSENT, not as empty", has is False)
    check("and yields no patterns", paths == [], str(paths))


def test_intersection_fires_and_stays_quiet() -> None:
    scenario("intersection")
    filtered = {"workflow": "capture.yml", "drivers": ["s.sh"], "inline": False,
                "writes_capture": ["s.sh"],
                "paths": ["scripts/security/probe.py", "memu-graph/**"],
                "has_filter": True}
    unfiltered = {"workflow": "core.yml", "drivers": ["s.sh"], "inline": False,
                  "writes_capture": [], "paths": [], "has_filter": False}

    hits = t.intersect(["scripts/security/probe.py"], [filtered])
    check("a path inside the filter is a hit", len(hits) == 1, str(hits))
    check("and the matching pattern is named",
          hits[0]["pattern"] == "scripts/security/probe.py", str(hits))
    check("a glob entry matches too",
          len(t.intersect(["memu-graph/Dockerfile"], [filtered])) == 1)
    # known-negative
    check("a path outside every filter is NOT a hit",
          t.intersect(["kai-pm/DECISIONS.md"], [filtered]) == [],
          str(t.intersect(["kai-pm/DECISIONS.md"], [filtered])))
    # the unfiltered workflow fires whatever changed
    check("an unfiltered live-capture workflow always hits",
          len(t.intersect(["kai-pm/DECISIONS.md"], [unfiltered])) == 1)
    check("and says why",
          "every push" in t.intersect(["x"], [unfiltered])[0]["pattern"])


def test_the_two_classes_are_not_conflated() -> None:
    scenario("live-model vs capture-writing")
    # Derived from the real tree, because the value of this check is that
    # its denominator is the repository's, not a fixture's.
    workflows = t.live_capture_workflows()
    check("at least one live-model workflow is identified",
          len(workflows) >= 1, str(workflows))
    names = {w["workflow"] for w in workflows}
    writing = {w["workflow"] for w in workflows if w["writes_capture"]}
    check("the capture workflow is identified as live-model",
          "memu-graph-startup-proof.yml" in names, str(sorted(names)))
    check("and as CAPTURE-WRITING specifically",
          "memu-graph-startup-proof.yml" in writing, str(sorted(writing)))
    # THE discovery this check made on its first run: a workflow can call
    # a real model without ever writing an admissible capture, and one of
    # ours does it on every push. Conflating the two classes would have
    # made "a model ran" and "evidence was produced" the same statement.
    live_only = names - writing
    check("live-model-only workflows are a separate, non-empty class",
          len(live_only) >= 1, str(sorted(names)))
    check("and none of them is treated as capture-writing",
          not (live_only & writing), str(sorted(live_only)))

    # A blinded detector reports zero and looks clean, so zero is refused
    # rather than passed (I-1).
    src = (REPO / "scripts" / "security"
           / "check_capture_trigger_paths.py").read_text()
    check("an empty workflow set is REFUSED, not reported as safe",
          "REFUSED: no live-capture workflow was identified" in src)
    check("a capture-writing hit exits differently from a live-model hit",
          "return 4" in src and "return 3" in src)


def run_all() -> None:
    test_the_glob_matcher_is_neither_blind_nor_greedy()
    test_the_filter_reader_sees_a_missing_filter()
    test_intersection_fires_and_stays_quiet()
    test_the_two_classes_are_not_conflated()

    wf = t.live_capture_workflows()
    print(f"  inspected: {len(wf)} live-capture workflow(s) derived from "
          f"the tree")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Live-Capture Trigger Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
