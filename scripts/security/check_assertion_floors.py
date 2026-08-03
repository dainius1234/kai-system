"""Assertion-count ratchet — a suite must never exercise less than before.

Five times in this programme a check has quietly stopped checking, and
every one was found by luck rather than by any mechanism:

  1. A negative test whose injected violation was never written to disk.
     It passed while asserting nothing.
  2. An architecture gate that reported a clean pass while silently
     omitting 6 of its 15 rules.
  3. `MANUAL` placeholders in the dashboard tracker reading, at a glance,
     as a clean bill of health.
  4. A UI-auth suite where an unresolved promise emptied node's event
     loop: the process exited 0 with **no output**, which on CI is
     indistinguishable from success.
  5. Ratchet tests that decremented a named column to prove the gate
     fires — and so stopped testing each column as that column reached
     zero. Coverage fell exactly as the system got healthier.

The operator named the pattern: a **self-consuming guard** — a
precondition that shrinks in scope because of the success of the
operation it guards, until the test silently tests nothing. Every guard
whose condition depends on state the test itself modifies is suspect.

They share one signature: **the number of things actually exercised fell,
and nothing complained.**

Two detectors, because the signature shows up in two ways:

  - **Floors** catch erosion *over time*: a suite exercising less than it
    did before.
  - **Determinism** catches erosion *at a point in time*: a suite whose
    count depends on context. That is how case 5 was actually spotted —
    not by a falling number, but by `test_hygiene_gate` reporting 34
    standalone and 31 under `make test-uh`. A count that varies with what
    else has run is a count that depends on state the test does not
    control, which is the definition of the pattern.

So: record a floor per suite and fail when it is not met. Floors move in
one direction — up. A suite that legitimately loses a test needs the
floor lowered deliberately, in a commit that says why, which is exactly
the conversation that should happen.

This is the same ratchet shape as `hygiene_survey.py`, inverted. There
the debt may only fall; here the coverage may only rise.

**This gate must not become case 6.** Three of its own rules exist only
to stop it eroding the way the things it watches did:

  - A suite in the floors that produces *no* count fails. Vanishing is
    the most complete way to stop checking, and it looks like silence.
  - A suite in the output that is *not* floored fails. An unfloored suite
    is unwatched, and can shrink to nothing unobserved.
  - A determinism sample naming a suite that no longer reports fails.
    Skipping it — the obvious `if label not in aggregate: continue` —
    would be this file quietly sampling fewer suites each time one was
    renamed. That is precisely case 5, rewritten in this file.

Exit codes:
  0  every suite met its floor
  1  a suite fell below its floor, produced no count, is unrecorded,
     or reports a different count depending on context
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
FLOORS = Path(__file__).resolve().parent / "assertion_floors.json"

# "Dashboard Auth Tests: 99 passed, 0 failed"
_SUITE_LINE = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9 &/,§.\-]*?):\s+"
                         r"(?P<passed>\d+)\s+passed,\s+(?P<failed>\d+)\s+failed\s*$")


def parse_counts(output: str) -> Dict[str, int]:
    """Extract per-suite assertion counts from a test run's output."""
    counts: Dict[str, int] = {}
    for line in output.splitlines():
        match = _SUITE_LINE.match(line.strip())
        if match:
            name = match.group("name").strip()
            counts[name] = int(match.group("passed"))
    return counts


def load_floors() -> Optional[Dict[str, int]]:
    try:
        return json.loads(FLOORS.read_text(encoding="utf-8"))["floors"]
    except (OSError, KeyError, ValueError):
        return None


def compare(counts: Dict[str, int],
            floors: Dict[str, int]) -> Tuple[List[str], List[str], List[str]]:
    """Return (fallen, missing, unrecorded).

    ``missing`` is the silent-exit case: a suite the floors expect that
    produced no count at all. That must fail — a suite which vanishes
    from the output is the most complete way for a check to stop
    checking, and it looks like nothing at all went wrong.

    ``unrecorded`` fails too. A suite nobody floored is a suite nobody
    watches, and the whole point here is that unwatched coverage drifts
    down. Recording it is one command.
    """
    fallen = [
        f"{name}: {floors[name]} → {counts[name]} ({counts[name] - floors[name]})"
        for name in sorted(floors)
        if name in counts and counts[name] < floors[name]
    ]
    missing = [name for name in sorted(floors) if name not in counts]
    unrecorded = [name for name in sorted(counts) if name not in floors]
    return fallen, missing, unrecorded


SUITE_TARGETS = {
    # Suite label as printed  ->  make target that runs it alone.
    "Hygiene Gate Tests": "test-hygiene-gate",
    "Dashboard Finding Tracker Tests": "test-dashboard-findings",
    "Dashboard Auth Tests": "test-dashboard-auth",
    "Dashboard UI Auth Tests": "test-dashboard-ui-auth",
    "Degraded State Tests": "test-degraded",
    "Architecture Rule Tests": "test-architecture-rules",
    "Deployment Preflight Tests": "test-preflight",
    "Assertion Floor Tests": "test-assertion-floors",
}


def check_determinism(aggregate: Dict[str, int]) -> List[str]:
    """Suites whose count changes when run alone.

    A count that depends on what else has run is a count that depends on
    state the suite does not control — a self-consuming guard in the act.

    A sampled label absent from the aggregate is an error, not a skip.
    Skipping would let this sample shrink silently as suites are renamed,
    which is the exact defect this file exists to catch.
    """
    drifted = []
    for label, target in sorted(SUITE_TARGETS.items()):
        if label not in aggregate:
            drifted.append(
                f"{label}: sampled by SUITE_TARGETS but absent from the "
                f"aggregate run — renamed, or no longer running")
            continue
        result = subprocess.run(["make", target], capture_output=True,
                                text=True, cwd=str(REPO))
        alone = parse_counts(result.stdout + result.stderr).get(label)
        if alone is None:
            drifted.append(f"{label}: produced no count when run alone")
        elif alone != aggregate[label]:
            drifted.append(
                f"{label}: {alone} alone vs {aggregate[label]} in aggregate")
    return drifted


def run_suites() -> Tuple[str, int]:
    """Run the aggregate suite and capture its output and exit status."""
    result = subprocess.run(
        ["make", "test-uh"], capture_output=True, text=True, cwd=str(REPO),
    )
    return result.stdout + result.stderr, result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-log", type=Path,
                        help="parse a captured test run instead of running one")
    parser.add_argument("--update-floors", action="store_true",
                        help="raise floors to the current counts (never lowers)")
    parser.add_argument("--determinism", action="store_true",
                        help="also verify each suite's count is context-free")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.from_log:
        output, status = args.from_log.read_text(encoding="utf-8"), 0
    else:
        output, status = run_suites()

    counts = parse_counts(output)

    if status != 0 and not args.update_floors:
        # Say so plainly rather than letting it arrive disguised as
        # "suites vanished" — a red test run and an eroded test surface
        # are different problems with different fixes.
        print("The aggregate test run failed "
              f"(make test-uh exited {status}).")
        print("Fix the failing suite first; assertion floors cannot be")
        print("judged against a run that did not complete.")
        print(f"\n{len(counts)} suite(s) reported before it stopped.")
        return 1

    if not counts:
        print("No suite counts found in the output.")
        print("A run that reports nothing is not a run that passed.")
        return 1

    floors = load_floors()

    if args.update_floors:
        previous = floors or {}
        lowered = {n: (previous[n], counts[n]) for n in counts
                   if n in previous and counts[n] < previous[n]}
        if lowered:
            print("REFUSED: floors may only rise. These would fall:")
            for name, (was, now) in sorted(lowered.items()):
                print(f"  - {name}: {was} → {now}")
            print("\nIf a suite genuinely lost coverage, lower its floor in a "
                  "separate commit\nthat says why. That is the conversation "
                  "this gate exists to force.")
            return 1
        merged = {**previous, **counts}
        FLOORS.write_text(
            json.dumps({
                "note": "Minimum assertions per suite. Raise only; see "
                        "scripts/security/check_assertion_floors.py.",
                "floors": dict(sorted(merged.items())),
                "total": sum(merged.values()),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Floors updated: {len(merged)} suites, {sum(merged.values())} assertions")
        return 0

    if floors is None:
        print("No floors recorded. Run with --update-floors.")
        return 1

    fallen, missing, unrecorded = compare(counts, floors)
    drifted = check_determinism(counts) if args.determinism else []

    if args.json:
        print(json.dumps({"counts": counts, "floors": floors,
                          "fallen": fallen, "missing": missing,
                          "unrecorded": unrecorded, "drifted": drifted},
                         indent=2))
        return 1 if fallen or missing or unrecorded or drifted else 0

    print(f"Assertion floors — {len(counts)} suites, "
          f"{sum(counts.values())} assertions\n")

    if fallen:
        print("  COVERAGE FELL — these suites exercise less than before:")
        for line in fallen:
            print(f"    - {line}")
        print("\n  A suite that shrinks is the signature of a check that has")
        print("  stopped checking. Find what stopped being exercised before")
        print("  lowering the floor.")

    if missing:
        print("\n  SUITES VANISHED — expected but produced no count:")
        for name in missing:
            print(f"    - {name}")
        print("\n  A suite that reports nothing is not a suite that passed.")

    if unrecorded:
        print("\n  UNFLOORED SUITES — running but unwatched:")
        for name in unrecorded:
            print(f"    - {name}")
        print("\n  Record them with `make assertion-floors-update`. Until then")
        print("  they can shrink to nothing without this gate noticing.")

    if drifted:
        print("\n  COUNT DEPENDS ON CONTEXT — a self-consuming guard in the act:")
        for line in drifted:
            print(f"    - {line}")
        print("\n  A suite whose count changes with what else has run is")
        print("  guarded on state it does not control. Drive it from")
        print("  synthetic inputs instead.")

    if fallen or missing or unrecorded or drifted:
        return 1

    print("  PASS: every suite met its floor.")
    if args.determinism:
        print("  PASS: every sampled suite reports the same count in isolation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
