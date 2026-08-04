"""The repo-wide test result is a ratchet: failures may fall, never rise.

The number this guards was **zero passing tests** on 2026-08-04, and had
been for at least a week. Not zero failures — zero *tests*. The run
aborted during collection, `python-app.yml` triggered on `main` only, and
a flake8 step failed ahead of the test step anyway. Three separate reasons
why "CI is red" carried no information about the suite.

It is now 4,208 passed, 0 failed, 0 errors. That number is worth more than
the fixes that produced it, because a number nothing defends drifts back.
This file defends it.

Three rules, and the third is the one that matters:

  - `failed` may not exceed the recorded maximum.
  - `errors` may not exceed the recorded maximum.
  - `passed` may not fall below the recorded minimum.

Without the third, deleting a test would be a way to pass this gate. The
assertion floors (`check_assertion_floors.py`) make the same argument for
the Unified Hunter suites; this is the same ratchet pointed at the
repo-wide run, which is a different denominator and so falsifiable in a
different direction.

Reads a captured run rather than producing one. The suite takes minutes,
CI already runs it, and running it twice to check it once would be its own
kind of waste — the same reason `check_assertion_floors.py` takes
`--from-log`.

Exit 0 = the suite is no worse than recorded.
Exit 1 = it regressed, the log is unreadable, or the log shows no tests
         ran at all, which is the failure this exists to make impossible
         to miss.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

REPO = Path(__file__).resolve().parent.parent.parent
FLOOR = Path(__file__).resolve().parent / "suite_floor.json"

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

# "4208 passed, 6 skipped, 322 warnings in 152.32s" — any order, any subset.
_COUNT = re.compile(r"(\d+) (passed|failed|error|errors|skipped)")


def parse(output: str) -> Optional[Dict[str, int]]:
    """Counts from a pytest run's summary line.

    Returns None when no summary is present at all. That is not zero
    failures — it is a run that did not report, which is exactly what the
    aborted collection looked like, and it must not read as a pass.
    """
    tallies: Dict[str, int] = {}
    for line in output.splitlines():
        found = _COUNT.findall(line)
        if found and ("passed" in line or "failed" in line or "error" in line):
            for number, word in found:
                key = "errors" if word.startswith("error") else word
                tallies[key] = int(number)
    return tallies or None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-log", type=Path, required=True,
                        help="a captured repo-wide pytest run")
    parser.add_argument("--record", action="store_true",
                        help="ratchet the floor to this run (refuses to loosen)")
    args = parser.parse_args()

    require(("scripts/security/suite_floor.json",))

    if not args.from_log.exists():
        print(f"REFUSED: no log at {args.from_log}.")
        print("  A missing log is not a passing suite.")
        return 1

    counts = parse(args.from_log.read_text(encoding="utf-8", errors="replace"))
    if counts is None:
        print("REFUSED: the log contains no pytest summary.")
        print("  A run that reports nothing is not a run that passed — that is")
        print("  precisely how the collection abort looked for a week.")
        return 1

    floor = json.loads(FLOOR.read_text(encoding="utf-8"))
    passed = counts.get("passed", 0)
    failed = counts.get("failed", 0)
    errors = counts.get("errors", 0)

    print(inspected(passed, "tests passed",
                    f"{failed} failed, {errors} error(s)"))

    if args.record:
        loosened = [
            name for name, (now, was) in {
                "max_failed": (failed, floor["max_failed"]),
                "max_errors": (errors, floor["max_errors"]),
            }.items() if now > was
        ]
        if passed < floor["min_passed"]:
            loosened.append("min_passed")
        if loosened:
            print(f"\nREFUSED: recording would loosen {', '.join(loosened)}.")
            print("  The floor moves one way. If the suite genuinely lost tests,")
            print("  lower it in a separate commit that says why.")
            return 1
        floor.update(max_failed=failed, max_errors=errors, min_passed=passed)
        FLOOR.write_text(json.dumps(floor, indent=2) + "\n", encoding="utf-8")
        print(f"\nFloor recorded: {passed} passed, {failed} failed, {errors} errors.")
        return 0

    problems = []
    if failed > floor["max_failed"]:
        problems.append(f"failures {floor['max_failed']} -> {failed}")
    if errors > floor["max_errors"]:
        problems.append(f"errors {floor['max_errors']} -> {errors}")
    if passed < floor["min_passed"]:
        problems.append(f"passing tests {floor['min_passed']} -> {passed}")

    if problems:
        print("\nFAIL: the repo-wide suite regressed:\n")
        for line in problems:
            print(f"  - {line}")
        print("\n  A falling pass count matters as much as a rising failure")
        print("  count: deleting a test would otherwise be a way to pass this")
        print("  gate. If a test was legitimately removed, lower the floor in a")
        print("  commit that says why.")
        return 1

    print("\nPASS: no more failures, no fewer passes than recorded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
