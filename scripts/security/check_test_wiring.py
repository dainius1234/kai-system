#!/usr/bin/env python3
"""A test that is never called is not a test.

Written after the operator's review of how much of this programme went
into fixing tests that gave false readings. The immediate finding was
**7 tests in the dashboard tracker, defined and never called — 16
assertions running nowhere.** All seven pass. Nothing failed, so nothing
drew attention to them.

**How this check was built matters more than what it found.** Four
attempts, three of them wrong, each wrong in the same way:

1. *"Any `test_` not called anywhere is dead."* → **1,555 hits.** Wrong:
   pytest and unittest *collect* tests; nothing calls them by name.
2. *"Exclude files containing `unittest.main()`."* → **1,813 hits.**
   Wrong: files run by `python -m pytest <file>` contain no such marker.
3. *"Look inside the file's `run()` function."* → **54 hits.** Wrong: it
   matched a *helper* named `run(...)` instead of `run_all()`.
4. Ask the Makefile **how each script is actually invoked**. → **10.**

The first three read a proxy — file contents — for the thing that
actually decides the answer, which is the command that runs the script.
`python -m pytest x.py` collects every test in it; `python x.py` runs
only what the file itself calls. That distinction is knowable exactly,
and guessing at it from the inside produced three confident wrong
numbers in a row.

So this file does two things that the first three attempts did not:

  - It reads the **invocation**, not the file.
  - It **calibrates against known-good suites before reporting.** If the
    detector disagrees with a suite whose behaviour is known, it says so
    and refuses, rather than reporting a number nobody can trust.

Exit 0 = every test in a self-run suite is invoked.  Exit 1 = one is not,
or the calibration failed.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected, require  # noqa: E402

# Suites whose dispatch is known-good. If the detector claims one of
# these has orphans, the detector is wrong and says so instead of
# reporting. Three earlier versions would have been stopped here.
CALIBRATION = (
    "test_compose_drift.py",
    "test_gate_registry.py",
    "test_secret_gates.py",
    "test_ci_tolerations.py",
    "test_assertion_floors.py",
)

_INVOCATION = re.compile(
    r"(python[0-9.]*\s+(?:-m\s+pytest\s+)?)(scripts/test_[a-z0-9_]+\.py)")


def invocations() -> Tuple[Set[str], Set[str]]:
    """(collected_by_pytest, run_as_plain_script) from the Makefile."""
    makefile = (REPO / "Makefile").read_text(encoding="utf-8")
    collected: Set[str] = set()
    self_run: Set[str] = set()
    for match in _INVOCATION.finditer(makefile):
        name = Path(match.group(2)).name
        (collected if "pytest" in match.group(1) else self_run).add(name)
    return collected, self_run


def orphans(path: Path) -> List[str]:
    """Tests defined in a self-run script that nothing in it calls."""
    text = path.read_text(encoding="utf-8")
    if "unittest.main()" in text:
        return []                       # unittest collects them
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return sorted(defined - called)


def calibrate() -> List[str]:
    """Disagreements with suites whose dispatch is known correct."""
    wrong = []
    for name in CALIBRATION:
        path = REPO / "scripts" / name
        if not path.exists():
            wrong.append(f"{name}: calibration suite is missing")
            continue
        found = orphans(path)
        if found:
            wrong.append(f"{name}: detector claims {len(found)} orphan(s) "
                         f"in a suite known to dispatch all of its tests")
    return wrong


def main() -> int:
    require(("Makefile", "scripts"))
    miscalibrated = calibrate()
    if miscalibrated:
        print("REFUSED: the detector disagrees with known-good suites.\n")
        for line in miscalibrated:
            print(f"  - {line}")
        print("\n  A detector that cannot reproduce a known answer cannot "
              "be trusted\n  with an unknown one. Three earlier versions of "
              "this check would\n  have been stopped here instead of "
              "reporting 1,555, 1,813 and 54.")
        return 1

    collected, self_run = invocations()
    dead: Dict[str, List[str]] = {}
    phantom: List[str] = []
    for name in sorted(self_run):
        path = REPO / "scripts" / name
        if not path.exists():
            # The Makefile names a script that is not there. That target
            # is broken, and skipping it would let this check certify a
            # suite it never opened — the same defect it exists to find,
            # in the file that finds it. I-1 caught this on the first run.
            phantom.append(name)
            continue
        found = orphans(path)
        if found:
            dead[name] = found

    print(inspected(len(self_run), "self-run suites",
                    f"{len(collected)} more are collected by pytest"))
    print(f"  calibrated against {len(CALIBRATION)} known-good suites")

    if phantom:
        print(f"\nFAIL: {len(phantom)} Makefile target(s) name a script "
              f"that does not exist:\n")
        for name in phantom:
            print(f"  - scripts/{name}")
        print("\n  That target cannot run, and this check cannot inspect "
              "what is not there.")
        return 1

    if dead:
        total = sum(len(v) for v in dead.values())
        print(f"\nFAIL: {total} test(s) defined and never called:\n")
        for name, found in dead.items():
            print(f"  {name}")
            for entry in found:
                print(f"    - {entry}")
        print("\n  A test that is never called asserts nothing, and nothing "
              "fails to\n  draw attention to it.")
        return 1

    print("\nPASS: every test in a self-run suite is dispatched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
