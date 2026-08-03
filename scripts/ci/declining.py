"""Declining to verify, out loud.

`pytest.skip()` is two different lies depending on how a script is run.
Under pytest it is **green** — a test that verified nothing counts as one
that passed. Run as a plain script it raises `Skipped`, the process exits
non-zero, and the output looks like a crash.

Four scripts in this repository used it, and I filed the resulting
failures under "pre-existing, needs a running stack" without looking.
Three of the six failing suites needed no stack at all.

The honest third option is the one this programme already applies to
services and to CI steps: **say that you did not verify, name what was
missing, and make it impossible to mistake for a pass.** The same shape as
`common/degraded.unavailable_metric` and the CI `# ci-toleration:`
markers — a result that reports its own absence.
"""
from __future__ import annotations

import sys
from typing import List

_DECLINED: List[str] = []


def declined(what: str, because: str) -> None:
    """Record that a check could not run, and why."""
    _DECLINED.append(f"{what}: {because}")
    print(f"  DID NOT VERIFY — {what}\n      because: {because}")


def report(suite: str, passed: int, failed: int) -> int:
    """Print the tally and return the exit code.

    A declined check is not a failure — the precondition genuinely was
    not there. It is also not a pass, and the count is printed separately
    so it can never be read as one.
    """
    print(f"\n{'=' * 60}")
    line = f"{suite}: {passed} passed, {failed} failed"
    if _DECLINED:
        line += f", {len(_DECLINED)} not verified"
    print(line)
    if _DECLINED:
        print("\n  Not verified (preconditions absent):")
        for entry in _DECLINED:
            print(f"    - {entry}")
        print("  These assert nothing. They are counted apart from passes"
              "\n  so the total cannot be read as coverage.")
    if failed:
        print("EXIT GATE: FAIL")
        return 1
    print("EXIT GATE: PASS")
    return 0


def reset() -> None:
    _DECLINED.clear()
