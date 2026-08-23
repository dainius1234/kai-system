#!/usr/bin/env python3
"""CALIBRATION TRACE HARNESS — the evidence channel for legs 2 and 3.

Census v1.0 could not distinguish "a fixture reached this value" from
"this value is spelled somewhere in a calibration file" (D341, method
correction). Grepping a suite proves nothing: a value can be named in an
assertion that never fires.

So the two legs are recorded SEPARATELY and only ever at runtime:

  OBSERVED  — leg 2. The value was actually emitted while a fixture ran.
  ASSERTED  — leg 3. A calibration assertion was ABOUT that value and
              PASSED, i.e. calibration discriminates it.

I-8: `assert_value` records only when the assertion passes. A failing
assertion proves nothing about discrimination, so it must not count
towards it.
"""
from __future__ import annotations
import collections

OBSERVED: collections.Counter = collections.Counter()
ASSERTED: collections.Counter = collections.Counter()
PASSED = 0
FAILED = 0
FAILURES: list = []


def reset():
    global PASSED, FAILED
    OBSERVED.clear()
    ASSERTED.clear()
    PASSED = FAILED = 0
    FAILURES.clear()


def observe(alphabet: str, value):
    """Leg 2 — a value was emitted by the instrument during a fixture."""
    if value is not None:
        OBSERVED[(alphabet, value)] += 1


def check(name: str, cond: bool, detail: str = ""):
    """A calibration assertion not tied to a single alphabet value."""
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name} :: {detail}")
    return bool(cond)


def assert_value(name: str, cond: bool, alphabet: str, value,
                 detail: str = ""):
    """Leg 3 — an assertion whose SUBJECT is `value`.

    Credits discrimination only on success.
    """
    ok = check(name, cond, detail)
    if ok:
        ASSERTED[(alphabet, value)] += 1
    return ok


def summary():
    return {"passed": PASSED, "failed": FAILED,
            "failures": list(FAILURES),
            "observed": {f"{a}::{v}": n for (a, v), n in OBSERVED.items()},
            "asserted": {f"{a}::{v}": n for (a, v), n in ASSERTED.items()}}
