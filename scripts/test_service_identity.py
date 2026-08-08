#!/usr/bin/env python3
"""The identity audit must classify every endpoint and assume nothing safe."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.report_service_identity import (  # noqa: E402
    _CLASS, _protected_ops, audit)

PASSED = FAILED = 0


def check(label, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def main() -> int:
    rows, n, counts = audit(REPO)
    check("CALIBRATION: 32 protected endpoints found", n == 32)
    check("CALIBRATION: across 8 services", counts.get("_services") == 8)
    check("CALIBRATION: 6 are A", counts.get("A") == 6)
    check("CALIBRATION: 26 are B", counts.get("B") == 26)
    check("every endpoint carries a verdict",
          counts.get("A", 0) + counts.get("B", 0)
          + counts.get("UNCLASSIFIED", 0) == n)

    # ── the false positive this instrument produced on its FIRST run ──
    #
    # The regex version reported a ninth service, `common/db_restore`,
    # from the usage example inside `common/service_auth.py`'s docstring.
    # A scope larger than reality reports failure over things that are
    # right — and it did so here, on the instrument built to measure it.
    check("a docstring example is NOT counted as an endpoint",
          _protected_ops('"""x: require_service_auth(\'db_restore\')"""')
          == [])
    check("a real call IS counted",
          _protected_ops('require_service_auth("tool_execute")')
          == [("tool_execute", 1)])
    check("and the phantom service is gone from the live tree",
          not any("db_restore" in r for r in rows))

    # The load-bearing property: an operation nobody has judged must NOT
    # default to A. Absence of a judgement is not evidence of safety.
    unjudged = "an_operation_nobody_has_judged"
    check("an unjudged operation is UNCLASSIFIED, never A",
          _CLASS.get(unjudged) is None)
    check("UNCLASSIFIED is the label an unjudged op receives",
          _CLASS.get(unjudged, ("UNCLASSIFIED", ""))[0] == "UNCLASSIFIED")

    # I-3: prove the UNCLASSIFIED branch can still fire, since nothing on
    # the live tree exercises it now that the phantom is gone.
    import scripts.security.report_service_identity as mod
    saved = dict(mod._CLASS)
    try:
        mod._CLASS.pop("tool_execute")
        rows3, n3, counts3 = audit(REPO)
        check("I-3: un-declaring an operation makes it UNCLASSIFIED",
              counts3.get("UNCLASSIFIED") == 1
              and any("tool_execute" in r and "UNCLASSIFIED" in r
                      for r in rows3))
        check("I-3: and it is NOT counted as A",
              counts3.get("A") == counts.get("A"))
    finally:
        mod._CLASS.clear()
        mod._CLASS.update(saved)
    check("I-3: the declaration is restored",
          audit(REPO)[2].get("UNCLASSIFIED", 0) == 0)

    # I-1: an empty scan is a broken scan, not a clean system.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rows2, n2, counts2 = audit(Path(d))
        check("an empty tree says the SCAN is broken, not the system",
              n2 == 0 and "scan is broken" in rows2[0])

    # Every declared class is one of the two, with a reason.
    check("every declared entry is A or B",
          all(k in ("A", "B") for k, _ in _CLASS.values()))
    check("every declared entry gives a reason",
          all(len(why) > 10 for _, why in _CLASS.values()))

    # The most authority-critical endpoints must be B.
    for op in ("tool_execute", "postgres_restore", "subject_erasure",
               "cortex_observe_turn", "monitor_rule_disable"):
        check(f"{op} is classified B", _CLASS[op][0] == "B")

    print("=" * 60)
    print(f"Service identity audit tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Service identity audit")
    print("=" * 60)
    sys.exit(main())
