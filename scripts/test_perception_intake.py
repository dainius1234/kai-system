#!/usr/bin/env python3
"""The intake report must classify honestly and never infer WORKING.

The denominator moved three times in one afternoon — 7, then 11, then 44
— each time because the previous count started from a list of things
that already existed. So the property under test is not "does it find
the failures", it is **does its population come from the tree**.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.report_perception_intake import (  # noqa: E402
    audit, candidate_sources)

PASSED = 0
FAILED = 0


def check(label: str, condition: bool) -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}")


def main() -> int:
    import tempfile

    # ── the denominator rule, which is the thing that keeps being wrong ──
    services = {
        "audio-service": {"profiles": ["sensors"], "networks": ["sensor-net"]},
        "sysmetrics": {"profiles": ["watchers"], "networks": ["observability-net"]},
        "email-reader": {"profiles": ["external-egress"], "networks": ["egress-net"]},
        "redis": {"networks": ["observability-net", "data-net"]},
        "grafana": {"networks": ["observability-net"]},
        "dashboard": {"networks": ["observability-net", "edge-net"]},
    }
    cand = candidate_sources(services)
    check("a `sensors` service is a candidate", "audio-service" in cand)
    check("a `watchers` service is a candidate", "sysmetrics" in cand)
    check("an egress reader with an adapter is a candidate",
          "email-reader" in cand)

    # The first version of this rule was "on sensor-net or
    # observability-net", which called redis, grafana and dashboard
    # missing perception sources — 49 findings, almost all nonsense.
    check("redis is NOT a perception source", "redis" not in cand)
    check("grafana is NOT a perception source", "grafana" not in cand)
    check("dashboard is NOT a perception source", "dashboard" not in cand)
    check("the narrowed rule yields exactly the three real ones",
          cand == {"audio-service", "sysmetrics", "email-reader"})

    # ── I-1: no input is UNKNOWN, never a clean surface ──
    with tempfile.TemporaryDirectory() as empty:
        rows, n, counts = audit(Path(empty))
        check("an empty tree refuses rather than reporting a clean surface",
              n == 0 and any("must not be read as a clean surface" in r
                             for r in rows))
        check("an empty tree claims no verdicts", counts == {})

    # ── CALIBRATION against the live tree ──
    rows, n, counts = audit(REPO)
    check("CALIBRATION: 44 source-profile pairs inspected", n == 44)
    check("CALIBRATION: exactly 2 are WORKING", counts.get("WORKING") == 2)
    check("CALIBRATION: 5 BROKEN", counts.get("BROKEN") == 5)
    check("CALIBRATION: 3 DORMANT", counts.get("DORMANT") == 3)
    check("CALIBRATION: 3 SUPERSEDED", counts.get("SUPERSEDED") == 3)
    check("CALIBRATION: verdicts account for every inspected pair",
          sum(counts.values()) == n)
    check("CALIBRATION: nothing is UNKNOWN today",
          counts.get("UNKNOWN", 0) == 0)

    # The load-bearing property. WORKING must be rare and earned; if a
    # future change makes most sources WORKING without the intake being
    # rebuilt, the detector has gone blind rather than the tree improving.
    check("WORKING is a small minority — a detector that goes blind "
          "reports everything fine",
          counts.get("WORKING", 0) < n // 4)

    text = "\n".join(rows)
    check("BROKEN findings name the unreachable network",
          "sensor-net" in text and "observability-net" in text)
    check("DORMANT findings say nothing invokes them",
          "nothing invokes it" in text)
    check("MISSING findings distinguish absent-service from no-adapter",
          "is not defined in this profile" in text
          and "no adapter and no event contract" in text)

    print("=" * 60)
    print(f"Perception intake tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Perception intake report tests")
    print("=" * 60)
    sys.exit(main())
