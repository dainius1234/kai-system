#!/usr/bin/env python3
"""The reachability report must fire only on intent declared twice.

The narrowing is the whole design, so it is the thing under test: a
`depends_on` alone must not fire, a URL alone must not fire, and both
together on a shared network must not fire.

Calibration figures asserted, not left in prose: the live tree reports
**5** findings from **51** doubly-declared edges. Both numbers matter —
a detector that has gone blind reports 0 findings, and a detector whose
traversal has narrowed reports a smaller denominator while still saying
"clean".
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.check_service_reachability import (  # noqa: E402
    _depends, _environment, audit)

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


def write(tmp: Path, body: str) -> None:
    (tmp / "docker-compose.t.yml").write_text(textwrap.dedent(body),
                                              encoding="utf-8")


def main() -> int:
    import tempfile

    # ── environment parsing, both compose spellings ──
    check("mapping environment is read",
          _environment({"environment": {"A": "x"}}) == {"A": "x"})
    check("list environment is read",
          _environment({"environment": ["A=x", "B=y"]}) == {"A": "x", "B": "y"})
    check("a list entry without = is skipped",
          _environment({"environment": ["BARE"]}) == {})
    check("non-string values are dropped, not crashed on",
          _environment({"environment": {"A": 5}}) == {})
    check("mapping depends_on is read",
          _depends({"depends_on": {"a": {"condition": "service_healthy"}}}) == {"a"})
    check("list depends_on is read", _depends({"depends_on": ["a", "b"]}) == {"a", "b"})
    check("absent depends_on is empty", _depends({}) == set())

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)

        # ── I-3: both declarations, disjoint networks -> fires ──
        write(tmp, """
            services:
              caller:
                build: {context: ., dockerfile: c/Dockerfile}
                networks: [alpha]
                depends_on: {peer: {condition: service_healthy}}
                environment: {PEER_URL: "http://peer:8000"}
              peer:
                build: {context: ., dockerfile: p/Dockerfile}
                networks: [beta]
            """)
        f, checked, _ = audit(tmp)
        check("both declared + disjoint networks fires", len(f) == 1)
        check("the finding names both services",
              "caller" in f[0] and "peer" in f[0])
        check("the finding names the variable", "PEER_URL" in f[0])
        check("the finding names both networks",
              "alpha" in f[0] and "beta" in f[0])
        check("the denominator counts the edge", checked == 1)

        # ── a shared network is enough, even if only one of several ──
        write(tmp, """
            services:
              caller:
                build: {context: ., dockerfile: c/Dockerfile}
                networks: [alpha, shared]
                depends_on: [peer]
                environment: {PEER_URL: "http://peer:8000"}
              peer:
                build: {context: ., dockerfile: p/Dockerfile}
                networks: [beta, shared]
            """)
        f, checked, _ = audit(tmp)
        check("one shared network is enough", f == [])
        check("and the edge is still counted", checked == 1)

        # ── depends_on ALONE must not fire (heartbeat's shape) ──
        write(tmp, """
            services:
              caller:
                build: {context: ., dockerfile: c/Dockerfile}
                networks: [alpha]
                depends_on: [peer]
              peer:
                build: {context: ., dockerfile: p/Dockerfile}
                networks: [beta]
            """)
        f, checked, _ = audit(tmp)
        check("depends_on alone does NOT fire", f == [])
        check("depends_on alone is not even counted", checked == 0)

        # ── a URL ALONE must not fire (dashboard's shape) ──
        write(tmp, """
            services:
              caller:
                build: {context: ., dockerfile: c/Dockerfile}
                networks: [alpha]
                environment: {PEER_URL: "http://peer:8000"}
              peer:
                build: {context: ., dockerfile: p/Dockerfile}
                networks: [beta]
            """)
        f, checked, _ = audit(tmp)
        check("a URL alone does NOT fire", f == [])
        check("a URL alone is not counted", checked == 0)

        # ── a URL naming something that is not a service is ignored ──
        write(tmp, """
            services:
              caller:
                build: {context: ., dockerfile: c/Dockerfile}
                networks: [alpha]
                depends_on: [peer]
                environment: {EXT_URL: "https://api.example.com/v1"}
              peer:
                build: {context: ., dockerfile: p/Dockerfile}
                networks: [beta]
            """)
        f, checked, _ = audit(tmp)
        check("an external URL is ignored", f == [] and checked == 0)

        # ── I-1: an empty tree refuses rather than passing ──
        with tempfile.TemporaryDirectory() as empty:
            f, checked, files = audit(Path(empty))
            check("an empty tree REFUSES rather than passing",
                  len(f) == 1 and "inspected nothing" in f[0])
            check("an empty tree reports zero inspected", checked == 0)

    # ── CALIBRATION against the live tree ──
    f, checked, files = audit(REPO)
    check("CALIBRATION: the live tree reports exactly 5", len(f) == 5)
    check("CALIBRATION: from exactly 51 doubly-declared edges", checked == 51)
    check("CALIBRATION: it traverses all three compose files", files == 3)
    check("CALIBRATION: executor -> memu-core is among them",
          any("executor" in x and "memu-core" in x for x in f))
    check("CALIBRATION: supervisor -> heartbeat is among them",
          any("supervisor" in x and "heartbeat" in x for x in f))
    check("CALIBRATION: dashboard is NOT flagged (optional peers)",
          not any("`dashboard`" in x for x in f))
    check("CALIBRATION: heartbeat is not flagged as a CALLER",
          not any(x.split("`")[1] == "heartbeat" for x in f))

    print("=" * 60)
    print(f"Service reachability tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Service reachability tests")
    print("=" * 60)
    sys.exit(main())
