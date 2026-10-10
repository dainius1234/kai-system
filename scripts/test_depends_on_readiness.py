"""Tests for `check_depends_on_readiness` — the wait that waited for nothing.

`docker-compose.full.yml` declared sixteen services with a bare
`depends_on` list. A bare list waits for the dependency container to be
**created** — not started, not listening, not healthy. They were fixed on
2026-08-07 in `e47622b`, the register entry was closed, and eleven more
instances of the identical class survived in the same tree on the same
day:

    full.yml 0 | minimal.yml 1 | sovereign.yml 10

Nothing found them because nothing was looking: thirty gate scripts and
none mentioned `depends_on`. The fix's scope was one file; the class's
scope was the tree.

The calibration test below is the load-bearing one. Pointed at
`full.yml` as it stood before its fix, the rule must report exactly
sixteen — the known answer. A rule that reports more is a rule that will
fail on things that are right, which is how this programme produced 100
findings for 1 real and 69 findings on a correct tree.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_depends_on_readiness as gate  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

passed = 0
failed = 0
EXPECTED_SCENARIOS = 9
executed: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def write(tmp: str, body: str) -> Path:
    root = Path(tmp)
    (root / "docker-compose.test.yml").write_text(body, encoding="utf-8")
    return root


# ── I-3: prove it can fail ───────────────────────────────────────────

def test_a_bare_list_is_a_finding() -> None:
    scenario("bare list fails")
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, """
services:
  db:
    image: postgres:15
  api:
    image: python:3.11-slim
    depends_on:
      - db
""")
        r = gate.audit(root)
        check("one finding", len(r.findings) == 1, str(r.findings))
        check("it names the service", "`api`" in r.findings[0], r.findings[0])
        check("and counts the edge", r.edges == 1, str(r.edges))


def test_a_mapping_without_a_condition_is_a_finding() -> None:
    """The half-migrated shape: mapping syntax, no condition."""
    scenario("condition omitted fails")
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, """
services:
  db:
    image: postgres:15
  api:
    image: python:3.11-slim
    depends_on:
      db: {}
""")
        r = gate.audit(root)
        check("one finding", len(r.findings) == 1, str(r.findings))
        check("it says creation only",
              "creation only" in r.findings[0], r.findings[0])


def test_a_condition_compose_rejects_is_a_finding() -> None:
    """Compose rejects this too — but during `up`, on a runner, minutes in."""
    scenario("invalid condition fails")
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, """
services:
  db:
    image: postgres:15
  api:
    image: python:3.11-slim
    depends_on:
      db:
        condition: service_ready
""")
        r = gate.audit(root)
        check("one finding", len(r.findings) == 1, str(r.findings))
        check("it lists the valid values",
              "service_healthy" in r.findings[0], r.findings[0])


def test_an_explicit_condition_passes_and_is_counted() -> None:
    scenario("explicit condition passes")
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, """
services:
  db:
    image: postgres:15
    healthcheck:
      test: ["CMD", "pg_isready"]
  api:
    image: python:3.11-slim
    depends_on:
      db:
        condition: service_healthy
""")
        r = gate.audit(root)
        check("clean", r.findings == [], str(r.findings))
        check("edge counted", r.edges == 1, str(r.edges))
        check("no service_started", r.started == 0, str(r.started))


# ── the reported-not-enforced clause ─────────────────────────────────

def test_declining_an_available_probe_is_reported_not_enforced() -> None:
    """`dashboard` does this four times in minimal.yml and that profile
    is green in CI. Blocking a UI on a slow optional dependency is a
    design decision, not a defect — so it is listed, not failed."""
    scenario("advisory not enforced")
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, """
services:
  tts:
    image: python:3.11-slim
    healthcheck:
      test: ["CMD", "true"]
  ui:
    image: python:3.11-slim
    depends_on:
      tts:
        condition: service_started
""")
        r = gate.audit(root)
        check("exit stays clean", r.findings == [], str(r.findings))
        check("but it is reported", len(r.advisories) == 1, str(r.advisories))
        check("and counted", r.started == 1, str(r.started))


def test_service_started_on_a_probeless_target_is_silent() -> None:
    """No healthcheck means no signal to wait for. This gate does not
    invent readiness it cannot observe."""
    scenario("probeless target silent")
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, """
services:
  vault:
    image: hashicorp/vault:1.18
  rotator:
    image: hashicorp/vault:1.18
    depends_on:
      vault:
        condition: service_started
""")
        r = gate.audit(root)
        check("clean", r.findings == [], str(r.findings))
        check("and not even advised", r.advisories == [], str(r.advisories))


# ── I-1 and calibration ──────────────────────────────────────────────

def test_a_tree_with_no_compose_files_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        r = gate.audit(Path(tmp))
        check("it refuses", r.findings != [], str(r.findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in r.findings),
              str(r.findings))
        check("with a zero denominator", r.edges == 0, str(r.edges))


def test_calibration_against_the_known_answer() -> None:
    """The rule must report exactly the sixteen defects that were really
    there — not more. Over-reporting is the failure mode that costs most.

    Skipped rather than failed if git cannot produce the old blob; a
    test that cannot fetch its input has not proven anything either way,
    and saying so is better than a green tick.
    """
    scenario("calibration")
    try:
        blob = subprocess.run(
            ["git", "show", "e47622b~1:docker-compose.full.yml"],
            cwd=REPO, capture_output=True, text=True, timeout=30)
    except Exception as exc:                            # noqa: BLE001
        print(f"  SKIP: calibration — git unavailable ({exc})")
        return
    if blob.returncode != 0 or not blob.stdout:
        print("  SKIP: calibration — e47622b~1 not in this checkout "
              "(shallow clone?)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        root = write(tmp, blob.stdout)
        (root / "docker-compose.test.yml").rename(
            root / "docker-compose.full.yml")
        r = gate.audit(root)
        check("exactly the sixteen known defects", len(r.findings) == 16,
              f"{len(r.findings)} findings")
        check("and no advisories invented", r.advisories == [],
              str(r.advisories))


def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    r = gate.audit()
    check("no undeclared wait", r.findings == [], str(r.findings[:2]))
    check("across a real number of edges", r.edges > 50, str(r.edges))


def run_all() -> None:
    test_a_bare_list_is_a_finding()
    test_a_mapping_without_a_condition_is_a_finding()
    test_a_condition_compose_rejects_is_a_finding()
    test_an_explicit_condition_passes_and_is_counted()
    test_declining_an_available_probe_is_reported_not_enforced()
    test_service_started_on_a_probeless_target_is_silent()
    test_a_tree_with_no_compose_files_refuses()
    test_calibration_against_the_known_answer()
    test_the_repository_passes_today()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Depends-On Readiness Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
