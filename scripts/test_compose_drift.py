"""Compose drift gate tests — proving a directional ratchet fires correctly.

`check_compose_drift` was one of eight gates that had **never been
observed failing** (KAI-GATE-003). It also compared only two of three
compose profiles, so `docker-compose.sovereign.yml` — the profile named
for being hardened — was the one never checked.

The most important case here is `test_stricter_is_not_a_violation`.
Sovereign deliberately hardens its executor beyond the baseline
(`runtime: gvisor`, `apparmor:executor-aa`, `read_only`). An
equality-based drift check flags that, and the cheapest way to make it
green is to **weaken sovereign**. A gate that pushes toward less security
is worse than no gate, so the ratchet direction is asserted explicitly
rather than left to be inferred from the implementation.

`test_a_profile_may_omit_a_network_it_does_not_use` is a regression test
for a false positive this gate had *during* its rewrite: `minimal` has no
`execution-net` because it runs no executor, and the first version
reported that as isolation drift. Flagging it would have invited someone
to declare a network nothing attaches to — defect 7's shape, which is
worse than the gap it replaces.

Every fixture is a synthetic compose tree in a temp directory. Nothing
reads the repository's own compose files, so these cases keep testing
what they claim to test however the real profiles evolve.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_compose_drift as drift  # noqa: E402
from scripts.security.gate_inputs import MissingInputs, resolve  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 18
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def scenario(name: str) -> None:
    executed.append(name)


# ── Synthetic profiles ───────────────────────────────────────────────

DEFAULTS = """x-defaults: &d
  restart: unless-stopped
  security_opt:
    - no-new-privileges:true
  logging:
    driver: json-file
"""


def profile(services: str, networks: str = "", defaults: str = DEFAULTS) -> str:
    body = defaults + "services:\n" + services
    body += "networks:\n" + (networks or "  agent-net:\n    internal: true\n")
    return body


GUARDED = ("  api:\n"
           "    image: x\n"
           "    restart: unless-stopped\n"
           "    security_opt:\n"
           "      - no-new-privileges:true\n")


def run(full: str, minimal: str = None, sovereign: str = None):
    """Write three synthetic profiles and run the gate over them."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docker-compose.full.yml").write_text(full)
        (root / "docker-compose.minimal.yml").write_text(minimal or full)
        (root / "docker-compose.sovereign.yml").write_text(sovereign or full)
        return drift.check_drift(root=root)


# ── The ratchet direction — the case that matters most ───────────────

def test_stricter_is_not_a_violation():
    """A profile hardening beyond the baseline must never be flagged."""
    scenario("stricter")
    stricter = ("  api:\n"
                "    image: x\n"
                "    restart: on-failure\n"
                "    security_opt:\n"
                "      - no-new-privileges:true\n"
                "      - apparmor:api-aa\n")
    violations, hardening, _, _ = run(profile(GUARDED),
                                      sovereign=profile(stricter))
    check("hardening is not a violation", not violations, str(violations))
    check("hardening is recorded so it cannot regress",
          any("apparmor:api-aa" in h for h in hardening), str(hardening))


def test_weaker_is_a_violation():
    scenario("weaker")
    weaker = ("  api:\n"
              "    image: x\n"
              "    restart: unless-stopped\n"
              "    security_opt:\n"
              "      - seccomp:unconfined\n")
    violations, _, _, _ = run(profile(GUARDED), sovereign=profile(weaker))
    check("a weaker profile fails", violations, "no violation reported")
    check("the missing setting is named",
          any("no-new-privileges" in v for v in violations), str(violations))


def test_absent_is_a_violation_not_a_difference():
    scenario("absent")
    bare = "  api:\n    image: x\n    restart: unless-stopped\n"
    violations, _, _, _ = run(profile(GUARDED), sovereign=profile(bare))
    check("an unset security_opt fails", violations, "no violation reported")
    check("it is called unguarded, not different",
          any("unguarded" in v for v in violations), str(violations))


# ── restart: presence required, value free ───────────────────────────

def test_a_different_restart_value_is_allowed():
    scenario("restart-value")
    other = ("  api:\n    image: x\n    restart: on-failure\n"
             "    security_opt:\n      - no-new-privileges:true\n")
    violations, _, _, _ = run(profile(GUARDED), sovereign=profile(other))
    check("containment-vs-availability is the profile's call",
          not violations, str(violations))


def test_a_missing_restart_is_a_violation():
    scenario("restart-absent")
    none = ("  api:\n    image: x\n"
            "    security_opt:\n      - no-new-privileges:true\n")
    violations, _, _, _ = run(profile(GUARDED), sovereign=profile(none))
    check("no restart policy fails",
          any("no restart policy" in v for v in violations), str(violations))


# ── Networks — including the false positive caught in review ─────────

def test_a_profile_may_omit_a_network_it_does_not_use():
    """Regression: `minimal` has no execution-net because it runs no executor."""
    scenario("network-omitted")
    both = "  agent-net:\n    internal: true\n  execution-net:\n    internal: true\n"
    one = "  agent-net:\n    internal: true\n"
    violations, _, _, _ = run(profile(GUARDED, both),
                              minimal=profile(GUARDED, one))
    check("an unused network may be absent", not violations, str(violations))


def test_downgrading_an_internal_network_is_a_violation():
    scenario("network-downgrade")
    internal = "  agent-net:\n    internal: true\n"
    external = "  agent-net:\n    internal: false\n"
    violations, _, _, _ = run(profile(GUARDED, internal),
                              sovereign=profile(GUARDED, external))
    check("isolation downgrade fails",
          any("downgraded" in v for v in violations), str(violations))


def test_attaching_to_an_undeclared_network_is_a_violation():
    scenario("network-undeclared")
    svc = ("  api:\n    image: x\n    restart: unless-stopped\n"
           "    security_opt:\n      - no-new-privileges:true\n"
           "    networks:\n      - ghost-net\n")
    violations, _, _, _ = run(profile(svc))
    check("attaching to an undeclared network fails",
          any("undeclared network" in v for v in violations), str(violations))


# ── The anchor-bypass category stays separate ────────────────────────

def test_skipping_a_profiles_own_stricter_anchor_is_reported_not_failed():
    """Two different defects. Folding them together would demand
    `cap_drop: ALL` on Postgres, which needs SETUID to start."""
    scenario("bypass")
    strict_defaults = DEFAULTS.replace(
        "  logging:\n    driver: json-file\n",
        "  cap_drop:\n    - ALL\n  logging:\n    driver: json-file\n")
    violations, _, bypassed, _ = run(
        profile(GUARDED), sovereign=profile(GUARDED, defaults=strict_defaults))
    check("an anchor bypass is not a hard failure", not violations,
          str(violations))
    check("but it is reported", any("does not inherit" in b for b in bypassed),
          str(bypassed))


# ── Every service, not only shared ones ──────────────────────────────

def test_a_service_in_only_one_profile_is_still_checked():
    """The old shared-only comparison never looked at these at all."""
    scenario("unshared")
    only_here = "  lonely:\n    image: x\n    restart: unless-stopped\n"
    violations, _, _, _ = run(profile(GUARDED),
                              sovereign=profile(GUARDED + only_here))
    check("a service in one profile only is still checked",
          any("lonely" in v for v in violations), str(violations))


# ── I-1 and I-2, on this gate ────────────────────────────────────────

def test_a_missing_profile_refuses_rather_than_passing():
    scenario("fail-closed")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docker-compose.full.yml").write_text(profile(GUARDED))
        raised = False
        try:
            resolve(drift.PROFILES, root=root)
        except MissingInputs as exc:
            raised = True
            named = "docker-compose.sovereign.yml" in exc.missing
    check("a missing profile is refused, not skipped", raised)
    check("the missing file is named", raised and named)


def test_the_denominator_counts_every_service_definition():
    scenario("denominator")
    _, _, _, compared = run(profile(GUARDED))
    check("all three profiles are counted", compared == 3, str(compared))


def test_zero_inputs_is_reported_as_not_a_pass():
    scenario("zero-warning")
    from scripts.security.gate_inputs import inspected
    check("a zero denominator carries a warning",
          "not a pass" in inspected(0, "services"), inspected(0, "services"))


def test_the_baseline_itself_is_checked():
    """A defect in `full` must not pass merely by being the baseline."""
    scenario("baseline-checked")
    bare = "  api:\n    image: x\n"
    violations, _, _, _ = run(profile(bare))
    check("the baseline is held to its own floor",
          any("docker-compose.full.yml" in v for v in violations),
          str(violations))



# ── shared container names (KAI-GATE-031) ────────────────────────────
# Docker container names are global to the daemon, not scoped to a
# compose project, so two profiles that claim the same one cannot both be
# up. Six are shared between minimal and sovereign today. That is
# survivable only because CI tears each stack down before the next, which
# is a property of the workflow rather than of these files.

def test_a_name_claimed_by_two_profiles_is_reported():
    scenario("container-name-shared")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docker-compose.full.yml").write_text(
            "services:\n  a:\n    container_name: shared-one\n", encoding="utf-8")
        (root / "docker-compose.minimal.yml").write_text(
            "services:\n  a:\n    container_name: shared-one\n", encoding="utf-8")
        (root / "docker-compose.sovereign.yml").write_text(
            "services:\n  b:\n    container_name: its-own\n", encoding="utf-8")
        shared = drift.shared_container_names(root)
        check("the collision is reported", len(shared) == 1, str(shared))
        check("and names both claimants",
              shared and "minimal" in shared[0] and "full" in shared[0],
              str(shared))
        check("the unique name is not reported",
              all("its-own" not in line for line in shared), str(shared))


def test_distinct_names_are_not_reported():
    scenario("container-name-distinct")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for i, f in enumerate(("docker-compose.full.yml",
                               "docker-compose.minimal.yml",
                               "docker-compose.sovereign.yml")):
            (root / f).write_text(
                f"services:\n  a:\n    container_name: name-{i}\n",
                encoding="utf-8")
        check("nothing is reported", drift.shared_container_names(root) == [],
              str(drift.shared_container_names(root)))


def test_services_without_a_container_name_are_ignored():
    """Most services have none; compose derives one per project, which
    does not collide. Flagging those would bury the six that matter."""
    scenario("container-name-absent")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for f in ("docker-compose.full.yml", "docker-compose.minimal.yml",
                  "docker-compose.sovereign.yml"):
            (root / f).write_text("services:\n  a:\n    image: x:1\n",
                                  encoding="utf-8")
        check("no false positives", drift.shared_container_names(root) == [],
              str(drift.shared_container_names(root)))


def test_the_real_tree_reports_the_six_that_are_known():
    """Calibration against the repository. Reads it, and can only ever
    get stronger: if these six are renamed the count falls, and the
    assertion is a ceiling, not an equality."""
    scenario("container-name-real-tree")
    shared = drift.shared_container_names()
    check("the known collisions are still visible", len(shared) >= 1,
          str(shared))
    check("and they are the minimal/sovereign pair",
          all("minimal" in line and "sovereign" in line for line in shared),
          str(shared))


def run_all() -> None:
    test_stricter_is_not_a_violation()
    test_weaker_is_a_violation()
    test_absent_is_a_violation_not_a_difference()
    test_a_different_restart_value_is_allowed()
    test_a_missing_restart_is_a_violation()
    test_a_profile_may_omit_a_network_it_does_not_use()
    test_downgrading_an_internal_network_is_a_violation()
    test_attaching_to_an_undeclared_network_is_a_violation()
    test_skipping_a_profiles_own_stricter_anchor_is_reported_not_failed()
    test_a_service_in_only_one_profile_is_still_checked()
    test_a_missing_profile_refuses_rather_than_passing()
    test_the_denominator_counts_every_service_definition()
    test_zero_inputs_is_reported_as_not_a_pass()
    test_the_baseline_itself_is_checked()

    test_a_name_claimed_by_two_profiles_is_reported()
    test_distinct_names_are_not_reported()
    test_services_without_a_container_name_are_ignored()
    test_the_real_tree_reports_the_six_that_are_known()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Compose Drift Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
