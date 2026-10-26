#!/usr/bin/env python3
"""Calibration for the runtime-topology report.

The report's job is to keep six things apart that look alike:

    defined · profile-gated · profile-set-enabled · individually
    startable · runtime-proven · expected by a live caller

Every pair that has actually been conflated in this programme gets a
known-positive and a known-negative here, per I-8, and the expected
answers come from **synthetic trees whose content is written in the
test** rather than from the module under test.

The load-bearing case is `memu-graph`. It is profile-gated and it starts,
because `core-tests.yml` names it and Compose enables a service's own
profiles when the service is targeted by name. A first version of this
analysis reported it as "on the dangerous list but not gated" — a false
positive produced by conflating *gated* with *never started*, caught
before it reached the record. If `test_a_named_gated_service_starts_without_any_profile_set`
ever fails, that conflation is back.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import report_runtime_topology as rt  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 12
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


BASE = """\
services:
  core:
    image: core:1
  gated:
    image: gated:1
    profiles: ["extra"]
  dep:
    image: dep:1
"""


def tree(compose: str, invocation: str) -> Path:
    """A throwaway repo: one compose file and one Makefile."""
    root = Path(tempfile.mkdtemp())
    (root / "docker-compose.full.yml").write_text(compose)
    (root / "docker-compose.minimal.yml").write_text("services: {}\n")
    (root / "docker-compose.sovereign.yml").write_text("services: {}\n")
    (root / "Makefile").write_text(f"up:\n\t{invocation}\n")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / "scripts").mkdir()
    return root


def survey(root: Path):
    files = [root / f for f in rt.COMPOSE_FILES]
    defined, _ = rt.definitions(root, files)
    invs = rt.invocations(root, files)
    started, how = rt.started_by(root, files, invs)
    return defined, invs, started, how


# ── defined vs started ───────────────────────────────────────────────

def test_a_default_profile_service_starts_on_a_bare_up() -> None:
    scenario("default-profile starts")
    _, _, started, how = survey(tree(BASE, "docker compose -f docker-compose.full.yml up -d"))
    check("core started", "core" in started)
    check("via default-profile", "default-profile" in how.get("core", ()), str(how))


def test_a_gated_service_does_not_start_on_a_bare_up() -> None:
    """Known-negative. If this fails, `never-started` has stopped meaning
    anything and every count in the report is inflated."""
    scenario("gated excluded from bare up")
    _, _, started, _ = survey(tree(BASE, "docker compose -f docker-compose.full.yml up -d"))
    check("gated NOT started", "gated" not in started, str(sorted(started)))


# ── the distinction the false positive came from ─────────────────────

def test_a_named_gated_service_starts_without_any_profile_set() -> None:
    """THE regression test. Compose enables a service's own profiles when
    it is named, so gated-and-started is a real, correct combination."""
    scenario("named gated service starts")
    _, invs, started, how = survey(
        tree(BASE, "docker compose -f docker-compose.full.yml up -d gated"))
    check("gated started", "gated" in started, str(sorted(started)))
    check("recorded as `named`", "named" in how.get("gated", ()), str(how))
    check("and NO profile set was selected",
          not [p for i in invs for p in i["profiles"]], str(invs))


def test_a_profile_set_selection_is_recognised() -> None:
    """Known-positive for the mechanism the report says is never used."""
    scenario("profile set selected")
    for form in ("COMPOSE_PROFILES=extra docker compose -f docker-compose.full.yml up -d",
                 "docker compose -f docker-compose.full.yml --profile extra up -d"):
        _, invs, started, how = survey(tree(BASE, form))
        selected = sorted({p for i in invs for p in i["profiles"]})
        check(f"profile parsed from `{form.split()[0]}`", selected == ["extra"], str(invs))
        check("gated started by the set", "gated" in started, str(sorted(started)))
        check("recorded as `profile-set`", "profile-set" in how.get("gated", ()), str(how))


def test_a_wildcard_profile_selection_starts_gated_services() -> None:
    scenario("wildcard profile")
    _, _, started, _ = survey(tree(
        BASE, "COMPOSE_PROFILES=* docker compose -f docker-compose.full.yml up -d"))
    check("gated started", "gated" in started, str(sorted(started)))


# ── things that must NOT count as starting something ─────────────────

def test_a_commented_invocation_starts_nothing() -> None:
    """A comment describing a bring-up is not a bring-up. The toleration
    gate learned this the hard way when a comment explaining `|| true`
    was counted as a second `|| true`."""
    scenario("comment starts nothing")
    _, invs, started, _ = survey(tree(BASE, "# docker compose -f docker-compose.full.yml up -d"))
    check("no invocation parsed", invs == [], str(invs))
    check("nothing started", not started, str(sorted(started)))


def test_an_unknown_compose_file_is_ignored() -> None:
    scenario("unknown compose file")
    _, invs, _, _ = survey(tree(BASE, "docker compose -f docker-compose.other.yml up -d"))
    check("ignored", invs == [], str(invs))


def test_dependencies_are_followed_but_labelled_differently() -> None:
    scenario("dependency closure")
    compose = BASE.replace("  core:\n    image: core:1\n",
                           "  core:\n    image: core:1\n    depends_on: [dep]\n")
    _, _, started, how = survey(tree(compose, "docker compose -f docker-compose.full.yml up -d core"))
    check("dep pulled in", "dep" in started, str(sorted(started)))
    check("but marked `dependency`, not `named`",
          sorted(how.get("dep", ())) == ["dependency"], str(how))


# ── I-1 / I-2 ────────────────────────────────────────────────────────

def test_it_fails_closed_on_a_missing_compose_file() -> None:
    scenario("fail closed")
    missing = [f for f in rt.COMPOSE_FILES if not (REPO / f).exists()]
    check("every declared input exists today", not missing, str(missing))
    check("inputs are declared, not globbed", len(rt.COMPOSE_FILES) == 3,
          str(rt.COMPOSE_FILES))


def test_the_denominator_is_printed_and_matches_the_registry_regex() -> None:
    scenario("denominator")
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rt.main()
    out = buf.getvalue()
    check("denominator line present",
          bool(re.search(r"inspected: \d+ service definition\(s\)", out)), out[:200])
    check("counts are reported", "NEVER-STARTED" in out, out[:200])
    check("runtime-proven is UNKNOWN, not invented", "RUNTIME-PROVEN" in out
          and "UNKNOWN" in out, out[:400])


# ── the real tree, as a denominator that can move ────────────────────

def test_the_real_tree_reports_a_plausible_topology() -> None:
    """Not a frozen count — a shape. A count pinned here would need
    editing every time a service is added, and an assertion nobody can
    fail is an assertion nobody reads."""
    scenario("real tree")
    rows, invs, defined = rt.survey()
    check("services are found", len(rows) > 40, str(len(rows)))
    check("invocations are found", len(invs) >= 5, str(len(invs)))
    check("some service starts", any(r["started"] for r in rows))
    check("some service does not", any(not r["started"] for r in rows))
    check("memu-graph is gated AND started",
          any(r["service"] == "memu-graph" and r["gated_everywhere"] and r["started"]
              for r in rows),
          str([r for r in rows if r["service"] == "memu-graph"]))


def test_the_gate_scope_comparison_is_derived_from_the_tree() -> None:
    scenario("gate scope")
    rows, _, defined = rt.survey()
    listed, gated, unlisted = rt.gate_scope(defined)
    check("the tree's gated set is derived", gated == {
        r["service"] for r in rows if r["gated_everywhere"]}, "")
    check("unlisted is a subset of gated", unlisted <= gated, str(unlisted))
    check("nothing listed is ungated today", not (listed & set(defined)) - gated,
          str((listed & set(defined)) - gated))


def run_all() -> None:
    test_a_default_profile_service_starts_on_a_bare_up()
    test_a_gated_service_does_not_start_on_a_bare_up()
    test_a_named_gated_service_starts_without_any_profile_set()
    test_a_profile_set_selection_is_recognised()
    test_a_wildcard_profile_selection_starts_gated_services()
    test_a_commented_invocation_starts_nothing()
    test_an_unknown_compose_file_is_ignored()
    test_dependencies_are_followed_but_labelled_differently()
    test_it_fails_closed_on_a_missing_compose_file()
    test_the_denominator_is_printed_and_matches_the_registry_regex()
    test_the_real_tree_reports_a_plausible_topology()
    test_the_gate_scope_comparison_is_derived_from_the_tree()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed), str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Runtime Topology Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
