"""Tests for `check_dockerfile_coverage` — the asymmetry that hid a typo.

`document-parser/Dockerfile` carried `--start_period` for
`--start-period`. Docker rejects that at parse time, so it stopped every
image build and the thirteen steps behind it. The typo was not subtle;
what let it survive was that **nothing had ever parsed the file**:

    Dockerfiles in the tree                     52
    referenced by docker-compose.full.yml       30   <- all CI built
    in another profile but not in full.yml      19   <- incl. the typo
    referenced by no profile at all              3

Twenty-two of fifty-two, 42%, never built. `full.yml` sounds like a
superset and is the smaller set.

Everything here is synthetic except the last three, which read the tree.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_dockerfile_coverage as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 11
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


# ── resolving what a profile builds ──────────────────────────────────

def test_every_build_spelling_resolves() -> None:
    """`build: ./x`, `{context: ./x}` and `{context: ., dockerfile: ...}`
    are all in use here. Understanding only some of them is how a gate
    reports PASS for the profile holding the defect."""
    scenario("all build spellings")
    doc = {"services": {
        "one": {"build": "./one"},
        "two": {"build": {"context": "./two"}},
        "three": {"build": {"context": ".", "dockerfile": "three/Dockerfile"}},
    }}
    got = gate.profile_dockerfiles(doc)
    check("all three resolve",
          got == {"one": "one/Dockerfile", "two": "two/Dockerfile",
                  "three": "three/Dockerfile"}, str(got))


def test_an_image_only_service_builds_nothing() -> None:
    """`redis:7-alpine` exercises no Dockerfile in this tree, so counting
    it as coverage would overstate what is checked."""
    scenario("image-only ignored")
    doc = {"services": {"redis": {"image": "redis:7-alpine"}}}
    check("nothing resolved", gate.profile_dockerfiles(doc) == {},
          str(gate.profile_dockerfiles(doc)))


def test_a_service_with_neither_build_nor_image_is_ignored() -> None:
    scenario("empty service ignored")
    doc = {"services": {"x": None, "y": {}}}
    check("nothing resolved", gate.profile_dockerfiles(doc) == {}, "")


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, count, profiles = gate.audit()
    check("no coverage problems", findings == [], str(findings))
    check("every profile was read", profiles == 3, str(profiles))
    check("and the tree walk found the Dockerfiles",
          count > 40, str(count))


def test_the_walk_sees_more_than_the_full_profile() -> None:
    """The finding itself, asserted. If `full.yml` ever becomes a true
    superset this can be revisited — but it must not silently become the
    assumption again."""
    scenario("walk exceeds full profile")
    import yaml
    full = yaml.safe_load(
        (gate.REPO / "docker-compose.full.yml").read_text(encoding="utf-8"))
    built_by_full = set(gate.profile_dockerfiles(full).values())
    tree = gate.tree_dockerfiles()
    check("the tree holds Dockerfiles full.yml never builds",
          len(tree - built_by_full) > 0,
          f"{len(tree)} in tree, {len(built_by_full)} built by full")


def test_a_declaration_that_stopped_being_true_is_reported() -> None:
    """Driven by sets, not by the tree. These two rules used to be
    checked only against the real `DECLARED`, so when the operator
    resolved all three orphans they began looping over an empty tuple
    and asserting nothing. The assertion ratchet caught the drop."""
    scenario("declarations still apply")
    found = gate.coverage_findings(
        built={"a/Dockerfile"}, tree={"a/Dockerfile"},
        declared={"a/Dockerfile"})
    check("a declared file a profile now builds is reported",
          len(found) == 1, str(found))
    check("and it says to remove the declaration",
          found and "remove the declaration" in found[0], str(found))
    check("while a declaration that is still true is silent",
          gate.coverage_findings(built=set(), tree={"a/Dockerfile"},
                                 declared={"a/Dockerfile"}) == [],
          "a still-unbuilt declaration was reported")


def test_an_undeclared_orphan_is_reported() -> None:
    scenario("orphan reported")
    found = gate.coverage_findings(
        built=set(), tree={"x/Dockerfile"}, declared=set())
    check("reported", len(found) == 1, str(found))
    check("and the remedy is spelled out",
          found and "declare it in DECLARED" in found[0], str(found))
    check("declaring it silences exactly that one",
          gate.coverage_findings(built=set(), tree={"x/Dockerfile"},
                                 declared={"x/Dockerfile"}) == [], "")


def test_a_profile_naming_a_missing_dockerfile_is_reported() -> None:
    """I-1: a `build:` pointing at nothing is a finding, not a skip."""
    scenario("declarations exist")
    found = gate.coverage_findings(
        built={"ghost/Dockerfile"}, tree=set(), declared=set())
    check("reported", len(found) == 1, str(found))
    check("named as absent from the tree",
          found and "not present in the tree" in found[0], str(found))
    # Declaring it does not excuse it — and it picks up a second finding,
    # because a declaration for a file a profile builds is itself stale.
    both = gate.coverage_findings(built={"ghost/Dockerfile"}, tree=set(),
                                  declared={"ghost/Dockerfile"})
    check("a declaration does not excuse the missing file",
          any("not present in the tree" in f for f in both), str(both))
    check("and the stale declaration is reported alongside it",
          any("remove the declaration" in f for f in both), str(both))


def test_a_fully_covered_tree_is_silent() -> None:
    scenario("covered tree silent")
    check("nothing reported",
          gate.coverage_findings(built={"a/Dockerfile", "b/Dockerfile"},
                                 tree={"a/Dockerfile", "b/Dockerfile"},
                                 declared=set()) == [], "")


def _well_formed(entry: "gate.Unbuilt") -> bool:
    """The rule a declaration must satisfy, as a function of one entry.

    Extracted so it can be driven by a synthetic entry. `DECLARED` is
    empty since the operator resolved all three orphans by deleting the
    code, and a loop over an empty tuple asserts nothing — a test that
    passes by not running is the failure mode this repository hunts.
    """
    return (bool(entry.owner)
            and len(entry.review_by) == 10 and entry.review_by[4] == "-"
            and len(entry.reason) > 40)


def test_every_declaration_carries_an_owner_and_a_date() -> None:
    """A skip without an expiry is forever — the ci-toleration rule,
    applied to the same class of debt."""
    scenario("declarations dated")
    good = gate.Unbuilt(
        path="x/Dockerfile",
        reason="a reason long enough to actually say something about why "
               "this file is not built by any profile",
        owner="orion", review_by="2026-11-01")
    check("a well-formed declaration passes", _well_formed(good))
    # I-3: prove it can fail, one missing field at a time.
    from dataclasses import replace
    check("no owner fails", not _well_formed(replace(good, owner="")))
    check("no date fails", not _well_formed(replace(good, review_by="soon")))
    check("no reason fails", not _well_formed(replace(good, reason="tbd")))
    for entry in gate.DECLARED:
        check(f"{entry.path} is well-formed", _well_formed(entry), entry.path)


def test_the_declared_set_is_small() -> None:
    """A gate whose exception list grows without bound has become a
    record of defeat. Zero today; this is a ceiling, not a target."""
    scenario("declared set bounded")
    check("no more than five declared orphans", len(gate.DECLARED) <= 5,
          str(len(gate.DECLARED)))
    check("and the ceiling is a real one", 5 < 50)


def run_all() -> None:
    test_every_build_spelling_resolves()
    test_an_image_only_service_builds_nothing()
    test_a_service_with_neither_build_nor_image_is_ignored()
    test_the_repository_passes_today()
    test_the_walk_sees_more_than_the_full_profile()
    test_a_declaration_that_stopped_being_true_is_reported()
    test_an_undeclared_orphan_is_reported()
    test_a_profile_naming_a_missing_dockerfile_is_reported()
    test_a_fully_covered_tree_is_silent()
    test_every_declaration_carries_an_owner_and_a_date()
    test_the_declared_set_is_small()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Dockerfile Coverage Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
