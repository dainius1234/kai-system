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


def test_every_declaration_is_still_unbuilt() -> None:
    """A declaration that stopped being true is noise, and noise teaches
    people to ignore the list."""
    scenario("declarations still apply")
    import yaml
    built = set()
    for name in gate.PROFILES:
        doc = yaml.safe_load((gate.REPO / name).read_text(encoding="utf-8"))
        built |= set(gate.profile_dockerfiles(doc).values())
    stale = [d.path for d in gate.DECLARED if d.path in built]
    check("none of the declared orphans is built", stale == [], str(stale))


def test_every_declaration_names_an_existing_file() -> None:
    """Declaring a file that is not there would quietly shrink the list."""
    scenario("declarations exist")
    tree = gate.tree_dockerfiles()
    missing = [d.path for d in gate.DECLARED if d.path not in tree]
    check("every declared path is in the tree", missing == [], str(missing))


def test_every_declaration_carries_an_owner_and_a_date() -> None:
    """A skip without an expiry is forever — the ci-toleration rule,
    applied to the same class of debt."""
    scenario("declarations dated")
    for entry in gate.DECLARED:
        check(f"{entry.path} has an owner", bool(entry.owner), entry.path)
        check(f"{entry.path} has a review date",
              len(entry.review_by) == 10 and entry.review_by[4] == "-",
              entry.review_by)
        check(f"{entry.path} says why", len(entry.reason) > 40, entry.reason)


def test_the_declared_set_is_small() -> None:
    """A gate whose exception list grows without bound has become a
    record of defeat. Three today; this is a ceiling, not a target."""
    scenario("declared set bounded")
    check("no more than five declared orphans", len(gate.DECLARED) <= 5,
          str(len(gate.DECLARED)))


def run_all() -> None:
    test_every_build_spelling_resolves()
    test_an_image_only_service_builds_nothing()
    test_a_service_with_neither_build_nor_image_is_ignored()
    test_the_repository_passes_today()
    test_the_walk_sees_more_than_the_full_profile()
    test_every_declaration_is_still_unbuilt()
    test_every_declaration_names_an_existing_file()
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
