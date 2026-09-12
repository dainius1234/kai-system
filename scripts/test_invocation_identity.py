#!/usr/bin/env python3
"""Calibration for the model-facing invocation-identity check.

The claim the check exists to support is "the repair changed the
instrument and not the experiment". A check that says that is worthless
unless it can also say the opposite, so every case here is a KNOWN
POSITIVE or a KNOWN NEGATIVE with the answer supplied by the mutation,
not by the checker (I-8):

  * mutate a definition INSIDE the derived surface  -> must BREACH
  * mutate one OUTSIDE it                           -> must NOT breach,
                                                       but must still be
                                                       reported as changed
  * delete an in-surface definition                 -> must BREACH
    (a removal that took its own scope with it would hide itself)
  * no change at all                                -> must NOT breach

The fixture is the shipped file itself, read from the tree, so a
definition renamed upstream changes this too. A copy maintained beside
it would be exactly the defect R5 names.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import check_invocation_identity as ci  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 4
executed: list[str] = []

SRC = (REPO / "scripts" / "security" / "stage1_replay.py").read_text()


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def mutate(old: str, new: str) -> str:
    """Text substitution that REFUSES to silently not apply.

    An edit whose anchor no longer matches is how a repair ships
    unapplied and a test goes on passing for the wrong reason.
    """
    if SRC.count(old) != 1:
        raise SystemExit(f"anchor matched {SRC.count(old)} times, not 1: "
                         f"{old!r}")
    return SRC.replace(old, new)


def names(result: dict) -> dict[str, str]:
    return {r["name"]: r["status"] for r in result["rows"]}


def test_the_surface_is_derived_from_the_seeds() -> None:
    scenario("surface derivation")
    defs = ci.top_level(SRC)
    surface = ci.surface_of(defs)
    for name in ("freeze", "send_once", "N1", "SENDABLE", "rebuild",
                 "_digest", "EXPECTED_RESPONSE_FORMAT"):
        check(f"{name} is inside the model-facing surface", name in surface,
              str(sorted(surface)))
    for name in ("main", "loggable", "instrument_identity", "open_output",
                 "probe_output_path", "load_replay_evidence"):
        check(f"{name} is OUTSIDE it", name not in surface,
              str(sorted(surface)))
    check("the surface is a strict subset of the file",
          surface < set(defs), f"{len(surface)}/{len(defs)}")
    mods = ci.reached_modules(defs, surface, ci.import_aliases(SRC))
    check("an aliased repo module resolves to its FILE, not its alias",
          ci.module_file(mods.get("s1", "")) ==
          "scripts/security/select_replay_subject.py", str(mods))
    check("and a stdlib module resolves to nothing",
          ci.module_file("json") is None)


def test_no_change_is_not_a_breach() -> None:
    scenario("known negative: identical")
    res = ci.compare(SRC, SRC)
    check("nothing breaches", res["breaches"] == [], str(res["breaches"]))
    check("and every definition reads unchanged",
          set(names(res).values()) == {"unchanged"}, str(names(res)))


def test_a_change_inside_the_surface_breaches() -> None:
    scenario("known positive: in-surface")
    # the sender's timeout handling — squarely model-facing
    res = ci.compare(SRC, mutate("urllib.request.urlopen(req, timeout=timeout)",
                                 "urllib.request.urlopen(req, timeout=1)"))
    check("send_once is reported CHANGED", names(res).get("send_once") ==
          "CHANGED", str(names(res).get("send_once")))
    check("and it is a breach",
          [b["name"] for b in res["breaches"]] == ["send_once"],
          str(res["breaches"]))
    # the precommitted denominator
    res = ci.compare(SRC, mutate("N1 = 10", "N1 = 9"))
    check("a changed N1 breaches",
          [b["name"] for b in res["breaches"]] == ["N1"], str(res["breaches"]))
    # a deletion must not take its own scope with it
    defs = ci.top_level(SRC)
    dropped = SRC.replace(defs["rebuild"], "")
    res = ci.compare(SRC, dropped)
    check("a REMOVED in-surface definition breaches",
          any(b["name"] == "rebuild" and b["status"] == "REMOVED"
              for b in res["breaches"]), str(res["breaches"]))


def test_a_change_outside_the_surface_does_not() -> None:
    scenario("known negative: out-of-surface")
    res = ci.compare(SRC, mutate(
        '    return (f"[withheld — derived from the reply body: '
        '{len(detail)} chars, "',
        '    return (f"[withheld: {len(detail)} chars, "'))
    check("loggable is reported CHANGED",
          names(res).get("loggable") == "CHANGED", str(names(res)))
    check("but it is NOT a breach", res["breaches"] == [],
          str(res["breaches"]))
    # and the repair actually under review
    res = ci.compare(SRC, SRC)
    check("the check can therefore distinguish the two directions",
          res["breaches"] == [])


def run_all() -> None:
    test_the_surface_is_derived_from_the_seeds()
    test_no_change_is_not_a_breach()
    test_a_change_inside_the_surface_breaches()
    test_a_change_outside_the_surface_does_not()
    defs = ci.top_level(SRC)
    print(f"  inspected: {len(defs)} top-level definition(s), "
          f"{len(ci.surface_of(defs))} in the model-facing surface")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Invocation Identity Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
