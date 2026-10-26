#!/usr/bin/env python3
"""Calibration for the doctrine-integrity gate.

The gate exists because two records of the same rules diverged and the
casualty was rule 4 — the one forbidding remembered identifiers — in the
record kept by the party that fetches identifiers (D272).

So the properties under test are the three ways this gate could fail to
notice that, and the two ways it could cry wolf. Every fixture is a
doctrine file written on disk and parsed by the shipped code, because
the defect being guarded lives in reading a real document, and both of
this gate's own bugs were found only by running it against one:

  * the first draft matched single-line bold openers only, "found" 23 of
    27 rules and reported four phantom gaps;
  * the second scanned the whole file and reported rule 7 duplicated,
    because section 0's proactive-duty step 7 is also a bold numbered
    item.

Both are this programme's one finding — a population that is not the
population the check's name claims — inside the check written to catch
it. They are exercised below so they cannot come back.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import check_doctrine_integrity as di  # noqa: E402

GATE = REPO / "scripts" / "security" / "check_doctrine_integrity.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 5
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


def doctrine(rules: list[tuple[int, str]], provenance: list[int] | None = None,
             preamble: str = "") -> str:
    """A minimal doctrine file with a section 0 that must NOT be counted."""
    prov = provenance if provenance is not None else [n for n, _ in rules]
    body = [
        "# Engineering doctrine", "",
        "## 0. Proactive engineering duty", "",
        "1. state what you observed;",
        "7. **do not implement scope expansion without authorisation.**",
        "",
        preamble,
        "## The rules", "",
    ]
    body += [f"{n}. **{t}**" for n, t in rules]
    body += ["", "## Where each rule was earned", "",
             "| rule | the failure that earned it |", "|---|---|"]
    body += [f"| {n} | a failure |" for n in prov]
    return "\n".join(body) + "\n"


def run(text: str) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "DOCTRINE.md"
        p.write_text(text)
        r = subprocess.run([sys.executable, str(GATE), "--file", str(p)],
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr


CLEAN = [(1, "First rule."), (2, "Second rule."), (3, "Third rule.")]


def test_a_clean_doctrine_passes() -> None:
    scenario("clean passes")
    code, out = run(doctrine(CLEAN))
    check("a contiguous, provenanced doctrine PASSES", code == 0, out)
    check("and prints its fingerprint", "DOCTRINE FINGERPRINT:" in out, out)
    check("and states the denominator",
          "inspected: 3 rule(s) across 3 provenance entry(s)" in out, out)


def test_a_dropped_rule_is_caught() -> None:
    """D272's actual failure: a rule silently absent."""
    scenario("dropped rule")
    code, out = run(doctrine([(1, "First rule."), (3, "Third rule.")],
                             provenance=[1, 3]))
    check("a gap FAILS", code == 1, out)
    check("and names the missing id", "rule id(s) [2]" in out, out)
    check("and calls a gap a dropped rule",
          "dropped rule until proven otherwise" in out, out)


def test_a_split_rule_is_caught() -> None:
    """D272's other half: reaching the same count by splitting one rule."""
    scenario("split rule")
    code, out = run(doctrine([(1, "First."), (2, "Second."), (2, "Second b.")],
                             provenance=[1, 2]))
    check("a duplicated id FAILS", code == 1, out)
    check("and names it", "rule id(s) [2]" in out, out)
    check("and says why it is ambiguous", "split or a paste" in out, out)


def test_the_population_is_only_the_rules_section() -> None:
    """Both of this gate's own bugs, made permanent as fixtures."""
    scenario("population is exactly the rules")
    # section 0's bold step 7 must NOT be counted as rule 7
    code, out = run(doctrine(CLEAN))
    check("section 0's bold step 7 is not counted as a rule", code == 0, out)
    check("and the rule count excludes it", "rules   : 3" in out, out)
    # a rule whose bold statement WRAPS must still be found
    wrapped = doctrine(CLEAN).replace(
        "3. **Third rule.**",
        "3. **Third rule, whose statement is long enough that it\n"
        "   wraps across two lines in the source.**")
    code, out = run(wrapped)
    check("a rule whose bold text wraps IS found", code == 0, out)
    check("and the count is still 3", "rules   : 3" in out, out)
    # no rules section at all is a refusal, not an empty pass
    code, out = run("# Doctrine\n\nsome prose and no rules section\n")
    check("a file with no rules section REFUSES", code == 1, out)
    check("and says a fingerprint over an unknown region is worthless",
          "worthless" in out, out)


def test_the_fingerprint_moves_when_the_rules_move() -> None:
    scenario("fingerprint discriminates")
    base = di.fingerprint(dict(CLEAN))[0]
    check("identical rule sets give identical fingerprints",
          di.fingerprint(dict(CLEAN))[0] == base)
    check("a REWORDED rule moves it",
          di.fingerprint({1: "First rule.", 2: "Second rule.",
                          3: "Third rule changed."})[0] != base)
    check("a DROPPED rule moves it",
          di.fingerprint({1: "First rule.", 2: "Second rule."})[0] != base)
    check("a RENUMBERED rule moves it",
          di.fingerprint({1: "First rule.", 2: "Second rule.",
                          4: "Third rule."})[0] != base)
    # prose outside the rules must NOT move it, or every edit invalidates
    # every external copy and the fingerprint becomes noise
    a = run(doctrine(CLEAN))[1]
    b = run(doctrine(CLEAN, preamble="Some unrelated new prose.\n"))[1]

    def fp(out: str) -> str:
        return next(l.split(":")[1].strip() for l in out.splitlines()
                    if "DOCTRINE FINGERPRINT" in l)
    check("prose outside the rules does NOT move it", fp(a) == fp(b),
          f"{fp(a)} vs {fp(b)}")
    # --quiet is the reconciliation surface: one value, nothing else
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "d.md"
        p.write_text(doctrine(CLEAN))
        r = subprocess.run([sys.executable, str(GATE), "--file", str(p),
                            "--quiet"], capture_output=True, text=True)
        check("--quiet emits exactly one line",
              len(r.stdout.strip().splitlines()) == 1, r.stdout)
        check("and it is the fingerprint", r.stdout.strip() == fp(a), r.stdout)


def run_all() -> None:
    test_a_clean_doctrine_passes()
    test_a_dropped_rule_is_caught()
    test_a_split_rule_is_caught()
    test_the_population_is_only_the_rules_section()
    test_the_fingerprint_moves_when_the_rules_move()
    live = di.rules((REPO / "kai-pm" / "ENGINEERING_DOCTRINE.md").read_text())
    print(f"  inspected: {EXPECTED_SCENARIOS} doctrine-integrity scenario(s) "
          f"across 1 gate, and {len(live)} live rule(s)")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Doctrine Integrity Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
