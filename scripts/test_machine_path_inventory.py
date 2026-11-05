#!/usr/bin/env python3
"""Calibration for the machine-path inventory REPORT.

**NO LIVE-DEFECT FLOOR ANYWHERE IN THIS SUITE.** Every expected answer
comes from a synthetic subject built here, whose contents are known
before the instrument is pointed at them. Nothing asserts a count taken
from the real repository, in either direction.

That constraint is the whole design, and it was corrected into this
tranche rather than discovered later. An earlier draft proposed asserting
that the live tree still reports at least the seven occurrences RC-1
comprises. That would have turned today's defects into a permanent
minimum: repair three of them legitimately and a healthier repository
fails the calibration guarding it. A control that punishes remediation is
the gate shape this programme exists to find, and it must not be built
into the fix for one.

Anti-blinding is therefore proven the only honest way — by mutating the
predicate against a fixture whose answer is fixed independently, and
requiring the known-positives to go red.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import machine_path_inventory as inv  # noqa: E402

PASSED = 0
FAILED = 0
FAILURES = []

# Built from fragments for the same reason the instrument does it: a
# suite that writes the literal becomes an instance of what it tests.
HOME = "/" + "home" + "/" + "someone"
SCRATCH = "/" + "tmp" + "/claude-0/session/scratch"


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")


def build(files: dict) -> Path:
    """A synthetic tracked subject. git, because the population is git's."""
    root = Path(tempfile.mkdtemp(prefix="mpi-"))
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(body, bytes):
            path.write_bytes(body)
        else:
            path.write_text(body, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True,
                   capture_output=True)
    return root


def run_on(files: dict):
    root = build(files)
    original = inv.REPO
    try:
        inv.REPO = root
        return inv.scan(inv.tracked_files())
    finally:
        inv.REPO = original


def test_known_positive_one_per_syntactic_class():
    report = run_on({
        "a.py": (
            '"""A docstring naming %s."""\n'
            '# a comment naming %s\n'
            'BOUND = "%s/checkout"\n'
            'import pathlib\n'
            'pathlib.Path("%s/direct")\n' % (HOME, HOME, HOME, HOME)),
        "notes.md": "prose naming %s here\n" % HOME,
    })
    classes = report["by_syntactic_class"]
    check("a docstring occurrence is classified DOCSTRING",
          classes.get(inv.DOCSTRING, 0) >= 1, str(classes))
    check("a comment occurrence is classified COMMENT",
          classes.get(inv.COMMENT, 0) >= 1, str(classes))
    check("an assignment occurrence is classified ASSIGNMENT_TARGET",
          classes.get(inv.ASSIGNMENT_TARGET, 0) >= 1, str(classes))
    check("a direct filesystem argument is classified CALL_ARGUMENT",
          classes.get(inv.CALL_ARGUMENT, 0) >= 1, str(classes))
    check("a non-Python file is classified NON_PYTHON_TEXT",
          classes.get(inv.NON_PYTHON_TEXT, 0) >= 1, str(classes))


def test_session_scratch_is_its_own_root_class():
    """The class the retired lexical detector could not see at all."""
    report = run_on({"a.py": 'import pathlib\npathlib.Path("%s/x")\n' % SCRATCH})
    roots = {o["matched_root_class"] for o in report["occurrences"]}
    check("an ephemeral session path is SESSION_SCRATCH",
          inv.SESSION_SCRATCH in roots, str(roots))


def test_known_negative_deployment_paths_are_not_reported():
    report = run_on({
        "a.py": ('import pathlib\n'
                 'pathlib.Path("/data/SOUL.md")\n'
                 'pathlib.Path("/opt/pw-browsers")\n'
                 'pathlib.Path("/var/log/sovereign")\n'),
        "b.md": "the container writes to /data and /usr/share\n",
    })
    check("deployment-fixed paths produce no occurrence",
          report["occurrences"] == [],
          f"{len(report['occurrences'])} false positive(s): "
          f"{report['occurrences'][:2]}")


def test_flows_is_never_falsely_negative():
    """`false` is a claim. It is made only where prose cannot flow."""
    report = run_on({
        "a.py": ('"""%s in prose."""\n'
                 'BOUND = "%s/x"\n' % (HOME, HOME)),
    })
    by_class = {o["syntactic_class"]: o["flows_to_filesystem_call"]
                for o in report["occurrences"]}
    check("a docstring cannot reach a filesystem call, so flows is False",
          by_class.get(inv.DOCSTRING) is False, str(by_class))
    check("an assignment reports UNKNOWN rather than guessing False",
          by_class.get(inv.ASSIGNMENT_TARGET) == inv.UNKNOWN,
          "an unanalysed binding was reported as not flowing — dataflow is "
          "Control B's job and this report must not pretend to it")


def test_non_text_is_named_not_counted_as_clean():
    report = run_on({
        "binary.bin": b"\x00\x01\x02binary" + HOME.encode(),
        "a.py": "x = 1\n",
    })
    not_text = [e["path"] for e in report["non_text_not_scanned"]]
    check("a binary file is named as not text-scanned",
          "binary.bin" in not_text, str(not_text))
    check("a file we could not read is not counted as having no occurrences",
          report["occurrences"] == [],
          "a NUL-bearing file was scanned as text")
    check("the text rule is stated in the report",
          "NUL" in str(report["text_rule"]) and "UTF-8" in str(report["text_rule"]),
          str(report["text_rule"]))


def test_exclusion_set_is_empty_and_says_so():
    report = run_on({"a.py": "x = 1\n"})
    check("the excluded population is empty",
          report["excluded_population"] == [],
          "a policy exclusion exists — this report must skip nothing")
    check("tracked total and text count are both reported",
          report["tracked_total"] >= 1 and report["text_inspected"] >= 1)


def test_population_comes_from_git_not_a_directory_walk():
    """An untracked file is outside the population, by the tree's own rule."""
    root = build({"tracked.py": 'import pathlib\npathlib.Path("%s/a")\n' % HOME})
    (root / "untracked.py").write_text(
        'import pathlib\npathlib.Path("%s/b")\n' % HOME, encoding="utf-8")
    original = inv.REPO
    try:
        inv.REPO = root
        report = inv.scan(inv.tracked_files())
    finally:
        inv.REPO = original
    paths = {o["path"] for o in report["occurrences"]}
    check("a tracked file is in the population", "tracked.py" in paths, str(paths))
    check("an untracked file is not", "untracked.py" not in paths,
          "the population was a directory walk, not the tree's membership")


def test_report_never_gates():
    """The property that makes this safe to run on every push."""
    root = build({"a.py": 'import pathlib\npathlib.Path("%s/a")\n' % HOME})
    original = inv.REPO
    try:
        inv.REPO = root
        code = inv.main([])
        check("exits 0 with occurrences present", code == 0,
              f"a REPORT returned {code} — it must never gate")
        code_json = inv.main(["--json"])
        check("exits 0 in --json form too", code_json == 0, str(code_json))
    finally:
        inv.REPO = original


def test_anti_blinding_predicate_mutation():
    """A detector that stops detecting must fail this suite, not go quiet.

    The honest form: mutate the PREDICATE against a fixture whose answer
    was fixed before the instrument saw it, and require the positives to
    disappear. No live-tree count is involved, so a legitimate repair of
    the real repository can never fail this.
    """
    files = {"a.py": 'import pathlib\npathlib.Path("%s/a")\n' % HOME}
    baseline = run_on(files)
    check("the fixture is detected before mutation",
          len(baseline["occurrences"]) >= 1, "known-positive did not fire")

    original_patterns = inv._ROOT_PATTERNS
    try:
        inv._ROOT_PATTERNS = ((inv.USER_HOME, re.compile("BOGUS NEVER MATCHES")),)
        blinded = run_on(files)
        check("a blinded predicate reports nothing — proving the count is "
              "load-bearing", blinded["occurrences"] == [],
              "the occurrences did not come from the predicate under test")
    finally:
        inv._ROOT_PATTERNS = original_patterns

    restored = run_on(files)
    check("the predicate is restored after the mutation",
          len(restored["occurrences"]) == len(baseline["occurrences"]),
          "the mutation leaked out of its own test")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print("")
    print("=" * 60)
    print(f"Machine-path inventory calibration — {len(tests)} scenarios, "
          f"{PASSED} passed, {FAILED} failed")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
