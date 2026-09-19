#!/usr/bin/env python3
"""Calibration for the shared execution-surface mechanism.

The functions in `scripts/security/execution_surface.py` were MOVED from
`check_gate_registry.py`, not reimplemented. Their behaviour is proven
preserved two ways: the meta-gate's own output is byte-identical across
the extraction, and its 82-assertion suite passes unchanged.

This suite exists for the third thing neither of those gives — **the
scars survive the move as ASSERTIONS rather than as docstrings.** Every
case below is a defect this repository actually paid for, and a future
edit that quietly undoes one now fails a test instead of a review.

Every fixture is synthetic. Nothing here asserts a count taken from the
live repository, in either direction: a calibration that requires today's
tree to keep its current shape would punish a healthy repair, which is
the defect corrected out of Control A's design in the same tranche.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import execution_surface as es  # noqa: E402

PASSED = 0
FAILED = 0
FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")


def _with_repo(files: dict):
    """Run against a synthetic repository root. Restores REPO afterwards."""
    tmp = tempfile.mkdtemp(prefix="exec-surface-")
    root = Path(tmp)
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def test_invocation_matches_a_run_not_a_mention():
    """A comment naming a gate does not run it, and neither does `echo`."""
    check("python invocation is an invocation",
          es._INVOCATION.findall("python3 scripts/security/check_ports.py")
          == ["check_ports"])
    check("a comment naming a script is not an invocation",
          es._INVOCATION.findall("# see scripts/security/check_ports.py") == [],
          "a comment about a gate was counted as configuring it")
    check("an echo naming a script is not an invocation",
          es._INVOCATION.findall("echo scripts/security/check_ports.py") == [],
          "a log message naming a gate was counted as running it")


def test_swallowed_exit_codes_all_three_shapes():
    check("|| true swallows", es._swallows("x || true", {}, "x || true"))
    check("|| echo swallows", es._swallows("x || echo hi", {}, "x || echo hi"))
    check("continue-on-error swallows",
          es._swallows("x", {"continue-on-error": True}, "x"))
    check("set +e swallows", es._swallows("x", {}, "set +e\nx\n"))
    check("a trailing exit 0 swallows", es._swallows("x", {}, "x\nexit 0\n"))
    check("a plain invocation does not swallow",
          not es._swallows("python3 a.py", {}, "python3 a.py"),
          "an enforcing step was classified as advisory")


def test_lost_step_name_duplicate_run_key():
    """The defect that made a green job run a different gate than it named.

    A step that loses its `- name:` turns its `run:` into a SECOND `run:`
    key on the step above. YAML keeps the last one. The job displayed the
    compose-env gate and executed the test-wiring gate, and
    `check_compose_env.py` never ran in CI at all while the file's text
    said it did. Only the parse knows which one runs.
    """
    root = _with_repo({
        ".github/workflows/w.yml": (
            "name: W\n"
            "jobs:\n"
            "  j:\n"
            "    steps:\n"
            "      - name: names the first gate\n"
            "        run: python3 scripts/security/check_first.py\n"
            "        run: python3 scripts/security/check_second.py\n"
        ),
        "Makefile": "noop:\n\t@true\n",
    })
    original = es.REPO
    try:
        es.REPO = root
        found = es.discover_workflows()
        check("the parse keeps the LAST run key, as YAML does",
              "check_second" in found,
              "the executed gate was not discovered")
        check("the shadowed run key is NOT reported as invoked",
              "check_first" not in found,
              "a gate that never runs was reported as wired — the exact "
              "defect that let check_compose_env.py go unrun while green")
    finally:
        es.REPO = original


def test_makefile_recipe_is_followed_but_comments_are_not():
    root = _with_repo({
        ".github/workflows/w.yml": (
            "name: W\njobs:\n  j:\n    steps:\n"
            "      - name: via make\n        run: make check-docs\n"),
        "Makefile": (
            "check-docs:\n"
            "\t# python3 scripts/security/check_mentioned_only.py\n"
            "\tpython3 scripts/sync_docs.py --check\n"),
    })
    original = es.REPO
    try:
        es.REPO = root
        found = es.discover_workflows()
        check("a make-invoked script is discovered as invoked",
              "sync_docs" in found,
              "a gate reachable only through a make target was invisible")
        check("a script named in a RECIPE COMMENT is not invoked",
              "check_mentioned_only" not in found,
              "a comment explaining a gate counted as configuring it")
    finally:
        es.REPO = original


def test_policy_check_block_stops_at_the_next_target():
    root = _with_repo({
        "Makefile": (
            "policy-check: lint\n"
            "\tpython3 scripts/security/check_in_block.py\n"
            "\t# python3 scripts/security/check_commented.py\n"
            "\n"
            "other-target:\n"
            "\tpython3 scripts/security/check_outside_block.py\n"),
    })
    original = es.REPO
    try:
        es.REPO = root
        found = es.discover_policy_check()
        check("a script in the policy-check recipe is found",
              "check_in_block" in found)
        check("a commented script in the block is not found",
              "check_commented" not in found,
              "a comment inside policy-check counted as enforcement")
        check("a script under a LATER target is not found",
              "check_outside_block" not in found,
              "the block scan ran past its own target")
    finally:
        es.REPO = original


def test_workflow_files_accepts_both_extensions():
    root = _with_repo({
        ".github/workflows/a.yml": "name: A\n",
        ".github/workflows/b.yaml": "name: B\n",
    })
    original = es.REPO
    try:
        es.REPO = root
        names = sorted(p.name for p in es.workflow_files())
        check("both .yml and .yaml are workflows", names == ["a.yml", "b.yaml"],
              f"got {names} — GitHub accepts both and so must this")
    finally:
        es.REPO = original


def test_module_file_resolves_and_refuses():
    """`module_file` is NEW in this tranche, not moved. It is calibrated here.

    It must refuse anything outside the repository: a third-party
    module's paths are not this repository's portability, and following
    one would silently widen every consumer's population.
    """
    check("a security module resolves",
          es.module_file("check_gate_registry") is not None)
    check("a dotted import resolves",
          es.module_file("scripts.security.gate_registry") is not None)
    check("a stdlib module resolves to None, not to site-packages",
          es.module_file("os") is None,
          "resolution escaped the repository")
    check("an unknown name resolves to None rather than guessing",
          es.module_file("no_such_module_anywhere") is None)
    check("the empty name is refused", es.module_file("") is None)


def test_unparseable_workflow_is_not_silently_skipped():
    """I-1: a workflow that cannot be parsed must not shrink the survey.

    `discover_workflows` deliberately does NOT catch the parse error —
    `check_gate_registry.main()` owns surfacing it as a wiring
    disagreement. Swallowing it here would turn an unreadable workflow
    into a clean result, which is boundary blindness.
    """
    root = _with_repo({".github/workflows/bad.yml": "jobs: [ unclosed\n"})
    original = es.REPO
    try:
        es.REPO = root
        raised = False
        try:
            es.discover_workflows()
        except Exception:
            raised = True
        check("an unparseable workflow raises rather than returning clean",
              raised,
              "a broken workflow silently became an empty survey")
    finally:
        es.REPO = original


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print("")
    print("=" * 60)
    print(f"Execution surface calibration — {len(tests)} scenarios, "
          f"{PASSED} passed, {FAILED} failed")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
