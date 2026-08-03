"""Instrumentation meta-check tests — the terminus of the recursion.

`check_gate_registry.py` asserts four invariants about every check in
`scripts/security/`. The operator's criterion for when the regress stops
is the one this file has to satisfy:

> Does the watcher survive the same scrutiny it applies?

and their practical test for A-04a:

> When you point the meta-check at a deliberately malformed registry,
> does it fail with a signal, not noise? When you point it at the real
> registry, does it pass?

Both are asserted below. Every case drives `cross_check()` from a
synthetic registry and a synthetic filesystem — nothing here depends on
the repository's current state, because a meta-check testable only
against the real repo would be guarded on state its own tests modify,
which is the self-consuming shape inside the file written to detect it.

Two cases are regression tests for defects this file's subject actually
had:

  - `test_the_meta_check_never_probes_itself` — the first run spawned
    itself by subprocess and recursed until the process tree had to be
    killed. Depth-one was a property of the design; nothing in the code
    enforced it.
  - `test_an_empty_registry_is_refused` — a meta-check that cross-checks
    zero things must not report a clean pass.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_gate_registry as meta  # noqa: E402
from scripts.security import gate_registry as registry_module  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 21
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


# ── Synthetic fixtures ───────────────────────────────────────────────

@dataclass(frozen=True)
class FakeGate:
    """Structurally what the registry declares, with nothing real behind it."""
    module: str
    kind: str = meta.GATE_KIND
    inputs: Tuple[str, ...] = ()
    optional_inputs: Tuple[str, ...] = ()
    denominator: Optional[str] = None
    probe: bool = True
    probe_skip_reason: Optional[str] = None
    proven_by: Optional[str] = "scripts/test_thing.py"
    in_policy_check: bool = True
    in_workflows: Tuple[str, ...] = ("policy-checks.yml",)
    pending_wiring: Optional[str] = None
    summary: str = "synthetic"
    findings: Tuple[str, ...] = field(default_factory=tuple)


def run(registry, on_disk=None, in_policy=None, in_flows=None,
        blind=None, probe=None, repo_has=None):
    """Drive the cross-check against a wholly synthetic world."""
    modules = [g.module for g in registry]
    return meta.cross_check(
        registry,
        on_disk=modules if on_disk is None else on_disk,
        in_policy=([g.module for g in registry if g.in_policy_check]
                   if in_policy is None else in_policy),
        in_flows=({g.module: list(g.in_workflows)
                   for g in registry if g.in_workflows}
                  if in_flows is None else in_flows),
        blind=blind, probe=probe, repo_has=repo_has,
    )


def clean(problems) -> bool:
    return not any(v for k, v in problems.items() if k != "pending")


# ── I-4: three sources must agree ────────────────────────────────────

def test_a_matching_world_is_clean():
    scenario("clean")
    p = run([FakeGate("check_a"), FakeGate("check_b")])
    check("a consistent registry reports nothing", clean(p), str(p))


def test_an_unregistered_check_fails():
    scenario("unregistered")
    p = run([FakeGate("check_a")], on_disk=["check_a", "check_sneaky"])
    check("a check nobody declared is caught", len(p["unregistered"]) == 1,
          str(p["unregistered"]))
    check("it is named", "check_sneaky" in str(p["unregistered"]),
          str(p["unregistered"]))


def test_a_registered_check_with_no_file_fails():
    scenario("phantom")
    p = run([FakeGate("check_a"), FakeGate("check_ghost")],
            on_disk=["check_a"])
    check("a declaration with no file is caught", len(p["phantom"]) == 1,
          str(p["phantom"]))


def test_a_declaration_that_disagrees_with_the_makefile_fails():
    scenario("wiring-policy")
    p = run([FakeGate("check_a", in_policy_check=True)], in_policy=[])
    check("declared in policy-check but absent is caught",
          any("in_policy_check" in x for x in p["wiring"]), str(p["wiring"]))


def test_a_declaration_that_disagrees_with_the_workflows_fails():
    scenario("wiring-flows")
    p = run([FakeGate("check_a", in_workflows=("policy-checks.yml",))],
            in_flows={"check_a": ["some-other.yml"]})
    check("a workflow mismatch is caught",
          any("workflows" in x for x in p["wiring"]), str(p["wiring"]))


def test_a_report_wired_as_a_gate_fails():
    """A report that gates makes the build depend on information."""
    scenario("report-gated")
    p = run([FakeGate("check_r", kind=meta.REPORT_KIND, in_policy_check=True)])
    check("a REPORT in policy-check is caught",
          any("REPORT" in x for x in p["wiring"]), str(p["wiring"]))


def test_a_gate_nothing_invokes_fails():
    scenario("orphan-gate")
    p = run([FakeGate("check_a", in_policy_check=False, in_workflows=())])
    check("a gate nothing runs is caught",
          any("nothing invokes" in x for x in p["wiring"]), str(p["wiring"]))


def test_pending_wiring_is_reported_but_does_not_fail():
    """The encoded form of 'not enforced yet' — visible, not silent."""
    scenario("pending")
    p = run([FakeGate("check_a", in_policy_check=False, in_workflows=(),
                      pending_wiring="A-04e")])
    check("a declared-pending gate is not a wiring failure",
          not p["wiring"], str(p["wiring"]))
    check("but it is reported every run", len(p["pending"]) == 1,
          str(p["pending"]))
    check("and the step is named", "A-04e" in str(p["pending"]),
          str(p["pending"]))


# ── I-3: proven able to fail ─────────────────────────────────────────

def test_a_check_with_no_failure_suite_fails():
    scenario("unproven")
    p = run([FakeGate("check_a", proven_by=None)])
    check("no proven_by is caught", len(p["unproven"]) == 1,
          str(p["unproven"]))


def test_a_failure_suite_that_does_not_exist_fails():
    scenario("unproven-missing")
    p = run([FakeGate("check_a", proven_by="scripts/test_gone.py")],
            repo_has=lambda rel: False)
    check("a proven_by pointing at nothing is caught",
          any("does not exist" in x for x in p["unproven"]),
          str(p["unproven"]))


# ── I-1: boundary blindness ──────────────────────────────────────────

def test_a_check_that_skips_absent_inputs_is_caught():
    scenario("blind-skip")
    p = run([FakeGate("check_a")], blind=lambda m: [70])
    check("the skip shape is caught", len(p["blind"]) == 1, str(p["blind"]))
    check("the line is named", ":70" in str(p["blind"]), str(p["blind"]))


def test_a_missing_required_input_is_caught():
    scenario("blind-input")
    p = run([FakeGate("check_a", inputs=("docker-compose.full.yml",))],
            repo_has=lambda rel: not rel.endswith(".yml"))
    check("a missing required input is caught",
          any("cannot answer its question" in x for x in p["blind"]),
          str(p["blind"]))


def test_an_optional_input_may_be_absent():
    scenario("blind-optional")
    p = run([FakeGate("check_a", optional_inputs=("docker-compose.sovereign.yml",))],
            repo_has=lambda rel: not rel.endswith(".yml"))
    check("an input declared optional does not fail", not p["blind"],
          str(p["blind"]))


# ── The AST detector, on real source rather than a stub ──────────────

def test_the_ast_detector_finds_the_real_shape():
    scenario("ast-positive")
    lines = meta.skips_absent_input("check_port_bindings")
    check("the known fail-open site is found", lines == [70], str(lines))


def test_the_ast_detector_does_not_flag_a_positive_guard():
    """Defect 7 was a survey with false positives. Not repeating it."""
    scenario("ast-negative")
    import ast
    tree = ast.parse("if path.exists():\n    do_the_work()\n")
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.If))
    check("`if X.exists():` is not an absence test",
          not meta._is_absence_test(node.test))


# ── I-2: a denominator, or an explicit decline ───────────────────────

def test_a_check_with_no_denominator_is_caught():
    scenario("denominator")
    p = run([FakeGate("check_a")], probe=lambda g: ("missing", "none declared"))
    check("a pass with no denominator is caught",
          len(p["denominator"]) == 1, str(p["denominator"]))


def test_a_declared_expensive_probe_is_not_a_failure():
    scenario("denominator-skipped")
    p = run([FakeGate("check_a")], probe=lambda g: ("skipped", "runs make"))
    check("an explicit decline is not a failure", not p["denominator"],
          str(p["denominator"]))


# ── The terminus: does the watcher survive its own scrutiny? ─────────

def test_the_meta_check_never_probes_itself():
    """Regression: the first run recursed until it had to be killed.

    Depth-one was a property of the design and of nothing else. This is
    the code that enforces it.
    """
    scenario("no-self-probe")
    own = registry_module.BY_MODULE["check_gate_registry"]
    calls = []
    original = meta.subprocess.run

    def _trap(argv, **kwargs):
        calls.append(argv)
        raise AssertionError("the meta-check spawned a subprocess for itself")

    meta.subprocess.run = _trap
    try:
        status, detail = meta.probe_denominator(own)
    finally:
        meta.subprocess.run = original
    check("no subprocess is spawned for itself", not calls, str(calls))
    check("it reports the in-process terminus", status == "self", detail)


def test_the_meta_check_prints_the_denominator_it_declares():
    """Its own I-2, asserted from outside against real output."""
    scenario("own-denominator")
    own = registry_module.BY_MODULE["check_gate_registry"]
    buffer = io.StringIO()
    argv, sys.argv = sys.argv, ["check_gate_registry.py"]
    try:
        with contextlib.redirect_stdout(buffer):
            meta.main()
    finally:
        sys.argv = argv
    printed = buffer.getvalue()
    # Asserted against what it actually printed, not against a string
    # rebuilt here — rebuilding it would assert the test's own format.
    check("the declared pattern matches its real output",
          re.search(own.denominator, printed) is not None,
          f"/{own.denominator}/ not found in output")


def test_an_empty_registry_is_refused():
    """A meta-check that cross-checks nothing must not report a pass."""
    scenario("empty-registry")
    original = registry_module.REGISTRY
    registry_module.REGISTRY = ()
    try:
        raised = False
        try:
            meta._load_registry()
        except SystemExit:
            raised = True
    finally:
        registry_module.REGISTRY = original
    check("an empty registry fails closed", raised)


def test_the_real_registry_passes_its_own_structural_rules():
    """The other half of the operator's criterion: real registry, no noise."""
    scenario("real-registry")
    problems = meta.evaluate()
    check("no check on disk is unregistered", not problems["unregistered"],
          str(problems["unregistered"]))
    check("no registered check is missing", not problems["phantom"],
          str(problems["phantom"]))
    check("every declaration matches the Makefile and workflows",
          not problems["wiring"], str(problems["wiring"]))


def run_all() -> None:
    test_a_matching_world_is_clean()
    test_an_unregistered_check_fails()
    test_a_registered_check_with_no_file_fails()
    test_a_declaration_that_disagrees_with_the_makefile_fails()
    test_a_declaration_that_disagrees_with_the_workflows_fails()
    test_a_report_wired_as_a_gate_fails()
    test_a_gate_nothing_invokes_fails()
    test_pending_wiring_is_reported_but_does_not_fail()
    test_a_check_with_no_failure_suite_fails()
    test_a_failure_suite_that_does_not_exist_fails()
    test_a_check_that_skips_absent_inputs_is_caught()
    test_a_missing_required_input_is_caught()
    test_an_optional_input_may_be_absent()
    test_the_ast_detector_finds_the_real_shape()
    test_the_ast_detector_does_not_flag_a_positive_guard()
    test_a_check_with_no_denominator_is_caught()
    test_a_declared_expensive_probe_is_not_a_failure()
    test_the_meta_check_never_probes_itself()
    test_the_meta_check_prints_the_denominator_it_declares()
    test_an_empty_registry_is_refused()
    test_the_real_registry_passes_its_own_structural_rules()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'='*60}")
    print(f"Gate Registry Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
