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
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_gate_registry as meta  # noqa: E402
from scripts.security import gate_registry as registry_module  # noqa: E402

passed = 0
failed = 0

EXPECTED_SCENARIOS = 37
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
    # Driven from a synthetic module, not from a real gate.
    #
    # Two earlier versions of this assertion were wrong in the two ways
    # this programme keeps correcting. The first pinned `== [70]` and
    # broke when an edit moved the site to 112 — guarded on state it did
    # not control. The second asserted "exactly one site in
    # check_port_bindings", and broke the moment that site was **fixed**
    # — a test that required the defect to persist, which is a
    # self-consuming guard in its purest form.
    #
    # The detector's behaviour is the invariant. The repository's current
    # defect count is not.
    import tempfile, pathlib as _p
    tmp = _p.Path(tempfile.mkdtemp())
    (tmp / "check_synth.py").write_text(
        "from pathlib import Path\n"
        "def f(paths):\n"
        "    for p in paths:\n"
        "        if not p.exists():\n"
        "            continue\n"
        "        yield p\n")
    original, meta.SECURITY = meta.SECURITY, tmp
    try:
        lines = meta.skips_absent_input("check_synth")
    finally:
        meta.SECURITY = original
    check("the skip shape is detected", len(lines) == 1, str(lines))
    check("the line is reported", lines and lines[0] > 0, str(lines))


def test_a_return_that_refuses_is_not_counted_as_a_skip():
    """The detector over-counted by 4 of 15 before this distinction.

    A mechanical sweep would have replaced four correct fail-closed
    returns with `require()` calls and called it a fix.
    """
    scenario("ast-refusal")
    import ast
    def first_stmt(src):
        return ast.parse(src).body[0]
    skips = ["continue" if False else "return", "return None", "return []",
             "return violations", "pass"]
    for src in skips:
        node = first_stmt(f"def f():\n    {src}\n").body[0]
        check(f"`{src}` is a skip", meta._is_skip(node), src)
    for src in ["return AnchorFailure('absent', 'gone')",
                "return [Violation(5, 'x')]",
                "return (LIVE, 'no auth.js')"]:
        node = first_stmt(f"def f():\n    {src}\n").body[0]
        check(f"`{src[:28]}...` is a refusal", not meta._is_skip(node), src)


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


# ── A-04e: partial enforcement, and a ratchet on the ratchet ─────────

def test_invariants_are_counted_per_dimension():
    scenario("invariant-counts")
    # Buckets derived from the real category list, so a new category
    # cannot make this test raise KeyError instead of asserting. It did
    # exactly that when "misreported" was added.
    problems = meta.empty_problems()
    problems.update({"unregistered": ["a"], "wiring": ["b"],
                     "blind": ["c", "d"], "unproven": ["e"]})
    counts = meta.invariant_counts(problems)
    check("I-4 sums its three keys", counts["I-4"] == 2, str(counts))
    check("I-1 counts blindness", counts["I-1"] == 2, str(counts))
    check("I-2 at zero", counts["I-2"] == 0, str(counts))


def test_an_enforced_invariant_with_breaches_fails():
    scenario("enforced-breach")
    breaches, _ = meta.verdict({"I-1": 5, "I-2": 3, "I-3": 1, "I-4": 2},
                               enforced=("I-4",))
    check("breaches of an enforced invariant are counted", breaches == 2,
          str(breaches))


def test_a_reported_invariant_with_breaches_does_not_fail():
    scenario("reported-breach")
    breaches, _ = meta.verdict({"I-1": 15, "I-2": 7, "I-3": 7, "I-4": 0},
                               enforced=("I-4",))
    check("reported debt does not fail the gate", breaches == 0,
          str(breaches))


def test_an_invariant_at_zero_must_be_enforced():
    """The ratchet advances itself. Zero without enforcement is debt —
    nothing stops it drifting back."""
    scenario("ready-to-enforce")
    _, ready = meta.verdict({"I-1": 0, "I-2": 7, "I-3": 7, "I-4": 0},
                            enforced=("I-4",))
    check("an invariant at zero and unenforced is flagged", ready == ["I-1"],
          str(ready))
    _, none_ready = meta.verdict({"I-1": 1, "I-2": 7, "I-3": 7, "I-4": 0},
                                 enforced=("I-4",))
    check("a non-zero invariant is not flagged", none_ready == [],
          str(none_ready))


def test_an_inert_constant_is_detected():
    """A policy-shaped constant nobody reads is a lapsed assertion."""
    scenario("inert-constant")
    found = meta.inert_rules("check_restart_recovery")
    check("the now-wired allowlist is not reported", not found, str(found))


def test_an_inert_pass_branch_is_detected():
    scenario("inert-branch")
    import tempfile, pathlib as _p
    src = "X = 1\ndef f(a):\n    if a:\n        pass\n    return a\n"
    tmp = _p.Path(tempfile.mkdtemp())
    (tmp / "check_fake.py").write_text(src)
    original, meta.SECURITY = meta.SECURITY, tmp
    try:
        found = meta.inert_rules("check_fake")
    finally:
        meta.SECURITY = original
    check("a `pass`-bodied condition is caught", len(found) == 1, str(found))
    check("it is named as doing nothing",
          found and "does nothing" in found[0], str(found))


def test_a_non_policy_constant_is_not_reported():
    """An unused URL constant is cruft; an unused ALLOWED_ is a lapse."""
    scenario("inert-nonpolicy")
    import tempfile, pathlib as _p
    tmp = _p.Path(tempfile.mkdtemp())
    (tmp / "check_fake2.py").write_text("SOME_URL = 'x'\nALLOWED_THINGS = {1}\n")
    original, meta.SECURITY = meta.SECURITY, tmp
    try:
        found = meta.inert_rules("check_fake2")
    finally:
        meta.SECURITY = original
    check("only the policy-shaped name is reported", len(found) == 1,
          str(found))
    check("and it is the ALLOWED_ one",
          found and "ALLOWED_THINGS" in found[0], str(found))


# ── I-6: closure is a claim the system re-checks ─────────────────────

def test_every_closure_currently_holds():
    scenario("closure-holds")
    from scripts.security.closure_register import lapsed
    out = lapsed()
    check("no closed finding has silently re-opened", not out, str(out))


def test_removing_a_prevention_reopens_its_findings():
    """Closure that nobody re-checks decays into a rubber stamp."""
    scenario("closure-lapses")
    from scripts.security import closure_register as cr
    # Drop exactly one invariant and expect exactly the closures that
    # depend on it. An earlier version set ENFORCED to ("I-4",) and
    # asserted `len(out) == 2` — then closing 001-003 made it 5, because
    # the assertion was pinned to how many findings happened to exist
    # rather than to the behaviour. Third time this file has been guarded
    # on state it does not control; the rule is the same every time.
    depends_on_i5 = {c.finding for c in cr.CLOSED
                     if "I-5" in c.prevention}
    original = meta.ENFORCED
    meta.ENFORCED = tuple(n for n in original if n != "I-5")
    try:
        out = cr.lapsed()
    finally:
        meta.ENFORCED = original
    reopened = {line.split(":")[0] for line in out}
    check("dropping I-5 re-opens exactly the closures that name it",
          reopened == depends_on_i5, f"{reopened} vs {depends_on_i5}")
    check("at least one closure depends on it", depends_on_i5, "none")
    check("and they close again when restored", not cr.lapsed())


def test_every_closure_records_the_full_template():
    """The operator set the template; a closure missing evidence is a
    note, not a review."""
    scenario("closure-template")
    from scripts.security.closure_register import CLOSED
    incomplete = [
        c.finding for c in CLOSED
        if not (c.defect and c.fix and c.prevention
                and c.proven_by and c.verified_on)
    ]
    check("every closure carries defect, fix, prevention, proof and date",
          not incomplete, str(incomplete))
    # `proven_by` may name several suites. Check every path it mentions,
    # not just the first token — the first version split on " " and took
    # element zero, which for a comma-separated list is a filename with a
    # trailing comma that exists nowhere.
    import re as _re
    missing = []
    for closure in CLOSED:
        for rel in _re.findall(r"scripts/[\w/]+\.py", closure.proven_by):
            if not pathlib_exists(rel):
                missing.append(f"{closure.finding}: {rel}")
    check("every named proof file exists", not missing, str(missing))
    check("every closure names at least one proof file",
          all(_re.search(r"scripts/[\w/]+\.py", c.proven_by) for c in CLOSED),
          str([c.finding for c in CLOSED
               if not _re.search(r"scripts/[\w/]+\.py", c.proven_by)]))


def pathlib_exists(rel: str) -> bool:
    import pathlib
    return (pathlib.Path(__file__).resolve().parent.parent / rel).exists()


def test_the_enforced_set_never_shrinks():
    """A floor on the ratchet itself.

    Removing an invariant from ENFORCED is the regression this whole file
    exists to prevent, so each is named against a literal rather than
    against the current value.

    The count is a FLOOR, not an equality. It was `== 6`, which meant a
    test named "never shrinks" also forbade growth: adding I-7 failed it.
    An assertion that blocks the improvement it was written to protect is
    its own small version of the ratchet problem.
    """
    scenario("enforced-floor")
    check("I-4 is enforced", "I-4" in meta.ENFORCED, str(meta.ENFORCED))
    check("I-5 is enforced", "I-5" in meta.ENFORCED, str(meta.ENFORCED))
    check("I-6 is enforced", "I-6" in meta.ENFORCED, str(meta.ENFORCED))
    check("I-2 is enforced", "I-2" in meta.ENFORCED, str(meta.ENFORCED))
    check("I-7 is enforced", "I-7" in meta.ENFORCED, str(meta.ENFORCED))
    # A FLOOR, not an equality. The literal `== 6` blocked I-7 from being
    # added at all — a test named "never shrinks" that also forbids
    # growth. Every invariant that has ever been enforced is named above,
    # so removing one still fails; adding one does not.
    check("the enforced set has not shrunk", len(meta.ENFORCED) >= 7,
          str(meta.ENFORCED))
    check("every invariant defined is also enforced",
          set(meta.ENFORCED) == set(meta.INVARIANTS), str(meta.ENFORCED))
    check("every enforced name is a real invariant",
          all(n in meta.INVARIANTS for n in meta.ENFORCED), str(meta.ENFORCED))


def test_the_human_summary_agrees_with_the_register():
    """I-4, pointed at the documentation layer.

    The closure register is enforced — every record carries a
    `still_holds` predicate re-evaluated on each run. The status table in
    `INSTRUMENTATION_ARCHITECTURE.md` is a *second* declaration of the
    same state, hand-maintained, and on 2026-08-05 **all ten** closures
    were misreported there: the machine said CLOSED, the document a human
    reads said OPEN or REMEDIATED. Nothing was cross-checking them, which
    is the exact defect KAI-GATE-004 describes, surviving in the one
    place the invariant had never been aimed.
    """
    scenario("human summary agrees with the register")
    from scripts.security.check_gate_registry import misreported_closures
    check("the doc table matches the register today",
          misreported_closures() == [], str(misreported_closures()))


def test_a_misreported_closure_is_detected():
    """Synthetic, so the assertion does not depend on the repo being wrong."""
    scenario("a misreported closure is caught")
    import re as _re
    from scripts.security import check_gate_registry as meta
    from scripts.security.closure_register import CLOSED

    with tempfile.TemporaryDirectory() as tmp:
        doc = Path(tmp) / "doc.md"
        rows = [f"| `{c.finding}` | HIGH | whatever | **CLOSED** |" for c in CLOSED]
        original = meta.ARCHITECTURE_DOC
        try:
            doc.write_text("\n".join(rows), encoding="utf-8")
            meta.ARCHITECTURE_DOC = doc
            check("a table that agrees reports nothing",
                  meta.misreported_closures() == [], str(meta.misreported_closures()))

            first = CLOSED[0].finding
            doc.write_text("\n".join(rows).replace(
                f"| `{first}` | HIGH | whatever | **CLOSED** |",
                f"| `{first}` | HIGH | whatever | **OPEN** |"), encoding="utf-8")
            found = meta.misreported_closures()
            check("a row that disagrees is reported", len(found) == 1, str(found))
            check("and it names the finding", first in found[0], str(found))

            # Fails closed: no table at all is a finding, not a pass. A
            # cross-check that silently passes when it has nothing to
            # compare against is the category confusion this programme
            # has paid for repeatedly.
            doc.write_text("no table here at all\n", encoding="utf-8")
            check("a document with no table is a finding",
                  len(meta.misreported_closures()) == 1,
                  str(meta.misreported_closures()))

            doc.unlink()
            check("an unreadable document is a finding",
                  len(meta.misreported_closures()) == 1,
                  str(meta.misreported_closures()))
        finally:
            meta.ARCHITECTURE_DOC = original


def test_every_ratchet_declares_its_calibration():
    """I-7. A bound that zero satisfies needs proof zero is real.

    A gate bounding a MAXIMUM is satisfied by zero, and zero is exactly
    what a detector that has stopped detecting reports. On 2026-08-05 a
    tokenising bug took the hygiene survey's `clients` from 16 to 0 and
    adoption from 149 to 0, and the gate passed — doing precisely what a
    ratchet does. It was caught because 0 was implausible and somebody
    looked, which is not a control.
    """
    scenario("every ratchet declares calibration")
    from scripts.security.check_gate_registry import uncalibrated_ratchets
    found = uncalibrated_ratchets(lambda rel: (meta.REPO / rel).exists())
    check("no ratchet is uncalibrated", found == [], str(found))

    ratchets = [g for g in registry_module.REGISTRY if g.ratchet]
    check("some gate is actually declared a ratchet",
          len(ratchets) >= 3, str(len(ratchets)))


def test_an_uncalibrated_ratchet_is_detected():
    """Synthetic, so this does not depend on the repo being wrong."""
    scenario("an uncalibrated ratchet is caught")
    from dataclasses import replace
    from scripts.security import check_gate_registry as m

    original = registry_module.REGISTRY
    victim = next(g for g in original if g.ratchet)
    try:
        registry_module.REGISTRY = tuple(
            replace(g, calibrated_by=None) if g is victim else g
            for g in original)
        found = m.uncalibrated_ratchets(lambda rel: (m.REPO / rel).exists())
        check("a ratchet with no calibration is reported",
              any(victim.module in f for f in found), str(found))

        # Fails closed: naming a suite that is not there must not satisfy
        # the check that exists to require one, or deleting the
        # calibration would "fix" it.
        registry_module.REGISTRY = tuple(
            replace(g, calibrated_by="scripts/test_absent_xyz.py — gone")
            if g is victim else g
            for g in original)
        found = m.uncalibrated_ratchets(lambda rel: (m.REPO / rel).exists())
        check("a calibration naming a missing file is reported",
              any(victim.module in f for f in found), str(found))
    finally:
        registry_module.REGISTRY = original


def run_all() -> None:
    test_invariants_are_counted_per_dimension()
    test_an_enforced_invariant_with_breaches_fails()
    test_a_reported_invariant_with_breaches_does_not_fail()
    test_an_invariant_at_zero_must_be_enforced()
    test_every_closure_currently_holds()
    test_removing_a_prevention_reopens_its_findings()
    test_every_closure_records_the_full_template()
    test_an_inert_constant_is_detected()
    test_an_inert_pass_branch_is_detected()
    test_a_non_policy_constant_is_not_reported()
    test_the_enforced_set_never_shrinks()
    test_the_human_summary_agrees_with_the_register()
    test_a_misreported_closure_is_detected()
    test_every_ratchet_declares_its_calibration()
    test_an_uncalibrated_ratchet_is_detected()
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
    test_a_return_that_refuses_is_not_counted_as_a_skip()
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
