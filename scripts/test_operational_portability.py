#!/usr/bin/env python3
"""Calibration for the operational portability GATE.

Known-positives P1-P4 and known-negatives N1-N4, every one on a synthetic
subject whose answer is fixed before the instrument sees it. **No live-tree
count is asserted in either direction**, so a legitimate repair of the real
repository can never fail this suite.

The four positives are the classes the gate must catch, and three of them
are classes the retired lexical detector could not:

  P1  a developer-checkout literal reaching a filesystem call
  P2  an EPHEMERAL SESSION SCRATCH path                      (was invisible)
  P3  a sys.path insertion that resolves an import           (was invisible)
  P4  the same defect in genuinely ACTIVE code reached through the derived
      execution surface — the population calibration

**P4 is the one that matters most and the one that proves least.** It
shows the gate fires on active code the derivation DOES reach. It cannot
show the derivation reaches everything; that limitation is stated in the
gate's own report and in D377, and no test here pretends otherwise.

The negatives are the RC-1 classes the old control could not separate: a
docstring, and calibration fixture data. Confusing either with a
dependency is what suppressed sixteen unrelated live-stack steps.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_operational_portability as cop  # noqa: E402
from scripts.security import execution_surface as es  # noqa: E402

PASSED = 0
FAILED = 0
FAILURES = []

# Fragments, so this suite is not itself an instance of what it tests.
CHECKOUT = "/" + "home" + "/dev/" + "kai-system"
SCRATCH = "/" + "tmp" + "/claude-0/sess/scratch"
CONTAINER_HOME = "/" + "home" + "/appuser/app"


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")


def build(files: dict) -> Path:
    root = Path(tempfile.mkdtemp(prefix="cop-"))
    for rel, body in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root


def scan_source(body: str):
    """Findings for one synthetic module, named so REPO.name matches."""
    root = build({"m.py": body})
    original_cop, original_es = cop.REPO, es.REPO
    try:
        # The gate's USER_HOME class requires the literal to name THIS
        # repository's checkout, so the fixture root must be named for it.
        named = root / "kai-system"
        named.mkdir()
        (named / "m.py").write_text(body, encoding="utf-8")
        cop.REPO = named
        es.REPO = named
        return cop.scan_module(named / "m.py")
    finally:
        cop.REPO, es.REPO = original_cop, original_es


def classes_of(findings):
    return {f["dependency_class"] for f in findings}


def test_P1_developer_checkout_literal_in_filesystem_call():
    findings = scan_source(
        'import pathlib\n'
        'pathlib.Path("%s/data/x.md").read_text()\n' % CHECKOUT)
    check("P1 a developer-checkout literal in a filesystem call fires",
          cop.DEP_LITERAL_IN_CALL in classes_of(findings), str(findings))
    check("P1 is classified USER_HOME",
          any(f["root_class"] == cop.USER_HOME for f in findings),
          str(findings))


def test_P2_ephemeral_session_scratch_dependency():
    """Invisible to the retired detector; live at cal_env.py:15."""
    findings = scan_source(
        'import pathlib\n'
        'SHALLOW = pathlib.Path("%s/h1")\n'
        'SHALLOW.read_text()\n' % SCRATCH)
    check("P2 an ephemeral session path fires",
          len(findings) >= 1, "the class the old lexical needle could not see")
    check("P2 is classified SESSION_SCRATCH",
          any(f["root_class"] == cop.SESSION_SCRATCH for f in findings),
          str(findings))


def test_P3_sys_path_import_resolution_dependency():
    """Invisible to the retired detector; live at run_mutations.py:13->16->17."""
    findings = scan_source(
        'import sys\n'
        'sys.path.insert(0, "%s/kai-pm/pkg")\n'
        'import something\n' % CHECKOUT)
    check("P3 a machine-bound sys.path insertion fires",
          cop.DEP_SYS_PATH in classes_of(findings), str(findings))


def test_P4_population_calibration_active_code_on_the_derived_surface():
    """The gate must reach code the enforcement surface actually runs.

    This is the calibration that guards the narrowing: Control B blocks on
    a SMALLER population than the control it replaces, so the derivation
    itself has to be exercised. If this cannot pass, the architecture does
    not ship.
    """
    root = build({
        ".github/workflows/policy.yml": (
            "name: Policy\njobs:\n  j:\n    steps:\n"
            "      - name: a real gate\n"
            "        run: python3 scripts/security/check_active.py\n"),
        "Makefile": "noop:\n\t@true\n",
        "scripts/security/check_active.py": (
            'import pathlib\n'
            'pathlib.Path("%s/x").read_text()\n' % CHECKOUT),
    })
    named = root / "kai-system"
    named.mkdir()
    for rel in (".github/workflows/policy.yml", "Makefile",
                "scripts/security/check_active.py"):
        target = named / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text((root / rel).read_text(encoding="utf-8"),
                          encoding="utf-8")
    original_cop, original_es = cop.REPO, es.REPO
    try:
        cop.REPO, es.REPO = named, named
        roots, unresolved = cop.enforcing_roots()
        check("P4 the derivation reaches a workflow-invoked gate",
              any(p.name == "check_active.py" for p in roots),
              f"roots={[p.name for p in roots]} unresolved={unresolved}")
        findings = []
        inspected, missing = cop.closure(roots)
        check("P4 no derived root was unopenable", missing == [], str(missing))
        for module in inspected:
            findings.extend(cop.scan_module(module))
        check("P4 a machine-bound dependency in ACTIVE code fires",
              len(findings) >= 1,
              "the gate's narrowed population created a blind spot")
    finally:
        cop.REPO, es.REPO = original_cop, original_es


def test_N1_docstring_prose_is_silent():
    """RC-1 class C, live at h2_v11/passa.py:5."""
    findings = scan_source(
        '"""v1.0 hard-coded FULL=%s and a scratch path, which is why it\n'
        'was unreproducible."""\n'
        'x = 1\n' % CHECKOUT)
    check("N1 a docstring describing the defect does not fire",
          findings == [],
          f"prose was treated as a filesystem dependency: {findings}")


def test_N2_calibration_fixture_literal_is_silent():
    """RC-1 class B, live at cal_claims.py:108 and :113."""
    findings = scan_source(
        'def td(paths, target):\n'
        '    return "COULD_REACH_T", None, None\n'
        'd, w, _ = td(["%s/data/SOUL.md"], "data/SOUL.md")\n' % CHECKOUT)
    check("N2 fixture data passed to a test helper does not fire",
          findings == [],
          f"calibration inputs were treated as dependencies: {findings}")


def test_N3_deployment_and_container_paths_are_silent():
    findings = scan_source(
        'import pathlib\n'
        'pathlib.Path("/data/SOUL.md").read_text()\n'
        'pathlib.Path("/opt/pw-browsers").exists()\n'
        'pathlib.Path("%s/config").read_text()\n' % CONTAINER_HOME)
    check("N3 deployment-fixed paths do not fire", findings == [],
          f"a fixed container path was called a developer checkout: {findings}")
    check("N3 a container/service home is a known negative",
          not any(CONTAINER_HOME in str(f["matched_text"]) for f in findings),
          "every path under a home directory was treated as a checkout")


def test_N4_clean_module_is_silent():
    findings = scan_source(
        'import pathlib\n'
        'ROOT = pathlib.Path(__file__).resolve().parents[2]\n'
        'ROOT.joinpath("data").read_text()\n')
    check("N4 a self-derived root does not fire", findings == [],
          str(findings))


def test_no_subject_is_a_refusal_not_a_pass():
    """R11: an empty surface is a failure to derive, never a clean result."""
    root = build({"Makefile": "noop:\n\t@true\n"})
    (root / ".github" / "workflows").mkdir(parents=True)
    original_cop, original_es = cop.REPO, es.REPO
    try:
        cop.REPO, es.REPO = root, root
        code = cop.main([])
        check("a gate with no derived root refuses rather than passing",
              code == 1,
              "an empty execution surface reported PASS — boundary blindness")
    finally:
        cop.REPO, es.REPO = original_cop, original_es


def test_anti_blinding_root_taxonomy_mutation():
    """Mutate the taxonomy: every known-positive must go red."""
    body = ('import pathlib\n'
            'pathlib.Path("%s/x").read_text()\n' % CHECKOUT)
    check("the fixture fires before mutation", len(scan_source(body)) >= 1)

    original = cop.classify_root
    try:
        cop.classify_root = lambda text: None      # blinded taxonomy
        check("a blinded taxonomy reports nothing — proving the findings "
              "come from the predicate under test", scan_source(body) == [],
              "findings survived a blinded classifier")
    finally:
        cop.classify_root = original

    check("the taxonomy is restored", len(scan_source(body)) >= 1,
          "the mutation leaked out of its own test")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print("")
    print("=" * 60)
    print(f"Operational portability calibration — {len(tests)} scenarios, "
          f"{PASSED} passed, {FAILED} failed")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
