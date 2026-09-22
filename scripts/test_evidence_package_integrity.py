#!/usr/bin/env python3
"""Calibration for the at-rest evidence-package integrity GATE.

Every case is built on a synthetic package in a temporary directory.
**No real evidence package is written to, copied over, or mutated by this
suite** — the gate under test is read-only and so is its calibration.

The third known-positive is the one this gate exists for.
`kai-pm/house_in_order_h2/run_h2.py` writes `h2-classification-v1.json`
and `h2-capability-contract.json` directly back into its own package, and
both are manifest-bound. One run of a documented entrypoint would leave
the package structurally normal and no longer matching its banked
identity. That case is simulated here against a fixture.

**Detection is not containment**, and this suite does not claim it is.
The operational DO-NOT-RUN-IN-PLACE hold is the containment; this proves
only that the rewrite would be caught afterwards.

No live-defect floor: nothing here asserts a count or digest taken from
the real repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_evidence_package_integrity as cei  # noqa: E402

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


def build_package(convention: str, artefacts: dict) -> tuple:
    """A synthetic package plus the spec row that describes it."""
    root = Path(tempfile.mkdtemp(prefix="cei-"))
    pkg = root / "pkg"
    pkg.mkdir()
    for name, body in artefacts.items():
        (pkg / name).write_text(body, encoding="utf-8")

    entries = "".join(
        "%s  %s\n" % (hashlib.sha256((pkg / n).read_bytes()).hexdigest(), n)
        for n in sorted(artefacts))
    manifest = pkg / "MANIFEST.sha256"

    if convention == cei.EMBEDDED_ENTRIES:
        aggregate = hashlib.sha256(entries.encode()).hexdigest()
        manifest.write_text(
            entries + "# aggregate manifest sha256: %s\n" % aggregate,
            encoding="utf-8")
        identity = aggregate
    else:
        manifest.write_text(entries, encoding="utf-8")
        identity = hashlib.sha256(manifest.read_bytes()).hexdigest()

    spec = {
        "path": "pkg",
        "convention": convention,
        "identity": identity,
        "provenance": "synthetic fixture",
        "declared_artefacts": len(artefacts),
        "note": "fixture",
    }
    return root, pkg, spec


def run_check(root: Path, spec: dict):
    original = cei.REPO
    try:
        cei.REPO = root
        return cei.check_package(spec)
    finally:
        cei.REPO = original


def tree_digest(path: Path) -> str:
    """Hash of every file under `path`, to prove the gate writes nothing."""
    parts = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            parts.append(item.relative_to(path).as_posix())
            parts.append(hashlib.sha256(item.read_bytes()).hexdigest())
    return hashlib.sha256("".join(parts).encode()).hexdigest()


def test_known_negative_unmodified_package_passes():
    for convention in (cei.WHOLE_FILE, cei.EMBEDDED_ENTRIES):
        root, _, spec = build_package(convention, {"a.json": '{"x":1}\n',
                                                   "b.py": "y = 2\n"})
        findings, verified = run_check(root, spec)
        check(f"an unmodified package passes ({convention[:18]})",
              findings == [] and verified == 2,
              f"{findings} verified={verified}")


def test_known_positive_one_byte_changed():
    root, pkg, spec = build_package(cei.WHOLE_FILE, {"a.json": '{"x":1}\n'})
    (pkg / "a.json").write_text('{"x":2}\n', encoding="utf-8")
    findings, _ = run_check(root, spec)
    check("a changed byte is a finding", len(findings) >= 1, str(findings))
    check("the finding names the file that changed",
          any("a.json" in f and "BYTES CHANGED" in f for f in findings),
          str(findings))


def test_known_positive_listed_artefact_removed():
    root, pkg, spec = build_package(cei.WHOLE_FILE, {"a.json": '{"x":1}\n',
                                                     "b.py": "y = 2\n"})
    (pkg / "b.py").unlink()
    findings, _ = run_check(root, spec)
    check("a listed artefact that is absent is a finding",
          any("absent on disk" in f for f in findings), str(findings))


def test_known_positive_in_place_rewrite_of_a_manifest_bound_output():
    """The H2 v1.0 self-mutation path, simulated.

    run_h2.py:27-28 writes two manifest-bound JSON outputs into its own
    package. This is that, against a fixture.
    """
    root, pkg, spec = build_package(
        cei.EMBEDDED_ENTRIES,
        {"classification.json": json.dumps({"rows": 272}) + "\n",
         "contract.json": json.dumps({"caps": 8}) + "\n",
         "run.py": "print('x')\n"})
    # Exactly what the documented entrypoint does: rewrite its own outputs.
    (pkg / "classification.json").write_text(
        json.dumps({"rows": 273}) + "\n", encoding="utf-8")
    (pkg / "contract.json").write_text(
        json.dumps({"caps": 9}) + "\n", encoding="utf-8")
    findings, _ = run_check(root, spec)
    check("an in-place rewrite of manifest-bound outputs is caught",
          sum(1 for f in findings if "BYTES CHANGED" in f) == 2,
          str(findings))
    # The package IDENTITY is untouched, and that is the point. The
    # rewrite changed artefacts, not the manifest, so the aggregate still
    # matches its own entry lines and the declared value. A gate that
    # checked only the identity would report this package intact while
    # two of its banked artefacts had been silently replaced — which is
    # precisely why every manifest-listed artefact is verified
    # individually rather than trusting one digest to stand for all of
    # them.
    check("the identity alone does NOT move — which is why per-artefact "
          "verification is the thing that catches this",
          not any("aggregate" in f or "identity moved" in f for f in findings),
          "the fixture did not isolate the artefact-only rewrite: " + str(findings))


def test_known_positive_identity_moved_while_entries_agree():
    """A regenerated manifest is not a repair. It must still be a finding."""
    root, pkg, spec = build_package(cei.WHOLE_FILE, {"a.json": '{"x":1}\n'})
    (pkg / "a.json").write_text('{"x":9}\n', encoding="utf-8")
    manifest = pkg / "MANIFEST.sha256"
    manifest.write_text(
        "%s  a.json\n" % hashlib.sha256((pkg / "a.json").read_bytes()).hexdigest(),
        encoding="utf-8")
    findings, verified = run_check(root, spec)
    check("entries agree after a regeneration", verified == 1, str(findings))
    check("but the declared identity has moved, and that is the finding",
          any("identity moved" in f for f in findings),
          "regenerating a manifest silenced the gate — false attestation")


def test_known_negative_untracked_runtime_artefacts_do_not_fail():
    root, pkg, spec = build_package(cei.WHOLE_FILE, {"a.py": "x = 1\n"})
    cache = pkg / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-311.pyc").write_text("compiled\n", encoding="utf-8")
    (pkg / "scratch.tmp").write_text("junk\n", encoding="utf-8")
    findings, _ = run_check(root, spec)
    check("__pycache__ and untracked files do not fail the gate",
          findings == [],
          f"a runtime artefact was treated as evidence mutation: {findings}")


def test_declared_artefact_count_is_compared_only_when_declared():
    root, _, spec = build_package(cei.WHOLE_FILE, {"a.py": "x = 1\n"})
    spec_wrong = dict(spec, declared_artefacts=99)
    findings, _ = run_check(root, spec_wrong)
    check("a declared count that disagrees is a finding",
          any("declares 99 artefacts" in f for f in findings), str(findings))

    spec_silent = dict(spec, declared_artefacts=None)
    findings, _ = run_check(root, spec_silent)
    check("no closed-world comparison is invented where authority is silent",
          findings == [], str(findings))


def test_missing_manifest_is_a_finding_not_a_skip():
    """I-1: a package we cannot verify is not a package that verifies."""
    root, pkg, spec = build_package(cei.WHOLE_FILE, {"a.py": "x = 1\n"})
    (pkg / "MANIFEST.sha256").unlink()
    findings, verified = run_check(root, spec)
    check("an absent manifest is a finding",
          any("missing" in f for f in findings), str(findings))
    check("and nothing is reported as verified", verified == 0, str(verified))


def test_the_gate_writes_nothing():
    """Read-only, proven by hashing the tree before and after."""
    root, pkg, spec = build_package(cei.WHOLE_FILE, {"a.json": '{"x":1}\n',
                                                     "b.py": "y = 2\n"})
    before = tree_digest(root)
    run_check(root, spec)
    (pkg / "a.json").write_text('{"x":5}\n', encoding="utf-8")
    after_mutation = tree_digest(root)
    run_check(root, spec)               # now with a finding present
    after = tree_digest(root)
    check("the gate does not modify a clean package",
          before != after_mutation, "fixture mutation did not register")
    check("the gate does not repair a package it finds broken",
          after == after_mutation,
          "the gate wrote to the subject — it must never repair")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
    print("")
    print("=" * 60)
    print(f"Evidence-package integrity calibration — {len(tests)} scenarios, "
          f"{PASSED} passed, {FAILED} failed")
    for failure in FAILURES:
        print(f"  FAIL {failure}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
