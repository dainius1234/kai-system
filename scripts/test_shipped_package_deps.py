"""Tests for `check_shipped_package_deps` — the policy nothing ever read.

`security/policy.yml` calls itself, in its own header, *"the single
source of truth — every runtime decision reads from this file."*
Thirty-five service images `COPY common/`, `common/policy.py` reads that
file with pyyaml, and **none of the thirty-five declared pyyaml**. The
import failed in every one of them, a fallback ran `json.loads` on a
YAML document, and the policy loaded empty — every permission dropping
to its most restrictive default.

Proven on 2026-08-06, when the sovereign profile started tool-gate for
the first time:

    JSONDecodeError: Expecting value: line 14 column 1 (char 13)
    POLICY FILE CORRUPT OR UNREADABLE — failing closed.

The day's pattern with the subject changed: not code that never
executed, but configuration that was never loaded.

The scope tests below are the point of this file. The first draft of the
gate flagged every import anywhere in the copied package and produced
over a hundred findings, of which one was real — including `torch`
against `weather-service`, because `common/gpu_utils.py` probes for CUDA
inside a `try:`. Acting on that means adding a two-gigabyte dependency
to a weather service to satisfy a check.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_shipped_package_deps as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 7
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


def tree(pkg_body: str, requirements: str = "fastapi\n"):
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    pkg = root / "common"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "policy.py").write_text(pkg_body)
    svc = root / "svc"
    svc.mkdir()
    (svc / "app.py").write_text("from common.policy import POLICY\n")
    (svc / "requirements.txt").write_text(requirements)
    (svc / "Dockerfile").write_text(
        "FROM python:3.11-slim\n"
        "COPY svc/requirements.txt ./\n"
        "COPY svc/app.py ./app.py\n"
        "COPY common/ ./common/\n"
        'CMD ["python", "app.py"]\n')
    return tmp, root


def test_an_unguarded_missing_import_is_reported() -> None:
    scenario("unguarded missing import")
    tmp, root = tree("import httpx\nPOLICY = {}\n")
    with tmp:
        findings, images, copies = gate.audit(root)
        check("reported", len(findings) == 1, str(findings))
        check("and names the distribution",
              findings and "httpx" in findings[0], str(findings))
        check("with a real denominator", copies == 1, str(copies))


def test_a_declared_import_passes() -> None:
    scenario("declared import passes")
    tmp, root = tree("import httpx\nPOLICY = {}\n", "fastapi\nhttpx>=0.27\n")
    with tmp:
        findings, _, _ = gate.audit(root)
        check("nothing reported", findings == [], str(findings))


def test_an_import_name_is_mapped_to_its_distribution() -> None:
    """`import yaml` is installed by `pyyaml`. Comparing import names to
    requirements lines directly would report every such pair."""
    scenario("distribution name mapped")
    tmp, root = tree("import yaml\nPOLICY = {}\n", "fastapi\npyyaml>=6.0\n")
    with tmp:
        check("pyyaml satisfies `import yaml`", gate.audit(root)[0] == [],
              str(gate.audit(root)[0]))


def test_a_guarded_import_is_not_reported() -> None:
    """The scope decision this gate turns on. `common/gpu_utils.py` does
    `import torch` inside a function inside a `try:`, with a correct
    fallback. Reporting it means telling somebody to add two gigabytes
    to a weather service."""
    scenario("guarded import ignored")
    tmp, root = tree("def probe():\n    try:\n        import torch\n"
                     "        return True\n    except Exception:\n"
                     "        return False\nPOLICY = {}\n")
    with tmp:
        check("nothing reported", gate.audit(root)[0] == [],
              str(gate.audit(root)[0]))


def test_an_unreached_module_is_not_reported() -> None:
    """A package module the entry point never imports is not this
    image's dependency. The first draft ignored reachability and
    produced a hundred findings from one real one."""
    scenario("unreached module ignored")
    tmp, root = tree("POLICY = {}\n")
    with tmp:
        (root / "common" / "llm.py").write_text("import torch\n")
        findings, _, _ = gate.audit(root)
        check("nothing reported", findings == [], str(findings))


def test_the_standard_library_is_not_a_dependency() -> None:
    scenario("stdlib ignored")
    tmp, root = tree("import json\nimport pathlib\nPOLICY = {}\n")
    with tmp:
        check("nothing reported", gate.audit(root)[0] == [],
              str(gate.audit(root)[0]))


def test_a_tree_with_no_dockerfiles_refuses() -> None:
    """I-1: inspecting nothing is not a pass."""
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, images, copies = gate.audit(Path(tmp))
        check("it refuses", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", (images, copies) == (0, 0),
              f"{images}, {copies}")


def run_all() -> None:
    test_an_unguarded_missing_import_is_reported()
    test_a_declared_import_passes()
    test_an_import_name_is_mapped_to_its_distribution()
    test_a_guarded_import_is_not_reported()
    test_an_unreached_module_is_not_reported()
    test_the_standard_library_is_not_a_dependency()
    test_a_tree_with_no_dockerfiles_refuses()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Shipped Package Deps Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
