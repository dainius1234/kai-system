"""Tests for `check_image_modules` — ten COPY lines, thirty-seven modules.

`agentic/Dockerfile` listed ten files by name into a directory holding
thirty-seven, and `app.py` imports twenty-seven of the ones it omitted:

    File "/app/app.py", line 27, in <module>
        from system_fsm import KaiEvent as SysEvent, ...
    ModuleNotFoundError: No module named 'system_fsm'

The container died at import on every boot it has ever had. It blocked
nothing until 2026-08-06, because until that day nothing got far enough
to start it — and on that day it was the last service standing between
CI and thirteen steps that had never run.

The list-beside-the-thing pattern, fourteenth venue, and the same remedy
every time: derive the denominator from the tree. `COPY agentic/ ./`
cannot go stale.

Both false positives found while calibrating are asserted here, because
each would have sent somebody to fix code that works:

  * `Dockerfile.introspect` runs `introspect_app.py`, not `app.py`
  * `COPY vault-sync/ ./` brings in every module in the directory

Verified against the real file: `git show 189313d:agentic/Dockerfile`
gives exactly one finding naming 27 modules; the fixed tree gives none.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_image_modules as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 14
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


def tree(files: dict):
    """A synthetic service directory. Returns (root, service dir)."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    svc = root / "svc"
    svc.mkdir()
    for name, body in files.items():
        (svc / name).write_text(body)
    return tmp, root, svc


# ── the defect itself ────────────────────────────────────────────────

def test_the_real_defect_is_caught() -> None:
    scenario("real defect caught")
    tmp, root, svc = tree({
        "app.py": "from system_fsm import fire\nimport cortex\n",
        "system_fsm.py": "def fire(): pass\n",
        "cortex.py": "x = 1\n",
    })
    with tmp:
        text = 'COPY svc/app.py ./\nCMD ["python", "app.py"]\n'
        f = gate.findings_in(text, svc, "svc/Dockerfile", root)
        check("it is reported", len(f) == 1, str(f))
        check("both missing modules are counted",
              f and "missing 2 module(s)" in f[0], str(f))
        check("they are named",
              f and "cortex" in f[0] and "system_fsm" in f[0], str(f))
        check("and the remedy is the directory",
              f and "cannot go stale" in f[0], str(f))


def test_copying_the_directory_fixes_it() -> None:
    """The actual fix applied to agentic."""
    scenario("directory copy passes")
    tmp, root, svc = tree({
        "app.py": "from system_fsm import fire\n",
        "system_fsm.py": "def fire(): pass\n",
    })
    with tmp:
        text = 'COPY svc/ ./\nCMD ["python", "app.py"]\n'
        check("nothing reported",
              gate.findings_in(text, svc, "d", root) == [],
              str(gate.findings_in(text, svc, "d", root)))


def test_the_walk_is_transitive() -> None:
    """`app.py` imports `a`, which imports `b`. Copying only `a` still
    breaks at import — one hop is not the denominator."""
    scenario("transitive walk")
    tmp, root, svc = tree({
        "app.py": "import a\n", "a.py": "import b\n", "b.py": "x = 1\n",
    })
    with tmp:
        text = 'COPY svc/app.py ./\nCOPY svc/a.py ./\nCMD ["python", "app.py"]\n'
        f = gate.findings_in(text, svc, "d", root)
        check("the second hop is reported", len(f) == 1, str(f))
        check("naming b", f and "b" in f[0], str(f))


# ── the two false positives, asserted ────────────────────────────────

def test_the_entry_point_comes_from_the_dockerfile() -> None:
    """`Dockerfile.introspect` runs `introspect_app.py`. Rooting every
    image at `app.py` reported 34 phantom misses against it — a survey
    with false positives sends people to fix working code."""
    scenario("entry point from CMD")
    tmp, root, svc = tree({
        "app.py": "import heavy\n",
        "heavy.py": "x = 1\n",
        "introspect_app.py": "import light\n",
        "light.py": "x = 1\n",
    })
    with tmp:
        text = ('COPY svc/introspect_app.py ./\nCOPY svc/light.py ./\n'
                'CMD ["python", "introspect_app.py"]\n')
        check("app.py's imports are not this image's problem",
              gate.findings_in(text, svc, "d", root) == [],
              str(gate.findings_in(text, svc, "d", root)))


def test_a_directory_copy_brings_everything_in_it() -> None:
    """`COPY vault-sync/ ./` made `parser`, `mapper` and `watcher` look
    missing to the first draft. They are not."""
    scenario("directory copy counted")
    tmp, root, svc = tree({
        "app.py": "import parser_mod\n", "parser_mod.py": "x = 1\n",
    })
    with tmp:
        text = 'COPY svc/ ./\nCMD ["uvicorn", "app:app"]\n'
        check("nothing reported",
              gate.findings_in(text, svc, "d", root) == [],
              str(gate.findings_in(text, svc, "d", root)))


def test_the_uvicorn_spelling_is_understood() -> None:
    """`CMD ["uvicorn", "app:app", ...]` runs `app.py`. Not knowing that
    would make this gate silent over every service that starts that
    way — including vault-sync."""
    scenario("uvicorn entry point")
    tmp, root, svc = tree({"app.py": "import missing_mod\n",
                           "missing_mod.py": "x = 1\n"})
    with tmp:
        text = ('COPY svc/app.py ./\n'
                'CMD ["uvicorn", "app:app", "--port", "8047"]\n')
        f = gate.findings_in(text, svc, "d", root)
        check("the entry point is found", len(f) == 1, str(f))
        check("and app.py is named as what it runs",
              f and "app.py" in f[0], str(f))


def test_a_third_party_import_is_not_a_local_module() -> None:
    """`import fastapi` is a dependency, not a file to copy. Flagging it
    would be a false positive on every service in the tree."""
    scenario("third-party ignored")
    tmp, root, svc = tree({"app.py": "import fastapi\nimport os\n"})
    with tmp:
        text = 'COPY svc/app.py ./\nCMD ["python", "app.py"]\n'
        check("nothing reported",
              gate.findings_in(text, svc, "d", root) == [],
              str(gate.findings_in(text, svc, "d", root)))


def test_a_non_python_image_is_ignored() -> None:
    """`CMD ["nginx", "-g", "daemon off;"]` has no entry point to root
    at, and inventing one would be a guess."""
    scenario("non-python ignored")
    tmp, root, svc = tree({"app.py": "import missing_mod\n",
                           "missing_mod.py": "x = 1\n"})
    with tmp:
        text = 'COPY svc/app.py ./\nCMD ["nginx", "-g", "daemon off;"]\n'
        check("nothing reported",
              gate.findings_in(text, svc, "d", root) == [],
              str(gate.findings_in(text, svc, "d", root)))


# ── I-1: zero inputs is not a pass ───────────────────────────────────

def test_a_tree_with_no_dockerfiles_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, services, files = gate.audit(Path(tmp))
        check("it fails rather than passing", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", (services, files) == (0, 0),
              f"{services}, {files}")


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, services, files = gate.audit()
    check("no image is missing its modules", findings == [], str(findings))
    check("across a real number of images", services > 30, str(services))
    check("and every Dockerfile was read", files >= services, str(files))


def test_the_dockerfile_list_comes_from_a_walk() -> None:
    scenario("walk not list")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel in ("a", "b/c"):
            (root / rel).mkdir(parents=True)
            (root / rel / "Dockerfile").write_text("FROM x\n")
        (root / "b" / "Dockerfile.introspect").write_text("FROM x\n")
        (root / "_archive").mkdir()
        (root / "_archive" / "Dockerfile").write_text("FROM x\n")
        got = gate.dockerfiles(root)
        check("it recurses and takes both spellings", len(got) == 3,
              str([str(p.relative_to(root)) for p in got]))
        check("and skips _archive",
              all("_archive" not in str(p) for p in got), str(got))


# ── the scope this gate shipped with ─────────────────────────────────

def test_a_repo_root_package_counts_as_a_module_the_image_needs() -> None:
    """The defect this gate shipped with, found by a CI run.

    `reachable` built its universe as `service.glob("*.py")` — the
    service's own directory and nothing else. So a service importing
    `common.http_hygiene` needed `common/` in its image and this gate
    could not see the need, because `common` was not a sibling.

    It printed `PASS: all 49 Python service image(s) contain the modules
    they import` while three of them could not start:

        backup-service-1 | from common.http_hygiene import pooled_client
        backup-service-1 | ModuleNotFoundError: No module named 'common'
    """
    scenario("root package is in scope")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "common").mkdir()
        (root / "common" / "__init__.py").write_text("")
        svc = root / "svc"
        svc.mkdir()
        (svc / "app.py").write_text(
            "from common.http_hygiene import pooled_client\n")
        text = ("FROM python:3.11-slim\n"
                "COPY svc/app.py ./app.py\n"
                'CMD ["python", "app.py"]\n')
        pkgs = gate.root_packages(root)
        check("common is discovered as a root package", pkgs == {"common"},
              str(pkgs))
        need = gate.reachable({svc / "app.py"}, svc, pkgs)
        check("and the entry point is seen to need it", need == {"common"},
              str(need))
        found = gate.findings_in(text, svc, "svc/Dockerfile", root)
        check("so the missing COPY is reported", len(found) == 1, str(found))


def test_copying_the_package_satisfies_it() -> None:
    """The inverse direction. A gate that reports the need but never
    sees it met would fail every correct image."""
    scenario("copied package satisfies")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "common").mkdir()
        (root / "common" / "__init__.py").write_text("")
        svc = root / "svc"
        svc.mkdir()
        (svc / "app.py").write_text("from common.errors import X\n")
        text = ("FROM python:3.11-slim\n"
                "COPY svc/app.py ./app.py\n"
                "COPY common/ ./common/\n"
                'CMD ["python", "app.py"]\n')
        check("nothing reported",
              gate.findings_in(text, svc, "svc/Dockerfile", root) == [],
              str(gate.findings_in(text, svc, "svc/Dockerfile", root)))


def test_a_third_party_import_is_not_a_missing_module() -> None:
    """`import fastapi` is pip's job, not COPY's. Reporting it would be
    a scope larger than reality — findings against working images."""
    scenario("third-party import ignored")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        svc = root / "svc"
        svc.mkdir()
        (svc / "app.py").write_text("import fastapi\nimport httpx\n")
        text = ("FROM python:3.11-slim\n"
                "COPY svc/app.py ./app.py\n"
                'CMD ["python", "app.py"]\n')
        check("nothing reported",
              gate.findings_in(text, svc, "svc/Dockerfile", root) == [], "")


def run_all() -> None:
    test_a_repo_root_package_counts_as_a_module_the_image_needs()
    test_copying_the_package_satisfies_it()
    test_a_third_party_import_is_not_a_missing_module()
    test_the_real_defect_is_caught()
    test_copying_the_directory_fixes_it()
    test_the_walk_is_transitive()
    test_the_entry_point_comes_from_the_dockerfile()
    test_a_directory_copy_brings_everything_in_it()
    test_the_uvicorn_spelling_is_understood()
    test_a_third_party_import_is_not_a_local_module()
    test_a_non_python_image_is_ignored()
    test_a_tree_with_no_dockerfiles_refuses()
    test_the_repository_passes_today()
    test_the_dockerfile_list_comes_from_a_walk()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Image Module Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
