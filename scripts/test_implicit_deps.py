"""Tests for `check_implicit_deps` — two containers that never started.

`document-parser` crash-looped in every deployment:

    RuntimeError: Form data requires "python-multipart" to be installed.

raised by `@app.post("/parse")` with `file: UploadFile = File(...)`.
FastAPI needs the package to *build* the route, so it raises at import.

So did the **dashboard**, with four such routes and the same omission.
That is the operator's entire interface, and it had never started.

Why no test could have caught it
--------------------------------

`scripts/test_dashboard.py` passes and always has. CI installs every
`requirements.txt` in the tree into one environment, and four other
services list `python-multipart`, so it is present by the time the
dashboard's tests run — installed by somebody else. A missing
*per-service* dependency is structurally invisible to a suite that runs
against the union of all of them.

`document-parser` never even blocked the bring-up: `dashboard` waits on
it with `condition: service_started`, which a container in a restart
loop satisfies.

Calibrated against both real files: reverting either produces its
finding, and the fixed tree produces none.

Everything here is synthetic except the last two, which read the tree.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_implicit_deps as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 11
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


def found(src: str, reqs: str) -> list:
    return gate.findings_in({"app.py": src}, reqs, "svc")


# ── the defect itself ────────────────────────────────────────────────

REAL = ('from fastapi import FastAPI, File, UploadFile\n'
        '@app.post("/parse")\n'
        'async def parse(file: UploadFile = File(...)):\n'
        '    return {}\n')


def test_the_real_defect_is_caught() -> None:
    scenario("real defect caught")
    f = found(REAL, "fastapi>=0.110.0\nuvicorn>=0.29.0\n")
    check("it is reported", len(f) == 1, str(f))
    check("the package is named",
          f and "python-multipart" in f[0], str(f))
    check("and it says the container never starts",
          f and "never starts" in f[0], str(f))
    check("and why no test sees it",
          f and "one environment" in f[0], str(f))


def test_declaring_it_silences_it() -> None:
    scenario("declared passes")
    reqs = "fastapi>=0.110.0\npython-multipart>=0.0.26\n"
    check("nothing reported", found(REAL, reqs) == [], str(found(REAL, reqs)))


def test_every_multipart_spelling_triggers_it() -> None:
    """`UploadFile`, `File(...)` and `Form(...)` all need the package.
    Knowing one of three would make this gate pass over two thirds of
    its own subject."""
    scenario("all multipart spellings")
    for usage in ("x: UploadFile", "f = File(...)", "n = Form(...)"):
        src = f"from fastapi import *\ndef r():\n    {usage}\n"
        check(f"`{usage}` is caught", len(found(src, "fastapi\n")) == 1,
              f"{usage}: {found(src, 'fastapi')}")


def test_the_other_four_rules_fire() -> None:
    """Same class, different library — an import statement names none of
    them, which is exactly why they go missing."""
    scenario("other implicit rules")
    for usage, package in (("EmailStr", "email-validator"),
                           ("SessionMiddleware", "itsdangerous"),
                           ("ORJSONResponse", "orjson"),
                           ("UJSONResponse", "ujson")):
        src = f"x = {usage}\n"
        f = found(src, "fastapi\n")
        check(f"{usage} needs {package}",
              len(f) == 1 and package in f[0], f"{usage}: {f}")


# ── name normalisation, where a near-miss reads as absent ────────────

def test_requirement_names_are_normalised() -> None:
    """PEP 503: `python_multipart`, `Python-Multipart` and
    `python-multipart` are one package. Treating them as three would
    report a defect in a service that is correctly specified."""
    scenario("names normalised")
    for spelling in ("python-multipart>=0.0.26", "python_multipart==0.0.26",
                     "Python-Multipart", "python.multipart >= 0.0.9"):
        check(f"`{spelling}` counts as declared",
              found(REAL, f"fastapi\n{spelling}\n") == [], spelling)


def test_comments_and_flags_are_not_package_names() -> None:
    scenario("comments ignored")
    reqs = ("# python-multipart is deliberately absent\n"
            "-r ../base.txt\n"
            "fastapi\n")
    f = found(REAL, reqs)
    check("a commented-out package does not count as declared",
          len(f) == 1, str(f))


def test_a_service_that_uses_none_of_them_is_silent() -> None:
    scenario("unrelated service silent")
    src = "from fastapi import FastAPI\napp = FastAPI()\n"
    check("nothing reported", found(src, "fastapi\n") == [], str(found(src, "fastapi\n")))


# ── I-1: zero inputs is not a pass ───────────────────────────────────

def test_a_tree_with_no_services_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, dirs, rules = gate.audit(Path(tmp))
        check("it fails rather than passing", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", dirs == 0, str(dirs))


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, dirs, rules = gate.audit()
    check("no undeclared implicit dependency", findings == [], str(findings))
    check("across a real number of services", dirs > 30, str(dirs))
    check("with every rule applied", rules == len(gate.RULES), str(rules))


def test_a_dev_only_requirements_file_still_counts() -> None:
    """The meta-check's I-1 catch on this gate's own first draft: it
    returned directories, then skipped any without a `requirements.txt`.
    A service pinning its deps in `requirements-dev.txt` alone would
    have been read as clean without ever being read at all."""
    scenario("dev-only requirements")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        svc = root / "svc"
        svc.mkdir()
        (svc / "requirements-dev.txt").write_text(
            "fastapi\npython-multipart>=0.0.26\n")
        (svc / "app.py").write_text(REAL)
        findings, dirs, _ = gate.audit(root)
        check("the directory is surveyed", dirs == 1, str(dirs))
        check("and its dev-only pin counts as declared",
              findings == [], str(findings))
        # …and the same directory without the pin is reported.
        (svc / "requirements-dev.txt").write_text("fastapi\n")
        findings, _, _ = gate.audit(root)
        check("while omitting it is still caught", len(findings) == 1,
              str(findings))


def test_the_service_list_comes_from_a_walk() -> None:
    """`document-parser` is exactly the service a hand-written list
    forgets — it is also the one `docker-compose.full.yml` forgot."""
    scenario("walk not list")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel in ("a", "b", "nested/c"):
            (root / rel).mkdir(parents=True)
            (root / rel / "requirements.txt").write_text("fastapi\n")
        (root / "requirements.txt").write_text("fastapi\n")   # repo root
        (root / "_archive").mkdir()
        (root / "_archive" / "requirements.txt").write_text("fastapi\n")
        got = gate.service_dirs(root)
        names = sorted(p.name for p in got)
        check("it recurses into nested services", "c" in names, str(names))
        check("skips the repo root's own file", len(got) == 3, str(names))
        check("and skips _archive", "_archive" not in names, str(names))


def run_all() -> None:
    test_the_real_defect_is_caught()
    test_declaring_it_silences_it()
    test_every_multipart_spelling_triggers_it()
    test_the_other_four_rules_fire()
    test_requirement_names_are_normalised()
    test_comments_and_flags_are_not_package_names()
    test_a_service_that_uses_none_of_them_is_silent()
    test_a_tree_with_no_services_refuses()
    test_the_repository_passes_today()
    test_a_dev_only_requirements_file_still_counts()
    test_the_service_list_comes_from_a_walk()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Implicit Dependency Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
