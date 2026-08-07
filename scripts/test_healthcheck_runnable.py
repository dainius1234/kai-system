"""Tests for `check_healthcheck_runnable` — the probe that could not run.

`docker-compose.sovereign.yml` healthchecked ten services with

    test: ["CMD-SHELL", "wget -qO- http://localhost:8000/health || exit 1"]

against `python:3.11-slim` images, and no Dockerfile in this repository
installs `wget`. The check could never pass.

On 2026-08-07 the sovereign profile finally got far enough to run it.
Both core services were fine —

    sovereign-memu-core | INFO: Uvicorn running on http://0.0.0.0:8001
    sovereign-tool-gate | INFO: Application startup complete.

— and the step still failed after exactly 180s. The services were
healthy; the instrument was broken. This programme's subject, moved into
the healthcheck: a probe reporting failure over something that is right.

The same services in `minimal` and `full`, and the `HEALTHCHECK` in
their own Dockerfiles, all use the `python -c` form. Three profiles, one
fact, one copy different — and the different one had never run.

The parser tests below exist because the first draft of this gate was a
false-positive machine: it split the healthcheck on `;` with a regex,
and the command it was meant to *bless* is

    python -c "import urllib.request; urllib.request.urlopen(...)"

whose `;` is inside the quoted Python. 69 findings against 49 images, on
a tree that was already correct.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_healthcheck_runnable as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 8
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


PY_CHECK = ['CMD-SHELL', 'python -c "import urllib.request; '
            'urllib.request.urlopen(\'http://localhost:8001/health\')"']
WGET_CHECK = ['CMD-SHELL', 'wget -qO- http://localhost:8001/health || exit 1']


# ── the parser, which got it wrong first ─────────────────────────────

def test_a_semicolon_inside_quotes_is_not_a_command_separator() -> None:
    """The false-positive the first draft shipped with."""
    scenario("quoted semicolon")
    cmds = gate.commands_in(PY_CHECK)
    check("only python is invoked", cmds == ["python"], str(cmds))


def test_a_real_separator_is_honoured() -> None:
    scenario("real separator")
    cmds = gate.commands_in(WGET_CHECK)
    check("both commands are seen", cmds == ["wget", "exit"], str(cmds))


def test_a_bare_command_is_read() -> None:
    scenario("bare command")
    check("pg_isready", gate.commands_in(
        ['CMD-SHELL', 'pg_isready -U keeper -d sovereign']) == ["pg_isready"])


def test_an_empty_healthcheck_yields_nothing() -> None:
    scenario("empty healthcheck")
    check("no commands", gate.commands_in(None) == [], "")


# ── what an image provides ───────────────────────────────────────────

def test_a_python_base_provides_python_but_not_wget() -> None:
    scenario("python base")
    with tempfile.TemporaryDirectory() as tmp:
        df = Path(tmp) / "Dockerfile"
        df.write_text("FROM python:3.11-slim\nCMD [\"python\", \"app.py\"]\n")
        have = gate.provided_by(df)
        check("python is provided", "python" in have, str(sorted(have)))
        check("wget is not", "wget" not in have, str(sorted(have)))
        check("and shell builtins are", "exit" in have, str(sorted(have)))


def test_an_apt_installed_binary_counts_as_provided() -> None:
    """Nothing is hard-coded as forbidden. `wget` is reported because
    nothing provides it — install it and the finding goes away."""
    scenario("apt install counts")
    with tempfile.TemporaryDirectory() as tmp:
        df = Path(tmp) / "Dockerfile"
        df.write_text("FROM python:3.11-slim\n"
                      "RUN apt-get update && apt-get install -y wget\n")
        check("wget now provided", "wget" in gate.provided_by(df), "")


# ── I-1 and the real tree ────────────────────────────────────────────

def test_a_tree_with_no_built_services_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, checked, images = gate.audit(Path(tmp))
        check("it refuses", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", (checked, images) == (0, 0),
              f"{checked}, {images}")


def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, checked, images = gate.audit()
    check("no unrunnable healthcheck", findings == [], str(findings[:2]))
    check("across a real number of healthchecks", checked > 50, str(checked))
    check("and a real number of images", images > 40, str(images))


def run_all() -> None:
    test_a_semicolon_inside_quotes_is_not_a_command_separator()
    test_a_real_separator_is_honoured()
    test_a_bare_command_is_read()
    test_an_empty_healthcheck_yields_nothing()
    test_a_python_base_provides_python_but_not_wget()
    test_an_apt_installed_binary_counts_as_provided()
    test_a_tree_with_no_built_services_refuses()
    test_the_repository_passes_today()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Healthcheck Runnable Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
