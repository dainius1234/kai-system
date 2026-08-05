"""Tests for `check_service_tokens` — every auth-enforcing service is given its token.

`require_service_auth` fails closed, so a service that enforces it and is
never given `KAI_SERVICE_TOKEN` is not an open endpoint — it is a service
that answers **503 to every protected call**, and the symptom appears at
the *caller*, which reads it as the callee being broken rather than
unconfigured.

Found auditing G-07's closure on 2026-08-05. The record said the token
was "wired into 8 service blocks across all three compose profiles"; it
was wired into 8 blocks in total, split 3/1/4, and `executor` — which
runs `POST /execute`, the endpoint that actually executes tools — had
none in `full` or `sovereign`.

Every assertion here runs against synthetic compose files, so none of
them depends on the repository being in any particular state. The two
that do read the real tree say so.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_service_tokens as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 10
executed: list = []


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


def _profile(root: Path, name: str, services: str) -> Path:
    path = root / name
    path.write_text("services:\n" + services, encoding="utf-8")
    return path


def _service(root: Path, directory: str, entry: str, enforces: bool) -> None:
    """Create a service directory with a Dockerfile and an entry module."""
    d = root / directory
    d.mkdir(parents=True, exist_ok=True)
    body = ("from fastapi import Depends\n"
            "@app.post('/x', dependencies=[Depends(require_service_auth('x'))])\n"
            "def handler():\n    return {}\n") if enforces else (
        "@app.post('/x')\ndef handler():\n    return {}\n")
    (d / entry).write_text(body, encoding="utf-8")
    (d / "Dockerfile").write_text(
        f'FROM python:3.11\nCMD ["python", "{entry}"]\n', encoding="utf-8")


def _run(root: Path, files):
    """Point the gate at a synthetic tree and collect its verdict."""
    original_repo, original_files = gate.REPO, gate.COMPOSE_FILES
    try:
        gate.REPO = root
        gate.COMPOSE_FILES = tuple(files)
        return gate.audit()
    finally:
        gate.REPO, gate.COMPOSE_FILES = original_repo, original_files


# ── The rule ─────────────────────────────────────────────────────────

def test_a_service_without_its_token_is_reported() -> None:
    scenario("missing token is reported")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _service(root, "executor", "app.py", enforces=True)
        _profile(root, "a.yml", '''  executor:
    build: ./executor
    environment:
      OTHER: "x"
''')
        findings, inspected, profiles = _run(root, ["a.yml"])
        check("it is inspected", inspected == 1, str(inspected))
        check("it is reported", len(findings) == 1, str(findings))
        check("the message names the service",
              findings and "executor" in findings[0], str(findings))
        check("and names the file that enforces auth",
              findings and "app.py" in findings[0], str(findings))


def test_a_service_with_its_token_passes() -> None:
    scenario("token present passes")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _service(root, "executor", "app.py", enforces=True)
        _profile(root, "a.yml", '''  executor:
    build: ./executor
    environment:
      KAI_SERVICE_TOKEN: "${KAI_SERVICE_TOKEN:-}"
''')
        findings, inspected, _ = _run(root, ["a.yml"])
        check("inspected", inspected == 1, str(inspected))
        check("no finding", findings == [], str(findings))


def test_an_empty_token_still_passes() -> None:
    """Empty means "not configured", which the code treats as fail-closed.
    Requiring a *value* here would push someone to invent one."""
    scenario("empty token is configuration, not absence")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _service(root, "executor", "app.py", enforces=True)
        _profile(root, "a.yml", '''  executor:
    build: ./executor
    environment:
      KAI_SERVICE_TOKEN: ""
''')
        findings, _, _ = _run(root, ["a.yml"])
        check("declared-but-empty is accepted", findings == [], str(findings))


def test_a_service_that_does_not_enforce_is_not_required_to_have_one() -> None:
    """Otherwise the gate would demand a token from every service in the
    stack, and a gate that flags innocent services gets ignored."""
    scenario("non-enforcing service is not flagged")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _service(root, "quiet", "app.py", enforces=False)
        _profile(root, "a.yml", '''  quiet:
    build: ./quiet
    environment:
      OTHER: "x"
''')
        findings, inspected, _ = _run(root, ["a.yml"])
        check("not inspected", inspected == 0, str(inspected))
        check("not reported", findings == [], str(findings))


# ── Granularity: the false positives that shaped this ────────────────

def test_two_services_from_one_directory_are_told_apart() -> None:
    """The first version asked "does any file under this directory
    enforce auth", which produced four false findings out of eight:
    `agentic-introspect` runs `introspect_app.py` (no protected routes)
    while `agentic/app.py` has them, and `avatar-service`/`tts-service`
    build from `output/avatar` and `output/tts` while only
    `output/notify/app.py` enforces anything."""
    scenario("sibling entry points are told apart")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "agentic"
        d.mkdir()
        _service(root, "agentic", "app.py", enforces=True)
        (d / "introspect_app.py").write_text(
            "@app.post('/x')\ndef h():\n    return {}\n", encoding="utf-8")
        (d / "Dockerfile.introspect").write_text(
            'FROM python:3.11\nCMD ["python", "introspect_app.py"]\n',
            encoding="utf-8")
        _profile(root, "a.yml", '''  agentic:
    build:
      context: .
      dockerfile: agentic/Dockerfile
    environment:
      KAI_SERVICE_TOKEN: "${KAI_SERVICE_TOKEN:-}"
  agentic-introspect:
    build:
      context: .
      dockerfile: agentic/Dockerfile.introspect
    environment:
      OTHER: "x"
''')
        findings, inspected, _ = _run(root, ["a.yml"])
        check("only the enforcing entry point is inspected",
              inspected == 1, str(inspected))
        check("the sibling is not falsely flagged", findings == [], str(findings))


def test_every_build_spelling_is_understood() -> None:
    """`build: ./x`, `{context: ./x}` and `{context: ., dockerfile:
    x/Dockerfile}` are all in use. Understanding only the first two made
    this check inspect 2 service definitions instead of 16 and report
    PASS for the profile holding the defect."""
    scenario("all three build spellings")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("one", "two", "three"):
            _service(root, name, "app.py", enforces=True)
        _profile(root, "a.yml", '''  one:
    build: ./one
    environment: {OTHER: "x"}
  two:
    build:
      context: ./two
    environment: {OTHER: "x"}
  three:
    build:
      context: .
      dockerfile: three/Dockerfile
    environment: {OTHER: "x"}
''')
        findings, inspected, _ = _run(root, ["a.yml"])
        check("all three spellings resolve", inspected == 3, str(inspected))
        check("all three are reported", len(findings) == 3, str(findings))


def test_a_uvicorn_entry_point_resolves() -> None:
    """`CMD ["uvicorn", "app:app", ...]` names a module, not a file."""
    scenario("uvicorn entry point")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "vault-sync"
        d.mkdir()
        (d / "app.py").write_text(
            "@app.post('/x', dependencies=[Depends(require_service_auth('x'))])\n"
            "def h():\n    return {}\n", encoding="utf-8")
        (d / "Dockerfile").write_text(
            'FROM python:3.11\nCMD ["uvicorn", "app:app", "--port", "8047"]\n',
            encoding="utf-8")
        _profile(root, "a.yml", '''  vault-sync:
    build: ./vault-sync
    environment: {OTHER: "x"}
''')
        findings, inspected, _ = _run(root, ["a.yml"])
        check("module:attr resolves to a file", inspected == 1, str(inspected))
        check("and is reported", len(findings) == 1, str(findings))


# ── Failing closed ───────────────────────────────────────────────────

def test_a_missing_profile_is_a_finding() -> None:
    """I-1. A profile that vanished is exactly when this is most needed."""
    scenario("missing profile fails closed")
    with tempfile.TemporaryDirectory() as tmp:
        findings, _, _ = _run(Path(tmp), ["absent.yml"])
        check("absence is reported", len(findings) == 1, str(findings))
        check("and says why",
              findings and "missing" in findings[0], str(findings))


def test_an_unresolvable_entry_point_is_a_finding() -> None:
    """Unknown is not the same as safe. A service whose entry point
    cannot be resolved is reported rather than skipped."""
    scenario("unresolvable entry point fails closed")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "odd"
        d.mkdir()
        (d / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
        _profile(root, "a.yml", '''  odd:
    build: ./odd
    environment: {OTHER: "x"}
''')
        findings, _, _ = _run(root, ["a.yml"])
        check("undecidable service is reported", len(findings) == 1, str(findings))
        check("and is described as unresolvable",
              findings and "unresolvable" in findings[0], str(findings))


# ── The real tree ────────────────────────────────────────────────────

def test_the_repository_currently_passes() -> None:
    scenario("the repository passes today")
    findings, inspected, profiles = gate.audit()
    check("every auth-enforcing service has its token",
          findings == [], str(findings))
    check("and something was actually inspected", inspected > 0, str(inspected))
    check("across every declared profile", profiles == 3, str(profiles))


def run_all() -> None:
    test_a_service_without_its_token_is_reported()
    test_a_service_with_its_token_passes()
    test_an_empty_token_still_passes()
    test_a_service_that_does_not_enforce_is_not_required_to_have_one()
    test_two_services_from_one_directory_are_told_apart()
    test_every_build_spelling_is_understood()
    test_a_uvicorn_entry_point_resolves()
    test_a_missing_profile_is_a_finding()
    test_an_unresolvable_entry_point_is_a_finding()
    test_the_repository_currently_passes()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Service Token Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
