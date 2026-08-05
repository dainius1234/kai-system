"""Tests for `check_dockerfile_context` — the profile that could not build.

`COPY` is relative to the build **context**, not to the Dockerfile, so the
same Dockerfile is correct under one profile and broken under another.
This repository had both, and only found out when CI first built a
profile other than `full.yml`:

    docker-compose.full.yml        30 builds,  0 broken COPY
    docker-compose.minimal.yml     32 builds, 10 broken COPY
    docker-compose.sovereign.yml   12 builds, 39 broken COPY

Every build service in the sovereign profile used `build: ./x`, making
the context the service directory, while the Dockerfiles use
root-relative paths. It looked healthy because the sovereign boot step
runs `up -d` without `--build` and reused images `full.yml` had built.

Synthetic except the last three, which read the tree.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_dockerfile_context as gate  # noqa: E402

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


# ── resolving the context ────────────────────────────────────────────

def test_the_short_form_makes_the_directory_the_context() -> None:
    """`build: ./tool-gate` — the spelling that broke the whole sovereign
    profile, because the Dockerfiles use root-relative COPY."""
    scenario("short form context")
    check("context is the directory",
          gate.build_target({"build": "./tool-gate"}) ==
          ("tool-gate", "tool-gate/Dockerfile"),
          str(gate.build_target({"build": "./tool-gate"})))


def test_the_long_form_keeps_the_declared_context() -> None:
    scenario("long form context")
    cfg = {"build": {"context": ".", "dockerfile": "tool-gate/Dockerfile"}}
    check("context is the root",
          gate.build_target(cfg) == (".", "tool-gate/Dockerfile"),
          str(gate.build_target(cfg)))


def test_an_image_only_service_has_no_build_target() -> None:
    scenario("image-only has no target")
    check("nothing to check", gate.build_target({"image": "redis:7"}) is None,
          "")


# ── reading COPY ─────────────────────────────────────────────────────

def test_copy_sources_exclude_flags_and_the_target() -> None:
    scenario("copy parsing")
    text = ("COPY --chown=app:app a.py b.py /app/\n"
            "COPY common/ ./common/\n"
            "RUN echo COPY not-a-copy\n")
    got = gate.copy_sources(text)
    check("two sources from the first COPY plus one from the second",
          got == [(1, "a.py"), (1, "b.py"), (2, "common/")], str(got))


def test_a_run_line_mentioning_copy_is_not_a_copy() -> None:
    scenario("run line ignored")
    check("ignored", gate.copy_sources("RUN cp COPY x y\n") == [], "")


# ── the two rules ────────────────────────────────────────────────────

def _tree(root: Path, files, compose: str) -> None:
    for f in files:
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
    (root / "docker-compose.full.yml").write_text(compose, encoding="utf-8")
    for other in ("docker-compose.minimal.yml", "docker-compose.sovereign.yml"):
        (root / other).write_text("services: {}\n", encoding="utf-8")


def _run(root: Path):
    return gate.audit(root)


def test_a_source_outside_the_context_is_reported() -> None:
    scenario("source outside context")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile", "svc/app.py"],
              "services:\n  s:\n    build: ./svc\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY svc/app.py ./\n", encoding="utf-8")
        findings, checked, builds = _run(root)
        check("it is reported", len(findings) == 1, str(findings))
        check("and names the context",
              findings and "'svc'" in findings[0], str(findings))
        check("one source checked", checked == 1, str(checked))
        check("one build read", builds == 1, str(builds))


def test_the_same_dockerfile_passes_under_the_right_context() -> None:
    """The whole point: the file is not wrong, the pairing is."""
    scenario("same file right context")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile", "svc/app.py"],
              "services:\n  s:\n    build:\n      context: .\n"
              "      dockerfile: svc/Dockerfile\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY svc/app.py ./\n", encoding="utf-8")
        findings, _, _ = _run(root)
        check("no finding", findings == [], str(findings))


def test_a_parent_escape_is_reported_under_any_context() -> None:
    """`COPY ../../common` is rejected by Docker whatever the context, so
    three images here could not build in any configuration."""
    scenario("parent escape")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile"], "services:\n  s:\n    build: ./svc\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY ../../common /app/common\n", encoding="utf-8")
        findings, _, _ = _run(root)
        check("it is reported", len(findings) == 1, str(findings))
        check("and says it escapes",
              findings and "escapes" in findings[0], str(findings))


def test_a_glob_is_skipped_rather_than_guessed() -> None:
    """Resolving a glob needs Docker's own matching. A wrong answer would
    report a defect against a working build, which is the failure mode
    this repository avoids hardest."""
    scenario("glob skipped")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile"], "services:\n  s:\n    build: ./svc\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY *.py ./\n", encoding="utf-8")
        findings, checked, _ = _run(root)
        check("not reported", findings == [], str(findings))
        check("and not counted as checked", checked == 0, str(checked))


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, checked, builds = gate.audit()
    check("every COPY resolves", findings == [], str(findings))
    check("and a real number was checked", checked > 200, str(checked))
    check("across every profile's builds", builds > 60, str(builds))


# ── the third rule: .dockerignore ────────────────────────────────────

def test_a_copy_source_excluded_by_dockerignore_is_reported() -> None:
    """Adding a `.dockerignore` is a large win and a new way to break
    every build at once. An excluded path fails at COPY, not at parse,
    so it would surface twenty minutes into a run — which is exactly
    where this class always surfaces."""
    scenario("dockerignore exclusion")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile", "common/x.py"],
              "services:\n  s:\n    build:\n      context: .\n"
              "      dockerfile: svc/Dockerfile\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY common/ ./common/\n", encoding="utf-8")
        (root / ".dockerignore").write_text("common\n", encoding="utf-8")
        findings, _, _ = _run(root)
        check("it is reported", len(findings) == 1, str(findings))
        check("and names the pattern",
              findings and "'common'" in findings[0], str(findings))


def test_a_negation_rescues_the_source() -> None:
    scenario("dockerignore negation")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile", ".env.example"],
              "services:\n  s:\n    build:\n      context: .\n"
              "      dockerfile: svc/Dockerfile\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY .env.example ./\n", encoding="utf-8")
        (root / ".dockerignore").write_text(
            ".env.*\n!.env.example\n", encoding="utf-8")
        findings, _, _ = _run(root)
        check("the negation is honoured", findings == [], str(findings))


def test_no_dockerignore_is_not_a_finding() -> None:
    """A repository may legitimately have none; absence only matters when
    a COPY collides with a pattern."""
    scenario("no dockerignore")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root, ["svc/Dockerfile", "common/x.py"],
              "services:\n  s:\n    build:\n      context: .\n"
              "      dockerfile: svc/Dockerfile\n")
        (root / "svc/Dockerfile").write_text(
            "FROM x\nCOPY common/ ./common/\n", encoding="utf-8")
        findings, _, _ = _run(root)
        check("nothing reported", findings == [], str(findings))


def test_the_repository_dockerignore_excludes_nothing_it_copies() -> None:
    """110 COPY sources under `context: .`, checked against the real
    `.dockerignore`. This is what makes that file safe to keep."""
    scenario("real dockerignore is safe")
    findings, _, _ = gate.audit()
    clashes = [f for f in findings if "excluded from the context" in f]
    check("no COPY source is excluded", clashes == [], str(clashes))


def run_all() -> None:
    test_the_short_form_makes_the_directory_the_context()
    test_the_long_form_keeps_the_declared_context()
    test_an_image_only_service_has_no_build_target()
    test_copy_sources_exclude_flags_and_the_target()
    test_a_run_line_mentioning_copy_is_not_a_copy()
    test_a_source_outside_the_context_is_reported()
    test_the_same_dockerfile_passes_under_the_right_context()
    test_a_parent_escape_is_reported_under_any_context()
    test_a_glob_is_skipped_rather_than_guessed()
    test_the_repository_passes_today()
    test_a_copy_source_excluded_by_dockerignore_is_reported()
    test_a_negation_rescues_the_source()
    test_no_dockerignore_is_not_a_finding()
    test_the_repository_dockerignore_excludes_nothing_it_copies()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Dockerfile Context Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
