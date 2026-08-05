"""Tests for `check_dockerfile_flags` — one character, thirteen dead steps.

`core-tests.yml` finally reached its build step and died on:

    unknown flag: --start_period (did you mean start-period?)

`HEALTHCHECK --start_period=20s` in `document-parser/Dockerfile`. That
stopped every image build, which stopped the bring-up, the live smoke,
kill-isolation, restart-persistence, the memu-graph cycle and the
sovereign boot — thirteen steps reporting nothing.

It was invisible structurally, not by bad luck: `document-parser` is one
of nineteen services in `docker-compose.minimal.yml` and not in
`docker-compose.full.yml`, and the only build step in CI built
`full.yml`. Nothing ever built that image, so nothing ever parsed that
Dockerfile.

Everything here is synthetic except the last two, which read the tree.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_dockerfile_flags as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 9
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


def test_the_real_defect_is_caught() -> None:
    scenario("real defect caught")
    text = ("HEALTHCHECK --interval=30s --timeout=5s --start_period=20s "
            "--retries=3 \\\n    CMD python -c \"pass\"\n")
    found = gate.findings_in(text, "document-parser/Dockerfile")
    check("it is reported", len(found) == 1, str(found))
    check("names the instruction", found and "HEALTHCHECK" in found[0],
          str(found))
    check("and suggests the fix",
          found and "--start-period" in found[0], str(found))


def test_correct_flags_pass() -> None:
    scenario("correct flags pass")
    text = ("HEALTHCHECK --interval=30s --timeout=5s --start-period=20s "
            "--retries=3 CMD true\n"
            "COPY --chown=app:app . /app\n"
            "FROM --platform=linux/amd64 python:3.11-slim\n")
    check("nothing reported", gate.findings_in(text, "D") == [],
          str(gate.findings_in(text, "D")))


def test_every_flagged_instruction_is_covered() -> None:
    """ADD, COPY, FROM, HEALTHCHECK and RUN all take Docker flags."""
    scenario("all instructions covered")
    for instruction in ("ADD", "COPY", "FROM", "RUN"):
        text = f"{instruction} --some_flag=1 x y\n"
        check(f"{instruction} is checked",
              len(gate.findings_in(text, "D")) == 1,
              f"{instruction}: {gate.findings_in(text, 'D')}")


def test_a_flag_after_CMD_belongs_to_the_command() -> None:
    """`HEALTHCHECK ... CMD curl --fail_fast` is the *program's* flag, and
    Docker never sees it. Flagging it would report a defect in code that
    works, which is the failure mode this repository avoids hardest."""
    scenario("post-CMD flags ignored")
    text = ("HEALTHCHECK --interval=30s CMD myprog --some_option=1\n")
    check("not reported", gate.findings_in(text, "D") == [],
          str(gate.findings_in(text, "D")))


def test_a_flag_in_a_run_command_is_still_docker_scope() -> None:
    """RUN takes `--mount`, so its flags are checked. A shell flag with an
    underscore inside a RUN would be a false positive — accepted here as
    the cost of covering `--mount`, and no such flag exists in this tree."""
    scenario("run flags checked")
    text = "RUN --mount=type=cache pip install .\n"
    check("valid mount passes", gate.findings_in(text, "D") == [],
          str(gate.findings_in(text, "D")))


def test_an_ordinary_line_is_ignored() -> None:
    scenario("ordinary lines ignored")
    for line in ("ENV FOO=bar\n", "EXPOSE 8032\n", "USER app\n",
                 "# a comment with --an_underscore\n"):
        check("ignored", gate.findings_in(line, "D") == [], line.strip())


def test_several_in_one_file_are_all_reported() -> None:
    scenario("all instances reported")
    text = ("COPY --some_flag=1 a b\n"
            "HEALTHCHECK --start_period=1s CMD true\n")
    check("both reported", len(gate.findings_in(text, "D")) == 2,
          str(gate.findings_in(text, "D")))


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, flags, files = gate.audit()
    check("no misspelled flags", findings == [], str(findings))
    check("and a real number of flags was inspected", flags > 100, str(flags))
    check("across a real number of Dockerfiles", files > 30, str(files))


def test_the_walk_finds_more_dockerfiles_than_the_full_profile_builds() -> None:
    """The point of the finding. `full.yml` builds 30 services; the tree
    holds far more Dockerfiles than that, and the one that broke CI was
    among the ones it never touched. A gate that took its file list from
    a compose profile would have missed it exactly as the build did."""
    scenario("walk beats any profile list")
    _, _, files = gate.audit()
    check("the walk covers more than any single profile builds",
          files > 30, str(files))


def run_all() -> None:
    test_the_real_defect_is_caught()
    test_correct_flags_pass()
    test_every_flagged_instruction_is_covered()
    test_a_flag_after_CMD_belongs_to_the_command()
    test_a_flag_in_a_run_command_is_still_docker_scope()
    test_an_ordinary_line_is_ignored()
    test_several_in_one_file_are_all_reported()
    test_the_repository_passes_today()
    test_the_walk_finds_more_dockerfiles_than_the_full_profile_builds()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Dockerfile Flag Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
