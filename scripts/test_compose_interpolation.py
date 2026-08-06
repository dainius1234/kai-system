"""Tests for `check_compose_interpolation` — a loop that never looped.

`docker-compose.sovereign.yml` gave the executor a retry-with-backoff:

    command: sh -c "set -e; for d in $BACKOFF_SCHEDULE; do
                      python app.py && exit 0 || sleep $d; done;
                    exec python app.py"

with `BACKOFF_SCHEDULE: "10 60 300"` three lines above it — in the
service's `environment:`, which is the *container's* environment.
Compose interpolates when it parses the file, before any container
exists, and knew neither `BACKOFF_SCHEDULE` nor `d`. Both became the
empty string, so the shell got `for d in ; do … done` — a loop over
nothing — and fell straight through to the bare `exec`. It has never
retried once.

The command still ran. That is why it lasted: nothing was broken, only
absent, and absence is what this repository keeps failing to see.

Directive 3 had already produced a check for this exact signature — CI
fails on `variable is not set` — but it greps the **minimal** profile's
bring-up log and this defect is in **sovereign**. A check whose scope
was smaller than its name, inside the fix for the last check whose scope
was smaller than its name.

Pointed at the tree, this gate found a third instance nobody had named:
`tailscale up --hostname=${TS_HOSTNAME}` with `TS_HOSTNAME:
sovereign-core` set only in the service environment, so the node has
never carried the name this file gives it.

Everything here is synthetic except the last two, which read the tree.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_compose_interpolation as gate  # noqa: E402

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


def doc(command: str, environment: dict = None) -> dict:
    return {"services": {"executor": {
        "command": command,
        "environment": environment or {},
    }}}


# ── the defect itself ────────────────────────────────────────────────

def test_the_real_defect_is_caught() -> None:
    scenario("real defect caught")
    found = gate.findings_in(
        doc("sh -c \"for d in $BACKOFF_SCHEDULE; do sleep $d; done\"",
            {"BACKOFF_SCHEDULE": "10 60 300"}),
        "sovereign.yml", set())
    check("both references are reported", len(found) == 2, str(found))
    check("the env-key case names the environment",
          any("own `environment:`" in f for f in found), str(found))
    check("and the loop variable is reported too",
          any("$d" in f for f in found), str(found))
    check("the fix is spelled out",
          all("$$" in f for f in found), str(found))


def test_the_fixed_form_passes() -> None:
    scenario("fixed form passes")
    found = gate.findings_in(
        doc("sh -c \"for d in $$BACKOFF_SCHEDULE; do sleep $$d; done\"",
            {"BACKOFF_SCHEDULE": "10 60 300"}),
        "sovereign.yml", set())
    check("nothing reported", found == [], str(found))


def test_the_tailscale_instance_is_caught() -> None:
    """The one the gate found that nobody had named."""
    scenario("tailscale instance")
    found = gate.findings_in(
        {"services": {"tailscale": {
            "command": "sh -c \"tailscale up --hostname=${TS_HOSTNAME}\"",
            "environment": {"TS_HOSTNAME": "sovereign-core"}}}},
        "sovereign.yml", set())
    check("reported", len(found) == 1, str(found))
    check("named as an environment key",
          found and "own `environment:`" in found[0], str(found))


# ── what must NOT be reported ────────────────────────────────────────

def test_a_default_is_not_a_defect() -> None:
    """`${NAME:-x}` means compose has something real to substitute."""
    scenario("defaults allowed")
    for spelling in ("${OLLAMA_MODEL:-qwen2.5:0.5b}", "${PORT:-8000}",
                     "${A:+set}", "${B:?required}"):
        found = gate.findings_in(doc(f"sh -c \"run {spelling}\""),
                                 "x.yml", set())
        check(f"{spelling} passes", found == [], str(found))


def test_a_name_env_example_supplies_is_not_a_defect() -> None:
    """Compose reads `.env` at run time, so this one resolves."""
    scenario("env.example supplies it")
    found = gate.findings_in(doc("sh -c \"run $KAI_SERVICE_TOKEN\""),
                             "x.yml", {"KAI_SERVICE_TOKEN"})
    check("not reported", found == [], str(found))
    # …and the same reference IS reported when nothing supplies it.
    found = gate.findings_in(doc("sh -c \"run $KAI_SERVICE_TOKEN\""),
                             "x.yml", set())
    check("but reported when nothing supplies it", len(found) == 1,
          str(found))


def test_a_service_with_no_command_is_ignored() -> None:
    scenario("no command")
    found = gate.findings_in(
        {"services": {"db": {"image": "postgres", "environment": {"A": "1"}}}},
        "x.yml", set())
    check("nothing reported", found == [], str(found))


def test_entrypoint_is_checked_as_well_as_command() -> None:
    """`ollama-pull` uses `entrypoint:`, not `command:`. A gate that read
    only one of the two would miss half the tree."""
    scenario("entrypoint checked")
    found = gate.findings_in(
        {"services": {"puller": {"entrypoint": "sh -c \"pull $MODEL\"",
                                 "environment": {"MODEL": "x"}}}},
        "x.yml", set())
    check("entrypoint is reported", len(found) == 1, str(found))
    check("and the field is named",
          found and "entrypoint" in found[0], str(found))


def test_a_list_form_command_is_joined_and_checked() -> None:
    """Both fields accept a list. A reference in one item still counts."""
    scenario("list form")
    found = gate.findings_in(
        {"services": {"puller": {
            "entrypoint": ["sh", "-c", "pull $MODEL"],
            "environment": {"MODEL": "x"}}}},
        "x.yml", set())
    check("the list form is reported", len(found) == 1, str(found))


# ── I-1: zero inputs is not a pass ───────────────────────────────────

def test_a_tree_with_no_compose_files_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, refs, files = gate.audit(Path(tmp))
        check("it fails rather than passing", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", (refs, files) == (0, 0),
              f"{refs}, {files}")


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, refs, files = gate.audit()
    check("no reference is eaten", findings == [], str(findings))
    check("across every compose file", files >= 3, str(files))
    check("and a real number of references was inspected", refs > 0,
          str(refs))


def test_the_file_list_comes_from_a_glob() -> None:
    """The point of the finding. The runtime check this replaces named
    one profile — `/tmp/bringup.log`, the minimal bring-up — and the
    defect was in sovereign. A list would have repeated that."""
    scenario("glob not list")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in ("docker-compose.yml", "docker-compose.override.yaml",
                     "docker-compose.experimental.yml"):
            (root / name).write_text("services:\n  a:\n    image: x\n")
        (root / "not-compose.yml").write_text("services: {}\n")
        found = gate.compose_files(root)
        check("every compose spelling is picked up", len(found) == 3,
              str([p.name for p in found]))
        check("including .yaml", any(p.suffix == ".yaml" for p in found),
              str([p.name for p in found]))
        check("and nothing else", all("docker-compose" in p.name
                                      for p in found),
              str([p.name for p in found]))


def run_all() -> None:
    test_the_real_defect_is_caught()
    test_the_fixed_form_passes()
    test_the_tailscale_instance_is_caught()
    test_a_default_is_not_a_defect()
    test_a_name_env_example_supplies_is_not_a_defect()
    test_a_service_with_no_command_is_ignored()
    test_entrypoint_is_checked_as_well_as_command()
    test_a_list_form_command_is_joined_and_checked()
    test_a_tree_with_no_compose_files_refuses()
    test_the_repository_passes_today()
    test_the_file_list_comes_from_a_glob()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Compose Interpolation Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
