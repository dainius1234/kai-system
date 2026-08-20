#!/usr/bin/env python3
"""Calibration for the STANDALONE MEASUREMENT, and nothing else.

WHY THIS FILE EXISTS AT ALL
===========================

`item8-preflight.yml` ran the VERDICT-LAYER calibration as its
calibration step — and that suite holds a module-level constant naming
the six-build runner, and executes it under a fake docker in most of its
fifty scenarios.

Neither that suite nor the runner is NAMED in this file, deliberately:
`check_preflight_reachability.py` counts a mention as a reference on
purpose, so a docstring explaining the exclusion would recreate it.

So the claim that the preflight workflow "cannot reach the experiment"
was made by grepping the YAML for the runner's name and finding nothing,
while a transitive path existed one file away. **Grepping a file for a
name is the wrong altitude for a reachability question**, and I made
that argument in writing before a reviewer took it apart.

This calibration reaches the parser, the preflight and the authority
guard. It does not import, name, or execute the six-build runner, the
subject deriver, or the claim engine — and the reachability check
asserts that mechanically rather than leaving it to the next person's
care.

WHAT IT DOES NOT DO
===================

It does not re-test the verdict layer. That layer is not on the
measurement's path: the preflight writes no result row, and nothing it
produces can become evidence about the contingency. Calibrating it here
would be the second entry point arriving as a dependency instead of as
a step. (D301)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PREFLIGHT = REPO / "scripts" / "security" / "preflight_buildkit_rawjson.py"
PARSER = REPO / "scripts" / "security" / "parse_buildkit_events.py"
AUTHORITY = REPO / "scripts" / "security" / "check_item8_authority.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 5
executed: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


# A docker that emits well-formed rawjson and never varies its digest,
# so the A/B/A differential finds no structural difference.
FLAT_DOCKER = r'''#!/usr/bin/env python3
"""Well-formed rawjson, every REQUIRED property satisfied, and ONE flat
digest -- so the only thing missing is the corroborator."""
import base64, hashlib, json, os, sys
argv = sys.argv[1:]
D = "sha256:" + "7" * 64
def ev(o): print(json.dumps(o), file=sys.stderr)
if argv and argv[0] == "build":
    df = argv[argv.index("-f") + 1] if "-f" in argv else ""
    text = open(df).read()
    cmd = ""
    for line in text.replace("\\\n", " ").splitlines():
        if line.startswith("RUN "):
            cmd = line[4:]
    failing = "exit 7" in cmd
    # `cached` MOVES: a second build of the same file, without
    # --no-cache, reports cached. Kept beside the fake, not in it.
    # hashlib, NOT hash(): Python randomises hash() per process, and
    # every fake-docker invocation is a separate process -- so the state
    # file was named differently each time and `cached` could never
    # move. A fixture defect that made a required property unprovable.
    state = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),
                         ".seen-" + hashlib.sha256(cmd.encode()).hexdigest()[:12])
    seen = os.path.exists(state)
    open(state, "w").write("1")
    cached = seen and "--no-cache" not in argv
    ev({"vertexes": [{"digest": D, "name": "[2/2] RUN " + cmd,
                      "cached": cached, "started": "t0"}]})
    n = 1 if failing else 3
    for _ in range(n):
        ev({"logs": [{"vertex": D, "stream": 1,
                      "data": base64.b64encode(
                          b"PREFLIGHT-RUNTIME-LINE x\n").decode()}]})
    ev({"vertexes": [{"digest": D, "name": "[2/2] RUN " + cmd,
                      "completed": "t1",
                      "error": "process did not complete successfully: "
                               "exit code: 7" if failing else ""}]})
    sys.exit(7 if failing else 0)
sys.exit(0)
'''


def run_preflight(docker: Path, *extra: str) -> tuple[int, str]:
    p = subprocess.run(
        [sys.executable, str(PREFLIGHT), "--docker", str(docker), *extra],
        capture_output=True, text=True, cwd=str(REPO))
    return p.returncode, p.stdout + p.stderr


def test_preflight_refuses_what_it_cannot_invoke() -> None:
    scenario("preflight: a toolchain it cannot invoke is a refusal")
    with tempfile.TemporaryDirectory() as d:
        code, out = run_preflight(Path(d) / "no-such-docker")
        check("an absent docker REFUSES", code == 2, out)
        check("and says it cannot qualify what it cannot invoke",
              "cannot qualify a toolchain it cannot invoke" in out, out)


def test_preflight_refuses_a_silent_daemon() -> None:
    scenario("preflight: a daemon emitting no rawjson is a refusal")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        fake = td / "silent"
        fake.write_text("#!/bin/sh\nexit 0\n")
        fake.chmod(0o755)
        code, out = run_preflight(fake)
        check("a daemon emitting no events REFUSES", code == 1, out)
        check("and names the rawjson possibility",
              "--progress=rawjson" in out, out)
        check("and says none of the denominator was spent",
              "ZERO Item-8 builds have been spent" in out, out)


def test_unstable_digest_is_recorded_not_fatal() -> None:
    """BuildKit does not license cross-invocation digest comparison."""
    scenario("preflight: a corroborator that is unavailable is not a failure")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        fake = td / "flat-docker"
        fake.write_text(FLAT_DOCKER)
        fake.chmod(0o755)
        rule = td / "binding-rule.json"
        code, out = run_preflight(fake, "--emit-binding-rule", str(rule))
        # Every REQUIRED property holds; only the digest corroborators do
        # not. That must not stop the measurement.
        check("netmode changing no digest is NOT fatal", code == 0, out)
        check("and it is stated rather than passed over in silence",
              "Recorded, not fatal" in out, out)
        check("the binding rule is still emitted", rule.is_file(), out)
        if rule.is_file():
            r = json.loads(rule.read_text())
            check("carrying the measured answer, including False",
                  r.get("netmode_changes_vertex_digest") is False, str(r))
            check("and the required property as true",
                  r.get("full_instruction_in_vertex_name") is True, str(r))


def test_captures_survive_only_with_keep() -> None:
    scenario("preflight: evidence is retained, or its absence is announced")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        fake = td / "flat-docker"
        fake.write_text(FLAT_DOCKER)
        fake.chmod(0o755)
        work = td / "work"
        work.mkdir()
        code, out = run_preflight(fake, "--workdir", str(work), "--keep")
        check("with --keep the captures are retained",
              "raw captures retained" in out, out)
        found = list(work.rglob("*.events-stderr.jsonl"))
        check("and they exist in the workspace directory",
              len(found) >= 3, str([str(f) for f in found]))
        # Without --keep they are deleted -- which is a fact the report
        # must state, because a report that describes deleted evidence
        # as archived is how "all three captures archived" became false.
        work2 = td / "work2"
        work2.mkdir()
        code, out = run_preflight(fake, "--workdir", str(work2))
        check("without --keep the deletion is ANNOUNCED",
              "raw captures DELETED" in out, out)
        check("and the workspace directory is empty of captures",
              not list(work2.rglob("*.events-stderr.jsonl")), str(work2))


def test_preflight_authority_is_a_separate_envelope() -> None:
    scenario("authority: a preflight envelope never authorises the experiment")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        s = td / "ITEM8_PREFLIGHT_GO"
        s.write_text("frozen_r2=x\napproved_commit=y\napproved_tree=z\n"
                     "authorises=preflight\n")
        p = subprocess.run(
            [sys.executable, str(AUTHORITY), "--sentinel", str(s),
             "--envelope-kind", "experiment", "--allow-no-ci"],
            capture_output=True, text=True, cwd=str(REPO))
        check("a PREFLIGHT envelope on the EXPERIMENT path REFUSES",
              p.returncode == 1, p.stdout)
        check("and says measuring is not spending",
              "not an authorisation to spend" in p.stdout, p.stdout)
        # An envelope naming no act at all authorises nothing.
        s.write_text("frozen_r2=x\napproved_commit=y\napproved_tree=z\n")
        p = subprocess.run(
            [sys.executable, str(AUTHORITY), "--sentinel", str(s),
             "--envelope-kind", "preflight", "--allow-no-ci"],
            capture_output=True, text=True, cwd=str(REPO))
        check("an envelope naming no act REFUSES", p.returncode == 1,
              p.stdout)
        check("and names the missing field", "authorises" in p.stdout,
              p.stdout)


def run_all() -> None:
    test_preflight_refuses_what_it_cannot_invoke()
    test_preflight_refuses_a_silent_daemon()
    test_unstable_digest_is_recorded_not_fatal()
    test_captures_survive_only_with_keep()
    test_preflight_authority_is_a_separate_envelope()
    print(f"  inspected: {EXPECTED_SCENARIOS} preflight scenario(s) across "
          f"2 shipped entry points, reaching NO subject-build machinery")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Item-8 Preflight Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
