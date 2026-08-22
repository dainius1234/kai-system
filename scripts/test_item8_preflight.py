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
REACH = (REPO / "scripts" / "security"
         / "check_preflight_reachability.py")
PREFLIGHT = REPO / "scripts" / "security" / "preflight_buildkit_rawjson.py"
PARSER = REPO / "scripts" / "security" / "parse_buildkit_events.py"
AUTHORITY = REPO / "scripts" / "security" / "check_item8_authority.py"

passed = 0
failed = 0
EXPECTED_SCENARIOS = 10
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


# A daemon on which the CORROBORATOR CANNOT BE MEASURED AT ALL: the
# A/B/A probes return no target vertex, while every REQUIRED property is
# still satisfied by the builds that prove them. "We could not measure
# the corroborator" and "a required property failed" are different
# facts, and collapsing them let a corroborator stop the run through the
# back door. (D302)
ABA_BLIND_DOCKER = FLAT_DOCKER.replace(
    'ev({"vertexes": [{"digest": D, "name": "[2/2] RUN " + cmd,',
    'if "-ABA" in cmd:\n'
    '        sys.exit(0)\n'
    '    ev({"vertexes": [{"digest": D, "name": "[2/2] RUN " + cmd,')


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


# ── D302: the reachability gate at the right altitude ──────────────────

def test_reachability_follows_real_python_imports() -> None:
    """A gate that matched only literal paths could not see `import X`."""
    scenario("reachability: an ordinary import is a dependency edge")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # THE HOSTILE CHAIN GPT NAMED: workflow -> A -> B -> a plain
        # Python import of a forbidden module. No literal "scripts/....py"
        # string anywhere in it, which is exactly what the first version
        # of this gate keyed on.
        sec = REPO / "scripts" / "security"
        a = sec / "_probe_reach_a.py"
        b = sec / "_probe_reach_b.py"
        wf = REPO / ".github" / "workflows" / "_probe-reach.yml"
        try:
            a.write_text("import _probe_reach_b\n")
            b.write_text("import derive_item8_dockerfile\n")
            wf.write_text("jobs:\n  x:\n    steps:\n      - run: "
                          "python3 scripts/security/_probe_reach_a.py\n")
            r = subprocess.run(
                [sys.executable, str(REACH), "--workflow",
                 ".github/workflows/_probe-reach.yml"],
                capture_output=True, text=True, cwd=str(REPO))
            check("a three-hop import chain is FOUND", r.returncode == 1,
                  r.stdout)
            check("and the forbidden module is named",
                  "derive_item8_dockerfile" in r.stdout, r.stdout)
            check("and the hop that reached it is named",
                  "_probe_reach_b" in r.stdout, r.stdout)
            # ...and the same chain WITHOUT the forbidden import passes,
            # so the fixture is not merely detecting the probe files.
            b.write_text("import json\n")
            r = subprocess.run(
                [sys.executable, str(REACH), "--workflow",
                 ".github/workflows/_probe-reach.yml"],
                capture_output=True, text=True, cwd=str(REPO))
            check("the same chain without the import PASSES",
                  r.returncode == 0, r.stdout)
        finally:
            for f in (a, b, wf):
                f.unlink(missing_ok=True)


def test_reachability_positive_closure_is_not_empty() -> None:
    """A gate that finds nothing passes for the wrong reason."""
    scenario("reachability: the closure must actually discover the parser")
    r = subprocess.run(
        [sys.executable, str(REACH), "--expect-reachable",
         "scripts/security/parse_buildkit_events.py"],
        capture_output=True, text=True, cwd=str(REPO))
    check("the real workflow PASSES", r.returncode == 0, r.stdout)
    check("having actually traversed to the parser",
          "parse_buildkit_events.py" in r.stdout, r.stdout)
    check("and it reports its denominator",
          "reachable script(s) against" in r.stdout, r.stdout)
    # the expectation itself must be able to fail
    r = subprocess.run(
        [sys.executable, str(REACH), "--expect-reachable",
         "scripts/security/nothing_reaches_this.py"],
        capture_output=True, text=True, cwd=str(REPO))
    check("an unmet --expect-reachable REFUSES", r.returncode == 1, r.stdout)


def _positive_authority_against(source: Path, d: Path, label: str) -> None:
    """Run the positive-authority case against ONE source repository.

    THE FIXTURE ESTABLISHES ITS OWN SENTINEL-ABSENT BASELINE.

    It used to clone the live repository and treat whatever HEAD held as
    its parent. That made it valid ONLY BEFORE THE FIRST REAL
    AUTHORISATION EVENT: the moment `kai-pm/ITEM8_PREFLIGHT_GO` existed
    upstream, the clone inherited it, the fixture's write became a
    MODIFY, and the guard correctly refused `'M', not 'A'`.

    So the fixture proving the guard can say YES was invalidated by the
    act it exists to validate. Discovered in production on run
    32575388846 — authority PASSED, this calibration then failed, and
    rawjson was never measured.

    That is rule 30: qualification and mutation sharing an uncontrolled
    subject state, in the one place where the state being mutated is the
    repository itself. The repair is not "delete the file if present" —
    it is that the calibration's parent state is CONSTRUCTED here, so
    the live repository's authority state has no bearing on it. (D309)
    """
    clone = d / f"clone-{label}"
    r = subprocess.run(["git", "clone", "--quiet", str(source), str(clone)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        check(f"[{label}] the clone could be made", False, r.stderr[:200])
        return

    def g(*a):
        return subprocess.run(["git", "-C", str(clone), *a],
                              capture_output=True, text=True).stdout.strip()

    g("checkout", "-B", "claude/project-rework-plan-pgvp35")
    g("config", "user.email", "calibration@local")
    g("config", "user.name", "calibration")

    # THE BASELINE, established rather than inherited. Any authority
    # state carried in from the source is removed and COMMITTED, so the
    # envelope below is a genuine ADD against a parent that provably has
    # none. Removing and re-adding within one commit would still read as
    # a MODIFY -- the removal has to be its own commit.
    carried = [s for s in ("kai-pm/ITEM8_PREFLIGHT_GO", "kai-pm/ITEM8_GO")
               if (clone / s).exists()]
    if carried:
        g("rm", "--quiet", *carried)
        g("commit", "-m", "calibration baseline: no authority state")
    for s in ("kai-pm/ITEM8_PREFLIGHT_GO", "kai-pm/ITEM8_GO"):
        check(f"[{label}] baseline carries no {s.split('/')[-1]}",
              not (clone / s).exists(), str(carried))

    frozen = subprocess.run(
        [sys.executable, str(clone / "scripts" / "security"
                             / "check_item8_design.py"), "--quiet"],
        capture_output=True, text=True).stdout.strip()
    parent = g("rev-parse", "HEAD")
    ptree = g("rev-parse", "HEAD^{tree}")
    (clone / "kai-pm" / "ITEM8_PREFLIGHT_GO").write_text(
        f"frozen_r2={frozen}\napproved_commit={parent}\n"
        f"approved_tree={ptree}\nauthorises=preflight\n")
    g("add", "kai-pm/ITEM8_PREFLIGHT_GO")
    g("commit", "-m", "authorise the standalone measurement")

    # THE DIFF MUST BE AN ADD. Asserted here rather than left to the
    # guard's message, because this is the exact property that silently
    # stopped holding in production.
    status = g("diff", "--name-status", parent, "HEAD")
    check(f"[{label}] the envelope commit is an ADD, not a MODIFY",
          status.startswith("A\t"), status or "(empty diff)")

    guard = clone / "scripts" / "security" / "check_item8_authority.py"
    env = {**os.environ, "GITHUB_RUN_ATTEMPT": "1",
           "GITHUB_EVENT_NAME": "push"}
    r = subprocess.run(
        [sys.executable, str(guard), "--sentinel",
         "kai-pm/ITEM8_PREFLIGHT_GO", "--envelope-kind", "preflight"],
        capture_output=True, text=True, cwd=str(clone), env=env)
    check(f"[{label}] a CORRECT preflight envelope is ACCEPTED",
          r.returncode == 0, r.stdout + r.stderr)
    check(f"[{label}] and says the artefact is the reviewed one",
          "was reviewed" in r.stdout, r.stdout)
    check(f"[{label}] and reports its own denominator",
          "authority envelope across" in r.stdout, r.stdout)

    r = subprocess.run(
        [sys.executable, str(guard), "--sentinel",
         "kai-pm/ITEM8_PREFLIGHT_GO", "--envelope-kind", "experiment"],
        capture_output=True, text=True, cwd=str(clone), env=env)
    check(f"[{label}] the same envelope REFUSES on the experiment path",
          r.returncode == 1, r.stdout)

    r = subprocess.run(
        [sys.executable, str(guard), "--sentinel",
         "kai-pm/ITEM8_PREFLIGHT_GO", "--envelope-kind", "preflight"],
        capture_output=True, text=True, cwd=str(clone),
        env={**env, "GITHUB_RUN_ATTEMPT": "2"})
    check(f"[{label}] attempt 2 of the accepted case REFUSES",
          r.returncode == 1, r.stdout)


def test_a_correct_preflight_envelope_actually_PASSES() -> None:
    """A guard that always refuses satisfies every negative fixture."""
    scenario("authority: the correct preflight envelope is ACCEPTED")
    with tempfile.TemporaryDirectory() as d:
        _positive_authority_against(REPO, Path(d), "live")


def test_positive_authority_survives_a_source_that_already_authorised() -> None:
    """The hostile property production exposed, CONSTRUCTED not inherited.

    Asserting this against the live repository would only hold while the
    live repository happens to carry a sentinel — which is the ambient
    state coupling that caused the defect. So the hostile source is
    BUILT here: a repository that already contains an authority
    envelope, exactly as the real one did on run 32575388846.

    If the isolation repair is removed, this fails. (D309, rule 29)
    """
    scenario("authority: the positive case survives an already-authorised "
             "source")
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "already-authorised"
        r = subprocess.run(["git", "clone", "--quiet", str(REPO), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            check("the hostile source could be made", False, r.stderr[:200])
            return

        def gs(*a):
            return subprocess.run(["git", "-C", str(src), *a],
                                  capture_output=True, text=True).stdout.strip()

        gs("checkout", "-B", "claude/project-rework-plan-pgvp35")
        gs("config", "user.email", "calibration@local")
        gs("config", "user.name", "calibration")
        # Force the hostile condition regardless of what the live repo
        # holds right now: this source HAS an authority envelope.
        (src / "kai-pm" / "ITEM8_PREFLIGHT_GO").write_text(
            "frozen_r2=x\napproved_commit=x\napproved_tree=x\n"
            "authorises=preflight\n")
        gs("add", "kai-pm/ITEM8_PREFLIGHT_GO")
        gs("commit", "-m", "a source that has already authorised once")
        check("the hostile source really does carry a sentinel",
              (src / "kai-pm" / "ITEM8_PREFLIGHT_GO").exists())
        _positive_authority_against(src, Path(d), "already-authorised")


def test_an_unmeasurable_corroborator_is_unresolved_not_fatal() -> None:
    """The back door: aba_err used to be appended to `failures`."""
    scenario("preflight: a corroborator that cannot be measured is UNRESOLVED")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        fake = td / "aba-blind-docker"
        fake.write_text(ABA_BLIND_DOCKER)
        fake.chmod(0o755)
        rule = td / "binding-rule.json"
        code, out = run_preflight(fake, "--emit-binding-rule", str(rule))
        check("an unmeasurable A/B/A probe does NOT fail the preflight",
              code == 0, out)
        check("and the state is UNRESOLVED, not silence",
              "UNRESOLVED" in out, out)
        check("naming why it could not be measured",
              "A/B/A probe" in out, out)
        check("while the REQUIRED properties still report as met",
              "PREFLIGHT PASS IS EVIDENCE" in out, out)
        check("and the binding rule is still emitted", rule.is_file(), out)


def run_all() -> None:
    test_preflight_refuses_what_it_cannot_invoke()
    test_preflight_refuses_a_silent_daemon()
    test_unstable_digest_is_recorded_not_fatal()
    test_an_unmeasurable_corroborator_is_unresolved_not_fatal()
    test_captures_survive_only_with_keep()
    test_preflight_authority_is_a_separate_envelope()
    test_reachability_follows_real_python_imports()
    test_reachability_positive_closure_is_not_empty()
    test_a_correct_preflight_envelope_actually_PASSES()
    test_positive_authority_survives_a_source_that_already_authorised()
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
