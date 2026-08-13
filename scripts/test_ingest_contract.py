#!/usr/bin/env python3
"""Calibration for the KAI-GATE-050 contract analyser.

The load-bearing assertions:

1. **HTTP 200 must not be the success predicate.** The 200 is the thing
   under suspicion. A detector that treats it as evidence of success
   agrees with the defect and can never find it (I-8).

2. **Known-positive and known-negative, both real.** `/graph/ingest`
   drives two cognee pipelines — `add`, which completes, and `cognify`,
   which failed in run 9 — so one request supplies a genuinely completed
   pipeline AND a genuinely failed one. Neither control is synthetic.
   The detector must classify them independently, or a CPU runner that
   never produces a successful cognify would leave it uncalibratable.

3. **A pipeline with no terminal marker is not a completed pipeline.**
   cognee swallows `PipelineRunFailedError` without re-raising, so a
   pipeline can end with neither `completed` nor `errored`. Reading that
   silence as success would reproduce the exact defect inside the
   instrument.

4. **The collector's command lines must be ones the probe accepts.** Run
   8's whole loss was an argv mismatch nobody checked. Same check here,
   derived from the collector's text, answered by the probe's own parser.

5. **Unmeasured must not read as clean.** An observation that established
   neither side exits non-zero.
"""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import summarise_ingest_contract as s  # noqa: E402
from scripts.security import probe_ingest_contract as p  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 8
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


# Shapes taken from run 31733359906's real logs, trimmed.
ADD_COMPLETED = """\
2026-08-13T19:00:19.661737 [INFO] Coroutine task completed: `ingest_data` [run_tasks_base]
2026-08-13T19:00:19.662065 [INFO] Pipeline run started: `93e2b017-80db-5af3-9404-20127e2539a6` [run_tasks_with_telemetry()]
2026-08-13T19:00:19.662065 [INFO] Pipeline run completed: `93e2b017-80db-5af3-9404-20127e2539a6` [run_tasks_with_telemetry()]
"""

COGNIFY_FAILED = """\
2026-08-13T19:00:23.191448 [INFO] Pipeline run started: `65e9ef5d-e908-594c-a079-d0e5bdbd90ec` [run_tasks_with_telemetry()]
2026-08-13T19:06:51.744032 [error] PipelineRunFailedError: Pipeline run failed. Data item could not be processed. (Status code: 422) [cognee.shared.logging_utils]
"""

COGNIFY_COMPLETED = """\
2026-08-13T19:00:23.191448 [INFO] Pipeline run started: `65e9ef5d-e908-594c-a079-d0e5bdbd90ec` [run_tasks_with_telemetry()]
2026-08-13T19:04:23.000000 [INFO] Pipeline run completed: `65e9ef5d-e908-594c-a079-d0e5bdbd90ec` [run_tasks_with_telemetry()]
"""

OK_200 = ("ENTERED  http-post  source_id=kai-gate-050-obs-1 wall=... budget=900.0s\n"
          "RETURNED http-post  status=200 elapsed=396.3s\n"
          'BODY {"status":"ingested","source_id":"kai-gate-050-obs-1","data_id":null}\n')

ERR_502 = ("ENTERED  http-post  source_id=kai-gate-050-obs-1 wall=... budget=900.0s\n"
           "RETURNED http-post  status=502 elapsed=396.3s\n"
           'BODY {"detail":"graph ingest failed: ..."}\n')


def stage_dir(root, files):
    d = Path(root)
    for name, body in files.items():
        (d / name).write_text(body)
    return d


def run(d):
    proc = subprocess.run(
        [sys.executable, "scripts/security/summarise_ingest_contract.py",
         "--stage-logs", str(d)],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    return proc.stdout, proc.returncode


# ── 1. the correlation, all four cells ───────────────────────────────

def test_failed_pipeline_plus_200_is_the_finding() -> None:
    """The known-negative: run 9's actual shape."""
    scenario("failed + 200 = finding")
    verdict, why = s.classify({"p1": "COMPLETED", "p2": "NO-TERMINAL-MARKER"},
                              ["Pipeline run failed. Data item could not be processed. (Status code: 422)"],
                              200)
    check("verdict is SUCCESS-SHAPED FAILURE",
          verdict == "SUCCESS-SHAPED FAILURE", f"{verdict}: {why}")
    check("and it names the incomplete pipeline", "p2" in why, why)


def test_failed_pipeline_plus_502_is_correct_behaviour() -> None:
    """The required behaviour must NOT be reported as a defect — a gate
    with false positives sends people to break working code (R4)."""
    scenario("failed + 502 = consistent")
    verdict, why = s.classify({"p1": "ERRORED"}, ["boom"], 502)
    check("failure propagated is CONSISTENT", verdict == "CONSISTENT",
          f"{verdict}: {why}")
    check("and it says the failure propagated", "propagated" in why, why)


def test_completed_pipelines_plus_200_is_consistent() -> None:
    """The known-positive."""
    scenario("completed + 200 = consistent")
    verdict, _ = s.classify({"p1": "COMPLETED", "p2": "COMPLETED"}, [], 200)
    check("genuine success is CONSISTENT", verdict == "CONSISTENT", verdict)


def test_completed_pipelines_plus_502_is_inverted() -> None:
    """The fourth cell. Not the finding, but not silence either."""
    scenario("completed + 502 = inverted")
    verdict, _ = s.classify({"p1": "COMPLETED"}, [], 502)
    check("a failure over a clean pipeline is INVERTED",
          verdict == "INVERTED", verdict)


def test_http_200_alone_never_establishes_success() -> None:
    """THE load-bearing assertion. If a 200 with no internal evidence
    reads as success, the detector agrees with the defect."""
    scenario("200 alone is not success")
    verdict, why = s.classify({}, [], 200)
    check("200 with no pipeline status is UNMEASURED",
          verdict == "UNMEASURED", f"{verdict}: {why}")
    check("and says the internal outcome is unknown",
          "internal outcome is unknown" in why, why)
    # and the converse: no HTTP result is equally unmeasured
    check("no HTTP result is UNMEASURED",
          s.classify({"p1": "COMPLETED"}, [], None)[0] == "UNMEASURED", "")


def test_a_pipeline_with_no_terminal_marker_is_not_completed() -> None:
    """cognee swallows PipelineRunFailedError, so silence is a real
    outcome and must not be read as success."""
    scenario("no terminal marker is not success")
    states, failures = s.pipeline_states(COGNIFY_FAILED)
    check("the started pipeline is tracked", len(states) == 1, str(states))
    check("its state is NO-TERMINAL-MARKER",
          list(states.values()) == ["NO-TERMINAL-MARKER"], str(states))
    check("the failure message is captured", len(failures) == 1, str(failures))
    check("and it carries the 422",
          "422" in failures[0] if failures else False, str(failures))
    # the known-positive parses too
    ok, no_failures = s.pipeline_states(ADD_COMPLETED)
    check("a completed pipeline reads as COMPLETED",
          list(ok.values()) == ["COMPLETED"], str(ok))
    check("with no failure messages", no_failures == [], str(no_failures))


# ── 2. end to end, both directions ───────────────────────────────────

def test_the_gate_fails_on_the_finding_and_passes_on_clean() -> None:
    scenario("gate both directions")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "ingest-1.log": OK_200,
            "cognee-log-1.log": ADD_COMPLETED + COGNIFY_FAILED,
            "ingest-2.log": OK_200,
            "cognee-log-2.log": ADD_COMPLETED + COGNIFY_FAILED,
        })
        out, rc = run(d)
        flat = " ".join(out.split())
        check("exits non-zero on the finding", rc != 0, f"rc={rc}")
        check("names the invariant violated",
              "THE INVARIANT IS VIOLATED" in out, out[-800:])
        check("reports the denominator", "DENOMINATOR: 2 observation(s)" in out,
              out[-600:])
        check("says it reproduced in both stacks",
              "Reproduced in 2/2" in flat, flat[-500:])
        check("refuses to authorise a remedy",
              "does NOT authorise a remedy" in flat, flat[-400:])
        check("and warns about the greener-acceptance trap",
              "greener" in flat, flat[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "ingest-1.log": OK_200,
            "cognee-log-1.log": ADD_COMPLETED + COGNIFY_COMPLETED,
        })
        out, rc = run(d)
        check("exits zero when every pipeline completed", rc == 0, out[-600:])
        check("but does not claim closure",
              "not closure" in out, out[-400:])
    # partial reproduction must not read as deterministic
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "ingest-1.log": OK_200,
            "cognee-log-1.log": ADD_COMPLETED + COGNIFY_FAILED,
            "ingest-2.log": OK_200,
            "cognee-log-2.log": ADD_COMPLETED + COGNIFY_COMPLETED,
        })
        out, rc = run(d)
        flat = " ".join(out.split())
        check("1-of-2 is not called deterministic",
              "NOT deterministic" in flat, flat[-500:])
        check("and still fails", rc != 0, f"rc={rc}")
    # THE KNOWN-POSITIVE FOR THE GATE ITSELF: a failed pipeline that DID
    # propagate as a 5xx is the required behaviour, and the gate must not
    # fire on it. A gate with false positives sends people to break
    # working code (R4).
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "ingest-1.log": ERR_502,
            "cognee-log-1.log": ADD_COMPLETED + COGNIFY_FAILED,
        })
        out, rc = run(d)
        check("a failure reported as a failure exits zero", rc == 0, out[-700:])
        check("and is not called a success-shaped failure",
              "SUCCESS-SHAPED FAILURE" not in out, out[-700:])
    # nothing collected at all
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(Path(tmp))
        check("an empty directory fails closed", rc != 0, f"rc={rc}")
        check("and says unmeasured is not clean",
              "not the same as clean" in out, out[:600])


# ── 3. the run-8 lesson, applied here ────────────────────────────────

COLLECTOR = REPO / "scripts" / "security" / "measure_ingest_contract.sh"
_INVOCATION = re.compile(
    r"python\s+-\s+([^\n<]*)<\s*scripts/security/probe_ingest_contract\.py")


def test_every_collector_invocation_is_one_the_probe_accepts() -> None:
    """Derived from the collector's text, answered by the probe's parser.
    Run 8 lost a whole 900s observation to an argv mismatch that this
    shape of check costs nothing to prevent."""
    scenario("collector invocations accepted")
    # Join shell line-continuations FIRST. A real invocation spans three
    # physical lines, and a regex that cannot cross a newline silently
    # finds fewer invocations than exist — a scope smaller than the
    # check's name implies (R5), which is the defect this whole check is
    # meant to catch, one level up.
    text = COLLECTOR.read_text(encoding="utf-8").replace("\\\n", " ")
    calls = []
    for raw in _INVOCATION.findall(text):
        # Substitute the EXPANSION only, never the surrounding quotes:
        # `"kai-gate-050-obs-${n}"` must stay one quoted word, or shlex
        # sees an unbalanced quote and the check dies instead of judging.
        resolved = re.sub(r"\$\{(\w+)\}|\$(\w+)", "1", raw)
        if "$" in resolved:
            raise AssertionError(f"unresolved shell expansion in {raw!r}")
        calls.append(["-"] + shlex.split(resolved))
    print(f"  inspected: {len(calls)} probe invocation(s) in "
          f"{COLLECTOR.relative_to(REPO)}")
    check("the collector invokes the probe at all", len(calls) >= 2, str(calls))
    for argv in calls:
        action, _b, _s, error = p.parse_argv(argv)
        check(f"probe accepts {' '.join(argv[1:])!r}", not error,
              f"{error} (from {argv})")
    kinds = {p.parse_argv(a)[0] for a in calls}
    check("both an ingest and a cognee-log invocation exist",
          kinds == {"ingest", "cognee-log"}, str(kinds))
    # known-negatives, including run 8's mistake in this probe's vocabulary
    check("a bare budget is rejected", bool(p.parse_argv(["-", "900"])[3]), "")
    check("ingest without a source_id is rejected",
          bool(p.parse_argv(["-", "ingest", "900"])[3]), "")
    check("an empty source_id is rejected",
          bool(p.parse_argv(["-", "ingest", "900", " "])[3]), "")
    check("cognee-log with a stray argument is rejected",
          bool(p.parse_argv(["-", "cognee-log", "x"])[3]), "")
    # known-positives, so the parser is not merely refusing everything
    check("the real ingest form is accepted",
          p.parse_argv(["-", "ingest", "900", "obs-1"])[0] == "ingest", "")
    check("the real cognee-log form is accepted",
          p.parse_argv(["-", "cognee-log"])[0] == "cognee-log", "")


def run_all() -> None:
    test_failed_pipeline_plus_200_is_the_finding()
    test_failed_pipeline_plus_502_is_correct_behaviour()
    test_completed_pipelines_plus_200_is_consistent()
    test_completed_pipelines_plus_502_is_inverted()
    test_http_200_alone_never_establishes_success()
    test_a_pipeline_with_no_terminal_marker_is_not_completed()
    test_the_gate_fails_on_the_finding_and_passes_on_clean()
    test_every_collector_invocation_is_one_the_probe_accepts()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Ingest Contract Analyser Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
