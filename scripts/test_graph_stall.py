#!/usr/bin/env python3
"""Calibration for the KAI-GATE-049 stall analyser.

The load-bearing assertions, in the order they matter:

1. **Four different states must stay four different verdicts.** CPU
   growth and delegate sockets are two independent bits; collapsing them
   would turn "waiting on ollama" and "genuinely slow" into one answer
   with one remedy, and only one of those remedies would be right.

2. **A non-return must never read as a hang.** Runs 4 and 6 agreed on
   ~291s because both used the same 300s client budget. Reading an
   instrument-determined number as a system property is the mistake this
   unit exists to undo; the analyser must not repeat it one level up.

3. **Marker pairing must find the unpaired one.** A task list is not a
   diagnosis, but the unpaired entry is the question's literal answer.

4. **The collector's command line must be one the probe accepts, and a
   run in which nothing was asked must not read as a diagnosis.** Run 8
   invoked the probe as `python - 900`: the budget became argv[1], the
   subcommand was absent, the probe exited 2 without sending anything,
   and every downstream section reported its own absence honestly while
   the job went green. The command line is checked HERE, from the
   collector's own text, because that costs nothing and happens before
   a stack exists — and the expected answer comes from the probe's
   parser rather than from a list kept beside it (R5, I-8).
"""
from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import summarise_graph_stall as s  # noqa: E402
from scripts.security import probe_graph_stall as p  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 12
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


SERVICE_LOG = """
    memu-graph-1 | 2026-08-13T16:55:51.164Z info Coroutine task started: `classify_documents`
    memu-graph-1 | 2026-08-13T16:55:51.165Z info Async Generator task started: `extract_chunks_from_documents`
    memu-graph-1 | 2026-08-13T16:55:51.174Z info Coroutine task started: `extract_graph_and_summarize`
    memu-graph-1 | 2026-08-13T16:55:51.180Z info Coroutine task completed: `classify_documents`
    """


def rows(n, utime_step, delegate, hz=100, step=20):
    out = []
    for i in range(n):
        out.append(json.dumps({
            "monotonic": float(i * step),
            "pid1_utime_ticks": i * utime_step,
            "pid1_stime_ticks": 0,
            "clock_ticks_per_sec": hz,
            "connections_total": 3,
            "delegate_connections": (
                [{"remote": "172.18.0.3:11434", "state": "ESTABLISHED"}]
                if delegate else []),
        }))
    return "\n".join(out) + "\n"


def stage_dir(root, files):
    d = Path(root)
    for name, body in files.items():
        (d / name).write_text(textwrap.dedent(body))
    return d


def run(d):
    p = subprocess.run(
        [sys.executable, "scripts/security/summarise_graph_stall.py",
         "--stage-logs", str(d)],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    return p.stdout


# ── 1. the four states ───────────────────────────────────────────────

def test_the_four_states_stay_distinct() -> None:
    scenario("four states distinct")
    # 20 ticks per 20s sample at 100Hz = 0.2s CPU per 20s = 1% -> flat.
    # 1200 ticks per 20s = 12s CPU per 20s = 60% -> growing.
    got = {
        "slow llm": s.cpu_verdict(json_rows(rows(5, 1200, True)))[0],
        "waiting": s.cpu_verdict(json_rows(rows(5, 5, True)))[0],
        "stuck": s.cpu_verdict(json_rows(rows(5, 5, False)))[0],
        "local": s.cpu_verdict(json_rows(rows(5, 1200, False)))[0],
    }
    check("four inputs give four distinct verdicts",
          len(set(got.values())) == 4, str(got))
    check("computing + delegate = SLOW LLM WORK",
          got["slow llm"] == "SLOW LLM WORK", got["slow llm"])
    check("flat + delegate = WAITING ON DELEGATE",
          got["waiting"] == "WAITING ON DELEGATE", got["waiting"])
    check("flat + no socket = STUCK ELSEWHERE",
          got["stuck"] == "STUCK ELSEWHERE", got["stuck"])
    check("computing + no socket = LOCAL COMPUTE",
          got["local"] == "LOCAL COMPUTE", got["local"])


def json_rows(text):
    return s.samples(text)


def test_one_sample_cannot_establish_growth() -> None:
    """Growth needs two points. One sample must be UNKNOWN, not a guess."""
    scenario("one sample is unknown")
    check("zero samples -> UNKNOWN", s.cpu_verdict([])[0] == "UNKNOWN", "")
    check("one sample -> UNKNOWN",
          s.cpu_verdict(json_rows(rows(1, 1200, True)))[0] == "UNKNOWN", "")
    check("two samples are enough to decide",
          s.cpu_verdict(json_rows(rows(2, 1200, True)))[0] != "UNKNOWN", "")


# ── 2. non-return is not a hang ──────────────────────────────────────

def test_a_non_return_is_reported_as_the_window_ending() -> None:
    scenario("non-return is not a hang")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "rc.env": "INGEST_RC=1\nWINDOW=900\nELAPSED=900\nLIVE_CYCLE_BUDGET=300\n",
            "ingest.log": "ENTERED  http-post monotonic=1.0\n"
                          "NO-RETURN http-post TimeoutError: timed out elapsed=900.0s\n",
            "samples.log": rows(5, 5, True),
            "service-logs.log": SERVICE_LOG,
            "ollama-after.log": "",
            "ollama-baseline.log": "",
        })
        out = run(d)
        check("says the window ended", "OBSERVATION WINDOW ending" in out, out[-800:])
        check("does NOT call it a hang", "proven hang" in out and
              "not a proven hang" in out, out[-800:])
        check("reports the window length", "900s" in out, out[:900])
        check("names the corrected premise",
              "not because the system did the same thing twice" in out, out[:900])


def test_a_return_is_reported_with_its_duration() -> None:
    """The whole point of watching past our own budget."""
    scenario("return is measured")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "rc.env": "INGEST_RC=0\nWINDOW=900\nELAPSED=520\nLIVE_CYCLE_BUDGET=300\n",
            "ingest.log": "ENTERED  http-post monotonic=1.0\n"
                          "RETURNED http-post  status=200 elapsed=517.3s\n"
                          'BODY {"status":"ingested"}\n',
            "samples.log": rows(5, 1200, True),
            "service-logs.log": SERVICE_LOG,
            "ollama-after.log": "",
            "ollama-baseline.log": "",
        })
        out = run(d)
        check("reports the status", "status=200" in out, out[:1200])
        check("reports the duration", "517.3s" in out, out[:1200])
        check("says a duration now exists",
              "measured duration and outcome" in out, out[:1200])
        check("and does not claim the window ended",
              "DID NOT RETURN" not in out, out[:1200])


# ── 3. marker pairing ────────────────────────────────────────────────

def test_the_unpaired_marker_is_found() -> None:
    scenario("unpaired marker found")
    unpaired, order = s.task_markers(textwrap.dedent(SERVICE_LOG))
    names = [n for n, _ in unpaired]
    check("three tasks entered", len(order) == 3, str(order))
    check("classify_documents returned",
          "classify_documents" not in names, str(names))
    check("extract_chunks_from_documents is unpaired",
          "extract_chunks_from_documents" in names, str(names))
    check("extract_graph_and_summarize is unpaired",
          "extract_graph_and_summarize" in names, str(names))
    check("timestamps are carried",
          all(st for _, st in unpaired), str(unpaired))


def test_a_fully_paired_log_reports_no_unpaired_task() -> None:
    """The known-negative. If every task returned, the silence is not an
    unpaired cognee task and saying otherwise would send someone into
    the wrong library."""
    scenario("fully paired log")
    paired = SERVICE_LOG + textwrap.dedent("""
        memu-graph-1 | 2026-08-13T16:56:01.000Z info Async Generator task completed: `extract_chunks_from_documents`
        memu-graph-1 | 2026-08-13T16:59:01.000Z info Coroutine task completed: `extract_graph_and_summarize`
        """)
    unpaired, order = s.task_markers(paired)
    check("nothing unpaired", unpaired == [], str(unpaired))
    check("but tasks were still seen", len(order) == 3, str(order))


def test_the_gather_fanout_is_named() -> None:
    """cognee emits ONE marker for asyncio.gather(extract_graph_from_data,
    summarize_text). The report must say so, or a reader will look for a
    single culprit that does not exist."""
    scenario("gather fan-out named")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "rc.env": "INGEST_RC=1\nWINDOW=900\nELAPSED=900\nLIVE_CYCLE_BUDGET=300\n",
            "ingest.log": "ENTERED  http-post  monotonic=1.0 budget=900.0s\n"
                          "NO-RETURN http-post TimeoutError: x elapsed=900.0s\n",
            "samples.log": rows(5, 5, True),
            "service-logs.log": SERVICE_LOG,
            "ollama-after.log": "",
            "ollama-baseline.log": "",
        })
        out = run(d)
        check("names the two concurrent paths",
              "extract_graph_from_data" in out and "summarize_text" in out,
              out[-900:])
        check("says cognee's logging cannot separate them",
              "cannot say which" in out, out[-900:])


def test_the_report_refuses_to_authorise_a_remedy() -> None:
    """It names a stage and a state. Three states, three owners."""
    scenario("no remedy authorised")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "rc.env": "INGEST_RC=1\nWINDOW=900\nELAPSED=900\nLIVE_CYCLE_BUDGET=300\n",
            "ingest.log": "ENTERED  http-post  monotonic=1.0 budget=900.0s\n"
                          "NO-RETURN http-post TimeoutError: x elapsed=900.0s\n",
            "samples.log": rows(5, 5, True),
            "service-logs.log": SERVICE_LOG,
            "ollama-after.log": "",
            "ollama-baseline.log": "",
        })
        out = run(d)
        # Whitespace-normalised: the report wraps at ~72 columns, so a
        # contiguous-substring match asserts the LAYOUT rather than the
        # content and breaks whenever a sentence re-wraps.
        flat = " ".join(out.split())
        check("says it does not authorise a remedy",
              "does NOT authorise a remedy" in flat, flat[-400:])
        check("says raising a timeout answers none of them",
              "raising a timeout would answer none of them" in flat,
              flat[-400:])
    with tempfile.TemporaryDirectory() as tmp:
        out = run(Path(tmp))
        check("an empty directory establishes nothing",
              "NOT COLLECTED" in out, out[:400])
        check("and does not invent an outcome",
              "OUTCOME NOT ESTABLISHED" in out or "0 of 5" in out, out[:600])


# ── 4. the collector's command line, and the unmeasured run ──────────

COLLECTOR = REPO / "scripts" / "security" / "diagnose_graph_stall.sh"

# `python - <args> < .../probe_graph_stall.py`. The args are whatever sits
# between the `-` and the redirection that feeds the script in.
_INVOCATION = re.compile(
    r"python\s+-\s+([^\n<]*)<\s*scripts/security/probe_graph_stall\.py")


def collector_invocations() -> list[list[str]]:
    """Every probe command line the collector actually runs.

    DERIVED FROM THE COLLECTOR, never listed beside it: a hand-kept tuple
    of expected invocations would have agreed with itself while the real
    one was broken. Shell variables collapse to the sentinel `1`, because
    what is under test is the SHAPE of the command line — subcommand
    present, arity right — not the numeric budget. That substitution is
    also what makes the run-8 defect visible: a bare `"$WINDOW"` in the
    subcommand position becomes `1`, and `1` is not a subcommand.
    """
    text = COLLECTOR.read_text(encoding="utf-8")
    out = []
    for raw in _INVOCATION.findall(text):
        resolved = re.sub(r'"?\$\{?(\w+)\}?"?', "1", raw)
        if "$" in resolved:
            raise AssertionError(
                f"unresolved shell expansion in {raw!r} — this calibration "
                f"cannot validate that invocation, and must not pretend to")
        out.append(["-"] + shlex.split(resolved))
    return out


def test_every_collector_invocation_is_one_the_probe_accepts() -> None:
    scenario("collector invocations accepted")
    calls = collector_invocations()
    # I-2. A check that finds nothing and a check that looks at nothing
    # print the same thing unless the denominator is stated.
    print(f"  inspected: {len(calls)} probe invocation(s) in "
          f"{COLLECTOR.relative_to(REPO)}")
    check("the collector invokes the probe at all", len(calls) >= 2, str(calls))
    for argv in calls:
        action, _budget, error = p.parse_argv(argv)
        check(f"probe accepts {' '.join(argv[1:])!r}", not error,
              f"{error} (from {argv})")
        check(f"{' '.join(argv[1:])!r} resolves to an action",
              action in ("ingest", "sample"), str(action))
    actions = {p.parse_argv(a)[0] for a in calls}
    check("both an ingest and a sample invocation exist",
          actions == {"ingest", "sample"}, str(actions))


def test_the_run_8_command_line_is_rejected() -> None:
    """The known-negative, and the reason this scenario exists at all.

    `python - "$WINDOW"` is exactly what shipped in run 8. If the check
    above cannot fail on it, it is decoration."""
    scenario("run-8 command line rejected")
    action, _b, error = p.parse_argv(["-", "900"])
    check("a bare budget is not a valid command line", bool(error), str(action))
    check("and the error names the missing subcommand",
          "unknown subcommand" in error, error)
    check("no action is returned", action is None, str(action))
    # the other arity mistakes in the same family
    check("ingest with no budget is rejected",
          bool(p.parse_argv(["-", "ingest"])[2]), "")
    check("ingest with a non-numeric budget is rejected",
          bool(p.parse_argv(["-", "ingest", "soon"])[2]), "")
    check("sample with a stray argument is rejected",
          bool(p.parse_argv(["-", "sample", "900"])[2]), "")
    # ...and the known-positives, so the parser is not simply refusing
    check("ingest with a budget is accepted",
          p.parse_argv(["-", "ingest", "900"])[:2] == ("ingest", 900.0), "")
    check("sample alone is accepted",
          p.parse_argv(["-", "sample"])[0] == "sample", "")


def test_a_run_that_asked_nothing_is_not_a_diagnosis() -> None:
    """Run 8's evidence, reduced. Three honest absences must not add up
    to a green diagnostic run."""
    scenario("unmeasured run fails closed")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "rc.env": "INGEST_RC=2\nWINDOW=900\nELAPSED=20\nLIVE_CYCLE_BUDGET=300\n",
            "ingest.log": "usage: probe_graph_stall.py "
                          "{ingest <budget-seconds>|sample}\n",
            "samples.log": rows(1, 5, False),
            "service-logs.log": "memu-graph-1 | 2026-08-13T18:30:31.1Z "
                                "INFO:     Application startup complete.\n",
            "ollama-after.log": "",
            "ollama-baseline.log": "",
        })
        proc = subprocess.run(
            [sys.executable, "scripts/security/summarise_graph_stall.py",
             "--stage-logs", str(d)],
            cwd=REPO, capture_output=True, text=True, timeout=60)
        out = proc.stdout
        check("exits non-zero — 'not measured' must not read as 'it works'",
              proc.returncode != 0, f"rc={proc.returncode}")
        check("says no request was sent",
              "NO REQUEST WAS EVER SENT" in out, out[:600])
        check("names exit 2 as the probe rejecting its command line",
              "REJECTING ITS OWN COMMAND LINE" in " ".join(out.split()),
              out[:900])
        check("calls it an instrument failure, not a system property",
              "INSTRUMENT INVOCATION FAILURE" in out, out[-700:])
        check("distinguishes unmeasured from measured-and-clean",
              "different from" in out and "measured-and-clean" in out,
              out[-500:])
        # and it must NOT go on to present the hierarchy as if it held
        check("does not report a stage-ownership verdict",
              "1. STAGE OWNERSHIP" not in out, out[:900])
        check("does not report an execution-state verdict",
              "STATE:" not in out, out[:900])


def test_a_real_observation_still_passes() -> None:
    """The known-positive for the same gate. A genuine non-return is a
    RESULT, not an instrument failure, and must still exit 0."""
    scenario("real observation still passes")
    with tempfile.TemporaryDirectory() as tmp:
        d = stage_dir(tmp, {
            "rc.env": "INGEST_RC=1\nWINDOW=900\nELAPSED=900\nLIVE_CYCLE_BUDGET=300\n",
            "ingest.log": "ENTERED  http-post  monotonic=1.0 budget=900.0s\n"
                          "NO-RETURN http-post TimeoutError: timed out "
                          "elapsed=900.0s\n",
            "samples.log": rows(5, 5, True),
            "service-logs.log": SERVICE_LOG,
            "ollama-after.log": "",
            "ollama-baseline.log": "",
        })
        proc = subprocess.run(
            [sys.executable, "scripts/security/summarise_graph_stall.py",
             "--stage-logs", str(d)],
            cwd=REPO, capture_output=True, text=True, timeout=60)
        check("a measured non-return exits 0", proc.returncode == 0,
              f"rc={proc.returncode}")
        check("and the hierarchy is reported",
              "1. STAGE OWNERSHIP" in proc.stdout, proc.stdout[:400])
        check("and it is not called an instrument failure",
              "INSTRUMENT INVOCATION FAILURE" not in proc.stdout,
              proc.stdout[-500:])


def run_all() -> None:
    test_the_four_states_stay_distinct()
    test_one_sample_cannot_establish_growth()
    test_a_non_return_is_reported_as_the_window_ending()
    test_a_return_is_reported_with_its_duration()
    test_the_unpaired_marker_is_found()
    test_a_fully_paired_log_reports_no_unpaired_task()
    test_the_gather_fanout_is_named()
    test_the_report_refuses_to_authorise_a_remedy()
    test_every_collector_invocation_is_one_the_probe_accepts()
    test_the_run_8_command_line_is_rejected()
    test_a_run_that_asked_nothing_is_not_a_diagnosis()
    test_a_real_observation_still_passes()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Graph Stall Analyser Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
