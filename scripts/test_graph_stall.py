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
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import summarise_graph_stall as s  # noqa: E402

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
            "ingest.log": "NO-RETURN http-post TimeoutError: x elapsed=900.0s\n",
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
            "ingest.log": "NO-RETURN http-post TimeoutError: x elapsed=900.0s\n",
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


def run_all() -> None:
    test_the_four_states_stay_distinct()
    test_one_sample_cannot_establish_growth()
    test_a_non_return_is_reported_as_the_window_ending()
    test_a_return_is_reported_with_its_duration()
    test_the_unpaired_marker_is_found()
    test_a_fully_paired_log_reports_no_unpaired_task()
    test_the_gather_fanout_is_named()
    test_the_report_refuses_to_authorise_a_remedy()

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
