#!/usr/bin/env python3
"""Calibration for the KAI-GATE-048 asset-contract summariser.

The summariser turns four stage logs into an answer to five questions,
and the answer will be used to plan a remediation. So the parsers get the
same treatment every other instrument in this repository has had: both
directions, on synthetic inputs whose answers are known before the
parser sees them.

THE ONE THAT MATTERS
====================

`CONTRACT PROVEN` may be printed only when the network-removed stage
SUCCEEDED on the asset set the fetch stage produced. A summariser that
printed it from stage A alone would be reporting a list of files nobody
had shown to be complete — and a bake built from that list would ship a
still-broken image while every log said the contract was defined.

So there are four assertions on that one sentence: proven, not-proven
(stage B absent), disproven (stage B failed), and ambiguous (stage B
succeeded but stage C did not fail). Each has a different remedy, and
collapsing any pair would hide the one case that matters.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import summarise_asset_contract as s  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 9
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


FETCH_OK = """
    huggingface_hub 0.34.1
    transformers 4.44.0
    HF_HOME               /probe-cache
    HF_HUB_CACHE          /probe-cache/hub
    HF_HUB_OFFLINE env    None
    model requested       bert-base-uncased
    RESULT: LOADED in 3.41s  class=BertTokenizerFast
    --- cache tree under /probe-cache ---
          1234  hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/config.json
         28112  hub/models--bert-base-uncased/blobs/deadbeef
        231508  hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/vocab.txt
    --- file count ---
    3
    --- total bytes ---
    466274
    """


def write(root, files: dict) -> Path:
    d = Path(root)
    for name, body in files.items():
        (d / f"{name}.log").write_text(textwrap.dedent(body))
    return d


def run(stage_dir) -> str:
    return subprocess.run(
        [sys.executable, "scripts/security/summarise_asset_contract.py",
         "--stage-logs", str(stage_dir)],
        cwd=REPO, capture_output=True, text=True, timeout=60).stdout


# ── the parsers, both directions ─────────────────────────────────────

def test_the_result_parser_separates_loaded_failed_and_absent() -> None:
    scenario("result parser")
    check("LOADED is True", s.loaded("RESULT: LOADED in 1.0s") is True, "")
    check("FAILED is False", s.loaded("RESULT: FAILED after 1.0s x") is False, "")
    check("a log with neither is None",
          s.loaded("the container exploded") is None,
          "an unparseable log must not read as a proven failure")
    check("a missing log is None", s.loaded(None) is None, "")


def test_the_timing_parser_reads_both_shapes() -> None:
    """The fail-fast comparison is the measurement behind obligation 2,
    and it is only available if BOTH the success and failure timings
    parse."""
    scenario("timing parser")
    check("success timing", s.seconds("RESULT: LOADED in 3.41s  class=X") == 3.41, "")
    check("failure timing", s.seconds("RESULT: FAILED after 47.2s  OSError") == 47.2, "")
    check("absent timing is None", s.seconds("no result line here") is None, "")


def test_the_cache_parser_counts_bytes_and_files() -> None:
    scenario("cache parser")
    n, total, files = s.cache_files(FETCH_OK)
    check("file count", n == 3, str(n))
    check("total bytes", total == 466274, str(total))
    check("file list is parsed", len(files) == 3, str(files))
    check("sizes are ints", all(isinstance(sz, int) for sz, _ in files), str(files))
    n2, total2, files2 = s.cache_files("nothing useful")
    check("garbage yields None, not zero", n2 is None and total2 is None,
          f"{n2} {total2} — an unparseable log must not read as an empty cache")
    check("and no files", files2 == [], str(files2))


def test_the_revision_is_read_off_the_cache_layout() -> None:
    """cognee never passes `revision=`, so the only place a pinnable
    revision exists is the snapshot path the stack itself created."""
    scenario("revision parser")
    _n, _t, files = s.cache_files(FETCH_OK)
    check("snapshot sha found",
          s.revision(files) == "86b5e0934494bd15c9632b12f734a8a67f723594",
          str(s.revision(files)))
    check("no snapshot path -> None",
          s.revision([(1, "hub/models--x/blobs/abc")]) is None, "")
    check("empty file list -> None", s.revision([]) is None, "")


# ── the verdict sentence, all four ways ──────────────────────────────

def test_contract_proven_requires_the_network_removed_stage() -> None:
    scenario("proven needs stage B")
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, {
            "A-fetch": FETCH_OK,
            "B-offline-with-asset": "RESULT: LOADED in 0.30s  class=BertTokenizerFast\n",
            "C-offline-no-asset": "RESULT: FAILED after 0.05s  OSError: offline\n",
            "D-noflag-no-asset": "RESULT: FAILED after 47.20s  OSError: name resolution\n",
        })
        out = run(tmp)
        check("prints CONTRACT PROVEN", "CONTRACT PROVEN" in out, out[-500:])
        check("reports 4 of 4 stages", "inspected: 4 of 4" in out, out[:200])
        check("names the model", "bert-base-uncased" in out, out[:600])
        check("reports the fail-closed cost",
              "FAIL-CLOSED COST" in out and "47.2s" in out, out[-400:])


def test_a_missing_stage_b_is_not_proven() -> None:
    """The load-bearing negative. Stage A alone is a list of files."""
    scenario("no stage B is not proven")
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, {"A-fetch": FETCH_OK})
        out = run(tmp)
        check("NOT proven", "CONTRACT PROVEN" not in out, out[-500:])
        check("says why", "CONTRACT NOT PROVEN" in out, out[-500:])
        check("names the uncollected stages",
              "NOT COLLECTED" in out and "B-offline-with-asset" in out, out[:400])
        check("and still answers what it can",
              "bert-base-uncased" in out, out[:600])


def test_a_failing_stage_b_is_disproven_not_merely_unproven() -> None:
    """Different remedy: NOT PROVEN means measure again, DISPROVEN means
    the asset list is wrong and a bake built from it would ship broken."""
    scenario("failing stage B is disproven")
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, {
            "A-fetch": FETCH_OK,
            "B-offline-with-asset": "RESULT: FAILED after 0.10s  OSError: missing file\n",
            "C-offline-no-asset": "RESULT: FAILED after 0.05s  OSError: offline\n",
        })
        out = run(tmp)
        check("DISPROVEN", "CONTRACT DISPROVEN" in out, out[-500:])
        check("warns a bake would ship broken", "still-broken" in out, out[-500:])
        check("not confused with NOT PROVEN",
              "CONTRACT NOT PROVEN" not in out, out[-500:])


def test_a_stage_c_that_succeeds_is_ambiguous() -> None:
    """If the asset is absent and it loads anyway, the flags are not
    doing what a remediation plan would assume — and that must not be
    reported as a proven contract."""
    scenario("stage C success is ambiguous")
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, {
            "A-fetch": FETCH_OK,
            "B-offline-with-asset": "RESULT: LOADED in 0.30s\n",
            "C-offline-no-asset": "RESULT: LOADED in 0.31s\n",
        })
        out = run(tmp)
        check("AMBIGUOUS", "CONTRACT AMBIGUOUS" in out, out[-500:])
        check("not PROVEN", "CONTRACT PROVEN" not in out, out[-500:])


def test_an_empty_stage_directory_answers_nothing() -> None:
    scenario("empty dir answers nothing")
    with tempfile.TemporaryDirectory() as tmp:
        out = run(tmp)
        check("0 of 4 stages", "inspected: 0 of 4" in out, out[:200])
        check("NOT PROVEN", "CONTRACT NOT PROVEN" in out, out[-400:])
        check("every question says NOT MEASURED",
              out.count("NOT MEASURED") >= 4, out)
        check("no fail-closed cost is invented",
              "FAIL-CLOSED COST" not in out, out[-300:])


def run_all() -> None:
    test_the_result_parser_separates_loaded_failed_and_absent()
    test_the_timing_parser_reads_both_shapes()
    test_the_cache_parser_counts_bytes_and_files()
    test_the_revision_is_read_off_the_cache_layout()
    test_contract_proven_requires_the_network_removed_stage()
    test_a_missing_stage_b_is_not_proven()
    test_a_failing_stage_b_is_disproven_not_merely_unproven()
    test_a_stage_c_that_succeeds_is_ambiguous()
    test_an_empty_stage_directory_answers_nothing()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Asset Contract Summariser Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
