#!/usr/bin/env python3
"""Calibration for the KAI-GATE-048 Phase 1 acceptance verdict.

This verdict decides whether a finding closes, so it gets the treatment
every instrument here gets: both directions, on synthetic stage logs
whose answers are known before the parser sees them.

THE TWO THAT MATTER MOST
========================

**D is inverted, and that is deliberate.** The can-fail stage PASSES on a
NON-ZERO exit — it withholds the asset and the check must fail. A
summariser that read exit codes uniformly would call a broken
calibration a pass, which is the precise shape of "a gate nobody has
seen fail".

**C must not be satisfiable by removing the network.** memu-graph
delegates embedding work to `ollama`. If the internal delegate is
unreachable, C must FAIL even when external egress is blocked and
everything else looks clean — otherwise `--network none` would satisfy
the criterion and we would have proved the wrong property. D190 proposed
exactly that criterion; D191 corrected it. The assertion below is what
stops it coming back.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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


GOOD = {
    "A-final-image-offline": "OFFLINE-VERIFIED bert-base-uncased in 0.31s class=BertTokenizerFast\n",
    "D-canfail-no-asset": "REFUSING: ... OSError: couldn't find them in the cached files\n",
    "B-chronology": "FIRST PASSING HEALTH PROBE at +5.4s from StartedAt\n",
    "B-maps-ready": "tokenizers: 0\ntorch: 0\nsafetensors: 0\n",
    "C1-external-egress": "huggingface.co:443 FAILED (gaierror) -> no external egress\n",
    "C2-internal-reachability": "ollama:11434 CONNECTED -> internal delegate REACHABLE\nollama /api/tags HTTP 200 -> delegate SERVING\n",
    "C3-live-cycle": "PASS: ingest -> cognify -> query -> forget cycle completed\n",
    "C4-service-logs": 'INFO: "POST /graph/ingest HTTP/1.1" 200 OK\n',
}
ARGS = ["--tree-sha", "a" * 40, "--commit", "b" * 40,
        "--image", "sha256:" + "c" * 40, "--dirty", "0",
        "--probe-rc", "0", "--a-rc", "0", "--d-rc", "1", "--c3-rc", "0"]


def run(stage_dir, extra=None) -> tuple[str, int]:
    args = list(ARGS)
    for k, v in (extra or {}).items():
        args[args.index(k) + 1] = str(v)
    p = subprocess.run(
        [sys.executable, "scripts/security/summarise_memu_graph_acceptance.py",
         "--stage-logs", str(stage_dir)] + args,
        cwd=REPO, capture_output=True, text=True, timeout=60)
    return p.stdout, p.returncode


def stages(root, overrides=None, drop=()):
    d = Path(root)
    body = dict(GOOD)
    body.update(overrides or {})
    for name in drop:
        body.pop(name, None)
    for name, text in body.items():
        (d / f"{name}.log").write_text(textwrap.dedent(text))
    return d


def test_the_fully_good_case_meets_acceptance() -> None:
    scenario("all good meets acceptance")
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(stages(tmp))
        check("exit 0", rc == 0, f"rc={rc}\n{out[-500:]}")
        check("ACCEPTANCE MET", "ACCEPTANCE MET" in out, out[-500:])
        check("all five report PASS", out.count("PASS") >= 5, out)
        check("8 of 8 stages", "inspected: 8 of 8" in out, out[:200])


def test_c_fails_when_the_internal_delegate_is_unreachable() -> None:
    """THE assertion that stops D190's mistake returning. External egress
    blocked + everything else clean must still FAIL if the delegate the
    service is designed to use has gone."""
    scenario("delegate unreachable fails C")
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(stages(tmp, {
            "C2-internal-reachability":
                "ollama:11434 FAILED (gaierror: no such host) -> delegate UNREACHABLE\n",
            "C3-live-cycle": "FAIL: ingest returned 502\n",
        }), {"--c3-rc": 1})
        check("NOT met", "ACCEPTANCE MET" not in out, out[-500:])
        check("C failed", "C  FAIL" in out, out[-600:])
        check("names the unreachable delegate", "UNREACHABLE" in out, out[-600:])
        check("exit 1", rc == 1, str(rc))


def test_blocking_all_networking_does_not_satisfy_c() -> None:
    """`--network none` blocks external egress AND the delegate. If that
    combination read as a pass, the acceptance would be satisfiable by
    testing a system we do not ship."""
    scenario("network none does not satisfy C")
    with tempfile.TemporaryDirectory() as tmp:
        out, _rc = run(stages(tmp, {
            "C1-external-egress": "huggingface.co:443 FAILED (gaierror) -> no external egress\n",
            "C2-internal-reachability": "ollama:11434 FAILED (OSError) -> delegate UNREACHABLE\n",
        }))
        check("not accepted", "ACCEPTANCE MET" not in out, out[-500:])
        check("C is the failing one", "C  FAIL" in out, out[-600:])


def test_the_can_fail_stage_is_inverted() -> None:
    """D PASSES on non-zero and FAILS on zero. Getting this backwards
    would turn a broken calibration into a green result."""
    scenario("D is inverted")
    with tempfile.TemporaryDirectory() as tmp:
        out, _ = run(stages(tmp), {"--d-rc": 1})
        check("non-zero is a PASS for D", "D  PASS" in out, out[-600:])
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(stages(tmp), {"--d-rc": 0})
        check("zero is a FAIL for D", "D  FAIL" in out, out[-600:])
        check("and says stage A proves nothing",
              "proves nothing" in out, out[-600:])
        check("overall not met", rc == 1, str(rc))


def test_a_premature_load_fails_b() -> None:
    """The lazy design is what we are NOT repairing. If the bake somehow
    dragged the tokenizer into startup, B must catch it."""
    scenario("premature load fails B")
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(stages(tmp, {
            "B-maps-ready": "tokenizers: 4\ntorch: 0\nsafetensors: 0\n"}))
        check("B failed", "B  FAIL" in out, out[-600:])
        check("names the regression", "lazy design regressed" in out, out[-600:])
        check("exit 1", rc == 1, str(rc))


def test_a_retry_storm_fails_c_even_if_the_request_succeeded() -> None:
    """A 200 that arrived after 47s of backoff is not the contract met —
    it would mean the asset was fetched, not baked."""
    scenario("retry storm fails C")
    with tempfile.TemporaryDirectory() as tmp:
        out, _ = run(stages(tmp, {
            "C4-service-logs":
                "Retrying in 8s [Retry 5/5].\n"
                'INFO: "POST /graph/ingest HTTP/1.1" 200 OK\n'}))
        check("C failed despite the 200", "C  FAIL" in out, out[-600:])
        check("names the retry sequence", "RETRY SEQUENCE PRESENT" in out,
              out[-600:])


def test_a_dirty_tree_fails_artefact_identity() -> None:
    scenario("dirty tree fails E")
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(stages(tmp), {"--dirty": "3"})
        check("E failed", "E  FAIL" in out, out[-600:])
        check("says the tree is not committed",
              "not a committed tree" in out, out[-600:])
        check("exit 1", rc == 1, str(rc))


def test_missing_stages_are_incomplete_not_failed() -> None:
    """"We did not measure" and "we measured and it was wrong" have
    different remedies, and only one of them means the change is bad."""
    scenario("missing is incomplete")
    with tempfile.TemporaryDirectory() as tmp:
        out, rc = run(stages(tmp, drop=("C3-live-cycle", "C4-service-logs")),
                      {"--c3-rc": None} if False else None)
        check("reports what was not collected",
              "NOT COLLECTED" in out and "C3-live-cycle" in out, out[:400])
    with tempfile.TemporaryDirectory() as tmp:
        # nothing at all collected
        out, rc = run(Path(tmp))
        check("INCOMPLETE, not NOT MET", "INCOMPLETE" in out, out[-500:])
        check("does not claim acceptance", "ACCEPTANCE MET" not in out, out[-500:])
        check("still exits non-zero", rc == 1, str(rc))
        check("says evidence absence differs from bad evidence",
              "different thing" in out, out[-400:])


def run_all() -> None:
    test_the_fully_good_case_meets_acceptance()
    test_c_fails_when_the_internal_delegate_is_unreachable()
    test_blocking_all_networking_does_not_satisfy_c()
    test_the_can_fail_stage_is_inverted()
    test_a_premature_load_fails_b()
    test_a_retry_storm_fails_c_even_if_the_request_succeeded()
    test_a_dirty_tree_fails_artefact_identity()
    test_missing_stages_are_incomplete_not_failed()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"memu-graph Acceptance Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
