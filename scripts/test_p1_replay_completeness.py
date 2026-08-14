#!/usr/bin/env python3
"""Calibration for the P1 replay-completeness instrument.

P1 exists to answer one question before D247's Stage 1 may run: does the
capture hold enough of the request to replay it faithfully? The operator's
condition, which is the whole reason this suite is separate and strict:

> A capture showing `other_params == []` tells us nothing about whether
> positional args were present. P1 cannot close merely from that.

So the property under test is not "does the census count rows" — it is
**the two axes must never substitute for one another**. A spotless axis A
with no source evidence for axis B must read as INCOMPLETE_POSITIONAL, and
the multiple-defect case must never collapse into either single verdict.

Every verdict gets a known-positive and a known-negative (I-8), and the
expected answers come from synthetic fixtures built here — not from the
instrument's own output about itself.

This suite is deliberately NOT part of scripts/test_llm_contract.py.
Sharing that file would have put P1 inside the LLM-capture workflow's
paths filter, so a change to a static census would have triggered a live
model capture that nobody authorised.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import p1_replay_completeness as p1  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 2
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


def _p1_capture(tmp: Path, name: str, rows: list[dict]) -> str:
    path = tmp / f"{name}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _p1_row(**over) -> dict:
    row = {"event": "llm-call", "phase": "capture",
           "messages": [{"role": "user", "content": "hi"}],
           "model": "qwen2.5:3b", "temperature": None,
           "response_format": {"type": "json_object"}, "tools": None,
           "other_params": []}
    row.update(over)
    return row


def _p1_source(tmp: Path, name: str, body: str) -> Path:
    root = tmp / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter.py").write_text(body)
    return root


CLEAN_CALL_SITE = (
    "def go(client):\n"
    "    return client.chat.completions.create(\n"
    "        model=m, messages=msgs, max_retries=2, response_model=R)\n")
DIRTY_CALL_SITE = (
    "def go(client):\n"
    "    return client.chat.completions.create(payload, model=m)\n")
CLEAN_FORWARDER = "def retry_sync(func, args, kwargs):\n    return func(*args, **kwargs)\n"
DIRTY_FORWARDER = "def retry_sync(func, args, kwargs):\n    return func(EXTRA, *args, **kwargs)\n"


def _p1_forwarder(tmp: Path, name: str, body: str) -> Path:
    root = tmp / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "retry.py").write_text(body)
    return root


def test_p1_never_lets_one_clean_axis_imply_the_other() -> None:
    scenario("p1 axes are independent")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clean_rows = _p1_capture(tmp, "clean", [_p1_row(), _p1_row()])
        clean_src = _p1_source(tmp, "clean", CLEAN_CALL_SITE)
        dirty_src = _p1_source(tmp, "dirty", DIRTY_CALL_SITE)
        clean_fwd = _p1_forwarder(tmp, "fwdok", CLEAN_FORWARDER)
        dirty_fwd = _p1_forwarder(tmp, "fwdbad", DIRTY_FORWARDER)

        def verdict(capture, call=None, fwd=None):
            kw = None
            if capture is not None:
                rows, _ = p1.read_rows(Path(capture))
                kw = p1.audit_kwargs(rows)
            pos = p1.audit_positional(call, fwd)
            return p1.classify(kw, pos)[0]

        # known-negative: both axes clean -> the only verdict that unblocks
        check("both axes clean is REQUEST_REPLAYABLE",
              verdict(clean_rows, clean_src, clean_fwd) == p1.REPLAYABLE,
              verdict(clean_rows, clean_src, clean_fwd))

        # THE load-bearing case: a spotless capture with no source evidence
        # must NOT read as replayable. This is the defect the operator named.
        check("clean capture WITHOUT call-path source is not replayable",
              verdict(clean_rows, None, None) == p1.INCOMPLETE_POSITIONAL,
              verdict(clean_rows, None, None))

        # known-positive, axis A only
        extra = _p1_capture(tmp, "extra",
                            [_p1_row(), _p1_row(other_params=["seed"])])
        check("an unrecorded kwarg value is REQUEST_INCOMPLETE_KWARGS",
              verdict(extra, clean_src, clean_fwd) == p1.INCOMPLETE_KWARGS,
              verdict(extra, clean_src, clean_fwd))

        # known-positive, axis B only, two distinct mechanisms
        check("a positional argument at the call site is INCOMPLETE_POSITIONAL",
              verdict(clean_rows, dirty_src, clean_fwd)
              == p1.INCOMPLETE_POSITIONAL, "")
        check("a forwarder that inserts an argument is INCOMPLETE_POSITIONAL",
              verdict(clean_rows, clean_src, dirty_fwd)
              == p1.INCOMPLETE_POSITIONAL, "")

        # both, and it must not collapse into either single verdict
        both = verdict(extra, dirty_src, clean_fwd)
        check("both defects give REQUEST_INCOMPLETE_MULTIPLE",
              both == p1.INCOMPLETE_MULTIPLE, both)
        check("the multiple verdict is not collapsed to kwargs",
              both != p1.INCOMPLETE_KWARGS, both)
        check("the multiple verdict is not collapsed to positional",
              both != p1.INCOMPLETE_POSITIONAL, both)

        # a source root with no create call at all is not the call path,
        # so it cannot certify anything (a scope smaller than its name).
        empty_src = _p1_source(tmp, "empty", "x = 1\n")
        check("a root with no create call site cannot establish axis B",
              verdict(clean_rows, empty_src, clean_fwd)
              == p1.INCOMPLETE_POSITIONAL, "")
        check("and it says why",
              "not the production call path"
              in p1.audit_positional(empty_src, clean_fwd)["reason"], "")

        # every verdict maps to a distinct exit code, so CI can branch
        check("five verdicts, five distinct exit codes",
              len(set(p1.EXIT.values())) == 5, str(p1.EXIT))
        check("only REQUEST_REPLAYABLE exits 0",
              [v for v, e in p1.EXIT.items() if e == 0] == [p1.REPLAYABLE], "")


def test_p1_refuses_rather_than_guessing() -> None:
    scenario("p1 refuses without a subject")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        clean_src = _p1_source(tmp, "clean", CLEAN_CALL_SITE)
        clean_fwd = _p1_forwarder(tmp, "fwdok", CLEAN_FORWARDER)

        def verdict(rows):
            kw = p1.audit_kwargs(rows)
            return p1.classify(kw, p1.audit_positional(clean_src, clean_fwd))[0]

        # R11: selftest rows are not the replay population. A capture of
        # nothing but selftest rows is UNRESOLVED, never REPLAYABLE.
        check("selftest-only capture is UNRESOLVED",
              verdict([_p1_row(phase="selftest")]) == p1.UNRESOLVED, "")
        check("an empty capture is UNRESOLVED",
              verdict([]) == p1.UNRESOLVED, "")
        check("a missing capture file is UNRESOLVED",
              p1.classify(None, p1.audit_positional(clean_src, clean_fwd))[0]
              == p1.UNRESOLVED, "")
        # a row that cannot describe a request is not a clean row
        check("a row with no messages is UNRESOLVED, not replayable",
              verdict([_p1_row(messages=None)]) == p1.UNRESOLVED, "")
        check("a row with no model is UNRESOLVED, not replayable",
              verdict([_p1_row(model=None)]) == p1.UNRESOLVED, "")

        # An unparseable line is a HOLE IN THE DENOMINATOR, not a
        # footnote. Run 1 of the P1 job returned REQUEST_REPLAYABLE while
        # one capture line had failed to parse; the population of 4 was a
        # lower bound and the verdict said nothing about that. A line
        # that did not parse could be the `llm-call` row carrying the
        # very kwarg that breaks axis A, and "it probably is not" is the
        # reasoning P1 exists to refuse.
        bad = tmp / "bad.jsonl"
        bad.write_text(json.dumps(_p1_row()) + "\n{ not json\n")
        rows, notes = p1.read_rows(bad)
        check("an unparseable capture line is reported", len(notes) == 1,
              str(notes))
        check("and the readable row still counts", len(rows) == 1, "")
        check("the note carries the line's CONTENT, not just a count",
              "not json" in notes[0], notes[0])
        check("and declares its own byte size (R10)",
              "bytes" in notes[0], notes[0])
        holed = p1.audit_kwargs(rows, notes)
        check("an unparseable line forces UNRESOLVED, not REPLAYABLE",
              p1.classify(holed, p1.audit_positional(clean_src, clean_fwd))[0]
              == p1.UNRESOLVED, "")
        check("and says the population is a LOWER BOUND",
              any("LOWER BOUND" in w for w in p1.classify(
                  holed, p1.audit_positional(clean_src, clean_fwd))[1]), "")
        # known-negative: the same rows with no parse hole DO certify, so
        # the refusal is caused by the hole and by nothing else.
        whole = p1.audit_kwargs(rows, [])
        check("the same rows without a parse hole are REPLAYABLE",
              p1.classify(whole, p1.audit_positional(clean_src, clean_fwd))[0]
              == p1.REPLAYABLE, "")

        # the null ambiguity is measured and stated, not assumed away
        kw = p1.audit_kwargs([_p1_row()])
        check("recorded-None is counted per key",
              kw["null_ambiguous"]["temperature"] == 1, str(kw))
        check("and a valued key is not counted as ambiguous",
              kw["null_ambiguous"]["model"] == 0, str(kw))

        # I-8: the key list must come from the probe, not be maintained
        # beside it. If the probe starts recording a sixth key's value,
        # this fails rather than silently narrowing the audit.
        probe = (REPO / "scripts" / "security"
                 / "probe_llm_contract.py").read_text()
        for key in p1.VALUED_KEYS:
            check(f"the probe really records {key} by value",
                  f'"{key}": kwargs.get("{key}")' in probe
                  or f'"{key}": _serialise(kwargs.get("{key}"))' in probe, "")
        check("the probe records other_params as NAMES only",
              'sorted(k for k in kwargs' in probe, "")
        check("the probe forwards positional args without recording them",
              "forward(self, *args, **kwargs)" in probe
              and '"positional"' not in probe, "")



def run_all() -> None:
    test_p1_never_lets_one_clean_axis_imply_the_other()
    test_p1_refuses_rather_than_guessing()

    print(f"  inspected: {len(p1.EXIT)} P1 verdict(s) discriminated")
    print(f"  axes: keyword (artifact) and positional (call-path source)")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"P1 Replay Completeness Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
