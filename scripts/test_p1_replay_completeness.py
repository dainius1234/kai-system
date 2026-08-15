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
EXPECTED_SCENARIOS = 4
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


def _p1_capture(tmp: Path, name: str, rows: list[dict],
                declared: int | None = None, extra: str = "") -> str:
    """A capture in the D250 format: rows plus a manifest that declares
    the production-call count from a counter, not from the rows."""
    path = tmp / f"{name}.jsonl"
    body = "".join(json.dumps(r) + "\n" for r in rows)
    if declared is None:
        declared = sum(1 for r in rows if r.get("phase") == "capture")
    if declared >= 0:
        body += json.dumps({"event": "capture-manifest",
                            "declared_production_calls": declared}) + "\n"
    path.write_text(body + extra)
    return str(path)


def _p1_row(**over) -> dict:
    row = {"event": "llm-call", "phase": "capture",
           "messages": [{"role": "user", "content": "hi"}],
           "model": "qwen2.5:3b", "temperature": None,
           "response_format": {"type": "json_object"}, "tools": None,
           "other_params": [],
           "args_state": {"messages": "VALUE", "model": "VALUE",
                          "temperature": "NULL", "response_format": "VALUE",
                          "tools": "ABSENT"},
           "positional_arg_count": 0, "positional_args": []}
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
                rows, notes, manifest = p1.read_rows(Path(capture))
                kw = p1.audit_kwargs(rows, notes, manifest)
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

        def verdict(rows, declared=None):
            if declared is None:
                declared = sum(1 for r in rows if r.get("phase") == "capture")
            kw = p1.audit_kwargs(rows, [], {"declared_production_calls": declared})
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
        bad.write_text(json.dumps(_p1_row()) + "\n{ not json\n"
                       + json.dumps({"event": "capture-manifest",
                                     "declared_production_calls": 1}) + "\n")
        rows, notes, _mf = p1.read_rows(bad)
        check("an unparseable capture line is reported", len(notes) == 1,
              str(notes))
        check("and the readable row still counts", len(rows) == 1, "")
        check("the note carries the line's CONTENT, not just a count",
              "not json" in notes[0], notes[0])
        check("and declares its own byte size (R10)",
              "bytes" in notes[0], notes[0])
        holed = p1.audit_kwargs(rows, notes, {"declared_production_calls": 1})
        check("an unparseable line forces UNRESOLVED, not REPLAYABLE",
              p1.classify(holed, p1.audit_positional(clean_src, clean_fwd))[0]
              == p1.UNRESOLVED, "")
        check("and says the population is a LOWER BOUND",
              any("LOWER BOUND" in w for w in p1.classify(
                  holed, p1.audit_positional(clean_src, clean_fwd))[1]), "")
        # known-negative: the same rows with no parse hole DO certify, so
        # the refusal is caused by the hole and by nothing else.
        whole = p1.audit_kwargs(rows, [], {"declared_production_calls": 1})
        check("the same rows without a parse hole are REPLAYABLE",
              p1.classify(whole, p1.audit_positional(clean_src, clean_fwd))[0]
              == p1.REPLAYABLE, "")

        # the null ambiguity is measured and stated, not assumed away
        kw = p1.audit_kwargs([_p1_row()], [], {"declared_production_calls": 1})
        check("NULL is counted as NULL, not as absent",
              kw["presence"]["temperature"]["NULL"] == 1, str(kw["presence"]))
        check("ABSENT is counted as ABSENT, not as null",
              kw["presence"]["tools"]["ABSENT"] == 1, str(kw["presence"]))
        check("and a valued key is counted as VALUE",
              kw["presence"]["model"]["VALUE"] == 1, str(kw["presence"]))

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
        check("the probe records the three presence states",
              '"ABSENT" if k not in kwargs' in probe
              and '"NULL" if kwargs[k] is None' in probe, "")
        check("the probe records a positional argument count",
              '"positional_arg_count": len(args)' in probe, "")



def test_p1_reconciles_and_refuses_the_new_holes() -> None:
    """D250. Four ways a capture can look complete and not be, each with
    a known-positive and the clean case beside it as known-negative."""
    scenario("p1 reconciliation and format refusals")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        src = _p1_source(tmp, "clean", CLEAN_CALL_SITE)
        fwd = _p1_forwarder(tmp, "fwdok", CLEAN_FORWARDER)

        def verdict(capture):
            rows, notes, manifest = p1.read_rows(Path(capture))
            kw = p1.audit_kwargs(rows, notes, manifest)
            return p1.classify(kw, p1.audit_positional(src, fwd))

        # known-negative: the clean D250 capture certifies.
        ok = _p1_capture(tmp, "ok", [_p1_row(), _p1_row()])
        check("a clean D250 capture is REQUEST_REPLAYABLE",
              verdict(ok)[0] == p1.REPLAYABLE, str(verdict(ok)))
        check("and says the reconciliation is internal, not independent",
              any("not independent proof" in w for w in verdict(ok)[1]), "")

        # 1. a dropped production row must break reconciliation
        dropped = _p1_capture(tmp, "dropped", [_p1_row()], declared=2)
        v, why = verdict(dropped)
        check("a dropped production row fails reconciliation",
              v == p1.UNRESOLVED, v)
        check("and names both counts",
              any("2 production call(s)" in w and "1 row(s)" in w
                  for w in why), str(why))

        # 2. prose in the machine stream
        prose = _p1_capture(tmp, "prose", [_p1_row()],
                            extra="  inspected: 1 model call(s) captured\n")
        check("prose in the data stream refuses", verdict(prose)[0]
              == p1.UNRESOLVED, "")
        check("and the note carries the prose itself",
              any("inspected: 1 model call" in w for w in verdict(prose)[1]),
              str(verdict(prose)[1]))

        # 3. malformed machine rows: valid JSON, not a record
        for name, blob in (("scalar", "42"),
                           ("array", '["llm-call"]'),
                           ("eventless", '{"phase": "capture"}')):
            cap = _p1_capture(tmp, f"bad-{name}", [_p1_row()],
                              extra=blob + "\n")
            check(f"a {name} line is a malformed machine row, and refuses",
                  verdict(cap)[0] == p1.UNRESOLVED, "")

        # 4. no manifest at all -> legacy or truncated, never upgraded
        legacy_path = tmp / "legacy.jsonl"
        legacy_path.write_text(json.dumps(_p1_row()) + "\n")
        check("a capture with no manifest is UNRESOLVED",
              verdict(str(legacy_path))[0] == p1.UNRESOLVED, "")
        check("and says past captures keep their historical meaning",
              any("historical meaning" in w
                  for w in verdict(str(legacy_path))[1]), "")

        # 5. ABSENT/NULL collapsed back into a bare value -> refuse
        old = _p1_row()
        old.pop("args_state")
        collapsed = _p1_capture(tmp, "collapsed", [old])
        v, why = verdict(collapsed)
        check("a row without args_state cannot certify", v == p1.UNRESOLVED, v)
        check("and says ABSENT and NULL are indistinguishable in it",
              any("indistinguishable" in w for w in why), str(why))

        # 6. positional arguments recorded but not reconstructable
        lost = _p1_capture(tmp, "lost",
                           [_p1_row(positional_arg_count=1,
                                    positional_args=[])])
        v, why = verdict(lost)
        check("unreconstructable positional args are INCOMPLETE_POSITIONAL",
              v == p1.INCOMPLETE_POSITIONAL, v)
        check("and refuse rather than infer",
              any("must not be inferred" in w for w in why), str(why))
        kept = _p1_capture(tmp, "kept",
                           [_p1_row(positional_arg_count=1,
                                    positional_args=["payload"])])
        check("but recorded positional VALUES are replayable",
              verdict(kept)[0] == p1.REPLAYABLE, str(verdict(kept)))
        # a row that records no count at all is the pre-D250 blind spot
        blind = _p1_row()
        blind.pop("positional_arg_count")
        check("a row with no positional count is INCOMPLETE_POSITIONAL",
              verdict(_p1_capture(tmp, "blind", [blind]))[0]
              == p1.INCOMPLETE_POSITIONAL, "")



def test_format_validation_cannot_become_a_verdict() -> None:
    """D251. The format check answers one question — is this capture
    written in the D250 format — and must not be able to answer any
    other. Each criterion gets a known-positive and a known-negative."""
    scenario("format validation is bounded")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        def fc(capture):
            rows, notes, manifest = p1.read_rows(Path(capture))
            return p1.format_check(p1.audit_kwargs(rows, notes, manifest))

        good = _p1_capture(tmp, "good", [_p1_row(), _p1_row()])
        ok, items = fc(good)
        check("a D250-format capture is FORMAT VALID", ok, str(items))
        check("four criteria are reported", len(items) == 4, str(items))

        # known-positives, one per criterion
        prose = _p1_capture(tmp, "prose", [_p1_row()],
                            extra="  inspected: 1 model call(s) captured\n")
        check("prose in the stream fails criterion 1",
              fc(prose)[1][0][1] is False, str(fc(prose)[1][0]))
        drop = _p1_capture(tmp, "drop", [_p1_row()], declared=2)
        check("a count mismatch fails criterion 2",
              fc(drop)[1][1][1] is False, str(fc(drop)[1][1]))
        nostate = _p1_row(); nostate.pop("args_state")
        check("a row without args_state fails criterion 3",
              fc(_p1_capture(tmp, "nostate", [nostate]))[1][2][1] is False, "")
        bogus = _p1_row(args_state={"messages": "MAYBE", "model": "VALUE",
                                    "temperature": "NULL",
                                    "response_format": "VALUE",
                                    "tools": "ABSENT"})
        check("an invalid state label fails criterion 3",
              fc(_p1_capture(tmp, "bogus", [bogus]))[1][2][1] is False, "")
        nopos = _p1_row(); nopos.pop("positional_arg_count")
        check("a row with no positional count fails criterion 4",
              fc(_p1_capture(tmp, "nopos", [nopos]))[1][3][1] is False, "")
        lost = _p1_row(positional_arg_count=1, positional_args=[])
        check("unreconstructable positional args fail criterion 4",
              fc(_p1_capture(tmp, "lost", [lost]))[1][3][1] is False, "")
        # an empty capture must not pass by having nothing to fail on
        check("a capture with no production rows is not FORMAT VALID",
              fc(_p1_capture(tmp, "empty", []))[0] is False, "")
        check("and an unreadable capture is not FORMAT VALID",
              p1.format_check(None)[0] is False, "")

        # THE bound. Format validation returns a boolean about the file's
        # shape; it has no way to express REQUEST_REPLAYABLE, so a green
        # format run cannot be mistaken for a certified replay subject.
        check("format_check returns a bool, never a P1 verdict",
              isinstance(fc(good)[0], bool), "")
        check("and no verdict string can leak out of it",
              not any(str(i) in p1.EXIT for i in fc(good)[1]), "")
        src = (REPO / "scripts" / "security"
               / "p1_replay_completeness.py").read_text()
        check("the format path reads no response field",
              "raw_response" not in src, "")
        check("and says out loud it is not a P1 verdict",
              "This is NOT a P1 verdict. It cannot become one." in src, "")



def run_all() -> None:
    test_p1_never_lets_one_clean_axis_imply_the_other()
    test_p1_refuses_rather_than_guessing()
    test_p1_reconciles_and_refuses_the_new_holes()
    test_format_validation_cannot_become_a_verdict()

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
