#!/usr/bin/env python3
"""Calibration for the S1 replay-subject selector.

Two properties, and the second is the one that would be expensive to get
wrong quietly:

1. **The five preconditions refuse.** Each gets a case that must fail and
   the clean case beside it, because a precondition that cannot fire is
   decoration and would let selection fall through to a row S1 never
   licensed.

2. **No response-bearing value can reach the output.** Asserted against
   rows whose response fields carry a sentinel string: if that string
   appears anywhere in the projection, the boundary D257 drew has been
   crossed. This is checked over the projection of a row that HAS every
   response field populated, so the test would notice a leak rather than
   merely observing that absent fields stayed absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import select_replay_subject as s  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 3
executed: list[str] = []
LEAK = "THIS-IS-A-RESPONSE-AND-MUST-NEVER-APPEAR"


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def row(**over) -> dict:
    """A production row with EVERY response field populated, so a leak
    has something to leak."""
    r = {
        "event": "llm-call", "phase": "capture", "seq": 5,
        "logical_call_id": "lc-abc123", "attempt_index": 1,
        "outside_logical_call": False,
        "messages": [{"role": "system", "content": "be brief"},
                     {"role": "user", "content": "hello"}],
        "model": "qwen2.5:3b", "temperature": None,
        "response_format": {"type": "json_object"}, "tools": None,
        "other_params": [],
        "args_state": {"messages": "VALUE", "model": "VALUE",
                       "temperature": "ABSENT",
                       "response_format": "VALUE", "tools": "ABSENT"},
        "positional_arg_count": 0, "positional_args": [],
        # every response-bearing field, each carrying the sentinel
        "raw_response": LEAK, "finish_reason": LEAK, "result_type": LEAK,
        "transport_error": LEAK, "raw_response_note": LEAK,
        "elapsed_s": 57.0, "layer": "RAW_MODEL_RESPONSE", "wall": LEAK,
    }
    r.update(over)
    return r


def test_lowest_seq_wins_and_only_that() -> None:
    scenario("lowest seq is selected")
    rows = [row(seq=9, logical_call_id="lc-late"),
            row(seq=3, logical_call_id="lc-first"),
            row(seq=7, logical_call_id="lc-mid")]
    sel, checks, refusals = s.select(rows)
    check("a row is selected", sel is not None, str(refusals))
    check("and it is the lowest seq", sel and sel["seq"] == 3, str(sel and sel["seq"]))
    check("not the first in file order", sel and sel["logical_call_id"] == "lc-first")
    check("no refusals on the clean case", refusals == [], str(refusals))
    check("all five preconditions are reported", len(checks) == 5, str(checks))
    # selftest rows are not candidates
    sel2, _, ref2 = s.select([row(seq=1, phase="selftest"), row(seq=4)])
    check("a selftest row is not selectable", sel2 and sel2["seq"] == 4, str(sel2))
    check("an empty production population refuses",
          s.select([row(phase="selftest")])[0] is None)


def test_each_precondition_refuses() -> None:
    scenario("preconditions refuse")

    def refused(rows) -> tuple[bool, str]:
        sel, _, ref = s.select(rows)
        return sel is None, " ".join(ref)

    # 2 — seq missing, wrong type, or out of range
    for label, bad in (("missing", {}), ("string", {"seq": "3"}),
                       ("bool", {"seq": True}), ("zero", {"seq": 0})):
        r = row()
        if label == "missing":
            r.pop("seq")
        else:
            r.update(bad)
        ok, why = refused([r])
        check(f"precondition 2 refuses a {label} seq", ok, why)
        check(f"and names precondition 2 for {label}", "precondition 2" in why, why)

    # 3 — duplicate minimum
    ok, why = refused([row(seq=2, logical_call_id="a"),
                       row(seq=2, logical_call_id="b")])
    check("precondition 3 refuses a duplicated minimum seq", ok, why)
    check("and calls it an instrument defect, not a choice",
          "not a choice to make" in why, why)

    # 4 — no logical_call_id
    for bad in (None, ""):
        ok, why = refused([row(logical_call_id=bad)])
        check(f"precondition 4 refuses logical_call_id={bad!r}", ok, why)
        check("and says the justification is unlicensed",
              "not licensed" in why, why)

    # 5 — attempt_index not 1, INCLUDING null
    for bad in (2, None, "1"):
        ok, why = refused([row(attempt_index=bad)])
        check(f"precondition 5 refuses attempt_index={bad!r}", ok, why)
    r = row(); r.pop("attempt_index")
    ok, why = refused([r])
    check("precondition 5 refuses a missing attempt_index", ok, why)
    check("and explains a later attempt is a different request",
          "different request" in why, why)

    # known-negative: the clean row still selects, so the refusals above
    # are caused by the defects and not by the harness.
    check("the clean row is still selected", s.select([row()])[0] is not None)


def test_no_response_value_can_reach_the_output() -> None:
    scenario("the response boundary holds")
    r = row()
    proj = s.projection(r)
    blob = json.dumps(proj, default=str)

    check("the sentinel appears in the SOURCE row", LEAK in json.dumps(r))
    check("and NOWHERE in the projection", LEAK not in blob, blob[:300])
    for field in s.RESPONSE_BEARING:
        check(f"{field} is not a projection key", field not in proj, str(sorted(proj)))
    check("elapsed_s's value does not appear either",
          "57.0" not in blob and "57" not in blob.replace("qwen2.5", ""),
          blob[:300])

    # the locator IS present, so the row can be identified later
    for field in ("seq", "logical_call_id", "attempt_index", "prompt_hash",
                  "contract_hash"):
        check(f"{field} IS published", field in proj, str(sorted(proj)))
    check("the prompt hash is a hash, not the prompt",
          "hello" not in blob and len(proj["prompt_hash"]) == 12, blob[:200])

    # no hash of the complete stored row — outcome-derived, withheld (D257)
    check("no full-row hash is published",
          not any("row_hash" in k or k == "row_sha" for k in proj),
          str(sorted(proj)))
    src = (REPO / "scripts" / "security" / "select_replay_subject.py").read_text()
    check("and the module says why it is withheld",
          "hash of the complete stored row" in src and "NOT published" in src)

    # the allow-list must be an allow-list: an unknown future field is
    # withheld by default rather than passed through.
    proj2 = s.projection(row(some_future_field=LEAK))
    check("an unclassified new field does not leak",
          LEAK not in json.dumps(proj2, default=str))


def run_all() -> None:
    test_lowest_seq_wins_and_only_that()
    test_each_precondition_refuses()
    test_no_response_value_can_reach_the_output()
    print(f"  inspected: {len(s.REQUEST_SIDE)} request-side field(s) "
          f"allowed, {len(s.RESPONSE_BEARING)} response-bearing field(s) "
          f"withheld")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"S1 Selection Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
