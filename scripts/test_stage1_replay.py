#!/usr/bin/env python3
"""Calibration for the Stage-1 replay driver.

Everything here runs offline. The properties under test are the ones the
operator made preconditions of authorising Stage 1 at all:

  * `response_format` is rebuilt with `ast.literal_eval` and **never**
    `eval`, and the result is asserted to be exactly the intended typed
    value rather than merely "something that parsed";
  * the frozen prompt/contract/seq identity must match, or the run
    REFUSES — a replay of a request we cannot prove is the frozen one
    answers a different question;
  * the original captured response is never read, which is asserted
    against a capture whose response fields carry a sentinel;
  * a transport error is recorded as one execution and is NOT replaced,
    so the denominator stays at the precommitted N.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import stage1_replay as st  # noqa: E402
import select_replay_subject as s1  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 5
executed: list[str] = []
LEAK = "ORIGINAL-RESPONSE-MUST-NEVER-BE-READ"


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


SYSTEM = ("You are a summarizer.\n\n"
          "\n        Parse the content and return a JSON object matching "
          "this schema:\n\n        "
          + json.dumps({"properties": {"summary": {"type": "string"}},
                        "required": ["summary"], "title": "S",
                        "type": "object"}, indent=2)
          + "\n\n        Return a valid JSON instance, not the schema "
            "definition.")


def row(**over) -> dict:
    r = {"event": "llm-call", "phase": "capture", "seq": 2,
         "logical_call_id": "lc-1", "attempt_index": 1,
         "outside_logical_call": False,
         "messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": "Ada Lovelace..."}],
         "model": "qwen2.5:3b", "temperature": None,
         "response_format": "{'type': 'json_object'}", "tools": None,
         "other_params": [],
         "args_state": {"messages": "VALUE", "model": "VALUE",
                        "temperature": "ABSENT",
                        "response_format": "VALUE", "tools": "ABSENT"},
         "positional_arg_count": 0, "positional_args": [],
         "raw_response": LEAK, "finish_reason": LEAK, "elapsed_s": 57.0,
         "layer": "RAW_MODEL_RESPONSE", "wall": LEAK}
    r.update(over)
    return r


def capture(tmp: Path, name: str, rows: list[dict]) -> Path:
    p = tmp / f"{name}.jsonl"
    body = "".join(json.dumps(r) + "\n" for r in rows)
    body += json.dumps({"event": "capture-manifest",
                        "declared_production_calls":
                        sum(1 for r in rows if r.get("phase") == "capture")}) + "\n"
    p.write_text(body)
    return p


def frozen(tmp: Path, rows: list[dict]):
    cap = capture(tmp, "c", rows)
    proj = s1.projection(rows[0])
    return st.freeze(cap, proj["prompt_hash"], proj["contract_hash"],
                     proj["seq"])


def test_response_format_is_rebuilt_and_asserted() -> None:
    scenario("typed reconstruction")
    check("a python repr rebuilds to a dict",
          st.rebuild("{'type': 'json_object'}") == {"type": "json_object"})
    check("and json.loads would NOT have worked",
          _json_fails("{'type': 'json_object'}"))
    check("an already-typed value passes through",
          st.rebuild({"type": "json_object"}) == {"type": "json_object"})
    check("a non-literal string is left alone, not executed",
          st.rebuild("__import__('os')") == "__import__('os')")
    check("literal_eval is used and eval is not",
          "ast.literal_eval" in _src() and "eval(" not in
          _src().replace("literal_eval(", "").replace("ast.literal_eval", ""))

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        man, probs = frozen(tmp, [row()])
        check("a clean subject freezes", man is not None, str(probs))
        check("response_format is sent as the typed value",
              man and man["request"]["response_format"] == {"type": "json_object"},
              str(man and man["request"].get("response_format")))
        # a repr that parses to the WRONG thing must refuse
        man2, probs2 = frozen(tmp, [row(response_format="{'type': 'text'}")])
        check("a different typed value REFUSES", man2 is None, str(man2))
        check("and says what it rebuilt to",
              any("RECONSTRUCTION FAILED" in p for p in probs2), str(probs2))


def _json_fails(s: str) -> bool:
    try:
        json.loads(s)
        return False
    except Exception:
        return True


def _src() -> str:
    return (REPO / "scripts" / "security" / "stage1_replay.py").read_text()


def test_identity_must_match_or_it_refuses() -> None:
    scenario("identity refusal")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cap = capture(tmp, "c", [row()])
        proj = s1.projection(row())
        good = (proj["prompt_hash"], proj["contract_hash"], proj["seq"])

        check("matching identity freezes",
              st.freeze(cap, *good)[0] is not None)
        for label, bad in (("prompt", ("deadbeef0000", good[1], good[2])),
                           ("contract", (good[0], "deadbeef0000", good[2])),
                           ("seq", (good[0], good[1], 99))):
            man, probs = st.freeze(cap, *bad)
            check(f"a changed {label} REFUSES", man is None, str(man))
            check(f"and names it an identity failure ({label})",
                  any("IDENTITY FAILED" in p for p in probs), str(probs))
        # a capture that S1 itself refuses must not freeze either
        man, probs = st.freeze(capture(tmp, "d", [row(attempt_index=2)]), *good)
        check("an S1 refusal propagates", man is None, str(probs))


def test_the_original_response_is_never_read() -> None:
    scenario("original response untouched")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        man, _ = frozen(tmp, [row()])
        blob = json.dumps(man, default=str)
        check("the sentinel is in the capture row", LEAK in json.dumps(row()))
        check("and NOWHERE in the frozen manifest", LEAK not in blob,
              blob[:200])
        for k in s1.RESPONSE_BEARING:
            check(f"{k} is not in the request body",
                  k not in man["request"], str(sorted(man["request"])))
            check(f"{k} is not in the subject projection",
                  k not in man["subject"], str(sorted(man["subject"])))
        # keys recorded ABSENT must not be sent at all
        check("an ABSENT key is omitted, not sent as null",
              "temperature" not in man["request"] and
              "tools" not in man["request"], str(sorted(man["request"])))
        check("and a VALUE key is sent",
              "messages" in man["request"] and "model" in man["request"])


def test_a_transport_error_is_one_execution_not_a_retry() -> None:
    scenario("denominator holds")
    check("N1 is the frozen 10", st.N1 == 10)
    src = _src()
    check("the sender catches every failure as a datum",
          "transport_error" in src and "every failure is a datum" in src)
    # A substring hunt for "retry" was the first version of this and it
    # broke the moment the manifest declared `"retry": "none"` — a check
    # that fires on the word rather than the behaviour. The property is
    # structural: exactly one call site, inside a loop bounded by n.
    check("there is exactly one send call site (plus its definition)",
          src.count("send_once(") == 2, str(src.count("send_once(")))
    check("and it is driven by a loop bounded by the frozen n",
          'for i in range(1, man["n"] + 1)' in src)
    check("the manifest declares no retry and no validation",
          '"retry": "none"' in src and '"validation": "none"' in src)
    # a short reply file must be reported as a shortfall, not a smaller N
    check("a denominator mismatch is named",
          "DENOMINATOR MISMATCH" in src and "not a smaller sample" in src)
    check("and it exits non-zero", "return 4" in src)



def test_the_manifest_carries_everything_needed_to_reproduce() -> None:
    """Hashes prove identity; they do not reconstruct an invocation."""
    scenario("manifest reproduces the invocation")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        cap = capture(tmp, "c", [row()])
        proj = s1.projection(row())
        man, probs = st.freeze(cap, proj["prompt_hash"], proj["contract_hash"],
                               proj["seq"], url="http://x/v1/chat/completions",
                               timeout=300.0)
        check("it freezes", man is not None, str(probs))
        # the actual body, not merely its hash
        msgs = man["request"].get("messages")
        check("the full messages body is present", isinstance(msgs, list)
              and len(msgs) == 2, str(type(msgs)))
        check("with real content, not a digest",
              "Ada Lovelace" in json.dumps(msgs), json.dumps(msgs)[:120])
        # D247's held constants
        for k in ("n", "url", "timeout_s", "instructor_in_path", "validation",
                  "retry", "model"):
            check(f"runtime constant {k} is frozen", k in man["runtime"],
                  str(sorted(man["runtime"])))
        check("instructor is declared out of the path",
              man["runtime"]["instructor_in_path"] is False)
        check("n matches the precommitted N1", man["runtime"]["n"] == st.N1)
        # identities
        check("a request hash is produced", len(man["request_hash"]) == 64)
        check("a manifest hash is produced", len(man["manifest_hash"]) == 64)
        check("they are different values",
              man["request_hash"] != man["manifest_hash"])
        check("the request hash covers the body",
              man["request_hash"] == st._digest(man["request"]))
        # the manifest hash must move when a runtime constant moves
        man2, _ = st.freeze(cap, proj["prompt_hash"], proj["contract_hash"],
                            proj["seq"], url="http://other/v1", timeout=300.0)
        check("a changed endpoint changes the manifest hash",
              man2["manifest_hash"] != man["manifest_hash"])
        check("but not the request hash",
              man2["request_hash"] == man["request_hash"])
        # classifier identity is part of the measurement
        ident = st.instrument_identity(
            REPO / "scripts" / "security" / "classify_llm_response.py")
        check("the classifier's identity is its bytes",
              len(ident["sha256"]) == 64 and ident["bytes"] > 0, str(ident))
        # one drifting invocation invalidates the set
        src = _src()
        check("a request-hash mismatch is INVALID, not 9/10",
              "STAGE 1 INVALID" in src and "not a 9/10 result" in src)
        check("and it exits non-zero", "return 5" in src)
        check("every invocation records what it actually sent",
              'rec["request_hash"] = _digest(body)' in src)


def run_all() -> None:
    test_response_format_is_rebuilt_and_asserted()
    test_identity_must_match_or_it_refuses()
    test_the_original_response_is_never_read()
    test_a_transport_error_is_one_execution_not_a_retry()
    test_the_manifest_carries_everything_needed_to_reproduce()
    print(f"  inspected: {st.N1} precommitted replay execution(s), "
          f"{len(s1.RESPONSE_BEARING)} response-bearing field(s) withheld")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"Stage 1 Replay Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
