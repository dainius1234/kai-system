#!/usr/bin/env python3
"""KAI-GATE-048 C: the Q1/Q2/Q6 table, from the captured JSONL.

Answers only what was measured, and names what it could not establish. It
authorises no remedy: whether the mismatch is owned by prompt
construction, adapter mode selection, model compliance or the validator is
a conclusion the operator draws from this table, not one this file asserts.

D222 changed where the contract comes from. Each attempt's schema is
recovered FROM THAT ATTEMPT's own system message, because instructor
compiles `response_model` into the system prose under `Mode.JSON` and the
Python object never reaches the raw boundary. `response_format` is
recorded as what it is — a mode directive — and is never treated as a
contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_llm_response import (  # noqa: E402
    classify, canonical, sha256_bytes, sha256_canonical,
    licenses_raw_response_claim, recover_contract, extraction_rule_provenance,
    logical_call_grouping, validator_status,
    CONTRACT_UNMEASURED, PASSING_VERDICTS, RAW_MODEL_RESPONSE,
    REQUIRED_FIELDS_PRESENT, SCHEMA_ECHO, VALID_INSTANCE)


def load(path: Path) -> List[dict]:
    rows = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def _mode_directive(call: Dict[str, Any]) -> str:
    """What `response_format` actually was. NOT the contract (D222 §2)."""
    rf = call.get("response_format")
    if rf is None:
        return "none"
    if isinstance(rf, dict):
        return canonical(rf)
    return str(rf)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    args = ap.parse_args()
    rows = load(Path(args.capture))

    calls = [r for r in rows if r.get("event") == "llm-call"
             and r.get("phase") != "selftest"]
    selftest_rows = [r for r in rows if r.get("event") == "llm-call"
                     and r.get("phase") == "selftest"]
    resolved = next((r for r in rows if r.get("event") == "resolved-config"), {})
    print(f"  inspected: {len(calls)} production model call(s) in {args.capture}")
    print(f"  excluded : {len(selftest_rows)} selftest row(s) — instrument "
          f"capability, not evidence about cognify")
    if not calls:
        print("  NOT COLLECTED: no production llm-call records. Q1/Q2/Q6 are")
        print("  UNMEASURED, which is not the same as 'no mismatch'.")
        return 1
    print()

    prov = extraction_rule_provenance()
    val = validator_status()

    # ── how this analysis was constituted ────────────────────────────
    print("  HOW THE CONTRACT WAS RECOVERED — provenance, not assumption")
    print(f"       rule                : {prov['primary_rule']}")
    print(f"       instructor available: {prov['instructor_available']}"
          f"  version {prov['instructor_version']}")
    print(f"       source              : {prov['source']}")
    print(f"       source sha256       : {(prov['source_sha256'] or '')[:16]}")
    print(f"       corroboration       : {prov.get('corroboration')}")
    print(f"       JSON Schema validator: {val['available']}"
          f"  {val.get('library')} {val.get('version')}")
    if not val["available"]:
        print(f"       -> {val.get('consequence')}")
    print()

    # ── Q1a: the effective mode, READ not inferred ───────────────────
    print("  Q1a. EFFECTIVE STRUCTURED-OUTPUT MODE — read at runtime")
    print(f"       config llm_instructor_mode : {resolved.get('config_llm_instructor_mode', '?')}")
    print(f"       adapter instructor_mode    : {resolved.get('adapter_instructor_mode', '?')}")
    print(f"       adapter class default      : {resolved.get('adapter_default_mode', '?')}")
    print(f"       instructor client .mode    : {resolved.get('instructor_client_mode', '?')}")
    print(f"       adapter class              : {resolved.get('adapter_class', '?')}")
    print("       An empty config field is NOT proof of the effective mode;")
    print("       the client's own value is the measurement.")
    print()

    print("  Q1b/Q2. PER ATTEMPT")
    print("    prompt/schema hashes are CANONICAL identity; the response")
    print("    hash is BYTE identity of the returned string. Different")
    print("    questions, never interchangeable (D209).")
    print(f"    {'#':>2} {'layer':<19} {'contract':<10} {'prompt~c':>9} "
          f"{'schema~c':>9} {'resp~b':>9}  {'classification':<24} {'elapsed':>8}")
    verdicts = []
    seen_prompt, seen_schema, seen_resp = set(), set(), set()
    for call in calls:
        layer = call.get("layer")
        messages = call.get("messages")
        schema, why_contract, detail = recover_contract(messages, prov)
        raw = call.get("raw_response")
        verdict, why = classify(raw, schema, val)

        ph = sha256_canonical(messages)
        sh = sha256_canonical(schema) if schema else ""
        rh = sha256_bytes(raw)
        seen_prompt.add(ph)
        seen_schema.add(sh)
        seen_resp.add(rh)
        verdicts.append({
            "attempt": call.get("attempt"), "layer": layer, "verdict": verdict,
            "why": why, "why_contract": why_contract, "detail": detail,
            "directive": _mode_directive(call), "schema": schema,
        })
        print(f"    {str(call.get('attempt', '?')):>2} {str(layer):<19} "
              f"{('RECOVERED' if schema else 'UNMEASURED'):<10} "
              f"{ph[:9]} {(sh[:9] or '—'):>9} {rh[:9]}  "
              f"{verdict:<24} {str(call.get('elapsed_s', '?')):>8}")
    print()

    for v in verdicts:
        print(f"    attempt {v['attempt']} [{v['layer']}]: {v['verdict']}")
        print(f"      contract: {v['why_contract']}")
        print(f"      {v['why']}")
        print(f"      response_format (MODE DIRECTIVE, not the contract): "
              f"{v['directive']}")
        if v["verdict"] in (VALID_INSTANCE, SCHEMA_ECHO, REQUIRED_FIELDS_PRESENT) \
                and not licenses_raw_response_claim(v["layer"]):
            print(f"      LAYER LIMIT: {v['layer']} does NOT license a claim")
            print(f"      about the MODEL's reply — instructor retries, parses")
            print(f"      and validates between there and {RAW_MODEL_RESPONSE}.")
    print()

    # ── Q6 ───────────────────────────────────────────────────────────
    grouping = logical_call_grouping(calls)
    print("  Q6. REPRODUCIBILITY — and why it is not answered here")
    print(f"      raw attempts       : {len(calls)}")
    print(f"      distinct prompts   : {len(seen_prompt)} (canonical)")
    print(f"      distinct contracts : {len(seen_schema)} (canonical)")
    print(f"      distinct responses : {len(seen_resp)} (byte)")
    if grouping["available"]:
        print(f"      logical calls      : {len(grouping['groups'])} "
              f"via '{grouping['key']}' -> {grouping['groups']}")
        retried = [c for c, n in grouping["groups"].items() if n > 1]
        print(f"      calls that RETRIED : {len(retried)}")
        print("      Q6 IS NOT ANSWERED BY THE EXISTENCE OF IDS. It needs")
        print("      logical calls that actually retried, and — if Q6's")
        print("      definition requires retry-level behaviour to recur")
        print("      ACROSS runs — a SECOND independent correlated capture.")
        print("      One correlated run is not two.")
    else:
        print("      logical calls      : UNAVAILABLE — "
              f"{grouping['why']}")
        print(f"      looked for         : {', '.join(grouping['looked_for'])}")
        print(f"      REFUSED as grouping signals: "
              f"{', '.join(grouping['refused_signals'])}")
        print("      cognee passes max_retries=2, so a logical call is at most")
        print("      TWO raw attempts; these rows therefore span several")
        print("      distinct calls. Counting them as one retry sequence would")
        print("      manufacture a denominator.")
        print(f"      NEXT MEASUREMENT REQUIREMENT: "
              f"{grouping['next_measurement_requirement']}")
    print()

    # ── Q6: per logical call, only when correlation exists ───────────
    if grouping["available"]:
        key = grouping["key"]
        order = []
        by_call: Dict[str, List[dict]] = {}
        for call, v in zip(calls, verdicts):
            cid = str(call.get(key))
            if cid not in by_call:
                by_call[cid] = []
                order.append(cid)
            by_call[cid].append({
                "attempt_index": call.get("attempt_index"),
                "verdict": v["verdict"],
                "schema": v["schema"],
                "prompt": sha256_canonical(call.get("messages")),
                "response": sha256_bytes(call.get("raw_response")),
                "messages": call.get("messages"),
            })
        print("  Q6a. PER LOGICAL CALL")
        print(f"    {'logical_call_id':<18} {'contract~c':>10} {'n':>2} "
              f"{'classifications':<26} {'prompt~c':<12} {'resp~b'}")
        for cid in order:
            rows_ = sorted(by_call[cid],
                           key=lambda r: (r["attempt_index"] or 0))
            kinds = ",".join(sorted({r["verdict"] for r in rows_}))
            print(f"    {cid:<18} "
                  f"{(sha256_canonical(rows_[0]['schema'])[:10] if rows_[0]['schema'] else '—'):>10} "
                  f"{len(rows_):>2} {kinds:<26} "
                  f"{','.join(r['prompt'][:8] for r in rows_):<12} "
                  f"{','.join(r['response'][:8] for r in rows_)}")
        print()

        print("  Q6b. WITHIN-CALL REPRODUCIBILITY")
        multi = [cid for cid in order if len(by_call[cid]) > 1]
        print(f"      logical calls           : {len(order)}")
        print(f"      with more than 1 attempt: {len(multi)}")
        if not multi:
            print("      NO logical call retried. A single-attempt call gives")
            print("      NO retry-reproducibility evidence — this is not")
            print("      'the retry behaved', it is 'no retry was observed'.")
        for cid in multi:
            rows_ = sorted(by_call[cid],
                           key=lambda r: (r["attempt_index"] or 0))
            same_contract = len({sha256_canonical(r["schema"]) for r in rows_}) == 1
            same_kind = len({r["verdict"] for r in rows_}) == 1
            same_bytes = len({r["response"] for r in rows_}) == 1
            first, last = rows_[0], rows_[-1]
            grew = (len(json.dumps(last["messages"] or []))
                    - len(json.dumps(first["messages"] or [])))
            print(f"      {cid}: {len(rows_)} attempt(s)")
            print(f"        same contract on retry : {same_contract}")
            print(f"        repair context changed : "
                  f"{'yes' if first['prompt'] != last['prompt'] else 'no'}"
                  f" ({grew:+d} bytes of messages)")
            print(f"        same failure class     : {same_kind} "
                  f"({', '.join(r['verdict'] for r in rows_)})")
            print(f"        byte-identical replies : {same_bytes}")
        print()

    # ── what this refuses to conclude ────────────────────────────────
    print("  OWNERSHIP — NOT CONCLUDED HERE")
    print("     Prompt construction, adapter mode selection, model compliance")
    print("     and the validator are four different owners with four")
    print("     different remedies. This table is the evidence for that")
    print("     decision; it is not the decision. No remedy is authorised.")

    raw_layer = [v for v in verdicts if licenses_raw_response_claim(v["layer"])]
    if not raw_layer:
        print()
        print(f"  Q2 — UNMEASURED. No production row came from")
        print(f"  {RAW_MODEL_RESPONSE}, so nothing here describes what the")
        print("  model actually returned per attempt (D215).")
        return 1

    kinds = sorted({v["verdict"] for v in raw_layer})
    print()
    print(f"  Q2 at {RAW_MODEL_RESPONSE}: {len(raw_layer)} attempt(s), "
          f"verdicts {', '.join(kinds)}")

    if any(v["verdict"] == CONTRACT_UNMEASURED for v in raw_layer):
        print("  CONTRACT UNMEASURED on at least one attempt: its schema could")
        print("  not be recovered from the attempt itself, so no verdict about")
        print("  the response kind is possible there. This is not a pass.")
        return 1
    if not val["available"]:
        print("  NO JSON SCHEMA VALIDATOR was available, so the strongest")
        print(f"  claim reachable is {REQUIRED_FIELDS_PRESENT}, which is NOT")
        print(f"  {VALID_INSTANCE}: types, nested objects and every other")
        print("  constraint went untested. Q2 is PARTIALLY measured.")
        return 1
    if any(v["verdict"] == SCHEMA_ECHO for v in raw_layer):
        print("  MEASURED: at least one attempt returned the SCHEMA where an")
        print("  INSTANCE was required. A measurement result, not an")
        print("  instrument failure.")
        return 1
    if all(v["verdict"] in PASSING_VERDICTS for v in raw_layer):
        print("  MEASURED: every raw attempt validated against the contract it")
        print("  was actually sent. The mismatch is NOT at the model's")
        print("  response kind.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
