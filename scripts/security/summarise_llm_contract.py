#!/usr/bin/env python3
"""KAI-GATE-048 C: the Q1/Q2/Q6 table, from the captured JSONL.

Answers only what was measured, and refuses the four questions it cannot
answer from the capture. It authorises no remedy: whether the mismatch is
owned by prompt construction, adapter mode selection, model compliance or
the validator is a conclusion the operator draws from this table, not one
this file asserts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from classify_llm_response import (  # noqa: E402
    classify, canonical, sha256, licenses_raw_response_claim,
    CLASSIFIER_UNMEASURED, RAW_MODEL_RESPONSE, SCHEMA_ECHO, VALID_INSTANCE)


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


def _schema_for(call: Dict[str, Any]) -> tuple:
    """(schema object, how it was conveyed, canonical text).

    PREFERS the structured `response_model` the record already carries.
    Run 15 fell back to scraping the schema out of message prose, which
    parses as text but not as JSON — so the requirements collapsed to
    empty and every verdict went vacuous (D216). Prose is still recorded
    as the CONVEYANCE, because for json_mode that is the truth; it is
    just not the thing to validate against.
    """
    rm = call.get("response_model")
    if isinstance(rm, dict) and rm:
        conveyance = "message text (json_mode)"
        if call.get("response_format"):
            conveyance = "response_format"
        elif call.get("tools"):
            conveyance = "tools"
        return rm, conveyance, canonical(rm)
    if call.get("response_format"):
        rf = call["response_format"]
        return rf, "response_format", canonical(rf)
    if call.get("tools"):
        return call["tools"], "tools", canonical(call["tools"])
    for msg in call.get("messages") or []:
        content = str(msg.get("content", ""))
        if '"properties"' in content or "'properties'" in content:
            # Conveyance is knowable; the schema OBJECT is not, and that
            # must surface as unmeasured rather than as an empty contract.
            return None, "message text (unparsed)", content
    return None, "NOT LOCATED", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True)
    args = ap.parse_args()
    rows = load(Path(args.capture))

    calls = [r for r in rows if r.get("event") == "llm-call"]
    resolved = next((r for r in rows if r.get("event") == "resolved-config"), {})
    print(f"  inspected: {len(calls)} model call(s) in {args.capture}")
    if not calls:
        print("  NOT COLLECTED: no llm-call records. Q1/Q2/Q6 are UNMEASURED,")
        print("  which is not the same as 'no mismatch'.")
        return 1
    print()

    # ── Q1: the effective mode, READ not inferred ────────────────────
    print("  Q1a. EFFECTIVE STRUCTURED-OUTPUT MODE — read at runtime")
    print(f"       config llm_instructor_mode : {resolved.get('config_llm_instructor_mode', '?')}")
    print(f"       adapter instructor_mode    : {resolved.get('adapter_instructor_mode', '?')}")
    print(f"       adapter class default      : {resolved.get('adapter_default_mode', '?')}")
    print(f"       instructor client .mode    : {resolved.get('instructor_client_mode', '?')}")
    print(f"       adapter class              : {resolved.get('adapter_class', '?')}")
    print("       An empty config field is NOT proof of the effective mode;")
    print("       the client's own value is the measurement.")
    print()

    print("  Q1b/Q2/Q6. PER ATTEMPT")
    print("    prompt/schema hashes are CANONICAL identity; the response")
    print("    hash is BYTE identity of the returned content string (D209).")
    print(f"    {'#':>2} {'layer':<19} {'conveyance':<26} {'prompt':>10} "
          f"{'schema':>10} {'response':>10}  {'classification':<22} {'elapsed':>8}")
    seen_prompt, seen_schema, seen_resp = set(), set(), set()
    verdicts = []
    for call in calls:
        layer = call.get("layer")
        prompt_text = canonical(call.get("messages"))
        schema_obj, conveyance, schema_text = _schema_for(call)
        raw = call.get("raw_response")
        verdict, why = classify(raw, schema_obj)
        verdicts.append((call.get("attempt"), layer, verdict, why, conveyance))
        ph, sh, rh = sha256(prompt_text), sha256(schema_text), sha256(raw)
        seen_prompt.add(ph); seen_schema.add(sh); seen_resp.add(rh)
        print(f"    {call.get('attempt', '?'):>2} {str(layer):<19} "
              f"{conveyance:<26} {ph[:10]} {sh[:10]} {rh[:10]}  "
              f"{verdict:<22} {call.get('elapsed_s', '?'):>8}")
    print()
    for attempt, layer, verdict, why, conveyance in verdicts:
        print(f"    attempt {attempt} [{layer}]: {verdict}")
        print(f"      {why}")
        print(f"      schema conveyance: {conveyance}")
        if verdict in (VALID_INSTANCE, SCHEMA_ECHO) \
                and not licenses_raw_response_claim(layer):
            print(f"      LAYER LIMIT: {layer} does NOT license a claim about")
            print(f"      the MODEL's reply — instructor retries, parses and")
            print(f"      validates between there and RAW_MODEL_RESPONSE.")
    print()

    # ── Q6 ────────────────────────────────────────────────────────────
    print("  Q6. REPRODUCIBILITY — measured by hash, not by eye")
    print(f"      distinct prompts   : {len(seen_prompt)} across {len(calls)} call(s)")
    print(f"      distinct schemas   : {len(seen_schema)}")
    print(f"      distinct responses : {len(seen_resp)}")
    kinds = sorted({v for _, _, v, _, _ in verdicts})
    print(f"      classifications    : {', '.join(kinds)}")
    if len(seen_resp) == 1 and len(calls) > 1:
        print("      Every attempt returned a BYTE-IDENTICAL response.")
    elif len(kinds) == 1 and len(calls) > 1:
        print("      Responses differ byte-wise but are the SAME KIND — "
              "distinct failures are not being collapsed.")
    print()

    # ── what this refuses to conclude ────────────────────────────────
    print("  OWNERSHIP — NOT CONCLUDED HERE")
    print("     Prompt construction, adapter mode selection, model compliance")
    print("     and the validator are four different owners with four")
    print("     different remedies. This table is the evidence for that")
    print("     decision; it is not the decision. No remedy is authorised:")
    print("     no mode change, no model swap, no timeout or retry change,")
    print("     no schema edit, no validator change.")

    raw_layer = [v for _, layer, v, _, _ in verdicts
                 if licenses_raw_response_claim(layer)]
    if not raw_layer:
        print()
        print("  Q2 — UNMEASURED. No captured row came from")
        print(f"  {RAW_MODEL_RESPONSE}, so nothing here describes what the")
        print("  model actually returned per attempt. A validated object, an")
        print("  instructor return value or a parsed result is NOT a")
        print("  substitute (D215).")
        return 1
    if any(v == CLASSIFIER_UNMEASURED for v in raw_layer):
        print()
        print("  CLASSIFIER UNMEASURED on at least one attempt: the schema")
        print("  requirements could not be established, so no verdict about")
        print("  the response kind is possible. This is not a pass.")
        return 1
    if any(v == SCHEMA_ECHO for v in raw_layer):
        print()
        print("  MEASURED at RAW_MODEL_RESPONSE: at least one attempt returned")
        print("  the SCHEMA where an INSTANCE was required. A measurement")
        print("  result, not an instrument failure.")
        return 1
    if all(v == VALID_INSTANCE for v in raw_layer):
        print()
        print("  MEASURED at RAW_MODEL_RESPONSE: every raw attempt returned a")
        print("  valid instance. The mismatch is NOT at the model's response")
        print("  kind.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
