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
    classify, required_fields_of, sha256, SCHEMA_ECHO, VALID_INSTANCE)


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


def _schema_text(call: Dict[str, Any]) -> str:
    """The schema material actually sent, wherever the adapter put it.

    json_mode embeds it in the messages; other modes use response_format
    or tools. Looking in only one place would report 'no schema sent' for
    a mode that simply carries it elsewhere — a scope narrower than the
    claim (R5)."""
    if call.get("response_format"):
        return json.dumps(call["response_format"], sort_keys=True, default=str)
    if call.get("tools"):
        return json.dumps(call["tools"], sort_keys=True, default=str)
    for msg in call.get("messages") or []:
        content = str(msg.get("content", ""))
        if '"properties"' in content or "'properties'" in content:
            return content
    return ""


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
    print(f"    {'#':>2}  {'prompt':>10} {'schema':>10} {'response':>10}  "
          f"{'classification':<24} {'elapsed':>8}")
    seen_prompt, seen_schema, seen_resp = set(), set(), set()
    verdicts = []
    for call in calls:
        prompt_text = json.dumps(call.get("messages"), sort_keys=True, default=str)
        schema_text = _schema_text(call)
        raw = call.get("raw_response")
        fields = required_fields_of(schema_text) if schema_text else []
        verdict, why = classify(raw, fields)
        verdicts.append((call.get("attempt"), verdict, why, fields))
        ph, sh, rh = sha256(prompt_text), sha256(schema_text), sha256(raw)
        seen_prompt.add(ph); seen_schema.add(sh); seen_resp.add(rh)
        print(f"    {call.get('attempt', '?'):>2}  {ph[:10]} {sh[:10]} "
              f"{rh[:10]}  {verdict:<24} {call.get('elapsed_s', '?'):>8}")
    print()
    for attempt, verdict, why, fields in verdicts:
        print(f"    attempt {attempt}: {verdict}")
        print(f"      {why}")
        print(f"      required fields derived from the schema SENT: "
              f"{fields or '(none found — schema not located in this call)'}")
    print()

    # ── Q6 ────────────────────────────────────────────────────────────
    print("  Q6. REPRODUCIBILITY — measured by hash, not by eye")
    print(f"      distinct prompts   : {len(seen_prompt)} across {len(calls)} call(s)")
    print(f"      distinct schemas   : {len(seen_schema)}")
    print(f"      distinct responses : {len(seen_resp)}")
    kinds = sorted({v for _, v, _, _ in verdicts})
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

    if any(v == SCHEMA_ECHO for _, v, _, _ in verdicts):
        # A finding, not an instrument failure — exit non-zero so it is
        # visible, and say which it is.
        print()
        print("  MEASURED: at least one attempt returned the SCHEMA where an")
        print("  INSTANCE was required. This is a measurement result, not an")
        print("  instrument failure.")
        return 1
    if all(v == VALID_INSTANCE for _, v, _, _ in verdicts):
        print()
        print("  MEASURED: every attempt returned a valid instance. The")
        print("  mismatch is NOT at the model's response kind.")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
