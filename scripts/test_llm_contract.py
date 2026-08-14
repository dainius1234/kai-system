#!/usr/bin/env python3
"""Calibration for the KAI-GATE-048 C contract recovery and classifier.

The operator's fitness condition, verbatim:

> If the instrument cannot distinguish schema-definition from
> schema-instance, it is not fit for this question.

and, after run 17:

> If the analyser says VALID_INSTANCE, that must mean the raw response
> satisfies the actual captured JSON Schema, not merely that it contains
> the required top-level keys.

So both pairs are asserted first and hardest: schema vs instance, and
required-fields-present vs schema-validated. Collapsing either hides which
of four different owners has the defect.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import classify_llm_response as c  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 17
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


# The real schema, from cognee/shared/data_models.py:270.
SCHEMA = {
    "description": "Bulleted memory record produced by chunk summarization.",
    "properties": {"summary": {"title": "Summary", "type": "string"}},
    "required": ["summary"],
    "title": "SummarizedContent",
    "type": "object",
}
SCHEMA_ECHO_RAW = json.dumps(SCHEMA)
INSTANCE_RAW = json.dumps({"summary": "Ada Lovelace wrote the first algorithm."})

# A real instructor-formatted system message, built the way
# providers/openai/utils.py:491 handle_json_modes builds it: the caller's
# system prompt, then the marker, the indent=2 schema, then the trailer.
INSTRUCTOR_SYSTEM = (
    "You are a summarizer.\n\n"
    "\n        Parse the content and return a JSON object matching this schema:\n\n"
    "        " + json.dumps(SCHEMA, indent=2) + "\n\n"
    "        Return a valid JSON instance, not the schema definition."
)


def messages(system_text: str, user: str = "Ada Lovelace...") -> list:
    return [{"role": "system", "content": system_text},
            {"role": "user", "content": user}]


# Forced-unavailable validator: safe to fabricate, because it only ever
# WEAKENS a claim. The reverse (fabricating availability) is refused by
# _validate returning None.
NO_VALIDATOR = {"available": False, "library": None, "version": None,
                "consequence": "narrower label only"}
REAL_VALIDATOR = c.validator_status()


def test_contract_is_recovered_from_the_attempt() -> None:
    """D222: the schema lives in THIS attempt's system message."""
    scenario("contract recovered from the row")
    schema, why, detail = c.recover_contract(messages(INSTRUCTOR_SYSTEM))
    check("a real instructor system message yields the schema",
          schema == SCHEMA, f"{why} / {schema}")
    check("and says where it came from", "system message" in why, why)
    check("exactly one candidate region was found",
          detail["candidates"] == 1, str(detail))
    check("the recovered contract names the right required field",
          c.required_fields_of(schema) == ["summary"], "")


def test_ambiguous_or_absent_contract_is_unmeasured() -> None:
    """Zero regions and two regions are BOTH refusals. Picking would guess."""
    scenario("ambiguous contract refused")
    none_schema, why_none, d0 = c.recover_contract(
        messages("You are a summarizer. No schema here."))
    check("no schema-shaped region -> no contract", none_schema is None, why_none)
    check("and says so", "no schema-shaped" in why_none, why_none)
    check("zero candidates counted", d0["candidates"] == 0, str(d0))

    two = INSTRUCTOR_SYSTEM + "\n\nAlso consider: " + json.dumps(
        {"properties": {"other": {}}, "required": ["other"], "type": "object"})
    amb, why_amb, d2 = c.recover_contract(messages(two))
    check("two schema-shaped regions -> no contract", amb is None, why_amb)
    check("and names the ambiguity", "ambiguous" in why_amb, why_amb)
    check("both candidates counted", d2["candidates"] == 2, str(d2))

    check("no system message at all -> no contract",
          c.recover_contract([{"role": "user", "content": "hi"}])[0] is None, "")
    check("messages missing entirely -> no contract",
          c.recover_contract(None)[0] is None, "")


def test_the_mode_directive_is_not_a_contract() -> None:
    """response_format={"type":"json_object"} carries no schema (D222 §2)."""
    scenario("mode directive is not a contract")
    directive = {"type": "json_object"}
    check("the directive is not schema-shaped",
          not c.is_schema_shaped(directive), "")
    verdict, why = c.classify(INSTANCE_RAW, directive, NO_VALIDATOR)
    check("classifying against it yields CONTRACT_UNMEASURED",
          verdict == c.CONTRACT_UNMEASURED, verdict)
    check("and explicitly not a pass", "not a pass" in why, why)
    # a directive sitting in a system message must not be mistaken for one
    check("a directive in the system text is not recovered as a contract",
          c.recover_contract(messages(
              'Reply with {"type": "json_object"} only.'))[0] is None, "")

    # ISOLATES the schema-shaped guard. A mutation test showed the empty-
    # required guard was MASKING it: deleting `is_schema_shaped` from step 1
    # changed no verdict, so nothing was testing it. This dict names a
    # required field, so the empty-required guard cannot fire, and only the
    # shape guard stands between it and a verdict.
    fragment = {"required": ["summary"]}
    check("a non-schema-shaped dict is not a contract even with required[]",
          not c.is_schema_shaped(fragment), "")
    check("and classifying against it is CONTRACT_UNMEASURED",
          c.classify(INSTANCE_RAW, fragment, NO_VALIDATOR)[0]
          == c.CONTRACT_UNMEASURED, "")
    check("the same holds WITH a validator available",
          c.classify(INSTANCE_RAW, fragment, REAL_VALIDATOR)[0]
          == c.CONTRACT_UNMEASURED, "")


def test_an_unusable_contract_is_not_an_invalid_response() -> None:
    """Different owners: a malformed REQUEST is not a bad REPLY.

    `check_schema` separates them. Reporting an unusable contract as
    INSTANCE_INVALID would hand the finding to the model when the defect is
    in what was sent."""
    scenario("unusable contract is not an invalid response")
    check("a JSON Schema validator is available to calibrate this",
          REAL_VALIDATOR["available"], str(REAL_VALIDATOR))
    if not REAL_VALIDATOR["available"]:
        return
    # schema-shaped (has properties) but not a valid JSON Schema
    broken = {"type": "not-a-type", "properties": {"summary": {}},
              "required": ["summary"]}
    check("the broken contract IS schema-shaped, so it reaches validation",
          c.is_schema_shaped(broken), "")
    verdict, why = c.classify(INSTANCE_RAW, broken, REAL_VALIDATOR)
    check("an unusable contract is CONTRACT_UNMEASURED",
          verdict == c.CONTRACT_UNMEASURED, f"{verdict}: {why}")
    check("and NOT INSTANCE_INVALID", verdict != c.INSTANCE_INVALID, verdict)
    check("the reason blames the contract, not the reply",
          "not in what came back" in why, why)
    check("and it is not a pass", verdict not in c.PASSING_VERDICTS, "")


def test_dicts_serialise_as_valid_json() -> None:
    """Run 17 recorded str(dict) — Python repr, unparseable (D222 §5.3)."""
    scenario("dicts serialise as JSON")
    text = c.canonical({"type": "json_object"})
    check("canonical output parses as JSON", json.loads(text) ==
          {"type": "json_object"}, text)
    check("and uses double quotes, not Python repr", "'" not in text, text)
    check("key order does not change canonical identity",
          c.sha256_canonical({"a": 1, "b": 2}) == c.sha256_canonical({"b": 2, "a": 1}), "")
    check("byte identity is a DIFFERENT question from canonical identity",
          c.sha256_bytes('{"b":2,"a":1}') != c.sha256_bytes('{"a":1,"b":2}'), "")
    check("byte identity of identical text matches",
          c.sha256_bytes(SCHEMA_ECHO_RAW) == c.sha256_bytes(SCHEMA_ECHO_RAW), "")
    check("None hashes to empty", c.sha256_bytes(None) == "", "")


def test_schema_echo_is_never_an_instance() -> None:
    """THE fitness condition. Both are JSON objects; only the KIND differs."""
    scenario("schema vs instance")
    echo, why_e = c.classify(SCHEMA_ECHO_RAW, SCHEMA, NO_VALIDATOR)
    inst, why_i = c.classify(INSTANCE_RAW, SCHEMA, NO_VALIDATOR)
    check("the schema is SCHEMA ECHO", echo == c.SCHEMA_ECHO, why_e)
    check("an instance is not", inst != c.SCHEMA_ECHO, why_i)
    check("and the echo verdict explains the difference",
          "not an instance" in why_e, why_e)
    check("it notes the echo IS the captured contract",
          "canonically IDENTICAL" in why_e, why_e)
    check("both are JSON objects — the split is on kind, not validity",
          isinstance(json.loads(INSTANCE_RAW), dict)
          and isinstance(json.loads(SCHEMA_ECHO_RAW), dict), "")


def test_required_fields_are_never_promoted() -> None:
    """THE operator's rule after run 17.

    Without a validator the analyser may say only what it tested. A
    top-level key check is not JSON Schema validation."""
    scenario("required fields never promoted")
    verdict, why = c.classify(INSTANCE_RAW, SCHEMA, NO_VALIDATOR)
    check("no validator -> REQUIRED_FIELDS_PRESENT",
          verdict == c.REQUIRED_FIELDS_PRESENT, verdict)
    check("and NOT VALID_INSTANCE", verdict != c.VALID_INSTANCE, verdict)
    check("the reason says it is not schema validation",
          "NOT SCHEMA VALIDATION" in why, why)
    check("and names what went untested", "types" in why, why)
    check("REQUIRED_FIELDS_PRESENT is not in the passing set",
          c.REQUIRED_FIELDS_PRESENT not in c.PASSING_VERDICTS, "")
    check("only VALID_INSTANCE passes",
          c.PASSING_VERDICTS == {c.VALID_INSTANCE}, str(c.PASSING_VERDICTS))
    missing, why_m = c.classify(json.dumps({"other": 1}), SCHEMA, NO_VALIDATOR)
    check("a missing required key is REQUIRED_FIELDS_MISSING",
          missing == c.REQUIRED_FIELDS_MISSING, missing)
    check("and says only presence was tested", "no JSON Schema validator" in why_m,
          why_m)


def test_real_validation_catches_what_key_presence_cannot() -> None:
    """The case that separates the two labels: required key present, but
    the schema violated in another way (wrong type)."""
    scenario("validator catches constraint violations")
    wrong_type = json.dumps({"summary": 12345})     # required key, wrong type
    weak, _ = c.classify(wrong_type, SCHEMA, NO_VALIDATOR)
    check("without a validator this looks like REQUIRED_FIELDS_PRESENT",
          weak == c.REQUIRED_FIELDS_PRESENT, weak)
    check("but it is never VALID_INSTANCE", weak != c.VALID_INSTANCE, weak)

    # I-1: absence of the validator is not the benign case. If it is not
    # installed, this scenario is NOT calibrated and must say so loudly.
    check("a JSON Schema validator is available to calibrate the strong path",
          REAL_VALIDATOR["available"],
          f"jsonschema missing: {REAL_VALIDATOR.get('why')}")
    if not REAL_VALIDATOR["available"]:
        return
    strong, why_s = c.classify(wrong_type, SCHEMA, REAL_VALIDATOR)
    check("WITH a validator the same object is INSTANCE_INVALID",
          strong == c.INSTANCE_INVALID, f"{strong}: {why_s}")
    check("and the violation is named", "violation" in why_s, why_s)
    good, why_g = c.classify(INSTANCE_RAW, SCHEMA, REAL_VALIDATOR)
    check("a genuinely valid instance IS VALID_INSTANCE",
          good == c.VALID_INSTANCE, f"{good}: {why_g}")
    check("and cites the validator that ran", "jsonschema" in why_g, why_g)
    echo, _ = c.classify(SCHEMA_ECHO_RAW, SCHEMA, REAL_VALIDATOR)
    check("the schema echo stays SCHEMA ECHO even with a validator",
          echo == c.SCHEMA_ECHO, echo)


def test_malformed_responses_stay_distinct() -> None:
    """Four failure kinds, four owners. Collapsing hides which."""
    scenario("failure kinds stay distinct")
    got = {
        "none": c.classify("", SCHEMA, NO_VALIDATOR)[0],
        "prose": c.classify("Here is your summary: Ada wrote...", SCHEMA,
                            NO_VALIDATOR)[0],
        "array": c.classify("[1,2,3]", SCHEMA, NO_VALIDATOR)[0],
        "wrongkey": c.classify(json.dumps({"summary_text": "x"}), SCHEMA,
                               NO_VALIDATOR)[0],
    }
    check("four inputs give four distinct verdicts",
          len(set(got.values())) == 4, str(got))
    check("empty is NO RESPONSE", got["none"] == c.NO_RESPONSE, got["none"])
    check("None is NO RESPONSE too",
          c.classify(None, SCHEMA, NO_VALIDATOR)[0] == c.NO_RESPONSE, "")
    check("prose is NOT JSON", got["prose"] == c.NOT_JSON, got["prose"])
    check("an array is NOT A JSON OBJECT",
          got["array"] == c.NOT_JSON_OBJECT, got["array"])
    check("a wrong-key object is REQUIRED_FIELDS_MISSING",
          got["wrongkey"] == c.REQUIRED_FIELDS_MISSING, got["wrongkey"])


def test_an_unusable_contract_is_never_a_pass() -> None:
    """D216's defect, re-pinned at the new boundary."""
    scenario("unusable contract is never a pass")
    for bad in (None, "", "not a schema", [], 0, {}, {"type": "json_object"}):
        v, why = c.classify(INSTANCE_RAW, bad, NO_VALIDATOR)
        check(f"{bad!r} yields CONTRACT_UNMEASURED",
              v == c.CONTRACT_UNMEASURED, f"{bad!r} -> {v}")
        check(f"{bad!r} is not a pass", v not in c.PASSING_VERDICTS, v)
    v, _ = c.classify(SCHEMA_ECHO_RAW, None, NO_VALIDATOR)
    check("a schema echo with no contract is NOT valid instance",
          v != c.VALID_INSTANCE, v)


def test_layers_gate_what_may_be_claimed() -> None:
    """D215. INSTRUCTOR_RETURN must never answer a RAW_MODEL_RESPONSE question."""
    scenario("layers gate claims")
    check("RAW_MODEL_RESPONSE licenses a raw-response claim",
          c.licenses_raw_response_claim(c.RAW_MODEL_RESPONSE), "")
    for layer in (c.INSTRUCTOR_RETURN, c.ADAPTER_INPUT, c.RAW_MODEL_REQUEST,
                  c.VALIDATION_RESULT, None, "MADE_UP"):
        check(f"{layer} does NOT license a raw-response claim",
              not c.licenses_raw_response_claim(layer), str(layer))
    check("the vocabulary is exactly the five agreed layers",
          c.LAYERS == {"ADAPTER_INPUT", "INSTRUCTOR_RETURN",
                       "RAW_MODEL_REQUEST", "RAW_MODEL_RESPONSE",
                       "VALIDATION_RESULT"}, str(sorted(c.LAYERS)))


def test_extraction_rule_records_its_provenance() -> None:
    """I-8: where the rule came from is recorded, never assumed."""
    scenario("extraction rule provenance")
    prov = c.extraction_rule_provenance()
    check("a primary rule is stated", bool(prov["primary_rule"]), "")
    check("the rule is structural, not a copied English string",
          "structural" in prov["primary_rule"], prov["primary_rule"])
    check("instructor availability is recorded either way",
          "instructor_available" in prov, "")
    check("corroboration status is always stated",
          bool(prov.get("corroboration")), str(prov))
    if prov["instructor_available"]:
        check("the source is named", prov["source"] is not None, "")
        check("and digested", bool(prov["source_sha256"]), "")
        check("markers are derived from that source", bool(prov["markers"]), "")
    else:
        check("absence of instructor is stated, not silently ignored",
              "instructor_import_error" in prov, str(prov))
        check("and the structural rule is declared to stand alone",
              "NONE" in prov["corroboration"], prov["corroboration"])
    val = c.validator_status()
    check("validator availability is recorded", "available" in val, "")
    if not val["available"]:
        check("and its consequence is spelled out", "consequence" in val, "")


def test_q6_grouping_is_refused_not_invented() -> None:
    """max_retries=2 means five raw rows span several logical calls."""
    scenario("Q6 grouping refused")
    rows = [{"attempt": n, "elapsed_s": 10.0 * n} for n in range(1, 6)]
    g = c.logical_call_grouping(rows)
    check("without a correlation id, grouping is UNAVAILABLE",
          g["available"] is False, str(g))
    check("it says why", "no attempt carries" in g["why"], g["why"])
    check("it names what it looked for", "logical_call_id" in g["looked_for"], "")
    check("it names the next measurement requirement",
          "correlation id" in g["next_measurement_requirement"], "")
    for signal in ("adjacency", "elapsed time", "prompt hash", "schema hash"):
        check(f"{signal} is explicitly refused as a grouping signal",
              signal in g["refused_signals"], str(g["refused_signals"]))
    tagged = [{"attempt": 1, "logical_call_id": "A"},
              {"attempt": 2, "logical_call_id": "A"},
              {"attempt": 3, "logical_call_id": "B"}]
    g2 = c.logical_call_grouping(tagged)
    check("WITH a correlation id, grouping becomes available",
          g2["available"] is True, str(g2))
    check("and the groups are counted", g2["groups"] == {"A": 2, "B": 1},
          str(g2.get("groups")))
    partial = [{"attempt": 1, "logical_call_id": "A"}, {"attempt": 2}]
    check("a partially tagged set is NOT grouped",
          c.logical_call_grouping(partial)["available"] is False, "")


def _run_summariser(rows):
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        path = fh.name
    return subprocess.run(
        [sys.executable, "scripts/security/summarise_llm_contract.py",
         "--capture", path], cwd=REPO, capture_output=True, text=True,
        timeout=60)


def test_the_summariser_reports_and_refuses() -> None:
    """End to end on the run-17 record shape."""
    scenario("summariser end to end")

    def row(n, layer, raw, system=INSTRUCTOR_SYSTEM, phase="capture"):
        return {"event": "llm-call", "attempt": n, "layer": layer,
                "phase": phase, "elapsed_s": 12.0 * n,
                "response_format": {"type": "json_object"},
                "messages": messages(system), "raw_response": raw}

    cfg = {"event": "resolved-config", "config_llm_instructor_mode": "''",
           "adapter_instructor_mode": "'json_mode'",
           "instructor_client_mode": "<Mode.JSON: 'json_mode'>",
           "adapter_class": "OllamaAPIAdapter",
           "adapter_default_mode": "'json_mode'"}

    # (a) raw-layer schema echoes — the finding, reportable
    proc = _run_summariser([cfg,
                            row(1, c.RAW_MODEL_RESPONSE, SCHEMA_ECHO_RAW),
                            row(2, c.RAW_MODEL_RESPONSE, SCHEMA_ECHO_RAW)])
    out = proc.stdout
    check("reports the production denominator",
          "inspected: 2 production model call(s)" in out, out[:400])
    check("recovers the contract per attempt", "RECOVERED" in out, out[:400])
    check("reports the RESOLVED mode", "Mode.JSON" in out, "")
    check("names response_format as a MODE DIRECTIVE",
          "MODE DIRECTIVE, not the contract" in out, "")
    check("reports schema echo at the raw layer", c.SCHEMA_ECHO in out, "")
    check("and refuses", proc.returncode == 1, str(proc.returncode))

    # (b) selftest rows are excluded from the production denominator
    proc = _run_summariser([cfg,
                            row(1, c.RAW_MODEL_RESPONSE, INSTANCE_RAW),
                            row(2, c.RAW_MODEL_RESPONSE, SCHEMA_ECHO_RAW,
                                phase="selftest")])
    check("selftest rows are excluded",
          "inspected: 1 production model call(s)" in proc.stdout, "")
    check("and counted separately",
          "excluded : 1 selftest row(s)" in proc.stdout, proc.stdout[:400])

    # (c) run 17's actual shape: no contract in the system message
    proc = _run_summariser([cfg,
                            row(1, c.RAW_MODEL_RESPONSE, INSTANCE_RAW,
                                system="You are a summarizer.")])
    check("no recoverable contract -> CONTRACT_UNMEASURED",
          c.CONTRACT_UNMEASURED in proc.stdout, proc.stdout[:400])
    check("and it is not a pass", proc.returncode == 1, "")
    check("and the row shows UNMEASURED for the contract column",
          "UNMEASURED" in proc.stdout, "")

    # (d) all valid at the WRONG layer must still refuse
    proc = _run_summariser([cfg,
                            row(1, c.INSTRUCTOR_RETURN, INSTANCE_RAW),
                            row(2, c.INSTRUCTOR_RETURN, INSTANCE_RAW)])
    check("no raw-layer row -> Q2 UNMEASURED",
          "Q2 — UNMEASURED" in proc.stdout, proc.stdout[-600:])
    check("and it refuses", proc.returncode == 1, "")

    # (e) Q6 grouping refused in the report itself
    proc = _run_summariser([cfg,
                            row(1, c.RAW_MODEL_RESPONSE, INSTANCE_RAW),
                            row(2, c.RAW_MODEL_RESPONSE, INSTANCE_RAW)])
    check("Q6 reports logical calls UNAVAILABLE",
          "logical calls      : UNAVAILABLE" in proc.stdout, proc.stdout[-900:])
    check("and names the next measurement requirement",
          "NEXT MEASUREMENT REQUIREMENT" in proc.stdout, "")
    check("and refuses adjacency as a grouping signal",
          "REFUSED as grouping signals" in proc.stdout, "")
    check("and explains the max_retries=2 arithmetic",
          "max_retries=2" in proc.stdout, "")

    # (f) the only exit-0 path: real validation, at the raw layer
    if REAL_VALIDATOR["available"]:
        check("valid instances at the raw layer with a validator pass",
              proc.returncode == 0, proc.stdout[-700:])
        check("and it says validation actually happened",
              "validated against the contract" in proc.stdout, "")


def test_the_collector_gates_on_the_selftest() -> None:
    """Run 13/16's lessons, read out of the shipped text."""
    scenario("selftest gates the run")
    collector = (REPO / "scripts" / "security" /
                 "capture_llm_contract.sh").read_text()
    probe = (REPO / "scripts" / "security" /
             "probe_llm_contract.py").read_text()
    i_self = collector.find("selftest")
    i_drive = collector.find("== CAPTURE — in-process")
    check("the collector runs a selftest", i_self > 0, "")
    check("and it runs BEFORE the expensive capture",
          0 < i_self < i_drive, f"{i_self} vs {i_drive}")
    check("a failed selftest aborts the run",
          "MEASUREMENT ABORTED: CAPTURE TRANSPARENCY NOT PROVEN" in collector, "")
    check("and says it is an instrument failure, not LLM evidence",
          "not evidence about the LLM contract" in collector, "")
    check("the abort no longer claims non-traversal generically",
          "MEASUREMENT ABORTED: THE CAPTURE POINT IS NOT TRAVERSED"
          not in collector, "")
    for state in ("NOT INSTALLED", "NOT TRAVERSED", "TRANSPARENCY NOT PROVEN"):
        check(f"the abort distinguishes {state}", state in collector, "")
        check(f"and the probe can emit {state}",
              f"SELFTEST-CLASS: {state}" in probe, "")
    check("the collector reads the class from the probe's own line",
          "grep -m1 '^SELFTEST-CLASS: '" in collector, "")
    check("and fails closed when no class was printed",
          "UNREPORTED" in collector, "")
    check("the three classes have distinct exit codes",
          "return 3" in probe and "return 4" in probe, "")
    check("the probe fails closed on zero captured calls",
          "if calls == 0:" in probe and "return 2" in probe, "")
    check("and refuses to read it as 'no mismatch'",
          "which is not 'no mismatch'" in probe, "")
    check("the selftest drives the production entry point",
          "acreate_structured_output" in probe, "")


def test_the_wrapper_preserves_the_callable_convention() -> None:
    """Run 14/16's invariants, asserted against the SHIPPED probe text."""
    scenario("wrapper preserves convention")
    probe = (REPO / "scripts" / "security" /
             "probe_llm_contract.py").read_text()
    check("no async wrapper remains", "async def capturing" not in probe, "")
    check("the wrapper is sync and descriptor-correct",
          "def capturing_create(self, *args, **kwargs):" in probe, "")
    check("it forwards self unchanged, synchronously",
          "result = forward(self, *args, **kwargs)" in probe
          and "await forward(" not in probe, "")
    check("the forward target is read from STATE at call time",
          'forward = STATE.get("original")' in probe, "")
    check("and is NOT a closure local captured at install",
          "result = original(self" not in probe, "")
    check("a missing forward target fails closed",
          "capture wrapper has no original to forward to" in probe, "")
    check("exception behaviour is preserved by re-raising",
          "raise          # exception type AND value propagate unchanged"
          in probe, "")
    check("the hook is Completions.create at class level",
          "Completions.create = capturing_create" in probe, "")
    check("and it is installed BEFORE adapter construction",
          probe.index("Completions.create = capturing_create")
          < probe.index("client = get_llm_client()"), "")
    check("the retry-loop call site is cited as the reason",
          "retry.py:193-198" in probe, "")
    check("raw rows are tagged RAW_MODEL_REQUEST",
          '"layer": "RAW_MODEL_REQUEST"' in probe, "")
    check("and promoted to RAW_MODEL_RESPONSE once the reply exists",
          'request["layer"] = "RAW_MODEL_RESPONSE"' in probe, "")
    for crit in ("traversed", "original_executed",
                 "exactly_once_per_wrapper_call", "convention_matches",
                 "exception_type_preserved", "exception_value_preserved",
                 "exception_not_swallowed", "rows_tagged_raw_layer",
                 "standin_executed_exactly_once", "exception_wrapper_traversed",
                 "exception_object_not_replaced", "original_restored_to_real",
                 "restore_check_rejects_a_standin"):
        check(f"selftest asserts {crit}", f'checks["{crit}"]' in probe, "")
    check("the stand-in counts its OWN executions",
          'standin["calls"] += 1' in probe
          and 'standin["calls"] == 1' in probe, "")
    check("the restore is in a finally, not a trailing statement",
          probe.index("finally:\n        # try/finally")
          > probe.index('STATE["original"] = raising_original'), "")
    check("the restore check has a known-negative and a known-positive",
          "not forwards_to_the_real_original()" in probe
          and 'checks["original_restored_to_real"] = '
              'forwards_to_the_real_original()' in probe, "")
    check("openai's module path is RECORDED, not used as a gate",
          "provenance_of_original" in probe
          and 'module.startswith("openai.")' not in probe, "")
    check("the selftest refuses the run when not transparent",
          "NOT PROVEN TRANSPARENT" in probe, "")
    check("and the selftest's own rows are excluded from Q6",
          "selftest_rows_excluded_from_q6" in probe, "")


SHIPPED_WRAPPER_HARNESS = r'''
import sys, types, importlib.util, json
real_calls = {"n": 0}
mod = types.ModuleType("openai.resources.chat.completions")
class Completions:
    def create(self, *a, **k):
        real_calls["n"] += 1
        return "REAL RESULT"
mod.Completions = Completions
pkg = types.ModuleType("openai"); pkg.__version__ = "stub"
for name, m in (("openai", pkg),
                ("openai.resources", types.ModuleType("openai.resources")),
                ("openai.resources.chat", types.ModuleType("openai.resources.chat")),
                ("openai.resources.chat.completions", mod)):
    sys.modules[name] = m

spec = importlib.util.spec_from_file_location("probe", sys.argv[1])
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)
probe.OUT = sys.argv[2]
open(probe.OUT, "w").close(); probe.STATE["phase"] = "harness"
try:
    probe.install_capture()
except ModuleNotFoundError:
    pass

wrapper = Completions.create
out = {"wrapper_is_shipped": wrapper is probe.STATE["wrapper"],
       "known_positive": probe.forwards_to_the_real_original()}

ns = {}
exec("def raising_original(_self,*a,**k):\n"
     "    standin['n'] += 1\n"
     "    raise sentinel\n", probe.__dict__, ns)
sentinel = RuntimeError("kai-gate-048c selftest sentinel")
standin = {"n": 0}
probe.__dict__["sentinel"] = sentinel; probe.__dict__["standin"] = standin
raising_original = ns["raising_original"]
out["standin_defined_in_probe"] = raising_original.__module__ == probe.__name__

saved = probe.STATE["original"]; caught = None
before = probe.STATE["capture"]["attempt"]
try:
    probe.STATE["original"] = raising_original
    out["known_negative"] = not probe.forwards_to_the_real_original()
    class _Fake: pass
    try:
        wrapper(_Fake(), messages=[], model="x")
    except Exception as exc:
        caught = exc
finally:
    probe.STATE["original"] = saved

out["standin_ran_once"] = standin["n"] == 1
out["real_not_called"] = real_calls["n"] == 0
out["traversed_once"] = probe.STATE["capture"]["attempt"] - before == 1
out["type_preserved"] = type(caught) is RuntimeError
out["value_preserved"] = str(caught) == str(sentinel)
out["object_not_replaced"] = caught is sentinel
out["restored"] = probe.forwards_to_the_real_original()
rows = [json.loads(l) for l in open(probe.OUT) if '"llm-call"' in l]
out["row_has_transport_error"] = bool(rows) and bool(rows[-1].get("transport_error"))
out["row_has_no_raw_response"] = bool(rows) and rows[-1].get("raw_response") is None
print(json.dumps(out))
'''


def test_the_shipped_wrapper_is_transparent_in_process() -> None:
    """The SHIPPED wrapper, executed — not its text, and not a rewrite."""
    scenario("shipped wrapper runs transparently")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(SHIPPED_WRAPPER_HARNESS)
        harness = fh.name
    capture = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False).name
    proc = subprocess.run(
        [sys.executable, harness,
         str(REPO / "scripts" / "security" / "probe_llm_contract.py"), capture],
        cwd=REPO, capture_output=True, text=True, timeout=60)
    check("the harness ran", proc.returncode == 0, proc.stderr[-400:])
    if proc.returncode != 0:
        return
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    expected = {
        "wrapper_is_shipped": "the installed callable is the shipped wrapper",
        "known_positive": "the restore check accepts the real original",
        "known_negative": "and REJECTS an injected stand-in",
        "standin_defined_in_probe": "the stand-in lives where selftest's does",
        "standin_ran_once": "the injected stand-in actually executed, once",
        "real_not_called": "and the real original was not called instead",
        "traversed_once": "the wrapper was traversed exactly once",
        "type_preserved": "the exception type survives the wrapper",
        "value_preserved": "the exception value survives the wrapper",
        "object_not_replaced": "the exception object itself is not replaced",
        "restored": "the forward target is the real original afterwards",
        "row_has_transport_error": "the row records the transport error",
        "row_has_no_raw_response": "and carries NO raw response for it",
    }
    for key, name in expected.items():
        check(f"shipped: {name}", got.get(key) is True, f"{key}={got.get(key)}")


def run_all() -> None:
    test_contract_is_recovered_from_the_attempt()
    test_ambiguous_or_absent_contract_is_unmeasured()
    test_the_mode_directive_is_not_a_contract()
    test_an_unusable_contract_is_not_an_invalid_response()
    test_dicts_serialise_as_valid_json()
    test_schema_echo_is_never_an_instance()
    test_required_fields_are_never_promoted()
    test_real_validation_catches_what_key_presence_cannot()
    test_malformed_responses_stay_distinct()
    test_an_unusable_contract_is_never_a_pass()
    test_layers_gate_what_may_be_claimed()
    test_extraction_rule_records_its_provenance()
    test_q6_grouping_is_refused_not_invented()
    test_the_summariser_reports_and_refuses()
    test_the_collector_gates_on_the_selftest()
    test_the_wrapper_preserves_the_callable_convention()
    test_the_shipped_wrapper_is_transparent_in_process()

    kinds = [c.VALID_INSTANCE, c.INSTANCE_INVALID, c.REQUIRED_FIELDS_PRESENT,
             c.REQUIRED_FIELDS_MISSING, c.SCHEMA_ECHO, c.NOT_JSON,
             c.NOT_JSON_OBJECT, c.NO_RESPONSE, c.CONTRACT_UNMEASURED]
    print(f"  inspected: {len(kinds)} response verdict(s) discriminated")
    print(f"  validator: {REAL_VALIDATOR}")
    print(f"  extraction rule: {c.extraction_rule_provenance()['primary_rule']}")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")


if __name__ == "__main__":
    print("=" * 60)
    run_all()
    print()
    print("=" * 60)
    print(f"LLM Response Contract Calibration: {passed} passed, {failed} failed")
    print(f"EXIT GATE: {'PASS' if failed == 0 else 'FAIL'}")
    sys.exit(1 if failed else 0)
