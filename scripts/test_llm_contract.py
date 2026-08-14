#!/usr/bin/env python3
"""Calibration for the KAI-GATE-048 C response classifier.

The operator's fitness condition, verbatim:

> If the instrument cannot distinguish schema-definition from
> schema-instance, it is not fit for this question.

So that pair is asserted first and hardest. The four kinds must stay four
verdicts: a valid instance, the schema itself, another malformed object,
and nothing at all have four different owners, and collapsing any two
would hide which.
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
EXPECTED_SCENARIOS = 10
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
# The real observed response, from run 12's cognee log.
SCHEMA_ECHO_RAW = json.dumps(SCHEMA)
INSTANCE_RAW = json.dumps({"summary": "Ada Lovelace wrote the first algorithm."})
FIELDS = ["summary"]


def test_schema_and_instance_are_never_confused() -> None:
    """THE fitness condition. Both are valid JSON objects; only the KIND
    differs, and the whole question turns on that difference."""
    scenario("schema vs instance")
    inst, why_i = c.classify(INSTANCE_RAW, SCHEMA)
    echo, why_e = c.classify(SCHEMA_ECHO_RAW, SCHEMA)
    check("an instance is VALID INSTANCE", inst == c.VALID_INSTANCE, why_i)
    check("the schema is SCHEMA ECHO", echo == c.SCHEMA_ECHO, why_e)
    check("they are different verdicts", inst != echo, f"{inst} / {echo}")
    check("and the schema verdict explains the difference",
          "not an instance" in why_e, why_e)
    # both parse as JSON objects, so the distinction is not accidental
    check("both are JSON objects — the split is on kind, not validity",
          isinstance(json.loads(INSTANCE_RAW), dict)
          and isinstance(json.loads(SCHEMA_ECHO_RAW), dict), "")


def test_all_four_kinds_stay_four_verdicts() -> None:
    scenario("four kinds distinct")
    got = {
        "instance": c.classify(INSTANCE_RAW, SCHEMA)[0],
        "schema": c.classify(SCHEMA_ECHO_RAW, SCHEMA)[0],
        "other": c.classify(json.dumps({"nonsense": 1}), SCHEMA)[0],
        "none": c.classify("", SCHEMA)[0],
    }
    check("four inputs give four distinct verdicts",
          len(set(got.values())) == 4, str(got))
    check("other malformed object is OTHER INVALID STRUCTURE",
          got["other"] == c.OTHER_INVALID, got["other"])
    check("empty is NO RESPONSE", got["none"] == c.NO_RESPONSE, got["none"])
    check("None is NO RESPONSE too",
          c.classify(None, SCHEMA)[0] == c.NO_RESPONSE, "")


def test_validator_failures_are_not_collapsed() -> None:
    """The operator's rule: do not collapse every validator failure into
    one 422. A schema echo and a random object are both 'invalid' to
    pydantic and must NOT be the same verdict here."""
    scenario("failures not collapsed")
    echo = c.classify(SCHEMA_ECHO_RAW, SCHEMA)[0]
    junk = c.classify(json.dumps({"summary_text": "wrong key"}), SCHEMA)[0]
    notjson = c.classify("Here is your summary: Ada wrote...", SCHEMA)[0]
    check("schema echo is not the same as a wrong-key object",
          echo != junk, f"{echo} / {junk}")
    check("non-JSON prose is OTHER INVALID STRUCTURE",
          notjson == c.OTHER_INVALID, notjson)
    check("and says it is not JSON", "not JSON" in
          c.classify("Here is your summary", SCHEMA)[1], "")
    check("a JSON array is not an object",
          c.classify("[1,2,3]", SCHEMA)[0] == c.OTHER_INVALID, "")


def test_required_fields_come_from_the_schema_sent() -> None:
    """R5: derived from the payload under test, never a tuple kept here.
    A schema change must not leave the classifier measuring the old one."""
    scenario("fields derived from schema")
    check("required[] is read", c.required_fields_of(SCHEMA) == ["summary"],
          str(c.required_fields_of(SCHEMA)))
    check("a JSON string schema works too",
          c.required_fields_of(json.dumps(SCHEMA)) == ["summary"], "")
    other = {"properties": {"a": {}, "b": {}}, "type": "object"}
    check("falls back to properties when required[] is absent",
          c.required_fields_of(other) == ["a", "b"],
          str(c.required_fields_of(other)))
    check("an unusable schema yields no fields",
          c.required_fields_of("not a schema") == [], "")
    # and with a DIFFERENT schema, the same echo is still an echo
    two = {"properties": {"x": {}, "y": {}}, "required": ["x", "y"],
           "type": "object", "title": "Other"}
    check("a different schema's echo is still SCHEMA ECHO",
          c.classify(json.dumps(two), two)[0] == c.SCHEMA_ECHO, "")


def test_hashes_measure_identity() -> None:
    """Q6 must be measured, not eyeballed."""
    scenario("hashes measure identity")
    check("identical text hashes identically",
          c.sha256(SCHEMA_ECHO_RAW) == c.sha256(SCHEMA_ECHO_RAW), "")
    check("a one-character difference does not",
          c.sha256(SCHEMA_ECHO_RAW) != c.sha256(SCHEMA_ECHO_RAW + " "), "")
    check("None hashes to empty", c.sha256(None) == "", "")


def test_the_summariser_reports_and_refuses() -> None:
    """End to end, on the NEW record shape: every row declares its layer
    and carries the structured response_model."""
    scenario("summariser end to end")

    def row(n, layer, raw, elapsed):
        return {"event": "llm-call", "attempt": n, "layer": layer,
                "elapsed_s": elapsed, "response_model": SCHEMA,
                "messages": [{"role": "system",
                              "content": "…schema… " + json.dumps(SCHEMA)},
                             {"role": "user", "content": "Ada Lovelace..."}],
                "raw_response": raw}

    cfg = {"event": "resolved-config", "config_llm_instructor_mode": "''",
           "adapter_instructor_mode": "'json_mode'",
           "instructor_client_mode": "<Mode.JSON: 'json_mode'>",
           "adapter_class": "OllamaAPIAdapter",
           "adapter_default_mode": "'json_mode'"}

    def run(rows):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            path = fh.name
        return subprocess.run(
            [sys.executable, "scripts/security/summarise_llm_contract.py",
             "--capture", path], cwd=REPO, capture_output=True, text=True,
            timeout=60)

    # (a) raw-layer schema echoes — the finding, reportable
    proc = run([cfg, row(1, c.RAW_MODEL_RESPONSE, SCHEMA_ECHO_RAW, 240.1),
                row(2, c.RAW_MODEL_RESPONSE, SCHEMA_ECHO_RAW, 64.0)])
    out = proc.stdout
    check("reports the denominator", "inspected: 2 model call(s)" in out, out[:300])
    check("reports the RESOLVED mode, not the config default",
          "json_mode" in out and "instructor client .mode" in out, out[:900])
    check("says an empty config field is not proof",
          "NOT proof of the effective mode" in out, out[:900])
    check("classifies both attempts as schema echo",
          out.count("SCHEMA ECHO") >= 2, out[:2000])
    check("measures byte-identity across attempts",
          "BYTE-IDENTICAL" in out, out[-1600:])
    check("labels prompt/schema hashes as canonical, response as byte",
          "CANONICAL identity" in out and "BYTE identity" in out, out[:1200])
    check("names the schema conveyance", "json_mode" in out, out[:2000])
    check("refuses to assign ownership", "NOT CONCLUDED HERE" in out, out[-1400:])
    check("and authorises no remedy",
          "No remedy is authorised" in " ".join(out.split()), out[-1400:])
    check("exits non-zero on a measured schema echo", proc.returncode != 0,
          f"rc={proc.returncode}")

    # (b) THE D215 GATE: instructor-layer evidence may not answer Q2
    proc = run([cfg, row(1, c.INSTRUCTOR_RETURN, INSTANCE_RAW, 583.2),
                row(2, c.INSTRUCTOR_RETURN, INSTANCE_RAW, 120.7)])
    out = proc.stdout
    check("an instructor-layer row is flagged as not licensing the claim",
          "LAYER LIMIT" in out, out[:2200])
    check("Q2 is declared UNMEASURED at that layer",
          "Q2 — UNMEASURED" in out, out[-1000:])
    # Whitespace-normalised: the report wraps at ~72 columns, so a
    # contiguous match asserts the LAYOUT rather than the content.
    check("and a validated object is refused as a substitute",
          "is NOT a substitute" in " ".join(out.split()), out[-1000:])
    check("even though every row classified VALID INSTANCE",
          out.count("VALID INSTANCE") >= 2, out[:2200])
    check("it still exits non-zero", proc.returncode != 0, f"rc={proc.returncode}")

    # (c) raw-layer valid instances — the only path to exit 0
    proc = run([cfg, row(1, c.RAW_MODEL_RESPONSE, INSTANCE_RAW, 12.0)])
    check("raw-layer valid instances exit zero", proc.returncode == 0,
          proc.stdout[-800:])
    check("and the claim is explicitly scoped to RAW_MODEL_RESPONSE",
          "MEASURED at RAW_MODEL_RESPONSE" in proc.stdout, proc.stdout[-800:])

    # (d) an empty capture still fails closed
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        empty = fh.name
    proc2 = subprocess.run(
        [sys.executable, "scripts/security/summarise_llm_contract.py",
         "--capture", empty], cwd=REPO, capture_output=True, text=True,
        timeout=60)
    check("an empty capture fails closed", proc2.returncode != 0,
          f"rc={proc2.returncode}")
    check("and says UNMEASURED is not 'no mismatch'",
          "not the same as 'no mismatch'" in proc2.stdout, proc2.stdout[:400])


def test_the_collector_gates_on_the_selftest() -> None:
    """Run 13's lesson, checked without a stack.

    The expensive capture must not run unless the capture point has been
    proven traversable, and the probe must not report success having
    observed nothing. Both are read out of the shipped text, because a
    rule that is only in a commit message is not enforced."""
    scenario("selftest gates the run")
    collector = (REPO / "scripts" / "security" /
                 "capture_llm_contract.sh").read_text()
    probe = (REPO / "scripts" / "security" /
             "probe_llm_contract.py").read_text()
    # ordering: the selftest must precede the expensive drive
    i_self = collector.find("selftest")
    i_drive = collector.find("== CAPTURE — in-process")
    check("the collector runs a selftest", i_self > 0, "")
    check("and it runs BEFORE the expensive capture",
          0 < i_self < i_drive, f"{i_self} vs {i_drive}")
    check("a failed selftest aborts the run",
          "MEASUREMENT ABORTED: THE CAPTURE POINT IS NOT TRAVERSED"
          in collector, "")
    check("and says it is an instrument failure, not LLM evidence",
          "not evidence about the LLM contract" in collector, "")
    # fail-closed on zero observations
    check("the probe fails closed on zero captured calls",
          "if calls == 0:" in probe and "return 2" in probe, "")
    check("and names it an instrument failure",
          "INSTRUMENT FAILURE: 0 model call(s) captured" in probe, "")
    check("and refuses to read it as 'no mismatch'",
          "which is not 'no mismatch'" in probe, "")
    # the selftest must use the production entry point, not the wrapper
    check("the selftest drives the production entry point",
          "acreate_structured_output" in probe, "")
    check("and says why calling the wrapper directly would prove nothing",
          "which was never in doubt" in probe, "")
    # both candidate hypotheses are recorded as object facts
    check("H2 (client caching) is measured by object identity",
          "get_llm_client_stable" in probe, "")
    check("H1 (instructor binding) is measured by what it holds",
          "instructor_holds_" in probe, "")


def test_the_wrapper_preserves_the_callable_convention() -> None:
    """The invariant from run 14, plus D216's altitude and the operator's
    transparency criteria — asserted against the SHIPPED probe text,
    because a rule that lives only in a commit message is not enforced."""
    scenario("wrapper preserves convention")
    probe = (REPO / "scripts" / "security" /
             "probe_llm_contract.py").read_text()
    # convention
    check("no async wrapper remains", "async def capturing" not in probe, "")
    check("the wrapper is sync and descriptor-correct",
          "def capturing_create(self, *args, **kwargs):" in probe, "")
    check("it forwards self unchanged, synchronously",
          "result = original(self, *args, **kwargs)" in probe
          and "await original(" not in probe, "")
    check("exception behaviour is preserved by re-raising",
          "raise          # exception type AND value propagate unchanged"
          in probe, "")
    # altitude (D216)
    check("the hook is Completions.create at class level",
          "Completions.create = capturing_create" in probe, "")
    check("and it is installed BEFORE adapter construction",
          probe.index("Completions.create = capturing_create")
          < probe.index("client = get_llm_client()"), "")
    check("the retry-loop call site is cited as the reason",
          "retry.py:193-198" in probe, "")
    check("and the @wraps name trap is recorded",
          "@wraps copied" in probe or "@wraps(func)" in probe, "")
    # layer tagging (D215)
    check("raw rows are tagged RAW_MODEL_REQUEST",
          '"layer": "RAW_MODEL_REQUEST"' in probe, "")
    check("and promoted to RAW_MODEL_RESPONSE once the reply exists",
          'request["layer"] = "RAW_MODEL_RESPONSE"' in probe, "")
    # transparency criteria present as runtime checks
    for crit in ("traversed", "original_executed",
                 "exactly_once_per_wrapper_call", "convention_matches",
                 "exception_type_preserved", "exception_value_preserved",
                 "exception_not_swallowed", "rows_tagged_raw_layer"):
        check(f"selftest asserts {crit}", f'checks["{crit}"]' in probe, "")
    check("the selftest refuses the run when not transparent",
          "CAPTURE POINT NOT TRANSPARENT" in probe, "")
    check("and the selftest's own rows are excluded from Q6",
          "selftest_rows_excluded_from_q6" in probe
          and '"phase"' in probe, "")
    check("the exception control needs no model run",
          "No model needed" in probe or "needs no model run" in probe, "")


def test_an_unestablished_schema_is_never_a_pass() -> None:
    """D216'S DEFECT, pinned.

    Run 15 classified every response VALID INSTANCE because the required
    field list arrived EMPTY — the schema had been found inside message
    prose, json.loads failed, and 'carries every required field' became
    trivially true. A check that cannot fail is not a check.

    The contract question must be asked BEFORE any instance test, and an
    unestablished contract must produce an explicit not-measured."""
    scenario("unestablished schema is not a pass")
    for bad in (None, "", "not json at all", "[1,2,3]", 42,
                "Parse the content and return a JSON object matching this "
                "schema: {…prose…}"):
        v, why = c.classify(INSTANCE_RAW, bad)
        check(f"schema={bad!r:.34} -> CLASSIFIER_UNMEASURED",
              v == c.CLASSIFIER_UNMEASURED, f"{v}: {why}")
    # a schema with no requirements at all is equally unmeasurable
    v, why = c.classify(INSTANCE_RAW, {"type": "object"})
    check("a schema naming nothing required is not a pass",
          v == c.CLASSIFIER_UNMEASURED, f"{v}: {why}")
    check("and it says requirements were not established",
          "SCHEMA REQUIREMENTS NOT ESTABLISHED" in why, why)
    check("and explicitly that this is not a pass", "not a pass" in why, why)
    # THE REGRESSION: a schema echo must not sail through an empty contract
    v, _ = c.classify(SCHEMA_ECHO_RAW, None)
    check("a schema echo with an unreadable schema is NOT valid instance",
          v != c.VALID_INSTANCE, v)


def test_layers_gate_what_may_be_claimed() -> None:
    """D215. INSTRUCTOR_RETURN = VALID INSTANCE must never answer a
    question about RAW_MODEL_RESPONSE — instructor retried, parsed and
    validated in between."""
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


def run_all() -> None:
    test_schema_and_instance_are_never_confused()
    test_all_four_kinds_stay_four_verdicts()
    test_validator_failures_are_not_collapsed()
    test_required_fields_come_from_the_schema_sent()
    test_hashes_measure_identity()
    test_the_summariser_reports_and_refuses()
    test_the_collector_gates_on_the_selftest()
    test_the_wrapper_preserves_the_callable_convention()
    test_an_unestablished_schema_is_never_a_pass()
    test_layers_gate_what_may_be_claimed()

    kinds = [c.VALID_INSTANCE, c.SCHEMA_ECHO, c.OTHER_INVALID, c.NO_RESPONSE]
    print(f"  inspected: {len(kinds)} response kind(s) discriminated: {kinds}")
    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS, f"{len(executed)}: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"LLM Response Contract Calibration: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
