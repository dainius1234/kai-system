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
EXPECTED_SCENARIOS = 7
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
    inst, why_i = c.classify(INSTANCE_RAW, FIELDS)
    echo, why_e = c.classify(SCHEMA_ECHO_RAW, FIELDS)
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
        "instance": c.classify(INSTANCE_RAW, FIELDS)[0],
        "schema": c.classify(SCHEMA_ECHO_RAW, FIELDS)[0],
        "other": c.classify(json.dumps({"nonsense": 1}), FIELDS)[0],
        "none": c.classify("", FIELDS)[0],
    }
    check("four inputs give four distinct verdicts",
          len(set(got.values())) == 4, str(got))
    check("other malformed object is OTHER INVALID STRUCTURE",
          got["other"] == c.OTHER_INVALID, got["other"])
    check("empty is NO RESPONSE", got["none"] == c.NO_RESPONSE, got["none"])
    check("None is NO RESPONSE too",
          c.classify(None, FIELDS)[0] == c.NO_RESPONSE, "")


def test_validator_failures_are_not_collapsed() -> None:
    """The operator's rule: do not collapse every validator failure into
    one 422. A schema echo and a random object are both 'invalid' to
    pydantic and must NOT be the same verdict here."""
    scenario("failures not collapsed")
    echo = c.classify(SCHEMA_ECHO_RAW, FIELDS)[0]
    junk = c.classify(json.dumps({"summary_text": "wrong key"}), FIELDS)[0]
    notjson = c.classify("Here is your summary: Ada wrote...", FIELDS)[0]
    check("schema echo is not the same as a wrong-key object",
          echo != junk, f"{echo} / {junk}")
    check("non-JSON prose is OTHER INVALID STRUCTURE",
          notjson == c.OTHER_INVALID, notjson)
    check("and says it is not JSON", "not JSON" in
          c.classify("Here is your summary", FIELDS)[1], "")
    check("a JSON array is not an object",
          c.classify("[1,2,3]", FIELDS)[0] == c.OTHER_INVALID, "")


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
          c.classify(json.dumps(two), ["x", "y"])[0] == c.SCHEMA_ECHO, "")


def test_hashes_measure_identity() -> None:
    """Q6 must be measured, not eyeballed."""
    scenario("hashes measure identity")
    check("identical text hashes identically",
          c.sha256(SCHEMA_ECHO_RAW) == c.sha256(SCHEMA_ECHO_RAW), "")
    check("a one-character difference does not",
          c.sha256(SCHEMA_ECHO_RAW) != c.sha256(SCHEMA_ECHO_RAW + " "), "")
    check("None hashes to empty", c.sha256(None) == "", "")


def test_the_summariser_reports_and_refuses() -> None:
    scenario("summariser end to end")
    cap = [
        {"event": "resolved-config", "config_llm_instructor_mode": "''",
         "adapter_instructor_mode": "'json_mode'",
         "instructor_client_mode": "<Mode.JSON: 'json_mode'>",
         "adapter_class": "OllamaAPIAdapter",
         "adapter_default_mode": "'json_mode'"},
        {"event": "llm-call", "attempt": 1, "elapsed_s": 240.1,
         "messages": [{"role": "system", "content": json.dumps(SCHEMA)},
                      {"role": "user", "content": "Ada Lovelace..."}],
         "raw_response": SCHEMA_ECHO_RAW},
        {"event": "llm-call", "attempt": 2, "elapsed_s": 64.0,
         "messages": [{"role": "system", "content": json.dumps(SCHEMA)},
                      {"role": "user", "content": "Ada Lovelace..."}],
         "raw_response": SCHEMA_ECHO_RAW},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        for row in cap:
            fh.write(json.dumps(row) + "\n")
        path = fh.name
    proc = subprocess.run(
        [sys.executable, "scripts/security/summarise_llm_contract.py",
         "--capture", path], cwd=REPO, capture_output=True, text=True,
        timeout=60)
    out = proc.stdout
    check("reports the denominator", "inspected: 2 model call(s)" in out, out[:300])
    check("reports the RESOLVED mode, not the config default",
          "json_mode" in out and "instructor client .mode" in out, out[:900])
    check("says an empty config field is not proof",
          "NOT proof of the effective mode" in out, out[:900])
    check("classifies both attempts as schema echo",
          out.count("SCHEMA ECHO") >= 2, out[:1600])
    check("measures byte-identity across attempts",
          "BYTE-IDENTICAL" in out, out[-1200:])
    check("refuses to assign ownership",
          "NOT CONCLUDED HERE" in out, out[-900:])
    check("and authorises no remedy",
          "No remedy is authorised" in " ".join(out.split()), out[-600:])
    check("exits non-zero on a measured schema echo", proc.returncode != 0,
          f"rc={proc.returncode}")
    # an empty capture must not read as 'no mismatch'
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


def run_all() -> None:
    test_schema_and_instance_are_never_confused()
    test_all_four_kinds_stay_four_verdicts()
    test_validator_failures_are_not_collapsed()
    test_required_fields_come_from_the_schema_sent()
    test_hashes_measure_identity()
    test_the_summariser_reports_and_refuses()
    test_the_collector_gates_on_the_selftest()

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
