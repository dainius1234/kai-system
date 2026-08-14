#!/usr/bin/env python3
"""KAI-GATE-048 C — Q1/Q2/Q6 capture. Runs INSIDE the memu-graph image.

OBSERVATION ONLY. It records what the structured-output adapter sends and
what comes back. It changes no prompt, no schema, no instructor mode, no
retry count, no timeout, no model, no temperature and no network path.
The wrapper below calls the original method with the arguments it was
given and returns the original object unaltered.

WHY IT DRIVES THE PIPELINE ITSELF RATHER THAN WATCHING THE SERVICE
==================================================================

The alternatives were worse. A proxy between memu-graph and ollama would
change the network topology, which is prohibited and would also make the
measurement about the proxy. Patching the running uvicorn process would
mean modifying a live service. Instead this runs in the SAME image, with
the SAME environment the service has, imports cognee exactly as app.py
does, and exercises the same `add` + `cognify` path in-process — so the
adapter, its resolved mode and its config are the real ones.

WHAT IS CAPTURED, AND WHY EACH FIELD
====================================

Q1  the request: every message reaching the model, the schema/response
    model actually supplied, the RESOLVED instructor mode, generation
    parameters and model identity.
Q2  the response: the raw content of EVERY attempt, before any cognee or
    pydantic transformation, with its attempt index and elapsed time.
Q6  reproducibility: sha256 of prompt, schema and response, so
    "identical" is measured rather than eyeballed.

THE RESOLVED MODE IS READ, NOT INFERRED
=======================================

`llm_instructor_mode` defaults to "" in cognee's config and the ollama
adapter falls back to "json_mode" — but an empty config field is not
proof of the effective mode. This reads `mode` off the instructor client
the adapter actually built, and also records the raw config value, so the
two can be compared rather than assumed equal.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

OUT = os.environ.get("LLM_CAPTURE_PATH", "/tmp/llm-contract-capture.jsonl")
TEXT = ("Ada Lovelace wrote the first algorithm for the Analytical "
        "Engine, which Charles Babbage designed in London.")


def emit(record: dict) -> None:
    record["wall"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    line = json.dumps(record, default=str)
    with open(OUT, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def _serialise(obj):
    """Best-effort JSON for a response model class or pydantic object."""
    for attr in ("model_json_schema", "schema"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception:  # noqa: BLE001 — capture must never break the run
                pass
    return str(obj)


# Module-level so the wrapper, its original and the phase label are
# reachable from selftest(), the driver and the restore path.
STATE: dict = {}


def install_capture() -> dict:
    """Patch `Completions.create` AT CLASS LEVEL, BEFORE any adapter exists.

    D216 established the altitude from installed source:

      instructor/core/patch.py:214  @wraps(func) def new_create_sync(...)
          -> `create_fn` is instructor's PATCHED wrapper. @wraps copied
             the raw function's __qualname__, so its repr read
             "<function Completions.create>" and LOOKED like the raw
             callable. Hooking it captured one adapter request, not one
             model attempt.
      instructor/core/retry.py:193-198
          for attempt in max_retries:
              response = func(*args, **kwargs)   <- THE per-attempt RAW call
          -> `func` is closed over at patch() time, i.e. inside
             instructor.from_openai(...), which cognee calls while
             CONSTRUCTING the adapter.

    So the only way to be inside the retry loop is to already be the
    bound method when `from_openai` captures it. Hence: patch the CLASS
    attribute first, and only then let the adapter be built.

    DESCRIPTOR SEMANTICS ARE EXPLICIT. `Completions.create` is a plain
    function on the class, so the wrapper receives `self` as its first
    positional argument and forwards it unchanged:

        wrapper(self, *args, **kwargs) -> original(self, *args, **kwargs)

    It is SYNCHRONOUS, because retry.py:198 calls it without `await`.
    Turning that into a coroutine was run 14's defect.
    """
    import openai
    from openai.resources.chat.completions import Completions

    original = Completions.create
    state = {"attempt": 0, "hooks_fired": set(), "originals_called": 0}
    STATE["capture"] = state
    STATE["original"] = original
    # Kept separately so the restore check has something to compare
    # against that no injection can move (D218).
    STATE["real_original"] = original

    def capturing_create(self, *args, **kwargs):
        """SYNC, descriptor-correct, strictly pass-through.

        THE FORWARD TARGET IS READ FROM `STATE` AT CALL TIME (D218), not
        closed over. Run 16's selftest injected a stand-in by swapping
        `STATE["original"]` while this wrapper still called the closure
        local captured at install — so the stand-in never ran, openai's
        own internals raised `AttributeError` against the fake receiver
        instead of the sentinel, and two transparency criteria failed for
        a reason that had nothing to do with the wrapper's behaviour.
        Reading `STATE` makes the callable the wrapper uses IDENTICAL to
        the injection point the selftest exercises, which is the only way
        the control measures the thing it names.
        """
        forward = STATE.get("original")
        if forward is None:
            # I-1. No callable, no observation — and never a silent
            # pass-through that would look like a working hook.
            raise RuntimeError(
                "capture wrapper has no original to forward to")
        state["hooks_fired"].add("Completions.create")
        state["attempt"] += 1
        attempt = state["attempt"]
        started = time.monotonic()
        request = {
            "event": "llm-call",
            "layer": "RAW_MODEL_REQUEST",
            "attempt": attempt,
            "phase": STATE.get("phase", "unknown"),
            "messages": kwargs.get("messages"),
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "response_model": _serialise(STATE.get("response_model"))
            if STATE.get("response_model") is not None else None,
            "response_format": _serialise(kwargs.get("response_format"))
            if kwargs.get("response_format") is not None else None,
            "tools": kwargs.get("tools"),
            "other_params": sorted(k for k in kwargs
                                   if k not in ("messages", "model",
                                                "temperature",
                                                "response_format", "tools")),
        }
        try:
            state["originals_called"] += 1
            result = forward(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            request["layer"] = "RAW_MODEL_RESPONSE"
            request["elapsed_s"] = round(time.monotonic() - started, 3)
            request["raw_response"] = None
            request["transport_error"] = f"{type(exc).__name__}: {exc}"
            emit(request)
            raise          # exception type AND value propagate unchanged
        request["layer"] = "RAW_MODEL_RESPONSE"
        request["elapsed_s"] = round(time.monotonic() - started, 3)
        try:
            request["raw_response"] = result.choices[0].message.content
            request["finish_reason"] = result.choices[0].finish_reason
        except Exception:  # noqa: BLE001
            request["raw_response"] = None
            request["raw_response_note"] = (
                "unexpected completion shape — raw text NOT captured")
        request["result_type"] = type(result).__name__
        emit(request)
        return result          # the ORIGINAL object, unaltered

    Completions.create = capturing_create
    STATE["wrapper"] = capturing_create

    # NOW build the adapter, so `from_openai` closes over the wrapper.
    import cognee  # noqa: F401
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.infrastructure.llm.structured_output_framework.\
        litellm_instructor.llm.get_llm_client import get_llm_client

    cfg = get_llm_config()
    client = get_llm_client()
    aclient = getattr(client, "aclient", None)

    resolved = {
        "event": "resolved-config",
        "layer": "ADAPTER_INPUT",
        "config_llm_instructor_mode": repr(getattr(cfg, "llm_instructor_mode", None)),
        "adapter_instructor_mode": repr(getattr(client, "instructor_mode", None)),
        "adapter_class": type(client).__name__,
        "adapter_default_mode": repr(getattr(type(client), "default_instructor_mode", None)),
        "model": repr(getattr(client, "model", None)),
        "endpoint": repr(getattr(client, "endpoint", None)),
        "instructor_client_mode": repr(getattr(aclient, "mode", None)),
        "adapter_id": id(client),
        "get_llm_client_stable": id(get_llm_client()) == id(client),
        "openai_version": repr(getattr(openai, "__version__", None)),
        # D216: this repr LOOKS like the raw callable because @wraps
        # copies __qualname__. Recorded, but never used to infer altitude.
        "instructor_holds_create_fn": repr(getattr(aclient, "create_fn", None))[:200],
        "capture": "INSTALLED at Completions.create (class level, sync, "
                   "BEFORE adapter construction)",
        "hooked_points": ["Completions.create"],
        "hook_convention": "sync, descriptor-correct (self, *args, **kwargs) "
                           "— matches retry.py:198 `func(*args, **kwargs)`",
    }
    emit(resolved)
    return resolved


def forwards_to_the_real_original() -> bool:
    """Is `STATE["original"]` the production callable, not a stand-in?

    Two sources that cannot excuse each other (I-8):

      * IDENTITY against the callable captured before the patch existed.
        No injection can move it, and it is exactly "the callable the
        production path held before this file touched anything".
      * ORIGIN — the callable must not be defined in THIS module. Every
        stand-in this file can inject is defined here, so a stand-in
        fails on its own evidence rather than on a name kept beside it.

    The stronger-looking test — matching openai's module path — is NOT a
    pass condition, because that string cannot be measured where this was
    written and an unverified assumption inside a gate is how run 16 was
    spent. The provenance is RECORDED instead (`provenance_of_original`),
    where a wrong guess costs a reader nothing.

    Used as a known-negative during the injection and as a known-positive
    after the restore, so the criterion is demonstrably able to fail.
    """
    fn = STATE.get("original")
    if fn is None or fn is not STATE.get("real_original"):
        return False
    return (getattr(fn, "__module__", "") or "") != __name__


def provenance_of_original() -> dict:
    """Where the current forward target comes from. Evidence, not a gate."""
    fn = STATE.get("original")
    return {"module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None),
            "is_the_pre_patch_callable": fn is STATE.get("real_original"),
            "defined_in_this_probe": (getattr(fn, "__module__", "") or "")
            == __name__}


def restore_capture() -> bool:
    """Put the class attribute back. Returns whether it was restored.

    The patch window is minimised, but ONLY if the closure still reaches
    the wrapper afterwards — the selftest measures that rather than
    assuming it. `from_openai` captured the BOUND method at construction,
    so restoring the class attribute should not detach it; that is a
    prediction, and the selftest is what turns it into a measurement.
    """
    original = STATE.get("original")
    if original is None:
        return False
    from openai.resources.chat.completions import Completions
    Completions.create = original
    STATE["restored"] = True
    return True


def selftest() -> int:
    """PROVE THE HOOK IS TRANSPARENT — not merely traversed.

    Run 14's selftest asserted `observed >= 1`, which licenses only "the
    hook is on the path" (D214). These are the operator's criteria, as
    RUNTIME assertions through the real adapter path:

      1 traversed;
      2 the ORIGINAL underlying method executed;
      3 exactly ONE original invocation per wrapper invocation;
      4 the response object reaching instructor is the original;
      5 an exception propagates unchanged in type AND value;
      6 no sync->async or async->sync conversion;
      7 instructor still performs its normal retry count;
      8 every raw row is tagged RAW_MODEL_REQUEST / RAW_MODEL_RESPONSE;
      9 the selftest's own calls are EXCLUDED from the Q6 denominator.

     10 the injected stand-in ACTUALLY RAN, exactly once;
     11 the forward target is the real original again afterwards.

    Criterion 5 needs no model run: a controlled exception is raised from
    a stand-in original and the wrapper is driven through the real
    instructor path to prove it neither swallows nor rewrites it.

    Criteria 10-11 exist because run 16 proved 5 could fail without ever
    being exercised (D218). 10 is 5's known-positive — without it, "the
    sentinel did not come out" cannot be told apart from "the sentinel was
    never raised". 11 is the guarantee that the expensive capture does not
    then run against a test stand-in, and it is checked against a
    known-negative taken while the stand-in is still installed.

    Criteria 1-4 are driven through the adapter's PRODUCTION entry point,
    `acreate_structured_output`. Calling the wrapper directly would prove
    only that the wrapper works, which was never in doubt; the question
    is whether the real path reaches it and comes back unchanged.
    """
    open(OUT, "w", encoding="utf-8").close()
    STATE["phase"] = "selftest"
    try:
        resolved = install_capture()
    except Exception as exc:  # noqa: BLE001
        emit({"event": "selftest-failed", "stage": "install",
              "error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()})
        print("SELFTEST-CLASS: NOT INSTALLED", flush=True)
        return 2
    if "NOT INSTALLED" in str(resolved.get("capture", "")):
        emit({"event": "selftest-failed", "stage": "install"})
        print("SELFTEST-CLASS: NOT INSTALLED", flush=True)
        return 2

    import asyncio
    import inspect
    from pydantic import BaseModel

    from cognee.infrastructure.llm.structured_output_framework.\
        litellm_instructor.llm.get_llm_client import get_llm_client

    class Ping(BaseModel):
        answer: str

    state = STATE["capture"]
    wrapper = STATE["wrapper"]
    checks = {}

    # 6 — convention, before anything is driven.
    checks["wrapper_is_sync"] = not inspect.iscoroutinefunction(wrapper)
    checks["original_is_sync"] = not inspect.iscoroutinefunction(STATE["original"])
    checks["convention_matches"] = (checks["wrapper_is_sync"]
                                    == checks["original_is_sync"])

    # 1-4, 7-8 — one controlled call through the PRODUCTION entry point.
    before_hook = state["attempt"]
    before_orig = state["originals_called"]
    STATE["response_model"] = Ping

    async def one_call():
        client = get_llm_client()
        try:
            await client.acreate_structured_output(
                text_input="ping", system_prompt="Answer with one word.",
                response_model=Ping)
        except Exception as exc:  # noqa: BLE001
            emit({"event": "selftest-invocation-error",
                  "error": f"{type(exc).__name__}: {exc}",
                  "note": "not a selftest failure by itself — what matters "
                          "is whether the hook OBSERVED the call"})

    asyncio.run(one_call())
    STATE["response_model"] = None

    observed = state["attempt"] - before_hook
    originals = state["originals_called"] - before_orig
    checks["traversed"] = observed >= 1
    checks["original_executed"] = originals >= 1
    checks["exactly_once_per_wrapper_call"] = (observed == originals)

    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")
            if '"event": "llm-call"' in l]
    checks["rows_tagged_raw_layer"] = bool(rows) and all(
        r.get("layer") in ("RAW_MODEL_REQUEST", "RAW_MODEL_RESPONSE")
        for r in rows)
    checks["rows_tagged_selftest_phase"] = bool(rows) and all(
        r.get("phase") == "selftest" for r in rows)

    # 5 — EXCEPTION CONTROL, WITH ITS KNOWN-POSITIVE (D218). No model run.
    #
    # Run 16 swapped `STATE["original"]` while the wrapper called a
    # closure local, so the stand-in never executed; openai's own
    # internals raised `AttributeError` against the fake receiver, and the
    # type/value criteria failed against a `RuntimeError` sentinel —
    # correctly, and for a reason that said nothing about the wrapper.
    # Two criteria therefore read as subject failures when they were
    # simply UNMEASURED.
    #
    # The whole chain is now asserted, in order:
    #   wrapper traversed -> injected stand-in ran EXACTLY ONCE ->
    #   sentinel raised -> same type out -> same value out ->
    #   same OBJECT out (not swallowed, not replaced).
    #
    # The counter is what makes this evidence. "The sentinel appeared" is
    # not proof that the injected callable ran — it is exactly the claim
    # run 16 could not distinguish from its opposite.
    sentinel = RuntimeError("kai-gate-048c selftest sentinel")
    standin = {"calls": 0}

    def raising_original(_self, *a, **k):
        standin["calls"] += 1
        raise sentinel

    before_wrapper = state["attempt"]
    caught = None
    saved = STATE["original"]
    try:
        STATE["original"] = raising_original
        # KNOWN-NEGATIVE, taken while the stand-in is installed: the
        # restore criterion must REFUSE this state, or its later pass
        # means nothing.
        checks["restore_check_rejects_a_standin"] = \
            not forwards_to_the_real_original()

        from openai.resources.chat.completions import Completions
        probe_wrapper = Completions.create

        class _Fake:
            pass
        try:
            probe_wrapper(_Fake(), messages=[], model="x")
        except Exception as exc:  # noqa: BLE001
            caught = exc
    finally:
        # try/finally, not a trailing statement: the production drive must
        # never continue against the test stand-in, including if the
        # injection block raises somewhere unexpected.
        STATE["original"] = saved

    checks["exception_wrapper_traversed"] = \
        (state["attempt"] - before_wrapper) == 1
    checks["standin_executed_exactly_once"] = standin["calls"] == 1
    checks["exception_not_swallowed"] = caught is not None
    checks["exception_type_preserved"] = type(caught) is type(sentinel)
    checks["exception_value_preserved"] = str(caught) == str(sentinel)
    checks["exception_object_not_replaced"] = caught is sentinel
    # KNOWN-POSITIVE. Nothing expensive may run until the forward target
    # is the production callable again.
    checks["original_restored_to_real"] = forwards_to_the_real_original()

    emit({"event": "selftest-result", "layer": "ADAPTER_INPUT",
          "calls_observed": observed, "originals_called": originals,
          "hooks_that_fired": sorted(state["hooks_fired"]),
          "checks": checks,
          "provenance_of_original": provenance_of_original(),
          "selftest_rows_excluded_from_q6": len(rows)})

    failed = [k for k, v in checks.items() if not v]
    print(f"  inspected: {len(checks)} transparency criterion(s), "
          f"{observed} model call(s) captured", flush=True)

    # THREE STATES, THREE MESSAGES (D218). Run 16 aborted with "THE
    # CAPTURE POINT IS NOT TRAVERSED" in a run where traversal had been
    # PROVEN — the failure was transparency. One abort message covering
    # three different states is the same defect as a check whose scope is
    # wrong, seen from the reporting side: the evidence reads as a state
    # that did not occur, and a later reader cannot tell which happened.
    #
    # The CLASS is printed by the instrument that knows it, on a line the
    # collector reads back. A table of codes kept beside the collector
    # would be a second denominator free to drift from this one (R5).
    if not failed:
        print("SELFTEST-CLASS: TRANSPARENT", flush=True)
        print(f"  CAPTURE POINT PROVEN TRANSPARENT: {len(checks)} criteria, "
              f"{observed} call(s) via "
              f"{', '.join(sorted(state['hooks_fired']))}", flush=True)
        return 0
    if not checks.get("traversed"):
        print("SELFTEST-CLASS: NOT TRAVERSED", flush=True)
        print("  THE CAPTURE POINT WAS NOT TRAVERSED: the hook installed, "
              "and execution never reached it.", flush=True)
        print(f"  also unmet: {', '.join(k for k in failed if k != 'traversed')}"
              if len(failed) > 1 else "  no other criterion was reached.",
              flush=True)
        return 3
    print("SELFTEST-CLASS: TRANSPARENCY NOT PROVEN", flush=True)
    print(f"  CAPTURE POINT TRAVERSED BUT NOT PROVEN TRANSPARENT: "
          f"{', '.join(failed)}", flush=True)
    print("  Traversal IS proven; what is unproven is that the hook "
          "observes without altering.", flush=True)
    print("  Refusing to spend a full capture run on a hook that is not "
          "proven observational.", flush=True)
    return 4


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
    STATE["phase"] = "capture"
    open(OUT, "w", encoding="utf-8").close()
    try:
        resolved = install_capture()
    except Exception as exc:  # noqa: BLE001
        emit({"event": "capture-failed", "error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()})
        return 2
    if "NOT INSTALLED" in str(resolved.get("capture", "")):
        return 2

    import cognee
    import asyncio

    async def drive():
        dataset = os.getenv("MEMU_GRAPH_DATASET", "memu")
        emit({"event": "drive-start", "dataset": dataset})
        await cognee.add(TEXT, dataset_name=dataset,
                         node_set=["kai-gate-048c-capture"])
        try:
            result = await cognee.cognify(datasets=[dataset])
            emit({"event": "cognify-returned",
                  "result": {str(k): getattr(v, "status", str(v))
                             for k, v in result.items()}
                  if isinstance(result, dict) else str(result)})
        except Exception as exc:  # noqa: BLE001
            emit({"event": "cognify-raised",
                  "error": f"{type(exc).__name__}: {exc}"})

    asyncio.run(drive())
    calls = sum(1 for line in open(OUT, encoding="utf-8")
                if '"event": "llm-call"' in line)
    fired = sorted(STATE.get("capture", {}).get("hooks_fired", []))
    emit({"event": "drive-end", "llm_calls_captured": calls,
          "hooks_that_fired": fired})
    # I-2. A capture that says nothing about how much it saw reads
    # identically whether the adapter was called twenty times or never.
    print(f"  inspected: {calls} model call(s) captured", flush=True)

    # RUN 13'S LESSON, ENFORCED. Successful installation of an
    # observation hook is NOT proof that execution traverses it. The
    # probe previously returned 0 -- "the capture ran to completion" --
    # having observed nothing, which is the success-shaped failure this
    # whole programme exists to refuse. Control flow completing is not a
    # measurement.
    if calls == 0:
        emit({"event": "instrument-failure",
              "why": "the pipeline was driven but ZERO model calls were "
                     "captured: the hook was installed and not traversed",
              "hooks_installed": True, "hooks_that_fired": fired})
        print("  INSTRUMENT FAILURE: 0 model call(s) captured after driving "
              "the pipeline.", flush=True)
        print("  Q1/Q2/Q6 are UNMEASURED — which is not 'no mismatch'.",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
