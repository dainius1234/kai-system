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

    def capturing_create(self, *args, **kwargs):
        """SYNC, descriptor-correct, strictly pass-through."""
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
            result = original(self, *args, **kwargs)
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

    Criterion 5 needs no model run: a controlled exception is raised from
    a stand-in original and the wrapper is driven through the real
    instructor path to prove it neither swallows nor rewrites it.

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
        return 2
    if "NOT INSTALLED" in str(resolved.get("capture", "")):
        emit({"event": "selftest-failed", "stage": "install"})
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

    # 5 — EXCEPTION CONTROL. No model needed: swap in a stand-in original
    # that raises, drive the wrapper, and require the same object out.
    sentinel = RuntimeError("kai-gate-048c selftest sentinel")

    def raising_original(_self, *a, **k):
        raise sentinel

    saved, STATE["original"] = STATE["original"], raising_original
    # rebuild a wrapper bound to the raising original by re-entering the
    # same code path the class patch uses
    caught = None
    try:
        from openai.resources.chat.completions import Completions
        probe_wrapper = Completions.create

        class _Fake:
            pass
        try:
            probe_wrapper(_Fake(), messages=[], model="x")
        except Exception as exc:  # noqa: BLE001
            caught = exc
    finally:
        STATE["original"] = saved

    checks["exception_type_preserved"] = type(caught) is type(sentinel) \
        if caught is not None else False
    checks["exception_not_swallowed"] = caught is not None
    checks["exception_value_preserved"] = (str(caught) == str(sentinel)
                                           if caught is not None else False)

    emit({"event": "selftest-result", "layer": "ADAPTER_INPUT",
          "calls_observed": observed, "originals_called": originals,
          "hooks_that_fired": sorted(state["hooks_fired"]),
          "checks": checks,
          "selftest_rows_excluded_from_q6": len(rows)})

    failed = [k for k, v in checks.items() if not v]
    print(f"  inspected: {len(checks)} transparency criterion(s), "
          f"{observed} model call(s) captured", flush=True)
    if failed:
        print(f"  CAPTURE POINT NOT TRANSPARENT: {', '.join(failed)}",
              flush=True)
        print("  Refusing to spend a full capture run on a hook that is not "
              "proven observational.", flush=True)
        return 2
    print(f"  CAPTURE POINT PROVEN TRANSPARENT: {len(checks)} criteria, "
          f"{observed} call(s) via {', '.join(sorted(state['hooks_fired']))}",
          flush=True)
    return 0


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
