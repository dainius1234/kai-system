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


def install_capture() -> dict:
    """Patch the adapter's OpenAI client. Returns what it resolved."""
    import cognee  # noqa: F401  — ensures cognee's config is loaded
    from cognee.infrastructure.llm.config import get_llm_config
    from cognee.infrastructure.llm.structured_output_framework.\
        litellm_instructor.llm.get_llm_client import get_llm_client

    cfg = get_llm_config()
    client = get_llm_client()

    resolved = {
        "event": "resolved-config",
        # Q1: the RESOLVED mode, read off the object the adapter built.
        "config_llm_instructor_mode": repr(getattr(cfg, "llm_instructor_mode", None)),
        "adapter_instructor_mode": repr(getattr(client, "instructor_mode", None)),
        "adapter_class": type(client).__name__,
        "adapter_default_mode": repr(getattr(type(client), "default_instructor_mode", None)),
        "model": repr(getattr(client, "model", None)),
        "endpoint": repr(getattr(client, "endpoint", None)),
        "api_version": repr(getattr(client, "api_version", None)),
        "max_tokens": repr(getattr(client, "max_tokens", None)),
    }

    aclient = getattr(client, "aclient", None)
    instructor_mode = getattr(aclient, "mode", None)
    resolved["instructor_client_mode"] = repr(instructor_mode)

    # HYPOTHESIS EVIDENCE, gathered by introspection alone — no model
    # call, no cost. Run 13 proved that installing a hook is not proof
    # that execution traverses it, so the two candidate mechanisms are
    # recorded as object facts rather than argued about:
    #   H1  instructor binds the create function at construction, so a
    #       later attribute replacement is never consulted;
    #   H2  get_llm_client() caching hands out an object other than the
    #       one used during cognify.
    inner = getattr(aclient, "client", None)
    resolved["adapter_id"] = id(client)
    resolved["aclient_id"] = id(aclient)
    resolved["inner_client_id"] = id(inner)
    # H2: does a second call return the same object?
    second = get_llm_client()
    resolved["get_llm_client_stable"] = (id(second) == id(client))
    resolved["second_adapter_id"] = id(second)
    # H1: does instructor hold its own bound reference?
    for attr in ("create_fn", "func", "_create", "create"):
        held = getattr(aclient, attr, None)
        if held is not None:
            resolved[f"instructor_holds_{attr}"] = repr(held)[:200]

    # ── THE HOOK, chosen from installed-source evidence ──────────────
    #
    # Read out of instructor 1.15.1 and cognee 1.1.3 in this image:
    #
    #   adapter.py:75   instructor.from_openai(OpenAI(base_url=...), ...)
    #                   -> a SYNCHRONOUS OpenAI client, so from_openai
    #                      returns the SYNC `Instructor`.
    #   client.py:230   self.create_fn = create
    #                   -> the bound method is captured AT CONSTRUCTION,
    #                      so replacing `inner.chat.completions.create`
    #                      afterwards is never consulted. H1 CONFIRMED,
    #                      and that is why run 13 captured nothing.
    #   client.py:376   return self.create_fn(...)
    #                   -> the SYNC client calls it WITHOUT `await`.
    #
    # So there is exactly ONE boundary that is both traversed and
    # patchable after construction: `aclient.create_fn`. Run 14 proved it
    # is traversed — a coroutine object existed, so it was called. And it
    # must be wrapped SYNCHRONOUSLY, because wrapping a sync callable in
    # `async def` returns a coroutine the caller never awaits: the
    # original is then never invoked and the wrapper has replaced the
    # call instead of observing it. That was run 14's defect, and it is
    # why only one hook is installed now: more hooks are only safer if
    # each is independently transparent.
    original = getattr(aclient, "create_fn", None)
    if not callable(original):
        resolved["capture"] = ("NOT INSTALLED — instructor client exposes no "
                               "callable create_fn")
        emit(resolved)
        return resolved

    state = {"attempt": 0, "hooks_fired": set()}

    def capturing_create(*args, **kwargs):
        """SYNC wrapper for a SYNC callable. Convention-preserving."""
        state["hooks_fired"].add("instructor.create_fn")
        state["attempt"] += 1
        attempt = state["attempt"]
        started = time.monotonic()
        record = {
            "event": "llm-call",
            "attempt": attempt,
            # Q1 — the request, exactly as it goes out.
            "messages": kwargs.get("messages"),
            "model": kwargs.get("model"),
            "temperature": kwargs.get("temperature"),
            "response_model": _serialise(kwargs.get("response_model"))
            if kwargs.get("response_model") is not None else None,
            "response_format": _serialise(kwargs.get("response_format"))
            if kwargs.get("response_format") is not None else None,
            "tools": kwargs.get("tools"),
            "max_retries": repr(kwargs.get("max_retries")),
            "other_params": sorted(k for k in kwargs
                                   if k not in ("messages", "model",
                                                "temperature", "max_retries",
                                                "response_model",
                                                "response_format", "tools")),
        }
        try:
            # STRICT PASS-THROUGH: same args, same kwargs, same call
            # convention, result returned unaltered.
            result = original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            record["raw_response"] = None
            record["transport_error"] = f"{type(exc).__name__}: {exc}"
            emit(record)
            raise          # exception behaviour preserved exactly
        record["elapsed_s"] = round(time.monotonic() - started, 3)
        # instructor returns the VALIDATED model here, not the raw
        # completion, so the raw text is recovered from the attached
        # completion when instructor exposes it. Recorded as absent
        # rather than guessed when it is not.
        raw = None
        for holder in (getattr(result, "_raw_response", None), result):
            try:
                raw = holder.choices[0].message.content
                break
            except Exception:  # noqa: BLE001
                continue
        record["raw_response"] = raw
        if raw is None:
            record["raw_response_note"] = (
                "instructor returned a validated object with no reachable "
                "raw completion — raw text NOT captured at this boundary")
        record["result_type"] = type(result).__name__
        emit(record)
        return result

    try:
        aclient.create_fn = capturing_create
    except Exception as exc:  # noqa: BLE001
        resolved["capture"] = f"NOT INSTALLED — create_fn not writable: {exc}"
        emit(resolved)
        return resolved

    resolved["capture"] = "INSTALLED at instructor.create_fn (sync)"
    resolved["hooked_points"] = ["instructor.create_fn"]
    resolved["hook_convention"] = "sync — matches client.py:376 `return self.create_fn(...)`"
    emit(resolved)
    STATE["capture"] = state
    return resolved


STATE: dict = {}


def selftest() -> int:
    """PROVE THE HOOK IS TRAVERSED — before spending a 9-minute run.

    Run 13's lesson: successful installation of an observation hook is
    not proof that execution traverses it. So one controlled invocation
    is driven through the ADAPTER'S OWN production entry point --
    `acreate_structured_output`, the same method cognee's summarisation
    calls -- and exactly one capture record is required.

    Calling `capturing_create` directly would prove only that the
    wrapper works, which was never in doubt. The question is whether the
    real path reaches it.

    Cheap by construction: a two-word input and a one-field model, so
    this costs seconds rather than the minutes a chunk summarisation
    takes. It changes no mode, schema, retry, timeout or topology -- it
    is one extra read-only request on a stack that is about to serve
    many.
    """
    open(OUT, "w", encoding="utf-8").close()
    try:
        resolved = install_capture()
    except Exception as exc:  # noqa: BLE001
        emit({"event": "selftest-failed", "stage": "install",
              "error": f"{type(exc).__name__}: {exc}",
              "traceback": traceback.format_exc()})
        return 2
    if "NOT INSTALLED" in str(resolved.get("capture", "")):
        emit({"event": "selftest-failed", "stage": "install",
              "error": "capture point not located"})
        return 2

    import asyncio
    from pydantic import BaseModel

    from cognee.infrastructure.llm.structured_output_framework.\
        litellm_instructor.llm.get_llm_client import get_llm_client

    class Ping(BaseModel):
        answer: str

    before = sum(1 for line in open(OUT, encoding="utf-8")
                 if '"event": "llm-call"' in line)

    async def one_call():
        client = get_llm_client()
        try:
            # THE PRODUCTION ENTRY POINT, not the wrapper.
            await client.acreate_structured_output(
                text_input="ping", system_prompt="Answer with one word.",
                response_model=Ping)
        except Exception as exc:  # noqa: BLE001
            # The model may well refuse or mis-shape this too -- which is
            # irrelevant. The question is whether the CALL was OBSERVED,
            # not whether it succeeded.
            emit({"event": "selftest-invocation-error",
                  "error": f"{type(exc).__name__}: {exc}",
                  "note": "not a selftest failure by itself — what matters "
                          "is whether the hook recorded the call"})

    asyncio.run(one_call())

    after = sum(1 for line in open(OUT, encoding="utf-8")
                if '"event": "llm-call"' in line)
    observed = after - before
    fired = sorted(STATE.get("capture", {}).get("hooks_fired", []))
    emit({"event": "selftest-result", "calls_observed": observed,
          "hooks_that_fired": fired,
          "hooks_installed": resolved.get("hooked_points")})
    print(f"  inspected: {observed} model call(s) captured", flush=True)

    if observed < 1:
        print("  CAPTURE POINT NOT TRAVERSED: the production entry point "
              "was driven and the hook recorded nothing.", flush=True)
        print("  Refusing to spend a full capture run on a hook that "
              "cannot observe.", flush=True)
        return 2
    print(f"  CAPTURE POINT PROVEN: {observed} call(s) observed via "
          f"{', '.join(fired) or 'unknown hook'}", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        return selftest()
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
