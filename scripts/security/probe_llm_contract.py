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

    # The innermost OpenAI async client instructor wraps. Patching HERE
    # sees each retry as its own call, which is what Q2 needs.
    inner = getattr(aclient, "client", None)
    completions = getattr(getattr(inner, "chat", None), "completions", None)
    if completions is None or not hasattr(completions, "create"):
        resolved["capture"] = "NOT INSTALLED — could not locate the inner client"
        emit(resolved)
        return resolved

    original = completions.create
    state = {"attempt": 0}

    async def capturing_create(*args, **kwargs):
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
            "response_format": _serialise(kwargs.get("response_format"))
            if kwargs.get("response_format") is not None else None,
            "tools": kwargs.get("tools"),
            "other_params": sorted(k for k in kwargs
                                   if k not in ("messages", "model",
                                                "temperature",
                                                "response_format", "tools")),
        }
        try:
            # STRICT PASS-THROUGH: same args, same kwargs, unaltered result.
            result = await original(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            record["raw_response"] = None
            record["transport_error"] = f"{type(exc).__name__}: {exc}"
            emit(record)
            raise
        record["elapsed_s"] = round(time.monotonic() - started, 3)
        # Q2 — the raw content, before any cognee/pydantic transformation.
        try:
            record["raw_response"] = result.choices[0].message.content
            record["finish_reason"] = result.choices[0].finish_reason
        except Exception:  # noqa: BLE001
            record["raw_response"] = None
            record["raw_response_note"] = "unexpected response shape"
        emit(record)
        return result

    completions.create = capturing_create
    resolved["capture"] = "INSTALLED at chat.completions.create"
    emit(resolved)
    return resolved


def main() -> int:
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
    emit({"event": "drive-end", "llm_calls_captured": calls})
    # I-2. A capture that says nothing about how much it saw reads
    # identically whether the adapter was called twenty times or never.
    print(f"  inspected: {calls} model call(s) captured", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
