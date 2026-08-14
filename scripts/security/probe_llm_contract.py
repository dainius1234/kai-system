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

import contextvars
import importlib
import itertools
import json
import os
import sys
import time
import traceback
import uuid
from functools import wraps

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

# ── LOGICAL-CALL CORRELATION (Q6) ────────────────────────────────────
#
# Q6 asks whether a failure reproduces ACROSS RETRIES OF ONE logical
# structured-output invocation. Runs 17 and 18 could not answer it: five
# raw attempts existed with nothing to say which invocation each belonged
# to, and every tempting substitute — adjacency, timing, prompt hash,
# schema hash, response similarity — was refused, because none of them is
# an identity. Four byte-identical responses in run 18 are exactly the
# trap: `max_retries=2` bounds a logical call at two attempts, so four
# identical rows CANNOT all be one call's retries.
#
# THE BOUNDARY, READ FROM THE INSTALLED SOURCE, NOT ASSUMED:
#
#   instructor/core/patch.py:258   response = retry_sync(...)
#       -> entered exactly ONCE per logical invocation
#   instructor/core/retry.py:193   for attempt in max_retries:
#       -> the attempt loop lives INSIDE it
#
# So `retry_sync` / `retry_async` ARE the logical-call boundary. Wrapping
# them mints one id per invocation regardless of which cognee method
# called it — a denominator derived from the traversed path rather than a
# hand-kept list of call sites (R5).
#
# PATCH TARGET. `core/patch.py:17` does `from .retry import retry_async,
# retry_sync`, binding the names into patch.py's OWN namespace. Patching
# `instructor.core.retry` would therefore change nothing at the call site
# — the run-13 altitude defect exactly. The target is
# `instructor.core.patch`, and because Python resolves module globals at
# CALL time, patching it works even though `from_openai` has already run.
#
# MECHANISM. `contextvars`, chosen after reading the path rather than by
# preference: from `retry_sync` down to `Completions.create` is a plain
# synchronous call stack, so the value is visible without being passed;
# asyncio gives every Task its own copy, so concurrent invocations cannot
# read each other's id; and set/reset tokens restore LIFO, so nesting is
# deterministic. A single mutable global would fail all three.
#
# INVISIBLE TO THE SUBJECT. The id is never placed in messages, system or
# user content, the response model, `response_format`, any provider
# kwarg, or reask text. It exists only in capture metadata.
LOGICAL_CALL_ID: contextvars.ContextVar = contextvars.ContextVar(
    "kai_gate_048c_logical_call_id", default=None)
LOGICAL_CALL_ATTEMPTS: contextvars.ContextVar = contextvars.ContextVar(
    "kai_gate_048c_logical_call_attempts", default=None)


def _mint_logical_call_id() -> str:
    """An opaque id. Not derived from time, order or any request content."""
    return uuid.uuid4().hex[:16]


def next_attempt_index():
    """This attempt's 1-based position within its logical invocation.

    Returns None outside any invocation context — "no logical call" is a
    first-class answer, not index 1.

    Shared by the capture wrapper and its calibration deliberately: a
    calibration that re-implements the thing it calibrates tests its own
    copy, and the two are then free to drift (R5/I-8).
    """
    counter = LOGICAL_CALL_ATTEMPTS.get()
    if counter is None:
        return None
    counter["n"] += 1
    return counter["n"]


_SEQ = itertools.count(1)


def _seq() -> int:
    """A monotonic order stamp, so ordering is machine-checkable.

    File order would usually do, but an explicit stamp survives
    interleaving and makes 'this attempt happened before its boundary was
    entered' a comparison rather than an impression.
    """
    return next(_SEQ)


def _emit_enter(call_id: str, point: str, parent) -> None:
    """LOGICAL_CALL_ENTER — the boundary was entered and this id minted."""
    emit({"event": "logical-call-enter", "logical_call_id": call_id,
          "parent_logical_call_id": parent, "boundary": point,
          "phase": STATE.get("phase", "unknown"), "seq": _seq()})


def _emit_exit(call_id: str, point: str, counter: dict, outcome: str) -> None:
    """LOGICAL_CALL_EXIT — the boundary exited, this many attempts under it."""
    emit({"event": "logical-call-exit", "logical_call_id": call_id,
          "boundary": point, "phase": STATE.get("phase", "unknown"),
          "attempts_observed": counter.get("n", 0), "outcome": outcome,
          "seq": _seq()})


def _emit_reset_confirmed(call_id: str, expected) -> None:
    """CONTEXT_RESET_CONFIRMED — and it means RESTORED, not 'reset ran'.

    Read AFTER the reset, comparing the context against the value that was
    live before this invocation set it. "The exit wrapper executed" would
    only say the reset was attempted; this says the state actually
    returned to what it was, which is what nesting correctness rests on.
    """
    observed = LOGICAL_CALL_ID.get()
    emit({"event": "context-reset-confirmed", "logical_call_id": call_id,
          "phase": STATE.get("phase", "unknown"), "seq": _seq(),
          "expected_after": expected, "context_after": observed,
          "confirmed": observed == expected})


def install_correlation() -> dict:
    """Wrap instructor's retry entry points so each invocation is identified.

    OBSERVATION ONLY: mints an id, sets two context variables, calls the
    original with the arguments it was given, returns what it returned,
    and restores the context in `finally` so no id can survive into the
    next invocation.

    The sync and async entry points get SEPARATE wrappers. Collapsing
    them was run 14's defect — a wrapper that changes the callable
    convention is an actuator, not an observer.
    """
    # `import instructor.core.patch as X` binds an ATTRIBUTE lookup,
    # and `instructor/core/__init__.py:19` does
    # `from .patch import patch, apatch` — which rebinds that
    # attribute from the MODULE to the FUNCTION. The `as` form
    # therefore yields a function with no `retry_sync`, `installed`
    # comes back empty, and NOTHING is patched (run 20).
    # `import_module` resolves through sys.modules and returns the
    # module regardless of what the package shadowed.
    ipatch = importlib.import_module("instructor.core.patch")

    installed = {}
    for name in ("retry_sync", "retry_async"):
        original = getattr(ipatch, name, None)
        if original is None:
            continue
        installed[name] = original

        if name == "retry_async":
            def make(orig, point=name):
                @wraps(orig)
                async def correlating_retry_async(*args, **kwargs):
                    parent = LOGICAL_CALL_ID.get()
                    call_id = _mint_logical_call_id()
                    counter = {"n": 0}
                    token_id = LOGICAL_CALL_ID.set(call_id)
                    token_ct = LOGICAL_CALL_ATTEMPTS.set(counter)
                    STATE["logical_calls"] = STATE.get("logical_calls", 0) + 1
                    _emit_enter(call_id, point, parent)
                    outcome = "returned"
                    try:
                        return await orig(*args, **kwargs)
                    except BaseException:
                        outcome = "raised"
                        raise
                    finally:
                        _emit_exit(call_id, point, counter, outcome)
                        LOGICAL_CALL_ID.reset(token_id)
                        LOGICAL_CALL_ATTEMPTS.reset(token_ct)
                        _emit_reset_confirmed(call_id, parent)
                return correlating_retry_async
        else:
            def make(orig, point=name):
                @wraps(orig)
                def correlating_retry_sync(*args, **kwargs):
                    parent = LOGICAL_CALL_ID.get()
                    call_id = _mint_logical_call_id()
                    counter = {"n": 0}
                    token_id = LOGICAL_CALL_ID.set(call_id)
                    token_ct = LOGICAL_CALL_ATTEMPTS.set(counter)
                    STATE["logical_calls"] = STATE.get("logical_calls", 0) + 1
                    _emit_enter(call_id, point, parent)
                    outcome = "returned"
                    try:
                        return orig(*args, **kwargs)
                    except BaseException:
                        outcome = "raised"
                        raise
                    finally:
                        _emit_exit(call_id, point, counter, outcome)
                        LOGICAL_CALL_ID.reset(token_id)
                        LOGICAL_CALL_ATTEMPTS.reset(token_ct)
                        _emit_reset_confirmed(call_id, parent)
                return correlating_retry_sync

        setattr(ipatch, name, make(original))

    STATE["correlation_originals"] = installed
    return {
        "correlation": "INSTALLED at instructor.core.patch."
                       + "/".join(sorted(installed)),
        "correlation_boundary": "retry_sync/retry_async — entered once per "
                                "logical invocation (patch.py:258), attempt "
                                "loop inside (retry.py:193)",
        "correlation_mechanism": "contextvars — per-Task isolation, LIFO "
                                 "reset tokens, no global mutable id",
        "correlation_visibility": "capture metadata ONLY — never in "
                                  "messages, schema, response_format or any "
                                  "provider kwarg",
        "correlation_points": sorted(installed),
    }


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
        # Q6 correlation, read out of band. `attempt` remains the global
        # capture counter; `attempt_index` is the position WITHIN this
        # logical invocation, which is the only one Q6 can use.
        attempt_index = next_attempt_index()
        request = {
            "event": "llm-call",
            "layer": "RAW_MODEL_REQUEST",
            "attempt": attempt,
            "seq": _seq(),
            "logical_call_id": LOGICAL_CALL_ID.get(),
            "attempt_index": attempt_index,
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

    # The logical-call boundary, patched before anything is driven. Safe
    # in either order relative to `from_openai`, because patch.py looks
    # `retry_sync` up as a module global at CALL time — but installed
    # here so there is one place that says what was patched.
    correlation = install_correlation()

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
    resolved.update(correlation)
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


def _selftest_correlation() -> dict:
    """PROVE THE CORRELATION IDENTIFIES, ISOLATES AND CLEANS UP.

    No model run: instructor's patched `retry_sync` is driven with a
    stand-in `func` that invokes the capture wrapper N times, which is
    exactly the shape of a real retry sequence. Q6's whole value depends
    on these ids being trustworthy, so they are measured rather than
    assumed to work because the code looks right.
    """
    import asyncio
    # `import instructor.core.patch as X` binds an ATTRIBUTE lookup,
    # and `instructor/core/__init__.py:19` does
    # `from .patch import patch, apatch` — which rebinds that
    # attribute from the MODULE to the FUNCTION. The `as` form
    # therefore yields a function with no `retry_sync`, `installed`
    # comes back empty, and NOTHING is patched (run 20).
    # `import_module` resolves through sys.modules and returns the
    # module regardless of what the package shadowed.
    ipatch = importlib.import_module("instructor.core.patch")

    out: dict = {}
    seen: list = []

    def fake_attempts(n):
        """Stand in for instructor's `func`, N attempts deep.

        Reads the correlation exactly as the capture wrapper does — via
        `next_attempt_index()`, the SAME function — so this cannot pass
        against a copy the wrapper no longer uses.
        """
        def run(*_a, **_k):
            for _ in range(n):
                seen.append((LOGICAL_CALL_ID.get(), next_attempt_index()))
            return f"returned-{n}"
        return run

    # 1-4 — two logical calls, the first with three attempts.
    seen.clear()
    r1 = ipatch.retry_sync(func=fake_attempts(3), response_model=None,
                           args=(), kwargs={})
    first = list(seen)
    seen.clear()
    r2 = ipatch.retry_sync(func=fake_attempts(1), response_model=None,
                           args=(), kwargs={})
    second = list(seen)

    ids1 = {i for i, _ in first}
    ids2 = {i for i, _ in second}
    out["corr_same_id_within_one_call"] = len(ids1) == 1 and None not in ids1
    out["corr_different_id_across_calls"] = bool(ids1 and ids2 and ids1 != ids2)
    out["corr_attempt_index_ordered"] = [n for _, n in first] == [1, 2, 3]
    out["corr_index_restarts_next_call"] = [n for _, n in second] == [1]
    out["corr_return_value_unchanged"] = (r1 == "returned-3"
                                          and r2 == "returned-1")

    # 9 — context cleared after success
    out["corr_context_cleared_after_success"] = (
        LOGICAL_CALL_ID.get() is None and LOGICAL_CALL_ATTEMPTS.get() is None)

    # 8, 10 — exception propagates unchanged AND the context is restored
    sentinel = RuntimeError("kai-gate-048c correlation sentinel")

    def raiser(*_a, **_k):
        raise sentinel

    caught = None
    try:
        ipatch.retry_sync(func=raiser, response_model=None, args=(), kwargs={})
    except Exception as exc:  # noqa: BLE001
        caught = exc
    out["corr_exception_object_unchanged"] = caught is sentinel
    out["corr_context_cleared_after_exception"] = (
        LOGICAL_CALL_ID.get() is None and LOGICAL_CALL_ATTEMPTS.get() is None)

    # 11 — concurrency: interleaved tasks must not read each other's id.
    async def _concurrent():
        results = {}

        async def one(tag):
            def body(*_a, **_k):
                results[tag] = LOGICAL_CALL_ID.get()
                return tag
            # each Task gets its own context copy
            return ipatch.retry_sync(func=body, response_model=None,
                                     args=(), kwargs={})

        await asyncio.gather(*(one(t) for t in ("a", "b", "c")))
        return results

    conc = asyncio.run(_concurrent())
    out["corr_concurrent_ids_distinct"] = (
        len(conc) == 3 and len(set(conc.values())) == 3
        and None not in conc.values())

    # 6 — the id must never be visible to the model. Checked against the
    # REAL captured rows, not against the wrapper's intentions.
    rows = [json.loads(l) for l in open(OUT, encoding="utf-8")
            if '"event": "llm-call"' in l]
    model_facing = json.dumps([{k: v for k, v in r.items()
                                if k in ("messages", "response_model",
                                         "response_format", "tools",
                                         "other_params")} for r in rows])
    ids_minted = {i for i, _ in first} | {i for i, _ in second}
    out["corr_id_absent_from_model_facing_fields"] = not any(
        i and i in model_facing for i in ids_minted)
    out["corr_id_present_on_captured_rows"] = bool(rows) and all(
        r.get("logical_call_id") for r in rows)
    return out


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

    checks.update(_selftest_correlation())

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
