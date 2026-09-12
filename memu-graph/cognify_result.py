"""KAI-GATE-050: is a cognee pipeline result a terminal SUCCESS?

Pure stdlib, no third-party imports, no I/O — so the predicate can be
calibrated without cognee, without fastapi and without a container.

WHY THIS EXISTS
===============

`memu-graph/app.py` treated *"`await cognee.cognify(...)` returned
without raising"* as success. cognee's observed failure mode is
**returned, not raised**: `run_tasks.py:147` raises
`PipelineRunFailedError`, and `:185-187` catches it, yields
`PipelineRunErrored`, and deliberately does not re-raise — the one
exception type meaning *the pipeline failed* is the one excluded. The
return value was discarded, so the boundary stayed blind while its
exception handler remained intact. Reproduced 2/2 on clean stacks
(D201).

THE PREDICATE IS A CLASS, NOT THE OBSERVED INSTANCE
===================================================

This deliberately does **not** ask *"does the result mention
PipelineRunFailedError?"*. That would fix the one failure we happened to
see and stay blind to the next one. It asks:

    Did cognee return a terminal SUCCESSFUL pipeline result?

and refuses success otherwise — including for statuses that do not exist
yet, for a run left mid-flight, for an empty result, and for a shape this
code does not recognise. I-1: an input that cannot be understood is a
failure to answer, not a pass.

THE CONTRACT, READ FROM cognee 1.1.3
====================================

`cognify()` defaults to `run_in_background=False`
(`api/v1/cognify/cognify.py:53`), so it awaits `run_pipeline_blocking`
(`modules/pipelines/layers/pipeline_execution_mode.py`), which returns

    Dict[dataset_id -> the LAST run_info yielded for that dataset]

or, when a `run_info` carries no `dataset_id`, that bare `run_info`
instead. Every `run_info` is a `PipelineRunInfo` with a `.status` string
(`modules/pipelines/models/PipelineRunInfo.py`):

    PipelineRunStarted  PipelineRunYield  PipelineRunCompleted
    PipelineRunAlreadyCompleted           PipelineRunErrored

Because the dict keeps the LAST yield per dataset, that status IS the
terminal state.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

# The only two statuses that mean "this pipeline finished successfully".
# `AlreadyCompleted` is a success: re-ingesting unchanged data is a
# legitimate no-op, and calling it a failure would break idempotent
# callers to fix a different bug.
TERMINAL_SUCCESS = frozenset({
    "PipelineRunCompleted",
    "PipelineRunAlreadyCompleted",
})


def _status_of(run_info: Any) -> str:
    """The status string, or '' when this object does not carry one."""
    status = getattr(run_info, "status", None)
    if status is None and isinstance(run_info, dict):
        status = run_info.get("status")
    return str(status) if status is not None else ""


def evaluate(result: Any) -> Tuple[bool, Dict[str, str], str]:
    """(is a terminal success, {key: status}, why not).

    `key` is the dataset id where cognee provided one, so a caller can
    say *which* dataset failed rather than only that something did.
    """
    if result is None:
        return False, {}, ("cognify returned None — no pipeline run "
                           "information, so success cannot be established")

    # A mapping of dataset_id -> run_info is the normal shape.
    if isinstance(result, dict):
        if not result:
            return False, {}, ("cognify returned an empty result — no "
                               "pipeline ran, so success cannot be "
                               "established")
        states = {str(k): _status_of(v) for k, v in result.items()}
    else:
        # A bare run_info, which run_pipeline_blocking returns when the
        # run_info carries no dataset_id.
        states = {"<no-dataset-id>": _status_of(result)}

    unknown = [k for k, v in states.items() if not v]
    if unknown:
        # Fail closed on a shape we cannot read. If cognee changes its
        # return type, this must become a loud failure rather than a
        # silent success — the exact way the original defect worked.
        return False, states, (
            f"cognify returned {len(unknown)} entry(ies) with no readable "
            f"status ({', '.join(unknown)}) — the result shape is not "
            f"understood, so success cannot be established")

    bad = {k: v for k, v in states.items() if v not in TERMINAL_SUCCESS}
    if bad:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(bad.items()))
        return False, states, (
            f"{len(bad)} of {len(states)} pipeline(s) did not reach a "
            f"terminal successful state ({detail}); terminal success is "
            f"one of {sorted(TERMINAL_SUCCESS)}")

    return True, states, ""
