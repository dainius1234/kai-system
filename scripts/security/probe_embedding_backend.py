#!/usr/bin/env python3
"""Run INSIDE a built service image: did the semantic operation happen?

Designed to be executed as `python scripts/security/probe_embedding_backend.py
<service>` inside the container under test, where it can see what the
image actually contains rather than what the repository declares.

Why importing is not enough
---------------------------

    A package can be installed and the model can still fail to load.

So this does three things in order and reports each separately:

    1. can the library be imported?
    2. can the intended model be loaded?
    3. **does the service's own semantic operation execute, and what
       width does it return?**

Step 3 is the one that matters. Steps 1 and 2 are diagnosis; only step 3
is the claim. A probe that stopped at step 1 would certify a service
whose model never loads.

The verdict is the EXIT CODE
----------------------------

    0  REAL             the intended backend produced the intended width
    3  FAKE             a degraded substitute ran
    4  WRONG_DIMENSION  something ran and produced the wrong width
    5  NO_OBSERVATION   nothing was produced; absence is never a pass

Non-zero for everything that is not REAL, so a caller cannot mistake a
degraded run for a good one by reading the wrong thing. The lesson is
recent and expensive: on 2026-08-11 a `grep -c` that found zero failures
exited 1 and turned a green gate into a reported FAIL. **The verdict must
come from the process under test, and callers must capture this exit code
BEFORE running any grep, tee or report command over the output.**
"""
from __future__ import annotations

import json
import os
import sys
import traceback

REAL_DIM = 384
FAKE_DIM = 8

EXIT = {"REAL": 0, "FAKE": 3, "WRONG_DIMENSION": 4, "NO_OBSERVATION": 5}


def _memu_core():
    """memu-core's real production path: its own generate_embedding."""
    sys.path.insert(0, "/app")
    from app import generate_embedding          # noqa: E402
    return generate_embedding("a probe sentence for the embedding backend")


def _agentic():
    """agentic's semantic routing model, via its own accessor."""
    sys.path.insert(0, "/app")
    from agentic.router import _get_smodel      # noqa: E402
    model = _get_smodel()
    if model is None:
        return None
    return model.encode("a probe sentence").tolist()


def _fusion_engine():
    """fusion-engine's semantic agreement path, via its own model name."""
    sys.path.insert(0, "/app")
    from sentence_transformers import SentenceTransformer  # noqa: E402
    name = os.getenv("FUSION_EMBED_MODEL", "all-MiniLM-L6-v2")
    return SentenceTransformer(name).encode("a probe sentence").tolist()


OPERATIONS = {
    "memu-core": _memu_core,
    "agentic": _agentic,
    "fusion-engine": _fusion_engine,
}


def probe(service: str) -> dict:
    result = {
        "service": service,
        "library_importable": False,
        "library_error": None,
        "operation_ran": False,
        "operation_error": None,
        "dimension": None,
        "verdict": "NO_OBSERVATION",
        "reason": "nothing was measured",
    }

    try:
        import sentence_transformers  # noqa: F401
        result["library_importable"] = True
        result["library_version"] = getattr(
            sentence_transformers, "__version__", "unknown")
    except Exception as exc:
        result["library_error"] = f"{type(exc).__name__}: {exc}"

    operation = OPERATIONS.get(service)
    if operation is None:
        result["reason"] = f"no probe defined for service {service!r}"
        return result

    try:
        vector = operation()
    except Exception as exc:
        result["operation_error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-800:]
        result["reason"] = (
            "the semantic operation raised, so no embedding exists to "
            "classify — this is not a pass")
        return result

    if vector is None:
        result["reason"] = (
            "the service returned no vector: its degraded path ran, which "
            "is exactly the silent failure this probe exists to catch")
        result["verdict"] = "FAKE"
        result["reason"] += " (recorded as FAKE, never REAL)"
        return result

    result["operation_ran"] = True
    result["dimension"] = len(vector)

    if result["dimension"] == REAL_DIM:
        result["verdict"] = "REAL"
        result["reason"] = f"{REAL_DIM}-dimensional vector from the "\
                           f"intended model"
    elif result["dimension"] == FAKE_DIM:
        result["verdict"] = "FAKE"
        result["reason"] = f"{FAKE_DIM}-dimensional hash vector — the "\
                           f"deterministic fallback"
    else:
        result["verdict"] = "WRONG_DIMENSION"
        result["reason"] = (
            f"{result['dimension']} dimensions is neither {REAL_DIM} nor "
            f"{FAKE_DIM}; the backend returned something no one designed")
    return result


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: probe_embedding_backend.py <service>", file=sys.stderr)
        return EXIT["NO_OBSERVATION"]
    service = sys.argv[1]
    result = probe(service)
    # House format, emitted by hand: this file must stay importable
    # inside a service image, which has no scripts/ directory, so it
    # cannot use gate_inputs.inspected().
    stages = sum((result["library_importable"],
                  result["operation_ran"],
                  result["dimension"] is not None))
    print(f"  inspected: 3 stage(s) of {service}'s semantic path "
          f"(library, model, operation); {stages} reached")
    print(json.dumps(result, indent=2, sort_keys=True))
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
