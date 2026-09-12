#!/usr/bin/env python3
"""The embedding-backend denominator, and its detector's calibration.

Two things under test:

1. the population is DERIVED — every non-test module importing
   `sentence_transformers`, not a list of services someone remembered;
2. the detector cannot be talked into saying REAL.

The second is the load-bearing one. Several distinct failures all leave a
service that starts, serves and reports healthy, so a detector that can
be fooled by any of them would certify a system with no semantic
capability at all. Each of those failures gets a known-negative here, and
the real backend gets a known-positive, per I-8.

Note what is deliberately NOT asserted: that agentic or fusion-engine
lack the library at runtime. Dependency arithmetic is not a built
container. Those rows stay DECLARATION DEFECT until a probe inside the
image says otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.report_embedding_backends import (  # noqa: E402
    FAKE, FAKE_DIM, NO_OBSERVATION, REAL, REAL_DIM, WRONG_DIMENSION,
    audit, classify_signature, importers)

PASSED = 0
FAILED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok    {label}")
    else:
        FAILED += 1
        print(f"  FAIL  {label}" + (f"\n        {detail}" if detail else ""))


REAL_LOG = ("sentence-transformers loaded — model='all-MiniLM-L6-v2'  "
            "dim=384  embedding backend ready in 3.2s")
FAKE_LOG = ("sentence-transformers not available — using hash-based fake "
            "embeddings (MEMU_ALLOW_FAKE_EMBEDDINGS=true)")


def main() -> int:
    # ── the KNOWN-POSITIVE ──
    verdict, reason = classify_signature(REAL_DIM, REAL_LOG)
    check("KNOWN-POSITIVE: 384 dimensions with the real log is REAL",
          verdict == REAL, reason)

    # ── the six known-negatives the operator named ──

    # 1. fake embeddings explicitly enabled
    verdict, reason = classify_signature(FAKE_DIM, FAKE_LOG)
    check("explicit fake embeddings -> FAKE, never REAL",
          verdict == FAKE and verdict != REAL, reason)

    # 2. library absent — the service degraded and said so
    verdict, _ = classify_signature(FAKE_DIM, "ImportError: no module named "
                                              "sentence_transformers")
    check("library absent -> FAKE, never REAL", verdict == FAKE)

    # 3. baked model absent or corrupt — nothing was produced
    verdict, reason = classify_signature(None, "RuntimeError: Embedding "
                                               "backend unavailable")
    check("model absent, no vector produced -> NO_OBSERVATION, never REAL",
          verdict == NO_OBSERVATION and verdict != REAL, reason)

    # 4. model cache path wrong — same shape, still not a pass
    verdict, _ = classify_signature(None, "OSError: /opt/hf_cache not found")
    check("wrong cache path -> NO_OBSERVATION, never REAL",
          verdict == NO_OBSERVATION)

    # 5. backend reports success but returns the wrong width.
    #    This is the one that would pass every naive check: the log says
    #    it worked, the service is healthy, and the vectors are wrong.
    for wrong in (0, 1, 7, 9, 128, 383, 385, 768):
        verdict, reason = classify_signature(wrong, REAL_LOG)
        check(f"a {wrong}-dimensional vector is refused, despite a "
              f"success log", verdict == WRONG_DIMENSION, reason)

    # 6. silent degradation where the semantic backend was expected: the
    #    service claims the real backend and returns the fallback width.
    verdict, reason = classify_signature(FAKE_DIM, REAL_LOG)
    check("a FAKE-width vector under a REAL log is not REAL",
          verdict == FAKE and verdict != REAL, reason)

    # ...and the inverse disagreement, which is equally a defect.
    verdict, reason = classify_signature(REAL_DIM, FAKE_LOG)
    check("a REAL-width vector under a FAKE log is reported as a "
          "disagreement, not quietly accepted",
          verdict == WRONG_DIMENSION, reason)

    # ── absence is never success ──
    verdict, reason = classify_signature(None, "")
    check("NOTHING MEASURED IS NEVER A PASS",
          verdict == NO_OBSERVATION and verdict != REAL, reason)
    check("and it says so plainly", "nothing was measured" in reason)

    # A log alone cannot buy REAL: a claim is not a measurement.
    verdict, reason = classify_signature(None, REAL_LOG)
    check("a SUCCESS LOG WITH NO VECTOR is not REAL",
          verdict != REAL and "not a measurement" in reason, reason)

    # ── the population is derived from the tree ──
    found = importers(REPO)
    services = sorted({s for s, _ in found})
    check("the population is derived, and finds all three consumers",
          services == ["agentic", "fusion-engine", "memu-core"], str(services))
    check("test files and conftest are not counted as consumers",
          all("test_" not in path and "conftest" not in path
              for _, path in found), str(found))

    rows, n, counts = audit(REPO)
    check("CALIBRATION: 3 services inspected", n == 3)
    check("CALIBRATION: 2 degrade SILENTLY", counts.get("_silent") == 2)
    check("CALIBRATION: nothing is PROVEN — proof needs a built image",
          counts.get("PROVEN", 0) == 0)

    by_service = {r.service: r for r in rows}

    # memu-core: declared, baked, explicit refusal, still unproven.
    memu = by_service["memu-core"]
    check("memu-core declares the library", "sentence-transformers" in
          memu.declared_in)
    check("memu-core bakes the model at build time",
          "baked at build time" in memu.model_source)
    check("memu-core REFUSES rather than degrading",
          memu.silent is False and "raises RuntimeError" in
          memu.on_library_missing)
    check("memu-core distinguishes a missing MODEL from a missing library",
          "MODEL" in memu.on_model_missing)
    check("memu-core is UNKNOWN — a baked model is not a loaded model",
          memu.classification == "UNKNOWN")
    check("and no repo-defined CI path runs its production default",
          memu.ci_executes_production_default is False)

    # agentic and fusion-engine: declaration defect, NOT runtime FAIL.
    for name in ("agentic", "fusion-engine"):
        row = by_service[name]
        check(f"{name} does not declare the library",
              row.declared_in == "ABSENT")
        check(f"{name} degrades SILENTLY", row.silent is True)
        check(f"{name} is DECLARATION DEFECT, NOT runtime FAIL — "
              f"dependency arithmetic is not a built container",
              row.classification == "DECLARATION DEFECT")
        check(f"{name} records that runtime state is unproven",
              any("NOT YET PROVEN" in note for note in row.notes))

    # ── I-1: an empty tree is a broken scan ──
    import tempfile
    empty = Path(tempfile.mkdtemp())
    rows2, n2, _ = audit(empty)
    check("an empty tree yields no rows rather than a clean bill",
          n2 == 0 and rows2 == [])

    print("=" * 66)
    print(f"Embedding backend tests: {PASSED} passed, {FAILED} failed")
    print(f"EXIT GATE: {'PASS' if FAILED == 0 else 'FAIL'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    print("Embedding backends — denominator and detector calibration")
    print("=" * 66)
    sys.exit(main())
