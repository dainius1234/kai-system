"""Where this process keeps its state — one answer, read at call time.

Every persistent store in `agentic/` spells its own directory as a module
constant: `Path("data/trust")`, `Path("data/ohana/fingerprint.json")`,
`Path("data/wisdom")`, `Path("data/paper-trading")`. Relative, so they
resolve against the working directory — and pytest's working directory is
the repository root.

The consequence was found twice before it was understood once.

  1. `data/SOUL.md`. A test exercising the soul endpoint rewrote the
     tracked document, and the tests asserting on its contents then
     depended on whether they ran before or after it. `conftest.py`
     redirects `SOUL_PATH` for exactly this reason, and its comment
     records that `git checkout -- data/` "had become a reflex between
     local runs" — which is how a defect stays invisible: someone keeps
     paying for it by hand.

  2. 2026-08-05. A commit meant to touch six files carried a seventh:
     two signed, hash-chained AUTONOMOUS_ACTION events that
     `make test-uh` had just appended to the repository's trust ledger.
     Measured rather than guessed at — a full `pytest scripts/` run
     (4,324 passing) mutates exactly four tracked files:

         data/ohana/fingerprint.json
         data/trust-ledger/events.jsonl
         data/trust/audit_log.jsonl
         data/trust/trust_record.json

The first fix was per-path, and a per-path fix is a list beside the
thing: it is true of what someone remembered on the day. This is the
same fix made general. One variable moves every adopting store at once,
and `conftest.py` sets it to a scratch directory for the whole test
session, so a suite cannot write into the repository whether or not
anybody remembered to redirect it.

`data_root()` is read at **call time**, not captured at import. A module
constant computed during import cannot be redirected by a test that
imports the module, which is the shape that made the originals
untestable in the first place.
"""
from __future__ import annotations

import os
from pathlib import Path

#: The variable. Named once here so no caller has to spell it.
ENV_VAR = "KAI_DATA_ROOT"

#: What the repository has always used when nothing says otherwise.
DEFAULT_ROOT = "data"


def data_root() -> Path:
    """The directory this process keeps persistent state under."""
    return Path(os.getenv(ENV_VAR) or DEFAULT_ROOT)


def data_path(*parts: str) -> Path:
    """A path beneath the data root: `data_path("trust", "audit_log.jsonl")`."""
    return data_root().joinpath(*parts)
