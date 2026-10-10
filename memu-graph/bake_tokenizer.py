#!/usr/bin/env python3
"""Bake and verify memu-graph's tokenizer asset. Build-time only.

KAI-GATE-048 obligation 1. `memu-graph` runs on `agent-net`
(`internal: true`) and cognee's chunker builds a HuggingFace tokenizer
in-process on the first `/graph/*` request. Measured deployed: two
5-attempt retry sequences against huggingface.co, ~47s, HTTP 502.

Two subcommands, both used by `memu-graph/Dockerfile`:

    fetch    obtain the asset by the SAME ref the runtime resolves, then
             assert the resolved revision equals the pinned one
    verify   load it again with external access disabled, from the cache
             just written

A file rather than an inline `python -c`, because the Dockerfile's
`RUN` string is subject to Docker's own variable substitution before any
shell sees it — `memu-core`'s inline equivalent prints "attempt /5" for
exactly that reason. It also makes both halves testable without a
daemon.

THE REVISION IS ASSERTED, NOT REQUESTED
=======================================

cognee calls `AutoTokenizer.from_pretrained(model)` with no `revision=`
anywhere in the package, so at runtime it resolves `refs/main`. Fetching
here with an explicit revision would populate `snapshots/<sha>` without
necessarily writing the `refs/main` the runtime then reads — and the
offline load would fail inside an image whose build had looked correct.
So: fetch by `main`, exactly as the runtime does, then CHECK the sha.
Upstream moving fails the build rather than silently changing what
ships.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _name() -> str:
    """The tokenizer to bake. Read from the environment the Dockerfile
    sets, so this file never becomes a second place the name lives."""
    name = os.environ.get("MEMU_GRAPH_TOKENIZER", "").strip()
    if not name:
        sys.exit("REFUSING: MEMU_GRAPH_TOKENIZER is unset. This script "
                 "cannot guess which asset the image is supposed to carry.")
    return name


def _cache_root() -> Path:
    root = os.environ.get("HF_HOME", "").strip()
    if not root:
        sys.exit("REFUSING: HF_HOME is unset, so the asset would land "
                 "somewhere the runtime does not read.")
    return Path(root)


def _ref_path(name: str) -> Path:
    return _cache_root() / "hub" / f"models--{name.replace('/', '--')}" / "refs" / "main"


def fetch() -> int:
    name = _name()
    pinned = os.environ.get("MEMU_GRAPH_TOKENIZER_REVISION", "").strip()
    if not pinned:
        sys.exit("REFUSING: MEMU_GRAPH_TOKENIZER_REVISION is unset. An "
                 "unpinned bake ships whatever upstream happens to be.")

    import transformers
    import huggingface_hub
    from transformers import AutoTokenizer

    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(name)
    elapsed = time.monotonic() - started

    ref = _ref_path(name)
    if not ref.exists():
        sys.exit(f"REFUSING: the loader wrote no {ref}. The runtime "
                 f"resolves `main`, so an asset without that ref is an "
                 f"asset the running container cannot find.")
    resolved = ref.read_text(encoding="utf-8").strip()

    print(f"BAKED {name} revision={resolved} in {elapsed:.2f}s "
          f"class={type(tok).__name__}")
    # Recorded in the build log on purpose: the asset contract was
    # measured against specific versions and `transformers>=4.40.0` is
    # unbounded across a major version. That is separate reproducibility
    # debt (D190) and is deliberately NOT fixed here, but the evidence
    # must say what it was built against.
    print(f"VERSIONS transformers={transformers.__version__} "
          f"huggingface_hub={huggingface_hub.__version__}")

    if resolved != pinned:
        sys.exit(f"REFUSING: resolved revision {resolved} != pinned "
                 f"{pinned}. Upstream `main` has moved. Re-measure the "
                 f"asset contract and update the pin deliberately rather "
                 f"than shipping a different model than the one proven.")
    print(f"REVISION MATCHES PIN {pinned}")
    return 0


def verify() -> int:
    """Load from the cache with external access disabled.

    Called by the Dockerfile with HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
    set inline. Refuses if those are not actually in force — otherwise
    this step could pass by silently re-downloading, which is the exact
    thing it exists to rule out.
    """
    name = _name()
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if os.environ.get(var) != "1":
            sys.exit(f"REFUSING: {var} is not 1, so this verification "
                     f"could pass by fetching. It would prove nothing.")

    from huggingface_hub import constants
    if not constants.HF_HUB_OFFLINE:
        sys.exit("REFUSING: huggingface_hub.constants.HF_HUB_OFFLINE is "
                 "False despite the env var. The switch is not reaching "
                 "the library, so offline was not actually in force.")

    from transformers import AutoTokenizer
    started = time.monotonic()
    tok = AutoTokenizer.from_pretrained(name)
    elapsed = time.monotonic() - started
    print(f"OFFLINE-VERIFIED {name} in {elapsed:.2f}s "
          f"class={type(tok).__name__}")
    print(f"CACHE ROOT {_cache_root()}  ref={_ref_path(name).read_text().strip()}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("fetch", "verify"):
        print("usage: bake_tokenizer.py {fetch|verify}", file=sys.stderr)
        return 2
    return fetch() if argv[1] == "fetch" else verify()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
