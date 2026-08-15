#!/usr/bin/env python3
"""Is the EXACT model the replay will ask for actually available?

Attempt 2 (run 31906667051) sent ten requests 0.49 seconds after the
model-pull container started, and got ten HTTP 404s in about a
millisecond each. Every prior gate had passed. The readiness probe
waited on `--services ollama`, which proves a SERVER answers — a
different claim from *this model is loaded*.

WHY THIS IS A SEPARATE PROBE
============================

The obvious fix is to have the sender report what a 404 meant. But
`send_once` is inside the model-facing invocation surface, and editing
it breaks the byte-identity chain that makes attempts comparable. So
availability is established BEFORE the sender runs, by something that
is not the sender — which is also the better shape: a prerequisite
proven in advance beats a failure explained afterwards.

WHERE THE EXPECTED IDENTITY COMES FROM
======================================

From the FROZEN MANIFEST — `runtime.model`, the value the replay will
actually put on the wire. Not from a literal typed in here, and not
from the environment variable, because those are the things that can
disagree with it.

The env value IS checked, against the manifest: `OLLAMA_MODEL` is what
`ollama-pull` pulled, `runtime.model` is what will be asked for, and
if they differ the run has pulled one model and would request another.
That is a silent 404 waiting to happen, so it refuses.

WHAT COUNTS AS PRESENT
======================

An EXACT match in the server's own inventory. A prefix is not a match:
`qwen2.5` is a different tag from `qwen2.5:3b`, and treating one as the
other is the same defect as a check whose scope is wider than its name.

`/api/tags` is the gate — it is the server's inventory of what it
holds. `/v1/models` is queried too and reported as corroboration, but
it is NOT allowed to veto: refusing a working system because a second
endpoint was unavailable would be a false negative dressed as rigour.
When it cannot be read, that is said out loud rather than passed over.

Nothing here is a claim that generation will succeed. It is the
narrower, checkable claim: the server lists this exact model.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.error
import urllib.request

READY = 0
REFUSED = 7


def fetch(url: str, timeout: float) -> tuple[dict | None, str | None]:
    """(payload, failure). Never raises; the caller needs a verdict."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
    except Exception as exc:  # noqa: BLE001 — every failure is a datum
        return None, f"{type(exc).__name__}: {exc}"
    try:
        payload = json.loads(body)
    except ValueError as exc:
        return None, (f"{type(exc).__name__}: the endpoint answered "
                      f"{len(body)} byte(s) that are not JSON")
    if not isinstance(payload, dict):
        return None, f"the endpoint answered a {type(payload).__name__}, not an object"
    return payload, None


def names_in_tags(payload: dict) -> list[str]:
    """Model identities from ollama's own inventory (`/api/tags`)."""
    out: list[str] = []
    for entry in payload.get("models") or []:
        if not isinstance(entry, dict):
            continue
        for key in ("name", "model"):
            value = entry.get(key)
            if isinstance(value, str) and value not in out:
                out.append(value)
    return out


def names_in_models(payload: dict) -> list[str]:
    """Model identities from the OpenAI-compatible `/v1/models`."""
    out: list[str] = []
    for entry in payload.get("data") or []:
        if isinstance(entry, dict) and isinstance(entry.get("id"), str):
            if entry["id"] not in out:
                out.append(entry["id"])
    return out


def required_model(manifest: pathlib.Path) -> tuple[str | None, str | None]:
    """The model the replay will send, read from the frozen manifest."""
    if not manifest.is_file():
        return None, (f"{manifest} does not exist, so what the replay would "
                      f"ask for is unknown")
    try:
        man = json.loads(manifest.read_text())
    except (OSError, ValueError) as exc:
        return None, f"{manifest} could not be read: {type(exc).__name__}: {exc}"
    value = ((man.get("runtime") or {}).get("model")
             if isinstance(man.get("runtime"), dict) else None)
    if not isinstance(value, str) or not value.strip():
        return None, f"{manifest} carries no runtime.model ({value!r})"
    return value, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True,
                    help="the frozen replay manifest; runtime.model is "
                         "authoritative for what will be sent")
    ap.add_argument("--declared", default="",
                    help="the model identity the pull was configured with "
                         "(OLLAMA_MODEL); must agree with the manifest")
    ap.add_argument("--url", default="http://ollama:11434")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    print("STAGE 1 — MODEL READINESS")
    print("=" * 64)
    print(f"  server        : {base}")

    wanted, failure = required_model(pathlib.Path(args.manifest))
    if failure:
        print("  REFUSED — the required model identity is unknown")
        print(f"    {failure}")
        print("  inspected: 0 model identities across 1 server")
        return REFUSED
    print(f"  required      : {wanted!r}   (from runtime.model, the value "
          f"the replay will send)")

    if args.declared and args.declared != wanted:
        print(f"  declared      : {args.declared!r}   (OLLAMA_MODEL — what "
              f"the pull was told to fetch)")
        print("  REFUSED — the pull and the replay disagree about the model")
        print(f"    pulled {args.declared!r}, would request {wanted!r}. That")
        print("    is a 404 waiting to happen, and it would arrive as a")
        print("    transport error rather than as this refusal.")
        print("  inspected: 0 model identities across 1 server")
        return REFUSED

    tags, tags_failure = fetch(f"{base}/api/tags", args.timeout)
    if tags_failure:
        print(f"  REFUSED — the server's inventory could not be read")
        print(f"    GET {base}/api/tags -> {tags_failure}")
        print("    A server that cannot say what it holds has not proven it")
        print("    holds this model. Unproven is not present.")
        print("  inspected: 0 model identities across 1 server")
        return REFUSED

    available = names_in_tags(tags)
    print(f"  inventory     : {len(available)} model(s) — {available}")

    # Corroboration only. It may not veto a server that answered /api/tags.
    compat, compat_failure = fetch(f"{base}/v1/models", args.timeout)
    if compat_failure:
        print(f"  corroboration : NONE — GET {base}/v1/models -> "
              f"{compat_failure}")
        print("                  (the gate is /api/tags; this is reported, "
              "not enforced)")
    else:
        compat_names = names_in_models(compat)
        print(f"  corroboration : /v1/models lists {compat_names}"
              + ("" if wanted in compat_names else
                 f"  — and does NOT carry {wanted!r}"))

    # EXACT match. A prefix is not a match.
    if wanted not in available:
        near = [a for a in available if a.split(":")[0] == wanted.split(":")[0]]
        print(f"  REFUSED — {wanted!r} is not in the server's inventory")
        if near:
            print(f"    same family, different tag: {near}. A prefix is not a")
            print("    match; a different tag is a different model.")
        print("    The replay would send ten requests and receive ten")
        print("    transport errors, which is what happened in run")
        print("    31906667051. Nothing is sent.")
        print(f"  inspected: {len(available)} model identities across 1 server")
        return REFUSED

    print(f"  READY — {wanted!r} is listed by the server, exactly")
    print(f"  inspected: {len(available)} model identities across 1 server")
    print("  NOT PROVEN: that generation will succeed. This is the narrower")
    print("  claim that the exact model is present, which is the one that")
    print("  failed in attempt 2.")
    return READY


if __name__ == "__main__":
    sys.exit(main())
