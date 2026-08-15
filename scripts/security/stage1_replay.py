#!/usr/bin/env python3
"""Stage 1 — replay the exact captured request, with no Instructor.

D247 froze the experiment; D255-D259 froze the subject. This runs it and
nothing else. Three modes, deliberately separate so that the machine
which chooses cannot be the machine which judges:

    --freeze    read the capture, re-select under S1, verify the frozen
                hashes, and write a REQUEST-ONLY manifest.
    --replay    read that manifest, POST it N times, emit raw replies.
                Runs where the endpoint is reachable; knows nothing
                about verdicts.
    --classify  read those replies, classify each with the SHIPPED
                classifier, and report every one in order.

--- what this file may not do -----------------------------------------

**It never reads the selected row's response.** Not to check the replay
against it, not to report it, not incidentally. `--freeze` builds its
manifest from the allow-listed request-side projection, so the original
`raw_response` is not in scope at any point. The operator's constraint —
*do not open the original selected model response before the ten replay
executions are complete and sealed* — is therefore structural here
rather than a rule someone must remember.

**It changes nothing to obtain a nicer result.** No model, timeout,
schema or request parameter is altered. The request is rebuilt from what
was recorded, or the run refuses.

--- reconstruction, and why it needs care -----------------------------

D259 found that `response_format` is stored as a Python **repr** —
`"{'type': 'json_object'}"`, single quotes — because the probe's
`_serialise()` falls back to `str(obj)`. `json.loads` fails on it.

It is rebuilt with `ast.literal_eval`, **never `eval`**: `literal_eval`
parses literals only and cannot execute anything, which matters because
this string came out of a capture file rather than out of our own code.
The result is then asserted to be exactly the intended typed value —
a `dict` equal to `{"type": "json_object"}` — because "it parsed" is not
"it is what was sent".
"""

from __future__ import annotations

import argparse
import ast
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import classify_llm_response as classify  # noqa: E402
import p1_replay_completeness as p1  # noqa: E402
import select_replay_subject as s1  # noqa: E402

# D247's frozen parameter. A budget cap, not statistical power (D243).
N1 = 10

# The typed value `response_format` must reconstruct to. Asserted, not
# assumed: a repr that parses into something else is a different request.
EXPECTED_RESPONSE_FORMAT = {"type": "json_object"}

# Keys the manifest may carry to the sender. Response-bearing keys are
# absent by construction — the manifest is built from S1's projection.
SENDABLE = ("model", "messages", "temperature", "response_format", "tools")


def rebuild(value):
    """A recorded argument, back to its typed form.

    `ast.literal_eval` and never `eval`: this string came from a capture
    file, and `literal_eval` cannot execute code even if that file were
    hostile. A value that is already typed passes through.
    """
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def freeze(capture: pathlib.Path, expect_prompt: str, expect_contract: str,
           expect_seq: int) -> tuple[dict | None, list[str]]:
    """Re-select under S1 and verify the subject is the frozen one."""
    problems: list[str] = []
    rows, notes, manifest = p1.read_rows(capture)
    for n in notes:
        problems.append(f"capture note: {n}")
    row, checks, refusals = s1.select(rows)
    if row is None:
        return None, problems + refusals

    proj = s1.projection(row)
    if proj.get("seq") != expect_seq:
        problems.append(
            f"IDENTITY FAILED: re-selected seq {proj.get('seq')}, frozen "
            f"subject was seq {expect_seq}")
    if proj.get("prompt_hash") != expect_prompt:
        problems.append(
            f"IDENTITY FAILED: prompt hash {proj.get('prompt_hash')} != "
            f"frozen {expect_prompt}")
    if proj.get("contract_hash") != expect_contract:
        problems.append(
            f"IDENTITY FAILED: contract hash {proj.get('contract_hash')} != "
            f"frozen {expect_contract}")
    if problems:
        return None, problems

    body: dict = {}
    for key in SENDABLE:
        state = (row.get("args_state") or {}).get(key)
        if state == "ABSENT":
            continue          # it was not sent; sending it would differ
        body[key] = rebuild(row.get(key))

    rf = body.get("response_format")
    if rf is not None and rf != EXPECTED_RESPONSE_FORMAT:
        problems.append(
            f"RECONSTRUCTION FAILED: response_format rebuilt to {rf!r} "
            f"({type(rf).__name__}), expected exactly "
            f"{EXPECTED_RESPONSE_FORMAT!r}")
        return None, problems
    if rf is not None and not isinstance(rf, dict):
        problems.append("RECONSTRUCTION FAILED: response_format is not a dict")
        return None, problems

    # Response-bearing keys must be absent from what we are about to send.
    leaked = [k for k in s1.RESPONSE_BEARING if k in body]
    if leaked:
        problems.append(
            f"REFUSED: the manifest would carry response-bearing key(s) "
            f"{leaked}")
        return None, problems

    return {"request": body, "subject": proj, "n": N1, "checks": checks}, []


def send_once(url: str, body: dict, timeout: float) -> dict:
    """One raw POST. No Instructor, no validation, no retry."""
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json",
                                 "Authorization": "Bearer ollama"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode(errors="replace"))
        elapsed = round(time.monotonic() - started, 3)
        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return {"raw_response": None, "elapsed_s": elapsed,
                    "note": "unexpected completion shape"}
        return {"raw_response": text, "elapsed_s": elapsed,
                "finish_reason": payload["choices"][0].get("finish_reason")}
    except Exception as exc:  # noqa: BLE001 — every failure is a datum
        return {"raw_response": None,
                "elapsed_s": round(time.monotonic() - started, 3),
                "transport_error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--classify", action="store_true")
    ap.add_argument("--capture")
    ap.add_argument("--manifest")
    ap.add_argument("--replies")
    ap.add_argument("--url")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--expect-seq", type=int)
    ap.add_argument("--expect-prompt-hash")
    ap.add_argument("--expect-contract-hash")
    args = ap.parse_args()

    if args.freeze:
        cap = pathlib.Path(args.capture)
        if not cap.is_file():
            print("STAGE 1 NOT STARTED: the capture is absent. This is an "
                  "availability failure, not a replay result.")
            return 2
        man, problems = freeze(cap, args.expect_prompt_hash,
                               args.expect_contract_hash, args.expect_seq)
        print("STAGE 1 — FREEZE THE EXACT REQUEST")
        print("=" * 64)
        if man is None:
            print("REFUSED — exact-request identity is not established:")
            for p in problems:
                print(f"  - {p}")
            print()
            print("  Stage 1 does NOT run. A replay of a request we cannot")
            print("  prove is the frozen one answers a different question.")
            return 3
        for c in man["checks"]:
            print(f"  OK  {c}")
        print(f"  OK  subject identity: seq {man['subject']['seq']}, "
              f"prompt {man['subject']['prompt_hash']}, "
              f"contract {man['subject']['contract_hash']} — all match frozen")
        rf = man["request"].get("response_format")
        print(f"  OK  response_format rebuilt via ast.literal_eval to "
              f"{rf!r} ({type(rf).__name__}), exactly the intended value")
        print(f"  request keys sent: {sorted(man['request'])}")
        print(f"  keys recorded ABSENT and therefore NOT sent: "
              f"{[k for k in SENDABLE if k not in man['request']]}")
        print(f"  inspected: 1 replay subject across "
              f"{len(man['checks']) + 3} identity check(s)")
        pathlib.Path(args.manifest).write_text(json.dumps(man, indent=2))
        print(f"FROZEN -> {args.manifest}")
        return 0

    if args.replay:
        man = json.loads(pathlib.Path(args.manifest).read_text())
        body = man["request"]
        out = pathlib.Path(args.replies).open("w")
        for i in range(1, man["n"] + 1):
            rec = send_once(args.url, body, args.timeout)
            rec["replay_index"] = i
            out.write(json.dumps(rec) + "\n")
            out.flush()
            # Progress to stderr: stdout stays a machine stream (D250).
            print(f"  replay {i}/{man['n']} done", file=sys.stderr, flush=True)
        out.close()
        return 0

    # --classify
    man = json.loads(pathlib.Path(args.manifest).read_text())
    replies = [json.loads(l) for l in
               pathlib.Path(args.replies).read_text().splitlines() if l.strip()]
    schema, why, _ = classify.recover_contract(man["request"].get("messages"))
    validator = classify.validator_status()

    print("STAGE 1 — REPLAY RESULTS")
    print("=" * 64)
    print(f"  subject      : seq {man['subject']['seq']}, prompt "
          f"{man['subject']['prompt_hash']}, contract "
          f"{man['subject']['contract_hash']}")
    print(f"  contract     : {why}")
    print(f"  validator    : {validator}")
    print(f"  precommitted : N1 = {man['n']}, sequential, no Instructor, "
          f"transport error recorded and NOT replaced")
    print()
    counts: dict[str, int] = {}
    for rec in replies:
        i = rec.get("replay_index")
        if rec.get("raw_response") is None:
            verdict = classify.NO_RESPONSE
            detail = rec.get("transport_error") or rec.get("note") or ""
        else:
            verdict, detail = classify.classify(
                rec["raw_response"], schema, validator)
        counts[verdict] = counts.get(verdict, 0) + 1
        print(f"  {i:>2}. {verdict:<26} {rec.get('elapsed_s')}s  {detail[:60]}")
    print()
    print(f"  inspected: {len(replies)} replay execution(s) of {man['n']} "
          f"precommitted")
    for v in sorted(counts):
        print(f"    {v:<26} {counts[v]}")
    if len(replies) != man["n"]:
        print(f"  DENOMINATOR MISMATCH: {len(replies)} replies for "
              f"{man['n']} precommitted executions. The denominator stays "
              f"{man['n']}; the shortfall is a fact, not a smaller sample.")
        return 4
    print()
    print("  These are REPLAY outcomes. The original captured response has")
    print("  NOT been opened, and comparing them to it is a separate,")
    print("  separately authorised step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
