#!/usr/bin/env python3
"""Stage 1 — replay the exact captured request, with no Instructor.

D247 froze the experiment; D255-D259 froze the subject. This runs it and
nothing else. Three modes, deliberately separate so that the machine
which chooses cannot be the machine which judges:

    --freeze    read the capture, re-select under S1, verify the frozen
                hashes, and write a REQUEST-ONLY manifest.
    --preflight open the replay output path, write to it, read it back
                and delete it — in the SAME image, as the SAME user, at
                the SAME path the replay will use. Cheap, and it runs
                before anything expensive.
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

--- where the output goes, and why that is not a detail ---------------

Attempt 1 (run 31899571806) executed **zero** of the ten calls. The
driver opened its output file inside the bind-mounted repository, the
image declares `USER app` (`memu-graph/Dockerfile:32`), and the mount is
owned by the runner user — `PermissionError` before the first request.
Nothing about the model, the request or the hypothesis was measured.

Two properties follow, and both live in code rather than in a habit:

* the output goes somewhere the container's own user owns, and the
  location is opened through ONE function (`open_output`) so that
  `--preflight` exercises the path the replay will actually take rather
  than a lookalike. The runtime user is NOT changed to obtain write
  permission; the destination is.
* `--classify` treats absent, unreadable or short replay evidence as
  `STAGE 1 UNMEASURED — REPLAY INSTRUMENT FAILURE` and says which
  prerequisite failed. Attempt 1's classify step raised
  `FileNotFoundError` instead — a crash where a verdict was owed (R11).

--- what may appear in the CI log ------------------------------------

The sealed replies are an artifact, never log lines. Two of the
classifier's explanations are built partly FROM the reply — jsonschema's
message embeds the offending value, and a schema echo is described by
its own property names — so those are withheld from the log behind their
length and digest and survive in full in the sealed classification file.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
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


def _digest(obj) -> str:
    """A stable identity over a JSON-serialisable value.

    Full sha256, not a prefix: this is the value ten invocations are
    required to reproduce exactly, and a truncated digest is a weaker
    claim than the one being made.
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"),
                   default=str).encode()).hexdigest()


def instrument_identity(path: pathlib.Path) -> dict:
    """An instrument's exact code identity, part of the measurement.

    We have learned too much about instruments to leave which one ran
    implicit: a verdict is a claim about bytes produced by a specific
    analyser, not by a name.
    """
    text = path.read_bytes()
    return {"file": str(path.relative_to(REPO)),
            "sha256": hashlib.sha256(text).hexdigest(),
            "bytes": len(text)}


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
           expect_seq: int, url: str = "", timeout: float = 300.0
           ) -> tuple[dict | None, list[str]]:
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

    # D247's held constants, frozen INTO the manifest. The hashes prove
    # identity; they do not reconstruct an invocation, so everything
    # needed to reproduce it travels with it.
    runtime = {"n": N1, "url": url, "timeout_s": timeout,
               "instructor_in_path": False, "validation": "none",
               "retry": "none",
               "model": body.get("model")}
    man = {"request": body, "subject": proj, "n": N1, "runtime": runtime,
           "checks": checks}
    # ONE immutable identity over request + constants, and a separate one
    # over the request alone, because every invocation is required to
    # reproduce the second exactly.
    man["request_hash"] = _digest(body)
    man["manifest_hash"] = _digest({"request": body, "runtime": runtime,
                                    "subject": proj})
    return man, []


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


# ── the output path, opened in exactly one place ─────────────────────
#
# `--preflight` and `--replay` must not be able to disagree about how the
# destination is opened, because the whole point of the preflight is that
# it proves the thing the replay will do. One function, both callers.
def open_output(path: pathlib.Path):
    """Open the replay output for writing. The ONLY place that happens."""
    return path.open("w")


PREFLIGHT_LINE = '{"preflight": "stage1 output path", "replay_index": 0}'


def probe_output_path(path: pathlib.Path) -> tuple[bool, str]:
    """Write, read back and delete at `path`. No model, no network.

    Read-back rather than "open() did not raise": a destination that
    accepts an open and loses the bytes is the failure this is for, and
    an open that succeeds is not evidence that a file arrived.
    """
    try:
        handle = open_output(path)
    except OSError as exc:
        return False, f"open failed: {type(exc).__name__}: {exc}"
    try:
        with handle:
            handle.write(PREFLIGHT_LINE + "\n")
            handle.flush()
        back = path.read_text()
    except OSError as exc:
        return False, f"write/read-back failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            path.unlink()
        except OSError:
            pass
    if back != PREFLIGHT_LINE + "\n":
        return False, (f"the destination accepted the write but returned "
                       f"{len(back)} byte(s), not the {len(PREFLIGHT_LINE) + 1} "
                       f"written")
    if path.exists():
        return False, ("the probe file could not be removed, so a later run "
                       "could mistake it for replay evidence")
    return True, (f"wrote and read back {len(PREFLIGHT_LINE) + 1} byte(s) at "
                  f"{path}, then removed it")


# ── refusing, rather than crashing, on absent replay evidence ────────
UNMEASURED = "STAGE 1 UNMEASURED — REPLAY INSTRUMENT FAILURE"


def load_replay_evidence(manifest_path, replies_path
                         ) -> tuple[dict | None, list, str | None]:
    """(manifest, replies, failure). Never raises on missing evidence.

    Attempt 1's classify step ran under `if: always()`, found no replies
    file and raised `FileNotFoundError`. A crash is not a verdict, and a
    step that owes a verdict must produce one: R11 — no subject, no
    observation — requires naming the unmet prerequisite and saying what
    was therefore not measured.
    """
    if not manifest_path:
        return None, [], "no --manifest was given, so nothing was frozen to replay"
    man_p = pathlib.Path(manifest_path)
    if not man_p.is_file():
        return None, [], (f"the frozen manifest {manifest_path} does not exist: "
                          f"the freeze step did not complete, so no request "
                          f"was ever frozen and none could have been sent")
    try:
        man = json.loads(man_p.read_text())
    except (OSError, ValueError) as exc:
        return None, [], (f"the frozen manifest {manifest_path} could not be "
                          f"read: {type(exc).__name__}: {exc}")

    if not replies_path:
        return man, [], "no --replies was given, so no replay output was offered"
    rep_p = pathlib.Path(replies_path)
    if not rep_p.is_file():
        return man, [], (f"the replies file {replies_path} does not exist: the "
                         f"replay step produced no output at all, so 0 of the "
                         f"precommitted {man.get('n')} executions are recorded")
    try:
        text = rep_p.read_text()
    except OSError as exc:
        return man, [], (f"the replies file {replies_path} could not be read: "
                         f"{type(exc).__name__}: {exc}")

    rows, bad = [], []
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError as exc:
            bad.append(f"line {lineno} ({len(line)} bytes): {type(exc).__name__}")
    if bad:
        return man, rows, (f"the replies file is not wholly parseable, so the "
                           f"population it describes cannot be trusted — "
                           f"{'; '.join(bad)}")
    if not rows:
        return man, rows, (f"the replies file exists but carries no execution "
                           f"rows: 0 of the precommitted {man.get('n')}")
    return man, rows, None


# ── what may be printed, and what may only be sealed ─────────────────
#
# Two of the classifier's explanations are built partly from the REPLY.
# `INSTANCE_INVALID` carries jsonschema's message, which embeds the
# offending instance value; `SCHEMA_ECHO` lists the property names the
# reply described. Anything not named here is withheld by default, so a
# verdict added upstream later fails safe rather than leaking.
RESPONSE_FREE_DETAIL = frozenset({
    classify.CONTRACT_UNMEASURED,
    classify.NO_RESPONSE,
    classify.NOT_JSON,
    classify.NOT_JSON_OBJECT,
    classify.VALID_INSTANCE,
    classify.REQUIRED_FIELDS_PRESENT,
    classify.REQUIRED_FIELDS_MISSING,
})


def loggable(verdict: str, detail: str) -> str:
    """The part of an explanation that may enter the CI log.

    A withheld detail still declares its own size and identity, because
    an excerpt that does not announce itself reads like the whole thing
    (R10). The full text survives in the sealed classification file.
    """
    if verdict in RESPONSE_FREE_DETAIL:
        return detail
    return (f"[withheld — derived from the reply body: {len(detail)} chars, "
            f"sha256 {hashlib.sha256(detail.encode()).hexdigest()[:12]}, "
            f"full text in the sealed classification artifact]")


# ── value-level identity across two attempts ─────────────────────────
#
# Code identity and YAML identity are both arguments about the
# instrument. This is the thing itself: attempt 1 got as far as
# producing a freeze manifest, so attempt 2's request hash can be
# compared to it BEFORE anything is sent. `request_hash` is computed
# over the request body alone, so reading it opens nothing about the
# original model response.
#
# `manifest_hash` is deliberately NOT compared: it covers runtime
# metadata that the repair is allowed to have moved. Equal requests
# under different instrument metadata is the intended state.
def response_bearing_keys(obj, seen=None) -> list[str]:
    """Every response-bearing key anywhere in a structure.

    The reference manifest is request-only by construction, but "by
    construction" is a claim about code that has since been edited.
    This checks the artifact in hand instead.
    """
    found = seen if seen is not None else []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in s1.RESPONSE_BEARING:
                found.append(k)
            response_bearing_keys(v, found)
    elif isinstance(obj, list):
        for item in obj:
            response_bearing_keys(item, found)
    return found


def read_request_hash(path: pathlib.Path) -> tuple[str | None, str | None]:
    """(request_hash, failure). Reads one request-side field, never more."""
    if not path.is_file():
        return None, (f"{path} does not exist, so the reference request "
                      f"identity could not be read")
    try:
        man = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return None, f"{path} could not be read: {type(exc).__name__}: {exc}"
    leaked = sorted(set(response_bearing_keys(man)))
    if leaked:
        return None, (f"{path} carries response-bearing key(s) {leaked}; it "
                      f"is not the request-only manifest it is taken for, "
                      f"and reading it further would open the response")
    value = man.get("request_hash")
    if not isinstance(value, str) or len(value) != 64:
        return None, (f"{path} carries no usable request_hash "
                      f"({value!r})")
    return value, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--verify-request-hash", action="store_true")
    mode.add_argument("--replay", action="store_true")
    mode.add_argument("--classify", action="store_true")
    ap.add_argument("--capture")
    ap.add_argument("--manifest")
    ap.add_argument("--against", help="a previous attempt's freeze manifest")
    ap.add_argument("--attempt-replies",
                    help="a previous attempt's replies file, which must NOT "
                         "exist if that attempt is recorded as 0 executions")
    ap.add_argument("--replies")
    ap.add_argument("--classification")
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
                               args.expect_contract_hash, args.expect_seq,
                               url=args.url or "", timeout=args.timeout)
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
        print(f"  request body keys : {sorted(man['request'])}")
        print(f"  messages present  : "
              f"{len(man['request'].get('messages') or [])} message(s), "
              f"{sum(len(m.get('content') or '') for m in man['request'].get('messages') or [])} chars")
        print("  frozen runtime constants (D247):")
        for k in sorted(man["runtime"]):
            print(f"      {k:<20} {man['runtime'][k]!r}")
        print(f"  request_hash      : {man['request_hash']}")
        print(f"  manifest_hash     : {man['manifest_hash']}")
        print(f"  inspected: 1 replay subject across "
              f"{len(man['checks']) + 3} identity check(s)")
        pathlib.Path(args.manifest).write_text(json.dumps(man, indent=2))
        print(f"FROZEN -> {args.manifest}")
        return 0

    if args.preflight:
        # Deliberately BEFORE the expensive stack, in the same image and
        # as the same user. It sends nothing and needs no model: this
        # asks only whether the replay's output has anywhere to land.
        path = pathlib.Path(args.replies)
        print("STAGE 1 — OUTPUT WRITE PREFLIGHT")
        print("=" * 64)
        print(f"  destination  : {path}")
        print(f"  running as   : uid {os.getuid()} gid {os.getgid()}")
        print(f"  opened by    : open_output(), the same call --replay uses")
        ok, why = probe_output_path(path)
        print(f"  {'OK ' if ok else 'FAIL'}         : {why}")
        print("  inspected: 1 replay output path across "
              "1 write/read-back/delete cycle")
        if not ok:
            print()
            print("REFUSED — the replay has nowhere to write its output.")
            print("  Attempt 1 (run 31899571806) discovered this AFTER the")
            print("  model stack was up, and executed 0 of 10 calls. Failing")
            print("  here costs a container start and measures the same thing.")
            return 6
        return 0

    if args.verify_request_hash:
        print("STAGE 1 — REQUEST IDENTITY ACROSS ATTEMPTS")
        print("=" * 64)
        mine, why_mine = read_request_hash(pathlib.Path(args.manifest))
        theirs, why_theirs = read_request_hash(pathlib.Path(args.against))
        print(f"  this attempt   : {args.manifest}")
        print(f"  reference      : {args.against}")
        # A previous attempt recorded as ZERO executions cannot have left
        # a replies file. If one arrived with the reference artifact, our
        # record of that attempt is wrong and this one must not proceed
        # on top of it.
        if args.attempt_replies and pathlib.Path(args.attempt_replies).exists():
            print(f"  REFUSED: {args.attempt_replies} exists. The reference "
                  f"attempt is recorded as having executed nothing; a "
                  f"replies file contradicts that record, and this attempt "
                  f"must not be built on a record known to be wrong.")
            return 4
        for label, failure in (("this attempt", why_mine),
                               ("reference", why_theirs)):
            if failure:
                print(f"  {UNMEASURED}")
                print(f"  unmet prerequisite: {label} — {failure}")
                print()
                print("  NOT MEASURED: whether the two attempts send the same")
                print("  request. Equality was not established, and an")
                print("  unestablished equality is not a match.")
                return 4
        print(f"  reference hash : {theirs}")
        print(f"  this attempt   : {mine}")
        print("  inspected: 2 freeze manifest(s) across 1 request identity")
        if mine != theirs:
            print()
            print("STAGE 1 INVALID — the request identity changed between "
                  "attempts")
            print("  The instrument was repaired; the request was not "
                  "supposed to move.")
            print("  Do not explain this away mid-execution. Nothing is sent.")
            return 5
        print()
        print("  IDENTICAL — attempt 2 sends byte-for-byte the request "
              "attempt 1 froze.")
        print("  manifest_hash is deliberately NOT compared: it covers "
              "instrument")
        print("  metadata the repair was authorised to move.")
        return 0

    if args.replay:
        man = json.loads(pathlib.Path(args.manifest).read_text())
        body = man["request"]
        out = open_output(pathlib.Path(args.replies))
        for i in range(1, man["n"] + 1):
            rec = send_once(man["runtime"]["url"], body,
                            man["runtime"]["timeout_s"])
            rec["replay_index"] = i
            # The identity of what was ACTUALLY sent, per invocation.
            # Ten calls from one frozen manifest should all reproduce the
            # frozen request hash; if one does not, the experiment is
            # invalid rather than nine-tenths usable.
            rec["request_hash"] = _digest(body)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            # Progress to stderr: stdout stays a machine stream (D250).
            print(f"  replay {i}/{man['n']} done", file=sys.stderr, flush=True)
        out.close()
        return 0

    # --classify
    man, replies, failure = load_replay_evidence(args.manifest, args.replies)

    print("STAGE 1 — REPLAY RESULTS")
    print("=" * 64)
    if man is None:
        # No manifest means no subject, and R11 forbids reporting rows
        # against a subject that was never established.
        print(f"  {UNMEASURED}")
        print(f"  unmet prerequisite: {failure}")
        print()
        print("  NOT MEASURED: the Stage-1 question. No replay execution is")
        print("  recorded, so this is a fact about our instrument and not")
        print("  about the model, the request or D247's hypothesis.")
        return 4

    schema, why, _ = classify.recover_contract(man["request"].get("messages"))
    validator = classify.validator_status()

    print(f"  subject      : seq {man['subject']['seq']}, prompt "
          f"{man['subject']['prompt_hash']}, contract "
          f"{man['subject']['contract_hash']}")
    print(f"  contract     : {why}")
    print(f"  validator    : {validator}")
    print(f"  manifest     : {man['manifest_hash']}")
    print(f"  request id   : {man['request_hash']}")
    print(f"  runtime      : {man['runtime']}")
    print(f"  precommitted : N1 = {man['n']}, sequential, no Instructor, "
          f"transport error recorded and NOT replaced")
    print(f"  classifier   : {instrument_identity(REPO / 'scripts' / 'security' / 'classify_llm_response.py')}")
    print()
    if failure is not None:
        # The freeze succeeded, so a subject exists; the replay did not,
        # so there is nothing to say about it. Reporting the rows that
        # DID parse across a record known to be broken is the D250
        # defect — an instrument announcing its own invalidity and then
        # reporting across it.
        print(f"  {UNMEASURED}")
        print(f"  unmet prerequisite: {failure}")
        print(f"  parseable rows recovered: {len(replies)} of the "
              f"precommitted {man['n']}")
        print()
        print("  NOT MEASURED: what the model returns to this exact request.")
        print("  The frozen request identity above is intact and re-usable;")
        print("  what failed is the recording of the executions, which is")
        print("  ours. It is not evidence about the model or about D247.")
        return 4
    # EVERY invocation must reproduce the frozen request identity. One
    # that does not makes the experiment INVALID, not nine-tenths usable
    # -- a set of ten in which one call sent something else is not a
    # sample of ten replays of the same request.
    mismatched = [r for r in replies
                  if r.get("request_hash") != man["request_hash"]]
    if mismatched:
        print("STAGE 1 INVALID — request identity was not constant")
        for r in mismatched:
            print(f"    replay {r.get('replay_index')}: sent "
                  f"{r.get('request_hash')}, frozen {man['request_hash']}")
        print()
        print("  This is not a 9/10 result. Ten replays of one request is")
        print("  the experiment; a set in which one call sent something")
        print("  else measures something nobody designed.")
        return 5
    print(f"  request identity: all {len(replies)} invocation(s) reproduce "
          f"the frozen request hash")
    print()
    counts: dict[str, int] = {}
    sealed: list[dict] = []
    withheld = 0
    for rec in replies:
        i = rec.get("replay_index")
        if rec.get("raw_response") is None:
            verdict = classify.NO_RESPONSE
            detail = rec.get("transport_error") or rec.get("note") or ""
        else:
            verdict, detail = classify.classify(
                rec["raw_response"], schema, validator)
        counts[verdict] = counts.get(verdict, 0) + 1
        shown = loggable(verdict, detail)
        withheld += (shown != detail)
        sealed.append({"replay_index": i, "verdict": verdict, "why": detail,
                       "elapsed_s": rec.get("elapsed_s"),
                       "request_hash": rec.get("request_hash")})
        print(f"  {i:>2}. {verdict:<26} {rec.get('elapsed_s')}s  {shown}")
    if args.classification:
        pathlib.Path(args.classification).write_text(
            "".join(json.dumps(s) + "\n" for s in sealed))
        print(f"  sealed classification -> {args.classification} "
              f"({len(sealed)} row(s), {withheld} explanation(s) withheld "
              f"from this log)")
    elif withheld:
        print(f"  {withheld} explanation(s) withheld from this log and NOT "
              f"sealed anywhere: --classification was not given")
    print()
    print(f"  inspected: {len(replies)} replay execution(s) of {man['n']} "
          f"precommitted")
    for v in sorted(counts):
        print(f"    {v:<26} {counts[v]}")
    if len(replies) != man["n"]:
        print(f"  DENOMINATOR MISMATCH: {len(replies)} replies for "
              f"{man['n']} precommitted executions. The denominator stays "
              f"{man['n']}; the shortfall is a fact, not a smaller sample.")
        print(f"  {UNMEASURED}")
        print("  unmet prerequisite: the replay did not run to its "
              "precommitted denominator")
        return 4
    # THE FALSE GREEN (D265). Attempt 2 produced ten rows, all with
    # matching request hashes and a count equal to N1, and every rule
    # this classifier enforced was satisfied — so it exited 0 over an
    # experiment in which nothing reached a model. Each row was
    # truthful; the aggregate was not.
    #
    # Deliberately narrow: ONLY the proven hole. D247 specifies no
    # threshold for partial populations, so none is invented here — a
    # run with even one model response is reported as it always was.
    responded = [r for r in replies if r.get("raw_response") is not None]
    if not responded:
        print()
        print(f"  {UNMEASURED}")
        print(f"  unmet prerequisite: {len(replies)} execution(s) were "
              f"attempted and NONE reached a model — 0 model response(s)")
        print()
        print("  Every row above is individually true. The set is not a")
        print("  Stage-1 result: D247 asks what the model returns to this")
        print("  request, and no request was answered by one. A complete")
        print("  population of transport failures is a fact about the")
        print("  infrastructure, not a measurement of the model.")
        return 4
    print()
    print(f"  model responses: {len(responded)} of {len(replies)} execution(s) "
          f"reached a model")
    print("  These are REPLAY outcomes. The original captured response has")
    print("  NOT been opened, and comparing them to it is a separate,")
    print("  separately authorised step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
