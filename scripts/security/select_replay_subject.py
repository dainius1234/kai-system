#!/usr/bin/env python3
"""S1 — which captured request becomes the Stage-1 replay subject.

D255 froze the rule before the capture existed, because D247 named how
the chosen row is *identified* and never said which row is *chosen* —
and a capture whose rows produced different outcome classes makes that
gap a post-result selection decision waiting to happen.

    S1: the production `llm-call` row with the LOWEST `seq`.

`seq` is stamped when the request is recorded, before the call is
forwarded, so the rule is request-side by construction.

--- why this tool can print nothing that could bias the choice ---------

D257 established, from the probe's source, that an `llm-call` row is
**not** purely request-side: the probe fills one dict with the request,
forwards the call, then adds `raw_response`, `finish_reason`,
`result_type`, `transport_error`, `elapsed_s` and a flipped `layer` to
that same dict before emitting it. So "publish the selected row" would
publish the response.

This tool therefore emits an **ALLOW-LIST projection**. A deny-list would
let a newly added field leak by default; an allow-list withholds anything
nobody has classified. Three exclusions are worth their own sentence:

  * `elapsed_s` — an outcome side-channel we have MEASURED, not one we
    fear in theory. Run 24 recorded 6.4 s for a VALID INSTANCE and 57.0 s
    for a SCHEMA ECHO, so elapsed time is empirically correlated with
    outcome class in our own data.
  * `wall` — stamped in `emit()` after the call returns, so differences
    between rows reconstruct those durations.
  * **a hash of the complete stored row** — deliberately NOT published
    here. It is a deterministic function of a response-bearing row, and
    "probably does not reveal the response" is not the standard; it could
    become a comparison or lookup side channel later. The artifact is
    already immutably bound, so the locator below identifies the row
    without any outcome-derived value. The full row and its hash may be
    opened only once Stage-1 use is authorised.

--- the five preconditions --------------------------------------------

Any failure REFUSES. None permits falling through to another row, which
would be selection-by-convenience wearing a rule's clothes.

  1. P1 returned `REQUEST_REPLAYABLE` over the whole population.
     **Enforced structurally, not asserted here:** this job declares
     `needs:` on the P1 job, and only `REQUEST_REPLAYABLE` exits 0, so
     the selector cannot run unless that verdict was reached. The run id
     is recorded as provenance.
  2. every candidate row has an integer `seq` >= 1
  3. exactly one row holds the minimum `seq`
  4. the selected row's `logical_call_id` is present and non-null
  5. the selected row's `attempt_index` == 1, strictly (null refuses)

2 and 3 are guaranteed by construction within one probe process
(`_SEQ = itertools.count(1)`), and are verified anyway because that
guarantee is structural rather than enforced. 4 and 5 are NOT guaranteed
— both contextvars default to `None` — and without them S1's own
justification, "the first attempt of the first logical call", would be
unlicensed even with a sound ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts" / "security"))

import classify_llm_response as classify  # noqa: E402
import p1_replay_completeness as p1  # noqa: E402

# The ONLY keys this tool may print from a row. Everything else is
# withheld — including any key added to the format after this line was
# written, which is the entire point of an allow-list.
REQUEST_SIDE = (
    "seq", "logical_call_id", "attempt_index", "outside_logical_call",
    "phase", "model", "temperature", "response_format", "tools",
    "other_params", "args_state", "positional_arg_count", "positional_args",
)

# Named so the calibration can assert none of them ever reaches output.
RESPONSE_BEARING = (
    "raw_response", "finish_reason", "result_type", "transport_error",
    "raw_response_note", "elapsed_s", "layer", "wall",
)


def _hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def select(rows: list[dict]) -> tuple[dict | None, list[str], list[str]]:
    """(selected row, checks passed, refusals). Never raises on data."""
    checks: list[str] = []
    refusals: list[str] = []

    population = [r for r in rows if r.get("phase") == p1.PRODUCTION_PHASE]
    if not population:
        return None, checks, [
            f"no production rows (phase={p1.PRODUCTION_PHASE!r}) to select from"]
    checks.append(f"candidate population: {len(population)} production row(s)")

    bad_seq = [r for r in population
               if not isinstance(r.get("seq"), int)
               or isinstance(r.get("seq"), bool)
               or r.get("seq") < 1]
    if bad_seq:
        refusals.append(
            f"precondition 2 FAILED: {len(bad_seq)} of {len(population)} "
            "row(s) have a missing, non-integer or out-of-range `seq`")
        return None, checks, refusals
    checks.append("precondition 2: every candidate row has an integer seq >= 1")

    lowest = min(r["seq"] for r in population)
    holders = [r for r in population if r["seq"] == lowest]
    if len(holders) != 1:
        refusals.append(
            f"precondition 3 FAILED: {len(holders)} rows share the minimum "
            f"seq {lowest}. A duplicated counter is an instrument defect to "
            "investigate, not a choice to make")
        return None, checks, refusals
    checks.append(f"precondition 3: exactly one row holds the minimum seq "
                  f"({lowest})")

    row = holders[0]
    if row.get("logical_call_id") in (None, ""):
        refusals.append(
            "precondition 4 FAILED: the selected row has no "
            "`logical_call_id`, so S1's justification — 'the first attempt "
            "of the first logical call' — is not licensed for it")
        return None, checks, refusals
    checks.append("precondition 4: the selected row has a logical_call_id")

    if row.get("attempt_index") != 1:
        refusals.append(
            f"precondition 5 FAILED: the selected row's attempt_index is "
            f"{row.get('attempt_index')!r}, not 1. S1 selects a FIRST "
            "attempt; a later attempt carries repair context appended by "
            "the reask path and is a different request")
        return None, checks, refusals
    checks.append("precondition 5: the selected row's attempt_index == 1")

    return row, checks, refusals


def projection(row: dict) -> dict:
    """The allow-listed, request-side identity of the selected row.

    Both hashes are over request-side material: the prompt is what was
    sent, and the contract is recovered from THIS row's own system
    message by the shipped classifier — never from a response.
    """
    out = {k: row[k] for k in REQUEST_SIDE if k in row}
    out["prompt_hash"] = _hash(row.get("messages"))
    schema, why, _detail = classify.recover_contract(row.get("messages"))
    out["contract_hash"] = _hash(schema) if schema is not None else None
    out["contract_provenance"] = why
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--capture", required=True)
    ap.add_argument("--p1-run-id", required=True,
                    help="the P1 run whose REQUEST_REPLAYABLE verdict this "
                         "selection stands on (precondition 1)")
    ap.add_argument("--capture-run-id", required=True)
    ap.add_argument("--capture-commit", required=True)
    args = ap.parse_args()

    print("S1 — STAGE-1 REPLAY SUBJECT SELECTION")
    print("=" * 64)
    print("Request-side projection ONLY. No response field is printed, and")
    print("no outcome-derived value — including any hash of the complete")
    print("stored row — is published before Stage-1 use is authorised.")
    print()
    print(f"  artifact  : run {args.capture_run_id}, commit "
          f"{args.capture_commit}")
    print(f"  precondition 1: established by P1 run {args.p1_run_id}.")
    print("    Enforced structurally: this job `needs:` that one, and only")
    print("    REQUEST_REPLAYABLE exits 0, so this selector cannot run")
    print("    unless that verdict was reached. Not re-derived here.")
    print()

    cap = pathlib.Path(args.capture)
    if not cap.is_file():
        # R11: no subject, no observation.
        print("NO SELECTION: the capture file is absent, so there was")
        print("  nothing to select from. This is an availability failure,")
        print("  NOT a selection verdict.")
        return 2

    rows, notes, manifest = p1.read_rows(cap)
    for note in notes:
        print(f"  capture note: {note}")
    row, checks, refusals = select(rows)
    for c in checks:
        print(f"  OK  {c}")
    print()
    print(f"  inspected: {len([r for r in rows if r.get('phase') == p1.PRODUCTION_PHASE])} "
          f"production request row(s) across 5 S1 precondition(s)")

    if row is None:
        print("S1: REFUSED")
        for r in refusals:
            print(f"  - {r}")
        print()
        print("  No row is selected. Stage 1 stays BLOCKED, and no other")
        print("  row is substituted — falling through would be selection by")
        print("  convenience.")
        return 3

    proj = projection(row)
    print("S1: SELECTED")
    print()
    for k in sorted(proj):
        v = proj[k]
        s = json.dumps(v, default=str)
        print(f"    {k:<22} {s[:200]}" + ("…" if len(s) > 200 else ""))
    print()
    print("  WITHHELD until Stage-1 use is authorised: "
          + ", ".join(RESPONSE_BEARING) + ", the full `messages` body, and")
    print("  any hash of the complete stored row.")
    print()
    print("  STAGE 1 REMAINS BLOCKED. This publishes the subject's identity")
    print("  and nothing else; it does not authorise the replay.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
