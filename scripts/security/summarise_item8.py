#!/usr/bin/env python3
"""Item 8's six results, on two axes that may not launder one another.

WHY TWO AXES, AND A THIRD COLUMN
================================

This run does two jobs at once:

* **Axis 1 — the HuggingFace/network contingency.** Does the retry loop
  recover from a failing fetch and refuse after persistent denial?
* **Axis 2 — image provenance**, and the collectors' first qualification
  against a real Docker daemon.

Frozen R2: *"A collector fault leaves Axis 1's result standing and leaves
item 10's provenance unmoved; a clean binding cannot turn a failed
contingency into a success."*

The first implementation of this pair had **one** verdict field. A failed
`.Image` binding rewrote it to UNMEASURED, and this file then printed
that field under "AXIS 1". So an image-provenance fault silently became a
contingency measurement — precisely the laundering R2 forbids. Caught in
adversarial review before any build existed.

Two columns from the runner, computed independently, and a third derived
HERE and never taken from the producer of the first two:

    axis1_verdict          PASS / WRONG_FAILURE / UNMEASURED
    axis2_provenance       BOUND / MISMATCH / UNRECORDED /
                           IMAGE_NOT_PRODUCED_BY_DESIGN
    qualifies              derived below — and per branch, not by a
                           single "sound" set applied to all six; see
                           REQUIRED_A2, where a BOUND B3 is a
                           contradiction rather than acceptable evidence

WHY A ROW COUNT IS NOT A DENOMINATOR
====================================

The first implementation keyed rows by `(image, branch)` into a dict —
which silently collapses duplicates — and then checked only
`len(rows) == 6`. Six rows containing a duplicate and a missing branch
would have satisfied it while one of the six precommitted subjects had
never been measured at all.

**A denominator is the set of precommitted subjects, not a number of
lines.** This requires exactly the six expected keys, each once, no
extras, before any conclusion is drawn — and reports the mismatch
precisely when it is not so.

WHY THE TOOLCHAIN IS RE-HASHED HERE
===================================

The runner puts a `toolchain_sha256` in every row. That proves the six
rows agree with **each other** about which file they ran under; it does
not prove which file that was, and every one of them came from the same
producer. Rule 26 again, and I-8: the expected answer must not come from
the thing under test.

So `--toolchain` takes the artefact itself, recomputes its digest here,
and requires all six rows to carry exactly that value. A row bound to a
different toolchain than the one archived beside the results is a row
whose conditions are unknown, whatever its axes say.

**It is REQUIRED, and that is the whole point.** While the flag was
optional, six rows agreeing with each other reached ALL SIX QUALIFY on
the producer's word alone — the exact defect the paragraph above claims
to close, still reachable through the shipped entry point. Optional
independent evidence is not independent evidence.

Digest equality is also not the whole binding. `tree_sha` and `run_id`
are written into each row by the same runner that wrote the digest into
it, so comparing them with each other is the producer agreeing with
itself. Both are now reconciled against the artefact, as is the
base-image digest each branch observed at its own build — a mutable tag
that moves mid-experiment would otherwise become an unexplained
difference between arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

# The frozen denominator, in the frozen order.
EXPECTED = [(i, b) for i in ("memu-core", "memu-graph")
            for b in ("B1", "B2", "B3")]

PASS = "PASS"
WRONG = "WRONG_FAILURE"
UNMEASURED = "UNMEASURED"

# THE BRANCH CONTRACT, PER BRANCH. Not one "sound" set for all six.
#
# `SOUND_A2 = {BOUND, IMAGE_NOT_PRODUCED_BY_DESIGN}` was a set of states
# that are sound SOMEWHERE, applied EVERYWHERE. Under it a B3 row
# carrying BOUND qualified -- while B3's entire contract is that no image
# is produced -- and a B1 row carrying IMAGE_NOT_PRODUCED_BY_DESIGN could
# qualify on its iidfile alone. Both are contradictions, and a
# contradiction is not sound provenance to be tolerated: it is evidence
# that the row does not describe the branch it claims to.
#
# The frozen design assigns each branch exactly one admissible state, so
# this does too, and anything else REFUSES. (D295)
REQUIRED_A2 = {"B1": "BOUND", "B2": "BOUND",
               "B3": "IMAGE_NOT_PRODUCED_BY_DESIGN"}
SOUND_A2 = set(REQUIRED_A2.values())   # for reporting counts only


def qualifies(r: dict) -> tuple[bool, str]:
    """Closure qualification is DERIVED HERE, not trusted from the runner.

    The runner produces observations and per-axis classifications. It
    does not certify the composite claim — an observation producer that
    also certifies the conclusion drawn from it is a second authority for
    the same statement, and rule 26 says no consequential mechanism
    self-approves. So this recomputes it from the row's evidence, and a
    runner that shipped a `qualified_for_closure` field would be
    contradicted rather than believed.
    """
    # R2 records the toolchain with EVERY branch. A row that does not
    # name the toolchain it ran under is not bound to one.
    tc = r.get("toolchain_sha256")
    if not tc or tc == "ABSENT":
        return False, f"toolchain binding is {tc or 'missing'}"
    if r.get("axis1_verdict") != PASS:
        return False, f"Axis 1 is {r.get('axis1_verdict')}"
    branch = r.get("branch")
    a2 = r.get("axis2_provenance")
    want = REQUIRED_A2.get(branch)
    if want is None:
        return False, f"{branch} is not a precommitted branch"
    if a2 != want:
        if a2 in SOUND_A2:
            return False, (f"Axis 2 is {a2}, which {branch} may never be: "
                           f"the branch contract requires {want}, and a row "
                           f"claiming the other is describing a different "
                           f"branch than the one it is filed under")
        return False, f"Axis 2 is {a2}, not {want}"
    if branch == "B3":
        return True, "refused by design, no image to bind"
    # Positive branches need the iidfile corroboration R2 requires.
    # ABSENT is not "no objection": it is the corroboration missing.
    corr = r.get("iidfile_corroboration")
    if corr != "CORROBORATED":
        return False, f"iidfile corroboration is {corr}"
    return True, "Axis 1 PASS, bound, iidfile corroborated"


def refuse(reason: str, detail: str = "") -> int:
    print("ITEM 8 UNMEASURED — EXPERIMENT INSTRUMENT FAILURE")
    print(f"  unmet prerequisite: {reason}")
    if detail:
        print(f"  {detail}")
    print("  No conclusion is drawn about the contingency from a partial "
          "or malformed result set.")
    return 4


def validate_keys(rows: list[dict]) -> tuple[bool, list[str]]:
    """Exactly the six precommitted subjects, each exactly once."""
    seen = [(r.get("image"), r.get("branch")) for r in rows]
    problems: list[str] = []
    for key in EXPECTED:
        n = seen.count(key)
        if n == 0:
            problems.append(f"MISSING: {key[0]}/{key[1]} was never reported")
        elif n > 1:
            problems.append(f"DUPLICATE: {key[0]}/{key[1]} reported {n} times")
    for key in sorted(set(seen) - set(EXPECTED)):
        problems.append(f"UNEXPECTED: {key[0]}/{key[1]} is not a "
                        f"precommitted subject")
    return (not problems), problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    # REQUIRED. Optional independent evidence is not independent
    # evidence: with the flag omitted, six rows agreeing with each other
    # about a digest reached ALL SIX QUALIFY on the producer's word
    # alone, which is the exact I-8 defect D294 claimed to close. The
    # calibration was itself calling the summariser without it. (D295)
    ap.add_argument("--toolchain", required=True,
                    help="REQUIRED. The toolchain artefact itself. Its "
                         "digest is recomputed here, and its tree and run "
                         "identity are compared against every row -- six "
                         "rows agreeing with each other are six statements "
                         "from one producer")
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_file():
        return refuse(f"{path} does not exist",
                      "No branch result was recorded, so there is nothing "
                      "to report.")
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines()
                if l.strip()]
    except json.JSONDecodeError as e:
        return refuse(f"{path} is not readable as JSONL: {e}")
    if not rows:
        return refuse(f"{path} is empty",
                      "Six branches were precommitted and none reported.")

    ok, problems = validate_keys(rows)

    # THE TOOLCHAIN, RE-HASHED FROM THE ARTEFACT. Computed before the
    # tables so the refusal below can be unconditional: a result set whose
    # conditions are unknown is not summarised under a heading that
    # implies they are known.
    tc_expected = None
    tc_rec: dict[str, str] = {}
    tc_problems: list[str] = []
    tc_path = pathlib.Path(args.toolchain)
    if not tc_path.is_file():
        tc_problems.append(
            f"TOOLCHAIN: {tc_path} does not exist. R2 records these "
            f"identities with every branch; the rows name a digest of "
            f"nothing this summary can read")
    else:
        raw = tc_path.read_bytes()
        tc_expected = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8", "replace").splitlines():
            line = line.strip()
            if line and "=" in line:
                k, v = line.split("=", 1)
                tc_rec[k.strip()] = v.strip()
        # THE HASH IS NOT THE WHOLE BINDING. Digest equality proves a row
        # names THIS file. It does not prove the row was produced under
        # the tree and run the file describes -- `tree_sha` and `run_id`
        # are written into the row by the same runner that wrote the
        # digest into it, so on their own they are the producer agreeing
        # with itself. The artefact is the second source; compare them.
        # (D295)
        want_tree = tc_rec.get("tree_sha")
        want_run = tc_rec.get("run_id")
        for r in rows:
            who = f"{r.get('image')}/{r.get('branch')}"
            got = r.get("toolchain_sha256")
            if got != tc_expected:
                tc_problems.append(
                    f"TOOLCHAIN: {who} is bound to {str(got)[:12]}, not the "
                    f"archived artefact's {tc_expected[:12]}")
            if want_tree and r.get("tree_sha") != want_tree:
                tc_problems.append(
                    f"TOOLCHAIN: {who} names tree "
                    f"{str(r.get('tree_sha'))[:12]}, the artefact names "
                    f"{want_tree[:12]}. A row and its conditions must "
                    f"describe the same tree")
            if want_run and str(r.get("run_id")) != want_run:
                tc_problems.append(
                    f"TOOLCHAIN: {who} names run {r.get('run_id')}, the "
                    f"artefact names {want_run}")
        if not want_tree or not want_run:
            tc_problems.append(
                "TOOLCHAIN: the artefact does not name a tree_sha and a "
                "run_id, so the rows cannot be reconciled against it. "
                "check_item8_toolchain.py requires both before build 1; "
                "this record did not come from a validated run")

        # THE BASE IMAGE IS A MUTABLE TAG. `python:3.11-slim` can move
        # under the experiment, and six arms built against two different
        # base images are not six arms of one experiment. Pinning it
        # would change the subject, so instead each branch RECORDS what
        # the tag resolved to at its own build, and all six must agree
        # with each other and with the pre-run record. Observation, not
        # mutation -- and a divergence blocks interpretation rather than
        # being discovered later as an unexplained difference. (D295)
        want_base = tc_rec.get("base_image_digest")
        seen_base = {str(r.get("base_image_digest")) for r in rows}
        if want_base and seen_base - {want_base}:
            tc_problems.append(
                f"BASE IMAGE: the tag resolved to more than one digest "
                f"across the experiment — recorded before build 1 as "
                f"{want_base[:19]}, observed {sorted(seen_base)}. Six arms "
                f"built on two base images are not six arms of one "
                f"experiment, and which arms differ is not recoverable "
                f"afterwards")

    print("ITEM 8 — HUGGINGFACE/NETWORK CONTINGENCY")
    print("=" * 74)
    print()
    print("AXIS 1 — the contingency (computed with NO identity input)")
    print("-" * 74)
    for image, branch in EXPECTED:
        matches = [r for r in rows if (r.get("image"), r.get("branch"))
                   == (image, branch)]
        if not matches:
            print(f"  {image:<12} {branch}  NOT REPORTED")
            continue
        for r in matches:
            print(f"  {image:<12} {branch}  {r.get('axis1_verdict', '?'):<14}"
                  f" retries={r.get('runtime_retries_observed', '?')}"
                  f" elapsed={r.get('elapsed_seconds', '?')}s")
            if r.get("note"):
                print(f"  {'':<12}     {r['note']}")

    print()
    print("AXIS 2 — provenance (separate; may block closure, never Axis 1)")
    print("-" * 74)
    for image, branch in EXPECTED:
        for r in [r for r in rows if (r.get("image"), r.get("branch"))
                  == (image, branch)]:
            q, why = qualifies(r)
            print(f"  {image:<12} {branch}  "
                  f"{r.get('axis2_provenance', 'UNRECORDED'):<30}"
                  f" iidfile={r.get('iidfile_corroboration', 'n/a'):<14}"
                  f" qualifies={'yes' if q else 'NO'}")
            if not q:
                print(f"  {'':<12}     {why}")

    a1 = {v: sum(1 for r in rows if r.get("axis1_verdict") == v)
          for v in (PASS, WRONG, UNMEASURED)}
    a2_sound = sum(1 for r in rows if r.get("axis2_provenance") in SOUND_A2)
    quals = {(r.get("image"), r.get("branch")): qualifies(r) for r in rows}
    qualified = sum(1 for v in quals.values() if v[0])

    # A runner that certifies its own composite claim is contradicted,
    # not trusted. Nothing currently emits this field; if something does,
    # a disagreement is a finding.
    disagreements: list[str] = []
    for r in rows:
        if "qualified_for_closure" in r:
            got, why = qualifies(r)
            if bool(r["qualified_for_closure"]) != got:
                disagreements.append(
                    f"DISAGREEMENT: {r.get('image')}/{r.get('branch')} row "
                    f"claims qualified={r['qualified_for_closure']}, derived "
                    f"{got} ({why})")

    print()
    print(f"  inspected: {len(rows)} result row(s) against "
          f"{len(EXPECTED)} precommitted subject(s)")
    print(f"    AXIS 1   PASS {a1[PASS]}  WRONG_FAILURE {a1[WRONG]}  "
          f"UNMEASURED {a1[UNMEASURED]}")
    print(f"    AXIS 2   sound {a2_sound} of {len(rows)}")
    print(f"    QUALIFIED FOR CLOSURE  {qualified} of {len(EXPECTED)}")
    if tc_expected:
        print(f"    TOOLCHAIN  recomputed {tc_expected[:16]}… from the "
              f"artefact, reconciled against {len(rows)} row(s) on digest, "
              f"tree, run and base image")
    else:
        print("    TOOLCHAIN  NOT RECOMPUTED — the artefact is missing")

    print()
    print("  Reading rules, frozen before these results existed:")
    print("   * B2 measures recovery from an INJECTED FETCH-COMMAND failure.")
    print("     It does NOT measure recovery from a real network outage.")
    print("   * UNMEASURED is never an adverse result about the contingency.")
    print("   * WRONG_FAILURE is never a PASS and never a FAIL of it.")
    print("   * An Axis-2 fault blocks closure and leaves Axis 1 standing.")
    print("   * Closure qualification is DERIVED here from the evidence,")
    print("     never taken from the producer of it (rule 26).")
    print("   * No re-draws. An UNMEASURED branch stays UNMEASURED and Item 8")
    print("     is incomplete for that subject. (D247 §5, D289)")

    if tc_problems:
        print()
        for p in tc_problems:
            print(f"FAIL: {p}")
        print()
        print("Six rows agreeing with each other about a digest are six "
              "statements from one producer. The artefact is the "
              "independent evidence (I-8), and a row bound to a different "
              "toolchain ran under conditions this summary cannot name.")
        return 4

    if disagreements:
        print()
        for d in disagreements:
            print(f"FAIL: {d}")
        print()
        print("A row carrying a composite claim that contradicts the "
              "evidence is schema drift, in EITHER direction. The producer "
              "of an observation does not certify the conclusion drawn "
              "from it (rule 26), and a contradiction is refused rather "
              "than noted.")
        return 4

    if not ok:
        print()
        for p in problems:
            print(f"FAIL: {p}")
        print()
        print("The denominator is the six precommitted subjects, not a count "
              "of lines. It is not adjusted downward, and a duplicate does "
              "not substitute for a missing subject.")
        return 4

    if a1[PASS] == len(EXPECTED) and qualified == len(EXPECTED):
        print()
        print("ALL SIX QUALIFY: both contingencies recover from an injected "
              "failure and refuse after persistent denial, and every branch "
              "carries sound provenance.")
        return 0

    print()
    if a1[PASS] == len(EXPECTED):
        print(f"AXIS 1 COMPLETE, PROVENANCE INCOMPLETE: 6/6 contingency PASS "
              f"but only {qualified}/6 qualify for closure. The contingency "
              f"result stands; item 10's provenance does not move for the "
              f"branches whose Axis 2 is unsound.")
    else:
        print(f"NOT ALL SIX PASS: Axis 1 {a1[PASS]}/6. Item 8 is not "
              f"satisfied. Every outcome above is banked as it occurred.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
