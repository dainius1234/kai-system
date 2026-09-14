"""The Unified Hunter runner — the execution plan causes execution.

WF-3's defect was never that a regex was too narrow. It was that **stdout
text was allowed to infer the test population**. A suite that renamed its
tally line vanished from the floors silently; a suite whose label carried
parentheses was never in the population at all; and a nested subject that
printed a tally of its own was admitted as though it were a suite.

So the authority moves. This runner traverses
`scripts/security/uh_execution_plan.json` and executes what it finds there.
The plan is not a list kept beside the measured thing — it is the object
that causes the measurement. Membership is defined there and nowhere else:
not by stdout, not by a filesystem glob, not by parsing the Make database.

Stdout may still report counts. It may never create or remove a member.

Two states are kept orthogonal, and the order they are decided in is the
whole of SC-1:

    execution_state      COMPLETED | FAILED | NOT_STARTED
    result_observation   RESOLVED | ABSENT | AMBIGUOUS | MALFORMED
                         | NOT_OBSERVED

**Execution status is adjudicated first.** A target that exits non-zero is
FAILED, whatever its output did or did not contain. That is not pedantry:
on 2026-09-12 `test-container-proof-harness` died on an uncaught 120-second
timeout *before* reaching the epilogue that prints its tally. Calling that a
result-contract conflict would say "the plan's contract is wrong" when the
truth is "the subject failed before it could report". The contract is
mandatory only for a target that exits 0.

**The runner does not own the aggregate evidence.** That boundary is
WF-2's and it is not this component's to cross:

    the workflow shell owns    the aggregate run.log, and run.log.status
                               captured from TOP-LEVEL `make test-uh`
                               under pipefail
    the runner owns            plan traversal, per-target execution and
                               result observations, and the final atomic
                               results.json

So the runner forwards each target's output to its own stdout and lets the
outer `make test-uh 2>&1 | tee "$ROOT/run.log"` remain the single producer
of the canonical log. It writes no status sidecar, and the manifest carries
no aggregate status field. The runner's process exit and the top-level Make
status are two different observations; naming them alike would invite a
later reader to treat either as completion authority.

Exit codes — a signal to Make, not the canonical status:
  0  every target COMPLETED and satisfied its result contract
  1  a target FAILED, or a completed target broke its contract
  2  the plan or the evidence context is unusable — nothing was run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
PLAN = Path(__file__).resolve().parent / "uh_execution_plan.json"

PLAN_SCHEMA = "kai.uh-plan/v1"
RUN_SCHEMA = "kai.uh-run/v1"

# The evidence root is the run identity. There is no separate run id that
# could disagree with the directory it names.
EVIDENCE_ENV = "KAI_UH_EVIDENCE_ROOT"

# Make options that would silently change what "serial, fail-fast" means.
# Production v3.3 is serial. Rather than reinterpret an unsupported
# invocation, refuse it.
#
# The letter table is not the dashed spelling, and that distinction was
# measured rather than assumed. GNU Make 4.3 packs single-letter options
# into the FIRST word of MAKEFLAGS **with no leading dash**:
#
#     make -k            -> MAKEFLAGS=[k]
#     make --keep-going  -> MAKEFLAGS=[k]
#     make -i            -> MAKEFLAGS=[i]
#     make -n            -> MAKEFLAGS=[n]
#     make -kj2          -> MAKEFLAGS=[k -j2 --jobserver-auth=3,4]
#     make -j4           -> MAKEFLAGS=[ -j4 --jobserver-auth=3,4]
#
# Only options carrying an argument keep their dash. A check written
# against "-k" therefore matches nothing at all, which is the worst
# possible failure for a guard: it reports clean while keep-going,
# ignore-errors and dry-run all pass straight through.
HOSTILE_LETTERS = {
    "j": "parallel execution",
    "i": "ignore-errors",
    "k": "keep-going",
    "n": "dry-run",
}
HOSTILE_LONG = {
    "--jobs": "parallel execution",
    "--ignore-errors": "ignore-errors",
    "--keep-going": "keep-going",
    "--just-print": "dry-run", "--dry-run": "dry-run", "--recon": "dry-run",
}


class Refusal(Exception):
    """Nothing was adjudicated, and why. Distinct from a finding."""

    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


# ── the plan ─────────────────────────────────────────────────────────

def load_plan(path: Path = PLAN) -> Tuple[List[dict], str]:
    """The plan's entries and the SHA-256 of its exact raw bytes.

    The digest is over bytes as they sit on disk. No canonicalisation is
    invented and no parsed structure is hashed: any byte change is a new
    digest, which is the point.
    """
    if not path.is_file():
        raise Refusal("PLAN_INVALID", f"no execution plan at {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise Refusal("PLAN_INVALID", f"plan is not readable JSON: {exc}")

    schema = doc.get("schema")
    if schema != PLAN_SCHEMA:
        raise Refusal("PLAN_INVALID",
                      f"unknown plan schema {schema!r}; this runner "
                      f"understands only {PLAN_SCHEMA!r}")

    entries = doc.get("targets")
    if not isinstance(entries, list) or not entries:
        raise Refusal("PLAN_INVALID", "plan declares no targets")

    seen = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise Refusal("PLAN_INVALID", f"entry {i} is not an object")
        target, label = e.get("make_target"), e.get("result_label")
        if not isinstance(target, str) or not target:
            raise Refusal("PLAN_INVALID", f"entry {i} has no make_target")
        if not isinstance(label, str) or not label:
            raise Refusal("PLAN_INVALID",
                          f"entry {i} ({target}) has no result_label")
        if target in seen:
            raise Refusal("PLAN_INVALID", f"duplicate target {target!r}")
        seen.add(target)
    return entries, digest


def result_pattern(label: str) -> "re.Pattern[str]":
    """The one line a target is contracted to print, matched literally.

    `re.escape` is not optional. Real labels in this repository contain
    parentheses and a leading slash — `Service identity (ed25519) tests`
    and `/observe_turn identity slice`. Interpolating either into a
    pattern turns punctuation into syntax, and punctuation must never
    decide membership.

    Anchored at both ends: no leading text, no trailing text.
    """
    return re.compile(rf"^{re.escape(label)}:\s+(\d+)\s+passed,\s+(\d+)\s+failed\s*$")


def label_pattern(label: str) -> "re.Pattern[str]":
    """Lines that CLAIM to be this target's result, valid or not.

    Separating the claim from the grammar is what makes MALFORMED a real
    state rather than a word in a docstring. A line reading
    `Alpha Tests: seventeen passed` is the target asserting its own
    result and getting it wrong — which is a contract defect. Folding it
    into ABSENT would report "the target said nothing", and the target
    said something quite loud.

    Also line-start anchored, so indented nested output cannot claim a
    parent's label.
    """
    return re.compile(rf"^{re.escape(label)}:")


# ── the environment ──────────────────────────────────────────────────

def check_makeflags(env: Dict[str, str]) -> None:
    flags = env.get("MAKEFLAGS", "")
    if not flags.strip():
        return

    def refuse(spelling: str, meaning: str) -> None:
        raise Refusal(
            "EVIDENCE_CONTEXT_INVALID",
            f"MAKEFLAGS carries {spelling} ({meaning}) in {flags!r}. This "
            f"runner is serial and fail-fast; it refuses an invocation whose "
            f"semantics it would otherwise have to reinterpret.")

    tokens = flags.split()
    # The undashed first word is the packed single-letter cluster.
    if tokens and not tokens[0].startswith("-"):
        for ch in tokens[0]:
            if ch in HOSTILE_LETTERS:
                refuse(f"{ch!r}", HOSTILE_LETTERS[ch])
    for tok in tokens:
        if tok.startswith("--"):
            base = tok.split("=", 1)[0]
            if base in HOSTILE_LONG:
                refuse(base, HOSTILE_LONG[base])
        elif tok.startswith("-"):
            for ch in tok[1:]:
                if ch.isdigit():
                    break            # the argument, e.g. the 4 in -j4
                if ch in HOSTILE_LETTERS:
                    refuse(f"-{ch}", HOSTILE_LETTERS[ch])


def evidence_root(explicit: Optional[str]) -> Path:
    """One root; siblings derive from it mechanically.

    Callers never choose log A with status B and manifest C. CI creates
    the directory and hands the absolute path through the environment;
    a local run gets an OS temp directory. There is no fixed `/tmp`
    fallback, because a fixed path is how two runs come to share one
    piece of evidence.
    """
    if explicit:
        root = Path(explicit)
        if not root.is_dir():
            raise Refusal("EVIDENCE_CONTEXT_INVALID",
                          f"{EVIDENCE_ENV} names {root}, which is not a "
                          f"directory")
        if not root.is_absolute():
            raise Refusal("EVIDENCE_CONTEXT_INVALID",
                          f"{EVIDENCE_ENV} must be absolute, got {root}")
        return root
    return Path(tempfile.mkdtemp(prefix="kai-uh-"))


# ── execution ────────────────────────────────────────────────────────

def classify(exit_code: int, output: str, label: str) -> Tuple[str, str, Optional[dict], Optional[str]]:
    """(execution_state, result_observation, result, refusal_code).

    SC-1: the exit status decides the execution state FIRST. The result
    contract binds only a target that exited 0.
    """
    claims = [l.rstrip() for l in output.splitlines()
              if label_pattern(label).match(l.rstrip())]
    valid = [m for m in (result_pattern(label).match(l) for l in claims) if m]

    if not claims:
        observed, result = "ABSENT", None
    elif not valid:
        # The target claimed its own result and the claim does not parse.
        observed, result = "MALFORMED", None
    elif len(valid) > 1:
        observed, result = "AMBIGUOUS", None
    elif len(claims) > 1:
        # Exactly one parses, but the target made another claim beside it.
        # Fail closed: we cannot say which line is the result. Recorded as
        # AMBIGUOUS rather than MALFORMED because the defect is that there
        # is more than one candidate, not that the winner was unreadable.
        observed, result = "AMBIGUOUS", None
    else:
        passed, failed = int(valid[0].group(1)), int(valid[0].group(2))
        observed, result = "RESOLVED", {"passed": passed, "failed": failed}

    if exit_code != 0:
        # FAILED, always. Whatever the output did or did not contain is
        # diagnostic; it does not replace the execution state, and its
        # absence is not a contract defect.
        return "FAILED", observed, result, None

    if observed in ("ABSENT", "MALFORMED", "AMBIGUOUS"):
        return "FAILED", observed, None, "RESULT_CONTRACT_CONFLICT"
    if result and result["failed"] > 0:
        # A green process contradicting its own tally must fail closed.
        return "FAILED", observed, result, "RESULT_CONTRACT_CONFLICT"
    return "COMPLETED", observed, result, None


def run(entries: List[dict], env: Dict[str, str]) -> List[dict]:
    slots: List[dict] = [
        {"position": i, "make_target": e["make_target"],
         "result_label": e["result_label"], "execution_state": "NOT_STARTED",
         "result_observation": "NOT_OBSERVED", "exit_code": None,
         "result": None, "refusal": None}
        for i, e in enumerate(entries)
    ]

    failed_at: Optional[int] = None
    for slot in slots:
        if failed_at is not None:
            break                      # serial, fail-fast; the rest stay
        target = slot["make_target"]   # NOT_STARTED rather than vanishing
        slot["execution_state"] = "RUNNING"
        proc = subprocess.run(["make", target], cwd=str(REPO), env=env,
                              capture_output=True, text=True)
        out = proc.stdout + proc.stderr
        # FORWARD, do not publish. The canonical aggregate log belongs to
        # the workflow shell's `make test-uh 2>&1 | tee $ROOT/run.log`.
        # The runner captures target output only to classify it, then
        # writes it through so the outer tee remains the single producer.
        #
        # CAPTURE-THEN-FORWARD IS A DELIBERATE CHOICE WITH A KNOWN COST:
        # a target's output appears all at once when it completes rather
        # than streaming live, so interleaving and timing differ from the
        # old aggregate. A streaming implementation would preserve timing
        # but changes signal/complexity semantics. This choice is an input
        # to the execution-semantics comparison, not a settled question.
        sys.stdout.write(out)
        sys.stdout.flush()

        state, observed, result, refusal = classify(
            proc.returncode, out, slot["result_label"])
        slot.update(execution_state=state, result_observation=observed,
                    exit_code=proc.returncode, result=result, refusal=refusal)
        if state == "FAILED":
            failed_at = slot["position"]
    return slots


def publish(root: Path, payload: dict) -> Path:
    """Write the manifest atomically; only the final path is evidence.

    A reader must never see a half-written manifest, and must never have
    to glob for something manifest-shaped. Serialise to a sibling, flush,
    close, then rename.
    """
    final = root / "results.json"
    tmp = root / "results.json.partial"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, final)
    return final


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, default=PLAN)
    ap.add_argument("--evidence-root", default=os.environ.get(EVIDENCE_ENV))
    args = ap.parse_args()

    try:
        entries, digest = load_plan(args.plan)
        env = dict(os.environ)
        check_makeflags(env)
        root = evidence_root(args.evidence_root)
    except Refusal as r:
        print(f"REFUSED ({r.code}): {r.message}")
        return 2

    print(f"Unified Hunter — {len(entries)} targets from {args.plan.name}")
    print(f"  plan digest   : {digest}")
    print(f"  evidence root : {root}")

    slots = run(entries, env)

    completed = sum(1 for s in slots if s["execution_state"] == "COMPLETED")
    failed = [s for s in slots if s["execution_state"] == "FAILED"]
    not_started = sum(1 for s in slots if s["execution_state"] == "NOT_STARTED")

    # A calibration plan may legitimately live outside the repository, and
    # recording its identity must not crash. A repo-relative path is only
    # meaningful for the canonical plan; anything else is recorded
    # absolutely and marked, so an external fixture can never be mistaken
    # for production authority.
    resolved = args.plan.resolve()
    try:
        plan_path, scope = str(resolved.relative_to(REPO)), "repository"
    except ValueError:
        plan_path, scope = str(resolved), "external"

    publish(root, {
        "schema": RUN_SCHEMA,
        "plan_digest": digest,
        "plan_path": plan_path,
        "plan_scope": scope,
        "evidence_root": str(root),
        "population": len(slots),
        "slots": slots,
    })

    print(f"\n  COMPLETED {completed} · FAILED {len(failed)} · "
          f"NOT_STARTED {not_started} of {len(slots)}")
    for s in failed:
        detail = f" [{s['refusal']}]" if s["refusal"] else ""
        print(f"  FAILED at #{s['position']} {s['make_target']} "
              f"(exit {s['exit_code']}, result {s['result_observation']})"
              f"{detail}")
    # The process exit exists so Make can observe success or failure. It is
    # NOT the canonical aggregate status: that is the top-level `make
    # test-uh` pipeline status, captured by the workflow shell into
    # run.log.status under the WF-2 pattern. What GNU Make returns to the
    # workflow for each of these values is a measurement owed to the
    # execution-semantics comparison, not something to assert here.
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
