"""The plan-aware assertion-floor consumer — SHADOW, not yet production.

This is the WF-3 side of the floor gate. It exists beside
`check_assertion_floors.py`, which is untouched and remains the production
path until the atomic cutover. Nothing in CI invokes this module.

**One evidence root, one authority.** The consumer is handed a single
absolute directory and derives its siblings mechanically:

    <root>/run.log          the canonical aggregate log   (workflow shell)
    <root>/run.log.status   the canonical aggregate status(workflow shell)
    <root>/results.json     the per-target manifest       (runner)

A caller may not select log A with status B and manifest C. That mixing is
exactly the authority-boundary fault recorded as INC-2026-09-14-16, and the
single root is the structural answer to it.

**Authority precedes supporting evidence.** The order below is semantic,
not stylistic. The manifest is never consulted first and then used to decide
whether the sidecar looks plausible:

    1. the evidence-root interface
    2. the exact sibling set
    3. the authoritative status sidecar
    4. non-zero status  -> AGGREGATE_INCOMPLETE, and stop
    5. zero status only -> the canonical plan
    6.                     the manifest
    7.                     plan <-> manifest reconciliation
    8.                     floor schema and set relations
    9.                     floor findings and unfloored targets
   10.                     determinism, where authorised

**`run.log` is never parsed for population or counts.** Population comes
from the execution plan; counts come from the target-bound manifest. The log
is evidence and context. If this module ever called a stdout parser to
discover a suite, WF-3's original lexical authority would have survived
inside the path built to remove it.

Exit codes:
  0  adjudicated, and every floored target met its floor
  1  adjudicated, and there is a finding
  2  refused — nothing was adjudicated
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
CANONICAL_PLAN = Path(__file__).resolve().parent / "uh_execution_plan.json"

PLAN_SCHEMA = "kai.uh-plan/v1"
RUN_SCHEMA = "kai.uh-run/v1"
FLOORS_SCHEMA = "kai.uh-floors/v2"
DETERMINISM_SCHEMA = "kai.uh-determinism/v1"

EVIDENCE_ENV = "KAI_UH_EVIDENCE_ROOT"
DETERMINISM_EXPECTED = 14          # machine-enforced, not a comment


class Refusal(Exception):
    """Nothing was adjudicated, and why. Never a finding."""

    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


# ── 1. the evidence-root interface ───────────────────────────────────

def resolve_root(cli: Optional[str], env: Optional[str]) -> Path:
    """Exactly one evidence context, from at most two agreeing sources.

    There is no fixed `/tmp` fallback and no legacy path. An absent root
    is a refusal, not a default: a default would silently adjudicate
    whatever happened to be lying in a well-known location.
    """
    if cli and env:
        a, b = Path(cli).resolve(), Path(env).resolve()
        if a != b:
            raise Refusal("EVIDENCE_CONTEXT_INVALID",
                          f"--evidence-root {a} and {EVIDENCE_ENV} {b} name "
                          f"different contexts; neither wins")
        chosen: Optional[str] = cli
    else:
        chosen = cli or env

    if not chosen:
        raise Refusal("EVIDENCE_CONTEXT_INVALID",
                      f"no evidence root: pass --evidence-root or set "
                      f"{EVIDENCE_ENV}. There is no default location.")
    root = Path(chosen)
    if not root.is_absolute():
        raise Refusal("EVIDENCE_CONTEXT_INVALID",
                      f"evidence root must be absolute, got {root}")
    if not root.is_dir():
        raise Refusal("EVIDENCE_CONTEXT_INVALID",
                      f"evidence root {root} is not a directory")
    return root.resolve()


# ── 2. the exact sibling set ─────────────────────────────────────────

def require_siblings(root: Path) -> Tuple[Path, Path, Path]:
    log, status, manifest = (root / "run.log", root / "run.log.status",
                             root / "results.json")
    missing = [p.name for p in (log, status, manifest) if not p.is_file()]
    if missing:
        raise Refusal(
            "REQUIRED_INPUT_MISSING",
            f"the evidence set is incomplete in {root}: missing "
            + ", ".join(missing)
            + ". A present manifest does not imply the aggregate ran.")
    return log, status, manifest


# ── 3. the authoritative status ──────────────────────────────────────

def read_status(path: Path) -> int:
    raw = path.read_text(encoding="utf-8")
    text = raw.strip()
    if not text or "\n" in text or not text.lstrip("-").isdigit():
        shown = raw[:80].replace("\n", "\\n")
        raise Refusal("STATUS_UNREADABLE",
                      f"{path.name} is not a single integer: {shown!r} "
                      f"({len(raw)} bytes)")
    return int(text)


# ── 5. the canonical plan ────────────────────────────────────────────

def load_canonical_plan() -> Tuple[List[dict], str]:
    if not CANONICAL_PLAN.is_file():
        raise Refusal("PLAN_INVALID", f"no plan at {CANONICAL_PLAN}")
    raw = CANONICAL_PLAN.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise Refusal("PLAN_INVALID", f"plan unreadable: {exc}")
    return validate_plan(doc), digest


def validate_plan(doc: object) -> List[dict]:
    """Validate the ONE canonical plan. This creates no second population.

    The consumer does not assume the runner already checked this. Two
    components reading one authority must each satisfy themselves it is
    well-formed, or a malformed plan is only caught by whichever happens
    to look first.
    """
    if not isinstance(doc, dict):
        raise Refusal("PLAN_INVALID", "plan is not an object")
    if doc.get("schema") != PLAN_SCHEMA:
        raise Refusal("PLAN_INVALID",
                      f"unknown plan schema {doc.get('schema')!r}")
    entries = doc.get("targets")
    if not isinstance(entries, list) or not entries:
        raise Refusal("PLAN_INVALID", "plan declares no targets")
    seen = set()
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            raise Refusal("PLAN_INVALID", f"plan entry {i} is not an object")
        target, label = e.get("make_target"), e.get("result_label")
        if not isinstance(target, str) or not target:
            raise Refusal("PLAN_INVALID",
                          f"plan entry {i} has no usable make_target")
        if not isinstance(label, str) or not label:
            raise Refusal("PLAN_INVALID",
                          f"plan entry {i} ({target}) has no usable "
                          f"result_label")
        if target in seen:
            raise Refusal("PLAN_INVALID",
                          f"plan repeats target {target!r}")
        seen.add(target)
    return entries


# ── 6/7. the manifest, and its reconciliation with the plan ──────────

VALID_STATES = {"COMPLETED", "FAILED", "NOT_STARTED"}


def load_manifest(path: Path, root: Path, plan: List[dict],
                  plan_digest: str) -> List[dict]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Refusal("MANIFEST_INVALID", f"manifest unreadable: {exc}")

    if doc.get("schema") != RUN_SCHEMA:
        raise Refusal("MANIFEST_INVALID",
                      f"unknown manifest schema {doc.get('schema')!r}; this "
                      f"gate adjudicates only {RUN_SCHEMA!r}. A calibration "
                      f"schema is not admissible to production adjudication.")

    # Same-context coherence, not independent corroboration: the manifest
    # must describe the root it was actually handed.
    declared = doc.get("evidence_root")
    if declared is None or Path(str(declared)).resolve() != root:
        raise Refusal("EVIDENCE_CONTEXT_INVALID",
                      f"manifest records evidence_root {declared!r}, which is "
                      f"not the supplied context {root}")

    if doc.get("plan_scope") != "repository":
        raise Refusal("PLAN_INVALID",
                      f"manifest plan_scope is {doc.get('plan_scope')!r}. "
                      f"Production adjudication requires the canonical "
                      f"repository plan; an external calibration plan carrying "
                      f"plausible target names is not authority.")

    if doc.get("plan_digest") != plan_digest:
        raise Refusal("PLAN_INVALID",
                      f"manifest plan_digest {doc.get('plan_digest')!r} does "
                      f"not match the canonical plan's raw-byte sha256 "
                      f"{plan_digest}")

    declared_path = doc.get("plan_path")
    canonical_rel = str(CANONICAL_PLAN.relative_to(REPO))
    if declared_path != canonical_rel:
        raise Refusal("MANIFEST_INVALID",
                      f"manifest plan_path is {declared_path!r}, not the "
                      f"canonical {canonical_rel!r}. A correct scope and "
                      f"digest do not entitle a false provenance field to "
                      f"travel as trustworthy evidence.")

    slots = doc.get("slots")
    if not isinstance(slots, list):
        raise Refusal("MANIFEST_INVALID", "manifest carries no slots")

    population = doc.get("population")
    if isinstance(population, bool) or not isinstance(population, int):
        raise Refusal("MANIFEST_INVALID",
                      f"manifest population {population!r} is not an integer")
    if population != len(plan):
        raise Refusal("MANIFEST_INVALID",
                      f"manifest declares population {population}, the "
                      f"canonical plan declares {len(plan)}")
    if population != len(slots):
        raise Refusal("MANIFEST_INVALID",
                      f"manifest declares population {population} but carries "
                      f"{len(slots)} slots")

    expected = [e["make_target"] for e in plan]
    got = [s.get("make_target") for s in slots]

    if len(got) != len(expected):
        raise Refusal("MANIFEST_INVALID",
                      f"population is {len(got)}, the plan declares "
                      f"{len(expected)}")
    if len(set(got)) != len(got):
        dupes = sorted({t for t in got if got.count(t) > 1})
        raise Refusal("MANIFEST_INVALID", f"duplicate target(s): {dupes}")
    foreign = [t for t in got if t not in set(expected)]
    if foreign:
        raise Refusal("MANIFEST_INVALID", f"foreign target(s): {foreign}")
    absent = [t for t in expected if t not in set(got)]
    if absent:
        raise Refusal("MANIFEST_INVALID", f"missing target(s): {absent}")
    if got != expected:
        first = next(i for i, (a, b) in enumerate(zip(got, expected)) if a != b)
        raise Refusal("MANIFEST_INVALID",
                      f"target order diverges at position {first}: manifest "
                      f"{got[first]!r}, plan {expected[first]!r}")

    for s in slots:
        if s.get("execution_state") not in VALID_STATES:
            raise Refusal("MANIFEST_INVALID",
                          f"{s.get('make_target')} has execution_state "
                          f"{s.get('execution_state')!r}")

    # Fail-fast shape: nothing may be COMPLETED after the first FAILED.
    seen_failure = False
    for s in slots:
        if seen_failure and s["execution_state"] != "NOT_STARTED":
            raise Refusal("MANIFEST_INVALID",
                          f"{s['make_target']} is {s['execution_state']} after "
                          f"an earlier FAILED; production traversal is serial "
                          f"and fail-fast")
        if s["execution_state"] == "FAILED":
            seen_failure = True
    return slots


def _count(value: object, field: str, target: str) -> int:
    """A count, or a governed refusal. Never a TypeError.

    `bool` is excluded explicitly because it is an `int` in Python, and
    `True` would otherwise pass as the number 1 — a count that is really a
    flag, which is the shape of defect this gate exists to catch.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise Refusal("RESULT_CONTRACT_CONFLICT",
                      f"{target}: result {field} is {value!r}, not an integer")
    if value < 0:
        raise Refusal("RESULT_CONTRACT_CONFLICT",
                      f"{target}: result {field} is {value}, which is negative")
    return value


def reconcile(slots: List[dict], plan: List[dict]) -> Dict[str, int]:
    """Prove every slot is the target-bound result the plan asked for.

    Reached only after the authoritative sidecar has been read and found
    to be zero. Identity is checked BY ORDINAL against the canonical plan:
    a slot must carry the plan's target AND the plan's exact result_label.

    The label check is the frozen invariant -- every population member
    proves one exact TARGET-BOUND assertion result. Without it a manifest
    could present a count under a label the plan never declared, and the
    count would be adjudicated anyway. The first version of this gate did
    exactly that, and its own positive fixture carried fabricated labels
    that adjudicated cleanly.

    Returns target -> passed, and nothing reaches the floor comparison
    that has not come through here.
    """
    counts: Dict[str, int] = {}
    for i, (slot, entry) in enumerate(zip(slots, plan)):
        target = entry["make_target"]

        if slot.get("position") != i:
            raise Refusal("MANIFEST_INVALID",
                          f"slot for {target} carries position "
                          f"{slot.get('position')!r}, expected ordinal {i}")
        if slot.get("make_target") != target:
            raise Refusal("MANIFEST_INVALID",
                          f"ordinal {i}: manifest {slot.get('make_target')!r}, "
                          f"plan {target!r}")
        if slot.get("result_label") != entry["result_label"]:
            raise Refusal("RESULT_CONTRACT_CONFLICT",
                          f"{target}: manifest result_label "
                          f"{slot.get('result_label')!r} is not the plan's "
                          f"{entry['result_label']!r}. The count is not bound "
                          f"to the contracted result.")
        if slot.get("execution_state") != "COMPLETED":
            raise Refusal("MANIFEST_INVALID",
                          f"the authoritative status is 0, but {target} is "
                          f"{slot.get('execution_state')!r}")
        if slot.get("exit_code") != 0:
            raise Refusal("MANIFEST_INVALID",
                          f"{target} is COMPLETED with exit_code "
                          f"{slot.get('exit_code')!r}")
        if slot.get("result_observation") != "RESOLVED":
            raise Refusal("RESULT_CONTRACT_CONFLICT",
                          f"{target} is COMPLETED but its result observation "
                          f"is {slot.get('result_observation')!r}")

        result = slot.get("result")
        if not isinstance(result, dict):
            raise Refusal("RESULT_CONTRACT_CONFLICT",
                          f"{target}: result is {result!r}, not an object")
        if "passed" not in result:
            raise Refusal("RESULT_CONTRACT_CONFLICT",
                          f"{target}: result carries no `passed`")
        if "failed" not in result:
            raise Refusal("RESULT_CONTRACT_CONFLICT",
                          f"{target}: result carries no `failed`")

        passed = _count(result["passed"], "passed", target)
        failed = _count(result["failed"], "failed", target)
        if failed != 0:
            # A green aggregate contradicting its own per-target tally must
            # fail closed. Quietly using `passed` and ignoring `failed`
            # would adjudicate a floor against a run that reported failures.
            raise Refusal("RESULT_CONTRACT_CONFLICT",
                          f"{target} reports {failed} failed while the "
                          f"authoritative aggregate status is 0")
        counts[target] = passed
    return counts


# ── 8. floors ────────────────────────────────────────────────────────

def load_floors(path: Path) -> Dict[str, int]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal("FLOORS_INVALID", f"floors unreadable: {exc}")
    schema = doc.get("schema")
    if schema != FLOORS_SCHEMA:
        raise Refusal(
            "FLOORS_INVALID",
            f"floors schema is {schema!r}. This path consumes only "
            f"{FLOORS_SCHEMA!r}, keyed by make target. Label-keyed v1 floors "
            f"are NOT translated at runtime: a silent compatibility mode is "
            f"how the old identity would survive inside the new authority.")
    floors = doc.get("floors")
    if not isinstance(floors, dict) or not floors:
        raise Refusal("FLOORS_INVALID", "floors carries no entries")
    for target, value in floors.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise Refusal("FLOORS_INVALID",
                          f"{target} floor {value!r} is not an integer")
        if value <= 0:
            raise Refusal("FLOORS_INVALID",
                          f"{target} floor is {value}; a zero or negative "
                          f"floor asserts nothing")
    return floors


def load_determinism(path: Optional[Path], P: set, F: set) -> List[str]:
    if path is None:
        return []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal("DETERMINISM_INVALID", f"policy unreadable: {exc}")
    if doc.get("schema") != DETERMINISM_SCHEMA:
        raise Refusal("DETERMINISM_INVALID",
                      f"unknown policy schema {doc.get('schema')!r}")
    targets = doc.get("targets")
    if not isinstance(targets, list):
        raise Refusal("DETERMINISM_INVALID", "policy declares no targets")
    D = set(targets)
    if len(D) != len(targets):
        raise Refusal("DETERMINISM_INVALID", "policy repeats a target")
    if len(D) != DETERMINISM_EXPECTED:
        raise Refusal("DETERMINISM_INVALID",
                      f"policy names {len(D)} targets; the authorised sample "
                      f"is exactly {DETERMINISM_EXPECTED}. A rename or removal "
                      f"must not silently shrink it, and growth is scope "
                      f"expansion requiring separate authorisation.")
    if not D <= P:
        raise Refusal("DETERMINISM_INVALID",
                      f"policy names target(s) outside the population: "
                      f"{sorted(D - P)}")
    if not D <= F:
        raise Refusal("DETERMINISM_INVALID",
                      f"policy names unfloored target(s): {sorted(D - F)}")
    return sorted(D)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-root")
    ap.add_argument("--floors", type=Path, required=True,
                    help="target-keyed schema-v2 floor registry")
    ap.add_argument("--determinism-policy", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    # Declared ONLY so they can be refused. The legacy interface has a
    # different authority model and the two cannot be blended; there is no
    # precedence rule, because a precedence rule is a blend.
    ap.add_argument("--from-log", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--update-floors", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        if args.from_log is not None:
            raise Refusal("EVIDENCE_CONTEXT_INVALID",
                          "--from-log belongs to the legacy label/stdout "
                          "authority model and cannot be combined with an "
                          "evidence root. Neither takes precedence.")
        if args.update_floors:
            raise Refusal("EVIDENCE_CONTEXT_INVALID",
                          "the plan-aware path is READ-ONLY against floors. "
                          "Floor mutation belongs to the legacy writer, which "
                          "is held separately under WF-2R.")

        root = resolve_root(args.evidence_root, os.environ.get(EVIDENCE_ENV))
        _log, status_path, manifest_path = require_siblings(root)
        status = read_status(status_path)

        if status != 0:
            # Diagnostic parsing is permitted; adjudication is not. The
            # findings buckets are OMITTED, never emitted empty: an empty
            # list says "checked and found none", which would be false.
            diagnostic = None
            try:
                doc = json.loads(manifest_path.read_text(encoding="utf-8"))
                slots = doc.get("slots", [])
                diagnostic = {
                    "completed": sum(1 for s in slots
                                     if s.get("execution_state") == "COMPLETED"),
                    "failed": [s["make_target"] for s in slots
                               if s.get("execution_state") == "FAILED"],
                    "not_started": sum(1 for s in slots
                                       if s.get("execution_state") == "NOT_STARTED"),
                }
            except (OSError, ValueError, KeyError):
                diagnostic = None
            payload = {
                "aggregate_status": status, "admissible": False,
                "adjudicated": False,
                "refusal": {"code": "aggregate_incomplete",
                            "message": f"the aggregate exited {status}; floors "
                                       f"cannot be judged against a run that "
                                       f"did not complete"},
            }
            if diagnostic is not None:
                payload["diagnostic"] = diagnostic
            print(json.dumps(payload, indent=2) if args.json
                  else f"REFUSED (aggregate_incomplete): aggregate exited "
                       f"{status}; no floor adjudication. {diagnostic}")
            return 2

        plan, plan_digest = load_canonical_plan()
        slots = load_manifest(manifest_path, root, plan, plan_digest)
        counts = reconcile(slots, plan)

        floors = load_floors(args.floors)
        P = {e["make_target"] for e in plan}
        F = set(floors)
        orphans = sorted(F - P)
        if orphans:
            raise Refusal("ORPHAN_FLOOR",
                          f"floor(s) for target(s) not in the population: "
                          f"{orphans}. This fails closed; it is not a fallen "
                          f"floor and must never read as a pass.")
        determinism = load_determinism(args.determinism_policy, P, F)

    except Refusal as r:
        payload = {"aggregate_status": None, "admissible": False,
                   "adjudicated": False,
                   "refusal": {"code": r.code, "message": r.message}}
        print(json.dumps(payload, indent=2) if args.json
              else f"REFUSED ({r.code}): {r.message}")
        return 2

    # ── 9. findings. Every count came through reconcile(), so each is
    #      bound to the plan's exact result_label for that ordinal. Nothing
    #      is read from run.log. ──
    fallen = [f"{t}: {floors[t]} → {counts[t]} ({counts[t] - floors[t]})"
              for t in sorted(floors) if counts[t] < floors[t]]
    unfloored = sorted(P - F)

    payload = {
        "aggregate_status": 0, "admissible": True, "adjudicated": True,
        "population": len(P), "floored": len(F),
        "plan_digest": plan_digest,
        "counts": counts, "floors": floors,
        "fallen": fallen,
        "unfloored_targets": unfloored,
        "determinism_policy": determinism,
        "determinism_executed": False,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Assertion floors — {len(P)} targets, {len(F)} floored, "
              f"{sum(counts.values())} assertions")
        if fallen:
            print("\n  COVERAGE FELL:")
            for line in fallen:
                print(f"    - {line}")
        if unfloored:
            print(f"\n  UNFLOORED TARGETS — {len(unfloored)} population "
                  f"member(s) carry no floor. Absence of a floor is NOT zero "
                  f"and is NOT a pass; these await adjudication:")
            for t in unfloored:
                print(f"    - {t}")
        if not fallen:
            print("\n  PASS: every floored target met its floor.")
    return 1 if fallen else 0


if __name__ == "__main__":
    sys.exit(main())
