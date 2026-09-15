"""Destructive calibration for the Unified Hunter runner — SHADOW.

Written against the runner's **frozen contract**, not against its code. The
expectations here come from what the component is required to do — the plan
is the population authority, SC-1 orders the two states, the runner does not
own the aggregate evidence — and each one is checked by injecting the defect
and proving the runner refuses it. A component that has only been shown
working input has been demonstrated, not tested.

THE RUNNER IS NEVER EXECUTED AGAINST THE REAL POPULATION HERE, and that is a
safety property rather than a convenience. `uh_runner.main()` takes no
required arguments: `--plan` defaults to the canonical 78-target plan and
`--evidence-root` falls back to a fresh temp directory. So running the module
with no arguments traverses all 78 targets for real, invoking `make` 78
times. Every case below drives `classify`, `check_makeflags`, `load_plan`,
`evidence_root`, `publish` and `run` directly, and `run` is given a stub in
place of `subprocess` so that fail-fast ordering is measured without a single
real target executing.

Fixtures derive from the canonical plan rather than inventing its contents.
`INC-2026-09-14-18` was a fixture that fabricated `result_label` and so
proved the gate accepted a field the contract turns on; the rule taken from
it is that anything the contract binds must come from the authority, never
from this file's imagination. The awkward labels used to test punctuation
are *found* in the plan here, not typed out.

The MAKEFLAGS spellings are the ones measured from GNU Make 4.3 and recorded
in the runner's own header — `make -k` yields `MAKEFLAGS=[k]`, undashed.
`INC-2026-09-14-15` was a guard written from intuition against `-k`, which
matches nothing, and it reported clean while keep-going, ignore-errors and
dry-run all passed through. The list below is written from the measured
forms, and a separate case asserts the module's tables do not exceed them.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from scripts.security import uh_runner as R            # noqa: E402

PLAN = REPO / "scripts" / "security" / "uh_execution_plan.json"

passed = 0
failed = 0
executed: list[str] = []
EXPECTED_SCENARIOS = 21


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


class _Stub:
    """Swap a module attribute for the duration of a block."""

    def __init__(self, module, **values):
        self.module, self.values, self.saved = module, values, {}

    def __enter__(self):
        for k, v in self.values.items():
            self.saved[k] = getattr(self.module, k)
            setattr(self.module, k, v)
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            setattr(self.module, k, v)


def refusal_of(fn, *args, **kwargs):
    """(code, message) if `fn` refuses, else None."""
    try:
        fn(*args, **kwargs)
    except R.Refusal as r:
        return r.code, r.message
    return None


def write_plan(root: Path, name: str, doc) -> Path:
    """Each fixture gets its OWN path.

    The first version of this helper wrote every case to one `plan.json`.
    The case list is built eagerly, so by the time the loop ran, all but
    the last fixture had been overwritten and seven refusal cases were
    silently exercising one document. A mutation that disabled the schema
    check survived the whole suite, which is how it was found — the same
    shape as INC-2026-09-14-18: a fixture that does not contain what its
    name says it contains.
    """
    path = root / f"{name}.json"
    path.write_text(
        doc if isinstance(doc, str) else json.dumps(doc), encoding="utf-8")
    return path


def canonical_targets() -> list[dict]:
    return json.loads(PLAN.read_bytes().decode("utf-8"))["targets"]


# ── the plan is the population authority ─────────────────────────────

def test_the_canonical_plan_hashes_its_own_raw_bytes() -> None:
    """The digest is over bytes as they sit on disk — recomputed here
    independently rather than read back out of the loader."""
    scenario("plan digest is the sha256 of the file")
    entries, digest = R.load_plan(PLAN)
    independent = hashlib.sha256(PLAN.read_bytes()).hexdigest()
    check("digest is sha256 of the raw file", digest == independent,
          f"{digest} != {independent}")
    check("entries are the plan's targets, in order",
          entries == canonical_targets())
    check("the population is non-empty", len(entries) > 0)


def test_one_changed_byte_is_a_different_plan() -> None:
    """Any byte change is a new digest. No canonicalisation, no hashing
    of a parsed structure — otherwise two different files could claim
    one identity."""
    scenario("a byte change changes the digest")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        raw = PLAN.read_bytes()
        same = root / "same.json"
        same.write_bytes(raw)
        _, d_same = R.load_plan(same)
        check("identical bytes, identical digest",
              d_same == hashlib.sha256(raw).hexdigest())
        nudged = root / "nudged.json"
        nudged.write_bytes(raw + b" ")
        _, d_nudged = R.load_plan(nudged)
        check("one trailing space is a different plan", d_nudged != d_same)


def test_an_unusable_plan_is_refused_not_worked_around() -> None:
    """Eight ways to be unusable. Each must REFUSE, because a runner that
    repairs its own authority is no longer traversing one."""
    scenario("plan refusals")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = canonical_targets()[:2]

        cases = [
            ("missing file", root / "nope.json"),
            ("not JSON", write_plan(root, "notjson", "{not json")),
            ("wrong schema", write_plan(
                root, "schema", {"schema": "kai.uh-plan/v99",
                                 "targets": good})),
            ("no targets key", write_plan(
                root, "notargets", {"schema": R.PLAN_SCHEMA})),
            ("empty targets", write_plan(
                root, "empty", {"schema": R.PLAN_SCHEMA, "targets": []})),
            ("entry is not an object", write_plan(
                root, "notobject",
                {"schema": R.PLAN_SCHEMA, "targets": ["test-a"]})),
            ("entry has no make_target", write_plan(
                root, "notarget", {"schema": R.PLAN_SCHEMA,
                                   "targets": [{"result_label": "A Tests"}]})),
            ("entry has no result_label", write_plan(
                root, "nolabel", {"schema": R.PLAN_SCHEMA,
                                  "targets": [{"make_target": "test-a"}]})),
            ("duplicate make_target", write_plan(
                root, "duplicate",
                {"schema": R.PLAN_SCHEMA, "targets": [good[0], good[0]]})),
        ]
        # Every fixture must be a distinct file, or the cases overwrite one
        # another and the loop measures the last one nine times.
        paths = [p for _, p in cases]
        check("each refusal case has its own fixture",
              len(set(paths)) == len(paths), str(paths))
        for name, path in cases:
            got = refusal_of(R.load_plan, path)
            check(f"{name} is refused", got is not None)
            if got:
                check(f"{name} refuses as PLAN_INVALID",
                      got[0] == "PLAN_INVALID", got[0])

        # Known-negative: the canonical plan must NOT refuse, or every
        # case above passes for the wrong reason.
        check("the canonical plan is accepted",
              refusal_of(R.load_plan, PLAN) is None)


# ── the result contract ──────────────────────────────────────────────

def awkward_labels() -> dict:
    """Labels the canonical plan actually carries that contain regex
    punctuation. Found, not typed — a hand-written label is a second
    declaration of something the plan already states."""
    labels = [e["result_label"] for e in canonical_targets()]
    return {
        "parenthesised": next((l for l in labels if "(" in l), None),
        "slashed": next((l for l in labels if "/" in l), None),
    }


def test_punctuation_in_a_real_label_decides_nothing() -> None:
    """`re.escape` is not optional. Interpolating `Service identity
    (ed25519) tests` or `/observe_turn identity slice` into a pattern
    turns punctuation into syntax, and punctuation must never decide
    membership."""
    scenario("real punctuated labels match literally")
    found = awkward_labels()
    for kind, label in found.items():
        check(f"the plan still carries a {kind} label", label is not None)
        if label is None:
            continue
        line = f"{label}: 12 passed, 0 failed"
        m = R.result_pattern(label).match(line)
        check(f"{kind} label matches its own result line", m is not None, line)
        if m:
            check(f"{kind} label reads its counts", m.groups() == ("12", "0"))
        check(f"{kind} label is not treated as a pattern",
              R.result_pattern(label).match("Service identity Xed25519Y tests: "
                                            "12 passed, 0 failed") is None)


def test_the_result_line_is_anchored_at_both_ends() -> None:
    """No leading text, no trailing text. An unanchored contract lets a
    nested subject's tally be read as its parent's."""
    scenario("result line anchoring")
    label = "Alpha Tests"
    ok = f"{label}: 3 passed, 0 failed"
    check("the exact line matches", R.result_pattern(label).match(ok))
    for bad in (f"  {label}: 3 passed, 0 failed",
                f"see {label}: 3 passed, 0 failed",
                f"{label}: 3 passed, 0 failed (cached)",
                f"{label}: 3 passed 0 failed",
                f"{label}: three passed, 0 failed"):
        check(f"rejected: {bad!r}", R.result_pattern(label).match(bad) is None)


def test_a_claim_is_separable_from_a_valid_result() -> None:
    """MALFORMED is a real state only because the claim and the grammar
    are matched separately. `Alpha Tests: seventeen passed` is the target
    asserting its result and getting it wrong — a contract defect, not
    silence."""
    scenario("claim versus grammar")
    label = "Alpha Tests"
    claim = f"{label}: seventeen passed"
    check("a malformed line still CLAIMS the label",
          R.label_pattern(label).match(claim) is not None)
    check("and does not parse as a result",
          R.result_pattern(label).match(claim) is None)
    check("an indented claim is not the parent's claim",
          R.label_pattern(label).match(f"    {claim}") is None)


# ── SC-1: execution status is adjudicated first ──────────────────────

def test_a_nonzero_exit_is_failed_whatever_the_output_said() -> None:
    """The measured case is plan member 27. `test-gate-registry` exited 2
    AND printed a perfectly well-formed `Gate Registry Tests: 81 passed,
    1 failed`. RESOLVED and FAILED at once: the tally is diagnostic, it
    does not replace the execution state, and it raises no refusal."""
    scenario("SC-1 — non-zero exit dominates")
    label = "Gate Registry Tests"
    state, observed, result, refusal = R.classify(
        2, f"{label}: 81 passed, 1 failed\n", label)
    check("FAILED", state == "FAILED", state)
    check("observation still RESOLVED", observed == "RESOLVED", observed)
    check("the tally is retained as diagnostic",
          result == {"passed": 81, "failed": 1}, str(result))
    check("no refusal is raised", refusal is None, str(refusal))


def test_absence_after_a_nonzero_exit_is_not_a_contract_defect() -> None:
    """`test-container-proof-harness` died on an uncaught 120-second
    timeout before reaching the epilogue that prints its tally. Calling
    that RESULT_CONTRACT_CONFLICT would say the plan's contract is wrong
    when the truth is the subject failed before it could report."""
    scenario("SC-1 — absence after failure is not a conflict")
    for out, expect in ((""                      , "ABSENT"),
                        ("Alpha Tests: nope\n"   , "MALFORMED"),
                        ("Alpha Tests: 1 passed, 0 failed\n"
                         "Alpha Tests: 2 passed, 0 failed\n", "AMBIGUOUS")):
        state, observed, _, refusal = R.classify(1, out, "Alpha Tests")
        check(f"{expect}: FAILED", state == "FAILED", state)
        check(f"{expect}: observed as {expect}", observed == expect, observed)
        check(f"{expect}: raises no contract refusal", refusal is None,
              str(refusal))


def test_a_green_target_must_satisfy_its_contract() -> None:
    """The contract is mandatory only for a target that exits 0 — and
    there it is mandatory."""
    scenario("SC-1 — the contract binds exit-zero targets")
    label = "Alpha Tests"
    for out, expect in ((""                                   , "ABSENT"),
                        (f"{label}: seventeen passed\n"       , "MALFORMED"),
                        (f"{label}: 1 passed, 0 failed\n"
                         f"{label}: 2 passed, 0 failed\n"     , "AMBIGUOUS"),
                        (f"{label}: 1 passed, 0 failed\n"
                         f"{label}: and another thing\n"      , "AMBIGUOUS")):
        state, observed, result, refusal = R.classify(0, out, label)
        check(f"{expect}: FAILED", state == "FAILED", state)
        check(f"{expect}: observed as {expect}", observed == expect, observed)
        check(f"{expect}: no result is published", result is None, str(result))
        check(f"{expect}: RESULT_CONTRACT_CONFLICT",
              refusal == "RESULT_CONTRACT_CONFLICT", str(refusal))


def test_a_green_process_contradicting_its_tally_fails_closed() -> None:
    scenario("SC-1 — exit 0 with failures in the tally")
    label = "Alpha Tests"
    state, observed, result, refusal = R.classify(
        0, f"{label}: 9 passed, 1 failed\n", label)
    check("FAILED", state == "FAILED", state)
    check("RESOLVED", observed == "RESOLVED", observed)
    check("the contradicting tally is kept",
          result == {"passed": 9, "failed": 1}, str(result))
    check("RESULT_CONTRACT_CONFLICT",
          refusal == "RESULT_CONTRACT_CONFLICT", str(refusal))


def test_the_only_completed_case() -> None:
    """The known-negative for everything above: if nothing can be
    COMPLETED, every rejection case passes for the wrong reason."""
    scenario("SC-1 — the one completed case")
    label = "Alpha Tests"
    state, observed, result, refusal = R.classify(
        0, f"noise\n{label}: 42 passed, 0 failed\nmore noise\n", label)
    check("COMPLETED", state == "COMPLETED", state)
    check("RESOLVED", observed == "RESOLVED", observed)
    check("counts read", result == {"passed": 42, "failed": 0}, str(result))
    check("no refusal", refusal is None, str(refusal))


def test_a_nested_tally_cannot_be_admitted_as_a_suite() -> None:
    """A nested subject that prints a tally of its own was admitted as
    though it were a suite. Line-start anchoring is what stops it."""
    scenario("SC-1 — nested output cannot claim the label")
    label = "Alpha Tests"
    state, observed, _, refusal = R.classify(
        0, f"    {label}: 5 passed, 0 failed\n", label)
    check("the indented tally is not seen", observed == "ABSENT", observed)
    check("so the green target breaks its contract", state == "FAILED", state)
    check("RESULT_CONTRACT_CONFLICT",
          refusal == "RESULT_CONTRACT_CONFLICT", str(refusal))


# ── the environment the evidence was produced in ─────────────────────

# Written from the forms MEASURED off GNU Make 4.3, not from the dashed
# spellings a person would reach for. Single-letter options are packed
# into the FIRST word with no leading dash.
MEASURED_HOSTILE = [
    ("k", "make -k"),
    ("k", "make --keep-going"),
    ("i", "make -i"),
    ("n", "make -n"),
    ("k -j2 --jobserver-auth=3,4", "make -kj2"),
    (" -j4 --jobserver-auth=3,4", "make -j4"),
    ("--keep-going", "long spelling, unpacked"),
    ("--ignore-errors", "long spelling, unpacked"),
    ("--dry-run", "long spelling, unpacked"),
    ("--just-print", "long spelling, unpacked"),
    ("--recon", "long spelling, unpacked"),
    ("--jobs=4", "long spelling with an argument"),
]

BENIGN = ["", "   ", "rR", "--warn-undefined-variables",
          "--output-sync=target", "-l4", "w"]


def test_every_measured_hostile_make_mode_is_refused() -> None:
    """The known-positive for INC-2026-09-14-15. A guard written against
    `-k` matches nothing at all, which is the worst failure a guard can
    have: it reports clean while five of nine hostile modes pass through."""
    scenario("hostile MAKEFLAGS refused")
    for flags, how in MEASURED_HOSTILE:
        got = refusal_of(R.check_makeflags, {"MAKEFLAGS": flags})
        check(f"{how} -> MAKEFLAGS={flags!r} refused", got is not None, flags)
        if got:
            check(f"{flags!r} refuses as EVIDENCE_CONTEXT_INVALID",
                  got[0] == "EVIDENCE_CONTEXT_INVALID", got[0])


def test_benign_make_modes_are_not_refused() -> None:
    """The known-negative. A guard that refuses everything passes every
    case above and makes the runner unusable."""
    scenario("benign MAKEFLAGS accepted")
    for flags in BENIGN:
        got = refusal_of(R.check_makeflags, {"MAKEFLAGS": flags})
        check(f"MAKEFLAGS={flags!r} accepted", got is None, str(got))
    check("an absent MAKEFLAGS is accepted",
          refusal_of(R.check_makeflags, {}) is None)


def test_the_guard_claims_no_mode_it_cannot_refuse() -> None:
    """Consistency, in the other direction: every mode the module's own
    tables declare hostile must actually be refused in the packed and the
    long spelling. This does not replace the measured list above — it
    stops the tables and the behaviour drifting apart."""
    scenario("declared hostile modes are all enforced")
    for letter in R.HOSTILE_LETTERS:
        check(f"packed {letter!r} refused",
              refusal_of(R.check_makeflags, {"MAKEFLAGS": letter}) is not None)
        check(f"dashed -{letter} refused",
              refusal_of(R.check_makeflags,
                         {"MAKEFLAGS": f"x -{letter}"}) is not None)
    for long in R.HOSTILE_LONG:
        check(f"{long} refused",
              refusal_of(R.check_makeflags, {"MAKEFLAGS": long}) is not None)


def test_the_evidence_root_is_the_run_identity() -> None:
    """One root; siblings derive from it. No fixed `/tmp` fallback,
    because a fixed path is how two runs come to share one piece of
    evidence."""
    scenario("evidence root")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        check("an absolute existing directory is accepted",
              R.evidence_root(str(root)) == root)

        got = refusal_of(R.evidence_root, str(root / "nope"))
        check("a path that is not a directory is refused", got is not None)
        if got:
            check("refused as EVIDENCE_CONTEXT_INVALID",
                  got[0] == "EVIDENCE_CONTEXT_INVALID", got[0])

        cwd = os.getcwd()
        try:
            os.chdir(tmp)
            (root / "rel").mkdir()
            got = refusal_of(R.evidence_root, "rel")
            check("a relative directory is refused", got is not None)
        finally:
            os.chdir(cwd)

    minted = R.evidence_root(None)
    try:
        check("an unset root is minted, absolute and fresh",
              minted.is_dir() and minted.is_absolute(), str(minted))
        second = R.evidence_root(None)
        try:
            check("two unset runs never share a root", minted != second)
        finally:
            second.rmdir()
    finally:
        minted.rmdir()


# ── the evidence-ownership boundary (WF-2's, not the runner's) ───────

def test_the_runner_writes_only_its_manifest() -> None:
    """INC-2026-09-14-16. The runner wrote run.log, run.log.status and an
    `aggregate_status` field — all three belong to the workflow shell's
    `make test-uh 2>&1 | tee $ROOT/run.log` under pipefail. The runner's
    process exit and the top-level Make status are two different
    observations; naming them alike invites a later reader to treat
    either as completion authority."""
    scenario("evidence ownership")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        payload = {"schema": R.RUN_SCHEMA, "plan_digest": "d",
                   "plan_path": "p", "plan_scope": "repository",
                   "evidence_root": str(root), "population": 0, "slots": []}
        final = R.publish(root, payload)
        names = sorted(p.name for p in root.iterdir())
        check("only results.json is written", names == ["results.json"],
              str(names))
        check("no partial file survives",
              not (root / "results.json.partial").exists())
        check("the runner never writes the aggregate log",
              "run.log" not in names and "run.log.status" not in names)
        written = json.loads(final.read_text(encoding="utf-8"))
        check("the manifest carries no aggregate_status",
              "aggregate_status" not in written, str(sorted(written)))
        check("and round-trips what it was given", written == payload)


def test_the_manifest_schema_is_declared() -> None:
    scenario("manifest schema")
    check("the run schema is distinct from the plan schema",
          R.RUN_SCHEMA != R.PLAN_SCHEMA)
    check("the run schema is versioned", R.RUN_SCHEMA.endswith("/v1"),
          R.RUN_SCHEMA)


# ── traversal: serial, fail-fast, and nothing vanishes ───────────────

class _FakeProc:
    def __init__(self, returncode: int, stdout: str):
        self.returncode, self.stdout, self.stderr = returncode, stdout, ""


class _FakeSubprocess:
    """Stands in for `subprocess` so traversal order is measured without
    a single real `make` invocation. The real module would run 78
    targets; that is what this suite exists to avoid."""

    def __init__(self, script):
        self.script, self.calls = script, []

    def run(self, argv, **kwargs):
        self.calls.append(list(argv))
        return self.script(argv[1])


def three_entries() -> list[dict]:
    return [{"make_target": f"test-{n}", "result_label": f"{n.title()} Tests"}
            for n in ("alpha", "beta", "gamma")]


def test_fail_fast_leaves_the_remainder_not_started() -> None:
    """Serial and fail-fast. The targets after a failure must remain
    NOT_STARTED rather than vanishing from the population — a member that
    disappears is a shrinking denominator, and progress and absence
    become indistinguishable."""
    scenario("fail-fast traversal")
    entries = three_entries()

    def script(target):
        if target == "test-alpha":
            return _FakeProc(0, "Alpha Tests: 1 passed, 0 failed\n")
        if target == "test-beta":
            return _FakeProc(2, "Beta Tests: 3 passed, 1 failed\n")
        raise AssertionError("test-gamma must never be executed")

    fake = _FakeSubprocess(script)
    with _Stub(R, subprocess=fake), contextlib.redirect_stdout(io.StringIO()):
        slots = R.run(entries, {})

    check("the population is preserved", len(slots) == len(entries))
    check("positions are 0-based and in plan order",
          [s["position"] for s in slots] == [0, 1, 2],
          str([s["position"] for s in slots]))
    check("make was invoked only up to the failure",
          fake.calls == [["make", "test-alpha"], ["make", "test-beta"]],
          str(fake.calls))
    check("alpha COMPLETED", slots[0]["execution_state"] == "COMPLETED",
          slots[0]["execution_state"])
    check("beta FAILED", slots[1]["execution_state"] == "FAILED",
          slots[1]["execution_state"])
    check("beta keeps its exit code", slots[1]["exit_code"] == 2)
    check("gamma NOT_STARTED", slots[2]["execution_state"] == "NOT_STARTED",
          slots[2]["execution_state"])
    check("gamma NOT_OBSERVED",
          slots[2]["result_observation"] == "NOT_OBSERVED")
    check("gamma has no exit code", slots[2]["exit_code"] is None)
    check("gamma carries its planned label",
          slots[2]["result_label"] == "Gamma Tests")


def test_target_output_is_forwarded_not_swallowed() -> None:
    """The runner captures only to classify, then writes through, so the
    outer `tee` remains the single producer of the canonical log. A
    runner that swallowed output would silently become that producer."""
    scenario("output forwarding")
    entries = three_entries()[:1]
    body = "Alpha Tests: 1 passed, 0 failed\nsome diagnostic detail\n"

    fake = _FakeSubprocess(lambda _t: _FakeProc(0, body))
    buf = io.StringIO()
    with _Stub(R, subprocess=fake), contextlib.redirect_stdout(buf):
        R.run(entries, {})
    check("every line the target printed reaches stdout",
          body in buf.getvalue(), repr(buf.getvalue()[:120]))


def test_the_plan_causes_the_traversal() -> None:
    """Membership is defined in the plan and nowhere else — not by
    stdout, not by a filesystem glob. A target that prints nothing is
    still a member; a tally for a name the plan does not carry creates
    nothing."""
    scenario("the plan defines membership")
    entries = three_entries()

    def script(_target):
        return _FakeProc(0, "Delta Tests: 99 passed, 0 failed\n")

    fake = _FakeSubprocess(script)
    with _Stub(R, subprocess=fake), contextlib.redirect_stdout(io.StringIO()):
        slots = R.run(entries, {})

    check("stdout did not add a member", len(slots) == 3)
    check("no slot is named for the unplanned tally",
          all(s["result_label"] != "Delta Tests" for s in slots))
    check("the first planned target broke its own contract",
          slots[0]["execution_state"] == "FAILED"
          and slots[0]["refusal"] == "RESULT_CONTRACT_CONFLICT",
          f"{slots[0]['execution_state']}/{slots[0]['refusal']}")
    check("and the rest stayed NOT_STARTED",
          [s["execution_state"] for s in slots[1:]]
          == ["NOT_STARTED", "NOT_STARTED"])


def run() -> None:
    test_the_canonical_plan_hashes_its_own_raw_bytes()
    test_one_changed_byte_is_a_different_plan()
    test_an_unusable_plan_is_refused_not_worked_around()
    test_punctuation_in_a_real_label_decides_nothing()
    test_the_result_line_is_anchored_at_both_ends()
    test_a_claim_is_separable_from_a_valid_result()
    test_a_nonzero_exit_is_failed_whatever_the_output_said()
    test_absence_after_a_nonzero_exit_is_not_a_contract_defect()
    test_a_green_target_must_satisfy_its_contract()
    test_a_green_process_contradicting_its_tally_fails_closed()
    test_the_only_completed_case()
    test_a_nested_tally_cannot_be_admitted_as_a_suite()
    test_every_measured_hostile_make_mode_is_refused()
    test_benign_make_modes_are_not_refused()
    test_the_guard_claims_no_mode_it_cannot_refuse()
    test_the_evidence_root_is_the_run_identity()
    test_the_runner_writes_only_its_manifest()
    test_the_manifest_schema_is_declared()
    test_fail_fast_leaves_the_remainder_not_started()
    test_target_output_is_forwarded_not_swallowed()
    test_the_plan_causes_the_traversal()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run()
    print("=" * 60)
    print(f"UH Runner Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
