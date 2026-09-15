"""Destructive calibration for the plan-aware floor consumer — SHADOW.

Every case here injects a defect and proves the gate refuses it. A gate that
has only ever been shown working evidence has not been tested; it has been
demonstrated.

Synthetic inputs only. Each case builds a complete evidence root in a
temporary directory — run.log, run.log.status, results.json — and a
target-keyed floor registry to match. Nothing reads the repository's live
counts, floors or CI artefacts, because a fixture that tracks live state
stops being a fixture.

The one exception is the canonical plan, which the gate hashes by design:
the manifest must carry that plan's real raw-byte digest or the run is not
about the plan we ship. Fixtures therefore read the canonical plan's target
list and digest and build manifests against it.

The critical case is H. An authoritative non-zero status with an
all-green-looking manifest must refuse. That is the known-negative for
INC-2026-09-14-16: if manifest evidence could ever substitute for a missing
or non-zero authoritative status, the authority boundary we corrected would
be back.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "scripts" / "security" / "uh_floor_gate.py"
PLAN = REPO / "scripts" / "security" / "uh_execution_plan.json"

passed = 0
failed = 0
executed: list[str] = []
# The denominator this gate is registered under. Declared here, beside
# the assertion that checks it, so the registry entry and its evidence
# cannot drift apart silently.
DECLARED_DENOMINATOR = (
    r"Assertion floors — \d+ targets, \d+ floored, \d+ assertions")

EXPECTED_SCENARIOS = 40


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def plan_pairs() -> tuple[list[tuple[str, str]], str]:
    """(make_target, result_label) PAIRS from the canonical plan.

    Deriving only the targets and inventing the labels is what made the
    first version of this suite a negative fixture the gate accepted: every
    "positive" case proved a weaker protocol than the one being shipped.
    The expected answer must come from the authority, whole -- not half
    from the authority and half from my imagination.
    """
    raw = PLAN.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    return ([(e["make_target"], e["result_label"]) for e in doc["targets"]],
            hashlib.sha256(raw).hexdigest())


PAIRS, PLAN_DIGEST = plan_pairs()
TARGETS = [t for t, _ in PAIRS]
LABELS = dict(PAIRS)
CANONICAL_PLAN_PATH = "scripts/security/uh_execution_plan.json"


def manifest(root: Path, *, targets=None, digest=None, scope="repository",
             schema="kai.uh-run/v1", states=None, evidence_root=None,
             observations=None, passes=None, labels=None, positions=None,
             exits=None, results=None, population=None,
             plan_path=None) -> dict:
    """A manifest whose defaults SATISFY the production contract.

    Labels default to the canonical plan's, so the known-positive is a real
    known-positive. Every override exists to build one specific negative.
    """
    targets = TARGETS if targets is None else targets
    slots = []
    for i, t in enumerate(targets):
        state = (states or {}).get(t, "COMPLETED")
        obs = (observations or {}).get(t, "RESOLVED" if state == "COMPLETED"
                                       else "NOT_OBSERVED")
        if results is not None and t in results:
            result = results[t]
        elif obs == "RESOLVED":
            result = {"passed": (passes or {}).get(t, 50), "failed": 0}
        else:
            result = None
        slots.append({"position": (positions or {}).get(t, i),
                      "make_target": t,
                      "result_label": (labels or {}).get(t, LABELS.get(t, "?")),
                      "execution_state": state, "result_observation": obs,
                      "exit_code": (exits or {}).get(
                          t, 0 if state == "COMPLETED" else 1),
                      "result": result, "refusal": None})
    return {"schema": schema,
            "plan_digest": PLAN_DIGEST if digest is None else digest,
            "plan_path": CANONICAL_PLAN_PATH if plan_path is None else plan_path,
            "plan_scope": scope,
            "evidence_root": str(evidence_root or root),
            "population": len(slots) if population is None else population,
            "slots": slots}


def floors_v2(targets, value=10, schema="kai.uh-floors/v2") -> dict:
    return {"schema": schema, "floors": {t: value for t in targets}}


def build(tmp: Path, *, status="0\n", man=None, omit=(), floors=None) -> Path:
    root = tmp / "root"
    root.mkdir(parents=True, exist_ok=True)
    if "run.log" not in omit:
        (root / "run.log").write_text("aggregate output\n", encoding="utf-8")
    if "run.log.status" not in omit:
        (root / "run.log.status").write_text(status, encoding="utf-8")
    if "results.json" not in omit:
        doc = manifest(root) if man is None else man(root)
        (root / "results.json").write_text(json.dumps(doc), encoding="utf-8")
    fl = tmp / "floors.json"
    fl.write_text(json.dumps(floors if floors is not None
                             else floors_v2(TARGETS)), encoding="utf-8")
    return root


def run_gate(root, floors_path=None, *extra, env_root=None) -> tuple[int, dict]:
    argv = [sys.executable, str(GATE), "--json"]
    if root is not None:
        argv += ["--evidence-root", str(root)]
    argv += ["--floors", str(floors_path)] if floors_path else []
    argv += list(extra)
    env = dict(os.environ)
    env.pop("KAI_UH_EVIDENCE_ROOT", None)
    if env_root:
        env["KAI_UH_EVIDENCE_ROOT"] = str(env_root)
    p = subprocess.run(argv, capture_output=True, text=True, cwd=str(REPO),
                       env=env)
    try:
        return p.returncode, json.loads(p.stdout)
    except ValueError:
        return p.returncode, {"_stdout": p.stdout, "_stderr": p.stderr}


def refused(name, code, root, floors_path=None, *extra, env_root=None):
    rc, out = run_gate(root, floors_path, *extra, env_root=env_root)
    got = out.get("refusal", {}).get("code")
    check(f"{name} refuses", rc == 2, f"rc={rc} out={out}")
    check(f"{name} -> {code}", got == code, f"got {got!r}")
    check(f"{name} publishes no findings",
          not any(k in out for k in ("fallen", "unfloored_targets")), str(out))
    check(f"{name} adjudicated=false", out.get("adjudicated") is False, str(out))


# ── A-C: the evidence-root interface ─────────────────────────────────

def test_root_absent():
    scenario("A-root-absent")
    with tempfile.TemporaryDirectory() as d:
        fl = Path(d) / "f.json"
        fl.write_text(json.dumps(floors_v2(TARGETS)), encoding="utf-8")
        refused("A no root", "EVIDENCE_CONTEXT_INVALID", None, fl)


def test_root_relative():
    scenario("B-root-relative")
    with tempfile.TemporaryDirectory() as d:
        fl = Path(d) / "f.json"
        fl.write_text(json.dumps(floors_v2(TARGETS)), encoding="utf-8")
        refused("B relative root", "EVIDENCE_CONTEXT_INVALID",
                Path("relative/root"), fl)


def test_cli_and_env_disagree():
    scenario("C-root-disagree")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        other = tmp / "other"
        other.mkdir()
        refused("C cli/env disagree", "EVIDENCE_CONTEXT_INVALID",
                root, tmp / "floors.json", env_root=other)


# ── D-F: the exact sibling set ───────────────────────────────────────

def _missing(letter, name):
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, omit=(name,))
        refused(f"{letter} missing {name}", "REQUIRED_INPUT_MISSING",
                root, tmp / "floors.json")


def test_missing_log():
    scenario("D-missing-log"); _missing("D", "run.log")


def test_missing_status():
    scenario("E-missing-status"); _missing("E", "run.log.status")


def test_missing_manifest():
    scenario("F-missing-manifest"); _missing("F", "results.json")


# ── G-I: status authority ────────────────────────────────────────────

def test_malformed_status():
    scenario("G-status-malformed")
    for raw in ("", "   \n", "two\nlines\n", "not-a-number\n"):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = build(tmp, status=raw)
            refused(f"G status {raw!r}", "STATUS_UNREADABLE",
                    root, tmp / "floors.json")


def test_nonzero_status_with_green_manifest():
    """THE known-negative for INC-2026-09-14-16.

    Every slot COMPLETED and RESOLVED, every floor met — and the
    authoritative sidecar says 2. If the manifest could override it, the
    authority boundary would be gone.
    """
    scenario("H-nonzero-vs-green-manifest")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, status="2\n")
        rc, out = run_gate(root, tmp / "floors.json")
        check("H all-green manifest cannot beat a non-zero status", rc == 2,
              f"rc={rc}")
        check("H names the aggregate, not erosion",
              out.get("refusal", {}).get("code") == "aggregate_incomplete",
              str(out.get("refusal")))
        check("H admissible=false", out.get("admissible") is False, str(out))
        check("H adjudicated=false", out.get("adjudicated") is False, str(out))
        for bucket in ("fallen", "missing", "unrecorded", "drifted",
                       "unfloored_targets"):
            check(f"H omits {bucket} rather than emitting it empty",
                  bucket not in out, str(out.keys()))
        check("H may still report diagnostically", "diagnostic" in out, str(out))


def test_zero_status_with_failed_manifest():
    scenario("I-zero-vs-failed-manifest")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        bad = TARGETS[5]
        root = build(tmp, status="0\n", man=lambda r: manifest(
            r, states={bad: "FAILED",
                       **{t: "NOT_STARTED" for t in TARGETS[6:]}}))
        refused("I zero status vs failed slots", "MANIFEST_INVALID",
                root, tmp / "floors.json")


# ── J-P: manifest integrity ──────────────────────────────────────────

def test_evidence_root_mismatch():
    scenario("J-root-mismatch")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, evidence_root="/somewhere/else"))
        refused("J manifest root mismatch", "EVIDENCE_CONTEXT_INVALID",
                root, tmp / "floors.json")


def test_external_plan_scope():
    scenario("K-external-plan")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(r, scope="external"))
        refused("K external plan scope", "PLAN_INVALID",
                root, tmp / "floors.json")


def test_wrong_plan_digest():
    scenario("L-wrong-digest")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(r, digest="0" * 64))
        refused("L wrong plan digest", "PLAN_INVALID",
                root, tmp / "floors.json")


def test_duplicate_target():
    scenario("M-duplicate-target")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        dupe = TARGETS[:-1] + [TARGETS[0]]
        root = build(tmp, man=lambda r: manifest(r, targets=dupe))
        refused("M duplicate target", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_missing_population_target():
    scenario("N-missing-target")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(r, targets=TARGETS[:-1]))
        refused("N short population", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_foreign_target():
    scenario("O-foreign-target")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        foreign = TARGETS[:-1] + ["test-not-in-the-plan"]
        root = build(tmp, man=lambda r: manifest(r, targets=foreign))
        refused("O foreign target", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_reordered_targets():
    scenario("P-reordered")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        swapped = list(TARGETS)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        root = build(tmp, man=lambda r: manifest(r, targets=swapped))
        refused("P reordered targets", "MANIFEST_INVALID",
                root, tmp / "floors.json")


# ── Q-T: floors ──────────────────────────────────────────────────────

def test_schema_v1_floors_refused():
    scenario("Q-schema-v1")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        v1 = tmp / "v1.json"
        v1.write_text(json.dumps({"note": "legacy",
                                  "floors": {"Alpha Tests": 10}}),
                      encoding="utf-8")
        refused("Q label-keyed v1 floors", "FLOORS_INVALID", root, v1)


def test_orphan_floor():
    scenario("R-orphan-floor")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        orph = tmp / "orphan.json"
        f = floors_v2(TARGETS)
        f["floors"]["test-removed-last-year"] = 12
        orph.write_text(json.dumps(f), encoding="utf-8")
        refused("R orphan floor", "ORPHAN_FLOOR", root, orph)


def test_unfloored_targets_reported():
    scenario("S-unfloored")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        partial = tmp / "partial.json"
        floored = TARGETS[:-17]
        partial.write_text(json.dumps(floors_v2(floored)), encoding="utf-8")
        rc, out = run_gate(root, partial)
        check("S adjudicates", out.get("adjudicated") is True, str(out)[:200])
        # THE ASSERTION THAT WAS MISSING. The first version checked the
        # report and never the return code, so the gate printed "NOT a pass"
        # and exited 0. CI reads the exit code, not the prose.
        check("S EXITS RED — unfloored targets are a finding", rc == 1,
              f"rc={rc}; an unwatched population member cannot exit green")
        check("S is admissible and adjudicated, not a refusal",
              out.get("admissible") is True and out.get("adjudicated") is True,
              str(out)[:160])
        check("S counts it as a finding",
              out.get("findings", {}).get("unfloored") == 17,
              str(out.get("findings")))
        check("S is not a floor erosion", out.get("fallen") == [],
              str(out.get("fallen")))
        check("S reports exactly the 17 unfloored",
              len(out.get("unfloored_targets", [])) == 17,
              str(len(out.get("unfloored_targets", []))))
        check("S assigns them no floor",
              all(t not in out.get("floors", {})
                  for t in out["unfloored_targets"]), "")
        check("S does not synthesise a zero",
              all(v > 0 for v in out.get("floors", {}).values()), "")


def test_bad_floor_values():
    scenario("T-bad-floor-values")
    for label, value in (("zero", 0), ("negative", -3), ("bool", True),
                         ("string", "12")):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = build(tmp)
            bad = tmp / "bad.json"
            f = floors_v2(TARGETS)
            f["floors"][TARGETS[0]] = value
            bad.write_text(json.dumps(f), encoding="utf-8")
            refused(f"T {label} floor", "FLOORS_INVALID", root, bad)


# ── U-V: interface exclusion ─────────────────────────────────────────

def test_root_plus_from_log_refused():
    scenario("U-root-plus-fromlog")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        refused("U root + --from-log", "EVIDENCE_CONTEXT_INVALID",
                root, tmp / "floors.json", "--from-log", "/tmp/x.log")


def test_root_plus_update_floors_refused():
    scenario("V-root-plus-update")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        refused("V root + --update-floors", "EVIDENCE_CONTEXT_INVALID",
                root, tmp / "floors.json", "--update-floors")


# ── W: the clean path, so refusal is not unconditional ───────────────

def test_clean_run_adjudicates():
    scenario("W-clean-adjudicates")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        rc, out = run_gate(root, tmp / "floors.json")
        check("W a clean run passes", rc == 0, f"rc={rc} {str(out)[:200]}")
        check("W admissible", out.get("admissible") is True, str(out)[:160])
        check("W adjudicated", out.get("adjudicated") is True, str(out)[:160])
        check("W findings buckets present when adjudicated",
              "fallen" in out and "unfloored_targets" in out, str(out.keys()))
        check("W population is the plan's", out.get("population") == len(TARGETS),
              str(out.get("population")))


def test_fallen_floor_is_a_finding_not_a_refusal():
    scenario("W2-fallen-floor")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        high = tmp / "high.json"
        f = floors_v2(TARGETS)
        f["floors"][TARGETS[3]] = 9999
        high.write_text(json.dumps(f), encoding="utf-8")
        rc, out = run_gate(root, high)
        check("W2 a fallen floor exits 1, not 2", rc == 1, f"rc={rc}")
        check("W2 it is adjudicated, not refused",
              out.get("adjudicated") is True, str(out)[:160])
        check("W2 the target is named",
              any(TARGETS[3] in line for line in out.get("fallen", [])),
              str(out.get("fallen")))


# ── X-AH: the target-bound result contract ───────────────────────────
#
# These exist because the first version of this suite did NOT have them,
# and its own positive fixture carried fabricated result labels that the
# gate adjudicated cleanly. X is that exact recovered defect.

def test_result_label_mismatch():
    """X — the recovered incident. Correct target, wrong contracted label."""
    scenario("X-label-mismatch")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        victim = TARGETS[7]
        root = build(tmp, man=lambda r: manifest(
            r, labels={victim: f"{victim} label"}))
        refused("X fabricated result_label", "RESULT_CONTRACT_CONFLICT",
                root, tmp / "floors.json")


def test_every_label_fabricated_is_refused():
    """X2 — the literal shape of the old fixture must now be rejected."""
    scenario("X2-all-labels-fabricated")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, labels={t: f"{t} label" for t in TARGETS}))
        refused("X2 every label fabricated", "RESULT_CONTRACT_CONFLICT",
                root, tmp / "floors.json")


def test_wrong_ordinal():
    scenario("Y-wrong-ordinal")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, positions={TARGETS[9]: 0}))
        refused("Y wrong ordinal", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_completed_with_nonzero_exit():
    scenario("Z-completed-nonzero-exit")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(r, exits={TARGETS[2]: 1}))
        refused("Z COMPLETED with exit 1", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_missing_passed():
    scenario("AA-missing-passed")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, results={TARGETS[1]: {"failed": 0}}))
        refused("AA no passed", "RESULT_CONTRACT_CONFLICT",
                root, tmp / "floors.json")


def test_invalid_passed():
    scenario("AB-invalid-passed")
    for label, value in (("True", True), ("string", "50"), ("negative", -1),
                         ("float", 1.5), ("null", None)):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = build(tmp, man=lambda r: manifest(
                r, results={TARGETS[1]: {"passed": value, "failed": 0}}))
            refused(f"AB passed={label}", "RESULT_CONTRACT_CONFLICT",
                    root, tmp / "floors.json")


def test_missing_failed():
    scenario("AC-missing-failed")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, results={TARGETS[1]: {"passed": 50}}))
        refused("AC no failed", "RESULT_CONTRACT_CONFLICT",
                root, tmp / "floors.json")


def test_invalid_failed():
    scenario("AD-invalid-failed")
    for label, value in (("True", True), ("string", "0"), ("negative", -1),
                         ("float", 0.0), ("null", None)):
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            root = build(tmp, man=lambda r: manifest(
                r, results={TARGETS[1]: {"passed": 50, "failed": value}}))
            refused(f"AD failed={label}", "RESULT_CONTRACT_CONFLICT",
                    root, tmp / "floors.json")


def test_resolved_result_reports_failures():
    """AE — a green aggregate contradicting its own per-target tally."""
    scenario("AE-result-reports-failures")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, results={TARGETS[4]: {"passed": 50, "failed": 1}}))
        rc, out = run_gate(root, tmp / "floors.json")
        check("AE refuses", rc == 2, f"rc={rc}")
        check("AE -> RESULT_CONTRACT_CONFLICT",
              out.get("refusal", {}).get("code") == "RESULT_CONTRACT_CONFLICT",
              str(out.get("refusal")))
        check("AE never adjudicates the passed count",
              "fallen" not in out and "counts" not in out, str(out.keys()))


def test_population_lies():
    scenario("AF-population-lies")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(r, population=77))
        refused("AF population lies", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_population_not_an_integer():
    scenario("AF2-population-bool")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(r, population=True))
        refused("AF2 population is bool", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_plan_path_lies():
    scenario("AG-plan-path-lies")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp, man=lambda r: manifest(
            r, plan_path="scripts/security/some_other_plan.json"))
        refused("AG plan_path lies", "MANIFEST_INVALID",
                root, tmp / "floors.json")


def test_plan_validator_known_negatives():
    """AH — the plan validator itself, without touching the live plan."""
    scenario("AH-plan-validator")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "g", REPO / "scripts" / "security" / "uh_floor_gate.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    good = {"make_target": "test-x", "result_label": "X Tests"}
    cases = [
        ("missing make_target", {"schema": "kai.uh-plan/v1",
                                 "targets": [{"result_label": "X"}]}),
        ("empty make_target", {"schema": "kai.uh-plan/v1",
                               "targets": [{"make_target": "",
                                            "result_label": "X"}]}),
        ("missing result_label", {"schema": "kai.uh-plan/v1",
                                  "targets": [{"make_target": "test-x"}]}),
        ("empty result_label", {"schema": "kai.uh-plan/v1",
                                "targets": [{"make_target": "test-x",
                                             "result_label": ""}]}),
        ("duplicate target", {"schema": "kai.uh-plan/v1",
                              "targets": [good, dict(good)]}),
        ("unknown schema", {"schema": "kai.uh-plan/v9", "targets": [good]}),
        ("empty targets", {"schema": "kai.uh-plan/v1", "targets": []}),
        ("entry not object", {"schema": "kai.uh-plan/v1", "targets": ["x"]}),
        ("not an object", ["not", "a", "dict"]),
    ]
    for name, doc in cases:
        try:
            g.validate_plan(doc)
            check(f"AH {name} refuses", False, "accepted a malformed plan")
        except g.Refusal as exc:
            check(f"AH {name} refuses", exc.code == "PLAN_INVALID", exc.code)
    # known-negative: the real plan must still validate
    try:
        entries = g.validate_plan(json.loads(PLAN.read_text()))
        check("AH the canonical plan still validates", len(entries) == 78,
              str(len(entries)))
    except g.Refusal as exc:
        check("AH the canonical plan still validates", False, str(exc))


def test_fallen_and_unfloored_together():
    """S2 — both findings at once must still be one red, adjudicated run."""
    scenario("S2-fallen-and-unfloored")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        both = tmp / "both.json"
        f = floors_v2(TARGETS[:-17])
        f["floors"][TARGETS[2]] = 9999
        both.write_text(json.dumps(f), encoding="utf-8")
        rc, out = run_gate(root, both)
        check("S2 exits red", rc == 1, f"rc={rc}")
        check("S2 adjudicated", out.get("adjudicated") is True, str(out)[:160])
        check("S2 reports both findings",
              out.get("findings") == {"fallen": 1, "unfloored": 17},
              str(out.get("findings")))


def test_fully_floored_population_passes():
    """S3 — the ONLY green: P floored entirely and every floor met."""
    scenario("S3-fully-floored-green")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        rc, out = run_gate(root, tmp / "floors.json")
        check("S3 exits green", rc == 0, f"rc={rc}")
        check("S3 has no unfloored member",
              out.get("unfloored_targets") == [], str(out.get("unfloored_targets")))
        check("S3 findings are both zero",
              out.get("findings") == {"fallen": 0, "unfloored": 0},
              str(out.get("findings")))


def test_the_declared_denominator_is_what_it_prints():
    """I-2, and the evidence `probe=False` rests on.

    The registry declares this gate's denominator and skips the live
    probe, because probing means handing it an evidence root and a floor
    registry that only a real run produces. A denominator nobody ever
    checks is the "pass that cannot be falsified" I-2 exists to prevent,
    so the declaration is verified here instead — against the gate's real
    human-readable output, which every other scenario bypasses with
    --json.
    """
    scenario("declared denominator matches real output")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        root = build(tmp)
        p = subprocess.run(
            [sys.executable, str(GATE), "--evidence-root", str(root),
             "--floors", str(tmp / "floors.json")],
            capture_output=True, text=True, cwd=str(REPO),
            env={k: v for k, v in os.environ.items()
                 if k != "KAI_UH_EVIDENCE_ROOT"})
        check("the non-JSON path exits green on a clean root",
              p.returncode == 0, f"rc={p.returncode} {p.stdout}{p.stderr}")
        check("output matches the registry's declared denominator",
              re.search(DECLARED_DENOMINATOR, p.stdout) is not None,
              p.stdout[:200])
        check("and the denominator names a non-zero population",
              re.search(r"Assertion floors — (\d+) targets", p.stdout)
              and int(re.search(r"Assertion floors — (\d+) targets",
                                p.stdout).group(1)) > 0,
              p.stdout[:200])


def run() -> None:
    test_root_absent()
    test_root_relative()
    test_cli_and_env_disagree()
    test_missing_log()
    test_missing_status()
    test_missing_manifest()
    test_malformed_status()
    test_nonzero_status_with_green_manifest()
    test_zero_status_with_failed_manifest()
    test_evidence_root_mismatch()
    test_external_plan_scope()
    test_wrong_plan_digest()
    test_duplicate_target()
    test_missing_population_target()
    test_foreign_target()
    test_reordered_targets()
    test_schema_v1_floors_refused()
    test_orphan_floor()
    test_unfloored_targets_reported()
    test_bad_floor_values()
    test_root_plus_from_log_refused()
    test_root_plus_update_floors_refused()
    test_clean_run_adjudicates()
    test_fallen_floor_is_a_finding_not_a_refusal()
    test_result_label_mismatch()
    test_every_label_fabricated_is_refused()
    test_wrong_ordinal()
    test_completed_with_nonzero_exit()
    test_missing_passed()
    test_invalid_passed()
    test_missing_failed()
    test_invalid_failed()
    test_resolved_result_reports_failures()
    test_population_lies()
    test_population_not_an_integer()
    test_plan_path_lies()
    test_plan_validator_known_negatives()
    test_fallen_and_unfloored_together()
    test_fully_floored_population_passes()
    test_the_declared_denominator_is_what_it_prints()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run()
    print("=" * 60)
    print(f"UH Floor Gate Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
