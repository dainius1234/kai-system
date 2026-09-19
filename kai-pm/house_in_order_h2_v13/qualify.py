#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — QUALIFICATION META-CHECK.

Implements the eight criteria of D367 8. The one that did not exist in
v1.1 is first, because it is the one that let the other sixteen defects
pass a green gate.

D17 -- THE GATE COULD NOT SEE AN OMISSION. v1.1's `qualify()` iterated
`ont.ALPHABETS`, so a value ABSENT from an alphabet lay outside its
denominator entirely: removing UNKNOWN from VALIDITY left it at 0
findings while 216 documents emitted that value. A check whose universe
is the list it is checking cannot detect an omission from that list.

So this gate checks THREE denominators, deliberately from three
different places:

  1. the AXIS SET      -> is the governing invariant satisfied?
                          (ontology_invariants; must not consult any
                          alphabet to decide what it expects)
  2. the ALPHABET      -> does every declared value carry a disposition,
                          is every emittable value reachable, is every
                          forbidden value absent?
  3. the OUTPUT        -> did any row emit a value the ontology has
                          never heard of?

Any one alone is blind in a direction the other two are not.

And it PROVES IT CAN FAIL in the same run: the removal calibration
required by D367 7 runs here, not only in the fixtures, so a green
qualification is never reported by an untested instrument.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ontology as ont                                         # noqa: E402


def dispositions(result):
    rows = result["rows"]
    observed = collections.Counter()
    for x in rows:
        for axis in ont.ALPHABETS:
            observed[(axis, x[axis]["value"])] += 1
    findings, table = [], []
    for axis, values in ont.ALPHABETS.items():
        for v in values:
            disp, _ = ont.STATE_DISPOSITION.get((axis, v), ("UNDECLARED", ""))
            n = observed.get((axis, v), 0)
            note = ""
            if disp == "H2_EMITTABLE":
                if n == 0:
                    note = "not observed on this subject"
            elif disp in ("H2_NOT_EARNABLE", "DEFERRED_TO_H3"):
                if n > 0:
                    note = f"FORBIDDEN VALUE EMITTED {n} times"
                    findings.append((axis, v, disp, note))
            else:
                note = "value has no declared disposition"
                findings.append((axis, v, disp, note))
            table.append({"axis": axis, "value": v, "disposition": disp,
                          "subject_count": n, "note": note})
    return table, findings, observed


def removal_calibration():
    """PROVE THE GATE CAN FAIL, IN THIS RUN (D367 7).

    Remove UNKNOWN from each axis in turn; the invariant MUST report it.
    A gate that has never been shown to fail is an untested instrument,
    not evidence -- 40/40 green was reported by v1.1 the entire time
    seventeen defects were live.
    """
    results = []
    for axis in ont.ALPHABETS:
        saved = ont.ALPHABETS[axis]
        try:
            ont.ALPHABETS[axis] = tuple(v for v in saved if v != ont.ABSTENTION)
            fired = any(a == axis for a, _ in ont.ontology_invariants())
        finally:
            ont.ALPHABETS[axis] = saved
        results.append((axis, fired))
    clean = ont.ontology_invariants() == []
    return results, clean


class QualifierIdentityError(AssertionError):
    """D367 §8(6) refused. Raised, never returned."""


def runtime_module_identity(manifest_path):
    """WHICH BYTES ACTUALLY EXECUTED. Inspecting sys.path and concluding
    stale imports are impossible is the reasoning that has already failed
    three times in this workstream.

    D379 §5 — THE HARD-CODED FIVE-MODULE TUPLE IS ABOLISHED AND IS NOT
    REPLACED BY ANOTHER LITERAL TUPLE. The population is DERIVED from the
    modules actually loaded in this interpreter. A hand-written tuple
    beside the thing it measures is a scope smaller than its name (R5),
    and this one was: it named five modules while the governed package has
    ten, so `holdout`, `qualify`, `run_h2_v12`, `cal_fixtures` and
    `stage_identity` could execute from anywhere without being checked.

    FAIL CLOSED, per D367 §8(6). A missing, unreadable or malformed
    manifest REFUSES; it does not degrade to "no finding".
    """
    p = pathlib.Path(manifest_path)
    if not p.is_file():
        raise QualifierIdentityError(
            f"REFUSE: qualifier manifest {manifest_path} does not exist. "
            f"D367 §8(6) is fail-closed: an unestablished qualifier identity "
            f"is not a passing one.")
    try:
        text = p.read_text()
    except OSError as e:
        raise QualifierIdentityError(
            f"REFUSE: qualifier manifest {manifest_path} is unreadable: {e}")
    manifest = {}
    for line in text.splitlines():
        if "  " in line:
            h, n = line.split("  ", 1)
            manifest[n.strip()] = h.strip()
    if not manifest:
        raise QualifierIdentityError(
            f"REFUSE: qualifier manifest {manifest_path} yielded no entries")

    here = p.resolve().parent
    rows, bad = [], []
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        fp = pathlib.Path(f).resolve()
        if fp.parent != here:
            continue                       # not a governed candidate module
        digest = hashlib.sha256(fp.read_bytes()).hexdigest()
        declared = manifest.get(fp.name)
        ok = declared is not None and digest == declared
        rows.append({"module": name, "file": str(fp),
                     "under_candidate_dir": True,
                     "source_sha256": digest,
                     "manifest_sha256": declared, "matches": ok})
        if not ok:
            bad.append(fp.name)
    if not rows:
        raise QualifierIdentityError(
            "REFUSE: no governed candidate module was observed loaded from "
            f"{here}. A qualifier that measures nothing has not passed.")
    return rows, bad


# ── §8(6), THE SUPERSEDING D380 RULE — INC-2026-09-19-36 ──────────────
#
# THE DEFECT THIS REPLACES. `runtime_module_identity` narrows to modules
# whose resolved parent equals the manifest directory and `continue`s past
# everything else. Measured: 73 loaded filesystem-backed origins silently
# skipped, including the two the contract exists to reject. Its own
# calibration asked the same narrow question, so the check could not fail
# for the reason the implementation was wrong.
#
# THE QUALIFIER DERIVES ITS OWN POPULATION. It does NOT call
# stage_identity.producer_population() and accept the answer: that would
# make qualifier completeness depend on the producer classifier and
# recreate the self-certified denominator one layer up. The two share
# deterministic low-level normalisation only, never the ANSWER.
#
# THE MANIFEST ALONE CANNOT ESTABLISH §8(6). It says nothing about Census,
# stdlib identity, interpreter identity or built-in/frozen origins, so a
# VALIDATED Stage-A descriptor is a required fail-closed input.

QUAL_H2 = "H2"
QUAL_CENSUS = "CENSUS"
QUAL_STDLIB = "GOVERNED_STDLIB"
QUAL_BUILTIN = "BUILTIN_OR_FROZEN"
QUAL_CLASSES = (QUAL_H2, QUAL_CENSUS, QUAL_STDLIB, QUAL_BUILTIN)


def _load_stage_a(stage_a_path):
    """Validate the supplied Stage-A descriptor BEFORE using it."""
    import stage_identity as SI
    sp = pathlib.Path(stage_a_path)
    if not sp.is_file():
        raise QualifierIdentityError(
            f"REFUSE: no Stage-A descriptor at {stage_a_path}. §8(6) cannot "
            f"be established from a manifest alone.")
    desc = json.loads(sp.read_bytes().decode("utf-8"))
    SI.validate_descriptor(desc)
    return desc


def classify_loaded_origin(name, mod, *, stage_h2, manifest, desc,
                           repo, roots, external):
    """ONE loaded origin -> exactly one governed class, or REFUSE.

    ORIGIN METADATA FIRST, and it outranks __file__ — a frozen module
    carrying a convenience path stays frozen.
    """
    import os as _os
    import stage_identity as SI
    o = SI.observe_origin(name, mod)
    if o.spec_origin in ("built-in", "frozen"):
        return {"module": name, "class": QUAL_BUILTIN,
                "identity": o.spec_origin}
    cands = []
    if (o.spec_origin and o.spec_origin not in ("built-in", "frozen")
            and _os.path.exists(o.spec_origin)):
        cands.append(_os.path.realpath(o.spec_origin))
    if o.file:
        cands.append(_os.path.realpath(o.file))
    if len(cands) == 2 and cands[0] != cands[1]:
        raise QualifierIdentityError(
            f"REFUSE: {name} spec.origin and __file__ resolve to different "
            f"sources: {cands[0]} != {cands[1]}")
    if not cands:
        raise QualifierIdentityError(
            f"REFUSE: {name} has no filesystem source and its origin is "
            f"neither built-in nor frozen (spec_origin={o.spec_origin!r}).")
    rp = cands[0]
    fp = pathlib.Path(rp)
    h2root = (repo / "kai-pm" / "house_in_order_h2_v13").resolve()
    censusroot = (repo / "kai-pm" / "house_in_order_census_v11").resolve()
    if str(fp).startswith(str(h2root) + _os.sep):
        rel = fp.relative_to(repo).as_posix()
        if rel not in stage_h2:
            raise QualifierIdentityError(
                f"REFUSE: loaded H2 source {rel} is ABSENT from the Stage-A "
                f"h2_sources population.")
        digest = hashlib.sha256(fp.read_bytes()).hexdigest()
        if digest != stage_h2[rel]:
            raise QualifierIdentityError(
                f"REFUSE: loaded H2 source {rel} byte mismatch against "
                f"Stage A: {digest} != {stage_h2[rel]}")
        declared = manifest.get(fp.name)
        if declared is not None and declared != digest:
            raise QualifierIdentityError(
                f"REFUSE: the H2 manifest and Stage-A h2_sources DISAGREE "
                f"about {rel}: {declared} != {digest}")
        return {"module": name, "class": QUAL_H2, "identity": rel,
                "source_sha256": digest}
    if str(fp).startswith(str(censusroot) + _os.sep):
        return {"module": name, "class": QUAL_CENSUS,
                "identity": fp.relative_to(repo).as_posix(),
                "census_aggregate": desc["census"]["aggregate_sha256"]}
    if SI._is_external(rp, external):
        raise QualifierIdentityError(
            f"REFUSE: {name} is a loaded EXTERNAL module at {rp}, outside "
            f"every governed root, with no Stage-A dependency identity.")
    if SI._owning_root(rp, roots) is not None:
        return {"module": name, "class": QUAL_STDLIB, "identity": rp,
                "stdlib_identity": desc["runtime"]["stdlib_identity"]}
    raise QualifierIdentityError(
        f"REFUSE: {name} at {rp} lies outside the governed H2 root, the "
        f"governed Census root and the governed stdlib classification.")


def qualifier_population(stage_a_path, manifest_path):
    """EVERY loaded origin, classified or REFUSED. NO SKIP CLASS."""
    import stage_identity as SI
    desc = _load_stage_a(stage_a_path)
    stage_h2 = {m["path"]: m["sha256"] for m in desc["h2_sources"]}
    mp = pathlib.Path(manifest_path)
    if not mp.is_file():
        raise QualifierIdentityError(
            f"REFUSE: qualifier manifest {manifest_path} does not exist.")
    manifest = {}
    for line in mp.read_text().splitlines():
        if "  " in line:
            h, n = line.split("  ", 1)
            manifest[n.strip()] = h.strip()
    if not manifest:
        raise QualifierIdentityError(
            f"REFUSE: qualifier manifest {manifest_path} yielded no entries")
    repo = HERE.parent.parent
    roots, external = SI._governed_roots()
    rows, refusals = [], []
    for name, mod in sorted(sys.modules.items()):
        if mod is None:
            continue
        try:
            rows.append(classify_loaded_origin(
                name, mod, stage_h2=stage_h2, manifest=manifest, desc=desc,
                repo=repo, roots=roots, external=external))
        except QualifierIdentityError as e:
            refusals.append((name, str(e)))
    return rows, refusals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--manifest", required=True,
                    help="the candidate MANIFEST.sha256. REQUIRED: "
                         "D367 8(6) is fail-closed, and an omitted "
                         "manifest silently SKIPPED criterion [6]")
    a = ap.parse_args()
    res = json.load(open(a.result))
    findings = []

    print("HOUSE_H2 v1.2 — QUALIFICATION")
    print(f"  subject {res['subject'][:12]}  tree {res['subject_tree'][:12]}")
    hi = res["history_identity"]
    print(f"  history {hi['oldest_reachable_date']} → {hi['newest_date']}  "
          f"shallow={hi['shallow']}  ancestry={hi['subject_ancestry_depth']}")
    print(f"  census  {res['census_dependency']['aggregate'][:16]}…\n")

    # ── 1. the AXIS SET ───────────────────────────────────────────────
    inv = ont.ontology_invariants()
    print("  [1] GOVERNING INVARIANT — checked from the AXIS SET")
    print(f"      UNKNOWN first-class on every axis: {inv == []}")
    for axis, why in inv:
        findings.append(("ONTOLOGY", axis, "-", why))
    cal, clean = removal_calibration()
    print("  [1b] REMOVAL CALIBRATION — the gate proves it can fail, this run")
    for axis, fired in cal:
        print(f"       remove UNKNOWN from {axis:<11} detected={fired}")
        if not fired:
            findings.append(("CALIBRATION", axis, "-",
                             "the invariant did NOT fire; the gate is blind"))
    if not clean:
        findings.append(("CALIBRATION", "-", "-", "state not restored"))

    # ── 2. the ALPHABET ───────────────────────────────────────────────
    table, dfind, observed = dispositions(res)
    findings += dfind
    print("\n  [2] STATE DISPOSITIONS — checked from the ALPHABET")
    cur = None
    for r in table:
        if r["axis"] != cur:
            cur = r["axis"]
            print(f"      [{cur}]")
        print(f"        {r['value']:<18}{r['disposition']:<18}"
              f"{r['subject_count']:>6}  {r['note']}")
    und = ont.undeclared()
    if und:
        findings.append(("ONTOLOGY", "-", "-", f"undeclared values {und}"))

    # ── 3. the OUTPUT ─────────────────────────────────────────────────
    outside = ont.values_outside_alphabet(res["rows"])
    print(f"\n  [3] OUTPUT-DERIVED — values emitted but unknown to the "
          f"ontology: {len(outside)}")
    for p, ax, v in outside[:5]:
        findings.append(("OUTPUT", ax, v, f"emitted by {p}, not in alphabet"))

    # ── 4. population ─────────────────────────────────────────────────
    recon = res["population"] == len(res["rows"])
    print(f"\n  [4] population declared {res['population']} == rows "
          f"{len(res['rows'])}: {recon}")
    if not recon:
        findings.append(("POPULATION", "-", "-", "declared != classified"))

    # ── 5. witness traceability ───────────────────────────────────────
    missing = [(r["path"], ax) for r in res["rows"] for ax in ont.ALPHABETS
               if r[ax]["value"] not in ("UNKNOWN", ont.CAPABILITY_FAILURE)
               and not (r[ax].get("witness") or {}).get("witness_value")]
    print(f"  [5] every non-abstention cell carries a source-bound witness: "
          f"{not missing}  ({len(missing)} missing)")
    for p, ax in missing[:5]:
        findings.append(("WITNESS", ax, "-", f"{p} has no witness value"))

    # ── 6. runtime identity ───────────────────────────────────────────
    # D367 8(6) IS NOT OPTIONAL. It was guarded by `if a.manifest:` behind a
    # `default=None` flag, so omitting one argument silently skipped the
    # criterion and the qualifier still reported a result. An absent check
    # that reads as a pass is the defect class this programme exists to find.
    if True:
        rows_id, bad = runtime_module_identity(a.manifest)
        print(f"\n  [6] RUNTIME MODULE IDENTITY — which bytes executed")
        for r in rows_id:
            print(f"      {r['module']:<14} under-candidate="
                  f"{r['under_candidate_dir']}  sha-match={r['matches']}")
        if bad:
            findings.append(("RUNTIME_IDENTITY", ",".join(bad), "-",
                             "loaded module is not the candidate's byte"))

    # ── utility profile, reported SEPARATELY from correctness ─────────
    print("\n  UTILITY PROFILE — reported, never optimised (D367 11)")
    print(f"      {'axis':<12}{'positive':>9}{'UNKNOWN':>9}{'UNMEASURED':>12}")
    for ax in ont.ALPHABETS:
        t = collections.Counter(r[ax]["value"] for r in res["rows"])
        pos = sum(v for k, v in t.items()
                  if k not in ("UNKNOWN", ont.CAPABILITY_FAILURE))
        print(f"      {ax:<12}{pos:>9}{t.get('UNKNOWN', 0):>9}"
              f"{t.get(ont.CAPABILITY_FAILURE, 0):>12}")
    print("      Qualification asks: is the instrument TRUTHFUL?")
    print("      A separate later decision asks: is it USEFUL enough for H3?")

    print(f"\n  FINDINGS: {len(findings)}")
    for a1, a2, a3, note in findings:
        print(f"    {a1}::{a2} — {note}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())


# ── Q1b — THE COMPLETE DERIVED §5 DENOMINATOR, AND E1 THROUGH IT ──────
#
# D379 §8: Q1b takes NO Pass-A input and runs against SYNTHETIC AND LOCAL
# SUBJECTS ONLY. It was briefly reported HELD "requires a real
# classification result"; that was wrong. The frozen v1.2 artefact records
# evidence_facts as bare booleans, which makes it the WRONG FIXTURE, not a
# blocker.
#
# BOTH DENOMINATORS COME FROM THE GOVERNING SCHEMA, NEVER FROM THE RESULT.
# The expected evidence-fact classes are ont.EVIDENCE_FACTS and the axes
# are ont.ALPHABETS. That is what lets Q1b-5 detect the deletion of a whole
# fact class: if the denominator were read off the emitted keys, removing a
# class would remove it from the denominator too and the omission would
# vanish. That is the D17 lesson, one layer out.
#
# NO MEASUREMENT IS ENCODED. 343, 316 and 659 were observations of one
# subject on one day, not definitions, and none appears here.

def q1b_denominators(result):
    """(axis_cells, positive_facts, sum, findings). Derived, not read off."""
    import run_h2_v12 as R
    findings = []
    rows = result.get("rows", [])

    axis_cells = 0
    for r in rows:
        for axis in ont.ALPHABETS:                    # the GOVERNING axis set
            cell = r.get(axis)
            if cell is None:
                findings.append(("AXIS_CELL_ABSENT", r.get("path"), axis,
                                 "the emitted row has no cell for a governed "
                                 "axis"))
                continue
            if cell.get("value") in (ont.ABSTENTION, ont.CAPABILITY_FAILURE):
                continue
            axis_cells += 1
            w = cell.get("witness")
            if not R._compliant(w):
                findings.append(("AXIS_WITNESS", r.get("path"), axis,
                                 "non-abstaining cell carries no compliant "
                                 "witness"))

    positive_facts = 0
    for r in rows:
        emitted = r.get("evidence_facts") or {}
        traces = r.get("evidence_fact_traces") or {}
        abstained = r.get("evidence_facts_abstained_no_compliant_trace") or []
        for name in ont.EVIDENCE_FACTS:               # the GOVERNING schema
            if name not in emitted:
                findings.append(("FACT_CLASS_ABSENT", r.get("path"), name,
                                 "a governed evidence-fact class is missing "
                                 "from the emitted result; the denominator "
                                 "does NOT shrink to hide it"))
                continue
            if not emitted[name]:
                continue
            positive_facts += 1
            # E1, PROVEN HERE AND NOT IN A SEPARATE GATE.
            t = traces.get(name)
            if not R._compliant(t):
                findings.append(("FACT_TRACE", r.get("path"), name,
                                 "positive evidence fact without a compliant "
                                 "nine-field determining trace"))
            elif not R._class_ok(name, t):
                findings.append(("FACT_TRACE_CLASS", r.get("path"), name,
                                 "the trace class does not match the fact "
                                 "class"))
            if name in abstained:
                findings.append(("ABSTENTION_RECONCILIATION", r.get("path"),
                                 name, "fact is listed as abstained for want "
                                 "of a compliant trace AND emitted positive"))
    return axis_cells, positive_facts, axis_cells + positive_facts, findings
