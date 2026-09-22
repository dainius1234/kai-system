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


def runtime_module_identity(manifest_path):
    """WHICH BYTES ACTUALLY EXECUTED. Inspecting sys.path and concluding
    stale imports are impossible is the reasoning that has already failed
    three times in this workstream."""
    import classify, envelope, ontology, passa, subjectbind
    manifest = {}
    for line in pathlib.Path(manifest_path).read_text().splitlines():
        if "  " in line:
            h, n = line.split("  ", 1)
            manifest[n.strip()] = h.strip()
    here = pathlib.Path(manifest_path).resolve().parent
    rows, bad = [], []
    for m in (classify, envelope, ontology, passa, subjectbind):
        f = pathlib.Path(m.__file__).resolve()
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = f.parent == here and digest == manifest.get(f.name)
        rows.append({"module": m.__name__, "file": str(f),
                     "under_candidate_dir": f.parent == here,
                     "source_sha256": digest,
                     "manifest_sha256": manifest.get(f.name), "matches": ok})
        if not ok:
            bad.append(f.name)
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--manifest", default=None)
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
    if a.manifest:
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
