"""At-rest integrity of the four implicated evidence-package lineages.

**READ-ONLY. THIS FILE NEVER WRITES, NEVER REPAIRS, AND MAY NEVER
REGENERATE A MANIFEST.** A manifest regenerated to make a gate green is
a false attestation, and this repository has already ruled on that: when
S1 O3 legitimately changed two Census files, retaining the old manifest
to preserve the prior digest was rejected for exactly that reason.

Authorised by D376 §3 as corrected by D377, and scoped by both to the
four implicated packages. **It is not a general evidence-package
ontology** — there is no universal aggregate formula here, because
assuming one is what produced the defect D376 §1.3 corrects.

WHAT THIS ADDS, AND WHAT IT DELIBERATELY DOES NOT DUPLICATE
------------------------------------------------------------
The accepted S1 mechanism at
`kai-pm/house_in_order_h2_v13/build_evidence/s1_toctou_controls.py`
is the authoritative control for the Census CONSUMPTION path:
READ → VERIFY THOSE BYTES → USE, one injected verifier, change-and-restore
protection, fail-closed refusal, reader-bypass guard. **That mechanism is
reused, not recreated.** Nothing here re-implements consumption-time
verification.

What no control covers is the packages AT REST. No workflow step and no
Makefile target references any of these packages, so nothing in CI has
ever asked whether their bytes still match their manifests. That is
doctrine 44 in its plainest form — `RULE_BANKED != CONTROL_OPERATIONALISED`
— and closing it is this file's entire job.

The concrete exposure it answers: `kai-pm/house_in_order_h2/run_h2.py`
writes `h2-classification-v1.json` and `h2-capability-contract.json`
directly back into its own package, and both are manifest-bound. A single
run of a documented entrypoint would leave the package structurally
normal and no longer matching its banked identity. **This gate detects
that. It does not prevent it** — the operational DO-NOT-RUN-IN-PLACE hold
is the containment, and detection is not containment.

TWO IDENTITY CONVENTIONS, BOTH REAL
-----------------------------------
`house_in_order_h2` embeds its own aggregate: `sha256` over the manifest's
ENTRY LINES, written back into the file as a comment. Because the comment
is appended, the whole-file hash necessarily differs from it and is NOT
that package's governance identity.

The later packages use `sha256` of the whole `MANIFEST.sha256` file.

Applying one formula to both is the defect D376 §1.3 records, so each
package carries its own convention and its own provenance below.

UNTRACKED FILES DO NOT FAIL
---------------------------
Importing a module creates `__pycache__`, and two of these packages
already carry one. A runtime artefact is not evidence mutation. Closed-
world artefact-population comparison is applied ONLY where governing
authority declares an exact artefact count, and is not invented where
authority is silent.

Exit codes:
  0  every manifest-listed artefact matches, and every declared identity holds
  1  any mismatch, any missing listed artefact, or an unreadable manifest
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

WHOLE_FILE = "whole-file sha256(MANIFEST.sha256)"
EMBEDDED_ENTRIES = "sha256(manifest entry lines), embedded as a comment"

#: One row per implicated lineage. `identity` is a DECLARATION bound to a
#: decision record, not a value derived from the subject — deriving the
#: expected answer from the thing under test is the error I-8 exists to
#: prevent. The provenance column is what makes each declaration
#: checkable by a human against the append-only log.
#:
#: `declared_artefacts` is populated only where a decision states an exact
#: count. `None` means authority is silent and no closed-world comparison
#: is made.
PACKAGES: Tuple[Dict[str, object], ...] = (
    {
        "path": "kai-pm/house_in_order_h2",
        "convention": EMBEDDED_ENTRIES,
        "identity": "fa8477261b83ef1bfdca273742554329918a80a8ac025f7ad2155e0d655145f4",
        "provenance": "D339 §7 — durable package aggregate",
        "declared_artefacts": 14,
        "note": "run_h2.py writes two manifest-bound outputs in place; "
                "DO-NOT-RUN-IN-PLACE hold is the containment, this is detection",
    },
    {
        "path": "kai-pm/house_in_order_census_v11",
        "convention": WHOLE_FILE,
        "identity": "29064d650a61296806df3c3bcab3322f7364da7df674ac93e79d0671475d757a",
        "provenance": "S1 O3 (bd1cbb4b) accepted hardened lineage; predecessor "
                      "eb7aad7c…fa0e frozen by D357 and preserved, not superseded",
        "declared_artefacts": 19,
        "note": "FROZEN EVIDENCE + AUTHORISED SHARED LIBRARY + AUTHORISED "
                "REPRODUCER — consumed by three later H2 generations",
    },
    {
        "path": "kai-pm/house_in_order_h2_v11",
        "convention": WHOLE_FILE,
        "identity": "be37a0aa5d56255a151c31361d93e8b4be94ab912ec9441c8ac3535a84fbf133",
        "provenance": "D363 FINAL CANDIDATE — NOT FROZEN, NOT ADMITTED",
        "declared_artefacts": 15,
        "note": "candidate identity, preserved as evidence",
    },
    {
        "path": "kai-pm/census_v11_claim_sensitivity",
        "convention": WHOLE_FILE,
        "identity": "397ceda087d5032484d0fdc08ee0e0308b17ea36d2ab54e9f83b2a2bb3a0b596",
        "provenance": "no freeze decision located; D354 §2 hashes PRECOMMIT.md "
                      "(a89bb9cc…). Recorded as measured, not as authority",
        "declared_artefacts": None,
        "note": "authority is silent on an artefact count; no closed-world "
                "comparison is made",
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_entries(manifest: Path) -> List[Tuple[str, str]]:
    """(digest, name) for every non-comment line. Names are bare, by design."""
    out = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def embedded_aggregate(manifest: Path) -> Tuple[Optional[str], str]:
    """The package's own `# aggregate` line, and the value it should hold.

    Computed over the entry lines only — the aggregate cannot include the
    comment that carries it.
    """
    text = manifest.read_text(encoding="utf-8")
    declared = None
    for line in text.splitlines():
        if "aggregate" in line and line.lstrip().startswith("#"):
            declared = line.rsplit(":", 1)[-1].strip()
    entries = "".join(
        l + "\n" for l in text.splitlines()
        if l.strip() and not l.lstrip().startswith("#"))
    return declared, hashlib.sha256(entries.encode()).hexdigest()


def check_package(spec: Dict[str, object]) -> Tuple[List[str], int]:
    """Returns (findings, artefacts_verified). Never writes anything."""
    findings: List[str] = []
    pkg = REPO / str(spec["path"])
    manifest = pkg / "MANIFEST.sha256"

    # I-1: an unreadable manifest is a failure, not a skip. A package we
    # cannot verify and a package that verifies are different answers.
    if not manifest.is_file():
        return [f"{spec['path']}: MANIFEST.sha256 is missing — cannot verify"], 0

    entries = manifest_entries(manifest)
    if not entries:
        return [f"{spec['path']}: MANIFEST.sha256 lists no artefacts"], 0

    verified = 0
    for digest, name in entries:
        artefact = pkg / name
        if not artefact.is_file():
            findings.append(
                f"{spec['path']}/{name}: listed in MANIFEST, absent on disk")
            continue
        actual = sha256_file(artefact)
        if actual != digest:
            findings.append(
                f"{spec['path']}/{name}: BYTES CHANGED — manifest "
                f"{digest[:16]}…, on disk {actual[:16]}…")
        else:
            verified += 1

    # Per-package identity convention. Never one formula for all four.
    if spec["convention"] == WHOLE_FILE:
        actual = sha256_file(manifest)
        if actual != spec["identity"]:
            findings.append(
                f"{spec['path']}: package identity moved — declared "
                f"{str(spec['identity'])[:16]}…, computed {actual[:16]}… "
                f"({spec['provenance']})")
    else:
        declared, computed = embedded_aggregate(manifest)
        if declared is None:
            findings.append(
                f"{spec['path']}: convention is {EMBEDDED_ENTRIES} but the "
                f"manifest carries no aggregate line")
        elif declared != computed:
            findings.append(
                f"{spec['path']}: embedded aggregate disagrees with its own "
                f"entries — line says {declared[:16]}…, entries hash "
                f"{computed[:16]}…")
        elif declared != spec["identity"]:
            findings.append(
                f"{spec['path']}: embedded aggregate moved — declared "
                f"{str(spec['identity'])[:16]}…, found {declared[:16]}… "
                f"({spec['provenance']})")

    # Closed-world comparison ONLY where authority states a count.
    declared_count = spec["declared_artefacts"]
    if declared_count is not None and len(entries) != declared_count:
        findings.append(
            f"{spec['path']}: authority declares {declared_count} artefacts, "
            f"MANIFEST lists {len(entries)}")

    return findings, verified


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    all_findings: List[str] = []
    total_verified = 0
    total_listed = 0
    rows: List[str] = []

    for spec in PACKAGES:
        findings, verified = check_package(spec)
        manifest = REPO / str(spec["path"]) / "MANIFEST.sha256"
        listed = len(manifest_entries(manifest)) if manifest.is_file() else 0
        total_verified += verified
        total_listed += listed
        all_findings.extend(findings)
        rows.append("    %-42s %2d/%-2d  %s"
                    % (spec["path"], verified, listed, spec["convention"]))

    print("Evidence-package integrity — %d packages, %d/%d manifest-listed "
          "artefacts verified, %d finding(s)"
          % (len(PACKAGES), total_verified, total_listed, len(all_findings)))
    print("  READ-ONLY. This gate never writes, never repairs, and may never")
    print("  regenerate a MANIFEST.")
    print("  AT-REST ONLY. Consumption-time verification for the Census path")
    print("  is the accepted S1 mechanism at kai-pm/house_in_order_h2_v13/")
    print("  build_evidence/s1_toctou_controls.py, reused and not recreated.")
    print("  Untracked runtime artefacts such as __pycache__ do not fail this")
    print("  gate. Closed-world artefact counts are compared only where a")
    print("  decision declares one.")
    print("")
    for row in rows:
        print(row)
    print("")
    for spec in PACKAGES:
        print("    %-42s %s" % (spec["path"], spec["provenance"]))

    if not all_findings:
        print("")
        print("  PASS: every manifest-listed artefact matches, and every "
              "declared identity holds.")
        return 0

    print("")
    for finding in all_findings:
        print("  FINDING %s" % finding)
    return 1


if __name__ == "__main__":
    sys.exit(main())
