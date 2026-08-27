#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — HOSTILE FIXTURES. Precommitted in PRECOMMIT.md §3.

Every fixture's expected result was fixed BEFORE this file existed
(contract sha256 fa1069103a721cf5911641cbe6447360069eb9f2a3873a4296531ae280f4258e).

F1 carries the FAIL-OLD/PASS-NEW pair the repair requires: v1.0's date
pattern is reproduced verbatim and must FAIL the same input v1.1 passes.
A repair that only proves the corrected case can silently destroy the
property it was protecting, so each pair also proves the opposite side.
"""
from __future__ import annotations
import json
import re
import sys

import classify2 as cl
import evidence as ev
import ontology as ont
import passa

P = F = 0
FAIL = []


def check(name, cond, detail=""):
    global P, F
    if cond:
        P += 1
    else:
        F += 1
        FAIL.append(f"{name} :: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        print(f"          {str(detail)[:150]}")


def row(**kw):
    base = dict(path="x.md", title="", bytes=500, sha256="0"*16,
                commits_in_window=1, last="2026-08-05", graphA_in=0,
                graphA_out=0, exe_ops=0, writers=[], readers=[],
                has_sha=False, has_run=False, has_date=False,
                superseded_by=None, says_supersedes=False,
                present_tense=False)
    base.update(kw)
    return base


# v1.0's pattern, reproduced verbatim for the fail-old half of F1.
DATE_V10 = re.compile(
    r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2} (?:January|February|March|April|May|"
    r"June|July|August|September|October|November|December) 20\d\d\b")


def f1_abbreviated_date():
    print("\nF1 — ABBREVIATED DATE (fail-old / pass-new, with boundary pair)")
    txt = "> Version: 1.0 — 2 Mar 2026\n\nThe current phase is delivery.\n"
    check("F1a v1.0 pattern MISSES '2 Mar 2026' (the original defect)",
          not DATE_V10.search(txt), "v1.0 pattern matched; cannot reproduce defect")
    check("F1b v1.1 pattern FINDS '2 Mar 2026'",
          bool(passa.DATE.search(txt)), "v1.1 pattern missed the date")
    old = cl.validity(row(has_date=bool(DATE_V10.search(txt)), present_tense=True))
    new = cl.validity(row(has_date=bool(passa.DATE.search(txt)), present_tense=True))
    check("F1c v1.0 evidence yields the FALSE POSITIVE CURRENT_TREE",
          old["value"] == "CURRENT_TREE", old)
    check("F1d v1.1 evidence yields TIME_BOUND", new["value"] == "TIME_BOUND", new)
    # boundary: an UNDATED present-tense document must STILL be CURRENT_TREE
    b = cl.validity(row(has_date=False, present_tense=True))
    check("F1e BOUNDARY: undated present-tense is still CURRENT_TREE",
          b["value"] == "CURRENT_TREE", b)
    for form in ("2 March 2026", "2026-03-02", "Mar 2, 2026", "2 Mar 2026"):
        check(f"F1f v1.1 recognises {form!r}", bool(passa.DATE.search(form)))
    check("F1g v1.1 does NOT match a bare month word in prose",
          not passa.DATE.search("we may revisit this later"))


def f2_self_claim_only():
    print("\nF2 — SELF-CURRENT CLAIM ONLY")
    txt = "# Thing\n\n**Status: ACTIVE**\n\nThis document is the current master.\n"
    r = row(path="thing.md", present_tense=True)
    facts = ev.facts(r, txt)
    check("F2a SELF_ASSERTS_CURRENT is present",
          facts["SELF_ASSERTS_CURRENT"]["present"], facts["SELF_ASSERTS_CURRENT"])
    lc = cl.lifecycle(cl.lifecycle_view(r), [])
    check("F2b LIFECYCLE=ACTIVE is FORBIDDEN — verdict is UNKNOWN",
          lc["value"] == "UNKNOWN", lc)
    check("F2c the abstention explains WHY ACTIVE was not awarded",
          "H2_NOT_EARNABLE" in lc.get("unresolved_reason", ""), lc)


def f3_reader_only():
    print("\nF3 — CODE READER ONLY")
    txt = "# Data file\n\nRows of stuff.\n"
    r = row(path="data/x.md", readers=["scripts/load.py"])
    facts = ev.facts(r, txt)
    check("F3a CONSUMED_AT_SUBJECT is present",
          facts["CONSUMED_AT_SUBJECT"]["present"], facts["CONSUMED_AT_SUBJECT"])
    check("F3b SELF_ASSERTS_CURRENT is ABSENT",
          not facts["SELF_ASSERTS_CURRENT"]["present"],
          facts["SELF_ASSERTS_CURRENT"])
    check("F3c LIFECYCLE=ACTIVE is FORBIDDEN",
          cl.lifecycle(cl.lifecycle_view(r), [])["value"] == "UNKNOWN")


def f4_no_evidence():
    print("\nF4 — NO MAINTENANCE, NO SELF-CLAIM, NO READER")
    facts = ev.facts(row(), "# Empty\n\nnothing here.\n")
    for f in ont.EVIDENCE_FACTS:
        check(f"F4 {f} is NOT manufactured", not facts[f]["present"], facts[f])


def f5_not_earnable():
    print("\nF5 — ATTEMPTED EMISSION OF AN H2_NOT_EARNABLE VALUE")
    check("F5a ACTIVE is declared H2_NOT_EARNABLE",
          ont.STATE_DISPOSITION[("LIFECYCLE", "ACTIVE")][0] == "H2_NOT_EARNABLE")
    check("F5b emittable() refuses ACTIVE", not ont.emittable("LIFECYCLE", "ACTIVE"))
    # a classifier that emitted ACTIVE must be caught by the self-check
    real = cl.lifecycle
    try:
        cl.lifecycle = lambda r, b: {"value": "ACTIVE", "witness": "injected"}
        try:
            cl.classify(row(), "# x\n", {"LIFECYCLE": []}, False, "n/a")
            check("F5c injected ACTIVE is REJECTED by the contract self-check",
                  False, "classify() accepted a forbidden value")
        except AssertionError as e:
            check("F5c injected ACTIVE is REJECTED by the contract self-check",
                  "CONTRACT VIOLATION" in str(e), str(e))
    finally:
        cl.lifecycle = real
    check("F5d AUTHORITY states are DEFERRED_TO_H3",
          all(ont.STATE_DISPOSITION[("AUTHORITY", v)][0] == "DEFERRED_TO_H3"
              for v in ("AUTHORITATIVE", "VERIFIED_DERIVED", "ADVISORY",
                        "NON_AUTHORITY")))


def f6_discovery_failure():
    print("\nF6 — DISCOVERY / TRAVERSAL FAILURE MUST NOT SHRINK SILENTLY")
    check("F6a every declared value carries a disposition (none forgotten)",
          ont.undeclared() == [], ont.undeclared())
    saved = ont.ALPHABETS["LIFECYCLE"]
    try:
        ont.ALPHABETS["LIFECYCLE"] = saved + ("SMUGGLED",)
        check("F6b an undeclared value is DETECTED, not ignored",
              ("LIFECYCLE", "SMUGGLED") in ont.undeclared(), ont.undeclared())
    finally:
        ont.ALPHABETS["LIFECYCLE"] = saved
    check("F6c capability failure yields UNMEASURED, never a plausible value",
          cl.lifecycle(cl.lifecycle_view(row()), ["HISTORY_SOURCE_NON_DEGENERATE"])["value"]
          == ont.CAPABILITY_FAILURE)


def f7_reference_and_other():
    print("\nF7 — REFERENCE AND OTHER ARE REACHABLE (D341 defect class)")
    ref = cl.function(row(path="kai-pm/NAVIGATION.md",
                          title="Navigation index for the programme"),
                      "# Navigation index\n", False, "")
    check("F7a REFERENCE is emitted", ref["value"] == "REFERENCE", ref)
    oth = cl.function(row(path="misc/thing.md", title="Shed inventory"),
                      "# Shed inventory\n\nThis document records the "
                      "contents of the shed.\n", False, "")
    check("F7b OTHER is emitted for a stated purpose outside the vocabulary",
          oth["value"] == "OTHER", oth)
    gap = cl.function(row(path="PLAYBOOKS/resume_session.md",
                          title="Resume session playbook"),
                      "# Resume session playbook\n", False, "")
    check("F7c nomination gap repaired: PLAYBOOKS/ can now reach a verdict",
          gap["value"] != "UNKNOWN", gap)
    check("F7d path alone still does NOT earn a verdict",
          cl.function(row(path="kai-pm/NAVIGATION.md", title="Weekly totals"),
                      "# Weekly totals\n", False, "")["value"] == "UNKNOWN")


def f8_unobserved_but_emittable():
    """PRECOMMIT §5.3 requires EVERY H2_EMITTABLE value be reachable.
    Three are zero on this subject — SUPERSEDED, TEMPLATE, RUN_ARTEFACT.
    Zero on a corpus is applicability, not unreachability, but the two
    are indistinguishable without a fixture. These prove reachability."""
    print("\nF8 — H2_EMITTABLE VALUES UNOBSERVED ON THE SUBJECT")
    lc = cl.lifecycle(cl.lifecycle_view(row(superseded_by="successor.md")), [])
    check("F8a SUPERSEDED is reachable", lc["value"] == "SUPERSEDED", lc)
    tm = cl.function(row(path="docs/operator-journal/_template.md",
                         title="Session template"),
                     "# Session template\n", False, "")
    check("F8b TEMPLATE is reachable", tm["value"] == "TEMPLATE", tm)
    ra = cl.validity(row(has_run=True))
    check("F8c RUN_ARTEFACT is reachable", ra["value"] == "RUN_ARTEFACT", ra)


def f9_evidence_cannot_reach_lifecycle():
    """KAI'S REQUIRED MUTATION FIXTURE. Vary every forbidden evidence
    field across its range; the lifecycle verdict BYTES must not move.

    This is the fixture the first build lacked. Without it, "lifecycle
    does not use the evidence" was an assertion about code I had read,
    not a property anything tested."""
    print("\nF9 — EVIDENCE MUTATION CANNOT MOVE THE LIFECYCLE VERDICT")
    forbidden = ("commits_in_window", "present_tense", "readers", "exe_ops",
                 "graphA_in", "graphA_out", "writers", "has_run", "has_date")
    check("F9a no forbidden field is in the authorised input tuple",
          not (set(forbidden) & set(cl.LIFECYCLE_AUTHORISED_INPUTS)),
          cl.LIFECYCLE_AUTHORISED_INPUTS)
    base = row(path="kai-pm/thing.md")
    ref = json.dumps(cl.lifecycle(cl.lifecycle_view(base), []), sort_keys=True)
    variants = dict(commits_in_window=[0, 1, 2, 99, 986],
                    present_tense=[True, False],
                    readers=[[], ["a.py"], ["a.py", "b.py"]],
                    exe_ops=[0, 1, 50], graphA_in=[0, 119], graphA_out=[0, 140],
                    writers=[[], ["w.py"]], has_run=[True, False],
                    has_date=[True, False])
    moved = []
    for field, vals in variants.items():
        for v in vals:
            got = json.dumps(cl.lifecycle(cl.lifecycle_view(row(
                path="kai-pm/thing.md", **{field: v})), []), sort_keys=True)
            if got != ref:
                moved.append((field, v))
    check("F9b varying ALL forbidden evidence leaves the verdict byte-identical",
          not moved, moved)
    # KNOWN-NEGATIVE: an AUTHORISED input MUST still move the verdict,
    # or the fixture would pass on a function that ignores everything.
    auth = cl.lifecycle(cl.lifecycle_view(
        row(path="kai-pm/thing.md", superseded_by="next.md")), [])
    check("F9c KNOWN-NEGATIVE: an authorised input DOES move the verdict",
          auth["value"] == "SUPERSEDED", auth)
    try:
        cl.lifecycle_view(base)["present_tense"]
        check("F9d forbidden field raises KeyError at the boundary", False,
              "field was reachable")
    except KeyError:
        check("F9d forbidden field raises KeyError at the boundary", True)


def run():
    f1_abbreviated_date(); f2_self_claim_only(); f3_reader_only()
    f4_no_evidence(); f5_not_earnable(); f6_discovery_failure()
    f7_reference_and_other(); f8_unobserved_but_emittable(); f9_evidence_cannot_reach_lifecycle()


if __name__ == "__main__":
    run()
    print(f"\nHOSTILE FIXTURES: {P} passed, {F} failed")
    for f in FAIL:
        print("  FAIL", f)
    sys.exit(1 if F else 0)
