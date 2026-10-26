#!/usr/bin/env python3
"""READ-ONLY CROSS-AXIS SEMANTIC AUDIT — evidence instrument, NOT a classifier.

Banked under D365 as EVIDENCE ONLY. It measures; it repairs nothing, it
is not part of the HOUSE_H2 package, and it must never be imported by a
classifier.

THE QUESTION, for every non-abstention verdict on FUNCTION, AUTHORITY,
GENERATION and SCOPE: does the evidence that earned the verdict prove
the semantic claim the label makes, at the scope the label claims?

WHY THE CLASSIFICATION IS DERIVED, NOT LISTED. The first pass of this
audit adjudicated rows by hand. A hand-maintained verdict list beside
the thing it judges is the R5 defect, and it is also unfalsifiable --
nobody can re-run a judgement held in prose. So the mechanical signals
are computed from the tree

    word_boundary_violation   the matched term sits inside a larger word
                              and the extension is not a plural 's'
    order_dependent           more than one nominated candidate's term
                              also matches, so PATH_NOMINATION ORDER --
                              not evidence strength -- picked the winner
    degenerate_capture        PURPOSE matched a TRIGGER-ONLY alternative,
                              so the capture cannot contain a function
                              term and a closed-vocabulary negative over
                              it is satisfied by construction

and the adjudication classes follow from them by declared rule. Re-run
it and the same rows come back.

THE CLASSES REMAIN AN AUTHOR NOMINATION. The rules encoding them are
mine, on an instrument I wrote, and a self-audit already graded one
population 83% sound that an independent held-out sample later
corrected. These counts carry no admission weight. Kai adjudicates.

    python3 cross_axis_semantic_audit.py \
        --subject-repo <exact checkout> --subject <40-hex> \
        --package <house_in_order_h2_v11 dir> --out audit.json
"""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
import re
import subprocess
import sys

IMMUTABLE_OID = re.compile(r"[0-9a-f]{40}")

# The PURPOSE alternatives that capture ONLY a trigger phrase. A trigger
# phrase can never contain a declared function term, so any negative
# computed over it is satisfied before the document is consulted.
TRIGGER_ONLY = re.compile(
    r"^\s*>?\s*(?:this (?:document|file|register)|defines|records|holds|tracks)",
    re.I)

# A neutral Pass A row for the mutation sweeps. Values are deliberately
# unremarkable: the sweep varies them, it does not rely on them.
BASE_ROW = dict(
    path="x.md", title="t", bytes=10, sha256="0" * 16, commits_in_window=1,
    last="2026-01-01", graphA_in=0, graphA_out=0, exe_ops=0, writers=[],
    readers=[], has_sha=False, has_run=False, has_date=False,
    superseded_by=None, says_supersedes=False, present_tense=False)
SWEEP = ([], ["a.py"], 0, 5, True, False, None, "zzz", "PLAYBOOKS/x.md")


def git(repo, *a):
    return subprocess.run(["git", *a], cwd=str(repo),
                          capture_output=True, text=True)


def reachable_values(cl, axis):
    """Which values does a bounded mutation sweep REACH? Mutation, not reading.

    THE RESULT IS A LOWER BOUND AND NOTHING MORE. The first version of
    this function returned {'UNKNOWN'} for FUNCTION and the caller read
    that as 'deferred by design' -- while FUNCTION emits 222 verdicts on
    the real corpus. The sweep had simply not reached them.

    That is UNKNOWN used as negative evidence: the abstention invariant
    (D340 7 / D358), broken inside the instrument written to detect it.
    A sweep that reaches nothing is silent, not negative. So this
    returns what it REACHED, the caller must union it with what the
    CORPUS emitted, and deferral is decided by is_deferred() below --
    never by this function alone."""
    seen = set()
    for field in BASE_ROW:
        for val in SWEEP:
            row = dict(BASE_ROW)
            row[field] = val
            try:
                seen.add(cl.classify(row, "# t\n", {}, False, "n/a")[axis]["value"])
            except Exception:
                pass                      # a rejected mutation proves nothing
    return sorted(seen), len(BASE_ROW) * len(SWEEP)


def is_deferred(cl, ont, axis, corpus_values):
    """Is an axis deferred BY DESIGN, or merely unobserved?

    Three conditions, and the third is the only one that is proof:

      1. every declared non-abstention value is H2_NOT_EARNABLE or
         DEFERRED_TO_H3 -- a DECLARATION, which is prose until tested;
      2. the corpus emits none of them -- an OBSERVATION, which cannot
         distinguish 'forbidden' from 'not encountered';
      3. INJECTING a forbidden value is REJECTED by the contract
         self-check in classify() -- a KNOWN-POSITIVE. This is the leg
         that carries the claim.
    """
    declared = {v for v in ont.ALPHABETS[axis]
                if v not in ABSTENTIONS(ont)}
    if not declared:
        return False, "axis declares no non-abstention value"
    dispositions = {ont.STATE_DISPOSITION.get((axis, v), ("UNDECLARED",))[0]
                    for v in declared}
    if not dispositions <= {"H2_NOT_EARNABLE", "DEFERRED_TO_H3"}:
        return False, f"declared dispositions {sorted(dispositions)} are earnable"
    if corpus_values - ABSTENTIONS(ont):
        return False, "corpus emits a non-abstention value"

    # Leg 3, the one that carries the claim: emit a forbidden value
    # THROUGH THE REAL CODE PATH and require the self-check to reject it.
    # These axes are assigned inline via _unknown(), so _unknown is what
    # gets poisoned -- patching classify() itself would bypass the guard
    # and prove nothing, and patching ont.emittable would be patching the
    # check rather than testing it.
    victim = sorted(declared)[0]
    real = cl._unknown
    fired = False
    try:
        def poisoned(ax, *a, **k):
            if ax == axis:
                return {"value": victim, "witness": "injected"}
            return real(ax, *a, **k)
        cl._unknown = poisoned
        try:
            cl.classify(dict(BASE_ROW), "# t\n", {}, False, "n/a")
        except AssertionError:
            fired = True
    finally:
        cl._unknown = real
    if not fired:
        return False, f"injected {victim!r} through _unknown() was NOT rejected"

    # Known-negative: the guard must not reject the legitimate abstention.
    try:
        cl.classify(dict(BASE_ROW), "# t\n", {}, False, "n/a")
    except AssertionError as e:
        return False, f"guard also rejects the CLEAN row: {e}"
    return True, (f"declared unearnable; unobserved on the corpus; injecting "
                  f"{victim!r} through the real path is REJECTED; and the "
                  f"clean row is still accepted (known-positive + "
                  f"known-negative)")


def ABSTENTIONS(ont):
    return {"UNKNOWN", ont.CAPABILITY_FAILURE}


def signals(cl, row, verdict):
    """The three mechanical signals, computed per row from the tree."""
    s = {"word_boundary_violation": False, "plural_extension": False,
         "order_dependent": False, "degenerate_capture": False,
         "nominated": [], "also_matched": []}
    value, wt = verdict["value"], verdict.get("witness_type")
    witness = verdict.get("witness") or ""

    if wt in ("title_states_purpose", "purpose_statement"):
        term = cl.FUNCTION_TERMS.get(value, "")
        m = re.search(term, witness, re.I) if term else None
        if m:
            a, b = m.span()
            pre = witness[a - 1] if a > 0 else " "
            post = witness[b] if b < len(witness) else " "
            if pre.isalpha() or post.isalpha():
                s["word_boundary_violation"] = True
                # 'Decisions' / 'Operating rules' are the SAME word
                # pluralised; 'Auditor' and 'Plane' are different words.
                s["plural_extension"] = (
                    not pre.isalpha() and post == "s"
                    and not (b + 1 < len(witness) and witness[b + 1].isalpha()))
        cands = [fn for pat, fn in cl.PATH_NOMINATION
                 if re.search(pat, row["path"], re.I)]
        s["nominated"] = cands
        hits = [c for c in cands if cl.FUNCTION_TERMS.get(c)
                and re.search(cl.FUNCTION_TERMS[c], witness, re.I)]
        s["also_matched"] = hits
        s["order_dependent"] = len(hits) > 1

    if wt == "purpose_outside_declared_vocabulary":
        s["degenerate_capture"] = bool(TRIGGER_ONLY.match(witness))
    return s


def adjudicate(verdict, sig, abstentions, deferred_axis):
    """Declared rules over the mechanical signals. No row is named."""
    if deferred_axis:
        return "DEFERRED_BY_DESIGN", "axis has no reachable non-abstention value"
    if verdict["value"] in abstentions:
        return "ABSTENTION", ""
    wt = verdict.get("witness_type")
    if sig["word_boundary_violation"] and not sig["plural_extension"]:
        return ("FALSE_POSITIVE",
                "matched term lies inside a different word, not a plural of it")
    if sig["degenerate_capture"]:
        return ("FALSE_POSITIVE",
                "closed-vocabulary negative computed over a trigger-only "
                "capture: satisfied by construction, not by the document")
    if sig["order_dependent"]:
        return ("AMBIGUOUS",
                f"candidates {sig['also_matched']} both match; "
                "PATH_NOMINATION order decided the verdict")
    if wt == "proven_family_rule":
        return ("PROVEN",
                "derived family class; all() guarantees per-document evidence")
    if wt == "size_and_role":
        return "PROVEN", "objective: byte count and path suffix"
    if wt in ("title_states_purpose", "purpose_statement"):
        return ("UNSUPPORTED_POSITIVE",
                "path and title are BOTH author self-description with common "
                "provenance: one source counted twice, not two")
    if wt in ("default", "default_pending_region"):
        return ("UNSUPPORTED_POSITIVE",
                "a default presented as a verdict; no determination performed")
    return "AMBIGUOUS", f"unhandled witness type {wt!r}"


def main():
    ap = argparse.ArgumentParser(description="read-only cross-axis audit")
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--package", required=True)
    ap.add_argument("--result", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # R11: prove the subject before measuring anything.
    if not IMMUTABLE_OID.fullmatch(a.subject):
        raise SystemExit("R11 ABORT: --subject must be an immutable 40-hex id")
    head = git(a.subject_repo, "rev-parse", "HEAD").stdout.strip()
    if head != a.subject:
        raise SystemExit(f"R11 ABORT: subject repo HEAD {head[:12]} != "
                         f"subject {a.subject[:12]}")

    pkg = pathlib.Path(a.package).resolve()
    sys.path.insert(0, str(pkg))
    import classify2 as cl, ontology as ont
    for mod in (cl, ont):
        if pathlib.Path(mod.__file__).resolve().parent != pkg:
            raise SystemExit(f"R11 ABORT: {mod.__name__} loaded from "
                             f"{mod.__file__}, not the named package")

    result = json.load(open(a.result or pkg / "h2v11-classification.json"))
    if result["subject"] != a.subject:
        raise SystemExit("R11 ABORT: result subject != --subject")

    abst = ABSTENTIONS(ont)
    # AXES IN SCOPE. LIFECYCLE was qualified under D361-D363 and VALIDITY
    # audited under D364; re-adjudicating them here with different rules
    # would produce two numbers for one question.
    AXES = ("FUNCTION", "AUTHORITY", "GENERATION", "SCOPE")

    corpus = {ax: {x[ax]["value"] for x in result["rows"]} for ax in AXES}
    reach, sweep_n = {}, 0
    for axis in AXES:
        reach[axis], sweep_n = reachable_values(cl, axis)
        # the sweep is a LOWER BOUND; union it with what the corpus did
        reach[axis] = sorted(set(reach[axis]) | corpus[axis])
    deferred, deferral_why = {}, {}
    for axis in AXES:
        deferred[axis], deferral_why[axis] = is_deferred(
            cl, ont, axis, corpus[axis])

    rows, by = [], collections.defaultdict(collections.Counter)
    for x in result["rows"]:
        for axis in AXES:
            v = x[axis]
            sig = signals(cl, x, v) if axis == "FUNCTION" else {
                "word_boundary_violation": False, "plural_extension": False,
                "order_dependent": False, "degenerate_capture": False,
                "nominated": [], "also_matched": []}
            klass, why = adjudicate(v, sig, abst, deferred[axis])
            by[axis][klass] += 1
            if klass in ("FALSE_POSITIVE", "AMBIGUOUS") or (
                    axis == "SCOPE" and v.get("witness_type")
                    == "default_pending_region"):
                rows.append({"path": x["path"], "axis": axis,
                             "value": v["value"],
                             "witness_type": v.get("witness_type"),
                             "witness": (v.get("witness") or "")[:120],
                             "class": klass, "why": why,
                             "signals": {k: val for k, val in sig.items() if val}})

    ORDER = ["PROVEN", "UNSUPPORTED_POSITIVE", "FALSE_POSITIVE",
             "AMBIGUOUS", "ABSTENTION", "DEFERRED_BY_DESIGN"]
    payload = {
        "audit": "CROSS_AXIS_SEMANTIC", "banked_under": "D365 (evidence only)",
        "subject": a.subject, "subject_tree": result["subject_tree"],
        "population": result["population"],
        "axes_in_scope": list(AXES),
        "axes_audited_elsewhere": {"LIFECYCLE": "D361-D363", "VALIDITY": "D364"},
        "mutation_sweep": f"{len(BASE_ROW)} fields x {len(SWEEP)} values "
                          f"= {sweep_n} mutations per axis (a LOWER BOUND, "
                          f"unioned with corpus-observed values)",
        "reachable_values": reach,
        "deferred_by_design": {k: deferral_why[k] for k, v in deferred.items() if v},
        "not_deferred": {k: deferral_why[k] for k, v in deferred.items() if not v},
        "tally_by_axis": {ax: dict(c) for ax, c in by.items()},
        "flagged_rows": rows,
        "caveat": "Adjudication classes are an AUTHOR NOMINATION derived by "
                  "declared rule from mechanical signals. They carry no "
                  "admission weight. Independent adjudication is required.",
    }
    pathlib.Path(a.out).write_text(json.dumps(payload, indent=1))

    print(f"CROSS-AXIS SEMANTIC AUDIT — subject {a.subject[:12]} "
          f"tree {result['subject_tree'][:12]}  N={result['population']}")
    print(f"  mutation sweep: {payload['mutation_sweep']}\n")
    print(f"  {'axis':<12}{'N':>4} " + "".join(f"{c[:9]:>10}" for c in ORDER))
    tot = collections.Counter()
    for axis in AXES:
        c = by[axis]
        print(f"  {axis:<12}{sum(c.values()):>4} " +
              "".join(f"{c[k]:>10}" for k in ORDER))
        tot.update(c)
    print(f"  {'TOTAL CELLS':<12}{sum(tot.values()):>4} " +
          "".join(f"{tot[k]:>10}" for k in ORDER))
    live = sum(tot[k] for k in ORDER[:4])
    print(f"\n  cells that are neither abstention nor deferred: {live}")
    for k in ORDER[:4]:
        print(f"    {k:<22}{tot[k]:>5}"
              f"   {round(100 * tot[k] / live) if live else 0}%")
    print("\n  REACHABILITY — sweep (lower bound) UNIONED WITH corpus")
    for ax in AXES:
        vals = reach[ax]
        if deferred[ax]:
            note = "  <- DEFERRED BY DESIGN"
        elif len(vals) == 1:
            note = "  <- CONSTANT: no determination is performed"
        else:
            note = ""
        print(f"    {ax:<12} {vals}{note}")
        print(f"    {'':<12} {deferral_why[ax]}")
    print(f"\n  flagged rows written: {len(rows)}")
    print("  AUTHOR NOMINATION. NO ADMISSION WEIGHT. NOTHING REPAIRED.")


if __name__ == "__main__":
    main()
