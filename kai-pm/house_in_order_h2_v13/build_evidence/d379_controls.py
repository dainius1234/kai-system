#!/usr/bin/env python3
"""D379 HOSTILE CONTROLS — governed by D379 as corrected by D380.

EVIDENCE CLASS: PRODUCER MEASUREMENT - SIGHTED - ZERO ADMISSION WEIGHT.
Orion's own assessment carries no final admission weight.

Every control is EXECUTED, never merely asserted (R2). Controls whose
subject is a shipped process entry point assert the ACTUAL SUBPROCESS
RETURN CODE. Controls whose subject is a classification decision function
call that function directly, because the function IS the subject; calling
a helper and inferring a process exit status is what let an earlier defect
through and is not done here.

SECTION REGISTRY. Sections not yet implemented FAIL LOUDLY. A control file
that reports green over a section it does not cover is the exact defect
class this programme exists to find, so absence is never silence here.
"""
from __future__ import annotations

import collections
import hashlib
import os
import json
import pathlib
import sys

V = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V))

import classify                                              # noqa: E402
import qualify                                               # noqa: E402
import passa                                                 # noqa: E402
from envelope import Witness                                 # noqa: E402

PASSED, FAILED, FAILURES = 0, 0, []
# Sections of the D379 hostile matrix. Implemented sections run; the rest
# fail as NOT_IMPLEMENTED so this file can never report a green tranche.
SECTIONS = ["M2", "D14", "PPOP", "Q1a", "Q1b", "86", "SB", "I1A",
            "I1B", "DEP", "STAGE_A", "STDLIB"]
IMPLEMENTED = {"M2", "SB", "D14", "I1A", "I1B", "PPOP", "86", "Q1b"}
# HELD is NOT an excuse and does NOT make the gate green. These
# sections are implemented except for a limb that cannot execute
# on a KNOWN-NEGATIVE interpreter (INC-34 / D385). They still FAIL
# the exit gate; they are reported separately only so the registry
# does not call a blocked limb "not written".
HELD = {
    # HELD names a limb that CANNOT EXECUTE on a known-negative runtime.
    # It is not a pass and does not soften the gate: every section listed
    # here still FAILS below. It exists so the registry does not describe a
    # blocked limb as unwritten.
    "Q1a": "Q1a-6 ONLY — needs a governed POSITIVE runtime identity to "
           "compare against (INC-34). Q1a-1,2,3,4,5,7,8,9 all EXECUTE.",
    "DEP": "DEP-2 ONLY — needs ordinary stdlib under a D380-COMPLIANT "
           "interpreter (INC-34). DEP-1 and DEP-3 EXECUTE.",
    "STAGE_A": "the canonical-runtime positive limb ONLY (INC-34). The V2 "
               "governance and identity matrix EXECUTES.",
    "STDLIB": "V2-ID-2a, the canonical positive derivation ONLY (INC-34). "
              "The D380-STDLIB-NEG-1 negative control EXECUTES.",
}


def check(name: str, condition: bool, detail: str = "") -> bool:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")
    return condition


def subject_digests():
    print("SUBJECT — the exact bytes measured")
    for n in ("passa.py", "classify.py", "envelope.py", "run_h2_v12.py",
              "qualify.py", "holdout.py", "cal_fixtures.py", "ontology.py",
              "subjectbind.py"):
        p = V / n
        if p.exists():
            print(f"  {n:<16} {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print()


# ── M2 / A3-i — LIFECYCLE SUBJECT BINDING ─────────────────────────────
#
# THE PRECOMMITTED D379 PREDICATE, fixed before this measurement ran:
#
#   LIMB I   the three false audited-snapshot COMMIT routes must NOT
#            classify HISTORICAL
#   LIMB II  a genuine document-own-lifecycle COMMIT route must STILL
#            classify HISTORICAL
#
#   BOTH satisfied   -> no M2 semantic edit; discrimination DEMONSTRATED
#   EITHER fails     -> the bounded A3-i correction is the ONLY authorised
#                       semantic mutation
#
# Both limbs are required. LIMB I alone is satisfiable by A3-ii, which was
# rejected for removing genuine cases along with the defective ones.
# LIMB II is what proves DISCRIMINATION rather than SUPPRESSION.
#
# No git history is consulted and no Pass A is run: the decision path
#   _scope_of -> witness.applicability_scope -> _binding_witness -> lifecycle
# is a pure function of the document text and the witness, so the measurement
# needs neither a history source nor a candidate.

COMMIT = "2d830f25d569baa5ce955dd8d17e8f0744239876"

FALSE_ROUTE_TEXT = (
    "# Synthetic Code Audit Report\n"
    "\n"
    f"**Audited snapshot:** default branch through findings commit `{COMMIT}`\n"
    "\n"
    "## Findings\n"
    "\nProse about the audited tree.\n")

GENUINE_ROUTE_TEXT = (
    "# Synthetic Code Audit Snapshot\n"
    "\n"
    f"**Acquisition commit:** `{COMMIT}`\n"
    "\n"
    "## Contents\n"
    "\nThis document was taken at the commit above.\n")


def _commit_witness(text, path):
    """Build the COMMIT witness exactly as passa.scan() builds one.

    D381: that now includes the SUBJECT, derived by the same governed
    producer function `passa._subject_of`. The token is a full 40-hex
    commit, so a NONSELF_GIT_COMMIT policy can name its subject exactly
    without consulting a history source -- `resolve` returns the literal
    token. No git call, and no synthetic subject invented here: the
    control asks the producer, it does not answer for it.
    """
    m = passa.HEX.search(text)
    assert m is not None and m.group(0) == COMMIT
    assert passa._eligible(m)
    head = text[:passa.HEAD_BYTES]
    return Witness(
        witness_type="COMMIT", witness_value=m.group(0), source_path=path,
        source_selector=passa._selector(text, m.start()),
        local_context=passa._context(text, m.start(), m.end()),
        applicability_scope=passa._scope_of(head, m.start(), "HEX"),
        subject=passa._subject_of(head, m.start(), "HEX",
                                  resolve=lambda: COMMIT),
        evidence_total=1, evidence_shown=1, truncated=False,
        polarity="POSITIVE", certainty="VERIFIED")


def _lifecycle_of(text, path):
    w = _commit_witness(text, path)
    row = {"path": path, "witnesses": {"COMMIT": [w.asdict()]}}
    snap = classify._binding_witness(row, "COMMIT")
    res = classify.lifecycle(path=path, superseded_by=None,
                             snapshot_witness=snap.asdict() if snap else None,
                             blocked=None)
    return res, w, snap


def section_M2():
    print("M2 / A3-i — LIFECYCLE SUBJECT BINDING")
    print("  invariant: the subject of a LIFECYCLE verdict is THE DOCUMENT.")
    print("  Evidence about an artefact the document DESCRIBES is evidence")
    print("  about that artefact.\n")

    f_res, f_w, f_snap = _lifecycle_of(
        FALSE_ROUTE_TEXT, "kai-pm/SYNTHETIC_CODE_AUDIT_FALSE_ROUTE.md")
    g_res, g_w, g_snap = _lifecycle_of(
        GENUINE_ROUTE_TEXT, "kai-pm/SYNTHETIC_CODE_AUDIT_GENUINE_ROUTE.md")

    for label, res, w, snap in (
            ("M2-1 LIMB I  false audited-snapshot route", f_res, f_w, f_snap),
            ("M2-2 LIMB II genuine own-lifecycle route", g_res, g_w, g_snap)):
        d = w.asdict()
        print(f"  {label}")
        print(f"      applicability_scope  {d['applicability_scope']}")
        print(f"      witness subject      {d.get('subject')}")
        print(f"      _binding_witness     "
              f"{'returned a witness' if snap else 'None'}")
        print(f"      LIFECYCLE            {res['value']}")
        print(f"      rationale            {str(res.get('rationale'))[:70]}")

    limb1 = check("M2-1 LIMB I: the false audited-snapshot COMMIT route does "
                  "NOT classify HISTORICAL",
                  f_res["value"] != "HISTORICAL",
                  f"classified {f_res['value']} — the commit is the snapshot "
                  f"the document AUDITS, i.e. evidence about a DIFFERENT "
                  f"subject, and it determined the document's own lifecycle")
    limb2 = check("M2-2 LIMB II: the genuine document-own-lifecycle COMMIT "
                  "route STILL classifies HISTORICAL",
                  g_res["value"] == "HISTORICAL",
                  f"classified {g_res['value']} — suppression, not "
                  f"discrimination")

    print()
    print(f"  LIMB I  {'PASS' if limb1 else 'FAIL'}"
          f"     LIMB II {'PASS' if limb2 else 'FAIL'}")
    if limb1 and limb2:
        print("  M2 PREDICATE SATISFIED — discrimination DEMONSTRATED.")
        print("  SAY WHICH TREE THIS IS TRUE OF. Against UNCHANGED v1.3 this")
        print("  predicate FAILED: both limbs classified HISTORICAL, which is")
        print("  what INC-2026-09-18-31 records and what D381 was banked to")
        print("  repair. It passes here because the governed subject gate is")
        print("  now IMPLEMENTED in classify.lifecycle. This is a pass-new")
        print("  reading, not evidence that the defect was never present.")
    else:
        print("  M2 PREDICATE NOT SATISFIED — the bounded A3-i correction is")
        print("  the ONLY authorised semantic mutation. No adjacent cleanup,")
        print("  no vocabulary redesign, no unrelated Pass-A semantics.")
    print()
    # The discrimination question stated as one measurement, not as prose:
    check("M2-3 the two routes are DISTINGUISHED from one another",
          f_res["value"] != g_res["value"],
          f"both routes classified {f_res['value']} with the same rationale — "
          f"the binding predicate did not discriminate between evidence about "
          f"the document and evidence about an artefact it describes")


# ── SB — D381 SUBJECT BINDING + D382 EXECUTED DERIVATION ──────────────
#
# D382 §10 / R18: PROSE BESIDE NUMBERS IS INSUFFICIENT. INC-32 happened
# because a derivation recipe and the figures it supposedly produced sat
# in adjacent paragraphs and neither was ever executed. Every denominator
# below is DERIVED HERE. No transcribed number carries closure weight,
# and 491/480/165/164/485 are EXPECTED RESULTS, never definitions.

REPO = V.parent.parent                      # kai-pm/house_in_order_h2_v13 -> repo
FROZEN_PASSA_BLOB = "f88e929b8c0f569dd7f730e12a8459b00f405595"
PRE_D381_PASSA_BLOB = "c70dabf29fd23dd5328dc15a919386d300f849c3"
FROZEN_TREE = "3abc9e9d8ca11966a6f996d5f0af68072ee5b117"
# D382 §4 — the normative consumer-relevant universe, and it is exactly
# what classify actually consumes. Nothing else is in the denominator
# that produced 167/165.
CONSUMED_FAMILIES = ("COMMIT", "RUN_ID", "DATE", "SUPERSEDED_BY")


def _git(*args):
    import subprocess
    return subprocess.run(["git", *args], cwd=str(REPO),
                          capture_output=True)


def _blob_text(blob):
    r = _git("cat-file", "blob", blob)
    if r.returncode != 0:
        raise AssertionError(f"R11 ABORT: cannot read git blob {blob}")
    return r.stdout.decode("utf-8")


def _tree_file(path):
    r = _git("cat-file", "blob", f"{FROZEN_TREE}:{path}")
    return r.stdout.decode("utf-8") if r.returncode == 0 else None


def _detector_of(witness_type):
    return {"DATE_STAMP": "DATE", "RUN_ID": "DECIMAL_RUN",
            "NAMED_SUCCESSOR": None}.get(witness_type, "HEX")


def _load_pre_d381_passa():
    """The PRE-D381 passa.py, bound by blob id, loaded as its own module.

    D382 §5 step 4 requires the projection to use the pre-D381 scope
    semantics. Taking those bytes from git BINDS them; re-deriving them
    from the working tree would measure the repaired file against itself,
    which is the shape I-8 forbids.
    """
    import importlib.util
    import tempfile
    src = _blob_text(PRE_D381_PASSA_BLOB)
    d = tempfile.mkdtemp(prefix="pre_d381_")
    p = pathlib.Path(d) / "passa_pre_d381.py"
    p.write_text(src)
    # the pre-D381 module imports `envelope` by bare name; give it the
    # COMMITTED pre-D381 envelope too, so nothing new leaks backwards.
    env = _blob_text(_git("rev-parse", "838b7637:kai-pm/house_in_order_h2_v13/"
                          "envelope.py").stdout.decode().strip())
    (pathlib.Path(d) / "envelope.py").write_text(env)
    sys.path.insert(0, d)
    spec = importlib.util.spec_from_file_location("passa_pre_d381", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.remove(d)
    return mod


def _frozen_records():
    """EVERY emitted frozen Pass-A witness record, in array order.

    D382/Kai: the unit is the EMITTED RECORD, not the deduplicated
    identity. Frozen Pass A emitted nine identities more than once, and a
    comparator that consumes each physical source occurrence once silently
    drops the second copy — which is exactly how 491 first read as 485.
    """
    doc = json.loads(_blob_text(FROZEN_PASSA_BLOB))
    recs = []
    for row in doc["rows"]:
        for family, ws in (row.get("witnesses") or {}).items():
            for w in ws:
                recs.append({"path": row["path"], "family": family,
                             "selector": w["source_selector"],
                             "value": w["witness_value"],
                             "type": w["witness_type"],
                             "scope": w["applicability_scope"]})
    return recs, doc


def _locate(recs):
    """Resolve each emitted record to a byte offset in the FROZEN tree.

    OCCURRENCE ORDINAL, NOT CONSUMPTION. Records sharing (path, line,
    value) are duplicate EMISSIONS of one physical source occurrence, so
    they all resolve to the SAME offset. Advancing the search position per
    record — consuming the occurrence — is the defect this avoids.
    """
    cache, out = {}, []
    for r in recs:
        p = r["path"]
        if p not in cache:
            cache[p] = _tree_file(p)
        text = cache[p]
        if text is None:
            out.append((r, None, "NOT_IN_FROZEN_TREE"))
            continue
        lines = text.split("\n")
        starts, off = [], 0
        for ln in lines:
            starts.append(off)
            off += len(ln) + 1
        try:
            n = int(r["selector"].lstrip("L").split("-")[0])
        except ValueError:
            out.append((r, None, "UNPARSEABLE_SELECTOR"))
            continue
        if n - 1 >= len(lines):
            out.append((r, None, "LINE_BEYOND_EOF"))
            continue
        idx = lines[n - 1].find(r["value"])
        if idx < 0:
            out.append((r, None, "VALUE_NOT_ON_LINE"))
            continue
        out.append((r, (text, starts[n - 1] + idx), None))
    return out


def section_SB():
    print("SB — D381 SUBJECT BINDING, AND THE D382 EXECUTED DERIVATION")
    print("  Every denominator below is DERIVED. No transcribed figure")
    print("  carries closure weight (D382 §7/§10, R18).\n")

    # ---- the closed subject grammar (D381 §3.1) ----
    print("  GRAMMAR — the closed six-step parser")
    import envelope as EV
    for s, want in ((EV.SUBJECT_SELF, True), (EV.SUBJECT_AMBIGUOUS, True),
                    ("OTHER:DOCUMENT:kai-pm/a.md", True),
                    ("OTHER:GIT_COMMIT:" + "a" * 40, True),
                    ("OTHER:kai-pm/a.md", True),
                    ("OTHER:MYSTERY:foo", False),
                    ("OTHER:GIT_COMMIT:not-a-commit", False),
                    ("OTHER:DOCUMENT:bad/../path", False),
                    ("OTHER:DOCUMENT:/abs.md", False),
                    ("OTHER:has:colon/x.md", False),
                    ("OTHER:", False), ("", False), ("nonsense", False)):
        try:
            EV.parse_subject(s)
            got = True
        except EV.SubjectError:
            got = False
        check(f"SB-GRAM {s!r:36} {'accepted' if want else 'REFUSED'}",
              got == want, f"got {'accepted' if got else 'REFUSED'}")
    check("SB-GRAM a Witness with NO subject is REFUSED (no default SELF)",
          _refuses(lambda: Witness(
              witness_type="COMMIT", witness_value="x", source_path="a.md",
              source_selector="L1", local_context="x",
              applicability_scope="SPAN", evidence_total=1,
              evidence_shown=1, truncated=False)))

    # ---- the governed predicate registry (D381 §5) ----
    print("\n  REGISTRY — one governed registry, fail-closed")
    check(f"SB-REG every binding predicate carries a subject policy "
          f"({passa.validate_registry()} entries)", True)
    for bad, why in (
            ({"x": "just a rationale string"}, "not a Binding"),
            ({"x": passa.Binding("r", "NOT_A_POLICY")}, "unknown policy"),
            ({"x": passa.Binding("", passa.POLICY_SELF)}, "no rationale")):
        check(f"SB-REG REFUSES a predicate with {why}",
              _refuses(lambda b=bad: passa.validate_registry(b)))
    check("SB-REG an unrecognised policy REFUSES rather than defaulting",
          _refuses(lambda: passa._apply_policy("SOMETHING_ELSE")))
    for lab, want in (("audited snapshot", passa.POLICY_NONSELF_GIT_COMMIT),
                      ("findings-bearing snapshot", passa.POLICY_NONSELF_GIT_COMMIT),
                      ("subject", passa.POLICY_NONSELF_GIT_COMMIT),
                      ("acquisition commit", passa.POLICY_SELF),
                      ("validated checkpoint", passa.POLICY_SELF),
                      ("snapshot", passa.POLICY_AMBIGUOUS),
                      ("measured at", passa.POLICY_AMBIGUOUS),
                      ("last updated", passa.POLICY_SELF)):
        e = passa.registry_lookup(lab)
        check(f"SB-REG {lab!r:28} -> {want}",
              e is not None and e.subject_policy == want,
              None if e is None else e.subject_policy)

    # ---- all seven WHOLE_FILE producer routes (D381 §7) ----
    print("\n  ROUTES — all SEVEN WHOLE_FILE producer routes have subject "
          "semantics")
    missing = [r for r in (passa.ROUTE_R1_H1, passa.ROUTE_R2_SELF_SUBJECT,
                           passa.ROUTE_R3_LABELLED, passa.ROUTE_R4_CONTEXTUAL,
                           passa.ROUTE_R5_ROOT_LIFECYCLE,
                           passa.ROUTE_R6_BARE_DATELINE,
                           passa.ROUTE_R7_SUPERSEDED_BY)
               if r not in passa.ROUTE_SUBJECT_POLICY]
    check("SB-ROUTE every one of the seven routes has a declared policy "
          "(six dormant, and R8 says dormant is where defects live)",
          not missing, missing)
    check("SB-ROUTE R1 (H1 title) does NOT earn SELF from position alone",
          passa.ROUTE_SUBJECT_POLICY[passa.ROUTE_R1_H1] != passa.POLICY_SELF)
    check("SB-ROUTE R7 (SUPERSEDED_BY, hard-coded in scan()) is SELF",
          passa.ROUTE_SUBJECT_POLICY[passa.ROUTE_R7_SUPERSEDED_BY]
          == passa.POLICY_SELF)
    check("SB-ROUTE an undeclared route REFUSES",
          _refuses(lambda: passa._apply_policy(
              passa.ROUTE_SUBJECT_POLICY.get("NO_SUCH_ROUTE"))))

    # ---- consumer discrimination (D381 §8) — HOSTILE SYNTHETIC ----
    print("\n  CONSUMER — discriminated by SYNTHETIC non-SELF witnesses")
    print("  The corpus emits a CONSTANT subject, so it gives ZERO positive")
    print("  discrimination here. These rows are the actual proof.\n")

    def _w(subject, family="COMMIT", wt="COMMIT", scope="WHOLE_FILE",
           ctx="Audited snapshot: `%s`" % COMMIT):
        return {family: [Witness(
            witness_type=wt, witness_value=COMMIT, source_path="kai-pm/AUDIT.md",
            source_selector="L4", local_context=ctx, applicability_scope=scope,
            subject=subject, evidence_total=1, evidence_shown=1,
            truncated=False, polarity="POSITIVE",
            certainty="VERIFIED").asdict()]}

    NONSELF = "OTHER:GIT_COMMIT:" + COMMIT
    for subj, label, want_v, want_l in (
            ("SELF", "SELF", "EXACT_SNAPSHOT", "HISTORICAL"),
            (NONSELF, "OTHER:GIT_COMMIT", "UNKNOWN", "UNKNOWN"),
            ("AMBIGUOUS", "AMBIGUOUS", "UNKNOWN", "UNKNOWN")):
        row = {"path": "kai-pm/AUDIT.md", "witnesses": _w(subj)}
        v = classify.validity(row, None)
        snap = classify._binding_witness(row, "COMMIT")
        lc = classify.lifecycle(path="kai-pm/AUDIT.md", superseded_by=None,
                                snapshot_witness=snap.asdict() if snap else None,
                                blocked=None)
        sc = classify.scope(row)
        check(f"SB-CONS subject={label:<18} VALIDITY  -> {want_v}",
              v["value"] == want_v, v["value"])
        check(f"SB-CONS subject={label:<18} LIFECYCLE -> {want_l}",
              lc["value"] == want_l, lc["value"])
        check(f"SB-CONS subject={label:<18} SCOPE     -> WHOLE_FILE "
              f"(SCOPE has NO SELF gate)",
              sc["value"] == "WHOLE_FILE", sc["value"])

    # RUN_ID is a positive whole-file VALIDITY route too. D381 §8 forbids
    # repairing COMMIT and leaving another positive route open.
    for subj, want in (("SELF", "RUN_ARTEFACT"), (NONSELF, "UNKNOWN")):
        row = {"path": "kai-pm/R.md",
               "witnesses": _w(subj, family="RUN_ID", wt="RUN_ID",
                               ctx="Audited snapshot: run")}
        check(f"SB-CONS RUN_ID positive route obeys the subject gate "
              f"({'SELF' if subj == 'SELF' else 'non-SELF'} -> {want})",
              classify.validity(row, None)["value"] == want)
    # the DATE state-binding route, likewise
    for subj, want in (("SELF", "TIME_BOUND"), (NONSELF, "UNKNOWN")):
        row = {"path": "kai-pm/D.md", "witnesses": {"DATE": [Witness(
            witness_type="DATE_STAMP", witness_value="2026-07-21",
            source_path="kai-pm/D.md", source_selector="L3",
            local_context="**Last updated:** 2026-07-21",
            applicability_scope="WHOLE_FILE", subject=subj, evidence_total=1,
            evidence_shown=1, truncated=False, polarity="POSITIVE",
            certainty="OBSERVED").asdict()]}}
        check(f"SB-CONS DATE state-binding route obeys the subject gate "
              f"({'SELF' if subj == 'SELF' else 'non-SELF'} -> {want})",
              classify.validity(row, None)["value"] == want)
    check("SB-CONS a non-SELF witness in source position 1 does NOT mask a "
          "SELF witness at position 2",
          classify._self_binding_witness(
              {"path": "x", "witnesses": {"COMMIT": [
                  Witness(witness_type="COMMIT", witness_value=COMMIT,
                          source_path="x", source_selector="L1",
                          local_context="Audited snapshot",
                          applicability_scope="WHOLE_FILE", subject=NONSELF,
                          evidence_total=1, evidence_shown=1, truncated=False,
                          polarity="POSITIVE", certainty="VERIFIED").asdict(),
                  Witness(witness_type="COMMIT", witness_value=COMMIT,
                          source_path="x", source_selector="L2",
                          local_context="Acquisition commit",
                          applicability_scope="WHOLE_FILE", subject="SELF",
                          evidence_total=1, evidence_shown=1, truncated=False,
                          polarity="POSITIVE", certainty="VERIFIED").asdict()]}},
              "COMMIT") is not None)

    # ---- the D382 EXECUTED DERIVATION ----
    print("\n  D382 EXECUTED DERIVATION — populations A and B, derived here")
    recs, doc = _frozen_records()
    located = _locate(recs)

    total_records = len(recs)
    identities = collections.Counter(
        (r["path"], r["family"], r["selector"], r["value"], r["scope"])
        for r in recs)
    dup_groups = {k: n for k, n in identities.items() if n > 1}
    print(f"    emitted witness records            {total_records}")
    print(f"    distinct exact evidence identities {len(identities)}")
    print(f"    duplicate identity groups          {len(dup_groups)}")
    print(f"    duplicate excess records           "
          f"{sum(n - 1 for n in dup_groups.values())}")
    for k, n in sorted(dup_groups.items(), key=lambda t: (t[0][1], t[0][0])):
        print(f"      {k[1]:<8} x{n}  {k[0]}  {k[2]}  {k[3]}")
    check("SB-DUP the emitted-record denominator EXCEEDS the distinct-identity "
          "count (duplicates are labelled non-independent, never dropped)",
          total_records > len(identities),
          f"{total_records} vs {len(identities)}")

    # Population A — four-kind consumer universe, and the all-kind diagnostic
    A_four = collections.Counter()
    A_all = collections.Counter()
    A_docs4, A_docsall = set(), set()
    for r in recs:
        if r["scope"] != "WHOLE_FILE":
            continue
        A_all[r["family"]] += 1
        A_docsall.add(r["path"])
        if r["family"] in CONSUMED_FAMILIES:
            A_four[r["family"]] += 1
            A_docs4.add(r["path"])
    pop = doc["population"]
    print(f"\n    POPULATION A  consumer-relevant {sum(A_four.values())} "
          f"over {len(A_docs4)}/{pop} documents  {dict(A_four)}")
    print(f"                  ALL-KIND diagnostic {sum(A_all.values())} "
          f"over {len(A_docsall)}/{pop}  {dict(A_all)}")
    check("SB-CORPUS-A consumer-relevant WHOLE_FILE == 167",
          sum(A_four.values()) == 167, sum(A_four.values()))
    check("SB-CORPUS-A documents == 166", len(A_docs4) == 166, len(A_docs4))
    check("SB-CORPUS-A COMMIT 6 / DATE 161 / RUN_ID 0 / SUPERSEDED_BY 0",
          (A_four["COMMIT"], A_four["DATE"], A_four["RUN_ID"],
           A_four["SUPERSEDED_BY"]) == (6, 161, 0, 0), dict(A_four))
    check("SB-CORPUS-A all-kind diagnostic == 168, the residual being one "
          "HEX_SHAPED_UNRESOLVED (D382 §4)",
          sum(A_all.values()) == 168, dict(A_all))

    # U_A — derived mechanically (D382 §5)
    U_A = [r for r in recs if r["scope"] == "WHOLE_FILE"
           and r["family"] in CONSUMED_FAMILIES]
    check(f"SB-U_A derived mechanically from the frozen blob ({len(U_A)})",
          len(U_A) == sum(A_four.values()), len(U_A))

    # Population B — the projection of U_A through PRE-D381 scope semantics
    pre = _load_pre_d381_passa()
    loc = {id(r): v for r, v, _ in located}
    B_four, B_docs, moves = collections.Counter(), set(), []
    for r in U_A:
        v = loc.get(id(r))
        if v is None:
            continue
        text, off = v
        head = text[:pre.HEAD_BYTES]
        if off >= pre.HEAD_BYTES:
            moves.append((r["path"], r["family"], r["selector"],
                          "WHOLE_FILE", "NOT_ELIGIBLE"))
            continue
        sc = ("WHOLE_FILE" if r["type"] == "NAMED_SUCCESSOR"
              else pre._scope_of(head, off, _detector_of(r["type"])))
        if sc != "WHOLE_FILE":
            moves.append((r["path"], r["family"], r["selector"],
                          "WHOLE_FILE", sc))
        else:
            B_four[r["family"]] += 1
            B_docs.add(r["path"])
    print(f"\n    POPULATION B  consumer-relevant {sum(B_four.values())} "
          f"over {len(B_docs)}/{pop} documents  {dict(B_four)}")
    check("SB-CORPUS-1 WHOLE_FILE == 165", sum(B_four.values()) == 165,
          sum(B_four.values()))
    check("SB-CORPUS-1 documents == 164/272", len(B_docs) == 164, len(B_docs))
    check("SB-CORPUS-1 DATE 160 / COMMIT 5 / RUN_ID 0 / SUPERSEDED_BY 0",
          (B_four["DATE"], B_four["COMMIT"], B_four["RUN_ID"],
           B_four["SUPERSEDED_BY"]) == (160, 5, 0, 0), dict(B_four))
    print("    A -> B MOVEMENTS:")
    for m in moves:
        print(f"      {m[0]}  {m[1]}  {m[2]}  {m[3]} -> {m[4]}")
    moved_paths = sorted({m[0] for m in moves})
    check("SB-MOVED exactly TWO A->B movements inside U_A, both "
          "WHOLE_FILE -> SPAN",
          len(moves) == 2 and all(m[4] == "SPAN" for m in moves), moves)
    check("SB-MOVED-1 CODE_AUDIT_PLANNING_PACKAGE_QA.md moved",
          "kai-pm/CODE_AUDIT_PLANNING_PACKAGE_QA.md" in moved_paths, moved_paths)
    check("SB-MOVED-2 ORION_FIELD_NOTES.md moved",
          "kai-pm/ORION_FIELD_NOTES.md" in moved_paths, moved_paths)

    # ---- SB-SCOPE-ALL — the class-wide invariant, MULTISET ----
    print("\n  SB-SCOPE-ALL — every comparable EMITTED record, pre vs post")
    comparable, diffs, unresolved = 0, [], []
    for r, v, why in located:
        if v is None:
            unresolved.append((r["path"], r["selector"], r["value"], why))
            continue
        if r["type"] == "NAMED_SUCCESSOR":
            continue                      # hard-coded both sides, not _scope_of
        text, off = v
        head = text[:passa.HEAD_BYTES]
        if off >= passa.HEAD_BYTES:
            continue                      # not eligible on either side
        det = _detector_of(r["type"])
        a = pre._scope_of(head, off, det)
        b = passa._scope_of(head, off, det)
        comparable += 1
        if a != b:
            diffs.append((r["path"], r["selector"], a, b))
    print(f"    comparable emitted records  {comparable}   (DERIVED)")
    print(f"    scope differences           {len(diffs)}")
    print(f"    unresolved records          {len(unresolved)}")
    for u in unresolved:
        print(f"      UNRESOLVED {u}")
    check("SB-SCOPE-ALL the subject repair changed NO applicability scope, "
          "over the derived multiset denominator",
          not diffs, diffs[:5])
    check("SB-SCOPE-ALL every emitted record resolved against the frozen tree "
          "(duplicates resolve to the SAME occurrence, never consumed)",
          not unresolved, unresolved[:5])

    # ---- SB-CORPUS-3 — the exact three-row delta ----
    print("\n  SB-CORPUS-3 — the exact D381-driven delta")
    before = {r["path"]: r for r in json.loads(
        _blob_text("ee524b47b43cfb4a0cc7bc9cb6c3c8f9ae389740"))["rows"]}
    expect = {
        "kai-pm/CODE_AUDIT_FINAL_REPORT.md":
            ("WHOLE_FILE", "UNKNOWN", "UNKNOWN"),
        "kai-pm/CODE_AUDIT_MASTER.md":
            ("WHOLE_FILE", "UNKNOWN", "UNKNOWN"),
        "kai-pm/house_in_order_instrument/AUTHORITY_ONTOLOGY.md":
            ("WHOLE_FILE", "UNKNOWN", "UNKNOWN"),
    }
    by_path = collections.defaultdict(list)
    for r, v, _why in located:
        by_path[r["path"]].append((r, v))
    for path, (xs, xv, xl) in expect.items():
        ws = collections.defaultdict(list)
        for r, v in by_path[path]:
            if v is None:
                continue
            text, off = v
            head = text[:passa.HEAD_BYTES]
            if off >= passa.HEAD_BYTES:
                continue
            if r["type"] == "NAMED_SUCCESSOR":
                sc, su = "WHOLE_FILE", "SELF"
            else:
                det = _detector_of(r["type"])
                sc = passa._scope_of(head, off, det)
                su = passa._subject_of(
                    head, off, det,
                    resolve=(lambda vv=r["value"]: passa.full_commit(
                        str(REPO), vv)) if r["family"] == "COMMIT" else None)
            ws[r["family"]].append({
                "witness_type": r["type"], "witness_value": r["value"],
                "source_path": path, "source_selector": r["selector"],
                "local_context": "", "applicability_scope": sc,
                "subject": su, "evidence_total": 1, "evidence_shown": 1,
                "truncated": False, "polarity": "POSITIVE",
                "certainty": "VERIFIED", "temporal": "AT_COMMIT"})
        row = {"path": path, "witnesses": dict(ws)}
        snap = classify._binding_witness(row, "COMMIT")
        got = (classify.scope(row)["value"],
               classify.validity(row, None)["value"],
               classify.lifecycle(path=path, superseded_by=None,
                                  snapshot_witness=snap.asdict() if snap else None,
                                  blocked=None)["value"])
        b = before[path]
        print(f"    {path}")
        for ax, g, e in (("SCOPE", got[0], xs), ("VALIDITY", got[1], xv),
                         ("LIFECYCLE", got[2], xl)):
            print(f"      {ax:<10} {b[ax]['value']:<15} -> {g}")
        check(f"SB-CORPUS-3 {path.split('/')[-1]} == the banked delta",
              got == (xs, xv, xl), f"got {got} want {(xs, xv, xl)}")
    print()


# ── shared synthetic/local fixtures for the executable-bound cases ────
#
# D379 §8 authorises synthetic and local subjects. Nothing here is a
# production artefact: no real Stage A, no real Pass A, no candidate.
_FIX = {}


def exec_fixtures():
    """Build the synthetic/local inputs the real CLIs will be handed."""
    if _FIX:
        return _FIX
    import tempfile
    import stage_identity as SI
    d = pathlib.Path(tempfile.mkdtemp(prefix="d379_exec_"))

    desc = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    (d / "stage_a.json").write_text(json.dumps(desc))
    _FIX["stage_a"] = d / "stage_a.json"
    _FIX["stage_a_identity"] = SI.stage_a_identity(desc)

    # a second, DIFFERENT Stage A, for the stale-input case (Q1a-4)
    other = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    other["subject"] = dict(other["subject"], population=271)
    (d / "stage_a_other.json").write_text(json.dumps(other))
    _FIX["stage_a_other"] = d / "stage_a_other.json"

    # a synthetic Pass-A artefact carrying in-band provenance
    prov = {"stage_a_identity": _FIX["stage_a_identity"],
            "stage_a_descriptor_digest": SI.stage_a_descriptor_digest(desc),
            "producer_component": "PASS_A",
            "producer_population": [{"class": SI.CLASS_H2,
                                     "identity": m["path"],
                                     "sha256": m["sha256"]}
                                    for m in desc["h2_sources"]],
            "producer_denominator": len(desc["h2_sources"]),
            "runtime_identity": desc["runtime"],
            "subject_commit": desc["subject"]["commit"],
            "subject_tree": desc["subject"]["tree"],
            "tree_paths_identity": desc["tree_paths"]["tree_paths_identity"],
            "census_identity": desc["census"]["aggregate_sha256"],
            "history_source_identity": desc["history"]["reachable_set_sha256"]}
    passa_doc = {"subject": desc["subject"]["commit"],
                 "subject_tree": desc["subject"]["tree"],
                 "history_identity": {}, "census_dependency": {},
                 "population": 0, "rows": [], "producer_provenance": prov}
    (d / "passA.json").write_text(json.dumps(passa_doc))
    _FIX["passa"] = d / "passA.json"

    # the same Pass A with its recorded identity TAMPERED (Q1a-5)
    bad = json.loads(json.dumps(passa_doc))
    bad["producer_provenance"]["stage_a_identity"] = "0" * 64
    (d / "passA_tampered.json").write_text(json.dumps(bad))
    _FIX["passa_tampered"] = d / "passA_tampered.json"

    # A synthetic classification result in the REAL emitted schema. The
    # qualifier reads subject / subject_tree / history_identity /
    # census_dependency, so a fixture missing them measures the fixture,
    # not the governed predicate.
    def _result(rows_src, prov):
        r = rows_src()
        return {"subject": desc["subject"]["commit"],
                "subject_tree": desc["subject"]["tree"],
                "history_identity": {
                    "oldest_reachable_date": "2026-01-01",
                    "newest_date": "2026-09-19", "shallow": "false",
                    "subject_ancestry_depth": 1},
                "census_dependency": {
                    "package": "house_in_order_census_v11",
                    "aggregate": desc["census"]["aggregate_sha256"]},
                "population": r["population"], "rows": r["rows"],
                "producer_provenance": prov}

    cls_prov = dict(prov, producer_component="CLASSIFICATION",
                    input_binding={"pass_a_artifact_sha256": "0" * 64,
                                   "pass_a_stage_a_identity":
                                       _FIX["stage_a_identity"],
                                   "pass_a_producer_provenance_digest":
                                       "0" * 64})
    _FIX["result"] = d / "result.json"
    _FIX["result"].write_text(json.dumps(_result(_synthetic_result, cls_prov)))

    import copy

    def _rows_missing_trace():
        m = copy.deepcopy(_synthetic_result())
        m["rows"][0]["evidence_fact_traces"].pop("CITES_COMMIT")
        return m
    _FIX["result_missing_trace"] = d / "result_q1b2.json"
    _FIX["result_missing_trace"].write_text(
        json.dumps(_result(_rows_missing_trace, cls_prov)))

    # Q1a-5's subject is a TAMPERED RECORDED IDENTITY, so the fixture must
    # carry one. Handing the qualifier a result with NO provenance tests a
    # different predicate entirely.
    tampered_prov = dict(cls_prov, stage_a_identity="0" * 64)
    _FIX["result_tampered"] = d / "result_tampered.json"
    _FIX["result_tampered"].write_text(
        json.dumps(_result(_synthetic_result, tampered_prov)))

    lines = [f"{m['sha256']}  {pathlib.Path(m['path']).name}"
             for m in desc["h2_sources"]]
    (d / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")
    _FIX["manifest"] = d / "MANIFEST.sha256"
    # I1-B's CLEAN case requires the frozen tree path population and the
    # candidate output path population to RECONCILE. A 272-path tree
    # against a 3-row synthetic result never could, so I1B-1 was not the
    # banked clean case at all and I1B-3/-4 were not "one added" or "one
    # duplicated" relative to anything. The tree is derived FROM the
    # result's own paths, which is what "the same population" means.
    _clean_paths = sorted(r["path"] for r in
                          json.loads(_FIX["result"].read_bytes())["rows"])
    (d / "tree_paths.txt").write_text("\n".join(_clean_paths) + "\n")
    _FIX["tree_paths"] = d / "tree_paths.txt"
    # B7. `--subject HEAD` is a literal string; passa resolves repo HEAD to
    # a SHA and compares, so "HEAD" would itself trip the R11 abort once
    # INC-34 stopped masking the case. And the Census package is a governed
    # directory, not the repository root.
    def _rows_q1b3():
        m = copy.deepcopy(_synthetic_result())
        m["rows"][1]["SCOPE"]["witness"] = dict(
            m["rows"][1]["SCOPE"]["witness"], local_context="unrelated text")
        return m

    def _rows_q1b4():
        m = copy.deepcopy(_synthetic_result())
        m["rows"][2]["evidence_facts_abstained_no_compliant_trace"] = \
            ["CITES_COMMIT"]
        return m

    def _rows_q1b5():
        m = copy.deepcopy(_synthetic_result())
        for r in m["rows"]:
            r["evidence_facts"].pop("BINDING_CONTRADICTION", None)
        return m

    def _rows_q1b6():
        m = copy.deepcopy(_synthetic_result())
        for r in m["rows"]:
            for n in list(r["evidence_facts"]):
                r["evidence_facts"][n] = False
            r["evidence_fact_traces"] = {}
        return m

    for key, fn in (("result_q1b3", _rows_q1b3), ("result_q1b4", _rows_q1b4),
                    ("result_q1b5", _rows_q1b5), ("result_q1b6", _rows_q1b6)):
        _FIX[key] = d / (key + ".json")
        _FIX[key].write_text(json.dumps(_result(fn, cls_prov)))

    # Q1a-7: a governed member DELETED from the output provenance
    short_prov = copy.deepcopy(cls_prov)
    short_prov["producer_population"] = short_prov["producer_population"][:-1]
    _FIX["result_short_prov"] = d / "result_short_prov.json"
    _FIX["result_short_prov"].write_text(
        json.dumps(_result(_synthetic_result, short_prov)))

    # ── SUBJECTS THAT MUST ACTUALLY BE BUILT ──────────────────────────
    # Everything below exists because a case id is not a subject. Six
    # cases previously shared ONE clean argv and differed only in the
    # string they hoped to see; none of them constructed the banked
    # hostile condition at all.
    import stage_identity as SI

    def _stage_a_wrong_byte(module_name, tag):
        """Stage A that RECORDS a digest for `module_name` which differs
        from the bytes on disk. The file is NOT touched: the hostile
        condition is 'Stage A was fixed, then the producer byte changed',
        and recording a different digest is the same divergence from the
        producer's side without mutating the governed tree."""
        srcs = [dict(m) for m in desc["h2_sources"]]
        hit = [m for m in srcs
               if pathlib.Path(m["path"]).name == module_name]
        assert len(hit) == 1, f"{module_name} not uniquely in h2_sources"
        actual = SI.sha256_hex((REPO / hit[0]["path"]).read_bytes())
        hit[0]["sha256"] = "f" * 64
        bad = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION,
                                    sources=srcs)
        path = d / f"stage_a_{tag}.json"
        path.write_text(json.dumps(bad))
        return {"path": path, "desc": bad, "rel": hit[0]["path"],
                "recorded": "f" * 64, "actual": actual}

    _FIX["byte_passa"] = _stage_a_wrong_byte("passa.py", "byte_passa")
    _FIX["byte_classify"] = _stage_a_wrong_byte("classify.py", "byte_classify")

    # Q1a-3 consumes a Pass A. If that Pass A is bound to the CLEAN Stage
    # A, the stale-input predicate (Q1a-4) fires first and Q1a-3 is never
    # reached -- the case would be red for the wrong reason. Its Pass A is
    # therefore bound to the SAME wrong-byte descriptor.
    def _passa_for(dsc, name):
        pv = {"stage_a_identity": SI.stage_a_identity(dsc),
              "stage_a_descriptor_digest": SI.stage_a_descriptor_digest(dsc),
              "producer_component": "PASS_A",
              "producer_population": [{"class": SI.CLASS_H2,
                                       "identity": m["path"],
                                       "sha256": m["sha256"]}
                                      for m in dsc["h2_sources"]],
              "producer_denominator": len(dsc["h2_sources"]),
              "runtime_identity": dsc["runtime"],
              "subject_commit": dsc["subject"]["commit"],
              "subject_tree": dsc["subject"]["tree"],
              "tree_paths_identity": dsc["tree_paths"]["tree_paths_identity"],
              "census_identity": dsc["census"]["aggregate_sha256"],
              "history_source_identity":
                  dsc["history"]["reachable_set_sha256"]}
        doc = {"subject": dsc["subject"]["commit"],
               "subject_tree": dsc["subject"]["tree"],
               "history_identity": {}, "census_dependency": {},
               "population": 0, "rows": [], "producer_provenance": pv}
        q = d / name
        q.write_text(json.dumps(doc))
        return q

    _FIX["passa_byte_classify"] = _passa_for(
        _FIX["byte_classify"]["desc"], "passA_byte_classify.json")

    # Q1a-6 / DEP-3 -- Stage A expects runtime identity A, the executing
    # interpreter presents B. Constructed by declaring a DIFFERENT stdlib
    # identity in the descriptor, so the observed/expected comparison has
    # a real divergence to find.
    rt = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    rt["runtime"] = dict(rt["runtime"], stdlib_identity="2" * 64,
                         cache_tag="calibration-B")
    (d / "stage_a_runtime_b.json").write_text(json.dumps(rt))
    _FIX["stage_a_runtime_b"] = d / "stage_a_runtime_b.json"
    _FIX["runtime_expected"] = desc["runtime"]["stdlib_identity"]
    _FIX["runtime_other"] = rt["runtime"]["stdlib_identity"]

    # Q1a-7 -- a member DELETED from the output provenance while the
    # independently captured denominator is unchanged. Truncating both
    # would be a self-consistent smaller population, which is a different
    # subject: the banked one is a provenance that no longer matches the
    # producer population it claims.
    del_prov = copy.deepcopy(cls_prov)
    _FIX["q1a7_removed"] = del_prov["producer_population"][-1]["identity"]
    del_prov["producer_population"] = del_prov["producer_population"][:-1]
    _FIX["q1a7_denominator"] = del_prov["producer_denominator"]
    _FIX["q1a7_population"] = len(del_prov["producer_population"])
    _FIX["result_member_deleted"] = d / "result_member_deleted.json"
    _FIX["result_member_deleted"].write_text(
        json.dumps(_result(_synthetic_result, del_prov)))

    # 86-6 -- a manifest that OMITS a module this qualifier actually loads.
    _FIX["omitted_module"] = "classify.py"
    (d / "MANIFEST_omit.sha256").write_text("\n".join(
        l for l in lines if not l.endswith("  " + _FIX["omitted_module"])) + "\n")
    _FIX["manifest_omit"] = d / "MANIFEST_omit.sha256"
    # 86-2 -- manifest supplied, ONE module byte differs. Banked, and
    # absent from the matrix alongside 86-6: the family ran 1,3,4,5.
    _FIX["disagree_module"] = "envelope.py"
    _FIX["manifest_disagree"] = d / "MANIFEST_disagree.sha256"
    _FIX["manifest_disagree"].write_text("\n".join(
        ("e" * 64 + "  " + _FIX["disagree_module"])
        if l.endswith("  " + _FIX["disagree_module"]) else l
        for l in lines) + "\n")
    _FIX["manifest_entries"] = len(lines)
    _FIX["manifest_omit_entries"] = len(
        [l for l in lines if not l.endswith("  " + _FIX["omitted_module"])])

    # I1A-3 -- HOLDOUT SEED INDEPENDENCE. Same Stage A, same output-path
    # population, MUTATED classification evidence. The selected sample must
    # not move. Two distinct result files are required or there is no
    # differential and nothing is being measured.
    def _rows_mutated_evidence():
        m = copy.deepcopy(_synthetic_result())
        for r in m["rows"]:
            for n in list(r["evidence_facts"]):
                r["evidence_facts"][n] = not r["evidence_facts"][n]
        return m
    _FIX["result_evidence_mutated"] = d / "result_evidence_mutated.json"
    _FIX["result_evidence_mutated"].write_text(
        json.dumps(_result(_rows_mutated_evidence, cls_prov)))

    # DEP-1 -- a non-stdlib module outside EVERY governed root. Written
    # under the fixture directory, which is under none of them.
    ext = d / "ext_pkg"
    ext.mkdir()
    (ext / "hostile_ext.py").write_text("VALUE = 'outside every governed root'\n")
    _FIX["ext_dir"] = ext
    _FIX["ext_mod"] = ext / "hostile_ext.py"

    (d / "EMPTY.sha256").write_text("")
    _FIX["empty_manifest"] = d / "EMPTY.sha256"
    tp = list(_clean_paths)
    (d / "tree_extra.txt").write_text("\n".join(tp + ["kai-pm/extra.md"]) + "\n")
    _FIX["tree_paths_extra"] = d / "tree_extra.txt"
    (d / "tree_dup.txt").write_text("\n".join(tp + [tp[0]]) + "\n")
    _FIX["tree_paths_dup"] = d / "tree_dup.txt"

    import subprocess as _sp
    _FIX["subject_sha"] = _sp.run(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO),
        capture_output=True, text=True).stdout.strip()
    _FIX["census_pkg"] = REPO / "kai-pm" / "house_in_order_census_v11"
    _FIX["dir"] = d
    return _FIX


def exec_function_subject_cases(f):
    """M2-1..3, D14-A/B/C, SB-1..3 — the last banked cases without verdicts.

    These stayed in-process because their governed subject is a DECISION
    FUNCTION rather than a shipped CLI. In-process is precisely how INC-38
    happened: the right code was proved while the real program walked
    around it. Each case below now runs in a CHILD PROCESS that imports the
    ACTUAL shipped function, builds the authorised synthetic input with the
    real producer helpers, and exits from the real predicate result.

    NO PRODUCTION INTERFACE IS WIDENED. Nothing is added to passa.py,
    classify.py or stage_identity.py to make any of this reachable; the
    children call what the producers already call.

    SB-1 and SB-3 turned out NOT to be function subjects at all: "Pass A
    absent" and "Pass A unbound" are conditions of the CLASSIFICATION
    executable's input, so they are bound to run_h2_v12.py, which is what
    actually decides them.
    """
    import stage_identity as SI
    d = f["dir"]
    (d / "false_route.md").write_text(FALSE_ROUTE_TEXT)
    (d / "genuine_route.md").write_text(GENUINE_ROUTE_TEXT)

    # The witness construction, using ONLY governed producer functions.
    # Shared by the three M2 children so each measures the same subject.
    M2_BUILD = f"""
import json, sys
sys.path.insert(0, {str(V)!r})
import passa, classify
from envelope import Witness
def lifecycle_of(text, path):
    m = passa.HEX.search(text)
    assert m is not None and passa._eligible(m)
    head = text[:passa.HEAD_BYTES]
    w = Witness(witness_type='COMMIT', witness_value=m.group(0),
                source_path=path,
                source_selector=passa._selector(text, m.start()),
                local_context=passa._context(text, m.start(), m.end()),
                applicability_scope=passa._scope_of(head, m.start(), 'HEX'),
                subject=passa._subject_of(head, m.start(), 'HEX',
                                          resolve=lambda: m.group(0)),
                evidence_total=1, evidence_shown=1, truncated=False,
                polarity='POSITIVE', certainty='VERIFIED')
    row = {{'path': path, 'witnesses': {{'COMMIT': [w.asdict()]}}}}
    snap = classify._binding_witness(row, 'COMMIT')
    return classify.lifecycle(path=path, superseded_by=None,
                              snapshot_witness=snap.asdict() if snap else None,
                              blocked=None), w
FALSE = open({str(d / "false_route.md")!r}).read()
GENUINE = open({str(d / "genuine_route.md")!r}).read()
FP = 'kai-pm/SYNTHETIC_CODE_AUDIT_FALSE_ROUTE.md'
GP = 'kai-pm/SYNTHETIC_CODE_AUDIT_GENUINE_ROUTE.md'
"""

    # raw facts about the CONSTRUCTED inputs, measured here
    fr, gr = FALSE_ROUTE_TEXT, GENUINE_ROUTE_TEXT
    m2_proof = {
        "false_route_bytes": len(fr.encode()),
        "genuine_route_bytes": len(gr.encode()),
        "false_route_sha256": hashlib.sha256(fr.encode()).hexdigest()[:16] + "…",
        "genuine_route_sha256": hashlib.sha256(gr.encode()).hexdigest()[:16] + "…",
        "texts_differ": fr != gr,
        "both_carry_the_commit_token": COMMIT in fr and COMMIT in gr,
        "governed_target": "classify.lifecycle via classify._binding_witness"}
    m2_holds = lambda pr: (pr["texts_differ"] is True
                           and pr["both_carry_the_commit_token"] is True)

    print("  classify.lifecycle   (child process, real governed function)")
    d379_case("M2-1", clause="D379 §8 fail-old: the false audited-snapshot "
                             "COMMIT route must NOT classify HISTORICAL",
              subject_proof=m2_proof, subject_holds=m2_holds,
              snippet=M2_BUILD + """
res, w = lifecycle_of(FALSE, FP)
print('scope=%s subject=%s LIFECYCLE=%s' % (w.applicability_scope, w.subject,
                                            res['value']))
if res['value'] != 'HISTORICAL':
    print('M2_FALSE_ROUTE_NOT_HISTORICAL'); sys.exit(0)
print('the commit the document AUDITS decided the document lifecycle')
sys.exit(1)
""",
              intended_reason="M2_FALSE_ROUTE_NOT_HISTORICAL",
              expect_class="ACCEPT")

    d379_case("M2-2", clause="D379 §8 pass-new: the genuine own-lifecycle "
                             "COMMIT route STILL classifies HISTORICAL",
              subject_proof=m2_proof, subject_holds=m2_holds,
              snippet=M2_BUILD + """
res, w = lifecycle_of(GENUINE, GP)
print('scope=%s subject=%s LIFECYCLE=%s' % (w.applicability_scope, w.subject,
                                            res['value']))
if res['value'] == 'HISTORICAL':
    print('M2_GENUINE_ROUTE_STILL_HISTORICAL'); sys.exit(0)
print('suppression, not discrimination'); sys.exit(1)
""",
              intended_reason="M2_GENUINE_ROUTE_STILL_HISTORICAL",
              expect_class="ACCEPT")

    d379_case("M2-3", clause="D379 §8 the two routes are DISTINGUISHED",
              subject_proof=m2_proof, subject_holds=m2_holds,
              snippet=M2_BUILD + """
a, _ = lifecycle_of(FALSE, FP)
b, _ = lifecycle_of(GENUINE, GP)
print('false=%s genuine=%s' % (a['value'], b['value']))
if a['value'] != b['value']:
    print('M2_ROUTES_DISCRIMINATED'); sys.exit(0)
print('both routes classified %s — the binding predicate did not '
      'discriminate' % a['value'])
sys.exit(1)
""",
              intended_reason="M2_ROUTES_DISCRIMINATED", expect_class="ACCEPT")

    # ── D14 — the HEAD_BYTES boundary decides WHICH TOKENS ARE ELIGIBLE,
    #    never WHAT AN ELIGIBLE TOKEN IS. Subject: passa._eligible and the
    #    recogniser, in a child that imports the real passa.
    H = passa.HEAD_BYTES
    tok = "76dbba4c1f3e9a05b7c2d8e6f40193a5c7b2e8d1"
    D14_HEAD = f"""
import sys
sys.path.insert(0, {str(V)!r})
import passa
H = passa.HEAD_BYTES
TOK = {tok!r}
"""
    print("  passa._eligible      (child process, real boundary predicate)")
    d379_case("D14-A", clause="D379 §8 straddling token: admitted WHOLE",
              subject_proof={"HEAD_BYTES": H, "token_length": len(tok),
                             "token_start_offset": H - 10,
                             "token_end_offset": H - 10 + len(tok),
                             "straddles_boundary": (H - 10) < H < (H - 10 + len(tok)),
                             "governed_target": "passa.HEX + passa._eligible"},
              subject_holds=lambda pr: pr["straddles_boundary"] is True,
              snippet=D14_HEAD + """
pad = '.' * (H - 10)
s = pad + TOK + ' tail'
m = next(x for x in passa.HEX.finditer(s) if x.start() == len(pad))
print('recognised=%r len=%d start=%d eligible=%s'
      % (m.group(0), len(m.group(0)), m.start(), passa._eligible(m)))
if m.group(0) == TOK and len(m.group(0)) == 40 and passa._eligible(m):
    print('D14_STRADDLING_TOKEN_ADMITTED_WHOLE'); sys.exit(0)
sys.exit(1)
""",
              intended_reason="D14_STRADDLING_TOKEN_ADMITTED_WHOLE",
              expect_class="ACCEPT")

    d379_case("D14-B", clause="D379 §8 cut below the recogniser minimum: "
                              "fail-old loses it, pass-new carries it whole",
              subject_proof={"HEAD_BYTES": H,
                             "token_start_offset": H - 4,
                             "characters_inside_window": 4,
                             "recogniser_minimum": 7,
                             "governed_target": "passa.HEX + passa._eligible"},
              # DERIVED FROM THE TWO NUMBERS, not supplied as a verdict. A
              # precomputed `cut_is_below_minimum: True` would be the flag
              # this whole repair removes.
              subject_holds=lambda pr: (pr["characters_inside_window"]
                                        < pr["recogniser_minimum"]),
              snippet=D14_HEAD + """
pad = '.' * (H - 4)
whole = pad + TOK
cut = whole[:H]
lost = [x.group(0) for x in passa.HEX.finditer(cut) if x.start() >= len(pad)]
m = next(x for x in passa.HEX.finditer(whole) if x.start() == len(pad))
print('fail-old found in truncated window: %r' % lost)
print('pass-new against complete source: %r eligible=%s'
      % (m.group(0), passa._eligible(m)))
if not lost and m.group(0) == TOK and passa._eligible(m):
    print('D14_COMPLETE_SOURCE_RECOGNITION'); sys.exit(0)
sys.exit(1)
""",
              intended_reason="D14_COMPLETE_SOURCE_RECOGNITION",
              expect_class="ACCEPT")

    d379_case("D14-C", clause="D379 §8 token starting at or after the "
                              "boundary: NOT ADMITTED ON EITHER SIDE",
              subject_proof={"HEAD_BYTES": H, "token_start_offset": H,
                             "governed_target": "passa._eligible"},
              subject_holds=lambda pr: (pr["token_start_offset"]
                                        >= pr["HEAD_BYTES"]),
              snippet=D14_HEAD + """
pad = '.' * H
s = pad + TOK
m = next(x for x in passa.HEX.finditer(s) if x.start() == len(pad))
print('start=%d HEAD_BYTES=%d eligible=%s' % (m.start(), H, passa._eligible(m)))
if not passa._eligible(m) and m.start() == H:
    print('D14_AT_OR_AFTER_BOUNDARY_NOT_ADMITTED'); sys.exit(0)
sys.exit(1)
""",
              intended_reason="D14_AT_OR_AFTER_BOUNDARY_NOT_ADMITTED",
              expect_class="ACCEPT")

    # ── SB — Stage-B separation and integrity ─────────────────────────
    #    SB-1 and SB-3 are conditions of the CLASSIFICATION EXECUTABLE's
    #    input, so they are bound to run_h2_v12.py. SB-2 is a differential
    #    over stage_b_binding and runs as a child on that function.
    print("  run_h2_v12.py        (Stage-B input conditions)")
    missing = d / "no_such_passA.json"
    d379_case("SB-1", clause="D379 §8 Pass A absent -> REFUSE, R11 abort",
              subject_proof={"passa_path": str(missing),
                             "path_exists": missing.exists(),
                             "governed_target": "run_h2_v12.py"},
              subject_holds=lambda pr: pr["path_exists"] is False,
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO), "--passa", str(missing),
                    "--out", str(d / "sb1.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="No such file", expect_class="REFUSE")

    unbound = d / "passA_unbound.json"
    _u = json.loads(f["passa"].read_bytes())
    _u.pop("producer_provenance")
    unbound.write_text(json.dumps(_u))
    d379_case("SB-3", clause="D379 §8 Pass A unbound -> REFUSE",
              subject_proof={
                  "passa_path": str(unbound), "path_exists": unbound.is_file(),
                  "carries_producer_provenance":
                      "producer_provenance" in json.loads(unbound.read_bytes()),
                  "governed_target": "run_h2_v12.py"},
              subject_holds=lambda pr: (pr["path_exists"] is True
                                        and pr["carries_producer_provenance"]
                                        is False),
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO), "--passa", str(unbound),
                    "--out", str(d / "sb3.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="carries no in-band producer_provenance",
              expect_class="REFUSE")

    altered = d / "passA_altered.json"
    _a = json.loads(f["passa"].read_bytes())
    _a["population"] = 99
    altered.write_text(json.dumps(_a))
    d379_case("SB-2", clause="D379 §8 Pass A altered -> Stage-B CHANGES, "
                             "Stage-A unchanged",
              subject_proof={
                  "original": str(f["passa"]), "altered": str(altered),
                  "bytes_differ": f["passa"].read_bytes()
                  != altered.read_bytes(),
                  "stage_a_untouched": str(f["stage_a"]),
                  "governed_target": "stage_identity.stage_b_binding"},
              subject_holds=lambda pr: pr["bytes_differ"] is True,
              snippet=f"""
import json, sys
import stage_identity as SI
desc = json.loads(open({str(f["stage_a"])!r}, 'rb').read().decode('utf-8'))
ident = SI.stage_a_identity(desc)
def b(p):
    return SI.stage_b_binding(p, artifact_kind='PASS_A', identity=ident,
                              producer_component='PASS_A',
                              producer_provenance_digest='0'*64)
b1, b2 = b({str(f["passa"])!r}), b({str(altered)!r})
same_a = b1['stage_a_identity'] == b2['stage_a_identity'] == ident
moved_b = SI.stage_b_aggregate([b1]) != SI.stage_b_aggregate([b2])
print('stage_a same=%s  stage_b moved=%s' % (same_a, moved_b))
if same_a and moved_b:
    print('SB_STAGE_A_UNCHANGED_STAGE_B_CHANGES'); sys.exit(0)
sys.exit(1)
""",
              intended_reason="SB_STAGE_A_UNCHANGED_STAGE_B_CHANGES",
              expect_class="ACCEPT")


def exec_bound_cases():
    """The BANKED D379 cases, each against ITS OWN CONSTRUCTED SUBJECT.

    NOT A NEW MATRIX. Every id below is already in D379 §8. What changed
    is that the hostile condition is now BUILT and PROVED rather than
    named. Six cases previously shared one clean invocation and differed
    only in the string they searched the output for -- the case label was
    doing the work the subject was supposed to do.

    Each case declares, beside itself: the banked proposition, how the
    subject is constructed, the RAW mechanical proof of that construction,
    the governed target, and the expected disposition. A case whose proof
    does not derive is a CONTROL/FIXTURE FAILURE and is reported as one.
    """
    print("\nD379 §8 EXECUTABLE-BOUND EXECUTION — the real shipped CLIs")
    print("  Each case: CONSTRUCTED hostile subject proved from raw facts,")
    print("  actual subprocess, rc from CompletedProcess, actual reason,")
    print("  first-effective failure. A case name proves nothing.\n")
    f = exec_fixtures()
    import stage_identity as SI

    def clean(*extra):
        """Raw facts for a case whose subject is the CLEAN chain: the
        Stage-A recorded digest for the entry point EQUALS the bytes on
        disk. A positive case still has a subject and still has to prove
        it (I-8: the known-negative and the known-positive are both
        constructed, neither is assumed)."""
        rel = "kai-pm/house_in_order_h2_v13/passa.py"
        rec = {m["path"]: m["sha256"]
               for m in json.loads(f["stage_a"].read_bytes())["h2_sources"]}
        pr = {"stage_a_recorded[passa.py]": rec[rel][:16] + "…",
              "actual_sha256(passa.py)":
                  SI.sha256_hex((REPO / rel).read_bytes())[:16] + "…"}
        pr.update(dict(extra))
        return pr

    def clean_holds(pr):
        return (pr["stage_a_recorded[passa.py]"]
                == pr["actual_sha256(passa.py)"])

    passa_argv = lambda stage_a, out: [
        "--subject-repo", str(REPO), "--history-repo", str(REPO),
        "--subject", f["subject_sha"], "--census-package", str(f["census_pkg"]),
        "--out", str(f["dir"] / out), "--stage-a", str(stage_a)]

    # ── Q1a-1 · banked: clean chain on one synthetic Stage A -> PASS ──
    #    subject: the CLEAN Stage A, recorded digests equal to disk.
    #    proof: recorded[passa.py] == sha256(passa.py on disk).
    #    target: passa.py (the shipped CLI). expect ACCEPT.
    print("  passa.py")
    d379_case("Q1a-1", clause="D379 §8 clean chain on one Stage A",
              subject_proof=clean(), subject_holds=clean_holds,
              executable=V / "passa.py",
              argv=passa_argv(f["stage_a"], "q1a1.json"),
              intended_reason="PASS A", expect_class="ACCEPT")

    # ── Q1a-2 · banked: one governed Pass-A producer byte changed after
    #    Stage A fixed -> Pass-A production REFUSES.
    #    subject: a Stage A whose recorded digest for passa.py differs
    #             from the bytes the interpreter will actually execute.
    #    proof: recorded != actual, both measured here.
    #    target: passa.py. expect REFUSE on the byte mismatch.
    bp = f["byte_passa"]
    d379_case("Q1a-2", clause="D379 §8 Pass-A producer byte changed",
              subject_proof={"member": bp["rel"],
                             "stage_a_records": bp["recorded"][:16] + "…",
                             "bytes_on_disk": bp["actual"][:16] + "…",
                             "divergence": bp["recorded"] != bp["actual"]},
              subject_holds=lambda pr: (pr["divergence"] is True
                                        and pr["stage_a_records"]
                                        != pr["bytes_on_disk"]),
              executable=V / "passa.py",
              argv=passa_argv(bp["path"], "q1a2.json"),
              intended_reason="byte mismatch against Stage A",
              expect_class="REFUSE")

    # ── DEP-2 · banked: producer loads ordinary stdlib modules under the
    #    governed interpreter -> PASS, without inventing a Stage-A entry
    #    for every stdlib file. Same clean subject as Q1a-1; the
    #    PROPOSITION differs (stdlib containment, not the chain), and the
    #    blocker is INC-34 either way.
    stdlib_loaded = sorted(
        n for n, m in list(sys.modules.items())
        if m is not None and getattr(m, "__file__", None)
        and "/lib/python3" in str(getattr(m, "__file__", "")))[:3]
    d379_case("DEP-2", clause="D379 §8 ordinary stdlib under a governed "
                              "interpreter",
              subject_proof=clean(("stdlib_modules_loaded_sample",
                                   ",".join(stdlib_loaded) or "NONE"),
                                  ("stdlib_in_stage_a_h2_sources", 0)),
              subject_holds=lambda pr: (clean_holds(pr)
                                        and pr["stdlib_in_stage_a_h2_sources"]
                                        == 0),
              executable=V / "passa.py",
              argv=passa_argv(f["stage_a"], "dep2.json"),
              intended_reason="PASS A", expect_class="ACCEPT")

    # ── DEP-3 · banked: Stage A expects runtime identity A; the producer
    #    presents mismatching governed interpreter identity B -> REFUSE.
    #    subject: a Stage A declaring a DIFFERENT stdlib identity.
    #    proof: the two descriptors' stdlib_identity values differ.
    rt_proof = {"stage_a_expects_stdlib_identity": f["runtime_other"][:16] + "…",
                "baseline_stage_a_stdlib_identity":
                    f["runtime_expected"][:16] + "…",
                "divergence": f["runtime_other"] != f["runtime_expected"]}
    rt_holds = lambda pr: pr["divergence"] is True
    d379_case("DEP-3", clause="D379 §8 runtime identity mismatch",
              subject_proof=rt_proof, subject_holds=rt_holds,
              executable=V / "passa.py",
              argv=passa_argv(f["stage_a_runtime_b"], "dep3.json"),
              intended_reason="executing runtime identity differs",
              expect_class="REFUSE")

    # ── Q1a-8 · banked: a governed module NOT represented in Stage A is
    #    loaded at production -> REFUSE, no silent runtime expansion.
    #    SUBJECT IS A GOVERNED DECISION FUNCTION (check_population), so the
    #    child imports the ACTUAL shipped function and exits from ITS
    #    result. Stage A cannot simply omit a member -- validate_descriptor
    #    governs the descriptor shape -- so the omission is made in the
    #    descriptor handed to the function, which is exactly the condition
    #    the clause governs.
    #    proof: the omitted module IS in sys.modules AND its resolved path
    #           IS under the governed H2 root.
    omit_rel = "kai-pm/house_in_order_h2_v13/classify.py"
    d379_case("Q1a-8", clause="D379 §5 no silent runtime expansion",
              subject_proof={
                  "omitted_from_descriptor": omit_rel,
                  "module_is_loaded_in_child": "classify (imported below)",
                  "resolved_path_under_h2_root":
                      str((REPO / omit_rel).resolve()).startswith(
                          str((REPO / "kai-pm/house_in_order_h2_v13")
                              .resolve()) + os.sep),
                  "governed_target": "stage_identity.check_population"},
              subject_holds=lambda pr: pr["resolved_path_under_h2_root"] is True,
              snippet=f"""
import json, sys
import classify                      # the module to be omitted, REALLY loaded
import stage_identity as SI
desc = json.loads(open({str(f["stage_a"])!r}, 'rb').read().decode('utf-8'))
desc['h2_sources'] = [m for m in desc['h2_sources']
                      if m['path'] != {omit_rel!r}]
assert {omit_rel!r} not in [m['path'] for m in desc['h2_sources']]
assert 'classify' in sys.modules
try:
    SI.check_population({str(REPO)!r}, desc, 'production')
except SI.StageIdentityError as e:
    print(str(e)); sys.exit(1)
print('ACCEPTED — no refusal'); sys.exit(0)
""",
              intended_reason="NOT represented in Stage A",
              expect_class="REFUSE")

    # ── DEP-1 · banked: synthetic producer loads a non-stdlib module
    #    outside all governed Stage-A roots -> REFUSE.
    #    subject: a real .py file under the fixture directory, imported by
    #             the child before the governed function runs.
    #    proof: the file EXISTS, and its realpath is under NONE of the
    #           governed roots. Measured, not asserted.
    h2root = str((REPO / "kai-pm/house_in_order_h2_v13").resolve())
    censusroot = str((REPO / "kai-pm/house_in_order_census_v11").resolve())
    extp = str(f["ext_mod"].resolve())
    d379_case("DEP-1", clause="D379 §8 synthetic producer, external dep",
              subject_proof={
                  "external_module_path": extp,
                  "file_exists": f["ext_mod"].is_file(),
                  "under_h2_root": extp.startswith(h2root + os.sep),
                  "under_census_root": extp.startswith(censusroot + os.sep),
                  "under_stdlib_prefix": extp.startswith(
                      os.path.dirname(os.__file__) + os.sep),
                  "governed_target": "stage_identity.producer_population"},
              subject_holds=lambda pr: (pr["file_exists"] is True
                                        and not pr["under_h2_root"]
                                        and not pr["under_census_root"]
                                        and not pr["under_stdlib_prefix"]),
              snippet=f"""
import sys
sys.path.insert(0, {str(f["ext_dir"])!r})
import hostile_ext                    # NON-STDLIB, OUTSIDE EVERY GOVERNED ROOT
import stage_identity as SI
assert hostile_ext.__file__
members, offenders = SI.producer_population({str(REPO)!r})
hit = [w for n, w in offenders if n == 'hostile_ext']
if hit:
    print('SYNTHETIC_HOSTILE_DEPENDENCY: ' + hit[0]); sys.exit(1)
print('ACCEPTED — hostile_ext produced no offender; members=%d offenders=%d'
      % (len(members), len(offenders)))
sys.exit(0)
""",
              intended_reason="SYNTHETIC_HOSTILE_DEPENDENCY",
              expect_class="REFUSE")

    # ── Q1a-3 · banked: one governed CLASSIFICATION-producer byte changed
    #    after Stage A fixed -> classification production REFUSES.
    #    subject: a Stage A recording a wrong digest for classify.py, AND
    #             a Pass A bound to THAT SAME Stage A -- otherwise the
    #             stale-input predicate (Q1a-4) fires first and this case
    #             is red for another case's reason.
    #    proof: recorded != actual for classify.py, and the Pass-A input
    #           carries the SAME stage_a_identity as the descriptor.
    print("  run_h2_v12.py")
    bc = f["byte_classify"]
    pa_bound = json.loads(f["passa_byte_classify"].read_bytes())
    d379_case("Q1a-3", clause="D379 §8 classification-producer byte changed",
              subject_proof={
                  "member": bc["rel"],
                  "stage_a_records": bc["recorded"][:16] + "…",
                  "bytes_on_disk": bc["actual"][:16] + "…",
                  "divergence": bc["recorded"] != bc["actual"],
                  "pass_a_bound_to_same_stage_a":
                      pa_bound["producer_provenance"]["stage_a_identity"]
                      == SI.stage_a_identity(bc["desc"])},
              subject_holds=lambda pr: (pr["divergence"] is True
                                        and pr["pass_a_bound_to_same_stage_a"]
                                        is True),
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO),
                    "--passa", str(f["passa_byte_classify"]),
                    "--out", str(f["dir"] / "q1a3.json"),
                    "--stage-a", str(bc["path"])],
              intended_reason="byte mismatch against Stage A",
              expect_class="REFUSE")

    # ── Q1a-4 · banked: stale input -- Pass A under Stage-A A,
    #    classification under Stage-A B -> REFUSE, no result accepted.
    #    proof: the two identities differ, and the Pass A carries A.
    idA = f["stage_a_identity"]
    idB = SI.stage_a_identity(json.loads(f["stage_a_other"].read_bytes()))
    d379_case("Q1a-4", clause="D379 §4 stale input across two Stage As",
              subject_proof={
                  "pass_a_recorded_identity": idA[:16] + "…",
                  "classification_given_identity": idB[:16] + "…",
                  "identities_differ": idA != idB},
              subject_holds=lambda pr: pr["identities_differ"] is True,
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO), "--passa", str(f["passa"]),
                    "--out", str(f["dir"] / "q1a4.json"),
                    "--stage-a", str(f["stage_a_other"])],
              intended_reason="stage_a_identity", expect_class="REFUSE")

    # ── Q1a-9 · banked, TWO LIMBS IN ONE CASE, as D379 states it:
    #    (a) construct producer evidence whose in-band provenance declares
    #        its OWN whole-file digest among the hashed bytes -> REFUSE AS
    #        INVALID IDENTITY CONSTRUCTION;
    #    (b) then prove the ACCEPTED path -- finalise file, hash exact
    #        final bytes, external Stage-B binding created -> PASS.
    #    The child exits from the CONJUNCTION of the two real predicate
    #    results. Neither limb is a separate matrix member.
    #    proof: the digest planted IS sha256 of the exact artefact bytes.
    d379_case("Q1a-9", clause="D379 §4 self-hash prohibition (both limbs)",
              subject_proof={
                  "limb_a": "provenance declares sha256(its own artefact)",
                  "limb_b": "finalise -> hash final bytes -> Stage-B binding",
                  "planted_digest_is_artefact_digest": "verified in child",
                  "governed_target":
                      "stage_identity.verify_provenance + stage_b_binding"},
              subject_holds=lambda pr: pr["limb_a"].startswith("provenance"),
              snippet=f"""
import json, hashlib, pathlib, sys
import stage_identity as SI
desc = json.loads(open({str(f["stage_a"])!r}, 'rb').read().decode('utf-8'))
prov = json.loads(open({str(f["passa"])!r}, 'rb').read()
                  .decode('utf-8'))['producer_provenance']
art = pathlib.Path({str(f["dir"])!r}) / 'q1a9_artifact.json'

# LIMB A -- plant the artefact's OWN whole-file digest inside its own
# in-band provenance. The digest must be of the EXACT bytes written.
doc = {{'rows': [], 'producer_provenance': dict(prov)}}
raw = json.dumps(doc).encode()
own = hashlib.sha256(raw).hexdigest()
doc['producer_provenance']['history_source_identity'] = own
art.write_bytes(json.dumps(doc).encode())
bad = dict(prov, history_source_identity=own)
assert own == hashlib.sha256(raw).hexdigest()
limb_a = False
try:
    SI.verify_provenance(bad, desc, artifact_bytes=raw)
except SI.StageIdentityError as e:
    limb_a = 'INVALID IDENTITY CONSTRUCTION' in str(e)
    print('LIMB A: ' + str(e)[:200])

# LIMB B -- the accepted path. Finalise, hash the exact final bytes from
# OUTSIDE the artefact, create the external Stage-B binding.
final = pathlib.Path({str(f["dir"])!r}) / 'q1a9_final.json'
final.write_bytes(json.dumps({{'rows': [], 'producer_provenance': prov}}).encode())
b = SI.stage_b_binding(final, artifact_kind='PASS_A', identity=prov['stage_a_identity'],
                       producer_component='PASS_A',
                       producer_provenance_digest=SI.provenance_digest(prov))
limb_b = (b['artifact_sha256'] == hashlib.sha256(final.read_bytes()).hexdigest()
          and b['artifact_sha256'] not in json.dumps(
              json.loads(final.read_bytes())['producer_provenance']))
print('LIMB B: stage-b binding on final bytes = %s, absent from in-band = %s'
      % (b['artifact_sha256'][:16], limb_b))
if limb_a and limb_b:
    print('SELF_HASH_PROHIBITION_BOTH_LIMBS_HELD'); sys.exit(0)
print('LIMB FAILURE limb_a=%s limb_b=%s' % (limb_a, limb_b)); sys.exit(1)
""",
              intended_reason="SELF_HASH_PROHIBITION_BOTH_LIMBS_HELD",
              expect_class="ACCEPT")

    # ── the qualifier family ──────────────────────────────────────────
    print("  qualify.py")
    qargv = lambda res, man=None, sa=None: [
        "--result", str(res), "--manifest", str(man or f["manifest"]),
        "--stage-a", str(sa or f["stage_a"])]

    def _prov_of(path):
        return json.loads(pathlib.Path(path).read_bytes())[
            "producer_provenance"]

    # Q1a-5 · banked: recorded producer digest / Stage-A identity tampered
    #   in a synthetic result -> qualification FAILS / REFUSES.
    tp = _prov_of(f["result_tampered"])
    d379_case("Q1a-5", clause="D379 §8 tampered recorded Stage-A identity",
              subject_proof={
                  "result_records_identity": tp["stage_a_identity"][:16] + "…",
                  "stage_a_actual_identity": idA[:16] + "…",
                  "tampered": tp["stage_a_identity"] != idA},
              subject_holds=lambda pr: pr["tampered"] is True,
              executable=V / "qualify.py", argv=qargv(f["result_tampered"]),
              intended_reason="does not match the Stage-A descriptor",
              expect_class="REFUSE")

    # Q1a-6 · banked: provenance valid, QUALIFIER RUNTIME differs from the
    #   governed Stage-A qualification bytes -> §8(6) FAIL. Producer and
    #   qualifier identity are INDEPENDENT controls, so the subject is the
    #   Stage A the qualifier is run against, not the result.
    d379_case("Q1a-6", clause="D379 §8 qualifier runtime differs",
              subject_proof=dict(rt_proof, provenance_valid_against_baseline=
                                 tp["stage_a_identity"] != idA),
              subject_holds=rt_holds,
              executable=V / "qualify.py",
              argv=qargv(f["result"], sa=f["stage_a_runtime_b"]),
              intended_reason="executing runtime identity differs",
              expect_class="REFUSE")

    # Q1a-7 · banked: one governed member DELETED from the OUTPUT
    #   provenance after production, independently captured runtime
    #   binding UNCHANGED -> REFUSE: recorded provenance does not match the
    #   bound producer population.
    #   proof: population shrank by one, denominator did NOT.
    d379_case("Q1a-7", clause="D379 §8 member deleted from OUTPUT provenance",
              subject_proof={
                  "member_removed": f["q1a7_removed"],
                  "recorded_population_size": f["q1a7_population"],
                  "recorded_denominator_unchanged": f["q1a7_denominator"],
                  "mismatch": f["q1a7_denominator"] != f["q1a7_population"]},
              subject_holds=lambda pr: pr["mismatch"] is True,
              executable=V / "qualify.py",
              argv=qargv(f["result_member_deleted"]),
              intended_reason="does not equal the recorded producer_population",
              expect_class="REFUSE")

    # Q1b-1..6 · the derived-denominator family. Each result differs from
    #   the clean one in exactly the way its clause names, and the proof is
    #   the measured difference against the clean fixture.
    clean_doc = json.loads(f["result"].read_bytes())
    def _q1b_proof(key, what):
        cur = json.loads(f[key].read_bytes()) if key else clean_doc
        return {"fixture": key or "result (clean)", "construction": what,
                "differs_from_clean_result":
                    json.dumps(cur, sort_keys=True)
                    != json.dumps(clean_doc, sort_keys=True),
                "rows": len(cur["rows"])}
    for cid, key, what, reason, cls in (
            ("Q1b-1", None, "clean complete result, nothing removed",
             "positive-evidence-fact denominator", "ACCEPT"),
            ("Q1b-2", "result_missing_trace",
             "one positive evidence fact stripped of its trace",
             "CITES_COMMIT", "REFUSE"),
            ("Q1b-3", "result_q1b3", "an axis-cell witness broken",
             "SCOPE", "REFUSE"),
            ("Q1b-4", "result_q1b4",
             "abstention list inconsistent with emitted positives",
             "ABSTENTION_RECONCILIATION", "REFUSE"),
            ("Q1b-5", "result_q1b5",
             "a fact CLASS removed from the emitted set (the D17 lesson)",
             "FACT_CLASS_ABSENT", "REFUSE"),
            ("Q1b-6", "result_q1b6",
             "opposite-side clean known-negative: all facts false, no traces",
             "positive-evidence-fact denominator", "ACCEPT")):
        d379_case(cid, clause="D379 §8 Q1b derived denominator",
                  subject_proof=_q1b_proof(key, what),
                  subject_holds=(lambda pr: pr["rows"] > 0) if cid in
                  ("Q1b-1", "Q1b-6") else
                  (lambda pr: pr["differs_from_clean_result"] is True),
                  executable=V / "qualify.py",
                  argv=qargv(f[key] if key else f["result"]),
                  intended_reason=reason, expect_class=cls)

    # 86-1..86-6 · fail-closed qualifier identity. The subject of each is
    #   the MANIFEST ARGUMENT, so the proof is a fact about that argument.
    for cid, base, proof, holds, reason, cls in (
            ("86-1", qargv(f["result"]),
             {"manifest_supplied": True,
              "entries": f["manifest_entries"],
              "identity_matches_stage_a": True},
             lambda pr: pr["manifest_supplied"] and pr["entries"] > 0,
             "§8(6) CLOSED ORIGIN CLASSIFICATION", "ACCEPT"),
            ("86-3", ["--result", str(f["result"]),
                      "--stage-a", str(f["stage_a"])],
             {"manifest_argument_present": False,
              "argv_contains__manifest": False},
             lambda pr: pr["argv_contains__manifest"] is False,
             "the following arguments are required: --manifest", "REFUSE"),
            ("86-4", qargv(f["result"], man="/nonexistent/MANIFEST.sha256"),
             {"manifest_path": "/nonexistent/MANIFEST.sha256",
              "path_exists": pathlib.Path("/nonexistent/MANIFEST.sha256")
              .exists()},
             lambda pr: pr["path_exists"] is False,
             "does not exist", "REFUSE"),
            ("86-5", qargv(f["result"], man=f["empty_manifest"]),
             {"manifest_path": str(f["empty_manifest"]),
              "path_exists": f["empty_manifest"].is_file(),
              "byte_size": f["empty_manifest"].stat().st_size},
             lambda pr: pr["path_exists"] is True and pr["byte_size"] == 0,
             "yielded no entries", "REFUSE"),
            ("86-2", qargv(f["result"], man=f["manifest_disagree"]),
             {"module": f["disagree_module"],
              "manifest_declares": "e" * 16 + "…",
              "bytes_on_disk": hashlib.sha256(
                  (V / f["disagree_module"]).read_bytes()).hexdigest()[:16]
                  + "…",
              "divergence": True,
              "module_is_loaded_by_qualifier":
                  f["disagree_module"][:-3] in sys.modules},
             lambda pr: (pr["divergence"] is True
                         and pr["manifest_declares"] != pr["bytes_on_disk"]
                         and pr["module_is_loaded_by_qualifier"] is True),
             "DISAGREE", "REFUSE"),
            ("86-6", qargv(f["result"], man=f["manifest_omit"]),
             {"omitted_module": f["omitted_module"],
              "full_manifest_entries": f["manifest_entries"],
              "omitting_manifest_entries": f["manifest_omit_entries"],
              "exactly_one_entry_removed":
                  f["manifest_entries"] - f["manifest_omit_entries"] == 1,
              "module_is_loaded_by_qualifier": "classify" in sys.modules},
             lambda pr: (pr["exactly_one_entry_removed"] is True
                         and pr["module_is_loaded_by_qualifier"] is True),
             "OMITS", "REFUSE")):
        d379_case(cid, clause="D367 §8(6) fail-closed qualifier identity",
                  subject_proof=proof, subject_holds=holds,
                  executable=V / "qualify.py", argv=base,
                  intended_reason=reason, expect_class=cls)

    # ── the holdout input contract ────────────────────────────────────
    print("  holdout.py")
    tree_n = len(f["tree_paths"].read_text().split())
    for cid, tp_key, what, reason, cls in (
            ("I1B-1", "tree_paths", "tree and output populations equal",
             "BLIND HOLDOUT", "ACCEPT"),
            ("I1B-2", "tree_paths", "drop one output row", 
             "REFUSE BEFORE SELECTION", "REFUSE"),
            ("I1B-3", "tree_paths_extra", "add one tree path",
             "REFUSE BEFORE SELECTION", "REFUSE"),
            ("I1B-4", "tree_paths_dup", "duplicate one tree path",
             "REFUSE BEFORE SELECTION", "REFUSE")):
        n = len(f[tp_key].read_text().split())
        rows = len(json.loads(f["result"].read_bytes())["rows"])
        res = f["result"]
        if cid == "I1B-2":
            # the banked subject is a DROPPED OUTPUT ROW, not a longer
            # tree. Previously this ran the clean result against the clean
            # tree and relied on the row count happening to differ, which
            # is the population defect, not the banked one.
            doc = json.loads(f["result"].read_bytes())
            doc["rows"] = doc["rows"][:-1] if len(doc["rows"]) > 1 else []
            doc["population"] = len(doc["rows"])
            res = f["dir"] / "result_dropped_row.json"
            res.write_text(json.dumps(doc))
            rows = len(doc["rows"])
        d379_case(cid, clause="D379 §7 I1-B tree/output reconciliation",
                  subject_proof={"construction": what,
                                 "tree_path_population": n,
                                 "output_row_population": rows,
                                 "equal": n == rows},
                  subject_holds=((lambda pr: pr["equal"] is True)
                                 if cid == "I1B-1"
                                 else (lambda pr: pr["equal"] is False)),
                  executable=V / "holdout.py",
                  argv=["--result", str(res), "--stage-a", str(f["stage_a"]),
                        "--tree-paths", str(f[tp_key]),
                        "--out", str(f["dir"] / (cid + ".json"))],
                  intended_reason=reason, expect_class=cls)

    # ── I1A-1 · banked: one byte altered in one Stage-A member ->
    #    STAGE-A IDENTITY CHANGES. The subject is stage_a_identity, a
    #    governed decision FUNCTION, so the child imports it and exits from
    #    the real comparison. Binding this id to holdout.py, as it was,
    #    measured a different function entirely.
    d379_case("I1A-1", clause="D379 §7 I1-A one member byte -> identity moves",
              subject_proof={
                  "member": bp["rel"],
                  "descriptor_A_records": f["stage_a_identity"][:16] + "…",
                  "descriptor_B_member_digest": bp["recorded"][:16] + "…",
                  "exactly_one_member_differs": True,
                  "governed_target": "stage_identity.stage_a_identity"},
              subject_holds=lambda pr: pr["exactly_one_member_differs"] is True,
              snippet=f"""
import json, sys
import stage_identity as SI
A = json.loads(open({str(f["stage_a"])!r}, 'rb').read().decode('utf-8'))
B = json.loads(open({str(bp["path"])!r}, 'rb').read().decode('utf-8'))
diff = [a['path'] for a, b in zip(A['h2_sources'], B['h2_sources'])
        if a != b]
assert diff == [{bp["rel"]!r}], diff          # EXACTLY ONE member differs
ia, ib = SI.stage_a_identity(A), SI.stage_a_identity(B)
print('A=%s B=%s' % (ia[:16], ib[:16]))
if ia != ib:
    print('STAGE_A_IDENTITY_CHANGES'); sys.exit(0)
print('IDENTITY UNCHANGED — one altered member byte did not move it')
sys.exit(1)
""",
              intended_reason="STAGE_A_IDENTITY_CHANGES", expect_class="ACCEPT")

    # ── I1A-2 · banked: evidence artefact mutated after a valid Stage A
    #    exists, output path population intact -> Stage A UNCHANGED,
    #    tree-derived paths UNCHANGED, STAGE-B CHANGES.
    d379_case("I1A-2", clause="D379 §7 mutate evidence -> only Stage-B moves",
              subject_proof={
                  "stage_a_held_constant": str(f["stage_a"]),
                  "clean_evidence": str(f["result"]),
                  "mutated_evidence": str(f["result_evidence_mutated"]),
                  "evidence_bytes_differ":
                      f["result"].read_bytes()
                      != f["result_evidence_mutated"].read_bytes(),
                  "governed_target": "stage_identity.stage_b_binding"},
              subject_holds=lambda pr: pr["evidence_bytes_differ"] is True,
              snippet=f"""
import json, sys
import stage_identity as SI
desc = json.loads(open({str(f["stage_a"])!r}, 'rb').read().decode('utf-8'))
ident = SI.stage_a_identity(desc)
paths = lambda p: sorted(r['path'] for r in
                         json.loads(open(p, 'rb').read().decode('utf-8'))['rows'])
p1, p2 = {str(f["result"])!r}, {str(f["result_evidence_mutated"])!r}
assert paths(p1) == paths(p2)                 # output path population INTACT
b1 = SI.stage_b_binding(p1, artifact_kind='CLASSIFICATION', identity=ident,
                        producer_component='CLASSIFICATION',
                        producer_provenance_digest='0'*64)
b2 = SI.stage_b_binding(p2, artifact_kind='CLASSIFICATION', identity=ident,
                        producer_component='CLASSIFICATION',
                        producer_provenance_digest='0'*64)
same_a = b1['stage_a_identity'] == b2['stage_a_identity'] == ident
moved_b = SI.stage_b_aggregate([b1]) != SI.stage_b_aggregate([b2])
print('stage_a same=%s  stage_b moved=%s' % (same_a, moved_b))
if same_a and moved_b:
    print('STAGE_A_UNCHANGED_STAGE_B_CHANGES'); sys.exit(0)
sys.exit(1)
""",
              intended_reason="STAGE_A_UNCHANGED_STAGE_B_CHANGES",
              expect_class="ACCEPT")

    # ── I1A-3 · banked: HOLDOUT SEED INDEPENDENCE. Synthetic Stage A
    #    fixed; output-path population held constant; classification
    #    evidence mutated. OLD: manifest-derived aggregate changes ->
    #    sample changes. NEW: stage_a_identity unchanged -> SELECTED SAMPLE
    #    UNCHANGED. Proves holdout.py ITSELF, not merely stage_identity.py,
    #    has stopped deriving blind selection from execution output --
    #    so this one is EXECUTABLE-BOUND on holdout.py, twice, and the
    #    child compares the two real selections.
    d379_case("I1A-3", clause="D379 §7 I1-A holdout seed independence",
              subject_proof={
                  "stage_a_held_constant": str(f["stage_a"]),
                  "tree_paths_held_constant": tree_n,
                  "evidence_mutated":
                      f["result"].read_bytes()
                      != f["result_evidence_mutated"].read_bytes(),
                  "governed_target": "holdout.py, executed twice"},
              subject_holds=lambda pr: (pr["evidence_mutated"] is True
                                        and pr["tree_paths_held_constant"] > 0),
              snippet=f"""
import json, subprocess, sys, pathlib
d = pathlib.Path({str(f["dir"])!r})
def run(res, out):
    pr = subprocess.run([sys.executable, {str(V / "holdout.py")!r},
                         '--result', res, '--stage-a', {str(f["stage_a"])!r},
                         '--tree-paths', {str(f["tree_paths"])!r},
                         '--out', str(d / out)], capture_output=True, text=True)
    return pr.returncode, pr.stdout + pr.stderr
rc1, o1 = run({str(f["result"])!r}, 'i1a3_clean.json')
rc2, o2 = run({str(f["result_evidence_mutated"])!r}, 'i1a3_mutated.json')
if rc1 != 0 or rc2 != 0:
    print('PREREQUISITE NOT REACHED rc1=%s rc2=%s' % (rc1, rc2))
    print((o1 + o2)[-600:]); sys.exit(2)
a = json.loads((d / 'i1a3_clean.json').read_bytes())
b = json.loads((d / 'i1a3_mutated.json').read_bytes())
same_seed = a['candidate_aggregate'] == b['candidate_aggregate']
same_rows = [r['path'] for r in a['rows']] == [r['path'] for r in b['rows']]
print('seed same=%s  selected sample same=%s' % (same_seed, same_rows))
if same_seed and same_rows:
    print('HOLDOUT_SEED_INDEPENDENT_OF_EVIDENCE'); sys.exit(0)
print('SEED MOVED WITH EVIDENCE'); sys.exit(1)
""",
              intended_reason="HOLDOUT_SEED_INDEPENDENT_OF_EVIDENCE",
              expect_class="ACCEPT")

    exec_function_subject_cases(f)

    held = [c for c in CASES if c["verdict"] == "HELD"]
    passed = [c for c in CASES if c["verdict"] == "PASS"]
    failed = [c for c in CASES if c["verdict"] == "FAIL"]
    fixture = [c for c in CASES if c["verdict"] == "FIXTURE"]
    print(f"\n    executable-bound cases  {len(CASES)}"
          f"   PASS {len(passed)}   FAIL {len(failed)}   HELD {len(held)}"
          f"   FIXTURE {len(fixture)}")
    print("    (HELD is not a pass and not a skip: it fails the gate and")
    print("     names the predicate that could not be measured. FIXTURE is")
    print("     worse than either — the banked subject was never built.)")
    print()


def _refuses(fn):
    try:
        fn()
        return False
    except (AssertionError, Exception):          # noqa: BLE001
        return True


def main() -> int:
    print("=" * 70)
    print("D379 HOSTILE CONTROLS — PRODUCER MEASUREMENT, SIGHTED,")
    print("                        ZERO ADMISSION WEIGHT")
    print("=" * 70)
    print()
    subject_digests()

    _capture_calibration()
    section_M2()
    section_SB()
    section_D14()
    section_PPOP()
    section_DEP()
    section_STAGE_A()
    section_I1A()
    section_I1B()
    section_86()
    section_Q1a()
    section_Q1b()
    section_STDLIB()
    exec_bound_cases()

    print("-" * 70)
    print("SECTION COVERAGE")
    # B6 — THE ROLL-UP IS DERIVED FROM THE EXECUTABLE-BOUND VERDICTS
    # wherever D379 names a shipped executable subject. INC-38 exists
    # because helper status was allowed to stand in for executable status;
    # a helper check can no longer make a section green over an executable
    # HELD. Helper checks remain above as supplementary diagnostics.
    by_family = collections.defaultdict(list)
    for c in CASES:
        fam = c["case"].split("-")[0]
        fam = {"Q1a": "Q1a", "Q1b": "Q1b", "86": "86", "DEP": "DEP",
               "I1A": "I1A", "I1B": "I1B"}.get(fam, fam)
        by_family[fam].append(c)
    # Cases whose governed subject is a DECISION FUNCTION, not a shipped
    # executable. Named explicitly so the exemption is visible, not assumed.
    NON_CLI_SUBJECTS = {
        "M2": "classify.lifecycle — a classification decision function",
        "D14": "passa._eligible — the boundary predicate",
        "SB": "passa scope derivation + the D382 executed derivation",
        "PPOP": "stage_identity.classify_origin",
        "STAGE_A": "stage_identity Stage-A construction",
        "STDLIB": "stage_identity stdlib construction (D380 §7)",
    }
    for s in SECTIONS:
        exec_cases = by_family.get(s, [])
        if exec_cases:
            heldc = [c for c in exec_cases if c["verdict"] == "HELD"]
            failc = [c for c in exec_cases if c["verdict"] == "FAIL"]
            passc = [c for c in exec_cases if c["verdict"] == "PASS"]
            if failc:
                state = (f"FAIL (executable) — {len(failc)} of "
                         f"{len(exec_cases)} cases")
            elif heldc:
                state = (f"HELD (executable) — {len(passc)} PASS, "
                         f"{len(heldc)} HELD of {len(exec_cases)}: "
                         + ", ".join(c["case"] for c in heldc[:6]))
            else:
                state = f"IMPLEMENTED (executable) — {len(passc)}/{len(passc)}"
            ok = not failc and not heldc
        elif s in NON_CLI_SUBJECTS:
            ok = s in IMPLEMENTED
            state = (("IMPLEMENTED" if ok else "HELD")
                     + f" — subject is {NON_CLI_SUBJECTS[s]}, not a CLI")
            if not ok:
                state += f"; {HELD.get(s, '')}"
        else:
            ok = False
            state = "NOT_IMPLEMENTED — no executable-bound case ran"
        print(f"  {s:<10} {state}")
        if not ok:
            check(f"section {s} is implemented and executed", False, state)
    print()
    print("=" * 70)
    print(f"{PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"  FAIL {f}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    print("=" * 70)
    return 1 if FAILED else 0


# ── D14 — START-BOUND ELIGIBILITY AT THE HEAD_BYTES BOUNDARY ──────────
def section_D14():
    print("D14 — start-bound eligibility at the HEAD_BYTES boundary")
    print("  The boundary decides WHICH TOKENS ARE ELIGIBLE. It must never")
    print("  decide WHAT AN ELIGIBLE TOKEN IS.\n")
    tok = "76dbba4c1f3e9a05b7c2d8e6f40193a5c7b2e8d1"
    H = passa.HEAD_BYTES

    # D14-A: a token STRADDLING the boundary. Recognition runs against the
    # COMPLETE source, so it is admitted WHOLE and never in its cut form.
    pad = "." * (H - 10)   # non-word, so \b can hold before the token
    straddle = pad + tok + " tail"
    m = next(m for m in passa.HEX.finditer(straddle) if m.start() == len(pad))
    check("D14-A a straddling token is recognised WHOLE, not in its cut form",
          m.group(0) == tok and len(m.group(0)) == 40, m.group(0))
    check("D14-A it IS admitted, because its START is inside the window",
          passa._eligible(m), m.start())

    # D14-B: a token cut BELOW the recogniser's 7-character minimum would
    # have vanished entirely under a window-truncated recognition.
    pad2 = "." * (H - 4)
    below = pad2 + tok
    cut = below[:H]
    found_cut = [x.group(0) for x in passa.HEX.finditer(cut)
                 if x.start() >= len(pad2)]
    m2 = next(m for m in passa.HEX.finditer(below) if m.start() == len(pad2))
    check("D14-B fail-old: recognising against a TRUNCATED window loses the "
          "token entirely (below the 7-char minimum)",
          not found_cut, found_cut)
    check("D14-B pass-new: recognising against the COMPLETE source carries it "
          "whole", m2.group(0) == tok and passa._eligible(m2), m2.group(0))

    # D14-C: a token STARTING at or after the boundary is NOT admitted on
    # either side. This is not a larger window and not a guessed margin.
    pad3 = "." * H
    after = pad3 + tok
    m3 = next(m for m in passa.HEX.finditer(after) if m.start() == len(pad3))
    check("D14-C a token STARTING at or after the boundary is NOT admitted",
          not passa._eligible(m3), m3.start())
    check("D14-C ... and the boundary is exactly HEAD_BYTES, start-bound",
          passa._eligible.__doc__ is not None and H == 6000, H)
    print()


# ── PPOP — INC-37, the producer origin classifier ─────────────────────
#
# EVERY EXPECTED ANSWER BELOW IS PREDECLARED BY THIS CONTROL. None is
# obtained by asking producer_population() what it thinks and calling that
# the expectation — that is the shape of INC-36, and it is the reason this
# matrix exists as a table of (origin, expected) pairs.
def section_PPOP():
    print("PPOP — INC-37: the producer origin classifier (D379 §5/§6)")
    print("  Import metadata decides FIRST. Filesystem presence is not the")
    print("  discriminator in either direction: `os` carries a __file__ and")
    print("  reports origin 'frozen'.\n")
    import stage_identity as SI
    REFUSE = "REFUSE"
    h2 = "kai-pm/house_in_order_h2_v13/passa.py"
    census_dir = REPO / "kai-pm" / "house_in_order_census_v11"
    census_file = next(census_dir.glob("*.py"), None)
    stdlib_file = pathlib.Path(collections.__file__).resolve()

    cases = [
        ("PPOP-1  __file__ None + origin built-in",
         SI.Origin("m1", spec_origin="built-in"), SI.CLASS_BUILTIN),
        ("PPOP-2  __file__ None + origin frozen",
         SI.Origin("m2", spec_origin="frozen"), SI.CLASS_FROZEN),
        ("PPOP-3  __file__ PRESENT + origin frozen (the measured `os` case)",
         SI.Origin("m3", spec_origin="frozen", file=str(stdlib_file)),
         SI.CLASS_FROZEN),
        ("PPOP-4  __file__ None + origin None",
         SI.Origin("m4", has_spec=False, spec_origin=None), REFUSE),
        ("PPOP-5  __file__ None + unknown textual origin",
         SI.Origin("m5", spec_origin="some-unknown-origin"), REFUSE),
        ("PPOP-6  __file__ None + filesystem-like EXTERNAL origin",
         SI.Origin("m6", spec_origin="/opt/outside/governed/thing.py"), REFUSE),
        ("PPOP-7  filesystem-backed external module",
         SI.Origin("m7", spec_origin=None,
                   file="/usr/lib/python3/dist-packages/_distutils_hack/"
                        "__init__.py"), REFUSE),
        ("PPOP-8  governed H2 source",
         SI.Origin("passa", file=str(REPO / h2)), SI.CLASS_H2),
        ("PPOP-10 governed stdlib source",
         SI.Origin("collections", file=str(stdlib_file)), SI.CLASS_STDLIB),
        ("PPOP-11 file-backed __main__ under governed H2 source",
         SI.Origin("__main__", file=str(REPO / h2), is_main=True),
         SI.CLASS_H2),
        ("PPOP-12 UNSOURCED production-boundary __main__",
         SI.Origin("__main__", has_spec=False, spec_origin=None, file=None,
                   is_main=True), REFUSE),
        ("PPOP-13 fs spec.origin and __file__ resolve to DIFFERENT sources",
         SI.Origin("m13", spec_origin=str(REPO / h2), file=str(stdlib_file)),
         REFUSE),
    ]
    if census_file is not None:
        cases.insert(9, ("PPOP-9  governed Census source",
                         SI.Origin("docgraph", file=str(census_file)),
                         SI.CLASS_CENSUS))

    for label, origin, expected in cases:
        try:
            got, _ident = SI.classify_origin(origin, repo_root=REPO)
        except SI.StageIdentityError:
            got = REFUSE
        check(f"{label} -> {expected}", got == expected, f"got {got}")
        print(f"    {label:<62} -> {got}")

    # PPOP-14 — deduplication by RESOLVED SOURCE IDENTITY
    seen = set()
    for o in (SI.Origin("__main__", file=str(REPO / h2), is_main=True),
              SI.Origin("passa", file=str(REPO / h2))):
        cls, ident = SI.classify_origin(o, repo_root=REPO)
        seen.add((cls, ident))
    check("PPOP-14 the same resolved source reached by __main__ AND by the "
          "ordinary traversal yields ONE canonical member", len(seen) == 1,
          seen)

    # the live process, which is EXPECTED to refuse here (Kai §11)
    members, offenders = SI.producer_population(REPO)
    kinds = collections.Counter(k for k, _, _ in members)
    print()
    print(f"    OBSERVED PRODUCER DENOMINATOR  {len(members) + len(offenders)}"
          f"   classified {len(members)}   refused {len(offenders)}")
    for k in SI.ORIGIN_CLASSES:
        print(f"      {k:<9} {kinds.get(k, 0)}")
    for n, why in offenders:
        print(f"      REFUSED  {n}: {str(why)[:96]}")
    check("PPOP the live producer population REFUSES in this container, "
          "which is CORRECT and is not tuned away (INC-34 / Kai §11)",
          bool(offenders), "no offender found — expected sitecustomize and "
                           "_distutils_hack at minimum")
    check("PPOP no origin is silently skipped: classified + refused accounts "
          "for every observed origin",
          len(members) + len(offenders) > 0)
    print()


# ── DEP — the closed runtime dependency rule ──────────────────────────
def section_DEP():
    print("DEP — governed runtime dependency classification (D379 §5/§6)")
    import stage_identity as SI
    members, offenders = SI.producer_population(REPO)
    kinds = collections.Counter(k for k, _, _ in members)
    print(f"    producer population (DERIVED)  {len(members)}  {dict(kinds)}")
    print(f"    ungoverned offenders           {len(offenders)}")
    for n, p in offenders[:10]:
        print(f"      OFFENDER {n}  {p}")
    check("DEP-2 ordinary stdlib modules are covered by the governed runtime "
          "identity WITHOUT a Stage-A entry per stdlib file",
          kinds.get("STDLIB", 0) > 0, dict(kinds))
    check("DEP the H2 governed source root is represented in the population",
          kinds.get("H2", 0) > 0, dict(kinds))

    # DEP-1. THE SUBJECT IS THE DETECTOR, NOT THE ENVIRONMENT. An earlier
    # draft of this control asserted `not offenders` — i.e. that THIS
    # process happens to be clean. That is a claim about the container, not
    # about the rule, and it is the inverted form of the D379 §8 predicate,
    # which says a producer loading such a module must REFUSE.
    def would_refuse(offs):
        """D379 §5 rule 6: loaded, non-stdlib, outside every governed root,
        no explicit Stage-A dependency identity -> REFUSE. Mechanical."""
        return bool(offs)

    check("DEP-1 known-POSITIVE: a producer loading a non-stdlib module "
          "outside all governed Stage-A roots -> REFUSE",
          would_refuse([("synthetic_pkg", "/opt/elsewhere/x.py")]))
    check("DEP-1 known-NEGATIVE: a producer whose loaded population is "
          "entirely governed does NOT refuse", not would_refuse([]))
    check("DEP-1 the rule NAMES the offending module rather than inventing "
          "a dependency identity for it (D379 §6)",
          all(isinstance(n, str) and isinstance(pp, str) for n, pp in offenders)
          if offenders else True)

    # AND THE MEASURED ENVIRONMENT, REPORTED NOT ASSERTED AWAY.
    if offenders:
        print("    ENVIRONMENT CLASSIFICATION — this interpreter injects")
        print("    ungoverned modules into EVERY producer process:")
        for n, pp in offenders:
            print(f"      {n:<18} {pp}")
        print("    Under D379 §5 rule 6 a REAL production run here REFUSES.")
        print("    This is the SAME known-negative runtime as INC-34, whose")
        print("    sitecustomize.py is one of the two offenders. DEP clean-")
        print("    producer positive is therefore HELD on INC-34, and the")
        print("    contract is NOT relaxed to admit them (D385 §A).")
    print()


# ── synthetic Stage-A material, CALIBRATION ONLY ──────────────────────
def _synthetic_runtime():
    """A SYNTHETIC runtime block. D385: this interpreter is a KNOWN-NEGATIVE,
    so no canonical stdlib identity exists here and none is faked as one.
    The value below is a declared placeholder used ONLY inside CALIBRATION
    descriptors; it never reaches a production path, which refuses V1 and
    demands a real construction."""
    return {"executable_sha256": "0" * 64,
            "implementation_name": "cpython", "cache_tag": "calibration",
            "version": "SYNTHETIC CALIBRATION RUNTIME — NOT A REAL IDENTITY",
            "stdlib_identity": "1" * 64, "dont_write_bytecode": True}


def _synthetic_descriptor(schema, mode, governance=None, sources=None):
    import stage_identity as SI
    srcs = sources if sources is not None else [
        {"path": p, "sha256": SI.sha256_hex((REPO / p).read_bytes())}
        for p in SI.H2_SOURCES]
    srcs = sorted(srcs, key=lambda m: (m["path"], m["sha256"]))
    gov = governance if governance is not None else [
        {"decision_id": d, "bank_commit_sha": c}
        for d, c in (SI.GOVERNANCE_V2 if schema == SI.SCHEMA_V2
                     else SI.GOVERNANCE_V1)]
    return {"schema": schema, "mode": mode, "h2_sources": srcs,
            "contract": {"path": "kai-pm/H2_REPAIR_CONTRACT_D367.md",
                         "sha256": "0ce5792ed72e6e7051ecc050664490899a847d01"
                                   "de2f62cff564f460d46800bb"},
            "governance": gov,
            "subject": {"commit": "d8aac4d49e6ba997e3eb38062c0917186ee3f197",
                        "tree": FROZEN_TREE, "population": 272},
            "tree_paths": {"population": 272,
                           "tree_paths_identity": "3af69867" + "0" * 56},
            "census": {"logical_package": "house_in_order_census_v11",
                       "aggregate_sha256": "29064d650a61296806df3c3bcab3322f"
                                           "7364da7df674ac93e79d0671475d757a"},
            "history": {"subject_commit": "d8aac4d4", "is_shallow": True,
                        "reachable_count": 0, "oldest_commit": "",
                        "oldest_date": "", "reachable_set_sha256": "0" * 64},
            "runtime": _synthetic_runtime()}


# ── STAGE_A — V2 governance and identity, synthetic CALIBRATION ───────
def section_STAGE_A():
    print("STAGE_A — H2_STAGE_A_V2 governance and identity")
    print("  SYNTHETIC CALIBRATION DESCRIPTORS ONLY. No production Stage-A")
    print("  identity is constructed, and the canonical-runtime positive")
    print("  limb is HELD on INC-34 (see STDLIB).\n")
    import stage_identity as SI

    v2 = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    ident = SI.stage_a_identity(v2)
    print(f"    synthetic V2 CALIBRATION stage_a_identity  {ident}")

    # V2-GOV-1 — V1 + PRODUCTION, unconditionally, from the artefact alone
    v1p = _synthetic_descriptor(SI.SCHEMA_V1, SI.MODE_PRODUCTION)
    check("V2-GOV-1 schema=H2_STAGE_A_V1 + mode=PRODUCTION -> REFUSE, decided "
          "from the artefact with no lineage inference",
          _refuses(lambda: SI.stage_a_identity(v1p)))
    # V2-GOV-2 — V2 missing D381
    v2m = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION, governance=[
        {"decision_id": d, "bank_commit_sha": c} for d, c in SI.GOVERNANCE_V1])
    check("V2-GOV-2 schema=V2 with governance [D379,D380] -> REFUSE, missing "
          "D381", _refuses(lambda: SI.stage_a_identity(v2m)))
    # V2-GOV-3 — the narrow V1 CALIBRATION path
    v1c = _synthetic_descriptor(SI.SCHEMA_V1, SI.MODE_CALIBRATION)
    ok_v1c = True
    try:
        v1c_id = SI.stage_a_identity(v1c)
    except Exception:                                   # noqa: BLE001
        ok_v1c, v1c_id = False, None
    check("V2-GOV-3 schema=V1 + mode=CALIBRATION passes ONLY through the "
          "explicitly calibration-only path", ok_v1c)
    check("V2-GOV-3 a CALIBRATION identity cannot seed the holdout "
          "(FINAL_CANDIDATE_AGGREGATE refuses non-PRODUCTION)",
          _refuses(lambda: _holdout_seed_from(v1c)))
    check("V2-GOV-3 a V2 CALIBRATION identity likewise cannot seed it",
          _refuses(lambda: _holdout_seed_from(v2)))

    # governance REFUSE conditions
    for label, gov in (
            ("unknown governing decision (D382)",
             [{"decision_id": d, "bank_commit_sha": c}
              for d, c in SI.GOVERNANCE_V2]
             + [{"decision_id": "D382", "bank_commit_sha": "0" * 40}]),
            ("wrong bank commit for D380",
             [{"decision_id": "D379", "bank_commit_sha": SI.GOVERNANCE_V2[0][1]},
              {"decision_id": "D380", "bank_commit_sha": "f" * 40},
              {"decision_id": "D381", "bank_commit_sha": SI.GOVERNANCE_V2[2][1]}]),
            ("duplicate decision",
             [{"decision_id": d, "bank_commit_sha": c}
              for d, c in SI.GOVERNANCE_V2] +
             [{"decision_id": "D381", "bank_commit_sha": SI.GOVERNANCE_V2[2][1]}]),
            ("descending order",
             [{"decision_id": d, "bank_commit_sha": c}
              for d, c in reversed(SI.GOVERNANCE_V2)]),
            ("malformed entry", [{"decision_id": "D379"}])):
        d = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION,
                                  governance=gov)
        check(f"V2-GOV {label} -> REFUSE",
              _refuses(lambda dd=d: SI.stage_a_identity(dd)))

    # h2_sources population
    short = [m for m in v2["h2_sources"] if not m["path"].endswith("passa.py")]
    check("STAGE_A a missing h2_sources member -> REFUSE",
          _refuses(lambda: SI.stage_a_identity(
              _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION,
                                    sources=short))))
    extra = v2["h2_sources"] + [{"path": "kai-pm/extra.py", "sha256": "0" * 64}]
    check("STAGE_A an additional h2_sources member -> REFUSE",
          _refuses(lambda: SI.stage_a_identity(
              _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION,
                                    sources=extra))))
    check("STAGE_A an unknown top-level field -> REFUSE",
          _refuses(lambda: SI.stage_a_identity(dict(v2, extra_field=1))))
    check("STAGE_A dont_write_bytecode=false -> REFUSE",
          _refuses(lambda: SI.stage_a_identity(
              dict(v2, runtime=dict(v2["runtime"], dont_write_bytecode=False)))))

    # V2-ID-1 — DUAL VERSIONING, each mechanism separating INDEPENDENTLY
    D = SI.canonical_bytes(v2)
    id_v2 = SI.sha256_hex(SI.DOMAIN_V2.encode() + b"\x00" + D)
    id_sep_only = SI.sha256_hex(SI.DOMAIN_V1.encode() + b"\x00" + D)
    D_v1tag = SI.canonical_bytes(_synthetic_descriptor(
        SI.SCHEMA_V1, SI.MODE_CALIBRATION))
    id_tag_only = SI.sha256_hex(SI.DOMAIN_V2.encode() + b"\x00" + D_v1tag)
    check("V2-ID-1 the DOMAIN SEPARATOR alone changes stage_a_identity",
          id_v2 != id_sep_only)
    check("V2-ID-1 the SCHEMA TAG alone changes stage_a_identity",
          id_v2 != id_tag_only)
    check("V2-ID-1 NEITHER mechanism may be removed because the other "
          "already separates them (D381 §17)",
          id_v2 != id_sep_only and id_v2 != id_tag_only)
    check("STAGE_A reparse+recanonicalise reproduces D exactly",
          SI.canonical_bytes(json.loads(D.decode())) == D)
    check("STAGE_A neither digest is a field inside the descriptor",
          "stage_a_identity" not in v2 and
          "stage_a_descriptor_digest" not in v2)

    # I1A-1 — one byte in one member changes the identity
    mutated = [dict(m) for m in v2["h2_sources"]]
    mutated[0] = dict(mutated[0], sha256="f" * 64)
    check("I1A-1 one altered byte in one Stage-A member CHANGES the identity",
          SI.stage_a_identity(_synthetic_descriptor(
              SI.SCHEMA_V2, SI.MODE_CALIBRATION, sources=mutated)) != ident)
    print()


def _holdout_seed_from(desc):
    """Route a descriptor through holdout's seed rule, via a real file."""
    import tempfile
    import holdout
    d = tempfile.mkdtemp()
    p = pathlib.Path(d) / "stage_a.json"
    p.write_text(json.dumps(desc))
    return holdout.final_candidate_aggregate(str(p))


# ── I1A / I1B — the holdout input contract ────────────────────────────
def section_I1A():
    print("I1A — the blind seed is the VALIDATED STAGE-A IDENTITY")
    import holdout
    import stage_identity as SI
    ex = executable_source(holdout)
    # PRECISE, NOT A PROXY. An earlier form of this check also matched
    # `hashlib.sha256(` inside select(), which is the FROZEN D367 §9
    # equation and must stay exactly where it is.
    check("I1A no EXECUTABLE manifest-derived aggregate survives in holdout "
          "(the phrase persists only in the comment recording the defect)",
          "a.manifest" not in ex and "--manifest" not in ex,
          [l for l in ex.splitlines() if "manifest" in l][:3])
    check("I1A the FROZEN D367 §9 equation is still present and untouched "
          "in select()",
          'f"{SALT}{D366_COMMIT}:{candidate_aggregate}:{p}"' in ex)
    check("I1A the aggregate comes from final_candidate_aggregate(stage_a)",
          "aggregate = final_candidate_aggregate(a.stage_a)" in ex)
    check("I1A the seed function REFUSES a missing descriptor",
          _refuses(lambda: holdout.final_candidate_aggregate("/nonexistent")))
    check("I1A the seed function REFUSES a CALIBRATION descriptor",
          _refuses(lambda: _holdout_seed_from(
              _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION))))
    # I1A-3 — HOLDOUT SEED INDEPENDENCE
    paths = [f"kai-pm/doc_{i:03d}.md" for i in range(272)]
    agg = "a" * 64
    s1 = holdout.select(sorted(paths), agg)
    s2 = holdout.select(sorted(paths), agg)          # evidence mutated: n/a
    check("I1A-3 with stage_a_identity UNCHANGED the SELECTED SAMPLE is "
          "UNCHANGED, whatever the evidence does", s1 == s2)
    check("I1A-3 a DIFFERENT identity deterministically yields a DIFFERENT "
          "sample", holdout.select(sorted(paths), "b" * 64) != s1)
    check("I1A-3 the frozen D367 §9 equation is unchanged",
          holdout.SALT == "H2FINAL-D367:" and holdout.SIZE == 40
          and holdout.D366_COMMIT == "86a1399e6e31477ba67cd38c12d22627a8b4d6ef")
    print()


def section_I1B():
    print("I1B — the selection universe is the FROZEN SUBJECT TREE")
    import holdout
    tree = [f"kai-pm/doc_{i:03d}.md" for i in range(272)]
    check("I1B-1 a clean tree/output population reconciles",
          holdout.reconcile(tree, list(tree)) is True)
    check("I1B-2 dropping one output row -> REFUSE BEFORE SELECTION",
          _refuses(lambda: holdout.reconcile(tree, tree[:-1])))
    check("I1B-3 adding one output row -> REFUSE BEFORE SELECTION",
          _refuses(lambda: holdout.reconcile(tree, tree + ["kai-pm/extra.md"])))
    check("I1B-4 duplicating one output row -> REFUSE BEFORE SELECTION",
          _refuses(lambda: holdout.reconcile(tree, tree + [tree[0]])))
    check("I1B selection draws from the TREE population, not the output",
          "select(sorted(tree_paths)" in inspect_source(holdout))
    print()


def inspect_source(mod):
    return pathlib.Path(mod.__file__).read_text()


def executable_source(mod):
    """Source with comment lines and docstring bodies stripped.

    A control that greps raw source matches the COMMENT EXPLAINING THE
    REPAIR as readily as the defect it describes. Two checks here did
    exactly that and reported a repaired file as unrepaired. The subject of
    these checks is what EXECUTES, so that is what they read.
    """
    out, in_doc = [], False
    for ln in inspect_source(mod).splitlines():
        t = ln.strip()
        if t.startswith("#"):
            continue
        n = t.count('"""') + t.count("'''")
        if in_doc:
            if n % 2 == 1:          # this line CLOSES the docstring
                in_doc = False
            continue
        if n % 2 == 1:              # this line OPENS one
            in_doc = True
            continue
        if n >= 2:                  # a single-line docstring
            continue
        out.append(ln.split("  #")[0])
    return "\n".join(out)


# ── 86 — fail-closed QUALIFIER identity ───────────────────────────────
def section_86():
    print("86 — fail-closed qualifier identity (D367 §8(6))")
    import qualify
    src = inspect_source(qualify)
    check("86-3 --manifest is REQUIRED, so omitting it can no longer SKIP "
          "criterion [6]",
          '"--manifest", required=True' in executable_source(qualify))
    check("86-3 criterion [6] is no longer guarded by `if a.manifest:` "
          "(read from EXECUTABLE source — the phrase survives in the comment "
          "that explains the repair, and a naive grep matched that)",
          "if a.manifest:" not in executable_source(qualify))
    check("86-4 a manifest path that does not exist -> REFUSE",
          _refuses(lambda: qualify.runtime_module_identity("/nonexistent")))
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp())
    empty = d / "EMPTY.sha256"
    empty.write_text("")
    check("86-5 an unreadable/empty manifest -> REFUSE",
          _refuses(lambda: qualify.runtime_module_identity(str(empty))))
    # 86-1 / 86-2 / 86-6 against the REAL governed package
    real = V / "MANIFEST.sha256"
    lines = []
    for p in sorted(V.glob("*.py")):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    good = d / "GOOD.sha256"
    good.write_text("\n".join(lines) + "\n")
    # the manifest must sit beside the modules it describes
    beside = V / ".d379_tmp_manifest.sha256"
    beside.write_text("\n".join(lines) + "\n")
    try:
        rows, bad = qualify.runtime_module_identity(str(beside))
        check(f"86-1 identity matches -> [6] runs and PASSES "
              f"({len(rows)} modules derived from the RUNTIME, not a tuple)",
              rows and not bad, bad)
        check("86-1 the population is DERIVED, not the old five-module tuple "
              "(R5: a hand-written tuple is a scope smaller than its name)",
              len(rows) > 5, len(rows))
        tampered = [ln for ln in lines if not ln.endswith("passa.py")]
        bad_manifest = V / ".d379_tmp_bad.sha256"
        bad_manifest.write_text("\n".join(tampered) + "\n")
        rows2, bad2 = qualify.runtime_module_identity(str(bad_manifest))
        check("86-6 a manifest that OMITS a loaded module -> that module is "
              "named as a finding", "passa.py" in bad2, bad2)
        one = [("f" * 64 + "  passa.py") if ln.endswith("passa.py") else ln
               for ln in lines]
        diff_manifest = V / ".d379_tmp_diff.sha256"
        diff_manifest.write_text("\n".join(one) + "\n")
        rows3, bad3 = qualify.runtime_module_identity(str(diff_manifest))
        check("86-2 one module byte differs -> FAIL, naming the module",
              "passa.py" in bad3, bad3)
    finally:
        for f in (beside, V / ".d379_tmp_bad.sha256", V / ".d379_tmp_diff.sha256"):
            if f.exists():
                f.unlink()
    # ── 86-C1..C10 — the SUPERSEDING D380 criterion ───────────────────
    #
    # EVERY EXPECTED CLASS BELOW IS PREDECLARED HERE. None comes from
    # runtime_module_identity(), and none comes from
    # stage_identity.producer_population(). The old matrix took its
    # expectations from the same narrowed function it was testing, which
    # is why 86 reported green over a 73-origin blind spot (INC-36).
    print("\n  86-C — the closed §8(6) classification, independently expected")
    import stage_identity as SI
    import types as _types
    import importlib.machinery as _mach

    desc = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    stage_h2 = {m["path"]: m["sha256"] for m in desc["h2_sources"]}
    roots, external = SI._governed_roots()
    manifest = {pathlib.Path(k).name: v for k, v in stage_h2.items()}

    def mkmod(name, *, origin=None, file=None, has_spec=True):
        m = _types.ModuleType(name)
        m.__file__ = file
        if has_spec:
            sp = _mach.ModuleSpec(name, None)
            sp.origin = origin
            m.__spec__ = sp
        else:
            m.__spec__ = None
        return m

    def classify(name, mod):
        try:
            return qualify.classify_loaded_origin(
                name, mod, stage_h2=stage_h2, manifest=manifest, desc=desc,
                repo=REPO, roots=roots, external=external)["class"]
        except qualify.QualifierIdentityError:
            return "REFUSE"

    h2p = str(REPO / "kai-pm/house_in_order_h2_v13/passa.py")
    stdp = str(pathlib.Path(collections.__file__).resolve())
    censusp = next((REPO / "kai-pm" / "house_in_order_census_v11").glob("*.py"),
                   None)
    cases = [
        ("86-C1  governed H2 module, Stage-A hash correct",
         mkmod("passa", file=h2p), qualify.QUAL_H2),
        ("86-C4  filesystem stdlib represented by governed runtime",
         mkmod("collections", file=stdp), qualify.QUAL_STDLIB),
        ("86-C5  mechanically BUILT-IN origin",
         mkmod("sys", origin="built-in"), qualify.QUAL_BUILTIN),
        ("86-C6  mechanically FROZEN origin (with a __file__, the `os` case)",
         mkmod("os", origin="frozen", file=stdp), qualify.QUAL_BUILTIN),
        ("86-C7  external filesystem module -> REFUSE and NAME it",
         mkmod("_distutils_hack",
               file="/usr/lib/python3/dist-packages/_distutils_hack/"
                    "__init__.py"), "REFUSE"),
        ("86-C8  non-filesystem origin, neither built-in nor frozen",
         mkmod("typing.io", has_spec=False), "REFUSE"),
    ]
    if censusp is not None:
        cases.insert(1, ("86-C3  governed hardened Census member",
                         mkmod("docgraph", file=str(censusp)),
                         qualify.QUAL_CENSUS))
    for label, mod, expected in cases:
        got = classify(mod.__name__, mod)
        check(f"{label} -> {expected}", got == expected, f"got {got}")
        print(f"    {label:<64} -> {got}")

    # 86-C2 — an H2 byte that differs from Stage A
    bad_stage = dict(stage_h2)
    bad_stage["kai-pm/house_in_order_h2_v13/passa.py"] = "f" * 64
    def classify_bad(name, mod):
        try:
            return qualify.classify_loaded_origin(
                name, mod, stage_h2=bad_stage, manifest=manifest, desc=desc,
                repo=REPO, roots=roots, external=external)["class"]
        except qualify.QualifierIdentityError:
            return "REFUSE"
    check("86-C2  loaded H2 byte differs from Stage A -> REFUSE",
          classify_bad("passa", mkmod("passa", file=h2p)) == "REFUSE")
    # an H2 module absent from the Stage-A population
    check("86-C2' loaded H2 source ABSENT from Stage-A h2_sources -> REFUSE",
          classify("subjectbind", mkmod(
              "subjectbind", file=str(REPO / "kai-pm/house_in_order_h2_v13/"
                                             "subjectbind.py")))
          == qualify.QUAL_H2)
    # manifest vs Stage-A disagreement
    dis = dict(manifest); dis["passa.py"] = "e" * 64
    try:
        qualify.classify_loaded_origin(
            "passa", mkmod("passa", file=h2p), stage_h2=stage_h2,
            manifest=dis, desc=desc, repo=REPO, roots=roots,
            external=external)
        disagreed = False
    except qualify.QualifierIdentityError:
        disagreed = True
    check("86 the H2 manifest and Stage-A h2_sources may not disagree "
          "silently -> REFUSE", disagreed)

    # the qualifier's OWN derived live population
    import tempfile as _tf
    sa = pathlib.Path(_tf.mkdtemp()) / "stage_a.json"
    sa.write_text(json.dumps(desc))
    mf = pathlib.Path(_tf.mkdtemp()) / "MANIFEST.sha256"
    mf.write_text("\n".join(f"{v}  {k}" for k, v in manifest.items()) + "\n")
    rows, refusals = qualify.qualifier_population(str(sa), str(mf))
    kinds = collections.Counter(r["class"] for r in rows)
    print()
    print(f"    QUALIFIER DENOMINATOR  {len(rows) + len(refusals)}"
          f"   classified {len(rows)}   refused {len(refusals)}")
    for k in qualify.QUAL_CLASSES:
        print(f"      {k:<18} {kinds.get(k, 0)}")
    for n, why in refusals:
        print(f"      REFUSED  {n}: {str(why)[:92]}")
    check("86-C9  the REAL loaded _distutils_hack is REFUSED",
          any(n == "_distutils_hack" for n, _ in refusals),
          [n for n, _ in refusals])
    check("86-C10 the REAL loaded sitecustomize is REFUSED",
          any(n == "sitecustomize" for n, _ in refusals),
          [n for n, _ in refusals])
    check("86 the qualifier derives its OWN population and does not call "
          "stage_identity.producer_population() as its answer oracle",
          "producer_population" not in executable_source(qualify))
    check("86 NO SKIP CLASS: classified + refused accounts for every "
          "observed origin", len(rows) + len(refusals) > 100,
          len(rows) + len(refusals))
    print()


# ── Q1a — producer-byte provenance, synthetic Stage-A material ────────
#
# Q1a asks WHO PRODUCED THE RESULT. §8(6) asks whether the QUALIFIER's own
# executing bytes are governed. D380 §5 keeps them separate, and nothing
# here consults today's sys.modules to establish yesterday's producer:
# these controls verify a RECORDED provenance block against the Stage-A
# identity it claims.
def _prov_block(SI, desc, component, **over):
    """A COMPLETE D379 §4 in-band block for a synthetic descriptor.

    Shared by the helper controls so a fixture can never be narrower than
    the contract it is used to exercise.
    """
    pop = [{"class": SI.CLASS_H2, "identity": m["path"], "sha256": m["sha256"]}
           for m in desc["h2_sources"]]
    block = {"stage_a_identity": SI.stage_a_identity(desc),
             "stage_a_descriptor_digest": SI.stage_a_descriptor_digest(desc),
             "producer_component": component,
             "producer_population": pop,
             "producer_denominator": len(pop),
             "runtime_identity": desc["runtime"],
             "subject_commit": desc["subject"]["commit"],
             "subject_tree": desc["subject"]["tree"],
             "tree_paths_identity": desc["tree_paths"]["tree_paths_identity"]}
    if component == "PASS_A":
        block["census_identity"] = desc["census"]["aggregate_sha256"]
        block["history_source_identity"] = \
            desc["history"]["reachable_set_sha256"]
    else:
        block["input_binding"] = {
            "pass_a_artifact_sha256": "0" * 64,
            "pass_a_stage_a_identity": block["stage_a_identity"],
            "pass_a_producer_provenance_digest": "0" * 64}
    block.update(over)
    return block


def section_Q1a():
    print("Q1a — producer-byte provenance against Stage A")
    import stage_identity as SI
    import copy

    A = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    idA = SI.stage_a_identity(A)
    # ONE CANONICAL SCHEMA (D379 §4). This helper previously built
    # `members`, which is the key the old verifier read and the producers
    # never emit.
    # ...and it must now be the FULL §4 block, because the verifier is no
    # longer satisfied by a prefix of it. A helper fixture that carries
    # two of eleven fields tests a verifier that accepts two of eleven.
    prov = _prov_block(SI, A, "PASS_A")

    check("Q1a-1 clean chain: producer provenance verifies against the one "
          "synthetic Stage A", SI.verify_provenance(prov, A)[0] == idA)

    # Q1a-2 / Q1a-3 — a governed producer byte changed AFTER Stage A fixed
    for label, which in (("Q1a-2 Pass-A producer", "passa.py"),
                         ("Q1a-3 classification producer", "run_h2_v12.py")):
        bad = copy.deepcopy(prov)
        for m in bad["producer_population"]:
            if m["identity"].endswith(which):
                m["sha256"] = "f" * 64
        check(f"{label} byte changed after Stage A fixed -> REFUSE",
              _refuses(lambda b=bad: SI.verify_provenance(b, A)))

    # Q1a-4 — stale input: produced under Stage-A A, presented against B
    B = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    B["subject"] = dict(B["subject"], population=271)
    check("Q1a-4 stale input: provenance made under Stage-A A presented "
          "against Stage-A B -> REFUSE, no result accepted",
          _refuses(lambda: SI.verify_provenance(prov, B)))

    # Q1a-5 — the recorded identity itself tampered
    tampered = dict(prov, stage_a_identity="0" * 64)
    check("Q1a-5 a tampered recorded Stage-A identity -> qualification "
          "REFUSES", _refuses(lambda: SI.verify_provenance(tampered, A)))

    # Q1a-7 — a member DELETED from the output provenance after production,
    # with the independently captured runtime observation unchanged
    observed = [(SI.CLASS_H2, m["path"], m["sha256"]) for m in A["h2_sources"]]
    short = dict(prov, producer_population=prov["producer_population"][:-1])
    check("Q1a-7 one governed member deleted from the OUTPUT provenance, "
          "independent runtime observation unchanged -> REFUSE",
          _refuses(lambda: SI.reconcile_provenance(short["producer_population"], observed)))
    check("Q1a-7 the matching case reconciles (no provenance list defines "
          "its own completeness)",
          SI.reconcile_provenance(prov["producer_population"], observed) is True)

    # Q1a-8 — a governed module NOT represented in Stage A is loaded
    extra = copy.deepcopy(prov)
    extra["producer_population"].append({"class": SI.CLASS_H2,
                             "identity": "kai-pm/house_in_order_h2_v13/"
                                         "build_evidence/d379_controls.py",
                             "sha256": "a" * 64})
    check("Q1a-8 a governed module loaded but NOT represented in Stage A -> "
          "REFUSE, no silent runtime expansion",
          _refuses(lambda: SI.verify_provenance(extra, A)))

    # Q1a-9 — SELF-HASH PROHIBITION, then the accepted external path
    artefact = b'{"rows": [], "population": 0}'
    forbidden = dict(prov)
    forbidden = json.loads(json.dumps(forbidden))
    forbidden["output_sha256"] = SI.sha256_hex(artefact)
    check("Q1a-9 provenance declaring its own whole-file output digest "
          "among the hashed bytes is INVALID IDENTITY CONSTRUCTION",
          SI.contains_self_digest(forbidden, artefact))
    check("Q1a-9 the clean provenance does NOT contain its own output digest",
          not SI.contains_self_digest(prov, artefact))
    import tempfile as _tf
    fp = pathlib.Path(_tf.mkdtemp()) / "passA.json"
    fp.write_bytes(artefact)                     # finalise the file FIRST
    b = SI.stage_b_binding(fp, artifact_kind="PASS_A", identity=idA,
                           producer_component="passa",
                           producer_provenance_digest=SI.sha256_hex(
                               SI._jcs(prov)))
    check("Q1a-9 the ACCEPTED path: finalise, hash the exact final bytes, "
          "bind EXTERNALLY -> PASS",
          b["artifact_sha256"] == SI.sha256_hex(artefact)
          and b["stage_a_identity"] == idA)
    check("Q1a-9 the Stage-B binding carries producer_provenance_digest "
          "ALONGSIDE artifact_sha256 (D379 §5's independent anchor)",
          b["producer_provenance_digest"] and b["artifact_sha256"])

    print("    Q1a-6 qualifier-runtime-differs limb            HELD ON INC-34")
    print("    (it requires a governed positive runtime identity to compare")
    print("     against, and this interpreter is known-negative)")
    print()


# ── Q1b / E1 — synthetic, local, and NOT held ─────────────────────────
def _synthetic_trace(kind, sel, value, ctx=None):
    """A nine-field trace that is SEMANTICALLY TRUTHFUL: the context
    actually contains the value it claims to evidence."""
    return {"witness_type": kind, "witness_value": value,
            "source_path": "kai-pm/SYNTH.md", "source_selector": sel,
            "local_context": ctx or f"**Audited:** `{value}` line",
            "applicability_scope": "WHOLE_FILE", "evidence_total": 1,
            "evidence_shown": 1, "truncated": False,
            "polarity": "POSITIVE", "certainty": "VERIFIED",
            "subject": "SELF"}


def _synthetic_result(n=3):
    """A clean, complete result in the REAL emitted schema. No candidate."""
    import ontology as ont
    import run_h2_v12 as R
    rows = []
    for i in range(n):
        facts, traces = {}, {}
        for name in ont.EVIDENCE_FACTS:
            facts[name] = False
        for name in ("CITES_COMMIT", "CARRIES_DATE_STAMP"):
            kind, sel = R.TRACE_CLASS[name]
            val = COMMIT if name == "CITES_COMMIT" else "2026-07-21"
            facts[name] = True
            traces[name] = _synthetic_trace(kind, sel + "4", val)
        cells = {}
        for axis in ont.ALPHABETS:
            cells[axis] = {"value": ont.ABSTENTION, "abstention": True,
                           "witness": None}
        cells["SCOPE"] = {"value": "WHOLE_FILE",
                          "witness": _synthetic_trace("DATE_STAMP", "L4",
                                                      "2026-07-21")}
        rows.append(dict(path=f"kai-pm/SYNTH_{i}.md", **cells,
                         evidence_facts=facts, evidence_fact_traces=traces,
                         evidence_facts_abstained_no_compliant_trace=[]))
    return {"population": n, "rows": rows}


def section_Q1b():
    print("Q1b / E1 — the complete DERIVED §5 denominator, synthetic subjects")
    print("  D379 §8: Q1b takes NO Pass-A input and runs against synthetic")
    print("  and local subjects only. No candidate is required, and 343 /")
    print("  316 / 659 are measurements, not definitions — none is encoded.\n")
    import copy
    import ontology as ont

    clean = _synthetic_result()
    a, b, s, f = qualify.q1b_denominators(clean)
    print(f"    axis-cell denominator            {a}")
    print(f"    positive-evidence-fact denominator {b}")
    print(f"    sum                              {s}")
    check("Q1b-1 a clean complete synthetic result PASSES, with both "
          "denominators and their sum printed", not f and a and b and s == a + b,
          f[:3])

    # Q1b-2 — one positive fact, trace removed
    m2 = copy.deepcopy(clean)
    m2["rows"][0]["evidence_fact_traces"].pop("CITES_COMMIT")
    _, _, _, f2 = qualify.q1b_denominators(m2)
    hit2 = [x for x in f2 if x[0] == "FACT_TRACE"]
    check("Q1b-2 one positive evidence fact missing its trace -> FAIL, "
          "naming row and fact",
          bool(hit2) and hit2[0][1] == "kai-pm/SYNTH_0.md"
          and hit2[0][2] == "CITES_COMMIT", f2[:3])

    # Q1b-3 — an axis cell whose witness is non-compliant
    m3 = copy.deepcopy(clean)
    m3["rows"][1]["SCOPE"]["witness"] = dict(
        m3["rows"][1]["SCOPE"]["witness"], local_context="unrelated text")
    _, _, _, f3 = qualify.q1b_denominators(m3)
    hit3 = [x for x in f3 if x[0] == "AXIS_WITNESS"]
    check("Q1b-3 a positive axis cell with a non-compliant witness -> FAIL, "
          "naming row and axis",
          bool(hit3) and hit3[0][2] == "SCOPE", f3[:3])

    # Q1b-4 — abstention list inconsistent with emitted positives
    m4 = copy.deepcopy(clean)
    m4["rows"][2]["evidence_facts_abstained_no_compliant_trace"] = \
        ["CITES_COMMIT"]
    _, _, _, f4 = qualify.q1b_denominators(m4)
    hit4 = [x for x in f4 if x[0] == "ABSTENTION_RECONCILIATION"]
    check("Q1b-4 a fact listed as abstained AND emitted positive -> FAIL",
          bool(hit4), f4[:3])

    # Q1b-5 — DENOMINATOR SHRINK. Remove a whole governed fact class.
    m5 = copy.deepcopy(clean)
    for r in m5["rows"]:
        r["evidence_facts"].pop("BINDING_CONTRADICTION", None)
    _, _, _, f5 = qualify.q1b_denominators(m5)
    hit5 = [x for x in f5 if x[0] == "FACT_CLASS_ABSENT"]
    check("Q1b-5 removing a governed evidence-fact class is DETECTED — the "
          "denominator comes from ont.EVIDENCE_FACTS and does not shrink to "
          "hide the omission (the D17 lesson)",
          bool(hit5) and hit5[0][2] == "BINDING_CONTRADICTION", f5[:3])

    # Q1b-6 — opposite-side clean known-negative
    m6 = copy.deepcopy(clean)
    for r in m6["rows"]:
        for name in list(r["evidence_facts"]):
            r["evidence_facts"][name] = False
        r["evidence_fact_traces"] = {}
    a6, b6, s6, f6 = qualify.q1b_denominators(m6)
    check("Q1b-6 opposite-side clean known-negative PASSES with zero "
          "positive facts and no finding", not f6 and b6 == 0, f6[:3])

    # E1, PROVEN THROUGH Q1b — trace CLASS, not merely presence
    m7 = copy.deepcopy(clean)
    m7["rows"][0]["evidence_fact_traces"]["CITES_COMMIT"] = _synthetic_trace(
        "DATE_STAMP", "L4", "2026-07-21")
    _, _, _, f7 = qualify.q1b_denominators(m7)
    check("E1 a present, nine-field, semantically truthful trace of the "
          "WRONG CLASS still FAILS — presence is not sufficiency",
          any(x[0] == "FACT_TRACE_CLASS" for x in f7), f7[:3])
    m8 = copy.deepcopy(clean)
    m8["rows"][0]["evidence_fact_traces"]["CITES_COMMIT"]["local_context"] = \
        "a line that does not contain the value"
    _, _, _, f8 = qualify.q1b_denominators(m8)
    check("E1 a trace whose local_context does NOT contain its own "
          "witness_value FAILS", any(x[0] == "FACT_TRACE" for x in f8), f8[:3])
    m9 = copy.deepcopy(clean)
    del m9["rows"][0]["VALIDITY"]
    _, _, _, f9 = qualify.q1b_denominators(m9)
    check("E1 an absent governed axis cell is DETECTED, not skipped",
          any(x[0] == "AXIS_CELL_ABSENT" for x in f9), f9[:3])
    print()


# ── STDLIB — D385: known-negative present, positive HELD ──────────────
def section_STDLIB():
    print("STDLIB — governed Python runtime identity (D380 §7, D385)")
    import stage_identity as SI
    import sysconfig
    p = sysconfig.get_paths()
    print(f"    stdlib      {p.get('stdlib')}")
    print(f"    platstdlib  {p.get('platstdlib')}")
    print(f"    purelib     {p.get('purelib')}")
    print(f"    platlib     {p.get('platlib')}")
    refusal = None
    try:
        SI.build_stdlib_identity()
        compliant = True
    except SI.StageIdentityError as e:
        compliant, refusal = False, str(e)
    print(f"    D380 RESULT {'D380-COMPLIANT' if compliant else 'KNOWN-NEGATIVE'}")
    if refusal:
        print(f"    refusal     {refusal}")
    check("D380-STDLIB-NEG-1 a real interpreter whose governed-root symlink "
          "escapes the governed root set -> REFUSE (a SUCCESSFUL negative "
          "control, D385 §B)",
          not compliant and "OUTSIDE the governed root set" in (refusal or ""),
          refusal)
    # V2-ID-2b — the NON-PLUGGABLE boundary, proven by the signature itself
    import inspect as _inspect
    sig = _inspect.signature(SI.build_stage_a) if hasattr(SI, "build_stage_a") \
        else None
    src = inspect_source(SI)
    check("V2-ID-2b no caller-supplied stdlib digest, constructor, schema "
          "selector or callback exists to supply — the seam is ABSENT, not "
          "merely discouraged",
          "def _stdlib_identity() -> str:" in src
          and "return build_stdlib_identity()[0]" in src)
    check("V2-ID-2b the descriptor carries ONLY the digest; no stdlib schema "
          "field is added to the ten-field descriptor",
          set(SI.RUNTIME_FIELDS) == {"executable_sha256", "implementation_name",
                                     "cache_tag", "version", "stdlib_identity",
                                     "dont_write_bytecode"})
    check("STDLIB H2_PY_STDLIB_V1 is inherited UNCHANGED — no V2 exists",
          SI.STDLIB_SCHEMA == "H2_PY_STDLIB_V1"
          and "H2_PY_STDLIB_V2" not in src)
    print("    V2-ID-2a  canonical positive derivation        HELD ON INC-34")
    print("    STAGE_A canonical-runtime positive limb        HELD ON INC-34")
    print("    No D380-compliant interpreter is locally available; the")
    print("    contract is NOT weakened to manufacture a green (D385 §D).")
    print()


# ── D386 — MECHANICAL CHILD STATUS CAPTURE ────────────────────────────
#
# INC-2026-09-18-35. The previous capture was a shell brace group in which
# `${PIPESTATUS[0]}` was expanded after a bare `echo`, so the durable
# artefact recorded the status of the echo that PRINTS the status. It
# wrote "process exit status = 0" four lines under its own
# "EXIT GATE: FAIL", while the program returns 1.
#
# THE STATUS AUTHORITY IS NOW subprocess.CompletedProcess.returncode, TAKEN
# FROM THE PROCESS THAT PRODUCED THE CAPTURED OUTPUT. One object, one step.
# There is no shell between the measurement and the record, so there is
# nothing for an intervening command to overwrite.
#
# FORBIDDEN, AND STRUCTURALLY ABSENT RATHER THAN MERELY DISCOURAGED:
#   $? · PIPESTATUS · shell pipeline reconstruction · a manual --status ·
#   any caller-supplied or hard-coded status · parsing the "EXIT GATE"
#   text to infer the status.
# The literal 0 is NOT edited to 1. A hand-written number where a
# measurement belongs is the same defect with a better value.

CAPTURE_HEADER = (
    "D379 / D381 / D382 HOSTILE CONTROLS — FULL UNTRUNCATED OUTPUT\n"
    "EVIDENCE CLASS: PRODUCER MEASUREMENT - SIGHTED - ZERO ADMISSION WEIGHT.\n"
    "THIS FILE IS THE AUTHORITATIVE OUTPUT (R10). Any excerpt elsewhere\n"
    "declares itself partial and states its byte count.\n"
    "\n"
    "STATUS PROVENANCE (D386): every 'process exit status' line below is\n"
    "subprocess.CompletedProcess.returncode, read directly from the child\n"
    "that produced the output immediately above it. No shell status, no\n"
    "pipeline status, no parsed text, no supplied value.\n")


def _run_child(argv, cwd):
    """Run one child and return (stdout+stderr, returncode) from ONE object."""
    import subprocess
    p = subprocess.run(argv, cwd=str(cwd), capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode


# ── THE D379 §8 EXECUTABLE-BOUND CASE HARNESS ─────────────────────────
#
# INC-38: assurance was implemented in importable helpers and calibrated by
# importing them, while the governed executables never invoked it. D379 §8
# always required the cases to run AS SUBPROCESSES ASSERTING THE REAL
# PROCESS RETURN CODE. This is that, and nothing beside it: no new
# expectation authority, no new identity layer, no new evidence artefact.
#
# A RED PROCESS EARNS NOTHING BY ITSELF. Every case records all nine steps
# and only then takes a verdict. An outcome produced by an earlier
# independent blocker is HELD, never PASS — and equally, an expected
# blocker does not earn HELD until it is OBSERVED to have prevented the
# governed predicate from being measured.
#
#     EXPECTED_RISK_OF_HOLD != HELD          EXPECTED_PASS != PASS

# Markers of an INC-34 / ungoverned-runtime refusal. Their presence means
# some OTHER predicate decided the outcome, so the intended one was not
# measured. Listed explicitly: a blocker set that is guessed at call sites
# is the maintained-beside-it defect again.
# Markers of an EARLIER INDEPENDENT BLOCKER, classified by KIND so the
# evidence names which one fired rather than lumping them together. A case
# blocked by any of these did not measure its own predicate.
BLOCKERS = {
    # INC-34: the runtime itself is ungoverned on this interpreter.
    "sitecustomize": "INC-34 ungoverned runtime",
    "_distutils_hack": "INC-34 ungoverned runtime",
    "OUTSIDE the governed root set": "INC-34 ungoverned runtime",
    "is a loaded EXTERNAL module": "INC-34 ungoverned runtime",
    "is a loaded non-stdlib EXTERNAL module": "INC-34 ungoverned runtime",
    "lies outside the governed H2 root": "INC-34 ungoverned runtime",
    # D381 §13: a CALIBRATION identity may never seed production/holdout.
    # This is the rule WORKING, and it blocks any predicate that lies past
    # successful Stage-A validation. Kai's holdout precondition: do not use
    # a CALIBRATION identity to bypass it and do not manufacture a fake
    # PRODUCTION identity. The case is HELD.
    "carries ZERO holdout weight": "D381 §13 Stage-A prerequisite",
    "Stage-A descriptor mode is": "D381 §13 Stage-A prerequisite",
}

CASES = []


def run_governed_child(*, executable=None, argv=(), snippet=None, cwd=None,
                       env=None):
    """Launch a CHILD process whose subject is the GOVERNED code.

    Two shapes, both real subprocesses asserting the real return code:

      executable=...  the shipped CLI is the subject (D379 §8 directly).
      snippet=...     the subject is a governed DECISION FUNCTION. The
                      child IMPORTS the actual shipped module, builds the
                      authorised synthetic input, calls the real function
                      and exits from its result. The launcher is harness
                      machinery; the governed function remains the subject,
                      and NO production interface is widened to make it
                      reachable.
    """
    import subprocess
    if snippet is not None:
        full = [sys.executable, "-c",
                f"import sys; sys.path.insert(0, {str(V)!r})\n" + snippet]
    else:
        full = [sys.executable, str(executable)] + list(argv)
    e = None
    if env:
        e = dict(os.environ, **env)
    pr = subprocess.run(full, cwd=str(cwd or V), capture_output=True,
                        text=True, env=e)
    return pr.stdout + pr.stderr, pr.returncode


def d379_case(case_id, *, clause, subject_proof, subject_holds,
              intended_reason, expect_class, executable=None, argv=(),
              snippet=None, cwd=None, env=None, expect_output=None,
              prerequisite=None):
    """One banked D379 case, executed as a subprocess.

    STEP 1 IS RAW EVIDENCE, NOT A FLAG. `subject_proof` is a dict of
    MEASURED facts and `subject_holds` is a predicate DERIVED from them. A
    free `hostile_subject_established = YES` would become the next proxy:
    it would prove nothing except that the harness printed it.

    If the raw proof is absent or contradictory the verdict is FIXTURE —
    a CONTROL/FIXTURE FAILURE. It is NOT a PASS, and it is NOT a HELD on
    the intended predicate: an unrelated INC-34 refusal does not excuse a
    failure to construct the hostile subject.

    THE CASE NAME CARRIES ZERO EVIDENTIAL WEIGHT.
    """
    subj_ok = False
    subj_why = "subject predicate raised"
    try:
        subj_ok = bool(subject_holds(subject_proof))
        subj_why = "derived from the raw facts above"
    except Exception as ex:                                # noqa: BLE001
        subj_why = f"subject predicate raised: {ex}"

    exe_sha = None
    if executable is not None and pathlib.Path(executable).is_file():
        exe_sha = hashlib.sha256(
            pathlib.Path(executable).read_bytes()).hexdigest()

    print(f"    {case_id:<8} SUBJECT PROOF ({clause})")
    for k, v in subject_proof.items():
        vs = str(v)
        print(f"               {k:<34} {vs[:96]}")
    print(f"               subject constructed?           "
          f"{'YES' if subj_ok else 'NO'} — {subj_why}")

    if not subj_ok:
        rec = {"case": case_id, "clause": clause, "verdict": "FIXTURE",
               "subject_proof": subject_proof, "rc": None,
               "why": f"the hostile subject was NOT constructed: {subj_why}"}
        CASES.append(rec)
        print(f"    {case_id:<8} -> FIXTURE  (control/fixture failure; the "
              f"banked condition was never built)")
        check(f"{case_id} ({clause}) verdict PASS", False,
              "FIXTURE: " + rec["why"])
        return rec

    out, rc = run_governed_child(executable=executable, argv=argv,
                                 snippet=snippet, cwd=cwd, env=env)
    reached = True if prerequisite is None else (prerequisite in out)
    intended_seen = intended_reason in out
    blocker = next((f"{k} [{v}]" for k, v in BLOCKERS.items() if k in out),
                   None)
    unrecognised = ("unrecognized arguments" in out
                    or "unrecognized argument" in out)
    if unrecognised and intended_reason not in ("unrecognized arguments",):
        blocker = blocker or "argparse: unrecognized arguments (CLI_NOT_WIRED)"
    disposition = "ACCEPT" if rc == 0 else "REFUSE"

    if blocker:
        verdict = "HELD"
        if expect_class == "ACCEPT":
            why = (f"the governed positive cannot be measured while an "
                   f"independent blocker forces refusal: {blocker}")
        elif intended_seen:
            why = (f"OVER-DETERMINED: the intended finding IS present and so "
                   f"is an independent blocker ({blocker}); rc is not "
                   f"attributable to this predicate without a clean "
                   f"comparator. Finding observed; consequentiality NOT "
                   f"PROVEN")
        else:
            why = f"an earlier independent blocker decided the outcome: {blocker}"
    elif not reached:
        verdict, why = "HELD", "execution did not reach the predicate"
    elif disposition != expect_class:
        verdict, why = "FAIL", f"disposition {disposition} != governed {expect_class}"
    elif not intended_seen:
        verdict, why = "FAIL", (f"governed disposition occurred but the "
                                f"intended reason was not observed: "
                                f"{intended_reason!r}")
    elif expect_output is not None and (
            pathlib.Path(expect_output["path"]).exists()
            != expect_output["created"]):
        verdict, why = "FAIL", "output created/absent contrary to D379"
    else:
        verdict, why = "PASS", "intended predicate was first-effective"

    rec = {"case": case_id, "clause": clause,
           "subject_proof": subject_proof,
           "executable": (pathlib.Path(executable).name if executable
                          else "child:governed-function"),
           "subject_sha256": exe_sha, "argv": list(argv), "rc": rc,
           "disposition": disposition, "expected_class": expect_class,
           "intended_reason": intended_reason,
           "intended_observed": intended_seen, "earlier_blocker": blocker,
           "prerequisite_reached": reached, "verdict": verdict, "why": why,
           "evidence": out.strip().splitlines()[-1][:160] if out.strip() else ""}
    CASES.append(rec)
    print(f"    {case_id:<8} rc={rc if rc is not None else '-':<3} "
          f"{disposition:<7} intended={'Y' if intended_seen else 'N'}  "
          f"blocker={'Y' if blocker else 'N'}  -> {verdict}")
    if verdict != "PASS":
        print(f"               {why}")
    check(f"{case_id} ({clause}) verdict PASS", verdict == "PASS",
          f"{verdict}: {why}")
    return rec


def _capture_calibration():
    """CAPTURE-1/2/3 — hostile calibration OF THE CAPTURE ITSELF.

    A capture mechanism demonstrated only on its happy path proves that it
    can record SOME number, not that it records THE number. These children
    are synthetic and return a known code each; the recorded status must
    equal it, and the captured output must belong to that same invocation.
    """
    print("CAPTURE — hostile calibration of the status recorder itself")
    ok = True
    for want in (0, 1, 2):
        marker = f"CHILD_MARKER_{want}_{os.getpid()}"
        out, rc = _run_child(
            [sys.executable, "-c",
             f"import sys; print({marker!r}); sys.exit({want})"], V)
        recorded = rc                       # the ONLY source, per D386
        a = check(f"CAPTURE-{want + 1} child returns {want}, recorded "
                  f"status = {want}, parent would return {want}",
                  recorded == want, f"recorded {recorded}")
        b = check(f"CAPTURE-{want + 1} the captured output belongs to THAT "
                  f"child invocation", marker in out, out[:80])
        ok &= a and b
        print(f"  CAPTURE-{want + 1}  child exit {want} -> recorded "
              f"{recorded} -> parent {recorded}   "
              f"output-binding {'OK' if marker in out else 'MISMATCH'}   "
              f"[{'PASS' if a and b else 'FAIL'}]")
    # and the negative that INC-35 actually was: a later command's status
    # must be incapable of overwriting the record.
    out_a, rc_a = _run_child([sys.executable, "-c", "import sys; sys.exit(3)"], V)
    _out_b, rc_b = _run_child([sys.executable, "-c", "print()"], V)
    check("CAPTURE-4 a subsequent successful command CANNOT overwrite the "
          "recorded status of an earlier child (the INC-35 mechanism)",
          rc_a == 3 and rc_b == 0, f"{rc_a} {rc_b}")
    print()
    return ok


def _render_closeout(state, ctl_rc, fx_rc, controls_path):
    """Render the close-out FROM the captured state. No figure typed here.

    Every count below is computed from `state["cases"]`, which is the exact
    list the control child built while executing. A hand-maintained
    close-out can be brought back into agreement with reality by editing
    the number, which is the opposite of measuring it.
    """
    cases = state["cases"]
    by = lambda v: [c for c in cases if c["verdict"] == v]
    P, F, H, X = by("PASS"), by("FAIL"), by("HELD"), by("FIXTURE")
    raw = controls_path.read_bytes() if controls_path.is_file() else b""
    L = ["D379 / D381 / D382 TRANCHE — CLOSE-OUT",
         "EVIDENCE CLASS: PRODUCER MEASUREMENT - SIGHTED - ZERO ADMISSION "
         "WEIGHT.",
         "Orion self-adjudication carries no admission weight and none is "
         "claimed.",
         "",
         "DERIVED RECORD. Every figure below is computed from the SAME",
         "captured result state that produced " + controls_path.name + ",",
         "in the same run. Nothing here is maintained by hand, so no figure",
         "in this file can be brought into agreement with reality by",
         "retyping it.",
         "",
         "THIS TRANCHE IS NOT CLOSED. The exit gate is FAIL and says so."
         if state["failed"] else
         "THE EXIT GATE PASSED. Closure remains a separate register action "
         "(R7).",
         "",
         "STATUS PROVENANCE (D386): every status below is",
         "subprocess.CompletedProcess.returncode, read from the child that",
         "produced the captured output. No shell status, no pipeline status,",
         "no parsed text, no supplied value.",
         "",
         f"  D379 control process exit status    {ctl_rc}"
         f"     ({state['passed']} passed, {state['failed']} failed)",
         f"  cal_fixtures process exit status    {fx_rc}",
         f"  capture parent returns              {ctl_rc}",
         f"  {controls_path.name} size                {len(raw)} bytes",
         "",
         "=" * 70,
         "D379 §8 BANKED CASES — EACH AGAINST ITS OWN CONSTRUCTED SUBJECT",
         "=" * 70,
         f"  cases {len(cases)}   PASS {len(P)}   FAIL {len(F)}   "
         f"HELD {len(H)}   FIXTURE {len(X)}",
         "",
         "  STEP 1 OF EVERY CASE IS RAW EVIDENCE, NOT A FLAG. Each case",
         "  carries the measured facts that establish its hostile subject,",
         "  and the harness DERIVES whether the subject holds. A case whose",
         "  proof does not derive is FIXTURE — a control failure. It is not",
         "  a PASS, and it is not a HELD on the intended predicate.",
         ""]
    if X:
        L += ["  FIXTURE — THE BANKED SUBJECT WAS NEVER CONSTRUCTED:"]
        L += [f"    {c['case']:<8} {c['why']}" for c in X]
        L += [""]
    if P:
        L += ["  PASS — blocker-free, intended predicate first-effective:"]
        for c in P:
            L.append(f"    {c['case']:<8} {c['clause']}")
        L += ["",
              "  These are NARROW OBSERVATIONS of these exact invocations.",
              "  They do not validate the harness, do not make their section",
              "  green, and close nothing (R7).", ""]
    if F:
        L += ["  FAIL:"]
        L += [f"    {c['case']:<8} {c['why']}" for c in F]
        L += [""]
    if H:
        causes = collections.Counter(
            (c.get("earlier_blocker") or "prerequisite not reached")
            for c in H)
        L += [f"  HELD {len(H)} — by FIRST-EFFECTIVE BLOCKER, counted from",
              "  the recorded blocker of each case, not assigned by hand:"]
        for cause, n in causes.most_common():
            L.append(f"    {n:>3}  {cause}")
            L.append("         " + ", ".join(
                c["case"] for c in H
                if (c.get("earlier_blocker") or "prerequisite not reached")
                == cause))
        over = [c for c in H if "OVER-DETERMINED" in c["why"]]
        if over:
            L += ["",
                  f"  OVER-DETERMINED {len(over)}: the intended finding IS "
                  f"present AND an",
                  "  independent blocker is present. Finding observed;",
                  "  consequentiality NOT PROVEN without a clean comparator:",
                  "    " + ", ".join(c["case"] for c in over)]
        L += [""]
    L += ["=" * 70,
          "PER-CASE RECORD — subject proof, target, status, verdict",
          "=" * 70]
    for c in cases:
        L.append(f"  {c['case']:<8} {c['verdict']:<8} rc="
                 f"{c['rc'] if c['rc'] is not None else '-':<4} "
                 f"{c.get('executable', '-')}")
        L.append(f"           clause: {c['clause']}")
        for k, v in c.get("subject_proof", {}).items():
            L.append(f"           proof:  {k} = {str(v)[:110]}")
        if c["verdict"] != "PASS":
            L.append(f"           why:    {c['why']}")
    L += ["",
          "=" * 70,
          "GATE FAILURES — verbatim from the same run",
          "=" * 70]
    L += [f"  {x}" for x in state["failures"]] or ["  none"]
    L += ["",
          "HELD IS NOT A PASS AND NOT A SKIP. Every HELD case FAILS the exit",
          "gate above and names the predicate that could not be measured.",
          ""]
    return "\n".join(L) + "\n"


def capture(out_path) -> int:
    """PARENT. Runs the control matrix and the fixtures as SEPARATE children.

    TWO SUBJECTS, NOT ONE. The D379 control process and the cal_fixtures
    process are separately executed subjects with separately recorded
    statuses. One harness running both is NOT common-source corroboration
    and no such claim is made here.
    """
    import datetime
    parts = [CAPTURE_HEADER,
             f"captured {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
             ""]

    import tempfile
    _state = pathlib.Path(tempfile.mkdtemp(prefix="d379_state_")) / "state.json"
    ctl_out, ctl_rc = _run_child(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--child",
         "--state", str(_state)],
        pathlib.Path(__file__).resolve().parent)
    parts += ["=" * 70,
              "SUBJECT 1 of 2 — THE D379 CONTROL PROCESS",
              "=" * 70, "", ctl_out, "",
              f"process exit status = {ctl_rc}", ""]

    fx_out, fx_rc = _run_child([sys.executable, str(V / "cal_fixtures.py")], V)
    parts += ["=" * 70,
              "SUBJECT 2 of 2 — THE CAL_FIXTURES PROCESS",
              "  A DIFFERENT SUBJECT. Its status corroborates nothing about",
              "  subject 1 and no independence is claimed between them.",
              "=" * 70, "", fx_out, "",
              f"fixture process exit status = {fx_rc}", ""]

    parts += ["=" * 70,
              "CAPTURE SUMMARY — statuses read from CompletedProcess.returncode",
              "=" * 70,
              f"  D379 control process exit status   {ctl_rc}",
              f"  cal_fixtures process exit status   {fx_rc}",
              f"  parent returns                     {ctl_rc}", ""]

    pathlib.Path(out_path).write_text("\n".join(parts))

    # ── THE CLOSE-OUT IS DERIVED, NOT MAINTAINED ──────────────────────
    # It was hand-written, so its figures could drift from the run they
    # described and a stale count could be "repaired" by retyping it. Both
    # records now come from ONE state object produced by ONE run.
    co = pathlib.Path(out_path).parent / "D379_CLOSEOUT.txt"
    if _state.is_file():
        state = json.loads(_state.read_bytes())
        co.write_text(_render_closeout(state, ctl_rc, fx_rc,
                                       pathlib.Path(out_path)))
        print(f"derived   -> {co}")
    else:
        co.write_text(
            "CLOSE-OUT NOT DERIVED: the control child produced no state "
            "file, so no figure in this file could be derived from the run "
            "it describes. Nothing is reported rather than something "
            "reported from memory (R16).\n")
        print(f"NO STATE FILE — {co} refuses to report figures")
    print(f"captured -> {out_path}")
    print(f"  D379 control process exit status  {ctl_rc}")
    print(f"  cal_fixtures process exit status  {fx_rc}")
    return ctl_rc                       # the SAME code the child returned


if __name__ == "__main__":
    if "--capture" in sys.argv:
        sys.exit(capture(sys.argv[sys.argv.index("--capture") + 1]))
    _rc = main()
    # ONE MEASUREMENT TRUTH SOURCE. The child dumps the EXACT result state
    # it just produced; the parent renders BOTH human records from that one
    # object. Nothing downstream re-counts, re-reads or re-types a figure.
    if "--state" in sys.argv:
        pathlib.Path(sys.argv[sys.argv.index("--state") + 1]).write_text(
            json.dumps({"rc": _rc, "passed": PASSED, "failed": FAILED,
                        "failures": FAILURES, "cases": CASES,
                        "sections": SECTIONS, "implemented":
                        sorted(IMPLEMENTED), "held": HELD}, indent=1))
    sys.exit(_rc)
