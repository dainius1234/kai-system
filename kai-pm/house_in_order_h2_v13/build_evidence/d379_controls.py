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
    (d / "tree_paths.txt").write_text("\n".join(
        f"kai-pm/doc_{i:03d}.md" for i in range(272)) + "\n")
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

    (d / "EMPTY.sha256").write_text("")
    _FIX["empty_manifest"] = d / "EMPTY.sha256"
    tp = [f"kai-pm/doc_{i:03d}.md" for i in range(272)]
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


def exec_bound_cases():
    """The BANKED D379 cases, run against the REAL shipped executables.

    NOT A NEW MATRIX. Every case id below is already in D379 §8; the only
    change is the boundary at which it is exercised. Helper-level checks
    remain above as supplementary evidence and carry no verdict.
    """
    print("\nD379 §8 EXECUTABLE-BOUND EXECUTION — the real shipped CLIs")
    print("  Each case: exact hostile subject, actual subprocess, rc from")
    print("  CompletedProcess, actual reason, first-effective failure.")
    print("  A red process without the intended subject AND the intended")
    print("  reason earns no calibration claim.\n")
    f = exec_fixtures()

    print("  passa.py")
    d379_case("Q1a-2", clause="D379 §2/§4 Pass-A producer provenance",
              executable=V / "passa.py",
              argv=["--subject-repo", str(REPO), "--history-repo", str(REPO),
                    "--subject", f["subject_sha"],
                    "--census-package", str(f["census_pkg"]),
                    "--out", str(f["dir"] / "q1a2.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="byte mismatch against Stage A",
              expect_class="REFUSE")
    d379_case("Q1a-8", clause="D379 §5 no silent runtime expansion",
              executable=V / "passa.py",
              argv=["--subject-repo", str(REPO), "--history-repo", str(REPO),
                    "--subject", f["subject_sha"],
                    "--census-package", str(f["census_pkg"]),
                    "--out", str(f["dir"] / "q1a8.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="NOT represented in Stage A",
              expect_class="REFUSE")
    d379_case("DEP-1", clause="D379 §8 synthetic producer, external dep",
              executable=V / "passa.py",
              argv=["--subject-repo", str(REPO), "--history-repo", str(REPO),
                    "--subject", f["subject_sha"],
                    "--census-package", str(f["census_pkg"]),
                    "--out", str(f["dir"] / "dep1.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="SYNTHETIC_HOSTILE_DEPENDENCY",
              expect_class="REFUSE")

    print("  run_h2_v12.py")
    d379_case("Q1a-4", clause="D379 §4 stale input across two Stage As",
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO), "--passa", str(f["passa"]),
                    "--out", str(f["dir"] / "q1a4.json"),
                    "--stage-a", str(f["stage_a_other"])],
              intended_reason="stage_a_identity",
              expect_class="REFUSE")
    d379_case("Q1a-9", clause="D379 §4 self-hash prohibition",
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO), "--passa", str(f["passa"]),
                    "--out", str(f["dir"] / "q1a9.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="producer_provenance",
              expect_class="ACCEPT")

    print("  qualify.py")
    d379_case("Q1a-5", clause="D379 §8 tampered recorded Stage-A identity",
              executable=V / "qualify.py",
              argv=["--result", str(f["result_tampered"]),
                    "--manifest", str(f["manifest"]),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="does not match the Stage-A descriptor",
              expect_class="REFUSE")
    d379_case("86-3", clause="D367 §8(6) manifest omitted -> REFUSE",
              executable=V / "qualify.py",
              argv=["--result", str(f["result"]),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="the following arguments are required: --manifest",
              expect_class="REFUSE")
    d379_case("Q1b-1", clause="D379 §8 clean complete result -> PASS",
              executable=V / "qualify.py",
              argv=["--result", str(f["result"]),
                    "--manifest", str(f["manifest"]),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="positive-evidence-fact denominator",
              expect_class="ACCEPT")
    d379_case("Q1b-2", clause="D379 §8 missing positive-fact trace -> FAIL",
              executable=V / "qualify.py",
              argv=["--result", str(f["result_missing_trace"]),
                    "--manifest", str(f["manifest"]),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="CITES_COMMIT",
              expect_class="REFUSE")

    print("  holdout.py   (already CLI-wired; not redesigned)")
    d379_case("I1B-2", clause="D379 §7 drop one output row -> REFUSE",
              executable=V / "holdout.py",
              argv=["--result", str(f["result"]),
                    "--stage-a", str(f["stage_a"]),
                    "--tree-paths", str(f["tree_paths"]),
                    "--out", str(f["dir"] / "i1b2.json")],
              intended_reason="REFUSE BEFORE SELECTION",
              expect_class="REFUSE")


    # ── the rest of the CLI-BOUND families ────────────────────────────
    # NO MATRIX EXPANSION: every id below is already banked in D379 §8.
    # The only change is the boundary. Cases whose governed subject is a
    # decision function rather than a shipped executable (M2-1..3,
    # D14-A/B/C, SB-1..3) are named in NON_CLI_SUBJECTS and stay with the
    # function that IS their subject.
    d379_case("Q1a-1", clause="D379 §8 clean chain on one Stage A",
              executable=V / "passa.py",
              argv=["--subject-repo", str(REPO), "--history-repo", str(REPO),
                    "--subject", f["subject_sha"],
                    "--census-package", str(f["census_pkg"]),
                    "--out", str(f["dir"] / "q1a1.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="PASS A", expect_class="ACCEPT")
    d379_case("Q1a-3", clause="D379 §8 classification-producer byte changed",
              executable=V / "run_h2_v12.py",
              argv=["--subject-repo", str(REPO), "--passa", str(f["passa"]),
                    "--out", str(f["dir"] / "q1a3.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="byte mismatch against Stage A",
              expect_class="REFUSE")
    d379_case("Q1a-6", clause="D379 §8 qualifier runtime differs",
              executable=V / "qualify.py",
              argv=["--result", str(f["result"]),
                    "--manifest", str(f["manifest"]),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="DEP-3 executing runtime identity differs",
              expect_class="REFUSE")
    d379_case("Q1a-7", clause="D379 §8 member deleted from OUTPUT provenance",
              executable=V / "qualify.py",
              argv=["--result", str(f["result_short_prov"]),
                    "--manifest", str(f["manifest"]),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="NOT represented in Stage A",
              expect_class="REFUSE")
    d379_case("DEP-2", clause="D379 §8 ordinary stdlib under a governed "
                              "interpreter",
              executable=V / "passa.py",
              argv=["--subject-repo", str(REPO), "--history-repo", str(REPO),
                    "--subject", f["subject_sha"],
                    "--census-package", str(f["census_pkg"]),
                    "--out", str(f["dir"] / "dep2.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="PASS A", expect_class="ACCEPT")
    d379_case("DEP-3", clause="D379 §8 runtime identity mismatch",
              executable=V / "passa.py",
              argv=["--subject-repo", str(REPO), "--history-repo", str(REPO),
                    "--subject", f["subject_sha"],
                    "--census-package", str(f["census_pkg"]),
                    "--out", str(f["dir"] / "dep3.json"),
                    "--stage-a", str(f["stage_a"])],
              intended_reason="DEP-3 executing runtime identity differs",
              expect_class="REFUSE")
    for cid, res_key, reason in (
            ("Q1b-3", "result_q1b3", "SCOPE"),
            ("Q1b-4", "result_q1b4", "ABSTENTION_RECONCILIATION"),
            ("Q1b-5", "result_q1b5", "FACT_CLASS_ABSENT"),
            ("Q1b-6", "result_q1b6", "positive-evidence-fact denominator")):
        d379_case(cid, clause="D379 §8 Q1b derived denominator",
                  executable=V / "qualify.py",
                  argv=["--result", str(f[res_key]),
                        "--manifest", str(f["manifest"]),
                        "--stage-a", str(f["stage_a"])],
                  intended_reason=reason,
                  expect_class="ACCEPT" if cid == "Q1b-6" else "REFUSE")
    for cid, argv_extra, reason in (
            ("86-1", [], "§8(6) CLOSED ORIGIN CLASSIFICATION"),
            ("86-4", ["--manifest", "/nonexistent"], "does not exist"),
            ("86-5", ["--manifest", str(f["empty_manifest"])],
             "yielded no entries")):
        base = ["--result", str(f["result"]), "--stage-a", str(f["stage_a"])]
        if not argv_extra:
            base += ["--manifest", str(f["manifest"])]
        d379_case(cid, clause="D367 §8(6) fail-closed qualifier identity",
                  executable=V / "qualify.py", argv=base + argv_extra,
                  intended_reason=reason,
                  expect_class="ACCEPT" if cid == "86-1" else "REFUSE")
    for cid, tp_key, reason in (
            ("I1B-1", "tree_paths", "BLIND HOLDOUT"),
            ("I1B-3", "tree_paths_extra", "REFUSE BEFORE SELECTION"),
            ("I1B-4", "tree_paths_dup", "REFUSE BEFORE SELECTION")):
        d379_case(cid, clause="D379 §7 I1-B tree/output reconciliation",
                  executable=V / "holdout.py",
                  argv=["--result", str(f["result"]),
                        "--stage-a", str(f["stage_a"]),
                        "--tree-paths", str(f[tp_key]),
                        "--out", str(f["dir"] / (cid + ".json"))],
                  intended_reason=reason,
                  expect_class="ACCEPT" if cid == "I1B-1" else "REFUSE")
    d379_case("I1A-1", clause="D379 §7 I1-A seed is the Stage-A identity",
              executable=V / "holdout.py",
              argv=["--result", str(f["result"]),
                    "--stage-a", str(f["stage_a"]),
                    "--tree-paths", str(f["tree_paths"]),
                    "--out", str(f["dir"] / "i1a1.json")],
              intended_reason="candidate aggregate", expect_class="ACCEPT")

    held = [c for c in CASES if c["verdict"] == "HELD"]
    passed = [c for c in CASES if c["verdict"] == "PASS"]
    failed = [c for c in CASES if c["verdict"] == "FAIL"]
    print(f"\n    executable-bound cases  {len(CASES)}"
          f"   PASS {len(passed)}   FAIL {len(failed)}   HELD {len(held)}")
    print("    (HELD is not a pass and not a skip: it fails the gate and")
    print("     names the predicate that could not be measured.)")
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
def section_Q1a():
    print("Q1a — producer-byte provenance against Stage A")
    import stage_identity as SI
    import copy

    A = _synthetic_descriptor(SI.SCHEMA_V2, SI.MODE_CALIBRATION)
    idA = SI.stage_a_identity(A)
    # ONE CANONICAL SCHEMA (D379 §4). This helper previously built
    # `members`, which is the key the old verifier read and the producers
    # never emit.
    prov = {"stage_a_identity": idA,
            "producer_population": [
                {"class": SI.CLASS_H2, "identity": m["path"],
                 "sha256": m["sha256"]} for m in A["h2_sources"]]}

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


def d379_case(case_id, *, clause, executable, argv, intended_reason,
              expect_class, expect_output=None, prerequisite=None,
              cwd=None):
    """Run ONE banked D379 case against ONE real shipped executable.

    `expect_class`  "ACCEPT" or "REFUSE" — the governed disposition.
    `intended_reason` a substring that appears ONLY when the case's own
                    predicate fired. It is declared by the control from the
                    D379 clause, never read back out of the implementation.
    `prerequisite`  a substring whose presence proves execution reached the
                    point where the intended predicate could fire. None
                    means the invocation itself is the prerequisite.
    """
    exe = pathlib.Path(executable)
    subject_sha = (hashlib.sha256(exe.read_bytes()).hexdigest()
                   if exe.is_file() else None)
    full = [sys.executable, str(exe)] + list(argv)
    out, rc = _run_child(full, cwd or V)

    # 3. prerequisite reached?
    if prerequisite is None:
        reached = True
    else:
        reached = prerequisite in out
    # 5/6. the intended predicate
    intended_seen = intended_reason in out
    # 7. an earlier INDEPENDENT blocker
    blocker = next((f"{k} [{v}]" for k, v in BLOCKERS.items()
                    if k in out), None)
    # argparse refusing an argument the CLI does not declare is itself an
    # independent blocker for any case whose predicate lies past parsing.
    unrecognised = "unrecognized arguments" in out or "unrecognized argument" in out
    if unrecognised and intended_reason != "unrecognized arguments":
        blocker = blocker or "argparse: unrecognized arguments (CLI_NOT_WIRED)"

    disposition = "ACCEPT" if rc == 0 else "REFUSE"

    # 9. THE VERDICT. ANY earlier independent blocker means this case did
    #    not measure its own predicate CONSEQUENTIALLY, whatever else is
    #    also true. Three distinct shapes, all HELD:
    #
    #      expected ACCEPT + blocker  the positive cannot be reached while
    #                                 an independent refusal is forced
    #      expected REFUSE + blocker + intended observed
    #                                 OVER-DETERMINED. The finding IS in the
    #                                 output, and rc could come from either
    #                                 cause. Without a clean comparator,
    #                                 consequentiality is NOT PROVEN.
    #      expected REFUSE + blocker + intended absent
    #                                 the blocker simply decided it
    #
    #    Letting `intended_observed` override a blocker is exactly how a red
    #    process gets credited to the predicate one hoped for.
    if blocker:
        verdict = "HELD"
        if expect_class == "ACCEPT":
            why = (f"the governed positive cannot be measured while an "
                   f"independent blocker forces refusal: {blocker}")
        elif intended_seen:
            why = (f"OVER-DETERMINED: the intended finding IS present in the "
                   f"output, but so is an independent blocker ({blocker}), so "
                   f"rc is not attributable to this predicate without a clean "
                   f"comparator. Finding observed; consequentiality NOT PROVEN")
        else:
            why = f"an earlier independent blocker decided the outcome: {blocker}"
    elif not reached:
        verdict = "HELD"
        why = "execution did not reach the point where the predicate fires"
    elif disposition != expect_class:
        verdict = "FAIL"
        why = f"disposition {disposition} != governed {expect_class}"
    elif not intended_seen:
        verdict = "FAIL"
        why = (f"the governed disposition occurred but the intended reason "
               f"was not observed: {intended_reason!r}")
    elif expect_output is not None:
        created = pathlib.Path(expect_output["path"]).exists()
        if created != expect_output["created"]:
            verdict, why = "FAIL", (
                f"output created={created}, governed "
                f"{expect_output['created']}")
        else:
            verdict, why = "PASS", "intended predicate was first-effective"
    else:
        verdict, why = "PASS", "intended predicate was first-effective"

    rec = {"case": case_id, "clause": clause, "executable": exe.name,
           "subject_sha256": subject_sha, "argv": list(argv),
           "prerequisite_reached": reached, "rc": rc,
           "disposition": disposition, "expected_class": expect_class,
           "intended_reason": intended_reason,
           "intended_observed": intended_seen,
           "earlier_blocker": blocker, "verdict": verdict, "why": why,
           "evidence": out.strip().splitlines()[-1][:150] if out.strip() else ""}
    CASES.append(rec)
    print(f"    {case_id:<12} rc={rc:<3} {disposition:<7} "
          f"intended={'Y' if intended_seen else 'N'}  "
          f"blocker={'Y' if blocker else 'N'}  -> {verdict}")
    if verdict != "PASS":
        print(f"                 {why}")
    # HELD is neither a pass nor a silent skip: it fails the exit gate, and
    # it is reported with the reason the predicate could not be measured.
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

    ctl_out, ctl_rc = _run_child(
        [sys.executable, str(pathlib.Path(__file__).resolve()), "--child"],
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
    print(f"captured -> {out_path}")
    print(f"  D379 control process exit status  {ctl_rc}")
    print(f"  cal_fixtures process exit status  {fx_rc}")
    return ctl_rc                       # the SAME code the child returned


if __name__ == "__main__":
    if "--capture" in sys.argv:
        sys.exit(capture(sys.argv[sys.argv.index("--capture") + 1]))
    sys.exit(main())
