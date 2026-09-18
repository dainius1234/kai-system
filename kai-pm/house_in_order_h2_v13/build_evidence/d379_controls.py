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
import json
import pathlib
import sys

V = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V))

import classify                                              # noqa: E402
import passa                                                 # noqa: E402
from envelope import Witness                                 # noqa: E402

PASSED, FAILED, FAILURES = 0, 0, []
# Sections of the D379 hostile matrix. Implemented sections run; the rest
# fail as NOT_IMPLEMENTED so this file can never report a green tranche.
SECTIONS = ["M2", "D14", "Q1a", "Q1b", "86", "SB", "I1A", "I1B",
            "DEP", "STAGE_A", "STDLIB"]
IMPLEMENTED = {"M2", "SB"}


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

    section_M2()
    section_SB()

    print("-" * 70)
    print("SECTION COVERAGE")
    for s in SECTIONS:
        state = "IMPLEMENTED" if s in IMPLEMENTED else "NOT_IMPLEMENTED"
        print(f"  {s:<10} {state}")
        if s not in IMPLEMENTED:
            check(f"section {s} is implemented", False,
                  "NOT_IMPLEMENTED — this control file does not yet cover "
                  "this section of the D379 hostile matrix")
    print()
    print("=" * 70)
    print(f"{PASSED} passed, {FAILED} failed")
    for f in FAILURES:
        print(f"  FAIL {f}")
    print("EXIT GATE:", "PASS" if FAILED == 0 else "FAIL")
    print("=" * 70)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
