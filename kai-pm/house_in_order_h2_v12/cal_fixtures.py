#!/usr/bin/env python3
"""HOUSE_H2 v1.2 — HOSTILE FIXTURES. One fail-old/pass-new pair per
registered defect class D1-D17, as D367 2 requires.

FAIL-OLD RUNS AGAINST THE ACTUAL COMMITTED v1.1, imported from
`../house_in_order_h2_v11`. Not a reconstruction, not a paraphrase of
the old rule -- the real bytes. A repair proved only against a
remembered version of the defect proves nothing about the defect.

Each pair also proves the OPPOSITE side where one exists. A repair that
only demonstrates the corrected case can silently destroy the property
it was protecting: the first v1.2 draft of `bind_claims` imposed a
6000-byte window and quietly destroyed two authority claims Kai had
adjudicated CORRECT. That is why the five correct rows are asserted
unchanged here.
"""
from __future__ import annotations
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
OLD = HERE.parent / "house_in_order_h2_v11"
sys.path.insert(0, str(HERE))

import classify as cl                                          # noqa: E402
import envelope as E                                           # noqa: E402
import ontology as ont                                         # noqa: E402
import passa                                                   # noqa: E402
import subjectbind as sb                                       # noqa: E402
from envelope import Witness                                   # noqa: E402

SUBJECT = pathlib.Path("/tmp/tmp.6xNl2hBs2V/subject")
HISTORY = "/tmp/tmp.849NW5Ho8U/recover"

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
        print(f"          {str(detail)[:160]}")


def old(snippet):
    """Run a snippet against the COMMITTED v1.1 package IN A SUBPROCESS.

    v1.1's modules import each other by bare name, and three of those
    names -- ontology, passa, cal_fixtures -- collide with v1.2's. An
    in-process import would let one version shadow the other, and a
    fail-old result contaminated by new code proves nothing. So the old
    code runs in its own interpreter with only its own directory on the
    path: exactly the committed bytes, exactly as they ran.

    The snippet must `print(json.dumps(...))` its result.
    """
    import subprocess
    r = subprocess.run(
        [sys.executable, "-c", "import sys, json, re\n"
         f"sys.path.insert(0, {str(OLD)!r})\n" + snippet],
        capture_output=True, text=True, cwd=str(OLD))
    if r.returncode != 0:
        raise RuntimeError(f"fail-old subprocess failed: {r.stderr[-400:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def w(**kw):
    d = dict(witness_type="commit_sha", witness_value="97a3a61",
             source_path="x.md", source_selector="L30",
             local_context="J1-J7 are DONE (see commit 97a3a61)",
             applicability_scope="SPAN", evidence_total=1, evidence_shown=1,
             truncated=False, polarity="POSITIVE")
    d.update(kw)
    return Witness(**d)


def row(**kw):
    base = dict(path="x.md", title="", bytes=500, sha256="0" * 16,
                commits_in_window=1, last="2026-08-05", graphA_in=0,
                graphA_out=0, exe_ops=0, writers=[], readers=[],
                says_supersedes=False, witnesses={})
    base.update(kw)
    return base


# ── D1 evidence token -> whole-file verdict ───────────────────────────
def d1():
    print("\nD1 — a commit cited for ONE SENTENCE is not a whole-file binding")
    o = old('import classify2 as c; print(json.dumps(c.validity('
            '{"has_run":False,"has_sha":True,"has_date":False,'
            '"present_tense":False})))')
    check("D1a FAIL-OLD: committed v1.1 emits EXACT_SNAPSHOT from a bare flag",
          o["value"] == "EXACT_SNAPSHOT", o)
    span = {"COMMIT": [w().asdict()]}
    n = cl.validity(row(witnesses=span), None)
    check("D1b PASS-NEW: v1.2 ABSTAINS on a SPAN-scoped citation",
          n["value"] == "UNKNOWN", n)
    bound = {"COMMIT": [w(applicability_scope="WHOLE_FILE",
                          local_context="Audited snapshot: `97a3a61`").asdict()]}
    b = cl.validity(row(witnesses=bound), None)
    check("D1c BOUNDARY: a document-level binding STILL earns EXACT_SNAPSHOT",
          b["value"] == "EXACT_SNAPSHOT", b)
    check("D1d the envelope refuses the widening structurally",
          _raises(lambda: E.claim(w(), "EXACT_SNAPSHOT", scope="WHOLE_FILE")))


# ── D2 witness kind assumed from shape ────────────────────────────────
def d2():
    print("\nD2 — witness KIND is discriminated, never assumed from shape")
    for tok, ctx in (("ed25519", "- **Alternative:** ed25519 per-service keys."),
                     ("1700000000", "from timestamp 1700000000 within"),
                     ("b5e68a3", "recorded (`sha256:b5e68a3…`) from the")):
        hit = old(f'import passa; print(json.dumps(bool(passa.SHA.search('
                  f'{ctx!r}))))')
        check(f"D2a FAIL-OLD: v1.1 SHA pattern accepts {tok!r}", hit, ctx)
        m = next(m for m in passa.HEX.finditer(ctx) if m.group(0) == tok)
        kind, is_commit = passa.classify_token_kind(ctx, m, HISTORY, "x")
        check(f"D2b PASS-NEW: v1.2 rejects {tok!r} as a commit (-> {kind})",
              not is_commit, kind)
    ctx = "Audited snapshot: commit `2d830f25d569baa5ce955dd8d17e8f0744239876`"
    m = passa.HEX.search(ctx)
    kind, is_commit = passa.classify_token_kind(ctx, m, HISTORY, "x")
    check("D2c KNOWN-NEGATIVE: a REAL commit still resolves", is_commit, kind)


# ── D3 self-claim never checked ───────────────────────────────────────
def d3():
    print("\nD3 — a currency self-claim the history refutes cannot earn a verdict")
    import run_h2_v12 as R
    r = row(last="2026-08-22", witnesses={"DATE": [w(
        witness_type="DATE_STAMP", witness_value="2026-07-21",
        applicability_scope="WHOLE_FILE",
        local_context="**Last updated:** 2026-07-21").asdict()]})
    c = R.contradiction_of(r)
    check("D3a the contradiction is DETECTED", c is not None and c["drift_days"] == 32, c)
    check("D3b PASS-NEW: the verdict ABSTAINS",
          cl.validity(r, c)["value"] == "UNKNOWN")
    r2 = row(last="2026-07-21", witnesses=r["witnesses"])
    c2 = R.contradiction_of(r2)
    check("D3c KNOWN-NEGATIVE: a consistent claim is NOT contradicted", c2 is None, c2)
    check("D3d ... and still earns TIME_BOUND",
          cl.validity(r2, c2)["value"] == "TIME_BOUND")
    r3 = row(last="2026-06-02", witnesses={"DATE": [w(
        witness_type="DATE_STAMP", witness_value="2 Mar 2026",
        applicability_scope="WHOLE_FILE",
        local_context="> Version: 1.0 — 2 Mar 2026").asdict()]})
    check("D3e a VERSION date is not a currency claim, so not a contradiction",
          R.contradiction_of(r3) is None)


# ── D4 detector cannot fire ───────────────────────────────────────────
def d4():
    print("\nD4 — the RUN detector fires through markdown")
    for lit in ("**Last run:** 31570714150", "run `31894868473` (tree",
                "Deployed run 1 (`31605138566`)"):
        miss = old(f'import passa; print(json.dumps(not passa.RUN.search('
                   f'{lit!r})))')
        check(f"D4a FAIL-OLD: v1.1 RUN misses {lit[:26]!r}", miss, lit)
        hit = any(passa.RUN_NEAR.search(lit[max(0, m.start() - 24):m.start()])
                  or passa.RUN_URL.search(lit[max(0, m.start() - 24):m.end()])
                  for m in passa.DECIMAL_RUN.finditer(lit))
        check(f"D4b PASS-NEW: v1.2 detects it", hit, lit)
    neg = "the run took 3 seconds and 12345678 rows"
    hit = any(passa.RUN_NEAR.search(neg[max(0, m.start() - 24):m.start()])
              for m in passa.DECIMAL_RUN.finditer(neg))
    check("D4c KNOWN-NEGATIVE: prose with letters between is NOT a run id", not hit)


# ── D5 raw flag earns a verdict ───────────────────────────────────────
def d5():
    print("\nD5 — the raw present-tense flag no longer exists as a verdict input")
    o = old('import classify2 as c; print(json.dumps(c.validity('
            '{"has_run":False,"has_sha":False,"has_date":False,'
            '"present_tense":True})))')
    check("D5a FAIL-OLD: v1.1 emits CURRENT_TREE from the raw flag",
          o["value"] == "CURRENT_TREE", o)
    check("D5b PASS-NEW: v1.2 has no present_tense input at all",
          "present_tense" not in json.dumps(
              cl.validity(row(), None)) and "present_tense" not in
          pathlib.Path(HERE / "classify.py").read_text())
    check("D5c and CURRENT_TREE is unreachable without a witness of that kind",
          cl.validity(row(), None)["value"] == "UNKNOWN")


# ── D6/D7 default emitted as verdict ──────────────────────────────────
def d6_d7():
    print("\nD6/D7 — SCOPE is EARNED or absent; no default, no contradiction")
    d = old('import ontology as o; print(json.dumps([o.STATE_DISPOSITION['
            '("SCOPE","WHOLE_FILE")][1], o.emittable("SCOPE","UNKNOWN"),'
            ' list(o.ALPHABETS["SCOPE"])]))')
    check("D6a FAIL-OLD: v1.1 dispositions WHOLE_FILE as a 'file-level default'",
          d[0] == "file-level default", d[0])
    check("D6b FAIL-OLD: v1.1 SCOPE cannot abstain", not d[1], d[1])
    check("D6c PASS-NEW: v1.2 SCOPE abstains with no binding witness",
          cl.scope(row())["value"] == "UNKNOWN")
    bound = {"COMMIT": [w(applicability_scope="WHOLE_FILE").asdict()]}
    check("D6d BOUNDARY: a binding witness STILL earns WHOLE_FILE",
          cl.scope(row(witnesses=bound))["value"] == "WHOLE_FILE")
    check("D7a a row with only SPAN witnesses cannot emit WHOLE_FILE",
          cl.scope(row(witnesses={"COMMIT": [w().asdict()]}))["value"] == "UNKNOWN")
    check("D6e v1.2 SCOPE=UNKNOWN is emittable", ont.emittable("SCOPE", "UNKNOWN"))


# ── D8 substring, no word boundary ────────────────────────────────────
def d8():
    print("\nD8 — term matching respects word boundaries, plurals excepted")
    for term, word in (("audit", "Auditor"), ("plan", "Plane"), ("plan", "Planning")):
        key = "EVIDENCE" if term == "audit" else "PLAN"
        hit = old(f'import classify2 as c; print(json.dumps(bool(re.search('
                  f'c.FUNCTION_TERMS[{key!r}], {word!r}, re.I))))')
        check(f"D8a FAIL-OLD: v1.1 matches {term!r} inside {word!r}", hit, word)
        check(f"D8b PASS-NEW: v1.2 does NOT match {term!r} in {word!r}",
              not cl.term_match(term, word), word)
    for term, word in (("audit", "Audits"), ("decision", "Decisions"),
                       ("plan", "Plans")):
        check(f"D8c KNOWN-NEGATIVE: a PLURAL still matches ({word!r})",
              bool(cl.term_match(term, word)), word)


# ── D9 degenerate closed-vocabulary negative ──────────────────────────
def d9():
    print("\nD9 — the purpose capture is SUBSTANTIVE, not a trigger phrase")
    for txt in ("This file is ~6,100 lines. Before editing, check which\n",
                "records *why* a commit was good rather than only that\n"):
        g = old(f'import classify2 as c; m=c.PURPOSE.search({txt!r}); '
                f'print(json.dumps(m.group(0) if m else None))')
        check(f"D9a FAIL-OLD: v1.1 captures only the trigger {str(g)[:20]!r}",
              g is not None and len(g) < 14, g)
    good = "This document records the qualification subject and its evidence\n"
    m = cl.PURPOSE.search(good)
    check("D9b PASS-NEW: v1.2 captures the BODY, not the trigger",
          m is not None and len(m.group("body")) >= 12,
          m.group("body") if m else None)
    check("D9c a trigger with no substantive body is NOT a purpose statement",
          cl.PURPOSE.search("This file is\n") is None)


# ── D10 never-executed branch ─────────────────────────────────────────
def d10():
    print("\nD10 — the purpose branch fires, or the fixture fails")
    txt = "# X\n\nThis document records the audit findings for the subject\n"
    r = row(path="x.md", title="")
    out = cl.function(r, txt)
    check("D10a the purpose path is REACHED (observed mentions the term)",
          "EVIDENCE" in str(out.get("observed", "")), out)


# ── D11 candidate order decides ───────────────────────────────────────
def d11():
    print("\nD11 — two corroborated roles are AMBIGUITY, not a list-order win")
    r = row(path="kai-pm/x.md", title="Audit Plan for the Census")
    out = cl.function(r, "# Audit Plan\n")
    check("D11a v1.2 abstains and NAMES both candidates",
          out["value"] == "UNKNOWN" and "EVIDENCE" in str(out) and
          "PLAN" in str(out), out)
    r2 = row(path="kai-pm/x.md", title="Census Audit Report")
    out2 = cl.function(r2, "# Census Audit Report\n")
    check("D11b KNOWN-NEGATIVE: a single role still resolves to one nominal",
          out2["value"] == "UNKNOWN" and "NOMINAL_FUNCTION=EVIDENCE" in
          str(out2.get("observed", "")), out2)


# ── D12/D13/D14 authority extractor ───────────────────────────────────
def d12_d13_d14():
    print("\nD12/D13/D14 — subject binding, polarity, and evidence truncation")
    txt = (SUBJECT / "README.md").read_text(errors="ignore")
    rd = str(SUBJECT / "README.md")
    oc_claim = old(f'import subjectbind2 as s; t=open({rd!r}).read(); '
                   f'c,_=s.bind_claims("README.md",t); '
                   f'print(json.dumps(s.authority_claim(c)))')
    check("D12a FAIL-OLD: committed v1.1 binds README's 'It' to SELF",
          oc_claim == "SELF_ASSERTS_AUTHORITY", oc_claim)
    nc, _ = sb.bind_claims("README.md", txt)
    check("D12b PASS-NEW: v1.2 binds it to the named antecedent",
          sb.authority_claim(nc) == "NO_SELF_CLAIM" and
          any(c["subject"].startswith("OTHER:") for c in nc), nc)
    check("D12c KNOWN-NEGATIVE: a bare pronoun with NO antecedent is SELF",
          sb.bind_subject("It is the source of truth.", 0,
                          "It is the source of truth.", "x.md")[0] == "SELF")

    for s in ("This document is non-authoritative.", "**Status:** non-authoritative"):
        pn = old(f'import subjectbind2 as sb; s={s!r}; print(json.dumps(['
                 f'any(re.search(p,s,re.I) for p in sb.AUTH_POS),'
                 f'any(re.search(p,s,re.I) for p in sb.AUTH_NEG)]))')
        oldpos, oldneg = pn
        check(f"D13a FAIL-OLD: v1.1 scores {s[:28]!r} POSITIVE",
              oldpos and not oldneg, f"pos={oldpos} neg={oldneg}")
        check("D13b PASS-NEW: v1.2 scores it NEGATIVE",
              sb.polarity_of(s) == "NEGATIVE", sb.polarity_of(s))
    check("D13c KNOWN-NEGATIVE: a genuine positive is still POSITIVE",
          sb.polarity_of("**Status:** authoritative") == "POSITIVE")

    dpath = "kai-pm/DECISIONS.md"
    dtxt = (SUBJECT / dpath).read_text(errors="ignore")
    dfull = str(SUBJECT / dpath)
    n_self6 = old(f'import subjectbind2 as s; t=open({dfull!r}).read(); '
                  f'c,_=s.bind_claims({dpath!r},t); '
                  f'print(json.dumps(len([x for x in c[:6] if x[1]=="SELF"])))')
    check("D14a FAIL-OLD: v1.1 truncates to 6 with no SELF row carried",
          n_self6 == 0, n_self6)
    nc2, st2 = sb.bind_claims(dpath, dtxt)
    det = sb.determining_claims(nc2)
    check("D14b PASS-NEW: the determining row IS carried, whatever the total",
          len(det) == 1 and st2["total"] > 6, f"{len(det)} of {st2['total']}")
    check("D14c the envelope refuses an undeclared truncation",
          _raises(lambda: Witness(
              witness_type="x", witness_value="y", source_path="p",
              source_selector="L1", local_context="c",
              applicability_scope="SPAN", evidence_total=43,
              evidence_shown=6, truncated=False)))


# ── D15 artefact cannot adjudicate its own cell ───────────────────────
def d15():
    print("\nD15 — every positive carries a source-bound witness")
    res = json.load(open(HERE / "h2v12-classification.json"))
    missing = []
    for r in res["rows"]:
        for ax in ont.ALPHABETS:
            c = r[ax]
            if c["value"] in ("UNKNOWN", "UNMEASURED"):
                continue
            wt = c.get("witness")
            if not wt or not wt.get("witness_value") or not wt.get(
                    "source_selector") or not wt.get("local_context"):
                missing.append((r["path"], ax))
    check(f"D15a every non-abstention cell carries value+selector+context "
          f"({len(missing)} missing)", not missing, missing[:5])
    old = json.load(open(OLD / "h2v11-classification.json"))
    stat = sum(1 for r in old["rows"] if r["VALIDITY"]["value"] != "UNKNOWN"
               and r["VALIDITY"].get("witness") == "cites a commit sha")
    check(f"D15b FAIL-OLD: v1.1 records a STATIC string on {stat} cells",
          stat > 0, stat)


# ── D16/D17 ontology and meta-check ───────────────────────────────────
def d16_d17():
    print("\nD16/D17 — the ontology invariant, and a gate that can see an omission")
    sc = old('import ontology as o; print(json.dumps(list(o.ALPHABETS["SCOPE"])))')
    check("D16a FAIL-OLD: v1.1 SCOPE alphabet omits UNKNOWN",
          "UNKNOWN" not in sc, sc)
    check("D16b PASS-NEW: v1.2 has UNKNOWN on EVERY axis",
          ont.ontology_invariants() == [], ont.ontology_invariants())
    for axis in ont.ALPHABETS:
        saved = ont.ALPHABETS[axis]
        try:
            ont.ALPHABETS[axis] = tuple(v for v in saved if v != "UNKNOWN")
            f = ont.ontology_invariants()
            check(f"D17a REMOVAL CALIBRATION: removing UNKNOWN from {axis} "
                  f"MUST be detected", len(f) == 1 and f[0][0] == axis, f)
        finally:
            ont.ALPHABETS[axis] = saved
    check("D17b restored: invariants clean again", ont.ontology_invariants() == [])
    rows = [{"path": "x.md", **{a: {"value": "UNKNOWN"} for a in ont.ALPHABETS}}]
    check("D17c a value OUTSIDE the alphabet is detected from OUTPUT",
          ont.values_outside_alphabet(
              [{**rows[0], "SCOPE": {"value": "SMUGGLED"}}]) != [])
    check("D17d KNOWN-NEGATIVE: clean rows produce no finding",
          ont.values_outside_alphabet(rows) == [])


# ── the five AUTHORITY rows Kai adjudicated CORRECT must not move ─────
def regression_five():
    print("\nREGRESSION — the 5 rows Kai adjudicated CORRECT must not move")
    want = {"kai-pm/CODE_AUDIT_MASTER.md": "SELF_ASSERTS_AUTHORITY",
            "kai-pm/DECISIONS.md": "SELF_ASSERTS_AUTHORITY",
            "kai-pm/KAI_ORION_CONTINUITY.md": "SELF_ASSERTS_NON_AUTHORITY",
            "kai-pm/KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md":
                "SELF_ASSERTS_AUTHORITY",
            "kai-pm/PHASE1_READINESS.md": "SELF_ASSERTS_AUTHORITY"}
    for p, v in want.items():
        c, _ = sb.bind_claims(p, (SUBJECT / p).read_text(errors="ignore"))
        check(f"R-{p.split('/')[-1][:26]} still {v}",
              sb.authority_claim(c) == v, sb.authority_claim(c))
    print("\nREGRESSION — EXACT_SNAPSHOT must equal Kai's independent D364 set")
    res = json.load(open(HERE / "h2v12-classification.json"))
    got = {r["path"] for r in res["rows"]
           if r["VALIDITY"]["value"] == "EXACT_SNAPSHOT"}
    kai6 = {"kai-pm/CODE_AUDIT_FINAL_REPORT.md", "kai-pm/CODE_AUDIT_MASTER.md",
            "kai-pm/CODE_AUDIT_PLANNING_PACKAGE_QA.md",
            "kai-pm/UH0_EVIDENCE_MANIFEST.md", "kai-pm/SERVICE_IDENTITY_STATE.md",
            "kai-pm/house_in_order_instrument/AUTHORITY_ONTOLOGY.md"}
    check("R-EXACT_SNAPSHOT == the 6 Kai adjudicated from source",
          got == kai6, f"extra={sorted(got - kai6)} missing={sorted(kai6 - got)}")


def _raises(fn):
    try:
        fn()
        return False
    except (E.PromotionError, AssertionError):
        return True


def main():
    print("HOUSE_H2 v1.2 — HOSTILE FIXTURES")
    print(f"  fail-old imported from COMMITTED {OLD.name}")
    for fn in (d1, d2, d3, d4, d5, d6_d7, d8, d9, d10, d11,
               d12_d13_d14, d15, d16_d17, regression_five):
        fn()
    print(f"\n  PASS {P}   FAIL {F}")
    for f in FAIL:
        print(f"    {f}")
    return 1 if F else 0


if __name__ == "__main__":
    sys.exit(main())
