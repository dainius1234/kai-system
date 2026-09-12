#!/usr/bin/env python3
"""H2 CLASSIFIER — corrected per Kai's D338 acceptance direction.

Consumes frozen HOUSE-IN-ORDER-CENSUS-INSTRUMENT v1.0 output.
Does NOT modify it.

Rulings implemented:
  * AUTHORITY is a QUALIFIED STATE. H2 verifies no claim, so H2 derives
    UNKNOWN. Self-declarations become authority_claim EVIDENCE.
  * FUNCTION requires a POSITIVE WITNESS. A path pattern may NOMINATE a
    candidate; it can never earn the verdict.
  * Every UNKNOWN carries required_evidence / observed_evidence /
    unresolved_reason.
  * An axis whose environment capability is undemonstrated emits
    UNMEASURED / ENVIRONMENT_CAPABILITY_MISSING, never a plausible value.
"""
from __future__ import annotations
import pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import subjectbind as sb, envcontract as ec

PURPOSE = re.compile(r"^\s*>?\s*(?:this (?:document|file|register)|defines|"
                     r"records|holds|tracks|the .{0,40}(?:register|log|plan|"
                     r"report|census|audit|protocol))", re.I)

FUNCTION_TERMS = {
 "GOVERNANCE": r"decision|doctrine|operating rule|protocol|governance|recovery",
 "STATUS":     r"status|progress|dashboard|report card|tracker",
 "PLAN":       r"plan|roadmap|backlog|sequence|next stint|shopping list",
 "EVIDENCE":   r"audit|census|review|post-?mortem|measurement|research|findings|analysis",
 "USER_GUIDE": r"setup|guide|how to|runbook|installation|demo",
 "RUNTIME_INPUT": r"loaded on startup|registry|agents|soul|teammate|skill",
 "TEMPLATE":   r"template",
}
PATH_NOMINATION = [
 (r"DECISIONS|DOCTRINE|CLAUDE\.md|CONTINUITY", "GOVERNANCE"),
 (r"STATUS|PROGRESS|TRACKER|REPORT_CARD",      "STATUS"),
 (r"PLAN|ROADMAP|BACKLOG|SEQUENCE|NEXT_STINT|SHOPPING", "PLAN"),
 (r"AUDIT|CENSUS|REVIEW|POSTMORTEM|MEASUREMENT|RESEARCH", "EVIDENCE"),
 (r"_template",                                 "TEMPLATE"),
 (r"^docs/",                                    "USER_GUIDE"),
 (r"^data/",                                    "RUNTIME_INPUT"),
]

def _unknown(axis, required, observed, why):
    return dict(value="UNKNOWN", required_evidence=required,
                observed_evidence=observed, unresolved_reason=why)
def _val(v, witness_type=None, witness=None):
    return dict(value=v, witness_type=witness_type, witness=witness)

def family_rule_proven(rows):
    """CODE_AUDIT_BATCH_* is a family ONLY if every member is structurally
    homogeneous. Proven, never assumed."""
    fam=[r for r in rows if "CODE_AUDIT_BATCH_" in r["path"]]
    if not fam: return False, "no members"
    ok=all(re.search(r"audit|finding", (r.get("title") or ""), re.I) for r in fam)
    return ok, f"{len(fam)} members; every title names audit/finding: {ok}"

def classify(r, text, cap_rows, fam_ok, fam_why):
    out={}
    title=r.get("title") or ""
    head=text[:6000]

    # ── LIFECYCLE ── gated on demonstrated history capability
    blocked=ec.axis_blocked("LIFECYCLE", cap_rows)
    if blocked:
        out["LIFECYCLE"]=dict(value="UNMEASURED",
            unresolved_reason="ENVIRONMENT_CAPABILITY_MISSING: "+",".join(blocked))
    elif r["superseded_by"]:
        out["LIFECYCLE"]=_val("SUPERSEDED","named_successor",r["superseded_by"])
    elif r["has_sha"] and re.search(r"CODE_AUDIT|AUDIT",r["path"],re.I):
        out["LIFECYCLE"]=_val("HISTORICAL","snapshot_binding","cites an audit snapshot commit")
    elif re.search(r"_20\d\d-\d\d-\d\d|POSTMORTEM_|REALITY_CHECK_",r["path"]):
        out["LIFECYCLE"]=_val("HISTORICAL","dated_artefact",r["path"])
    elif r["commits"]>1:
        out["LIFECYCLE"]=_val("ACTIVE","maintained_since_boundary",f"{r['commits']} commits")
    elif r["present_tense"]:
        out["LIFECYCLE"]=_val("ACTIVE","present_tense_claim","asserts current state")
    else:
        out["LIFECYCLE"]=_unknown("LIFECYCLE",
            "commits>1 since boundary, snapshot binding, or present-tense claim",
            f"commits={r['commits']}, sha={r['has_sha']}, present={r['present_tense']}",
            "no positive lifecycle witness")

    # ── FUNCTION ── path NOMINATES, witness EARNS
    cand=None
    for pat,f in PATH_NOMINATION:
        if re.search(pat,r["path"],re.I): cand=f; break
    if "CODE_AUDIT_BATCH_" in r["path"] and fam_ok:
        out["FUNCTION"]=_val("EVIDENCE","proven_family_rule",fam_why)
    elif r["bytes"]<200 and r["path"].endswith("README.md"):
        out["FUNCTION"]=_val("MARKER","size_and_role",f"{r['bytes']} bytes")
    elif cand:
        term=FUNCTION_TERMS.get(cand,"")
        m=re.search(term,title,re.I) if term else None
        pm=PURPOSE.search(head)
        if m:
            out["FUNCTION"]=_val(cand,"title_states_purpose",title[:70])
        elif pm and term and re.search(term,pm.group(0),re.I):
            out["FUNCTION"]=_val(cand,"purpose_statement",pm.group(0)[:70])
        else:
            out["FUNCTION"]=_unknown("FUNCTION",
                f"title or purpose statement corroborating candidate {cand}",
                f"path nominates {cand}; title={title[:50]!r}",
                "path pattern nominated but no positive witness corroborated it")
    else:
        out["FUNCTION"]=_unknown("FUNCTION",
            "path nomination plus a corroborating title/purpose witness",
            f"title={title[:50]!r}", "no function nomination and no witness")

    # ── AUTHORITY ── H2 never awards a qualified state
    claims,amb=sb.bind_claims(r["path"],text)
    ac=sb.authority_claim(claims)
    out["authority_claim"]=ac
    out["authority_claim_evidence"]=[f"{p}|{s}|{e[:60]}" for p,s,e in claims][:6]
    out["ambiguous_subject_claims"]=amb
    out["AUTHORITY"]=_unknown("AUTHORITY",
        "qualification of the document's claims against programme authority (H3)",
        f"authority_claim={ac}",
        "H2 verifies no claim; authority is earned at H3")

    # ── GENERATION ──
    if r["writers"]:
        out["GENERATION"]=_unknown("GENERATION",
            "write SCOPE (whole-file vs region) established",
            f"PROVEN_WRITE_RELATION {r['writers']}",
            "scope unestablished: cannot distinguish PARTIAL from FULL")
    else:
        out["GENERATION"]=_unknown("GENERATION",
            "a closed search space yielding NO_WRITER, or a proven writer",
            "NO_PROVEN_WRITER over an open search space",
            "MANUAL not admissible while unresolved operations remain")

    # ── VALIDITY ──
    if r["has_run"]: out["VALIDITY"]=_val("RUN_ARTEFACT","run_id","cites a workflow run")
    elif r["has_sha"]: out["VALIDITY"]=_val("EXACT_SNAPSHOT","commit_sha","cites a commit sha")
    elif r["has_date"] and r["present_tense"]:
        out["VALIDITY"]=_val("TIME_BOUND","dated_claim","dated present-tense claim")
    elif r["present_tense"]:
        out["VALIDITY"]=_val("CURRENT_TREE","unbound_claim","claims current state, unbound")
    else:
        out["VALIDITY"]=_unknown("VALIDITY","a run id, commit sha, date, or present-tense claim",
            "none present","no validity binding evidence")

    # ── SCOPE ──
    out["SCOPE"]=(_val("WHOLE_FILE","default_pending_region",
                       "proven writer present: REGION DETERMINATION REQUIRED")
                  if r["writers"] else _val("WHOLE_FILE","default","no region override evidence"))
    return out
