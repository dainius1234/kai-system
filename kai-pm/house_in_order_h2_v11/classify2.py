#!/usr/bin/env python3
"""HOUSE_H2 v1.1 — VERDICTS. Evidence facts are computed elsewhere.

STRUCTURAL SEPARATION IS THE POINT. `lifecycle()` below does not receive
the evidence facts at all. It cannot read `MAINTENANCE_OBSERVED`,
`SELF_ASSERTS_CURRENT` or `CONSUMED_AT_SUBJECT`, so it cannot convert
one into a verdict even by accident. That is what D360 §5 requires, and
a comment saying "do not use these" would not have prevented v1.0's
defect -- v1.0's own docstring already said authority was earned at H3
while its LIFECYCLE branch awarded ACTIVE from a self-claim.

REPAIRS IMPLEMENTED HERE
  * ACTIVE is NOT EARNABLE. Declared in ontology.py, enforced by the
    absence of any branch that emits it, and proven by qualification.
  * REFERENCE and OTHER implemented -- D341's class of defect was a
    declared value no code path could emit.
  * Nomination gaps repaired GENERICALLY. PLAYBOOKS/, TECH_WATCH and
    NAVIGATION nominated nothing, so the correct answer was unreachable.
    Path still only NOMINATES; a title or purpose witness still EARNS.
  * Multiple candidates are allowed and the witness selects among them.
    v1.0 took first-match-wins, so `docs/` nominated USER_GUIDE and
    stopped, and a specification under docs/ could never reach any other
    function however clearly its title said otherwise.
  * OTHER requires a purpose statement AND no match across the declared
    function vocabulary. The negative half is admissible only because
    that vocabulary is CLOSED BY DECLARATION -- an open-world "none of
    the above" would be the dismissed-on-absence defect P13 forbids.
"""
from __future__ import annotations
import re

import ontology as ont
import subjectbind2 as sb

# re.M added deliberately. v1.0 compiled this with `^` and NO multiline
# flag, so the anchor matched only at the very first character of the
# file. A purpose statement under a heading -- which is how essentially
# every document in this repository is written -- was invisible to the
# witness that exists to find it. Scope narrower than the name implied
# (R5), and it silently suppressed the corroborating evidence that lets
# a nominated candidate be earned.
PURPOSE = re.compile(r"^\s*>?\s*(?:this (?:document|file|register)|defines|"
                     r"records|holds|tracks|the .{0,40}(?:register|log|plan|"
                     r"report|census|audit|protocol|index|map))",
                     re.I | re.M)

FUNCTION_TERMS = {
    "GOVERNANCE": r"decision|doctrine|operating rule|protocol|governance|recovery|charter",
    "STATUS": r"status|progress|dashboard|report card|tracker|watch",
    "PLAN": r"plan|roadmap|backlog|sequence|next stint|shopping list|proposal",
    "EVIDENCE": r"audit|census|review|post-?mortem|measurement|research|findings|analysis|report",
    "REFERENCE": r"reference|index|navigation|map\b|catalogue|glossary|specification|spec\b|architecture|design",
    "USER_GUIDE": r"setup|guide|how to|runbook|installation|demo|playbook|checklist",
    "RUNTIME_INPUT": r"loaded on startup|registry|agents|soul|teammate|skill|prompt",
    "TEMPLATE": r"template",
}
# A path may nominate SEVERAL candidates; the witness decides. Order is
# not precedence.
PATH_NOMINATION = [
    (r"DECISIONS|DOCTRINE|CLAUDE\.md|CONTINUITY|PLAYBOOK", "GOVERNANCE"),
    (r"STATUS|PROGRESS|TRACKER|REPORT_CARD|TECH_WATCH|WATCH", "STATUS"),
    (r"PLAN|ROADMAP|BACKLOG|SEQUENCE|NEXT_STINT|SHOPPING|PROPOSAL", "PLAN"),
    (r"AUDIT|CENSUS|REVIEW|POSTMORTEM|MEASUREMENT|RESEARCH|EVIDENCE", "EVIDENCE"),
    (r"NAVIGATION|INDEX|_MAP|GLOSSARY|SPEC|ARCHITECTURE|DESIGN", "REFERENCE"),
    (r"_template", "TEMPLATE"),
    (r"^docs/|PLAYBOOKS/|RUNBOOK|GUIDE|SETUP|CHECKLIST", "USER_GUIDE"),
    (r"^data/", "RUNTIME_INPUT"),
]


def _unknown(axis, required, observed, why):
    return dict(value="UNKNOWN", required_evidence=required,
                observed_evidence=observed, unresolved_reason=why)


def _val(v, witness_type=None, witness=None):
    return dict(value=v, witness_type=witness_type, witness=witness)


def lifecycle(row, blocked):
    """DELIBERATELY receives no evidence facts. ACTIVE is unreachable."""
    if blocked:
        return dict(value=ont.CAPABILITY_FAILURE,
                    unresolved_reason="ENVIRONMENT_CAPABILITY_MISSING: "
                                      + ",".join(blocked))
    if row["superseded_by"]:
        return _val("SUPERSEDED", "named_successor", row["superseded_by"])
    if row["has_sha"] and re.search(r"CODE_AUDIT|AUDIT", row["path"], re.I):
        return _val("HISTORICAL", "snapshot_binding",
                    "cites an audit snapshot commit")
    if re.search(r"_20\d\d-\d\d-\d\d|POSTMORTEM_|REALITY_CHECK_", row["path"]):
        return _val("HISTORICAL", "dated_artefact", row["path"])
    return _unknown(
        "LIFECYCLE",
        "a named successor, or a bound snapshot/dated-artefact witness",
        f"superseded_by={row['superseded_by']}, has_sha={row['has_sha']}",
        "no positively earnable lifecycle witness. ACTIVE is "
        "H2_NOT_EARNABLE per D360 §5: maintenance, self-assertion and "
        "consumption are evidence facts, not verdicts.")


def function(row, text, fam_ok, fam_why):
    title = row.get("title") or ""
    head = text[:6000]
    if "CODE_AUDIT_BATCH_" in row["path"] and fam_ok:
        return _val("EVIDENCE", "proven_family_rule", fam_why)
    if row["bytes"] < 200 and row["path"].endswith("README.md"):
        return _val("MARKER", "size_and_role", f"{row['bytes']} bytes")

    cands = [f for pat, f in PATH_NOMINATION
             if re.search(pat, row["path"], re.I)]
    pm = PURPOSE.search(head)
    for c in cands:                       # witness selects among candidates
        term = FUNCTION_TERMS.get(c, "")
        if term and re.search(term, title, re.I):
            return _val(c, "title_states_purpose", title[:70])
    for c in cands:
        term = FUNCTION_TERMS.get(c, "")
        if pm and term and re.search(term, pm.group(0), re.I):
            return _val(c, "purpose_statement", pm.group(0)[:70])

    # OTHER: a stated purpose that matches NO declared function term.
    # Admissible only because the vocabulary is closed by declaration.
    if pm and not any(re.search(t, title + " " + pm.group(0), re.I)
                      for t in FUNCTION_TERMS.values()):
        return _val("OTHER", "purpose_outside_declared_vocabulary",
                    pm.group(0)[:70])

    if cands:
        return _unknown(
            "FUNCTION",
            f"a title or purpose witness corroborating one of {cands}",
            f"path nominates {cands}; title={title[:50]!r}",
            "path nominated but no positive witness corroborated it")
    return _unknown("FUNCTION",
                    "path nomination plus a corroborating title/purpose witness",
                    f"title={title[:50]!r}",
                    "no function nomination and no witness")


def validity(row):
    if row["has_run"]:
        return _val("RUN_ARTEFACT", "run_id", "cites a workflow run")
    if row["has_sha"]:
        return _val("EXACT_SNAPSHOT", "commit_sha", "cites a commit sha")
    if row["has_date"] and row["present_tense"]:
        return _val("TIME_BOUND", "dated_claim", "dated present-tense claim")
    if row["present_tense"]:
        return _val("CURRENT_TREE", "unbound_claim",
                    "claims current state, unbound")
    # v1.0 emitted "none present" here while has_date could be true. An
    # observed fact must never be erased because it was insufficient.
    seen = [k for k in ("has_run", "has_sha", "has_date", "present_tense")
            if row[k]]
    return _unknown("VALIDITY",
                    "a run id, commit sha, or a date WITH a present-tense claim",
                    f"observed: {seen or 'none of run/sha/date/present-tense'}",
                    "no sufficient validity binding")


def family_rule_proven(rows):
    fam = [r for r in rows if "CODE_AUDIT_BATCH_" in r["path"]]
    if not fam:
        return False, "no members"
    ok = all(re.search(r"audit|finding", (r.get("title") or ""), re.I)
             for r in fam)
    return ok, f"{len(fam)} members; every title names audit/finding: {ok}"


def classify(row, text, blocked_by_axis, fam_ok, fam_why):
    out = {}
    out["LIFECYCLE"] = lifecycle(row, blocked_by_axis.get("LIFECYCLE"))
    out["FUNCTION"] = function(row, text, fam_ok, fam_why)

    claims, amb = sb.bind_claims(row["path"], text)
    out["authority_claim"] = sb.authority_claim(claims)
    out["authority_claim_evidence"] = [f"{p}|{s}|{e[:60]}"
                                       for p, s, e in claims][:6]
    out["ambiguous_subject_claims"] = amb
    out["AUTHORITY"] = _unknown(
        "AUTHORITY",
        "qualification of the document's claims against programme authority",
        f"authority_claim={out['authority_claim']}",
        "H2 verifies no claim; AUTHORITY states are DEFERRED_TO_H3")

    out["GENERATION"] = _unknown(
        "GENERATION",
        "a closed search space yielding NO_WRITER, or a writer with "
        "established scope",
        f"writers={row['writers'] or 'none proven'}",
        "MANUAL/PARTIAL_DERIVED/FULL_DERIVED are all H2_NOT_EARNABLE: "
        "the write search space is open and region scope is not "
        "determined at H2")

    out["VALIDITY"] = validity(row)
    out["SCOPE"] = _val("WHOLE_FILE",
                        "default_pending_region" if row["writers"] else "default",
                        "proven writer present: REGION DETERMINATION REQUIRED"
                        if row["writers"] else "no region override evidence")

    # self-check: no verdict may be a value the contract forbids at H2
    for axis in ont.ALPHABETS:
        v = out[axis]["value"]
        if v == ont.CAPABILITY_FAILURE:
            continue
        if not ont.emittable(axis, v):
            raise AssertionError(
                f"CONTRACT VIOLATION: emitted {axis}={v}, disposition "
                f"{ont.STATE_DISPOSITION.get((axis, v))}")
    return out
