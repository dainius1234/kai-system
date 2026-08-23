#!/usr/bin/env python3
"""CLAIM SUBJECT-BINDING CALIBRATION — Kai's ten cases + boundary pairs."""
import sys, pathlib
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import subjectbind as sb
P=F=0; FAILS=[]
def ck(n,c,d=""):
    global P,F
    if c: P+=1
    else: F+=1; FAILS.append(f"{n} :: {d}")

def claim(path, text, fm=None):
    cl,amb = sb.bind_claims(path, text, fm)
    return sb.authority_claim(cl), cl, amb

# 1 SELF asserts authority
k,_,_ = claim("a.md","This document is the single source of truth.\n")
ck("1 SELF asserts authority", k=="SELF_ASSERTS_AUTHORITY", k)
# 2 SELF asserts non-authority
k,_,_ = claim("a.md","This document is not a source of programme truth.\n")
ck("2 SELF asserts non-authority", k=="SELF_ASSERTS_NON_AUTHORITY", k)
# 3 OTHER asserts authority
k,cl,_ = claim("a.md","See `b.md` which is the single source of truth.\n")
ck("3 OTHER-named authority is NOT a self claim", k=="NO_SELF_CLAIM", k)
ck("3b and the subject is recorded as OTHER",
   any(s=="OTHER" for _,s,_ in cl), str(cl))
# 4 OTHER asserts non-authority  <-- THE CODE_AUDIT_MASTER DEFECT
k,cl,_ = claim("a.md","Historical registers in `old.md` are not authoritative.\n")
ck("4 OTHER-named non-authority is NOT a self claim", k=="NO_SELF_CLAIM", k)
# 5 explicit named path subject == itself
k,_,_ = claim("a.md","`a.md` is authoritative for this programme.\n")
ck("5 naming ITSELF binds to SELF", k=="SELF_ASSERTS_AUTHORITY", k)
# 6 unresolved pronoun -> abstain
k,cl,amb = claim("a.md","They are not authoritative any more.\n")
ck("6 unresolved pronoun -> AMBIGUOUS, no self claim",
   k=="NO_SELF_CLAIM" and amb==1, f"{k} amb={amb}")
# 7 quote containing authority language
k,cl,_ = claim("a.md","> This document is the single source of truth.\n")
ck("7 a quotation is not a declaration", k=="NO_SELF_CLAIM", k)
ck("7b recorded as QUOTED_NOT_DECLARATION",
   any(s=="QUOTED_NOT_DECLARATION" for _,s,_ in cl), str(cl))
# 8 two conflicting SELF claims
k,_,_ = claim("a.md","This document is authoritative.\nThis file is not authoritative.\n")
ck("8 conflicting self claims -> CONFLICTING_SELF_CLAIMS",
   k=="CONFLICTING_SELF_CLAIMS", k)
# 9 explicit snapshot binding does not create authority
k,_,_ = claim("a.md","Audited snapshot 2d830f25d569baa5ce955dd8d17e8f0744239876.\n")
ck("9 a snapshot binding alone is no authority claim", k=="NO_SELF_CLAIM", k)
# 10 ambiguous negation/scope
k,_,amb = claim("a.md","Not everything here is authoritative.\n")
ck("10 ambiguous negation abstains rather than deciding",
   k in ("NO_SELF_CLAIM","SELF_ASSERTS_NON_AUTHORITY"), k)
# structured metadata is SELF by construction
k,_,_ = claim("a.md","body\n",{"authority":"not authoritative"})
ck("11 front-matter field binds to SELF", k=="SELF_ASSERTS_NON_AUTHORITY", k)

# ── BOUNDARY PAIRS: self/other must not collapse ────────────────────
k_self,_,_ = claim("m.md","`m.md` is the single source of truth.\n")
k_other,_,_ = claim("m.md","Historical registers are chronology only, see `old.md`.\n")
ck("PAIR-A self-named claim IS a self claim", k_self=="SELF_ASSERTS_AUTHORITY", k_self)
ck("PAIR-B other-named claim is NOT a self claim", k_other=="NO_SELF_CLAIM", k_other)

# THE REAL CASE, reduced: both statements in one document
real = ("# Master\nStatus: **FINAL — SINGLE SOURCE OF TRUTH**\n"
        "Historical working registers are retained for chronology only. "
        "They are not authoritative.\n")
k,cl,amb = claim("kai-pm/CODE_AUDIT_MASTER.md", real)
ck("REAL the master's own SINGLE SOURCE OF TRUTH is a SELF claim",
   k in ("SELF_ASSERTS_AUTHORITY","CONFLICTING_SELF_CLAIMS"), f"{k} {cl}")
ck("REAL 'not authoritative' about others must NOT flip it to non-authority",
   k!="SELF_ASSERTS_NON_AUTHORITY", f"{k} {cl}")

# 12 'final' is a STAGE word, not authority vocabulary
k,_,_ = claim("a.md","Status: CONFIRMED - pending final consolidation\n")
ck("12 'pending final consolidation' is NOT an authority claim",
   k=="NO_SELF_CLAIM", k)
k,_,_ = claim("a.md","It is not the final repository total.\n")
ck("12b 'not the final total' is NOT an authority claim",
   k!="SELF_ASSERTS_AUTHORITY", k)
# 13 blockquote CALLOUT carrying a controlled field IS a declaration
k,_,_ = claim("a.md","> **STATUS: RECOVERY PROTOCOL - NOT A SOURCE OF PROGRAMME TRUTH**\n")
ck("13 quoted controlled-field banner IS a self declaration",
   k=="SELF_ASSERTS_NON_AUTHORITY", k)
# 13b BOUNDARY: quoted FREE PROSE remains a quotation
k,_,_ = claim("a.md","> This document is the single source of truth.\n")
ck("13b quoted free prose remains a quotation, not a declaration",
   k=="NO_SELF_CLAIM", k)

print(f"SUBJECT-BINDING CALIBRATION: {P} passed, {F} failed")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
