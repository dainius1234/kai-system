#!/usr/bin/env python3
"""Capability-contract calibration: a shallow history source MUST block
the LIFECYCLE axis rather than yield a plausible number."""
import sys,pathlib,subprocess,tempfile
sys.path.insert(0,str(pathlib.Path(__file__).parent))
import envcontract as ec
P=F=0;FAILS=[]
def ck(n,c,d=""):
    global P,F
    if c:P+=1
    else:F+=1;FAILS.append(f"{n} :: {d}")

FULL=pathlib.Path("/home/user/kai-system")
SUBJ="d8aac4d49e6ba997e3eb38062c0917186ee3f197"
SHALLOW=pathlib.Path("/tmp/claude-0/-home-user-kai-system/84284242-f61e-5a69-9588-732883a5292c/scratchpad/h1")

rows,ok=ec.probe(SHALLOW,FULL,SUBJ)
ck("full-history source satisfies every capability", ok, str(rows))
ck("LIFECYCLE not blocked when history is full",
   ec.axis_blocked("LIFECYCLE",rows)==[], str(ec.axis_blocked("LIFECYCLE",rows)))

# KNOWN-NEGATIVE: use the SHALLOW checkout as the history source
rows2,ok2=ec.probe(SHALLOW,SHALLOW,SUBJ)
ck("shallow history source FAILS the contract", not ok2, str(rows2))
blocked=ec.axis_blocked("LIFECYCLE",rows2)
ck("LIFECYCLE is BLOCKED on a shallow source", blocked!=[], str(blocked))
ck("the blocking reason names the degenerate history source",
   "HISTORY_SOURCE_NON_DEGENERATE" in blocked, str(blocked))
ck("FUNCTION still permitted (needs only the exact tree)",
   ec.axis_blocked("FUNCTION",rows2)==[], str(ec.axis_blocked("FUNCTION",rows2)))
d={r["capability"]:r for r in rows2}
ck("the depth demonstration reports the ERASED value",
   d["HISTORY_SOURCE_NON_DEGENERATE"]["observed_state"]=="README.md = 1",
   d["HISTORY_SOURCE_NON_DEGENERATE"]["observed_state"])
ck("the boundary is DECLARED, never hidden",
   any(r["capability"]=="HISTORY_BOUNDARY" for r in rows), str(rows))
print(f"ENV CAPABILITY CALIBRATION: {P} passed, {F} failed")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
