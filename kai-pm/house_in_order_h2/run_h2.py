#!/usr/bin/env python3
"""H2 RUN — capability contract, then Pass A, then Pass B."""
import json,pathlib,sys,collections
HERE=pathlib.Path(__file__).parent; sys.path.insert(0,str(HERE))
import envcontract as ec, classify as cl
SUBJ="d8aac4d49e6ba997e3eb38062c0917186ee3f197"
TREE_SRC=pathlib.Path(sys.argv[1])           # exact-tree checkout
HIST_SRC=pathlib.Path(sys.argv[2])           # history source
PASSA=json.load(open(sys.argv[3]))

rows_cap,ok=ec.probe(TREE_SRC,HIST_SRC,SUBJ)
print("ENVIRONMENT-SUBJECT CAPABILITY CONTRACT")
for r in rows_cap:
    print(f"  {r['disposition']:<20} {r['capability']:<32} {r['observed_state'][:56]}")
print(f"  contract satisfied: {ok}")
if not ok:
    print("  -> axes requiring missing capabilities will emit UNMEASURED")

fam_ok,fam_why=cl.family_rule_proven(PASSA)
print(f"\nFAMILY RULE (CODE_AUDIT_BATCH_*): proven={fam_ok} — {fam_why}")

out=[]
for r in PASSA:
    text=(TREE_SRC/r["path"]).read_text(errors="ignore")
    out.append({**r, **cl.classify(r,text,rows_cap,fam_ok,fam_why)})
assert len(out)==len(PASSA), "classification population mismatch"
json.dump(out,open(f"{HERE}/h2-classification-v1.json","w"),indent=1)
json.dump(rows_cap,open(f"{HERE}/h2-capability-contract.json","w"),indent=1)

print(f"\nCLASSIFIED {len(out)} == population {len(PASSA)}")
for axis in ("LIFECYCLE","FUNCTION","AUTHORITY","GENERATION","VALIDITY","SCOPE"):
    c=collections.Counter(x[axis]["value"] for x in out)
    print(f"\n  {axis}")
    for k,v in c.most_common(): print(f"    {k:<18} {v:>4}")
print("\n  AUTHORITY_CLAIM (evidence, NOT a state)")
for k,v in collections.Counter(x["authority_claim"] for x in out).most_common():
    print(f"    {k:<28} {v:>4}")
amb=sum(x["ambiguous_subject_claims"] for x in out)
print(f"\n  ambiguous-subject claim sentences abstained: {amb}")
