#!/usr/bin/env python3
"""BOUNDARY-PRESERVING REPAIR (Kai, D333 review).

Every classifier repair must prove BOTH:
  (a) the corrected known-bad case, AND
  (b) a same-family counterexample on the OPPOSITE side of the boundary.

Earned by H0.2: the fix for "/dev/null read as repository dev/" broke
`str(tmp) + "/SOUL.md"`, which merely LOOKED absolute. CASE 4 caught it
by luck. These pairs make it deliberate.

Each test below is a PAIR. A repair that collapses the boundary fails
the pair even if it satisfies its own side.
"""
import pathlib, subprocess, sys, tempfile
sys.path.insert(0,'.'); import genlink3 as g3
P=F=0; FAILS=[]
def ck(n,c,d=""):
    global P,F
    if c: P+=1
    else: F+=1; FAILS.append(f"{n} :: {d}")
def mk(root,files):
    for rel,body in files.items():
        p=root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(body)
    subprocess.run(["git","init","-q"],cwd=root); subprocess.run(["git","add","-A"],cwd=root)
    subprocess.run(["git","-c","user.email=c@x","-c","user.name=c","commit","-q","-m","f"],cwd=root)
def run(files, target="data/SOUL.md"):
    with tempfile.TemporaryDirectory() as d:
        R=pathlib.Path(d); mk(R,files)
        docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
        return ops, g3.writer_claim_scoped(ops,target,tr), tr

BASE={"data/SOUL.md":"# soul\n"}

# ── PAIR 1 — absolute system path vs concatenation suffix ────────────
ops,(k,w,b,wit),_=run({**BASE,"a.sh":'echo x > /dev/null\n'})
ck("PAIR1a /dev/null -> excluded as ABSOLUTE_SYSTEM",
   "absolute system path" in " ".join(wit), str(wit))
ops,(k,w,b,wit),_=run({**BASE,
   "b.py":'import pathlib\npathlib.Path(str(tmp)+"/SOUL.md").write_text("x")\n'})
ck("PAIR1b dynamic prefix + '/SOUL.md' is NOT absolute -> COULD_REACH",
   b["COULD_REACH_T"]>=1 and b["EXCLUDED_FROM_T"]==0, f"{b} {wit}")

# ── PAIR 2 — dict key is not a path vs a real literal path IS ────────
ops,(k,w,b,wit),_=run({**BASE,
   "c.py":'import os,pathlib\npathlib.Path(os.environ["SOUL"]).write_text("x")\n'})
ck("PAIR2a subscript key 'SOUL' not treated as a path -> COULD_REACH",
   b["COULD_REACH_T"]>=1 and b["EXCLUDED_FROM_T"]==0, f"{b} {wit}")
ops,(k,w,b,wit),_=run({**BASE,
   "d.py":'import pathlib\npathlib.Path("data/SOUL.md").write_text("x")\n'})
ck("PAIR2b a real literal path IS still resolved -> PROVEN_WRITE_RELATION",
   k=="PROVEN_WRITE_RELATION" and w==["d.py"], f"{k} {w}")

# ── PAIR 3 — '>' is not a target vs a real redirect target IS ────────
ops,_,_=run({**BASE,"e.sh":'echo x >> notes.txt\n'})
frags=[f for o in ops for f in (o._frags or [])]
ck("PAIR3a '>' never captured as a path fragment", ">" not in frags, str(frags))
ck("PAIR3b the real redirect target IS captured", "notes.txt" in frags, str(frags))

# ── PAIR 4 — multi-segment resolves vs unrelated segments do not ─────
ops,(k,w,b,wit),_=run({**BASE,
   "f.py":'import pathlib\nROOT=pathlib.Path(".")\np=ROOT/"data"/"SOUL.md"\np.write_text("x")\n'})
ck("PAIR4a ROOT/'data'/'SOUL.md' resolves in source order",
   k=="PROVEN_WRITE_RELATION", f"{k} {b}")
ops,(k,w,b,wit),_=run({**BASE,
   "g.py":'import pathlib\nROOT=pathlib.Path(".")\np=ROOT/"other"/"NOTES.md"\np.write_text("x")\n'})
ck("PAIR4b an unrelated multi-segment path does NOT reach the target",
   b["REACHES_T"]==0, f"{k} {b}")

# ── PAIR 5 — non-document witness vs extensionless literal ───────────
ops,(k,w,b,wit),_=run({**BASE,
   "h.py":'import pathlib\npathlib.Path("cfg.json").write_text("{}")\n'})
ck("PAIR5a '.json' excluded on a POSITIVE extension witness",
   any("extension" in x for x in wit), str(wit))
ops,(k,w,b,wit),_=run({**BASE,
   "i.py":'import pathlib\npathlib.Path("somefile").write_text("x")\n'})
ck("PAIR5b an extensionless literal is NOT excluded -> COULD_REACH",
   b["COULD_REACH_T"]>=1, f"{b} {wit}")

print(f"BOUNDARY-PRESERVATION: {P} passed, {F} failed  ({P//2} pairs)")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
