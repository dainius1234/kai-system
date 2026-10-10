#!/usr/bin/env python3
"""P12 — ANALYSIS COVERAGE MUST BE EXPLICIT. Answers by construction."""
import pathlib, subprocess, sys, tempfile, collections
sys.path.insert(0,'.'); import genlink3 as g3
P=F=0; FAILS=[]
def ck(n,c,d=""):
    global P,F
    if c: P+=1
    else: F+=1; FAILS.append(f"{n} :: {d}")

FILES={
 "docs/a.md":"# a\n", "docs/b.md":"# b\n",
 # 1 resolved read, 1 resolved write, 1 rw, 1 unresolved(dynamic),
 # 1 unresolved(bare basename - NO FALLBACK), 1 non-document target
 "s.py":('import os,pathlib\n'
         'pathlib.Path("docs/a.md").read_text()\n'          # RESOLVED_READ
         'pathlib.Path("docs/b.md").write_text("x")\n'      # RESOLVED_WRITE
         'open("docs/a.md","r+").read()\n'                  # READ_AND_WRITE
         'pathlib.Path(os.environ["Z"]+".md").write_text("y")\n'  # UNRESOLVED
         'pathlib.Path("b.md").write_text("z")\n'           # UNRESOLVED (no fallback)
         'pathlib.Path("notes.txt").write_text("q")\n'                # NON_DOCUMENT_TARGET
         'ROOT = pathlib.Path(".")\n'
         'multi = ROOT / "docs" / "b.md"\n'
         'multi.write_text("m")\n'),   # RESOLVED_WRITE via multi-segment path
}
# Vocabulary updated for Kai's P12 correction: NON_DOCUMENT requires a
# positive witness (.txt extension here); unresolvable .md targets are
# UNRESOLVED_TARGET.
EXPECT=collections.Counter({"RESOLVED_READ":1,"RESOLVED_WRITE":2,
    "READ_AND_WRITE":1,"UNRESOLVED_TARGET":2,
    "RESOLVED_NON_DOCUMENT_TARGET":1})

with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    for rel,body in FILES.items():
        p=R/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(body)
    subprocess.run(["git","init","-q"],cwd=R); subprocess.run(["git","add","-A"],cwd=R)
    subprocess.run(["git","-c","user.email=c@x","-c","user.name=c","commit","-q","-m","f"],cwd=R)
    docs=g3.tracked(R,".md")
    ops=[o for o in g3.collect(R,docs) if o.src=="s.py"]
    n,tally,total=g3.account(ops)

    ck("P12 every operation has exactly one disposition",
       all(o.disposition in g3.DISPOSITIONS for o in ops),
       str([(o.mode,o.disposition) for o in ops]))
    ck("P12 candidates == sum(dispositions)", n==total, f"{n} vs {total}")
    ck("P12 disposition tally matches construction", tally==EXPECT,
       f"got {dict(tally)} want {dict(EXPECT)}")
    ck("P12 NO BASENAME FALLBACK: bare 'b.md' write is UNRESOLVED",
       not any(o.target=="docs/b.md" and o.expr.find("b.md")>=0 and
               o.disposition=="RESOLVED_WRITE" and "docs" not in o.expr
               for o in ops),
       str([(o.expr[:40],o.disposition,o.target) for o in ops]))
    ck("P12 unresolved ops are COUNTED, not dropped",
       tally["UNRESOLVED_TARGET"]==2, str(dict(tally)))
    ck("P12 multi-segment ROOT/'docs'/'b.md' resolves (source order)",
       any(o.target=="docs/b.md" and o.disposition=="RESOLVED_WRITE"
           and "multi" in o.expr for o in ops),
       str([(o.expr[:30],o.disposition,o.target) for o in ops]))
    ck("P12 non-document target needs a POSITIVE witness (.txt)",
       tally["RESOLVED_NON_DOCUMENT_TARGET"]==1, str(dict(tally)))
    ck("P12 source population is git-tracked, not rglob",
       "s.py" in g3.source_population(R), str(g3.source_population(R)))

print(f"P12 CALIBRATION: {P} passed, {F} failed")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
