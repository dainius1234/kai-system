#!/usr/bin/env python3
"""P13 — NEGATIVE CLAIMS REQUIRE A CLOSED SEARCH SPACE.
Plus the corrected P12 relevance model. Answers by construction."""
import pathlib, subprocess, sys, tempfile, collections
sys.path.insert(0,'.'); import genlink3 as g3
P=F=0; FAILS=[]
def ck(n,c,d=""):
    global P,F
    if c: P+=1
    else: F+=1; FAILS.append(f"{n} :: {d}")

def mk(root, files):
    for rel,body in files.items():
        p=root/rel; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(body)
    subprocess.run(["git","init","-q"],cwd=root); subprocess.run(["git","add","-A"],cwd=root)
    subprocess.run(["git","-c","user.email=c@x","-c","user.name=c","commit","-q","-m","f"],cwd=root)

# ── CLOSED search space: every op resolved, so NO_WRITER is admissible
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"docs/a.md":"# a\n","cfg.json":"{}\n",
          "s.py":('import pathlib\n'
                  'pathlib.Path("docs/a.md").read_text()\n'      # RESOLVED_READ
                  'pathlib.Path("cfg.json").write_text("{}")\n')}) # RESOLVED_NON_DOC
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs)
    tally=collections.Counter(o.disposition for o in ops)
    ck("P12 exact tracked non-.md target -> RESOLVED_NON_DOCUMENT_TARGET",
       tally.get("RESOLVED_NON_DOCUMENT_TARGET",0)==1, str(dict(tally)))
    ck("P13 CLOSED search space -> NO_WRITER admissible",
       g3.writer_claim(ops,"docs/a.md")[0]=="NO_WRITER",
       str(g3.writer_claim(ops,"docs/a.md")))

# ── OPEN search space: one dynamic op, so NO_WRITER is NOT admissible
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"docs/a.md":"# a\n",
          "s.py":('import os,pathlib\n'
                  'pathlib.Path("docs/a.md").read_text()\n'
                  'pathlib.Path(os.environ["X"]).write_text("y")\n')})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs)
    tally=collections.Counter(o.disposition for o in ops)
    kind,detail=g3.writer_claim(ops,"docs/a.md")
    ck("P12 dynamic target -> UNRESOLVED_RELEVANCE, not NON_DOCUMENT",
       tally.get("UNRESOLVED_RELEVANCE",0)==1, str(dict(tally)))
    ck("P13 OPEN search space -> NO_PROVEN_WRITER, never NO_WRITER",
       kind=="NO_PROVEN_WRITER", f"{kind} {detail}")
    ck("P13 the open claim REPORTS how open it is",
       detail.get("unresolved_operations",0)==1, str(detail))

# ── a real writer must be reported as a relation, not a negative
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"docs/a.md":"# a\n",
          "w.py":'import pathlib\npathlib.Path("docs/a.md").write_text("x")\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs)
    kind,who=g3.writer_claim(ops,"docs/a.md")
    ck("P13 a proven writer yields PROVEN_WRITE_RELATION",
       kind=="PROVEN_WRITE_RELATION" and who==["w.py"], f"{kind} {who}")

# ── literals present but none .md and no non-doc witness -> UNRESOLVED_RELEVANCE
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"docs/a.md":"# a\n",
          "s.py":'import pathlib\npathlib.Path("somefile").write_text("x")\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs)
    tally=collections.Counter(o.disposition for o in ops)
    ck("P12 extensionless literal is UNRESOLVED_RELEVANCE, not NON_DOCUMENT",
       tally.get("UNRESOLVED_RELEVANCE",0)==1, str(dict(tally)))

print(f"P13/P12 CALIBRATION: {P} passed, {F} failed")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
