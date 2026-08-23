#!/usr/bin/env python3
"""P11 — DIRECTIONALITY MUST BE PROVEN. Answers by construction."""
import pathlib, subprocess, sys, tempfile
sys.path.insert(0,'.'); import genlink as gl
P=F=0; FAILS=[]
def ck(n,c,d=""):
    global P,F
    if c: P+=1
    else: F+=1; FAILS.append(f"{n} :: {d}")

CASES = {
 "read_text_only.py":   ('t/a.md','import pathlib\npathlib.Path("t/a.md").read_text()\n','PROVEN_READER'),
 "open_r.py":           ('t/b.md','open("t/b.md","r").read()\n','PROVEN_READER'),
 "write_text.py":       ('t/c.md','import pathlib\npathlib.Path("t/c.md").write_text("x")\n','PROVEN_WRITER'),
 "open_w.py":           ('t/d.md','open("t/d.md","w").write("x")\n','PROVEN_WRITER'),
 "open_a.py":           ('t/e.md','open("t/e.md","a").write("x")\n','PROVEN_WRITER'),
 "both.py":             ('t/f.md','import pathlib\np=pathlib.Path("t/f.md")\np.read_text()\np.write_text("x")\n','READ_AND_WRITE'),
 "mention_near_write.py":('t/g.md','# t/g.md is described here\nopen("other.txt","w").write("x")\n',None),
 "shell_cat.sh":        ('t/h.md','cat t/h.md\n','PROVEN_READER'),
 "shell_redir.sh":      ('t/i.md','echo x > t/i.md\n','PROVEN_WRITER'),
 "shell_append.sh":     ('t/j.md','echo x >> t/j.md\n','PROVEN_WRITER'),
}
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d); (R/"t").mkdir()
    docs=[]
    for fn,(doc,src,_exp) in CASES.items():
        (R/fn).write_text(src)
        (R/doc).write_text("# doc\n"); docs.append(doc)
    # dynamic target -> UNRESOLVED, never forced into a bucket
    (R/"dynamic.py").write_text('import os,pathlib\n'
        'pathlib.Path(os.environ["X"]+".md").write_text("y")\n')
    (R/"dyn2.sh").write_text('echo x > "${OUT}/gen.md"\n')
    docs=sorted(set(docs))
    subprocess.run(["git","init","-q"],cwd=R)
    subprocess.run(["git","add","-A"],cwd=R)
    subprocess.run(["git","-c","user.email=c@x","-c","user.name=c",
                    "commit","-q","-m","f"],cwd=R)
    st,unres = gl.directionality(R,docs)
    for fn,(doc,_src,exp) in CASES.items():
        got = st[doc].get(fn)
        ck(f"P11 {fn} -> {exp}", got==exp, f"got {got!r} full={st[doc]}")
    ck("P11 reader never becomes writer",
       not any(v=="PROVEN_WRITER" for f,v in st['t/a.md'].items()), str(st['t/a.md']))
    ck("P11 dynamic python target -> UNRESOLVED, not a bucket",
       "dynamic.py" in unres, str(unres))
    ck("P11 dynamic shell target -> UNRESOLVED, not a bucket",
       "dyn2.sh" in unres, str(unres))
    ck("P11 unresolved target attributed to NO document",
       not any("dynamic.py" in v or "dyn2.sh" in v for v in st.values()),
       str({k:v for k,v in st.items() if v}))
print(f"P11 CALIBRATION: {P} passed, {F} failed")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
