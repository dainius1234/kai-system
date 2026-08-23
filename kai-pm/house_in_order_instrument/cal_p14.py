#!/usr/bin/env python3
"""P14 — CLAIM-SCOPED NEGATIVES REQUIRE CONSTRUCTIVE EXCLUSION."""
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

# CASE 1 — every write constructively excluded -> NO_WRITER admissible
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n","README.md":"# r\n","cfg.json":"{}\n",
          "s.py":('import pathlib\n'
                  'pathlib.Path("README.md").write_text("a")\n'      # other exact path
                  'pathlib.Path("cfg.json").write_text("b")\n')})     # non-md extension
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    kind,who,buckets,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 all writes constructively excluded -> NO_WRITER",
       kind=="NO_WRITER", f"{kind} {buckets} {wit}")
    ck("P14 every exclusion carries a positive witness",
       sum(wit.values())==buckets["EXCLUDED_FROM_T"], f"{wit} {buckets}")
    ck("P14 population reconciles",
       sum(buckets.values())==buckets["REACHES_T"]+buckets["EXCLUDED_FROM_T"]+buckets["COULD_REACH_T"])

# CASE 2 — one dynamic write cannot be excluded -> NO_PROVEN_WRITER
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n","README.md":"# r\n",
          "s.py":('import os,pathlib\n'
                  'pathlib.Path("README.md").write_text("a")\n'
                  'pathlib.Path(os.environ["P"]).write_text("b")\n')})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    kind,who,buckets,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 a dynamic write blocks NO_WRITER",
       kind=="NO_PROVEN_WRITER", f"{kind} {buckets}")
    ck("P14 the residual COULD_REACH count is reported",
       buckets["COULD_REACH_T"]==1, str(buckets))

# CASE 3 — a real writer of T is REACHES, never excluded
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n",
          "w.py":'import pathlib\npathlib.Path("data/SOUL.md").write_text("x")\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    kind,who,buckets,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 real writer -> PROVEN_WRITE_RELATION",
       kind=="PROVEN_WRITE_RELATION" and who==["w.py"], f"{kind} {who}")

# CASE 4 — MISBINDING guard: tmp fixture with the SAME basename must NOT
# be excluded by basename, and must NOT be counted as reaching the repo file
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n",
          "t.py":('import pathlib\n'
                  'pathlib.Path(str(tmp)+"/SOUL.md").write_text("x")\n')})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    kind,who,buckets,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 same-basename tmp write is NOT a proven writer of the repo file",
       kind!="PROVEN_WRITE_RELATION", f"{kind} {buckets}")
    ck("P14 and it is NOT silently excluded either",
       buckets["COULD_REACH_T"]>=1, str(buckets))

# CASE 5 — a GENUINELY DYNAMIC target (shell ${VAR}) must be COULD_REACH.
# Added because the earlier fixtures never reached the `not _fixed`
# branch: os.environ["P"] yields the DICT KEY "P" as a fragment, so the
# path looked fixed. A mutation planted in that branch passed 8/8 --
# the calibration was vacuous there.
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n",
          "w.sh":'echo x > "${OUT}/SOUL.md"\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    writes=[o for o in ops if o.mode in ("W","RW")]
    ck("P14 CASE5 the dynamic write is collected at all", len(writes)>=1,
       str([(o.src,o.mode,o._frags) for o in ops]))
    ck("P14 CASE5 it is NOT _fixed (reaches the dynamic branch)",
       any(not g3._fixed(o._frags) for o in writes),
       str([o._frags for o in writes]))
    kind,who,buckets,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 CASE5 dynamic target -> COULD_REACH, never excluded",
       buckets["COULD_REACH_T"]>=1, f"{kind} {buckets} {wit}")
    ck("P14 CASE5 therefore NO_WRITER is NOT admissible",
       kind=="NO_PROVEN_WRITER", f"{kind} {buckets}")

# CASE 6 — FULLY dynamic target, no basename at all. This is the case
# that discriminates constructive exclusion from absence-based exclusion:
# the target token is ABSENT from the expression, yet the operation could
# still reach the target at runtime. CASE 5 did not discriminate, because
# its path happened to CONTAIN "SOUL.md".
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n", "w.sh":'echo x > "$DEST"\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    writes=[o for o in ops if o.mode in ("W","RW")]
    ck("P14 CASE6 the fully dynamic write is collected", len(writes)>=1,
       str([(o.src,o.mode,o._frags) for o in ops]))
    kind,who,buckets,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 CASE6 target token ABSENT must NOT cause exclusion",
       buckets["EXCLUDED_FROM_T"]==0, f"{buckets} {wit}")
    ck("P14 CASE6 it is COULD_REACH", buckets["COULD_REACH_T"]>=1, str(buckets))
    ck("P14 CASE6 so NO_WRITER is inadmissible", kind=="NO_PROVEN_WRITER",
       f"{kind} {buckets}")

# CASE 7 — H0.1 REGRESSION: a dict key must NOT be a path fragment.
# This is the fixture whose ABSENCE let a vacuous mutation pass twice.
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n",
          "s.py":('import os,pathlib\n'
                  'pathlib.Path(os.environ["P"]).write_text("y")\n')})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    w=[o for o in ops if o.mode in ("W","RW")]
    ck("P14 CASE7 dict key 'P' is NOT extracted as a path fragment",
       all("P" not in (o._frags or []) for o in w), str([o._frags for o in w]))
    ck("P14 CASE7 the expression is therefore NOT _fixed",
       all(not g3._fixed(o._frags) for o in w), str([o._frags for o in w]))
    kind,who,b,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ck("P14 CASE7 env-var write is COULD_REACH, not excluded",
       b["COULD_REACH_T"]>=1 and b["EXCLUDED_FROM_T"]==0, f"{b} {wit}")

# CASE 8 — H0.2 PATH DOMAINS: /dev/null excluded as a SYSTEM path,
# never as a repository directory named 'dev'.
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n", "w.sh":'echo x > /dev/null\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs); tr=set(g3.tracked(R))
    kind,who,b,wit=g3.writer_claim_scoped(ops,"data/SOUL.md",tr)
    ws=" ".join(wit)
    ck("P14 CASE8 /dev/null excluded with a SYSTEM-PATH witness",
       "absolute system path" in ws, str(wit))
    ck("P14 CASE8 and NOT described as repository directory 'dev'",
       "directory 'dev'" not in ws, str(wit))

# CASE 9 — H0.2 shell regex: '>>' must not capture '>' as a target.
with tempfile.TemporaryDirectory() as d:
    R=pathlib.Path(d)
    mk(R,{"data/SOUL.md":"# soul\n", "w.sh":'echo x >> notes.txt\n'})
    docs=g3.tracked(R,".md"); ops=g3.collect(R,docs)
    frags=[f for o in ops for f in (o._frags or [])]
    ck("P14 CASE9 '>' is never captured as a path fragment",
       ">" not in frags, str(frags))

print(f"P14 CALIBRATION: {P} passed, {F} failed")
for f in FAILS: print("  FAIL",f)
sys.exit(1 if F else 0)
