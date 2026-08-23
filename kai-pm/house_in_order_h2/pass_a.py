#!/usr/bin/env python3
"""H2 PASS A — evidence packet for all 272. NO ROLES ASSIGNED.

Consumes frozen instrument v1.0. NOT part of it; not an amendment to it.
"""
import pathlib,sys,re,json,hashlib,collections
sys.path.insert(0,'.')
import doccensus2 as dc, genlink3 as g3
R=pathlib.Path("../..")
# The H1 subject checkout is DEPTH-1, so `rev-list --count` returns 1 for
# EVERY document -- maintenance history erased by the measurement
# environment. Commit history is therefore read from a full-history repo
# AT THE EXACT SUBJECT COMMIT, which is still bound to the subject.
FULL=pathlib.Path("/home/user/kai-system")
SUBJ="d8aac4d49e6ba997e3eb38062c0917186ee3f197"
tracked=dc.tracked_md(R)
edges=dc.build_graph(R,tracked); inc=dc.incoming(edges)
out_deg=collections.Counter(s for s,d,k,_,_ in edges if d)
ops=g3.collect(R,tracked)
exe=collections.Counter(); writers=collections.defaultdict(set); readers=collections.defaultdict(set)
for o in ops:
    if o.target and o.disposition in ("RESOLVED_READ","RESOLVED_WRITE","READ_AND_WRITE"):
        exe[o.target]+=1
        (writers if o.disposition!="RESOLVED_READ" else readers)[o.target].add(o.src)

SHA=re.compile(r"\b[0-9a-f]{7,40}\b")
RUN=re.compile(r"\brun[_ ]?(?:id)?[ :#]*\d{8,}\b|actions/runs/\d+",re.I)
DATE=re.compile(r"\b20\d\d-\d\d-\d\d\b|\b\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|November|December) 20\d\d\b")
# A successor must be a PATH-LIKE token. "superseded by the 4,580-finding
# reconciliation" captured "the" -- an article recorded as a named
# successor, and SUPERSEDED assigned on it. MISBINDING.
SUPBY=re.compile(r"superseded by\s+`?([A-Za-z0-9_./-]+\.md)`?",re.I)
SUPES=re.compile(r"\bsupersedes\b",re.I)
DECL=re.compile(r"SINGLE SOURCE OF TRUTH|NOT AUTHORITATIVE|not authoritative|AUTHORITATIVE|SUPERSEDED|superseded|HISTORICAL|historical|chronology only|DEPRECATED|STALE|stale|ACTIVE|FINAL|CURRENT|LATEST|Loaded on startup|not a source of programme truth|not programme authority",re.I)
PRESENT=re.compile(r"\bcurrent phase\b|\bcurrent focus\b|\bcurrently\b|\blast updated\b|\bnext\b:",re.I)

rows=[]
for d in tracked:
    p=R/d; txt=p.read_text(errors="ignore"); head=txt[:6000]
    title=""
    for ln in txt.splitlines():
        if ln.startswith("#"): title=ln.lstrip("#").strip()[:80]; break
    rows.append(dict(
        path=d, title=title, bytes=len(txt.encode()),
        sha256=hashlib.sha256(txt.encode()).hexdigest()[:16],
        commits=int(g3.git(FULL,"rev-list","--count",SUBJ,"--",d).strip() or 0),
        last=g3.git(FULL,"log","-1","--format=%ad","--date=short",SUBJ,"--",d).strip(),
        graphA_in=inc.get(d,0), graphA_out=out_deg.get(d,0),
        exe_ops=exe.get(d,0),
        writers=sorted(writers.get(d,())), readers=sorted(readers.get(d,())),
        declared=sorted({m.group(0) for m in DECL.finditer(head)})[:6],
        has_sha=bool(SHA.search(head)), has_run=bool(RUN.search(head)),
        has_date=bool(DATE.search(head)),
        superseded_by=(SUPBY.search(head).group(1) if SUPBY.search(head) else None),
        says_supersedes=bool(SUPES.search(head)),
        present_tense=bool(PRESENT.search(head)),
    ))
assert len(rows)==len(tracked), "PASS A population mismatch"
json.dump(rows,open("/tmp/claude-0/-home-user-kai-system/84284242-f61e-5a69-9588-732883a5292c/scratchpad/h2-passA.json","w"),indent=1)
print(f"PASS A COMPLETE — evidence packets: {len(rows)} == population {len(tracked)}")
print(f"  with proven writer      : {sum(1 for r in rows if r['writers'])}")
print(f"  with any executable op  : {sum(1 for r in rows if r['exe_ops'])}")
print(f"  with a 'superseded by'  : {sum(1 for r in rows if r['superseded_by'])}")
print(f"  saying 'supersedes'     : {sum(1 for r in rows if r['says_supersedes'])}")
print(f"  containing a commit sha : {sum(1 for r in rows if r['has_sha'])}")
print(f"  containing a run id     : {sum(1 for r in rows if r['has_run'])}")
print(f"  present-tense state     : {sum(1 for r in rows if r['present_tense'])}")
print("  NO ROLE ASSIGNED IN PASS A.")
