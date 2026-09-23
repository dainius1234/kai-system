"""Consumer inventory search. Universe and patterns are fixed here and
printed; every hit is emitted, nothing filtered after the predicate."""
import re, subprocess, json, sys
BASE = "194db0a0c13b4d5b322997fc1ceb33bdd21a77bc"
files = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", BASE],
                       capture_output=True).stdout.decode().split("\0")
files = [f for f in files if f]
EXEC = re.compile(r"(\.(py|sh|bash|ya?ml|toml|cfg|ini|json|js|mjs|ts|tsx)$|(^|/)Makefile[^/]*$|(^|/)Dockerfile[^/]*$|(^|/)[^/.]+$)")
DOC = re.compile(r"\.(md|txt|rst|html|csv)$")
SEP = r"""[\s"',\[\]]+"""        # shell spacing AND argv-list spelling
PATTERNS = {
 "P1 rev-parse HEAD":        r"rev-parse" + SEP + r"(?:--[\w-]+" + SEP + r")*HEAD\b",
 "P2 github.sha/GITHUB_SHA": r"github\.sha\b|\bGITHUB_SHA\b",
 "P3 github.ref family":     r"github\.(?:ref|ref_name|head_ref|base_ref)\b|\bGITHUB_(?:REF|REF_NAME|HEAD_REF|BASE_REF)\b",
 "P4 hard-coded main":       r"origin/main\b|refs/heads/main\b|(?:checkout|switch|--branch|-b)" + SEP + r"main\b|branches:\s*\[[^\]]*\bmain\b|^\s*-\s*main\s*$",
 "P5 current branch":        r"rev-parse" + SEP + r"--abbrev-ref|symbolic-ref\b|branch" + SEP + r"--show-current",
 "P6 git describe":          r"\bgit" + SEP + r"describe\b",
 "P7 other git cmd on HEAD": r"\bgit\b[^\n]*\b(?:log|diff|show|rev-list|cat-file|ls-tree|archive|diff-tree)\b[^\n]*\bHEAD\b",
}
LOOSE = re.compile(r"rev-parse|\bHEAD\b|GITHUB_SHA|github\.sha")
rx = {k: re.compile(v, re.M) for k, v in PATTERNS.items()}
exec_files = [f for f in files if EXEC.search(f) and not DOC.search(f)]
doc_files = [f for f in files if DOC.search(f)]
def scan(fs):
    hits = []
    for f in fs:
        blob = subprocess.run(["git", "show", f"{BASE}:{f}"], capture_output=True).stdout
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for k, r in rx.items():
                if r.search(line):
                    hits.append({"file": f, "line": i, "pattern": k, "text": line.strip()[:160]})
    return hits
eh, dh = scan(exec_files), scan(doc_files)
tight = {(h["file"], h["line"]) for h in eh}
loose_unmatched = []
for f in exec_files:
    blob = subprocess.run(["git", "show", f"{BASE}:{f}"], capture_output=True).stdout
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        continue
    for i, line in enumerate(text.splitlines(), 1):
        if LOOSE.search(line) and (f, i) not in tight:
            loose_unmatched.append({"file": f, "line": i, "text": line.strip()[:160]})
out = {"base": BASE, "tracked_files": len(files), "universe_exec_config": len(exec_files),
       "excluded_docs": len(doc_files), "other_excluded": len(files) - len(exec_files) - len(doc_files),
       "patterns": PATTERNS, "hits": eh, "doc_hits_count": len(dh),
       "doc_hits": dh, "loose_unmatched": loose_unmatched}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(f"tracked {len(files)} | universe {len(exec_files)} | docs excluded {len(doc_files)} (their hits: {len(dh)}) | other excluded {out['other_excluded']}")
from collections import Counter
print("hits in universe:", len(eh), dict(Counter(h['pattern'] for h in eh)))
print("files with hits:", len({h['file'] for h in eh}))
print("loose-superset lines NOT matched by a tight pattern:", len(loose_unmatched))
for u in loose_unmatched: print("   ", u["file"], u["line"], u["text"][:110])
