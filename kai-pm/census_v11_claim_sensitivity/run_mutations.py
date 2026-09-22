#!/usr/bin/env python3
"""CLAIM-SENSITIVITY MUTATIONS A and B — disposable copies of World A.

Precommitted in PRECOMMIT.md (sha256
a89bb9cc2f6aa9bf66096ebf424a9d6fd50dd328b04c39b7ea54bd7430efd5a0)
BEFORE any mutant was built. Census v1.1 is imported UNMODIFIED.

Each mutation must be APPLIED, REACHABLE, DISCRIMINATING and TARGETED —
D332 recorded two mutation attempts that passed while proving nothing,
because the mutated branch was never reached.
"""
import json, pathlib, shutil, subprocess, sys, tempfile
PKG = pathlib.Path("/home/user/kai-system/kai-pm/house_in_order_census_v11")
REPO = pathlib.Path("/home/user/kai-system")
REF = "d8aac4d49e6ba997e3eb38062c0917186ee3f197"
sys.path.insert(0, str(PKG))
import claims as C, docgraph as G, opscan as O, run_census as RC

RESULT = {"precommit_sha":
          "a89bb9cc2f6aa9bf66096ebf424a9d6fd50dd328b04c39b7ea54bd7430efd5a0",
          "subject": REF, "checks": [], "mutations": {}}
FAIL = []

def check(mut, cid, desc, ok, detail=""):
    RESULT["checks"].append({"mutation": mut, "id": cid, "expectation": desc,
                             "result": "PASS" if ok else "FAIL",
                             "detail": str(detail)[:400]})
    if not ok:
        FAIL.append(f"{mut}/{cid} {desc} :: {detail}")
    print(f"  [{'PASS' if ok else 'FAIL'}] {cid}  {desc}")
    if not ok:
        print(f"         detail: {str(detail)[:300]}")

def analyse(root):
    docs = G.tracked_md(root)
    ops, acc = O.collect(root, docs)
    C.classify(ops, docs, set(O.tracked(root)))
    ta = set(O.tracked(root))
    claims = {}
    for d in docs:
        v, srcs, b, _w, _s = C.scoped_claim(ops, d, ta)
        claims[d] = {"claim": v, "sources": srcs}
    return claims, ops, acc

def tally(claims):
    t = {}
    for v in claims.values():
        t[v["claim"]] = t.get(v["claim"], 0) + 1
    return t

def commit(root):
    subprocess.run(["git", "add", "-A", "--force"], cwd=root,
                   capture_output=True)
    subprocess.run(["git", "-c", "user.email=c@x", "-c", "user.name=c",
                    "commit", "-q", "-m", "mutant"], cwd=root,
                   capture_output=True)

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    base = td / "baseline"
    RC.materialise(REPO, REF, base)
    b_claims, b_ops, _ = analyse(base)
    b_tally = tally(b_claims)
    print(f"BASELINE  {b_tally}")
    RESULT["baseline_tally"] = b_tally
    if b_tally.get("PROVEN_WRITE_RELATION") != 5:
        raise SystemExit("R11 ABORT: baseline is not the expected 5 positives")

    # ── MUTATION A — POSITIVE INJECTION ──────────────────────────────
    print("\nMUTATION A — POSITIVE INJECTION (docs/DEMO.md)")
    ma = td / "mutantA"
    shutil.copytree(base, ma)
    tgt = ma / "scripts/sync_docs.py"
    before = tgt.read_text()
    tgt.write_text(before + '\n\nimport pathlib\n'
                   'pathlib.Path("docs/DEMO.md").write_text("mutation A probe")\n')
    commit(ma)
    check("A", "A0-APPLIED", "the mutant source differs from baseline",
          tgt.read_text() != before
          and 'pathlib.Path("docs/DEMO.md").write_text' in tgt.read_text())
    a_claims, a_ops, _ = analyse(ma)
    a_tally = tally(a_claims)
    RESULT["mutations"]["A"] = {"tally": a_tally,
                                "demo": a_claims.get("docs/DEMO.md")}
    print(f"  mutant tally {a_tally}")

    inj = [o for o in a_ops if o.target == "docs/DEMO.md" and o.mode == "W"]
    check("A", "A5-REACHABLE",
          "injected op is admitted, RESOLVED_WRITE, target docs/DEMO.md",
          len(inj) == 1 and inj[0].disposition == "RESOLVED_WRITE"
          and inj[0].src == "scripts/sync_docs.py",
          [(o.src, o.line, o.disposition) for o in inj])
    check("A", "A1", "PROVEN_WRITE_RELATION 5 -> 6",
          a_tally.get("PROVEN_WRITE_RELATION") == 6, a_tally)
    check("A", "A2", "docs/DEMO.md NO_PROVEN_WRITER -> PROVEN_WRITE_RELATION",
          b_claims["docs/DEMO.md"]["claim"] == "NO_PROVEN_WRITER"
          and a_claims["docs/DEMO.md"]["claim"] == "PROVEN_WRITE_RELATION",
          (b_claims["docs/DEMO.md"], a_claims["docs/DEMO.md"]))
    check("A", "A3", 'docs/DEMO.md sources == ["scripts/sync_docs.py"]',
          a_claims["docs/DEMO.md"]["sources"] == ["scripts/sync_docs.py"],
          a_claims["docs/DEMO.md"]["sources"])
    movedA = [d for d in b_claims
              if b_claims[d]["claim"] != a_claims.get(d, {}).get("claim")]
    check("A", "A4-TARGETED", "no unrelated document changed claim",
          movedA == ["docs/DEMO.md"], movedA)
    check("A", "A6-DISCRIMINATING", "baseline and mutant claim tables differ",
          b_tally != a_tally, (b_tally, a_tally))

    # ── MUTATION B — POSITIVE REMOVAL ────────────────────────────────
    print("\nMUTATION B — POSITIVE REMOVAL (CHANGELOG.md)")
    mb = td / "mutantB"
    shutil.copytree(base, mb)
    ac = mb / "scripts/auto_changelog.py"
    lines = ac.read_text().splitlines(keepends=True)
    anchor = lines[138]
    # rule 18: assert the anchor matched before editing on a line number
    check("B", "B0-ANCHOR", "line 139 is the expected write statement",
          "CHANGELOG.write_text(" in anchor, anchor.strip())
    if "CHANGELOG.write_text(" not in anchor:
        raise SystemExit("R11 ABORT: anchor did not match; refusing to edit")
    indent = anchor[:len(anchor) - len(anchor.lstrip())]
    lines[138] = f"{indent}pass  # neutralised for Mutation B\n"
    ac.write_text("".join(lines))
    commit(mb)
    check("B", "B0-APPLIED", "the write statement is gone from the mutant",
          "CHANGELOG.write_text(" not in ac.read_text())
    b2_claims, b2_ops, _ = analyse(mb)
    b2_tally = tally(b2_claims)
    RESULT["mutations"]["B"] = {"tally": b2_tally,
                                "changelog": b2_claims.get("CHANGELOG.md")}
    print(f"  mutant tally {b2_tally}")

    w = [o for o in b2_ops if o.target == "CHANGELOG.md" and o.mode == "W"]
    r = [o for o in b2_ops if o.target == "CHANGELOG.md" and o.mode == "R"]
    check("B", "B5-APPLIED",
          "no WRITE op resolves to CHANGELOG.md; the two READs survive",
          len(w) == 0 and len(r) == 2, (len(w), len(r)))
    check("B", "B1", "PROVEN_WRITE_RELATION 5 -> 4",
          b2_tally.get("PROVEN_WRITE_RELATION") == 4, b2_tally)
    check("B", "B2", "CHANGELOG.md PROVEN_WRITE_RELATION -> NO_PROVEN_WRITER",
          b_claims["CHANGELOG.md"]["claim"] == "PROVEN_WRITE_RELATION"
          and b2_claims["CHANGELOG.md"]["claim"] == "NO_PROVEN_WRITER",
          (b_claims["CHANGELOG.md"], b2_claims["CHANGELOG.md"]))
    check("B", "B3",
          "CHANGELOG.md must NOT become NO_WRITER_WITHIN_ANALYZED_SCOPE",
          b2_claims["CHANGELOG.md"]["claim"]
          != "NO_WRITER_WITHIN_ANALYZED_SCOPE",
          b2_claims["CHANGELOG.md"])
    movedB = [d for d in b_claims
              if b_claims[d]["claim"] != b2_claims.get(d, {}).get("claim")]
    check("B", "B4-TARGETED", "no unrelated document changed claim",
          movedB == ["CHANGELOG.md"], movedB)
    check("B", "B6-DISCRIMINATING", "baseline and mutant claim tables differ",
          b_tally != b2_tally, (b_tally, b2_tally))

RESULT["verdict"] = "ALL EXPECTATIONS MET" if not FAIL else "FAILED"
RESULT["failures"] = FAIL
pathlib.Path("MUTATION_RESULT.json").write_text(
    json.dumps(RESULT, indent=1, sort_keys=True))
print(f"\n{'='*66}\nVERDICT: {RESULT['verdict']}   "
      f"({sum(1 for c in RESULT['checks'] if c['result']=='PASS')}"
      f"/{len(RESULT['checks'])} checks passed)")
for f in FAIL:
    print("  FAIL", f)
sys.exit(1 if FAIL else 0)
