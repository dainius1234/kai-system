#!/usr/bin/env python3
"""TREE-BOUND canonical source-occurrence manifest. Replaces the source
binding of `make_record_manifest.py` for all NEW reference work.

WHY THIS EXISTS. `make_record_manifest.py` takes its path population from
`git ls-tree -r --name-only <tree>` but reads every document with
`Path(repo)/d.read_text()` -- from the WORKING FILESYSTEM. The
enumeration is bound to the requested tree; **the bytes are not**. So
`--tree` never actually bound the content being analysed, and a manifest
claiming tree identity could have been computed over content that tree
never contained.

REPO_CONFIRMED_DEFECT under doctrine 50, and LATENT rather than realised:
at the time of repair the subject worktree was clean and at tree
3abc9e9d, and all 272 documents matched the tree byte-for-byte. The
frozen 492-record manifest is therefore uncorrupted -- correct because
the worktree happened to agree, which is not the same as correct by
construction. This file makes it the second thing.

THE HISTORICAL BUILDER IS NOT MODIFIED. `make_record_manifest.py` and the
manifest it produced stay exactly as they are, as evidence of what
actually ran (Kai's ruling: repair forward, do not rewrite history).

WHAT CHANGED, and only this: content is read from the git object. There
is deliberately no filesystem read of any subject document anywhere in
this file. Detection, scoping, ordering, canonical form and the
uniqueness abort are byte-for-byte the same logic, which is why this
reproduces the frozen manifest exactly.

FAIL CLOSED. An unresolvable tree, a path that will not resolve to a
blob, or content that is not strictly decodable UTF-8 ABORTS. The old
path used `errors="ignore"`, which silently substitutes characters and
would shift every subsequent offset in the file -- an undetectable
corruption of the very identities this manifest exists to fix.

    python3 make_record_manifest_treebound.py --subject-repo R --tree T --out F
    python3 make_record_manifest_treebound.py --subject-repo R --tree T \\
        --prove-worktree-independence
"""
from __future__ import annotations
import argparse
import hashlib
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import passa as P                                              # noqa: E402

FIELDS = ("path", "start", "end", "detector", "source_selector",
          "current_applicability_scope")


def esc(v):
    return (str(v).replace("\\", "\\\\").replace("\t", "\\t")
            .replace("\n", "\\n").replace("\r", "\\r"))


def _git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed in {repo}: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout if binary else r.stdout.decode()


def read_from_tree(repo, tree, path):
    """The document AS THE TREE HOLDS IT. Never the filesystem.

    Fail-closed on both legs: an unresolvable object aborts, and so does
    content that is not strictly UTF-8. `errors="ignore"` would shift
    every offset after the bad byte while still producing a manifest.
    """
    blob = _git(repo, "show", f"{tree}:{path}", binary=True)
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(f"R11 ABORT: {path} at {tree[:12]} is not strictly "
                         f"UTF-8 ({e}). Refusing to emit a manifest whose "
                         f"offsets would be silently shifted.")


def occurrences(repo, tree):
    """Every emitted source occurrence, mirroring scan()'s own gating."""
    kind = _git(repo, "cat-file", "-t", tree).strip()
    if kind != "tree":
        raise SystemExit(f"R11 ABORT: {tree} resolves to {kind!r}, not a tree.")
    names = _git(repo, "ls-tree", "-r", "--name-only", tree).splitlines()
    rows = []
    for d in sorted(n for n in names if n.endswith(".md")):
        text = read_from_tree(repo, tree, d)          # <- the whole repair
        head = text[:P.HEAD_BYTES]

        def add(det, m):
            rows.append({"path": d, "start": m.start(), "end": m.end(),
                         "detector": det,
                         "source_selector": P._selector(text, m.start()),
                         "current_applicability_scope":
                             P._scope_of(head, m.start(), det)})

        for det, rx in (("HEX", P.HEX), ("DATE", P.DATE)):
            for m in rx.finditer(text):
                if P._eligible(m):
                    add(det, m)
        for m in P.DECIMAL_RUN.finditer(text):
            if not P._eligible(m):
                continue
            before = text[max(0, m.start() - 24):m.start()]
            window = text[max(0, m.start() - 24):m.end()]
            if P.RUN_NEAR.search(before) or P.RUN_URL.search(window):
                add("DECIMAL_RUN", m)
        sup = next((x for x in P.SUPBY.finditer(text) if P._eligible(x)), None)
        if sup:
            rows.append({"path": d, "start": sup.start(), "end": sup.end(),
                         "detector": "SUPBY",
                         "source_selector": P._selector(text, sup.start()),
                         # supersession is file-level by construction
                         "current_applicability_scope": "WHOLE_FILE"})
    return sorted(rows, key=lambda r: (r["path"], r["start"], r["end"],
                                       r["detector"]))


def serialise(rows):
    keys = [(r["path"], r["start"], r["end"], r["detector"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"R11 ABORT: path+start+end+detector is NOT unique "
                         f"({len(keys)} rows, {len(set(keys))} keys). An "
                         f"explicit ordinal is required; do not add one "
                         f"silently.")
    body = "\n".join("\t".join(esc(r[f]) for f in FIELDS) for r in rows)
    return "\t".join(FIELDS) + "\n" + body + "\n"


def _prove(repo, tree):
    """CAN-FAIL. A dirty worktree must not be able to move this output.

    The legacy path must CHANGE and the tree-bound path must NOT. If the
    legacy path did not change, this proof would be vacuous -- it would
    demonstrate nothing about where the bytes came from.
    """
    sys.path.insert(0, str(HERE))
    import make_record_manifest as legacy

    before_new = hashlib.sha256(
        serialise(occurrences(repo, tree)).encode()).hexdigest()
    before_old = hashlib.sha256(
        serialise(legacy.occurrences(str(repo), tree)).encode()).hexdigest()

    victim = pathlib.Path(repo) / "README.md"
    original = victim.read_bytes()
    try:
        victim.write_bytes(b"# DIRTY WORKTREE PROBE 2099-12-31\n\n" + original)
        after_new = hashlib.sha256(
            serialise(occurrences(repo, tree)).encode()).hexdigest()
        after_old = hashlib.sha256(
            serialise(legacy.occurrences(str(repo), tree)).encode()).hexdigest()
    finally:
        victim.write_bytes(original)

    dirty = pathlib.Path(repo)
    restored = _git(dirty, "status", "--porcelain").strip() == ""
    print("CAN-FAIL — a dirty worktree, the SAME tree argument throughout\n")
    print(f"  legacy builder   before {before_old[:16]}")
    print(f"                   after  {after_old[:16]}   "
          f"{'CHANGED -- defect demonstrated' if before_old != after_old else '<<< PROOF VACUOUS'}")
    print(f"  tree-bound       before {before_new[:16]}")
    print(f"                   after  {after_new[:16]}   "
          f"{'UNCHANGED -- repair demonstrated' if before_new == after_new else '<<< REPAIR FAILED'}")
    print(f"\n  worktree restored clean: {restored}")
    ok = (before_old != after_old) and (before_new == after_new) and restored
    print(f"  WORKTREE INDEPENDENCE: {'PROVEN' if ok else 'NOT PROVEN'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out")
    ap.add_argument("--prove-worktree-independence", action="store_true")
    a = ap.parse_args()

    if a.prove_worktree_independence:
        raise SystemExit(0 if _prove(a.subject_repo, a.tree) else 1)

    if not a.out:
        raise SystemExit("--out is required unless proving independence")
    rows = occurrences(a.subject_repo, a.tree)
    text = serialise(rows)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    wf = sum(1 for r in rows
             if r["current_applicability_scope"] == "WHOLE_FILE")
    print(f"rows              {len(rows)}")
    print(f"WHOLE_FILE        {wf}")
    print(f"SPAN              {len(rows) - wf}")
    print(f"content source    git object {a.tree[:12]}, never the filesystem")
    print(f"sha256(manifest)  "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
