#!/usr/bin/env python3
"""TREE-BOUND, IDENTITY-ONLY M3 record manifest.

PURPOSE. Give the future source-derived correctness-reference procedure
the record identities and the source navigation it needs to locate each
occurrence, WITHOUT handing it the candidate's own applicability-scope
answer. The existing manifest carries `current_applicability_scope`, so
feeding it to a correctness-reference derivation would leak the answer
into the input to that derivation. This artefact removes that leakage.

IT IS AN IDENTITY / SOURCE-LOCATION ARTEFACT. IT IS NOT AN ANSWER KEY.

EMITTED, and nothing else:

    path · start · end · detector · source_selector

DELIBERATELY ABSENT, enforced by `FORBIDDEN_FIELDS` below and by a
self-check that runs before the file is written:

    current_applicability_scope · any expected applicability scope ·
    any Cycle-5 derived scope · 214/278 reconciliation targets ·
    47/1 correction targets · any historical expected answer ·
    any adjudicated M3 answer · any reference-derived classification

`detector` is the RECOGNISER, not the witness_type. HEX and DECIMAL_RUN
can match the same span (F9-A), so the recogniser is load-bearing in the
identity and its removal would silently merge six pairs of distinct
records.

SOURCE BINDING. Content is read from the requested git object, never from
the working filesystem, and fails closed on an unresolvable tree, an
unresolvable path, or content that is not strictly decodable UTF-8. The
predecessor `make_record_manifest.py` took its paths from the tree and
its bytes from the filesystem; that defect is not reproduced here.

`_scope_of` IS NEVER CALLED BY THIS FILE. There is no scope computation
to leak, rather than a computed scope that is discarded before writing.

    python3 make_identity_manifest.py --subject-repo R --tree T --out F
    python3 make_identity_manifest.py --subject-repo R --tree T --prove
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

FIELDS = ("path", "start", "end", "detector", "source_selector")
FORBIDDEN_FIELDS = ("current_applicability_scope", "applicability_scope",
                    "expected_applicability_scope", "expected_scope",
                    "scope", "WHOLE_FILE", "SPAN", "UNKNOWN")


def esc(v):
    return (str(v).replace("\\", "\\\\").replace("\t", "\\t")
            .replace("\n", "\\n").replace("\r", "\\r"))


def _git(repo, *args, binary=False):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if r.returncode != 0:
        raise SystemExit(f"R11 ABORT: git {' '.join(args)} failed in {repo}: "
                         f"{r.stderr.decode(errors='replace').strip()[:200]}")
    return r.stdout if binary else r.stdout.decode()


def read_from_tree(repo, tree, path):
    """The document AS THE TREE HOLDS IT. Never the filesystem."""
    blob = _git(repo, "show", f"{tree}:{path}", binary=True)
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError as e:
        raise SystemExit(f"R11 ABORT: {path} at {tree[:12]} is not strictly "
                         f"UTF-8 ({e}). Refusing to emit identities whose "
                         f"offsets would be silently shifted.")


def identities(repo, tree):
    """Every emitted source occurrence, mirroring scan()'s own gating.

    Identity and navigation only. No scope is computed anywhere.
    """
    kind = _git(repo, "cat-file", "-t", tree).strip()
    if kind != "tree":
        raise SystemExit(f"R11 ABORT: {tree} resolves to {kind!r}, not a tree.")
    names = _git(repo, "ls-tree", "-r", "--name-only", tree).splitlines()
    rows = []
    for d in sorted(n for n in names if n.endswith(".md")):
        text = read_from_tree(repo, tree, d)

        def add(det, start, end):
            rows.append({"path": d, "start": start, "end": end,
                         "detector": det,
                         "source_selector": P._selector(text, start)})

        for det, rx in (("HEX", P.HEX), ("DATE", P.DATE)):
            for m in rx.finditer(text):
                if P._eligible(m):
                    add(det, m.start(), m.end())
        for m in P.DECIMAL_RUN.finditer(text):
            if not P._eligible(m):
                continue
            before = text[max(0, m.start() - 24):m.start()]
            window = text[max(0, m.start() - 24):m.end()]
            if P.RUN_NEAR.search(before) or P.RUN_URL.search(window):
                add("DECIMAL_RUN", m.start(), m.end())
        sup = next((x for x in P.SUPBY.finditer(text) if P._eligible(x)), None)
        if sup:
            add("SUPBY", sup.start(), sup.end())
    return sorted(rows, key=lambda r: (r["path"], r["start"], r["end"],
                                       r["detector"]))


def serialise(rows):
    keys = [(r["path"], r["start"], r["end"], r["detector"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"R11 ABORT: path+start+end+detector is NOT unique "
                         f"({len(keys)} rows, {len(set(keys))} keys).")
    body = "\n".join("\t".join(esc(r[f]) for f in FIELDS) for r in rows)
    text = "\t".join(FIELDS) + "\n" + body + "\n"
    # LEAKAGE SELF-CHECK, before a single byte is written. A field that
    # must not exist is worth failing over, not commenting about.
    header = text.split("\n", 1)[0]
    for bad in FORBIDDEN_FIELDS:
        if bad in header:
            raise SystemExit(f"R11 ABORT: forbidden field {bad!r} present in "
                             f"an identity-only artefact.")
    return text


def _prove(repo, tree):
    """Dirty worktree must not move the output; wrong tree must fail."""
    ok = True
    before = hashlib.sha256(serialise(identities(repo, tree)).encode()
                            ).hexdigest()
    again = hashlib.sha256(serialise(identities(repo, tree)).encode()
                           ).hexdigest()
    print(f"  deterministic regeneration : "
          f"{'IDENTICAL' if before == again else '<<< NON-DETERMINISTIC'}")
    ok &= before == again

    victim = pathlib.Path(repo) / "README.md"
    original = victim.read_bytes()
    try:
        victim.write_bytes(b"# DIRTY WORKTREE PROBE 2099-12-31\n\n" + original)
        dirty = hashlib.sha256(serialise(identities(repo, tree)).encode()
                               ).hexdigest()
    finally:
        victim.write_bytes(original)
    clean = _git(repo, "status", "--porcelain").strip() == ""
    print(f"  dirty-worktree can-fail    : "
          f"{'UNCHANGED, tree-bound' if dirty == before else '<<< LEAKED'}"
          f"   worktree restored clean: {clean}")
    ok &= dirty == before and clean

    for bad, label in ((tree[:-4] + "dead", "wrong tree object"),
                       ("HEAD", "non-tree object")):
        r = subprocess.run([sys.executable, __file__, "--subject-repo",
                            str(repo), "--tree", bad, "--out", "/dev/null"],
                           capture_output=True, text=True)
        fired = r.returncode != 0
        print(f"  fail-closed, {label:<18}: "
              f"{'ABORTS' if fired else '<<< PROCEEDED'}")
        ok &= fired
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-repo", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--out")
    ap.add_argument("--prove", action="store_true")
    a = ap.parse_args()

    if a.prove:
        raise SystemExit(0 if _prove(a.subject_repo, a.tree) else 1)
    if not a.out:
        raise SystemExit("--out is required unless --prove")
    rows = identities(a.subject_repo, a.tree)
    text = serialise(rows)
    pathlib.Path(a.out).write_text(text, encoding="utf-8")
    print(f"rows              {len(rows)}")
    print(f"fields            {' '.join(FIELDS)}")
    print(f"content source    git object {a.tree[:12]}, never the filesystem")
    print(f"scope fields      NONE — _scope_of is never called here")
    print(f"sha256(artefact)  "
          f"{hashlib.sha256(text.encode('utf-8')).hexdigest()}")


if __name__ == "__main__":
    main()
