#!/usr/bin/env python3
"""CAI v1.0 hostile calibration — path and mode scope (order §12, §18 S1-S7).

Every case: fresh real repository, a real baseline->candidate change, the
SHIPPED check_cai_scope.py run as a subprocess, exit code AND reason
asserted. Known-positive: S1. Everything else must REFUSE for its own
reason. Entries that are awkward on a working tree (gitlinks, case and
Unicode collisions, backslashes) are written straight into the index, so
the case tests Git object paths exactly as the contract says authority
reads them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cai_testkit import Keys, Suite, World, run_checker  # noqa: E402

DECLARED = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
SCOPE = {"allowed_paths": [], "allowed_prefixes": ["allowed/"],
         "forbidden_paths": [], "forbidden_prefixes": ["forbidden/"]}


def scope(w, cand):
    f = w.dir / "scope.json"
    f.write_text(json.dumps(SCOPE))
    return run_checker("check_cai_scope.py", "--repo", str(w.repo),
                       "--baseline", w.B, "--candidate", cand,
                       "--scope", str(f))


def index_entry(w, mode, sha, path):
    w.git("update-index", "--add", "--cacheinfo", f"{mode},{sha},{path}")


def blob(w, text):
    return w.git("hash-object", "-w", "--stdin",
                 input_bytes=text.encode()).strip()


def forge_commit(w, parent, extra: dict, msg):
    """Build a commit from RAW objects, bypassing the index.

    Git's index refuses some hostile names (a `.GIT` component, for one),
    but tree objects can be written directly with `git mktree`. An
    attacker is not obliged to use the index, so neither is this case.
    extra: {path: (mode, sha)} added on top of `parent`'s tree.
    """
    ents = {}
    # -z on BOTH ends: names pass byte-exact and are never C-quoted, which
    # is what non-ASCII and backslash names would otherwise become.
    for line in w.git("ls-tree", "-r", "-z", "--full-tree",
                      parent).split("\0"):
        if not line:
            continue
        meta, path = line.split("\t", 1)
        mode, typ, sha = meta.split()
        ents[path] = (mode, typ, sha)
    for path, (mode, sha) in extra.items():
        ents[path] = (mode, "commit" if mode == "160000" else "blob", sha)

    def build(prefix):
        rows, subdirs = [], set()
        for path, (mode, typ, sha) in ents.items():
            if not path.startswith(prefix):
                continue
            rest = path[len(prefix):]
            if "/" in rest:
                subdirs.add(rest.split("/", 1)[0])
            else:
                rows.append(f"{mode} {typ} {sha}\t{rest}")
        for d in sorted(subdirs):
            rows.append(f"040000 tree {build(prefix + d + '/')}\t{d}")
        return w.git("mktree", "-z", input_bytes=("\0".join(rows) + "\0")
                     .encode()).strip()

    return w.git("commit-tree", build(""), "-p", parent, "-m", msg).strip()


def commit_index(w, msg):
    w.git("commit", "-q", "-m", msg)
    return w.git("rev-parse", "HEAD").strip()


def main() -> int:
    keys = Keys()
    s = Suite("test_cai_scope")
    worlds = []

    def world():
        w = World(keys)
        worlds.append(w)
        w.checkout(w.B)
        return w

    try:
        w = world()
        c = w.change(w.B, {"allowed/a.txt": "a2\n"})
        rc, out = scope(w, c)
        s.expect("S1", "allowed path modified -> PASS", rc, out, want_rc=0,
                 must_contain=["changed entries evaluated: 1", "PASS"])

        w = world()
        c = w.change(w.B, {"docs/readme.md": "r2\n",
                           "allowed-evil/x.txt": "boundary\n"})
        rc, out = scope(w, c)
        s.expect("S2", "outside-scope path, and prefix-boundary look-alike "
                       "'allowed-evil/' -> REFUSE", rc, out, want_rc=1,
                 must_contain=["docs/readme.md (M): outside every allowed",
                               "allowed-evil/x.txt (A): outside every allowed"])

        w = world()
        w.git("mv", "allowed/a.txt", "forbidden/a.txt")
        c = commit_index(w, "rename into forbidden")
        rc, out = scope(w, c)
        s.expect("S3", "rename INTO forbidden scope -> REFUSE (new end)", rc,
                 out, want_rc=1,
                 must_contain=["forbidden/a.txt (A): under a forbidden prefix"])

        w = world()
        w.git("mv", "forbidden/x.txt", "allowed/x.txt")
        c = commit_index(w, "rename out of forbidden")
        rc, out = scope(w, c)
        s.expect("S4", "rename OUT OF forbidden scope -> REFUSE (old end)", rc,
                 out, want_rc=1,
                 must_contain=["forbidden/x.txt (D): under a forbidden prefix"])

        w = world()
        index_entry(w, "120000", blob(w, "../forbidden/x.txt"),
                    "allowed/link")
        c = commit_index(w, "symlink")
        rc, out = scope(w, c)
        s.expect("S5", "symlink (mode 120000) inside allowed scope -> REFUSE",
                 rc, out, want_rc=1,
                 must_contain=["symlink (mode 120000) refused"])

        w = world()
        index_entry(w, "160000", w.B, "allowed/sub")
        c = commit_index(w, "gitlink")
        rc, out = scope(w, c)
        s.expect("S6", "gitlink (mode 160000) inside allowed scope -> REFUSE",
                 rc, out, want_rc=1,
                 must_contain=["gitlink (mode 160000) refused"])

        w = world()
        b = blob(w, "x\n")
        index_entry(w, "100644", b, "allowed/A.txt")          # vs allowed/a.txt
        index_entry(w, "100644", b, "allowed/café.txt")  # NFD
        index_entry(w, "100644", b, "allowed/back\\slash.txt")
        c = commit_index(w, "ambiguous paths (index-admissible ones)")
        c = forge_commit(w, c, {"allowed/.GIT/config": ("100644", b)},
                         "hostile tree built without the index")
        rc, out = scope(w, c)
        s.expect("S7", "ambiguous path representations -> REFUSE", rc, out,
                 want_rc=1,
                 must_contain=["collides after case-fold/NFC",
                               "not NFC-normalised",
                               "contains a backslash",
                               "has a .git component"])
    finally:
        for w in worlds:
            w.cleanup()
        keys.cleanup()
    return s.finish(DECLARED)


if __name__ == "__main__":
    sys.exit(main())
