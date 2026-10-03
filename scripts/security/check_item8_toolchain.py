#!/usr/bin/env python3
"""A SHA-256 of an incomplete record is a perfect hash of bad evidence.

WHAT WENT WRONG
===============

Frozen R2 requires the toolchain recorded **with every branch**: the
pinned frontend, the Docker and buildx versions, the `python:3.11-slim`
base-image digest, the runner OS, and the tree and run identity.

The first implementation wrote one file, hashed it, and put the hash in
each row. That binds a row to *a file* — it does not establish the file
says anything. The calibration proved the point without meaning to: its
fixture toolchain contained two fields, and six branches qualified
against it.

Worse, the generator runs under `set -uo pipefail` rather than `-e`, so a
failed command inside a `$( )` leaves an EMPTY value while the enclosing
`echo` succeeds. `key=` is then a present key with nothing behind it, and
only the literal word `UNRESOLVED` was ever looked for.

So this validates the record BEFORE any build, and a failure here costs
**zero** experimental builds rather than six.

WHAT IT CHECKS
==============

1. every required key is present;
2. every value is non-empty — `key=` is absence wearing a key's clothes;
3. no value is `UNRESOLVED`;
4. the repository identities in the record match the execution actually
   about to happen, so a stale toolchain file cannot be carried forward.

(4) is the one a hash cannot give you. A digest proves two parties read
the same bytes; it says nothing about whether those bytes describe
*this* run.

Exit 0 = the record is complete and describes this execution.
Exit 1 = refused. No build may start.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

# Derived from frozen R2's sentence, not from a wish-list kept beside it:
# "the frontend identity, Docker/buildx versions, base-image digest,
#  runner OS, tree and run identity are recorded with every branch".
REQUIRED = (
    "frontend",
    "docker_version",
    "buildx_version",
    "base_image_digest",
    "runner_os",
    "commit_sha",
    "tree_sha",
    "run_id",
)

# The pinned frontend R2 names. A toolchain claiming a different one is
# describing a different experiment.
FRONTEND = ("docker/dockerfile:1.9.0@sha256:"
            "fe40cf4e92cd0c467be2cfc30657a680ae2398318afd50b0c80585784c604f28")


def git(*args: str) -> str | None:
    p = subprocess.run(["git", "-C", str(REPO), *args],
                       capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 and p.stdout.strip() else None


def parse(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def contract_problems(rec: dict[str, str]) -> list[str]:
    """THE CONTRACT, in ONE place, for BOTH boundaries.

    This module ran before build 1 and required eight identities. The
    summariser then re-derived a smaller contract of its own -- enough of
    the artefact to reconcile hash, tree, run and base image -- so an
    archived toolchain holding only those four could support a closure
    claim while lacking the frontend, the Docker and buildx versions, the
    runner OS and the commit.

    That made the evidence package non-self-validating: reading it later
    meant *remembering* that a stricter step had once run. Two
    interpretations of one contract is D272's failure shape, and the
    repair is the same one — a single function, imported, not copied.
    (D296)
    """
    problems: list[str] = []
    missing = [k for k in REQUIRED if k not in rec]
    if missing:
        problems.append(f"missing required identity(s): {', '.join(missing)}")
    empty = [k for k in REQUIRED if k in rec and not rec[k]]
    if empty:
        problems.append(f"present but EMPTY: {', '.join(empty)} — `key=` is "
                        f"absence wearing a key's clothes")
    unresolved = [k for k in REQUIRED if rec.get(k) == "UNRESOLVED"]
    if unresolved:
        problems.append(f"UNRESOLVED: {', '.join(unresolved)}")
    if rec.get("frontend") and rec["frontend"] != FRONTEND:
        problems.append(f"frontend is {rec['frontend']!r}, not the pinned "
                        f"value R2 froze")
    return problems


def refuse(msg: str) -> int:
    print(f"REFUSED: {msg}")
    print()
    print("NO BUILD MAY START. The toolchain record is the thing every "
          "branch result binds to; an incomplete one binds them to "
          "nothing, and it costs zero builds to find out now. (D294)")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--toolchain", required=True)
    ap.add_argument("--expect-run-id", help="the run this record must describe")
    ap.add_argument("--skip-repo-identity", action="store_true",
                    help="calibration only: omit the match against this "
                         "repository's current commit and tree")
    args = ap.parse_args()

    path = pathlib.Path(args.toolchain)
    if not path.is_file():
        return refuse(f"{path} does not exist. R2 requires the toolchain "
                      f"recorded with every branch, and there is no record")

    raw = path.read_bytes()
    rec = parse(raw.decode("utf-8", "replace"))

    # THE SHARED CONTRACT. Same function the summariser calls at closure,
    # so the two boundaries cannot drift into two different ideas of what
    # a complete record is.
    problems = contract_problems(rec)
    if problems:
        return refuse("; ".join(problems) + ". R2 names these identities; a "
                      "record without them describes a different, smaller "
                      "thing, and the generator runs without `set -e`, so a "
                      "failed lookup leaves exactly this")

    if not args.skip_repo_identity:
        head = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        if head is None or tree is None:
            return refuse("this repository's HEAD/tree could not be "
                          "resolved, so the record cannot be shown to "
                          "describe THIS execution")
        if rec["commit_sha"] != head:
            return refuse(f"commit_sha {rec['commit_sha'][:12]} is not this "
                          f"execution's HEAD {head[:12]}. A stale toolchain "
                          f"record cannot be carried forward")
        if rec["tree_sha"] != tree:
            return refuse(f"tree_sha {rec['tree_sha'][:12]} is not this "
                          f"execution's tree {tree[:12]}")
    if args.expect_run_id and rec["run_id"] != args.expect_run_id:
        return refuse(f"run_id {rec['run_id']} is not this run "
                      f"{args.expect_run_id}")

    digest = hashlib.sha256(raw).hexdigest()
    print("ITEM-8 TOOLCHAIN RECORD")
    print("=" * 68)
    for k in REQUIRED:
        print(f"  {k:<18} {rec[k][:80]}")
    print()
    print(f"  sha256: {digest}")
    print()
    print(f"  inspected: {len(REQUIRED)} required identity(s), "
          f"{len(rec)} recorded")
    print()
    print("PASS: complete, resolved, pinned, and describing this execution.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
