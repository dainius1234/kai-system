#!/usr/bin/env python3
"""Create throwaway secret files for a CI bring-up, derived from compose.

`docker-compose.full.yml` declares three file-backed Docker secrets:

    secrets:
      hmac_secret:
        file: ${SECRETS_DIR:-./runtime-secrets}/hmac_secret
      db_password:
        file: ${SECRETS_DIR:-./runtime-secrets}/db_password
      bridge_secret:
        file: ${SECRETS_DIR:-./runtime-secrets}/bridge_secret

and nothing in the repository creates that directory. A comment tells a
human to. So the first time anything ever brought the full profile up —
2026-08-06, run 31101212426 — it died:

    level=warning msg="secret file kai-system_db_password does not exist"
    Container kai-system-backup-service-1  Error response from daemon:
      invalid mount config for type "bind": bind source path does not
      exist: …/runtime-secrets/db_password

That is the day's pattern once more: a defect in code that had never
executed. Five service-secret bindings across four services (tool-gate,
agentic, camera-service, backup-service) could not have worked, and no
test could have said so, because nothing had ever started them.

Derived, not listed
-------------------

The obvious fix is three `echo` lines in the workflow. That is the
list-beside-the-thing pattern being created on purpose: a fourth secret
added to compose would not be created, and the bring-up would fail on
the same error a month from now with the same explanation.

So the names are read from the compose file's own `secrets:` block.

These are not secrets
---------------------

The values are fixed, public, and say so in their own text. They exist
so that four containers can start in CI, and their only security
property is being unmistakable if one is ever found somewhere real.

Two guards, because a script that writes files named `db_password` is
worth being careful with:

  * an existing file is **never** overwritten — if somebody runs this on
    a machine that has real secret material, it is left alone and named
    in the output;
  * the target directory is gitignored, so the documented workflow of
    writing a real secret there cannot commit it. That gitignore entry
    was missing until this was written, which was the second finding.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

VALUE = "ci-throwaway-{name}-not-a-real-secret"

_ENV_DEFAULT = re.compile(r"\$\{[A-Z_]+:-([^}]+)\}")


def declared_secrets(compose: Path) -> List[Tuple[str, Path]]:
    """`(name, path)` for every file-backed secret the profile declares.

    The path is taken from the declaration, with `${SECRETS_DIR:-…}`
    resolved to its default — the same value compose itself uses when
    the variable is unset, which is the case this runs in.
    """
    import yaml

    doc = yaml.safe_load(compose.read_text(encoding="utf-8")) or {}
    out: List[Tuple[str, Path]] = []
    for name, cfg in (doc.get("secrets") or {}).items():
        source = (cfg or {}).get("file")
        if not source:
            continue        # an external secret is not ours to create
        out.append((name, Path(_ENV_DEFAULT.sub(r"\1", source))))
    return out


def provision(compose: Path, root: Path = REPO) -> Tuple[List[str], List[str]]:
    """Create every missing secret file. Returns (created, left alone)."""
    created: List[str] = []
    kept: List[str] = []
    for name, rel in declared_secrets(compose):
        path = (root / rel).resolve()
        if path.exists():
            kept.append(str(rel))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(VALUE.format(name=name) + "\n", encoding="utf-8")
        path.chmod(0o600)
        created.append(str(rel))
    return created, kept


def main(argv: List[str]) -> int:
    compose = REPO / (argv[0] if argv else "docker-compose.full.yml")
    if not compose.exists():
        print(f"REFUSED: {compose} does not exist, so no secret could be "
              f"derived from it. A bring-up run after this would fail on "
              f"a missing mount with a less obvious message.")
        return 1

    declared = declared_secrets(compose)
    created, kept = provision(compose)

    print(f"  inspected: {len(declared)} file-backed secret(s) declared in "
          f"{compose.name}")
    if not declared:
        print("  WARNING: the profile declares no file-backed secrets — "
              "either\n  that is a change nobody told this script about, or "
              "it is reading\n  the wrong file. Not a pass.")
        return 1

    for rel in created:
        print(f"    created  {rel}  (throwaway, CI only)")
    for rel in kept:
        print(f"    kept     {rel}  (already exists — NOT overwritten)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
