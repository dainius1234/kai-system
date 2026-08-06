#!/usr/bin/env python3
"""A `$` in a compose `command:` belongs to compose, not to the shell.

`docker-compose.sovereign.yml` carried this executor command for as long
as the executor has existed:

    command: sh -c "set -e; for d in $BACKOFF_SCHEDULE; do
                      python app.py && exit 0 || sleep $d;
                    done; exec python app.py"

`BACKOFF_SCHEDULE: "10 60 300"` is set three lines above — in the
service's **`environment:`**, which is the *container's* environment.
Compose interpolates `$VAR` from its own environment when it parses the
file, long before any shell exists, and it knew neither name. Both were
replaced with the empty string, so the shell received:

    for d in ; do python app.py && exit 0 || sleep ; done; exec python app.py

An empty `for` list runs its body zero times. The retry-with-backoff
that line exists to provide has never retried once; every boot fell
through to the bare `exec` at the end. The command *worked*, which is
why it survived — it just never did the thing it was written to do.

It announced itself on every compose invocation in CI:

    The "BACKOFF_SCHEDULE" variable is not set. Defaulting to a blank string.
    The "d" variable is not set. Defaulting to a blank string.

Directive 3 — nothing repeats unexplained — put a runtime check in
`core-tests.yml` that fails on `variable is not set`. It greps
`/tmp/bringup.log`, which is the **minimal** profile's bring-up. This
defect is in **sovereign**. A check whose scope is smaller than its
name, again, and this time in the fix for the last one.

So the rule moves from a log grep over one profile to a static parse of
every compose file in the tree, found by a glob rather than a list.

The rule, per `$NAME` inside a `command:` or `entrypoint:`:

  ``$$NAME``           correct — compose passes a literal `$` through
  ``${NAME:-default}`` fine — compose substitutes something real
  NAME in the service's own ``environment:``
                       **defect** — it is plainly meant for the shell
                       inside the container, and compose is eating it
  NAME in ``.env.example``
                       fine — compose reads `.env` at run time
  anything else        **defect** — compose substitutes blank and the
                       command silently means something else

Exit 0 = every `$` in every command reaches whoever it was written for.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import (  # noqa: E402
    compose_files as _compose_files, inspected)

#: `$$NAME` first so an escaped reference is consumed before the bare
#: pattern can match its tail. `${NAME:-x}` and `${NAME}` are captured
#: with their default (if any) so the two cases can be told apart.
_REF = re.compile(r"""
    \$\$\w+                     # correctly escaped — matched, then skipped
  | \$\{(?P<braced>\w+)(?P<default>:?[-+?][^}]*)?\}
  | \$(?P<bare>\w+)
""", re.VERBOSE)


def compose_files(root: Path = None) -> List[Path]:
    """Every compose file at the repo root — see `gate_inputs.compose_files`.

    Derived rather than listed for the reason this gate exists: the
    runtime check it replaces named one profile and the defect was in
    another. Kept as a thin alias because this glob had been written out
    three times independently, which is the list-beside-the-thing
    pattern applied to the lists themselves.
    """
    return _compose_files(root or REPO)


def env_example_names(root: Path = None) -> Set[str]:
    """Names `.env` is expected to supply, so compose can substitute them."""
    root = root or REPO
    path = root / ".env.example"
    if not path.exists():
        return set()
    names: Set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        names.add(line.split("=", 1)[0].strip())
    return names


def _shell_strings(cfg: dict) -> List[Tuple[str, str]]:
    """(field, text) for every `command:`/`entrypoint:` spelling.

    Both accept a string or a list; a list is joined so a reference
    split across items is still seen as one text to scan.
    """
    out: List[Tuple[str, str]] = []
    for field in ("command", "entrypoint"):
        value = (cfg or {}).get(field)
        if isinstance(value, str):
            out.append((field, value))
        elif isinstance(value, list):
            out.append((field, " ".join(str(v) for v in value)))
    return out


def findings_in(doc: dict, origin: str, supplied: Set[str]) -> List[str]:
    """Every `$NAME` compose will eat that somebody meant for a shell."""
    findings: List[str] = []
    for service, cfg in sorted((doc.get("services") or {}).items()):
        own_env = set((cfg or {}).get("environment") or {})
        for field, text in _shell_strings(cfg):
            for match in _REF.finditer(text):
                name = match.group("braced") or match.group("bare")
                if not name:
                    continue                      # `$$NAME` — already right
                if match.group("default"):
                    continue                      # compose has something to put here
                if name in own_env:
                    findings.append(
                        f"{origin}: service '{service}' {field} uses "
                        f"`${name}`, and '{name}' is a key in that "
                        f"service's own `environment:` — compose "
                        f"substitutes it blank before the container "
                        f"exists. Write `$${name}` to reach the shell.")
                elif name not in supplied:
                    findings.append(
                        f"{origin}: service '{service}' {field} uses "
                        f"`${name}`, which nothing supplies — compose "
                        f"substitutes the empty string and the command "
                        f"quietly means something else. Escape it as "
                        f"`$${name}`, give it a default, or add it to "
                        f".env.example.")
    return findings


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, references inspected, compose files read)."""
    import yaml

    root = root or REPO
    supplied = env_example_names(root)
    findings: List[str] = []
    refs = 0
    paths = compose_files(root)
    if not paths:
        # I-1: zero inputs is a finding, not a pass.
        return ([f"{root}: no docker-compose*.yml found — this gate "
                 f"inspected nothing and must not report success"], 0, 0)
    for path in paths:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        origin = path.name
        for _, cfg in sorted((doc.get("services") or {}).items()):
            for _, text in _shell_strings(cfg):
                refs += len(_REF.findall(text))
        findings.extend(findings_in(doc, origin, supplied))
    return findings, refs, len(paths)


def main() -> int:
    findings, refs, files = audit()

    print(inspected(refs, "variable reference(s)",
                    f"in command:/entrypoint: across {files} compose file(s)"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} reference(s) compose will eat:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  A command that still runs is not a command that still "
              "works. The\n  executor's backoff loop ran its body zero "
              "times for its whole life\n  because `$BACKOFF_SCHEDULE` "
              "became the empty string at parse time.")
        return 1
    print("PASS: every `$` in every command reaches whoever wrote it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
