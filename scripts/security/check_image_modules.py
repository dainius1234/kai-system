#!/usr/bin/env python3
"""Every module a service imports is in the image that runs it.

`agentic/Dockerfile` copied ten files by name:

    COPY agentic/adversary.py ./
    COPY agentic/app.py ./
    COPY agentic/conviction.py ./
    ... seven more

into a directory holding thirty-seven modules. `app.py` imports
twenty-seven of the ones the list omitted, so the container died at
import on every boot it has ever had:

    File "/app/app.py", line 27, in <module>
        from system_fsm import KaiEvent as SysEvent, ...
    ModuleNotFoundError: No module named 'system_fsm'

It blocked nothing until 2026-08-06, because until that day nothing had
ever got far enough to start it. When it finally did, `agentic` was the
last service standing between CI and thirteen steps that had never run.

**The list-beside-the-thing pattern, fourteenth venue.** The remedy is
the same one it has been every time: state the denominator, and derive
it from the tree rather than from a list somebody must remember to
update. `COPY agentic/ ./` cannot go stale.

How this decides
----------------

Rooted at what the image *actually runs* — the `.py` named by its
`CMD`/`ENTRYPOINT`, including the `uvicorn app:app` spelling — then
walking imports transitively through modules that are siblings in the
service directory. Anything reachable that no `COPY` puts in the image
is a `ModuleNotFoundError` waiting for its first boot.

Two false positives were caught while calibrating this, and both are
handled rather than tolerated:

  * `agentic/Dockerfile.introspect` runs `introspect_app.py`, not
    `app.py`. Rooting every image at `app.py` reported 34 phantom
    misses. The entry point comes from the Dockerfile.
  * `vault-sync/Dockerfile` does `COPY vault-sync/ ./`, so `parser`,
    `mapper` and `watcher` are all present. A directory COPY brings in
    every module under it.

Exit 0 = every image contains the modules its entry point reaches.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_EXCLUDED = {".git", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", "site-packages"}

_COPY = re.compile(r"\s*COPY\s+(?:--\S+\s+)*(\S+)\s+(\S+)\s*$")
_RUNS = re.compile(r"\s*(?:CMD|ENTRYPOINT)\s+(.*)")


def dockerfiles(root: Path = None) -> List[Path]:
    """Every Dockerfile, from a walk."""
    root = root or REPO
    return sorted(p for p in root.rglob("Dockerfile*")
                  if p.is_file()
                  and not any(part in _EXCLUDED for part in p.parts))


def _tokens(rest: str) -> List[str]:
    """The words of a CMD/ENTRYPOINT, exec-form or shell-form."""
    rest = rest.strip()
    if rest.startswith("["):
        try:
            return [str(t) for t in json.loads(rest)]
        except json.JSONDecodeError:
            pass
    return rest.split()


def entry_points(text: str, service: Path) -> Set[Path]:
    """The `.py` files this image actually starts.

    Rooting at `app.py` regardless is what produced 34 phantom findings
    against `Dockerfile.introspect`, which runs `introspect_app.py`.
    """
    out: Set[Path] = set()
    for line in text.splitlines():
        match = _RUNS.match(line)
        if not match:
            continue
        for token in _tokens(match.group(1)):
            token = token.strip('",')
            if token.endswith(".py") and (service / token).exists():
                out.add(service / token)
            if ":" in token:                 # `uvicorn app:app`
                module = token.split(":", 1)[0]
                candidate = service / f"{module}.py"
                if candidate.exists():
                    out.add(candidate)
    return out


def copied_modules(text: str, root: Path = None) -> Set[str]:
    """Module names the image receives.

    A directory COPY brings in every module under it — missing that is
    what made `vault-sync` look broken when it is not.
    """
    root = root or REPO
    names: Set[str] = set()
    for line in text.splitlines():
        match = _COPY.match(line)
        if not match:
            continue
        source = match.group(1)
        path = root / source
        if source.endswith(".py"):
            names.add(Path(source).stem)
        elif path.is_dir():
            names |= {f.stem for f in path.glob("*.py")}
            names.add(path.name)
    return names


def root_packages(root: Path = None) -> Set[str]:
    """Importable packages at the repository root, e.g. `common`.

    Derived from `__init__.py`, not listed. These are the *other* thing
    a service image has to be given, and the first version of this gate
    could not see them at all — see `reachable`.
    """
    root = root or REPO
    return {p.name for p in root.iterdir()
            if p.is_dir() and (p / "__init__.py").exists()
            and p.name not in _EXCLUDED}


def _local_imports(pyfile: Path, known: Set[str]) -> Set[str]:
    """Modules this file imports that the image has to supply.

    `known` is sibling modules **and** repo-root packages. Relative
    imports are skipped — a package-relative import means the whole
    package came along.
    """
    out: Set[str] = set()
    try:
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in known:
                    out.add(root)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            root = node.module.split(".")[0]
            if root in known:
                out.add(root)
    return out


def reachable(roots: Set[Path], service: Path,
              packages: Set[str] = None) -> Set[str]:
    """Modules reachable from the entry points, transitively.

    **Siblings and repo-root packages.** This considered only

        siblings = {p.stem for p in service.glob("*.py")}

    — the service's own directory, and nothing else. So a service
    importing `common.http_hygiene` needed `common/` in its image and
    this gate could not see the need, because `common` was not a
    sibling. The gate is named *"every Python service image contains the
    modules it imports"* and its universe was one directory: the
    systemic finding again, in the gate written this morning to catch
    the systemic finding.

    It cost a CI run to find, on 2026-08-06, when the full profile
    finally started backup-service:

        backup-service-1 | from common.http_hygiene import pooled_client
        backup-service-1 | ModuleNotFoundError: No module named 'common'

    while this printed `PASS: all 49 Python service image(s) contain the
    modules they import`. Three images were affected — backup-service,
    broker-bridge, cortex — and it reported none of them.

    A root package is not recursed into: `COPY common/ ./common/` brings
    the whole directory, so its internal imports are satisfied by
    construction.
    """
    packages = root_packages() if packages is None else packages
    siblings = {p.stem for p in service.glob("*.py")}
    known = siblings | packages
    need: Set[str] = set()
    frontier = list(roots)
    seen = set(roots)
    while frontier:
        for module in _local_imports(frontier.pop(), known):
            if module in need:
                continue
            need.add(module)
            if module in packages:
                continue          # copied wholesale; nothing more to trace
            nxt = service / f"{module}.py"
            if nxt.exists() and nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    return need


def findings_in(text: str, service: Path, origin: str,
                root: Path = None) -> List[str]:
    """Modules the entry point reaches that no COPY puts in the image."""
    roots = entry_points(text, service)
    if not roots:
        return []                  # not a Python service image
    missing = sorted(reachable(roots, service) - copied_modules(text, root))
    if not missing:
        return []
    names = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
    return [f"{origin}: runs "
            f"{', '.join(sorted(p.name for p in roots))} but the image is "
            f"missing {len(missing)} module(s) its own code imports "
            f"({names}). The container raises ModuleNotFoundError at "
            f"import and never starts. Replace the per-file COPY list "
            f"with the directory — a list has to be remembered, a "
            f"directory cannot go stale."]


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, Python service images inspected, Dockerfiles read)."""
    root = root or REPO
    paths = dockerfiles(root)
    if not paths:
        # I-1: zero inputs is a finding, not a pass.
        return ([f"{root}: no Dockerfile found — this gate inspected "
                 f"nothing and must not report success"], 0, 0)
    findings: List[str] = []
    services = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        service = path.parent
        if not entry_points(text, service):
            continue
        services += 1
        findings.extend(findings_in(
            text, service, str(path.relative_to(root)), root))
    return findings, services, len(paths)


def main() -> int:
    findings, services, files = audit()

    print(inspected(services, "Python service image(s)",
                    f"out of {files} Dockerfiles"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} image(s) missing their own modules:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  agentic's Dockerfile named ten files by hand against a "
              "directory of\n  thirty-seven, and app.py imported "
              "twenty-seven of the ones it missed.\n  It died at import on "
              "every boot it ever had.")
        return 1
    print(f"PASS: all {services} Python service image(s) contain the "
          f"modules they import.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
