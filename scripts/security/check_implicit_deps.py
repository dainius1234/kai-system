#!/usr/bin/env python3
"""FastAPI's implicit dependencies, which no import statement names.

`document-parser` crash-looped in every deployment:

    Traceback (most recent call last):
      File "/app/app.py", line 370, in <module>
        @app.post("/parse")
      ...
        ensure_multipart_is_installed()
    RuntimeError: Form data requires "python-multipart" to be installed.

`@app.post("/parse")` takes `file: UploadFile = File(...)`. FastAPI needs
`python-multipart` to build that route and raises **at import**, so the
container has never started. Neither had the **dashboard**, which has
four such routes and the same omission — the operator's entire interface.

Why no test could see it
------------------------

`scripts/test_dashboard.py` passes, and always has. CI installs every
`requirements.txt` in the tree into **one** environment:

    find . -maxdepth 3 -name requirements.txt ... | while read req; do
      pip install -r "$req"

`browser-agent`, `perception/audio`, `perception/vision` and
`screen-capture` all list `python-multipart`, so by the time the
dashboard's tests run it is installed — by somebody else. The unit
tests are structurally incapable of catching a missing *per-service*
dependency. Only the container can, and the container only speaks when
something else lets the bring-up get that far.

`document-parser` never even blocked the bring-up: `dashboard` waits on
it with `condition: service_started`, which a container in a restart
loop satisfies perfectly well.

The rule
--------

These libraries are needed at import time and named by no `import`
statement — the dependency is expressed as a *usage*, which is exactly
why it goes missing:

    UploadFile / File(...) / Form(...)   ->  python-multipart
    EmailStr                             ->  email-validator
    SessionMiddleware                    ->  itsdangerous
    ORJSONResponse                       ->  orjson
    UJSONResponse                        ->  ujson

A service directory whose Python uses one and whose `requirements.txt`
does not list the package is a container that will not start.

Exit 0 = every implicit dependency is declared where it is used.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_EXCLUDED = {".git", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", "site-packages", "scripts", "tests"}

#: usage pattern -> the distribution that must be installed for it.
RULES: Tuple[Tuple[str, "re.Pattern[str]", str], ...] = (
    ("python-multipart",
     re.compile(r"\bUploadFile\b|\bFile\s*\(|\bForm\s*\("),
     "FastAPI builds multipart routes at import and raises without it"),
    ("email-validator",
     re.compile(r"\bEmailStr\b"),
     "pydantic validates EmailStr at model definition time"),
    ("itsdangerous",
     re.compile(r"\bSessionMiddleware\b"),
     "starlette's session middleware signs cookies with it"),
    ("orjson",
     re.compile(r"\bORJSONResponse\b"),
     "the response class imports it eagerly"),
    ("ujson",
     re.compile(r"\bUJSONResponse\b"),
     "the response class imports it eagerly"),
)


def requirement_files(root: Path = None) -> Dict[Path, List[Path]]:
    """directory -> every `requirements*.txt` in it, from a walk.

    A list would have omitted `document-parser` for the same reason
    `docker-compose.full.yml` did — nobody thinks of the quiet ones.

    Returns the *files*, not just the directories. An earlier version
    returned directories and then did `if not (d / "requirements.txt")
    .exists(): continue`, which the meta-check flagged as boundary
    blindness (I-1) and was right to: a service pinning its deps in
    `requirements-dev.txt` alone would have been silently skipped, and
    the gate would have reported success over a service it never read.
    Deriving the files directly means there is nothing left to be
    absent.
    """
    root = root or REPO
    out: Dict[Path, List[Path]] = {}
    for req in sorted(root.rglob("requirements*.txt")):
        if any(part in _EXCLUDED for part in req.parts):
            continue
        if req.parent == root:
            continue                    # the repo's own top-level deps
        out.setdefault(req.parent, []).append(req)
    return out


def service_dirs(root: Path = None) -> List[Path]:
    """Every directory that pins dependencies."""
    return sorted(requirement_files(root))


def declared(requirements: str) -> Set[str]:
    """Distribution names a requirements file pins, normalised.

    `python_multipart`, `Python-Multipart` and `python-multipart` are one
    package; PEP 503 says compare them that way.
    """
    names: Set[str] = set()
    for line in requirements.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = re.split(r"[<>=!~\[; ]", line, 1)[0].strip()
        if name:
            names.add(re.sub(r"[-_.]+", "-", name).lower())
    return names


def findings_in(sources: Dict[str, str], requirements: str,
                origin: str) -> List[str]:
    """Implicit dependencies used in `sources` but absent from the file."""
    have = declared(requirements)
    findings: List[str] = []
    for package, pattern, why in RULES:
        if package in have:
            continue
        for filename, text in sorted(sources.items()):
            match = pattern.search(text)
            if not match:
                continue
            findings.append(
                f"{origin}: {filename} uses `{match.group(0).strip('(')}` "
                f"but requirements.txt does not list `{package}` — "
                f"{why}. The container raises at import and never "
                f"starts; no unit test can see this, because CI "
                f"installs every requirements.txt into one environment.")
            break                       # one finding per package per dir
    return findings


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, service directories inspected, rules applied)."""
    root = root or REPO
    by_dir = requirement_files(root)
    if not by_dir:
        # I-1: zero inputs is a finding, not a pass.
        return ([f"{root}: no service requirements.txt found — this gate "
                 f"inspected nothing and must not report success"], 0, 0)
    findings: List[str] = []
    for directory, reqs in sorted(by_dir.items()):
        # Every requirements file in the directory counts as declaring —
        # a dependency pinned in `requirements-dev.txt` is still pinned.
        pinned = "\n".join(r.read_text(encoding="utf-8") for r in reqs)
        sources = {}
        for path in sorted(directory.glob("*.py")):
            try:
                sources[path.name] = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
        if not sources:
            continue                    # nothing to use a dependency
        findings.extend(findings_in(
            sources, pinned, str(directory.relative_to(root))))
    return findings, len(by_dir), len(RULES)


def main() -> int:
    findings, dirs, rules = audit()

    print(inspected(dirs, "service director(ies)",
                    f"against {rules} implicit-dependency rules"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} undeclared implicit dependency(ies):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  document-parser and the dashboard both carried this and "
              "both had\n  never started in any deployment. The dashboard is "
              "the operator's\n  entire interface.")
        return 1
    print("PASS: every implicit dependency is declared where it is used.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
