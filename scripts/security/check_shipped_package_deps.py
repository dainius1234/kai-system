#!/usr/bin/env python3
"""An image that ships a package must be able to import what it imports.

The defect
----------

`common/policy.py` opens with

    try:
        import yaml
        ...
    except Exception:
        import json
        def _load_yaml(path): ...   # json.loads on a YAML document

and `security/policy.yml` calls itself, in its own header, *"the single
source of truth — every runtime decision (tool-gate gating, verifier
verdicts, circuit breaker thresholds, quarantine triggers) reads from
this file."*

Thirty-five service images `COPY common/`. **None of them declared
pyyaml.** So in every one of those containers the import failed, the
fallback ran, `json.loads` met `version: "1.0.0"`, and the policy loaded
empty — every permission dropping to its most restrictive default.

Proven, not inferred, on 2026-08-06 when the sovereign profile started
tool-gate for the first time:

    JSONDecodeError: Expecting value: line 14 column 1 (char 13)
    POLICY FILE CORRUPT OR UNREADABLE — failing closed.

Today's pattern with the subject changed: not code that never executed,
but *configuration that was never loaded*.

The rule
--------

If an image copies a first-party package, the package's own third-party
imports are that image's dependencies. Copying the code does not copy
what the code needs.

`check_implicit_deps` asks a neighbouring question and answers it from a
hand-written list of five FastAPI/pydantic couplings — a list beside the
thing, which is why it could never have found this one. This derives
both sides: the packages an image copies, from its COPY lines; and what
those packages import, from their source.

Scope, stated — including what it does NOT catch
------------------------------------------------

**Module-level, unguarded imports only**, in the parts of the package
the image's entry point actually reaches. Those are the ones that run
the moment the module is imported and raise `ModuleNotFoundError` with
nothing to catch them.

**This rule would not have caught the pyyaml defect above.** That import
was inside `try: ... except:`, and saying otherwise would make this gate
a claim about its own origin story rather than about the code. Two
honest reasons for the narrowness:

  * `common/gpu_utils.py` does `import torch` inside a function inside a
    `try:`, and its fallback — a capability probe returning False — is
    correct. A rule broad enough to catch pyyaml also reports `torch`
    against `weather-service`, and acting on that means adding a
    two-gigabyte dependency to a weather service to satisfy a check.
    Measured: the broad rule produced over a hundred findings, of which
    one was real.
  * What actually distinguished the pyyaml case is that its *fallback
    was broken* — `json.loads` on a YAML document could only ever
    return `{}`. That is a property of one bug, not a rule a static
    check can generalise.

So the guarded-import class is left to a human, recorded rather than
pretended away, and this enforces the part that is decidable. A gate
with false positives sends people to break working code and buries the
one true finding — the failure mode this programme has now hit twice.

Exit 0 = every image installs the unguarded imports of the code it
ships.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_COPY_PKG = re.compile(r"^\s*COPY\s+(?:--\S+\s+)*([a-z0-9_]+)/\s+\S+\s*$",
                       re.M)

#: Distribution name for an import name where they differ. Import names
#: are what the source says; requirements files carry distributions.
_DIST = {
    "yaml": "pyyaml",
    "dateutil": "python-dateutil",
    "jwt": "pyjwt",
    "dotenv": "python-dotenv",
    "PIL": "pillow",
    "cv2": "opencv-python",
}


def first_party(root: Path = None) -> Set[str]:
    """Top-level names that are our own code, so never a dependency."""
    root = root or REPO
    return {p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")} | {
        p.stem for p in root.glob("*.py")}


def _imports_of(pyfile: Path, top_level_only: bool = False) -> Set[str]:
    """Dotted import names in one file, non-relative only.

    `top_level_only` restricts to statements in the module body — not
    inside a function, not inside `try:`. Those are the imports that run
    the instant the module is imported and raise `ModuleNotFoundError`
    with nothing to catch them. See `audit` for why the distinction is
    the whole gate.
    """
    out: Set[str] = set()
    try:
        tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return out
    nodes = tree.body if top_level_only else ast.walk(tree)
    for node in nodes:
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.add(node.module)
    return out


def reachable_third_party(entry: Path, pkg_name: str, pkg: Path,
                          own: Set[str]) -> Set[str]:
    """Third-party imports of the parts of `pkg` this entry point reaches.

    **Reachability, not the whole package.** The first draft took every
    import anywhere under `common/` and reported 100-odd findings —
    including `torch` and `tiktoken` against `weather-service`, because
    `common/llm.py` imports them and a weather service never touches it.
    Acting on that would have added a two-gigabyte dependency to a
    weather service to satisfy a check.

    That is the inverse defect and the worse one: a scope larger than
    reality reports failure over things that are right, sends people to
    break working code, and buries the one true finding. `common/policy`
    *is* reached — that is what makes pyyaml a real dependency and torch
    not.
    """
    needed: Set[str] = set()
    seen: Set[Path] = set()
    frontier = [entry]
    while frontier:
        current = frontier.pop()
        # No existence guard: every path that reaches this frontier has
        # already been checked. `entry` comes from `_entry_point`, which
        # returns None rather than a missing file; a submodule is only
        # appended when `sub.exists()`; and `__init__.py` is guaranteed
        # by `is_package`. A `not current.exists(): continue` here would
        # be a skip that cannot fire, which reads as tolerance for an
        # absence this gate would in fact never see.
        if current in seen:
            continue
        seen.add(current)
        for dotted in _imports_of(current):
            head = dotted.split(".")[0]
            if head == pkg_name:
                parts = dotted.split(".")
                sub = pkg.joinpath(*parts[1:]).with_suffix(".py")
                if sub.exists():
                    frontier.append(sub)
                else:
                    frontier.append(pkg / "__init__.py")
                continue
    # Second pass: only the reached package modules, and only their
    # module-level unguarded imports.
    for module in seen:
        if pkg not in module.parents:
            continue
        for dotted in _imports_of(module, top_level_only=True):
            head = dotted.split(".")[0]
            if head in own or head in sys.stdlib_module_names:
                continue
            needed.add(head)
    return needed


def declared(requirements: Path) -> Set[str]:
    """Distribution names a requirements file installs, lower-cased.

    A missing file returns an empty set and `audit` treats that as a
    finding rather than as nothing to check — an image that ships
    `common/` and installs nothing cannot import any of it. The first
    draft returned early here and `audit` skipped, which is I-1: absence
    reading as correctness, in a gate about absence.
    """
    if not requirements.exists():
        return set()
    out: Set[str] = set()
    for line in requirements.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        out.add(re.split(r"[<>=!\[; ]", line, 1)[0].strip().lower())
    return out


def _entry_point(dockerfile: Path, text: str):
    """The image's Python entry point, from its CMD/ENTRYPOINT.

    Reuses `check_image_modules.entry_points`, which already understands
    `uvicorn app:app` as well as `python app.py`. The first draft
    assumed `app.py`, which is a second, weaker copy of a question
    another gate had already answered properly.
    """
    from scripts.security.check_image_modules import entry_points
    roots = entry_points(text, dockerfile.parent)
    for candidate in sorted(roots):
        if candidate.exists():
            return candidate
    return None


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, images inspected, package copies seen)."""
    root = root or REPO
    own = first_party(root)
    findings: List[str] = []
    images = 0
    copies = 0
    cache: Dict[str, Set[str]] = {}

    dockerfiles = sorted(p for p in root.glob("*/Dockerfile*") if p.is_file())
    if not dockerfiles:
        # I-1: nothing to inspect is not a clean bill of health.
        return ([f"{root}: no service Dockerfiles found — this gate "
                 f"inspected nothing and must not report success"], 0, 0)

    for dockerfile in dockerfiles:
        text = dockerfile.read_text(encoding="utf-8")
        images += 1
        have = declared(dockerfile.parent / "requirements.txt")
        for pkg_name in _COPY_PKG.findall(text):
            pkg = root / pkg_name
            # Positive condition: a first-party package is a directory
            # with an `__init__.py`. Anything else is an ordinary
            # directory copy and not this gate's subject.
            is_package = pkg.is_dir() and (pkg / "__init__.py").exists()
            if is_package:
                copies += 1
            else:
                continue
            if not have:
                findings.append(
                    f"{dockerfile.relative_to(root)}: ships `{pkg_name}/` "
                    f"and installs nothing — there is no requirements.txt, "
                    f"so none of the package's imports can resolve.")
                continue
            entry = _entry_point(dockerfile, text)
            if entry is None:
                findings.append(
                    f"{dockerfile.relative_to(root)}: ships `{pkg_name}/` "
                    f"but no Python entry point could be read from its "
                    f"CMD/ENTRYPOINT, so nothing here was checked. That is "
                    f"not a clean image, it is an unread one.")
                continue
            key = f"{dockerfile.parent.name}:{pkg_name}"
            if key not in cache:
                cache[key] = reachable_third_party(entry, pkg_name, pkg, own)
            missing = sorted(
                _DIST.get(mod, mod) for mod in cache[key]
                if _DIST.get(mod, mod).lower() not in have)
            for dist in missing:
                findings.append(
                    f"{dockerfile.relative_to(root)}: ships `{pkg_name}/`, "
                    f"which imports `{dist}`, and its requirements.txt does "
                    f"not install it. Copying the code does not copy what "
                    f"the code needs.")
    return findings, images, copies


def main() -> int:
    findings, images, copies = audit()

    print(inspected(copies, "first-party package copy(ies)",
                    f"across {images} service image(s)"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} image(s) ship code they cannot "
              f"import:\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  35 images shipped `common/` without pyyaml, so "
              "`common/policy.py`\n  fell back to a loader that cannot "
              "parse YAML and every permission\n  took its most "
              "restrictive default. The policy file calls itself the\n  "
              "single source of truth; no container had ever read it.")
        return 1

    print(f"PASS: every image can import what the packages it ships "
          f"import ({copies} package copy(ies) inspected).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
