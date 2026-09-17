"""Machine- and session-bound dependencies on the enforcing execution surface.

**THE PROPOSITION, AND NOTHING WIDER:**

    No detected developer-checkout or ephemeral-session filesystem
    dependency exists on the workflow-and-Make enforcement surface
    derived by `execution_surface.py`.

Authorised by D376 §3 as corrected by D377. This gate takes over the
BLOCKING half of `test_p1_p4_enhancements.py::test_no_developer_home_paths`,
which is retired in the same tranche; `machine_path_inventory.py` takes
over the visibility half and reports every occurrence this gate declines
to adjudicate.

WHY THE PROPOSITION IS NARROW ON PURPOSE
----------------------------------------
"The repository is portable" is wider than any static predicate we know
how to enforce, so this gate does not say it. It says something smaller
and true: *on the part of the repository that CI actually executes, no
path names one contributor's checkout or one session's scratch
directory.*

The retired detector conflated two different questions — *does this
string appear?* and *does this code depend on that path?* — and RC-1 is
what that conflation costs. Of its seven occurrences, four were genuine
execution dependencies, two were deliberate calibration fixtures proving
a classifier discriminates absolute paths, and one was a docstring
sentence describing an already-repaired defect. A blocking control that
cannot separate those suppressed sixteen unrelated live-stack checks.

WHAT THIS GATE DOES NOT MEASURE
-------------------------------
The shared helper derives roots from workflow `run:` steps and Makefile
recipes. It does not derive:

    pytest collection · `python -m` module entrypoints · shell
    entrypoints · Docker ENTRYPOINT/CMD · Compose `command:` ·
    runtime/service startup roots

**Those root classes are UNMEASURED BY THIS GATE and are printed as
unmeasured in every report.** Extending the shared helper to cover them
needs independent fixtures per class and is deliberately outside this
tranche. A gate that quietly left them implicit would be claiming a
surface it never inspected — the defect this file exists to prevent.

FALSE-NEGATIVE BOUNDARY, PRINTED EVERY RUN
------------------------------------------
Runtime-composed f-strings · paths assembled across function boundaries ·
values read from data files or the environment alone · `getattr` and
dynamic attribute access · shell strings built at runtime · anything
reached through a non-static import. This gate under-reports by design,
because a portability survey with false positives sends people to break
working code.

ABSOLUTE IS NOT THE SAME AS MACHINE-BOUND
-----------------------------------------
Only two root classes are findings: a developer checkout and an ephemeral
session scratch directory. A fixed deployment path is NOT declared
portable here — `check_bind_mount_portability.py` records
`/var/log/sovereign` booting healthy and reading nothing on every host
but one, so "fixed prefix" and "portable" are different properties.
Everything outside the two finding classes is NOT_CLASSIFIED, is left to
the inventory report, and is not adjudicated by this gate in either
direction.

Exit codes:
  0  no finding on the derived enforcing surface
  1  a finding, or the surface could not be derived
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from scripts.security.execution_surface import (  # noqa: E402
    discover_policy_check,
    discover_workflows,
    enforcing_elsewhere,
    module_file,
)

PREDICATE_VERSION = "1.0.0"

#: Assembled from fragments so this file's own source is not an instance
#: of what it forbids. The trap is recorded five times in
#: check_bind_mount_portability.py: prose about a forbidden string is
#: still the forbidden string.
_HOME = "/" + "home" + "/"
_USERS = "/" + "Users" + "/"
_SCRATCH = ("/" + "tmp" + "/claude-", "/" + "var" + "/folders/")

USER_HOME = "USER_HOME"
SESSION_SCRATCH = "SESSION_SCRATCH"

#: Root classes this gate ADJUDICATES. Anything else absolute is
#: NOT_CLASSIFIED — see ABSOLUTE IS NOT THE SAME AS MACHINE-BOUND.
_UNMEASURED_ROOT_CLASSES = (
    "pytest collection",
    "python -m module entrypoints",
    "shell entrypoints",
    "Docker ENTRYPOINT/CMD",
    "Compose command:",
    "runtime/service startup roots",
)

#: Calls whose string argument is a filesystem or import-resolution
#: subject. `insert`/`append` are here for the sys.path class and are
#: confirmed against the receiver before they count.
_FS_FUNCS = {"Path", "open", "chdir", "copytree", "copy", "copy2", "rmtree",
             "move", "read_text", "write_text", "read_bytes", "write_bytes",
             "mkdir", "listdir", "walk", "glob", "exists", "join"}

DEP_LITERAL_IN_CALL = "LITERAL_IN_FILESYSTEM_CALL"
DEP_BOUND_NAME = "MODULE_LEVEL_BINDING_USED_IN_FILESYSTEM_CALL"
DEP_SYS_PATH = "SYS_PATH_IMPORT_RESOLUTION"
DEP_DEFAULT_ARG = "DEFAULT_ARGUMENT_VALUE"
DEP_ENV_FALLBACK = "ENVIRONMENT_DEFAULT_FALLBACK"


def classify_root(text: str) -> Optional[str]:
    """Which finding class, if any, does this literal belong to?

    `USER_HOME` means a DEVELOPER CHECKOUT, not every path beneath a home
    directory. A container or service home such as `/home/appuser/app` is
    a deployment path and is a known-negative: it is the same on every
    host that runs the image, which is the property that matters. The
    discriminator is whether the literal names this repository's own
    checkout — its directory name, or a path used as a repository or
    import root.
    """
    for prefix in _SCRATCH:
        if prefix in text:
            return SESSION_SCRATCH
    for prefix in (_HOME, _USERS):
        if prefix not in text:
            continue
        if REPO.name in text:
            return USER_HOME
    return None


def _is_fs_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name):
        return node.func.id in _FS_FUNCS
    if isinstance(node.func, ast.Attribute):
        if node.func.attr in ("insert", "append"):
            recv = node.func.value
            return (isinstance(recv, ast.Attribute) and recv.attr == "path") or \
                   (isinstance(recv, ast.Name) and recv.id == "path")
        return node.func.attr in _FS_FUNCS
    return False


def _is_sys_path_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in ("insert", "append"):
        return False
    recv = node.func.value
    return isinstance(recv, ast.Attribute) and recv.attr == "path"


def _is_env_get(node: ast.Call) -> bool:
    return (isinstance(node.func, ast.Attribute)
            and node.func.attr in ("get", "getenv"))


def _str_constants(node: ast.AST):
    """Every string constant directly inside `node`, one level of nesting."""
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child


def scan_module(path: Path) -> List[Dict[str, object]]:
    """Dependency classes 1-5 for one file. AST, never text matching."""
    findings: List[Dict[str, object]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return findings

    rel = path.relative_to(REPO).as_posix()

    def record(node: ast.AST, text: str, klass: str, dep: str) -> None:
        findings.append({
            "path": rel, "line": getattr(node, "lineno", 0),
            "matched_text": text, "root_class": klass,
            "dependency_class": dep,
        })

    # Class 2 needs the module-level bindings first: NAME -> literal.
    bound: Dict[str, Tuple[str, int]] = {}
    for stmt in tree.body:
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            value = stmt.value
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            literal = None
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                literal = value.value
            elif isinstance(value, ast.Call) and _is_fs_call(value):
                for const in _str_constants(value):
                    literal = const.value
                    break
            if literal is None:
                continue
            for target in targets:
                if isinstance(target, ast.Name):
                    bound[target.id] = (literal, stmt.lineno)

    used_names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        if _is_sys_path_call(node):
            for const in _str_constants(node):
                klass = classify_root(const.value)
                if klass:
                    record(const, const.value, klass, DEP_SYS_PATH)
            for arg in node.args:
                if isinstance(arg, ast.Call):
                    for const in _str_constants(arg):
                        klass = classify_root(const.value)
                        if klass:
                            record(const, const.value, klass, DEP_SYS_PATH)
                for name in (n.id for n in ast.walk(arg)
                             if isinstance(n, ast.Name)):
                    used_names.add(name)
            continue

        if _is_env_get(node) and len(node.args) >= 2:
            fallback = node.args[1]
            if isinstance(fallback, ast.Constant) and isinstance(
                    fallback.value, str):
                klass = classify_root(fallback.value)
                if klass:
                    record(fallback, fallback.value, klass, DEP_ENV_FALLBACK)
            continue

        if _is_fs_call(node):
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    klass = classify_root(arg.value)
                    if klass:
                        record(arg, arg.value, klass, DEP_LITERAL_IN_CALL)
                elif isinstance(arg, ast.Name):
                    used_names.add(arg.id)

    for name in sorted(used_names):
        if name not in bound:
            continue
        literal, lineno = bound[name]
        klass = classify_root(literal)
        if klass:
            findings.append({
                "path": rel, "line": lineno, "matched_text": literal,
                "root_class": klass, "dependency_class": DEP_BOUND_NAME,
            })

    # Class 4: a default argument value that is itself machine-bound.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        defaults = list(node.args.defaults) + [
            d for d in node.args.kw_defaults if d is not None]
        for default in defaults:
            for const in _str_constants(default):
                klass = classify_root(const.value)
                if klass:
                    record(const, const.value, klass, DEP_DEFAULT_ARG)

    seen, unique = set(), []
    for finding in findings:
        key = (finding["path"], finding["line"], finding["dependency_class"])
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


def _imports(path: Path) -> List[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, OSError):
        return []
    names: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
    return names


def closure(roots: List[Path]) -> List[Path]:
    """Transitive STATIC import closure, plus any root that is absent.

    Returns (inspected, missing). A missing subject is never dropped:
    the caller refuses rather than reporting a pass over code it could
    not open.

    Following them would need execution, and executing an arbitrary
    repository module to decide whether it is portable is a worse idea
    than under-reporting. The boundary is printed with the result.
    """
    seen: Set[Path] = set()
    missing: Set[Path] = set()
    queue = list(roots)
    while queue:
        current = queue.pop()
        if current in seen or current in missing:
            continue
        # I-1: a subject that is not there is NOT a subject with nothing
        # wrong in it. Skipping it would shrink the closure silently and
        # let this gate report PASS over code it never opened — absence
        # reading as correctness, which is the defect class this gate is
        # part of removing. It is returned and the caller refuses.
        if not current.is_file():
            missing.add(current)
            continue
        seen.add(current)
        for name in _imports(current):
            resolved = module_file(name)
            if resolved is not None and resolved not in seen:
                queue.append(resolved)
    return sorted(seen), sorted(missing)


def enforcing_roots() -> Tuple[List[Path], List[str]]:
    """Roots derived by the SHARED mechanism. No second parser here."""
    names: Set[str] = set()
    names |= set(discover_workflows().keys())
    names |= set(discover_policy_check())
    names |= set(enforcing_elsewhere())
    roots, unresolved = [], []
    for name in sorted(names):
        resolved = module_file(name)
        if resolved is None:
            unresolved.append(name)
        else:
            roots.append(resolved)
    return roots, unresolved


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None,
                        help="qualify an explicit subject instead of the "
                             "derived enforcing surface (use-time preflight)")
    args = parser.parse_args(argv)

    if args.root:
        roots = [Path(r) if Path(r).is_absolute() else REPO / r
                 for r in args.root]
        unresolved: List[str] = []
        surface = "explicit subject"
    else:
        roots, unresolved = enforcing_roots()
        surface = "workflow-and-Make enforcement surface"

    # I-1: no subject means no observation. An empty surface is a
    # failure to derive, never a clean result.
    if not roots:
        print("Operational portability — 0 roots, 0 modules, 0 finding(s)")
        print("  REFUSED: no execution root could be derived. A gate with no "
              "subject cannot report a pass (R11).")
        return 1

    modules, missing = closure(roots)

    # I-1 again, at the boundary that matters: if any named subject
    # could not be opened, this gate cannot certify the surface. A
    # pass here would mean 'nothing wrong in the files I managed to
    # read', while the sentence printed says something wider.
    if missing:
        print('Operational portability — %d roots, %d modules, %d finding(s)'
              % (len(roots), len(modules), len(missing)))
        print('  REFUSED: %d named subject(s) could not be opened. A gate\n'
              '  cannot certify a surface it did not read (R11, I-1):'
              % len(missing))
        for path in missing:
            print('    %s' % path)
        return 1
    findings: List[Dict[str, object]] = []
    for module in modules:
        findings.extend(scan_module(module))

    print("Operational portability — %d roots, %d modules, %d finding(s)"
          % (len(roots), len(modules), len(findings)))
    print("  proposition        no developer-checkout or ephemeral-session")
    print("                     filesystem dependency on the %s" % surface)
    print("  predicate version  %s" % PREDICATE_VERSION)
    print("  roots derived by   scripts/security/execution_surface.py "
          "(shared with check_gate_registry.py)")
    print("  NOT MEASURED BY THIS GATE — root classes the shared mechanism")
    print("  does not derive, so this gate makes no claim about them:")
    for klass in _UNMEASURED_ROOT_CLASSES:
        print("    - %s" % klass)
    print("  FALSE-NEGATIVE BOUNDARY: runtime-composed f-strings; paths")
    print("  assembled across function boundaries; env-only values; getattr")
    print("  and dynamic attribute access; runtime-built shell strings;")
    print("  anything reached through a non-static import.")
    print("  This gate does NOT claim the repository is portable.")
    if unresolved:
        print("  unresolved root names (reported, not silently dropped): %s"
              % ", ".join(unresolved))

    if not findings:
        print("  PASS: no machine- or session-bound dependency on the "
              "derived surface.")
        return 0

    print("")
    for finding in findings:
        print("  FINDING %s:%s  %s  %s"
              % (finding["path"], finding["line"], finding["root_class"],
                 finding["dependency_class"]))
        print("          %s" % finding["matched_text"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
