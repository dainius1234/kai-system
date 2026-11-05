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
from dataclasses import dataclass
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


# ── SUBJECT COMPLETENESS ──────────────────────────────────────────────
#
# THE INVARIANT THIS GATE IS BUILT AROUND:
#
#     derive the authoritative enforcing surface
#         -> establish that the SUBJECT IS COMPLETE
#             -> only then adjudicate portability
#
# A portability verdict is a statement about a subject. If the subject
# could not be fully obtained, no verdict about it is available — not
# "pass", not "fail", but REFUSE. That is R11 (no subject, no
# observation) and I-1 (fail closed on a missing input) applied to the
# thing this gate actually reasons over.
#
# There are three ways the subject can be incomplete, and they are ONE
# PREREQUISITE wearing three faces:
#
#     NO_AUTHORITATIVE_ROOT        the derivation produced nothing
#     ROOT_NAMED_BUT_UNRESOLVABLE  a workflow or Makefile names an
#                                  enforcing script that resolves to no
#                                  file in this repository
#     SUBJECT_NAMED_BUT_UNOPENABLE a resolved member of the closure
#                                  cannot be opened
#
# INC-2026-09-17-25 is what it costs to treat them as unrelated. Two of
# the three refused and the third was PRINTED AND THEN ADMITTED: the
# shipped gate certified PASS and exited 0 while an authoritative
# enforcing root named by a workflow had resolved to nothing. The
# relationship between the three existed only in prose, so the one that
# was written last simply did not inherit it.
#
# They are therefore not three `if` statements in `main()`. They are
# gaps recorded against ONE object, by ONE function, and consumed by
# ONE refusal branch. Adding a fourth way to be incomplete means adding
# a gap kind and a place that records it — it cannot mean forgetting to
# add a fourth conditional, because there are no conditionals to
# forget.

NO_AUTHORITATIVE_ROOT = "NO_AUTHORITATIVE_ROOT"
ROOT_NAMED_BUT_UNRESOLVABLE = "ROOT_NAMED_BUT_UNRESOLVABLE"
SUBJECT_NAMED_BUT_UNOPENABLE = "SUBJECT_NAMED_BUT_UNOPENABLE"

#: Human wording per gap kind. Kept beside the constant deliberately:
#: this is presentation, not policy, and nothing decides on it.
_GAP_WORDING = {
    NO_AUTHORITATIVE_ROOT:
        "the derivation produced no authoritative enforcing root",
    ROOT_NAMED_BUT_UNRESOLVABLE:
        "named as an enforcing script but resolves to no file here",
    SUBJECT_NAMED_BUT_UNOPENABLE:
        "resolved into the closure but could not be opened",
}


@dataclass(frozen=True)
class Subject:
    """What this gate adjudicates, and whether it was fully obtained.

    `gaps` is the whole completeness verdict. An empty `gaps` is the
    ONLY state in which a portability verdict may be issued, and
    `complete` is the single place that says so.
    """

    surface: str
    roots: Tuple[Path, ...]
    modules: Tuple[Path, ...]
    gaps: Tuple[Tuple[str, str], ...]

    @property
    def complete(self) -> bool:
        return not self.gaps


def closure(roots: List[Path]) -> Tuple[List[Path], List[Path]]:
    """Transitive STATIC import closure, plus any member that is absent.

    Returns (inspected, unopenable). Dynamic imports are not followed:
    resolving them would need execution, and executing an arbitrary
    repository module to decide whether it is portable is a worse idea
    than under-reporting. The boundary is printed with the result.
    """
    seen: Set[Path] = set()
    unopenable: Set[Path] = set()
    queue = list(roots)
    while queue:
        current = queue.pop()
        if current in seen or current in unopenable:
            continue
        if not current.is_file():
            unopenable.add(current)
            continue
        seen.add(current)
        for name in _imports(current):
            resolved = module_file(name)
            if resolved is not None and resolved not in seen:
                queue.append(resolved)
    return sorted(seen), sorted(unopenable)


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


def derive_subject(explicit: Optional[List[str]] = None) -> Subject:
    """Derive the subject AND its completeness in one place.

    Every way of failing to obtain the subject is recorded as a gap
    here. No caller re-derives completeness, and no caller may reach a
    verdict without consulting `Subject.complete`.
    """
    gaps: List[Tuple[str, str]] = []

    if explicit:
        surface = "explicit subject"
        roots: List[Path] = []
        for raw in explicit:
            candidate = Path(raw) if Path(raw).is_absolute() else REPO / raw
            # An explicitly named subject that does not exist is the
            # same prerequisite failure as a derived one that does not
            # resolve. The operator naming it does not make it present.
            if candidate.is_file():
                roots.append(candidate)
            else:
                gaps.append((ROOT_NAMED_BUT_UNRESOLVABLE, raw))
    else:
        surface = "workflow-and-Make enforcement surface"
        roots, unresolved = enforcing_roots()
        for name in unresolved:
            gaps.append((ROOT_NAMED_BUT_UNRESOLVABLE, name))

    if not roots:
        gaps.append((NO_AUTHORITATIVE_ROOT, surface))
        return Subject(surface, (), (), tuple(gaps))

    modules, unopenable = closure(roots)
    for path in unopenable:
        gaps.append((SUBJECT_NAMED_BUT_UNOPENABLE, str(path)))

    return Subject(surface, tuple(roots), tuple(modules), tuple(gaps))


def adjudicate(subject: Subject) -> List[Dict[str, object]]:
    """Portability findings over a subject. Callers must check `complete`.

    Refuses to run at all on an incomplete subject rather than
    returning an empty finding list that a caller could mistake for a
    clean result — the shape that produced INC-2026-09-17-25.
    """
    if not subject.complete:
        raise ValueError(
            "adjudicate() called on an incomplete subject: %d gap(s)"
            % len(subject.gaps))
    findings: List[Dict[str, object]] = []
    for module in subject.modules:
        findings.extend(scan_module(module))
    return findings


def _print_header(subject: Subject, findings: int) -> None:
    print("Operational portability — %d roots, %d modules, %d finding(s)"
          % (len(subject.roots), len(subject.modules), findings))
    print("  proposition        no developer-checkout or ephemeral-session")
    print("                     filesystem dependency on the %s"
          % subject.surface)
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", default=None,
                        help="qualify an explicit subject instead of the "
                             "derived enforcing surface (use-time preflight)")
    args = parser.parse_args(argv)

    subject = derive_subject(args.root)

    # THE INVARIANT, AS ONE BRANCH. Completeness is established before a
    # verdict is even attempted, and every way of being incomplete
    # arrives here through the same door.
    if not subject.complete:
        _print_header(subject, 0)
        print("")
        print("  SUBJECT INCOMPLETE — %d gap(s). REFUSED." % len(subject.gaps))
        print("  A portability verdict is a statement about a subject. This")
        print("  subject was not fully obtained, so no verdict about it is")
        print("  available — not PASS, not FAIL (R11, I-1).")
        for kind, detail in subject.gaps:
            print("    %-28s %s" % (kind, detail))
            print("    %-28s %s" % ("", _GAP_WORDING[kind]))
        return 1

    findings = adjudicate(subject)
    _print_header(subject, len(findings))
    print("  SUBJECT COMPLETE — every derived authoritative root resolved "
          "and opened.")

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
