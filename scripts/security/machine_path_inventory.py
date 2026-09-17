"""Whole-repository inventory of machine- and session-bound paths.

**THIS IS A REPORT. IT NEVER GATES.** It exits 0 at any occurrence count,
rising or falling, and it is declared `kind=REPORT` in `gate_registry.py`
with no `pending_wiring`: no promotion path is encoded, because promoting
a report to a gate is a governance decision and not a code change.

Authorised by D376 §3 as corrected by D377. It replaces the VISIBILITY
half of `test_p1_p4_enhancements.py::test_no_developer_home_paths`, which
is retired as a blocking control in the same tranche. **Nothing that test
could see stops being visible here.** The opposite: its population was
`.py` files under a five-name skip set; this one is every tracked text
file in the repository.

WHAT THIS ANSWERS, AND WHAT IT DOES NOT
---------------------------------------
It answers **"where does a machine- or session-bound path string appear?"**
That is an OCCURRENCE claim about text. It is not a claim that any of them
is an operational defect, and it must never be read as one: of the seven
occurrences that made up RC-1, four were genuine machine-bound execution
dependencies, two were deliberate calibration fixture data, and one was a
sentence in a docstring describing a defect that had already been
repaired. A control that cannot tell those apart should not be allowed to
stop a build, which is why this one cannot.

**It does not claim the repository is portable.** That proposition is
wider than any lexical predicate can support. `check_operational_
portability.py` makes the narrow enforceable claim; this file makes the
wide advisory one; neither impersonates the other.

THE DENOMINATOR IS NOT HIDDEN
-----------------------------
Every run prints the tracked total, how many files were inspected as
text, how many were not, and the exact rule that decided. **The exclusion
set is EMPTY and is printed as empty.** A file that cannot be decoded is
named, not silently counted as zero occurrences — "we found nothing in
it" and "we could not look in it" are different answers and this report
distinguishes them.

Population comes from `git ls-files`, which is the tree's own definition
of membership. That choice removes a skip list rather than maintaining
one: the old detector carried `{_archive, .venv, __pycache__,
node_modules, .git}`, and `_archive` does not exist in this repository at
all — a name kept beside the thing it describes, which is the defect
shape this file is part of removing.

SELF-INSPECTION
---------------
The patterns are assembled from fragments at runtime so this file's own
source is not a false occurrence. **This file is inside the population
and is not excluded from it.** The technique is the one the retired
detector used, and `check_bind_mount_portability.py` records the trap
being fallen into five separate times: prose about a forbidden string is
still the forbidden string.

Exit code: **0, always.**
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

#: Bumped whenever the predicate changes. It travels in the report so a
#: count from one version is never silently compared with another.
PREDICATE_VERSION = "1.0.0"

#: Assembled from fragments: see SELF-INSPECTION above.
_HOME = "/" + "home" + "/"
_USERS = "/" + "Users" + "/"
_SCRATCH_PREFIXES = ("/" + "tmp" + "/claude-", "/" + "var" + "/folders/")

#: The two classes this report names. Anything else absolute is
#: NOT_CLASSIFIED and is deliberately not called portable: a fixed
#: container path may still be machine-specific, and
#: `check_bind_mount_portability.py` documents exactly that happening.
USER_HOME = "USER_HOME"
SESSION_SCRATCH = "SESSION_SCRATCH"

_ROOT_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    (USER_HOME, re.compile(re.escape(_HOME) + r"[A-Za-z0-9._-]+")),
    (USER_HOME, re.compile(re.escape(_USERS) + r"[A-Za-z0-9._-]+")),
) + tuple(
    (SESSION_SCRATCH, re.compile(re.escape(p) + r"[A-Za-z0-9._/-]*"))
    for p in _SCRATCH_PREFIXES
)

# Syntactic classes. UNKNOWN is a real answer and is used wherever the
# class cannot be derived, never as a synonym for "none".
STRING_CONSTANT = "STRING_CONSTANT"
DOCSTRING = "DOCSTRING"
COMMENT = "COMMENT"
CALL_ARGUMENT = "CALL_ARGUMENT"
ASSIGNMENT_TARGET = "ASSIGNMENT_TARGET"
NON_PYTHON_TEXT = "NON_PYTHON_TEXT"
UNKNOWN = "UNKNOWN"

#: Callables whose argument is a filesystem subject. Used only to answer
#: `flows_to_filesystem_call` for a DIRECT argument; this report performs
#: no dataflow, and says UNKNOWN rather than guessing.
_FS_CALLS = {
    "Path", "open", "chdir", "join", "exists", "read_text", "write_text",
    "copytree", "copy", "copy2", "rmtree", "move", "insert", "append",
}

#: Read once per file; 8 KiB is enough to find a NUL in any real binary
#: and cheap enough to run over the whole tree.
_SNIFF_BYTES = 8192


def tracked_files() -> List[Path]:
    """Population: the tree's own membership list, not a directory walk.

    `git ls-files` is derived from the subject under inspection. A
    `rglob` plus a skip set is a list maintained beside the thing, and it
    was already wrong: the retired detector skipped `_archive`, which
    does not exist here.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO,
        capture_output=True, text=True, check=True).stdout
    return [REPO / p for p in out.split("\0") if p]


def is_text(path: Path) -> Tuple[bool, str]:
    """Decidability rule, declared rather than assumed.

    A file is text when its first 8 KiB contain no NUL byte AND the whole
    file decodes as UTF-8. Returns the verdict and the reason, because a
    file we could not read must be reported as unread rather than as
    clean.
    """
    try:
        head = path.open("rb").read(_SNIFF_BYTES)
    except OSError as exc:
        return False, f"unreadable: {exc.__class__.__name__}"
    if b"\0" in head:
        return False, "NUL byte in first 8192 bytes"
    try:
        path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return False, "does not decode as UTF-8"
    return True, "text"


def _py_classes(text: str) -> Dict[int, Tuple[str, Optional[bool]]]:
    """Per-line syntactic class for a Python source file.

    Line-granular on purpose. A finer answer would need column tracking
    through two different parsers, and the honest coarse answer is worth
    more than a precise one that is sometimes wrong.

    `flows_to_filesystem_call` is True only for a literal that is a
    DIRECT argument to a filesystem call. For an assignment it is None —
    reported as UNKNOWN — because the value may or may not reach such a
    call, and deciding that is dataflow, which is
    `check_operational_portability.py`'s job and not this report's.
    """
    classes: Dict[int, Tuple[str, Optional[bool]]] = {}

    # Comments are not in the AST; tokenize is the only source for them.
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                classes[tok.start[0]] = (COMMENT, False)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return classes

    docstring_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                first = body[0].value
                for ln in range(first.lineno,
                                (first.end_lineno or first.lineno) + 1):
                    docstring_lines.add(ln)

    fs_arg_lines = set()
    assign_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in _FS_CALLS:
                for arg in list(node.args) + [k.value for k in node.keywords]:
                    if isinstance(arg, ast.Constant) and isinstance(
                            arg.value, str):
                        fs_arg_lines.add(arg.lineno)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                assign_lines.add(value.lineno)

    for ln in docstring_lines:
        classes.setdefault(ln, (DOCSTRING, False))
    for ln in fs_arg_lines:
        classes[ln] = (CALL_ARGUMENT, True)
    for ln in assign_lines:
        classes.setdefault(ln, (ASSIGNMENT_TARGET, None))
    return classes


def scan(paths: List[Path]) -> Dict[str, object]:
    """Inventory every tracked file. Returns the whole report as data."""
    occurrences: List[Dict[str, object]] = []
    not_text: List[Dict[str, str]] = []
    text_count = 0

    for path in sorted(paths):
        rel = path.relative_to(REPO).as_posix()
        if not path.exists():           # tracked but deleted in worktree
            not_text.append({"path": rel, "reason": "tracked, absent on disk"})
            continue
        ok, reason = is_text(path)
        if not ok:
            not_text.append({"path": rel, "reason": reason})
            continue
        text_count += 1
        body = path.read_text(encoding="utf-8")
        classes = _py_classes(body) if path.suffix == ".py" else {}
        for num, line in enumerate(body.splitlines(), 1):
            for root_class, pattern in _ROOT_PATTERNS:
                for match in pattern.finditer(line):
                    if path.suffix == ".py":
                        klass, flows = classes.get(num, (STRING_CONSTANT, None))
                    else:
                        klass, flows = NON_PYTHON_TEXT, None
                    occurrences.append({
                        "path": rel,
                        "line": num,
                        "column": match.start() + 1,
                        "matched_root_class": root_class,
                        "matched_text": match.group(0),
                        "syntactic_class": klass,
                        "flows_to_filesystem_call":
                            UNKNOWN if flows is None else flows,
                    })

    by_class: Dict[str, int] = {}
    for occ in occurrences:
        key = str(occ["syntactic_class"])
        by_class[key] = by_class.get(key, 0) + 1

    return {
        "predicate_version": PREDICATE_VERSION,
        "tracked_total": len(paths),
        "text_inspected": text_count,
        "non_text_not_scanned": not_text,
        "text_rule": ("no NUL byte in first %d bytes AND decodes as UTF-8"
                      % _SNIFF_BYTES),
        "excluded_population": [],
        "occurrences": occurrences,
        "by_syntactic_class": by_class,
    }


def subject() -> Tuple[str, str]:
    """Exact subject. UNKNOWN rather than a guess if git cannot answer."""
    def rev(what: str) -> str:
        try:
            return subprocess.run(
                ["git", "rev-parse", what], cwd=REPO,
                capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return UNKNOWN
    return rev("HEAD"), rev("HEAD^{tree}")


def render(report: Dict[str, object]) -> str:
    """Human-readable form. The denominator line is the contract."""
    head, tree = subject()
    lines = [
        "Machine-path inventory — %d tracked, %d text, %d non-text, "
        "%d occurrences" % (
            report["tracked_total"], report["text_inspected"],
            len(report["non_text_not_scanned"]), len(report["occurrences"])),
        "  subject            %s" % head,
        "  tree               %s" % tree,
        "  predicate version  %s" % report["predicate_version"],
        "  text rule          %s" % report["text_rule"],
        "  excluded           EMPTY — no path is skipped by policy",
        "",
        "  THIS IS A REPORT. It makes an OCCURRENCE claim only, never a",
        "  portability claim, and it cannot fail a build.",
        "",
    ]
    if report["non_text_not_scanned"]:
        lines.append("  NOT TEXT-SCANNED (named, not counted as zero):")
        for entry in report["non_text_not_scanned"]:
            lines.append("    %-60s %s" % (entry["path"], entry["reason"]))
        lines.append("")
    if report["by_syntactic_class"]:
        lines.append("  by syntactic class:")
        for key in sorted(report["by_syntactic_class"]):
            lines.append("    %-20s %d" % (key, report["by_syntactic_class"][key]))
        lines.append("")
    if report["occurrences"]:
        lines.append("  occurrences:")
        for occ in report["occurrences"]:
            lines.append(
                "    %s:%s:%s  %-16s %-18s flows=%s"
                % (occ["path"], occ["line"], occ["column"],
                   occ["matched_root_class"], occ["syntactic_class"],
                   occ["flows_to_filesystem_call"]))
    else:
        lines.append("  no occurrences in the inspected population")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true",
                        help="emit the report as JSON")
    args = parser.parse_args(argv)
    report = scan(tracked_files())
    if args.json:
        head, tree = subject()
        report["subject"] = head
        report["tree"] = tree
        print(json.dumps(report, indent=1, sort_keys=True))
    else:
        print(render(report))
    return 0            # ALWAYS. This report does not gate.


if __name__ == "__main__":
    sys.exit(main())
