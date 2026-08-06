#!/usr/bin/env python3
"""A name bound on one path and used on another.

Twice in one constructor, in one day, both found only when a container
told us:

    AttributeError: 'TurboVecStore' object has no attribute '_pool_lock'
    UnboundLocalError: cannot access local variable 'socket'

The second is the one this gate detects, because it is decidable
statically. `memu-core/app.py` had:

    if self._tv_lock_path.exists():
        import socket                       # <- bound only here
        raise RuntimeError(...)             # <- and this always raises
    self._tv_lock_path.write_text(
        f"{socket.gethostname()}:{os.getpid()}")   # <- used here

The `if` body ends in `raise`, so by construction control never flows
from the import to the use. The import is reachable; the *binding* it
creates is not reachable from the site that needs it. Every execution
that gets to `write_text` has an unbound `socket`.

The cost was not the crash. It was that **the writer branch had never
completed in any deployment** — `.writer.lock` was never written, so the
single-writer guarantee that block exists to provide had never once been
in force, and nothing said so. A guarantee that has never executed looks
exactly like a guarantee that has never been needed.

Scope, stated because that is the rule here: this catches the decidable
case only — an `import` inside an `if`/`elif` body whose last statement
is `raise`, `return`, `continue` or `break`, where the imported name is
also referenced outside that block. It deliberately does **not** flag
`try: import x / except ImportError:`, which is idiomatic and correct,
nor an import in a branch that falls through, where the use may well be
reachable. Pointed at the tree before the fix it returned exactly one
site; after, zero. A survey with false positives invites fixes to
working code, so the narrow rule is the right one.

Exit 0 = every import binds on a path that reaches its uses.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_EXCLUDED = {".git", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", "site-packages"}

#: Statements after which control does not continue to the next line.
_TERMINAL = (ast.Raise, ast.Return, ast.Continue, ast.Break)


def source_files(root: Path = None) -> List[Path]:
    """Every .py file, from a walk rather than a list."""
    root = root or REPO
    return sorted(p for p in root.rglob("*.py")
                  if not any(part in _EXCLUDED for part in p.parts))


def _terminates(body: List[ast.stmt]) -> bool:
    """True when this block cannot fall through to what follows it."""
    return bool(body) and isinstance(body[-1], _TERMINAL)


def _linenos(node: ast.AST) -> Set[int]:
    return {n.lineno for n in ast.walk(node) if hasattr(n, "lineno")}


def _imported_names(body: List[ast.stmt]) -> Set[str]:
    """The names an import statement binds, as they are referenced."""
    names: Set[str] = set()
    for stmt in body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                if alias.name == "*":
                    continue           # a star import binds unknown names
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def findings_in(tree: ast.AST, origin: str) -> List[str]:
    """Bindings made on a path that cannot reach their use."""
    findings: List[str] = []
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for fn in functions:
        for node in ast.walk(fn):
            if not isinstance(node, ast.If) or not _terminates(node.body):
                continue
            bound = _imported_names(node.body)
            if not bound:
                continue
            block = _linenos(node)
            for name in sorted(bound):
                used = sorted({n.lineno for n in ast.walk(fn)
                               if isinstance(n, ast.Name) and n.id == name
                               and n.lineno not in block})
                if not used:
                    continue
                findings.append(
                    f"{origin}:{node.lineno}: `{fn.name}()` imports "
                    f"`{name}` inside an `if` whose body always "
                    f"{type(node.body[-1]).__name__.lower()}s, and uses "
                    f"it at line{'s' if len(used) > 1 else ''} "
                    f"{', '.join(str(u) for u in used)}. Control never "
                    f"flows from the binding to the use — every "
                    f"execution reaching those lines has `{name}` "
                    f"unbound. Move the import above the `if`.")
    return findings


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, functions inspected, files read)."""
    root = root or REPO
    paths = source_files(root)
    if not paths:
        # I-1: zero inputs is a finding, not a pass.
        return ([f"{root}: no .py files found — this gate inspected "
                 f"nothing and must not report success"], 0, 0)
    findings: List[str] = []
    functions = 0
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            # A file the interpreter cannot read is not this gate's
            # finding — `lint-blocking` fails on it first, by design.
            continue
        functions += sum(
            1 for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
        findings.extend(findings_in(tree, str(path.relative_to(root))))
    return findings, functions, len(paths)


def main() -> int:
    findings, functions, files = audit()

    print(inspected(functions, "function(s)", f"across {files} Python files"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} unreachable binding(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  memu-core's writer branch carried one of these and had "
              "therefore never\n  completed in any deployment — the "
              "single-writer lock it exists to take\n  was never once "
              "taken, and nothing said so.")
        return 1
    print("PASS: every import binds on a path that reaches its uses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
