"""Tests for `check_unreachable_bindings` — a lock never once taken.

`memu-core/app.py` had this, and had had it for as long as the writer
branch has existed:

    if self._tv_lock_path.exists():
        import socket
        existing = self._tv_lock_path.read_text().strip()
        raise RuntimeError(...)
    self._tv_lock_path.write_text(f"{socket.gethostname()}:{os.getpid()}")

The `if` body ends in `raise`. Control therefore never flows from the
`import` to the `write_text`, and every execution that reaches
`write_text` has `socket` unbound:

    UnboundLocalError: cannot access local variable 'socket'

The crash was the cheap part. The expensive part is what it means: the
writer branch had **never completed in any deployment**, so
`.writer.lock` was never written and the single-writer guarantee that
block exists to provide had never once been in force. A guarantee that
has never executed is indistinguishable from one that was never needed.

Verified against the real file: `git show e38695f:memu-core/app.py`
produces exactly one finding, and the fixed tree produces none.

Everything here is synthetic except the last two, which read the tree.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.security import check_unreachable_bindings as gate  # noqa: E402

passed = 0
failed = 0
EXPECTED_SCENARIOS = 12
executed: list = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL: {name}" + (f" — {detail}" if detail else ""))


def scenario(name: str) -> None:
    executed.append(name)


def found(src: str) -> list:
    return gate.findings_in(ast.parse(src), "x.py")


# ── the defect itself, in the shape it actually had ──────────────────

REAL = """
def __init__(self):
    if self._lock.exists():
        import socket
        raise RuntimeError("held")
    self._lock.write_text(socket.gethostname())
"""


def test_the_real_defect_is_caught() -> None:
    scenario("real defect caught")
    f = found(REAL)
    check("it is reported", len(f) == 1, str(f))
    check("the name is given", f and "`socket`" in f[0], str(f))
    check("the function is named", f and "__init__()" in f[0], str(f))
    check("and the remedy is spelled out",
          f and "Move the import above the `if`" in f[0], str(f))


def test_the_fixed_form_passes() -> None:
    scenario("fixed form passes")
    fixed = """
def __init__(self):
    import socket
    if self._lock.exists():
        raise RuntimeError("held")
    self._lock.write_text(socket.gethostname())
"""
    check("nothing reported", found(fixed) == [], str(found(fixed)))


def test_every_terminal_statement_counts() -> None:
    """`raise` is the one that bit. `return`, `continue` and `break`
    end the block just as absolutely, and a gate that knew only `raise`
    would report PASS over three quarters of its own subject."""
    scenario("all terminal statements")
    for stmt, wrap in (("return None", "    {}\n"),
                       ("continue", None), ("break", None)):
        if wrap:
            src = (f"def f():\n    if c:\n        import json\n"
                   f"        {stmt}\n    json.dumps(1)\n")
        else:
            src = (f"def f():\n    for i in r:\n        if c:\n"
                   f"            import json\n            {stmt}\n"
                   f"    json.dumps(1)\n")
        check(f"`{stmt}` is treated as terminal", len(found(src)) == 1,
              f"{stmt}: {found(src)}")


# ── what must NOT be reported ────────────────────────────────────────

def test_a_try_except_import_guard_is_not_a_defect() -> None:
    """The idiomatic optional-dependency guard. Flagging it would report
    a defect in code that works, which is the failure mode this
    repository avoids hardest — 20 sites in this tree use it."""
    scenario("try/except guard allowed")
    src = """
def f():
    try:
        import numpy as np
    except ImportError:
        return None
    return np.zeros(3)
"""
    check("not reported", found(src) == [], str(found(src)))


def test_a_branch_that_falls_through_is_not_a_defect() -> None:
    """If the body can fall through, the use may well be reachable and
    the binding may well have happened. Not decidable here, so silent."""
    scenario("fall-through allowed")
    src = """
def f(c):
    if c:
        import json
        x = 1
    return json.dumps(x)
"""
    check("not reported", found(src) == [], str(found(src)))


def test_a_name_used_only_inside_the_branch_is_not_a_defect() -> None:
    """`import socket` beside a message that formats with it is fine —
    the binding and the use are on the same path."""
    scenario("use inside branch allowed")
    src = """
def f():
    if c:
        import socket
        raise RuntimeError(socket.gethostname())
    return 1
"""
    check("not reported", found(src) == [], str(found(src)))


def test_a_star_import_is_not_guessed_at() -> None:
    """`from x import *` binds names this gate cannot enumerate.
    Reporting on a guess would be worse than saying nothing."""
    scenario("star import skipped")
    src = """
def f():
    if c:
        from mod import *
        raise RuntimeError("x")
    return anything
"""
    check("not reported", found(src) == [], str(found(src)))


def test_an_aliased_import_is_tracked_by_its_alias() -> None:
    scenario("alias tracked")
    src = """
def f():
    if c:
        import numpy as np
        raise RuntimeError("x")
    return np.zeros(3)
"""
    f = found(src)
    check("reported", len(f) == 1, str(f))
    check("under the alias, not the module", f and "`np`" in f[0], str(f))


def test_a_dotted_import_is_tracked_by_its_root() -> None:
    """`import os.path` binds `os`, not `os.path`."""
    scenario("dotted import")
    src = """
def f():
    if c:
        import os.path
        raise RuntimeError("x")
    return os.getcwd()
"""
    f = found(src)
    check("reported under the root name", f and "`os`" in f[0], str(f))


# ── I-1: zero inputs is not a pass ───────────────────────────────────

def test_a_tree_with_no_python_refuses() -> None:
    scenario("zero inputs refuses")
    with tempfile.TemporaryDirectory() as tmp:
        findings, functions, files = gate.audit(Path(tmp))
        check("it fails rather than passing", findings != [], str(findings))
        check("and says it inspected nothing",
              any("inspected nothing" in f for f in findings), str(findings))
        check("with a zero denominator", (functions, files) == (0, 0),
              f"{functions}, {files}")


# ── the real tree ────────────────────────────────────────────────────

def test_the_repository_passes_today() -> None:
    scenario("repository passes")
    findings, functions, files = gate.audit()
    check("no unreachable bindings", findings == [], str(findings))
    check("across a real number of functions", functions > 1000, str(functions))
    check("in a real number of files", files > 300, str(files))


def test_the_file_list_comes_from_a_walk() -> None:
    """A hand-written list of files to check is the defect this
    programme keeps finding. The denominator is the tree."""
    scenario("walk not list")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "pkg").mkdir()
        (root / "pkg" / "deep").mkdir()
        for rel in ("a.py", "pkg/b.py", "pkg/deep/c.py"):
            (root / rel).write_text("x = 1\n")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "d.py").write_text("x = 1\n")
        got = gate.source_files(root)
        check("it recurses", len(got) == 3, str([p.name for p in got]))
        check("and skips caches",
              all("__pycache__" not in str(p) for p in got),
              str([str(p) for p in got]))


def run_all() -> None:
    test_the_real_defect_is_caught()
    test_the_fixed_form_passes()
    test_every_terminal_statement_counts()
    test_a_try_except_import_guard_is_not_a_defect()
    test_a_branch_that_falls_through_is_not_a_defect()
    test_a_name_used_only_inside_the_branch_is_not_a_defect()
    test_a_star_import_is_not_guessed_at()
    test_an_aliased_import_is_tracked_by_its_alias()
    test_a_dotted_import_is_tracked_by_its_root()
    test_a_tree_with_no_python_refuses()
    test_the_repository_passes_today()
    test_the_file_list_comes_from_a_walk()

    check(f"all {EXPECTED_SCENARIOS} scenarios ran",
          len(executed) == EXPECTED_SCENARIOS,
          f"{len(executed)} ran: {executed}")
    check("no scenario ran twice", len(set(executed)) == len(executed),
          str(executed))


if __name__ == "__main__":
    run_all()
    print(f"\n{'=' * 60}")
    print(f"Unreachable Binding Tests: {passed} passed, {failed} failed")
    if failed:
        print("EXIT GATE: FAIL")
        sys.exit(1)
    print("EXIT GATE: PASS")
