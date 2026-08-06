#!/usr/bin/env python3
"""A test does not let the environment choose its own privilege level.

`scripts/security_fuzz_upload.py` opens with:

    _os.environ.setdefault("KAI_DASHBOARD_ROLE", "keeper")

which reads as *"this suite runs as keeper"*. It does not.
`setdefault` means **unless somebody else already decided**, and on
2026-08-06 somebody did: `KAI_DASHBOARD_ROLE: operator` went into
`core-tests.yml` at job scope so the live smoke could authenticate.

`/api/upload` requires `Scope.WRITE_EXTERNAL`. `keeper` has it;
`operator` does not. Eight of the suite's fourteen tests went from
asserting *upload validation* to asserting *authorisation*, and failed:

    FAILED test_one_byte_over_limit_returns_413
    FAILED test_oversized_payload_returns_413
    FAILED test_no_filename_rejected
    ... five more

The tests were not wrong and the endpoint was not broken. The suite's
**identity** was decided by an environment variable it did not control,
so what it verified depended on who ran it. Under CI it had only ever
run at whatever privilege happened to be ambient — which, until that
commit, was the permissive path.

That is the systemic finding again, aimed at a test rather than a check:
its scope — *which privilege am I exercising* — was not stated but
inherited, and it reported success over a question it never asked.

The rule
--------

In a test file, a variable naming an identity, a credential or a
privilege must be **assigned**, not `setdefault`. Everything else
(paths, feature flags, cache locations) may still `setdefault`: a test
that tolerates an ambient `/tmp` path is not thereby testing something
different.

Names treated as identity-bearing: any containing TOKEN, ROLE,
IDENTITY, SECRET, HMAC, PRINCIPAL, AUTH or KEY.

Exit 0 = every test pins the identity it claims to run as.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO))

from scripts.security.gate_inputs import inspected  # noqa: E402

_EXCLUDED = {".git", ".venv", "venv", "_archive", "node_modules",
             "__pycache__", "site-packages"}

#: A variable whose value decides *who the caller is*, not merely where
#: a file lives. Substring match, upper-cased.
_IDENTITY = re.compile(
    r"TOKEN|ROLE|IDENTITY|SECRET|HMAC|PRINCIPAL|AUTH|KEY")


def test_files(root: Path = None) -> List[Path]:
    """Every test module, from a walk.

    `conftest.py` counts: it sets the environment every suite inherits,
    so an ambient-dependent identity there reaches all of them.
    """
    root = root or REPO
    out = []
    for path in root.rglob("*.py"):
        if any(part in _EXCLUDED for part in path.parts):
            continue
        name = path.name
        if (name.startswith("test_") or name == "conftest.py"
                or "_fuzz_" in name or name.endswith("_test.py")):
            out.append(path)
    return sorted(out)


def findings_in(tree: ast.AST, origin: str) -> List[str]:
    """`environ.setdefault` calls that let the environment pick a privilege."""
    findings: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "setdefault"):
            continue
        # …the receiver must be `<something>.environ`
        owner = func.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == "environ"):
            continue
        if not node.args:
            continue
        key = node.args[0]
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
            continue
        name = key.value
        if not _IDENTITY.search(name.upper()):
            continue
        findings.append(
            f"{origin}:{node.lineno}: `environ.setdefault({name!r}, …)` "
            f"lets the surrounding environment choose this suite's "
            f"privilege. `setdefault` means 'unless somebody already "
            f"decided', and what this test verifies then depends on who "
            f"runs it. Assign it: `environ[{name!r}] = …`.")
    return findings


def audit(root: Path = None) -> Tuple[List[str], int, int]:
    """Return (findings, environ.setdefault calls seen, test files read)."""
    root = root or REPO
    paths = test_files(root)
    if not paths:
        # I-1: zero inputs is a finding, not a pass.
        return ([f"{root}: no test files found — this gate inspected "
                 f"nothing and must not report success"], 0, 0)
    findings: List[str] = []
    calls = 0
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setdefault"
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "environ"):
                calls += 1
        findings.extend(findings_in(tree, str(path.relative_to(root))))
    return findings, calls, len(paths)


def main() -> int:
    findings, calls, files = audit()

    print(inspected(calls, "environ.setdefault call(s)",
                    f"across {files} test files"))
    print()

    if findings:
        print(f"FAIL: {len(findings)} ambient-privilege setdefault(s):\n")
        for line in findings:
            print(f"  - {line}")
        print("\n  `security_fuzz_upload` believed it ran as keeper and ran "
              "as whatever\n  the environment said. Eight of its fourteen "
              "tests silently changed\n  from checking upload validation to "
              "checking authorisation.")
        return 1
    print(f"PASS: every test pins the identity it claims to run as "
          f"({calls} setdefault call(s) inspected, none identity-bearing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
