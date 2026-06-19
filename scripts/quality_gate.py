#!/usr/bin/env python3
"""
Script Quality Gate: Scan for stubs, TODOs, and missing docstrings in scripts/.
Fails with exit code 1 if any issues are found.

TODO/stub detection only matches real markers (a comment starting with
"TODO" or "FIXME", an actual "pass  # stub" line, or a "raise
NotImplementedError" call) — not the word "TODO" appearing inside string
literals or prose, which previously produced false positives in scripts
that reference these markers as data (e.g. a stub-cleanup tool whose own
source mentions "TODO" as a string to search for).
"""
import ast
import re
import sys
import tokenize
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

# Files with a known, intentionally-incomplete feature, tracked outside this
# gate. Listing a file here only suppresses the stub/TODO check — it still
# must have a module docstring explaining what's missing and why.
KNOWN_STUBS = {
    "hse_rams.py": "RAMS.docx generation not yet implemented; needs python-docx "
    "and a RAMS template — tracked as a real feature gap, not an accident.",
}

_TODO_MARKER = re.compile(r"^(TODO|FIXME)\b", re.IGNORECASE)


def _raises_not_implemented(mod: ast.AST) -> bool:
    """True if the module actually executes `raise NotImplementedError(...)` somewhere."""
    for node in ast.walk(mod):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exc = node.exc
        name = exc.func if isinstance(exc, ast.Call) else exc
        if isinstance(name, ast.Name) and name.id == "NotImplementedError":
            return True
    return False


def _has_marker_comment(src: str) -> bool:
    """True if a real comment (not a string literal) is a TODO/FIXME/stub marker."""
    try:
        tokens = list(tokenize.generate_tokens(StringIO(src).readline))
    except tokenize.TokenizeError:
        return False
    for i, tok in enumerate(tokens):
        if tok.type != tokenize.COMMENT:
            continue
        text = tok.string.lstrip("#").strip()
        if _TODO_MARKER.match(text):
            return True
        if "stub" in text.lower():
            # only counts as a stub marker if it trails a bare `pass` on the same line
            prev = [t for t in tokens[:i] if t.start[0] == tok.start[0] and t.type == tokenize.NAME]
            if prev and prev[-1].string == "pass":
                return True
    return False


failures = []

for pyfile in sorted(SCRIPTS.glob("*.py")):
    if pyfile.name.startswith("test_"):
        continue  # skip test scripts
    src = pyfile.read_text(encoding="utf-8")

    if pyfile.name not in KNOWN_STUBS:
        try:
            mod = ast.parse(src)
        except Exception:
            mod = None
        has_stub_marker = (mod is not None and _raises_not_implemented(mod)) or _has_marker_comment(src)
        if has_stub_marker:
            failures.append(f"{pyfile}: Contains TODO/stub/NotImplementedError")

    try:
        mod = ast.parse(src)
        if not ast.get_docstring(mod):
            failures.append(f"{pyfile}: Missing module docstring")
    except Exception as e:
        failures.append(f"{pyfile}: Parse error: {e}")

if failures:
    print("Script quality gate failed:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
else:
    print("All scripts pass quality gate.")
