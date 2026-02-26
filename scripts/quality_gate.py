#!/usr/bin/env python3
"""
Script Quality Gate: Scan for stubs, TODOs, and missing docstrings in scripts/.
Fails with exit code 1 if any issues are found.
"""
import os
import sys
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"

failures = []

for pyfile in SCRIPTS.glob("*.py"):
    if pyfile.name.startswith("test_"):
        continue  # skip test scripts
    src = pyfile.read_text(encoding="utf-8")
    # Check for TODO or stub
    if "TODO" in src or "pass  # stub" in src or "NotImplementedError" in src:
        failures.append(f"{pyfile}: Contains TODO/stub/NotImplementedError")
    # Check for module docstring
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
