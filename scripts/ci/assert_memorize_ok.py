#!/usr/bin/env python3
"""Assert a memu-core /memory/memorize response reports success.

Extracted from `core-tests.yml`, where it lived as a `python3 -c "` block
whose body began at column 0 — which terminates the enclosing `run: |`
scalar, so the workflow did not parse under a standard YAML parser.

A workflow that does not parse runs nothing, and running nothing is
indistinguishable from having no failures. The script cannot simply be
indented, because the indentation would land inside the Python string and
raise `IndentationError`; a file is the fix.

Reads the response on stdin. Exit 0 = ok, 1 = anything else.
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw)
    except ValueError:
        print(f"FAIL: memorize returned non-JSON: {raw[:200]!r}")
        return 1
    if payload.get("status") != "ok":
        print(f"FAIL: memorize returned {payload}")
        return 1
    print("OK: memorize succeeded after introspect stop")
    return 0


if __name__ == "__main__":
    sys.exit(main())
