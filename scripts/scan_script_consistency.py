from __future__ import annotations

from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1] / "scripts"
issues: list[str] = []

for path in sorted(root.glob("*")):
    if path.is_dir():
        continue
    name = path.name
    if not re.fullmatch(r"[a-z0-9_\-.]+", name):
        issues.append(f"bad filename: {name}")
    text = path.read_text(encoding="utf-8", errors="ignore")
    if name.endswith(".sh"):
        shebang_ok = text.startswith("#!/usr/bin/env sh") or text.startswith("#!/usr/bin/env bash")
        if not shebang_ok:
            issues.append(f"{name}: missing standard shell shebang")
        strict_mode_ok = ("set -eu" in text) or ("set -euo pipefail" in text)
        if not strict_mode_ok:
            issues.append(f"{name}: missing strict shell mode")
    if name.endswith(".py"):
        if "from __future__ import annotations" not in text:
            issues.append(f"{name}: missing future annotations header")

if issues:
    print("SCRIPT CONSISTENCY FAILED")
    for i in issues:
        print(i)
    sys.exit(1)

print("SCRIPT CONSISTENCY OK")
