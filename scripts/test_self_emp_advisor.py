from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.self_emp_advisor import advise

income = 48000
expenses = ["buy laptop 1200", "car service 300"]
out = advise(income_total=income, expenses_lines=expenses)
joined = "\n".join(out).lower()

assert "april" in joined
assert "car service" in joined
assert "135" in joined
assert "mtd" in joined
print("self-employment advisor tests passed")
