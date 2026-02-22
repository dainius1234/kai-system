from __future__ import annotations

import os

vat_threshold = float(os.getenv("VAT_THRESHOLD", "85000"))
income = float(os.getenv("CURRENT_INCOME", "0"))
if income < vat_threshold:
    print("Advice: Do not add VAT yet (below threshold).")
print("invoice.pdf generated (stub)")
