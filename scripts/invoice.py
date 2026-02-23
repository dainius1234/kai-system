from __future__ import annotations

import argparse
import os
from pathlib import Path


def should_add_vat(income: float, threshold: float) -> bool:
    """Return ``True`` if income has reached the VAT threshold."""
    return income >= threshold


def generate_invoice_file(path: str, income: float, vat_applied: bool) -> None:
    """Create a simple invoice file at ``path``.

    This is intentionally lightweight; real code can switch to a
    PDF library such as :mod:`fpdf` or :mod:`reportlab` when needed.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(f"Income: {income}\n")
        f.write(f"VAT applied: {vat_applied}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple invoice and check VAT eligibility.")
    parser.add_argument("--income", type=float, help="Total income", default=float(os.getenv("CURRENT_INCOME", "0")))
    parser.add_argument("--threshold", type=float, help="VAT threshold", default=float(os.getenv("VAT_THRESHOLD", "85000")))
    parser.add_argument("--output", help="Output file path", default="invoice.pdf")
    args = parser.parse_args()

    vat = should_add_vat(args.income, args.threshold)
    if not vat:
        print("Advice: Do not add VAT yet (below threshold).")
    generate_invoice_file(args.output, args.income, vat)
    print(f"{args.output} generated.")


if __name__ == "__main__":
    main()
