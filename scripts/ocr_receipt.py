from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None


AMOUNT_RE = re.compile(r"(\d+[\.,]\d{2})")


def parse_amount(text: str) -> float:
    matches = AMOUNT_RE.findall(text)
    if not matches:
        return 0.0
    return float(matches[-1].replace(",", "."))


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: ocr_receipt.py <image> <output_csv>")
    image = Path(sys.argv[1])
    out = Path(sys.argv[2])
    text = ""
    if pytesseract is not None and Image is not None and image.exists():
        text = pytesseract.image_to_string(Image.open(image))
    amount = parse_amount(text)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "amount", "raw_text"])
        w.writeheader()
        w.writerow({"source": str(image), "amount": f"{amount:.2f}", "raw_text": text.strip()[:2000]})
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
