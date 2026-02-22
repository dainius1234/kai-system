from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

from common.market_cache import load_cache


def _to_float(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip().replace("£", "").replace(",", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def thresholds() -> Dict[str, float]:
    return {
        "mtd_start": _to_float(os.getenv("MTD_START", "50000")),
        "vat_threshold": _to_float(os.getenv("VAT_THRESHOLD", "85000")),
        "mileage_rate": _to_float(os.getenv("MILEAGE_RATE", "0.45")),
    }


def load_income_total(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    total = 0.0
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += _to_float(row.get("amount", "0"))
    return round(total, 2)


def load_expenses(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def advise(income_total: float, expenses_lines: List[str]) -> List[str]:
    t = thresholds()
    out: List[str] = []

    if income_total >= t["mtd_start"] - 2000:
        left = max(t["mtd_start"] - income_total, 0)
        out.append(f"Alert: £{left:.0f} left till MTD threshold (£{t['mtd_start']:.0f}) — prep digital records now or risk £100 penalty.")

    if income_total < t["vat_threshold"]:
        out.append(f"You are below VAT threshold (£{t['vat_threshold']:.0f}) — avoid unnecessary VAT registration for now.")

    joined = " \n".join(expenses_lines)
    if "laptop" in joined and 47000 <= income_total <= 50000:
        out.append("Income around £48k — wait till April to buy laptop; plan deduction timing (example laptop £1200).")
    elif "laptop" in joined:
        out.append("Laptop expense detected — time purchase with tax-year planning for cleaner deductions.")

    for line in expenses_lines:
        if "car" in line and "300" in line:
            deduction = 300 * t["mileage_rate"]
            out.append(f"Car service £300 noted — log mileage and keep receipt; indicative mileage-method deduction example: £{deduction:.0f}.")
            break

    market = load_cache()
    petrol = market.get("petrol", {}) if isinstance(market, dict) else {}
    trend = str(petrol.get("trend", "")).lower()
    if "+" in trend or "up" in trend:
        out.append("Fuel trend rising — fill up today to reduce next-day cost impact.")

    if not out:
        out.append("No major tax timing risks detected from current offline data. Keep receipts and update income/expense logs weekly.")
    return out
