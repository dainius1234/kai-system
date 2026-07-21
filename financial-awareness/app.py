"""P29 — CIS-aware Financial Awareness service for UK construction subcontractors.

Endpoints:
  POST /finance/cis/record      — log a CIS payment received from a contractor
  GET  /finance/cis/summary     — YTD CIS deductions / net receipts
  POST /finance/invoice/generate — produce a CIS-compliant invoice payload
  GET  /finance/vat             — VAT position vs registration threshold
  GET  /finance/tax             — estimated Income Tax + Class 4 NI for the tax year
  GET  /finance/summary         — full financial overview
  GET  /health
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from common.runtime import setup_json_logger

# ── Config ────────────────────────────────────────────────────────────────────
FINANCE_ROOT = Path(os.getenv("FINANCE_ROOT", "/data/finance"))
CIS_RECORDS_FILE = FINANCE_ROOT / "cis_records.json"
PORT = int(os.getenv("PORT", "8063"))

# UK tax thresholds (overridable via env for future-proofing)
PERSONAL_ALLOWANCE   = float(os.getenv("UK_PERSONAL_ALLOWANCE",   "12570"))
BASIC_RATE_LIMIT     = float(os.getenv("UK_BASIC_RATE_LIMIT",      "50270"))
HIGHER_RATE_LIMIT    = float(os.getenv("UK_HIGHER_RATE_LIMIT",    "125140"))
MTD_THRESHOLD        = float(os.getenv("MTD_START",                "50000"))
VAT_THRESHOLD        = float(os.getenv("VAT_THRESHOLD",            "85000"))
MILEAGE_RATE         = float(os.getenv("MILEAGE_RATE",             "0.45"))

# CIS deduction rates
CIS_RATES: Dict[str, float] = {"registered": 0.20, "unregistered": 0.30, "gross": 0.00}

logger = setup_json_logger("financial-awareness", os.getenv("LOG_PATH", "/tmp/financial-awareness.json.log"))


# ── Persistence ───────────────────────────────────────────────────────────────
def _load_records() -> List[Dict[str, Any]]:
    if not CIS_RECORDS_FILE.exists():
        return []
    try:
        return json.loads(CIS_RECORDS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_records(records: List[Dict[str, Any]]) -> None:
    FINANCE_ROOT.mkdir(parents=True, exist_ok=True)
    CIS_RECORDS_FILE.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")


# ── Tax helpers ───────────────────────────────────────────────────────────────
def _income_tax(profit: float) -> float:
    """Estimate Income Tax on self-employment profit (England/Wales/NI rates 2024/25)."""
    taxable = max(profit - PERSONAL_ALLOWANCE, 0.0)
    basic   = min(taxable, BASIC_RATE_LIMIT - PERSONAL_ALLOWANCE)
    higher  = min(max(taxable - (BASIC_RATE_LIMIT - PERSONAL_ALLOWANCE), 0.0),
                  HIGHER_RATE_LIMIT - BASIC_RATE_LIMIT)
    additional = max(taxable - (HIGHER_RATE_LIMIT - PERSONAL_ALLOWANCE), 0.0)
    return round(basic * 0.20 + higher * 0.40 + additional * 0.45, 2)


def _class4_ni(profit: float) -> float:
    """Estimate Class 4 NI (2024/25 rates: 6% on £12,570–£50,270; 2% above)."""
    band1 = min(max(profit - 12570, 0.0), 50270 - 12570)
    band2 = max(profit - 50270, 0.0)
    return round(band1 * 0.06 + band2 * 0.02, 2)


def _ytd_income_from_records(records: List[Dict[str, Any]]) -> float:
    """Gross income YTD from CIS records (current tax year: April 6 → April 5)."""
    today = date.today()
    tax_year_start = date(today.year if today.month >= 4 and today.day >= 6 else today.year - 1, 4, 6)
    total = 0.0
    for r in records:
        try:
            rec_date = date.fromisoformat(r["date"])
        except Exception:
            continue
        if rec_date >= tax_year_start:
            total += float(r.get("gross_amount", 0))
    return round(total, 2)


def _ytd_deductions_from_records(records: List[Dict[str, Any]]) -> float:
    today = date.today()
    tax_year_start = date(today.year if today.month >= 4 and today.day >= 6 else today.year - 1, 4, 6)
    total = 0.0
    for r in records:
        try:
            rec_date = date.fromisoformat(r["date"])
        except Exception:
            continue
        if rec_date >= tax_year_start:
            total += float(r.get("deduction_amount", 0))
    return round(total, 2)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_app: FastAPI):
    FINANCE_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("financial-awareness starting on port %d, data root %s", PORT, FINANCE_ROOT)
    yield


app = FastAPI(title="financial-awareness", version="0.1.0", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────
class CISRecordRequest(BaseModel):
    """Log a CIS payment received from a contractor."""
    contractor_name: str
    contractor_utr: Optional[str] = None
    gross_amount: float
    materials_amount: float = 0.0          # materials are exempt from CIS deduction
    deduction_status: Literal["registered", "unregistered", "gross"] = "registered"
    work_description: str = ""
    payment_date: Optional[str] = None     # ISO date; defaults to today

    @field_validator("gross_amount", "materials_amount")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return round(v, 2)


class InvoiceRequest(BaseModel):
    """Generate a CIS-compliant invoice."""
    contractor_name: str
    contractor_address: Optional[str] = None
    subcontractor_name: str
    subcontractor_utr: Optional[str] = None
    work_description: str
    labour_amount: float
    materials_amount: float = 0.0
    deduction_status: Literal["registered", "unregistered", "gross"] = "registered"
    invoice_date: Optional[str] = None
    invoice_ref: Optional[str] = None

    @field_validator("labour_amount", "materials_amount")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be non-negative")
        return round(v, 2)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> Dict[str, Any]:
    records = _load_records()
    return {
        "status": "ok",
        "service": "financial-awareness",
        "cis_records": len(records),
        "finance_root": str(FINANCE_ROOT),
    }


@app.post("/finance/cis/record")
async def cis_record(req: CISRecordRequest) -> Dict[str, Any]:
    """Log a CIS payment and return the deduction breakdown."""
    rate = CIS_RATES[req.deduction_status]
    # CIS applies to labour only (gross minus materials)
    labour = max(req.gross_amount - req.materials_amount, 0.0)
    deduction_amount = round(labour * rate, 2)
    net_payment = round(req.gross_amount - deduction_amount, 2)
    payment_date = req.payment_date or date.today().isoformat()

    record: Dict[str, Any] = {
        "id": f"cis-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
        "date": payment_date,
        "contractor_name": req.contractor_name,
        "contractor_utr": req.contractor_utr,
        "gross_amount": req.gross_amount,
        "materials_amount": req.materials_amount,
        "labour_amount": round(labour, 2),
        "deduction_rate": rate,
        "deduction_status": req.deduction_status,
        "deduction_amount": deduction_amount,
        "net_payment": net_payment,
        "work_description": req.work_description,
    }

    records = _load_records()
    records.append(record)
    try:
        _save_records(records)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to persist record: {exc}")

    logger.info("CIS record saved: gross=%.2f deduction=%.2f net=%.2f contractor=%s",
                req.gross_amount, deduction_amount, net_payment, req.contractor_name)
    return record


@app.get("/finance/cis/summary")
async def cis_summary() -> Dict[str, Any]:
    """YTD CIS deductions breakdown and reclaim guidance."""
    records = _load_records()
    today = date.today()
    tax_year_start = date(today.year if today.month >= 4 and today.day >= 6 else today.year - 1, 4, 6)

    ytd_records = [
        r for r in records
        if _in_tax_year(r.get("date", ""), tax_year_start)
    ]

    gross_ytd      = round(sum(r.get("gross_amount",    0) for r in ytd_records), 2)
    deducted_ytd   = round(sum(r.get("deduction_amount", 0) for r in ytd_records), 2)
    net_ytd        = round(sum(r.get("net_payment",      0) for r in ytd_records), 2)
    materials_ytd  = round(sum(r.get("materials_amount", 0) for r in ytd_records), 2)

    estimated_tax  = _income_tax(gross_ytd)
    estimated_ni   = _class4_ni(gross_ytd)
    # CIS already deducted counts against the tax bill — show the credit
    cis_credit     = deducted_ytd
    net_tax_due    = max(round(estimated_tax + estimated_ni - cis_credit, 2), 0.0)

    alerts: List[str] = []
    if gross_ytd >= MTD_THRESHOLD - 2000:
        left = max(MTD_THRESHOLD - gross_ytd, 0)
        alerts.append(f"MTD: £{left:,.0f} to threshold (£{MTD_THRESHOLD:,.0f}) — ensure digital records are current.")
    if gross_ytd >= VAT_THRESHOLD - 5000:
        left = max(VAT_THRESHOLD - gross_ytd, 0)
        alerts.append(f"VAT: £{left:,.0f} to registration threshold (£{VAT_THRESHOLD:,.0f}) — consult HMRC before crossing.")

    return {
        "tax_year_start": tax_year_start.isoformat(),
        "record_count": len(ytd_records),
        "gross_income_ytd": gross_ytd,
        "materials_ytd": materials_ytd,
        "labour_ytd": round(gross_ytd - materials_ytd, 2),
        "cis_deducted_ytd": deducted_ytd,
        "net_received_ytd": net_ytd,
        "estimated_income_tax": estimated_tax,
        "estimated_class4_ni": estimated_ni,
        "cis_credit_against_tax": cis_credit,
        "estimated_net_tax_due": net_tax_due,
        "alerts": alerts,
    }


def _in_tax_year(date_str: str, tax_year_start: date) -> bool:
    try:
        return date.fromisoformat(date_str) >= tax_year_start
    except Exception:
        return False


@app.post("/finance/invoice/generate")
async def invoice_generate(req: InvoiceRequest) -> Dict[str, Any]:
    """Return a CIS-compliant invoice payload (JSON + plain-text rendering)."""
    rate = CIS_RATES[req.deduction_status]
    deduction = round(req.labour_amount * rate, 2)
    total_gross = round(req.labour_amount + req.materials_amount, 2)
    total_net   = round(total_gross - deduction, 2)
    inv_date    = req.invoice_date or date.today().isoformat()
    inv_ref     = req.invoice_ref or f"INV-{datetime.utcnow().strftime('%Y%m%d%H%M')}"

    lines = [
        f"INVOICE {inv_ref}",
        f"Date: {inv_date}",
        "",
        f"From: {req.subcontractor_name}" + (f" (UTR: {req.subcontractor_utr})" if req.subcontractor_utr else ""),
        f"To:   {req.contractor_name}" + (f"\n      {req.contractor_address}" if req.contractor_address else ""),
        "",
        f"Works: {req.work_description}",
        "",
        f"  Labour              £{req.labour_amount:>10,.2f}",
    ]
    if req.materials_amount:
        lines.append(f"  Materials           £{req.materials_amount:>10,.2f}")
    lines += [
        f"  {'─' * 30}",
        f"  Gross total         £{total_gross:>10,.2f}",
        f"  CIS deduction ({int(rate*100)}%) £{deduction:>10,.2f}",
        f"  {'─' * 30}",
        f"  NET PAYMENT DUE     £{total_net:>10,.2f}",
        "",
        f"CIS deduction status: {req.deduction_status.upper()}",
        "Please retain this invoice for your CIS monthly return.",
    ]

    return {
        "invoice_ref": inv_ref,
        "invoice_date": inv_date,
        "contractor_name": req.contractor_name,
        "subcontractor_name": req.subcontractor_name,
        "subcontractor_utr": req.subcontractor_utr,
        "work_description": req.work_description,
        "labour_amount": req.labour_amount,
        "materials_amount": req.materials_amount,
        "gross_total": total_gross,
        "cis_deduction_rate": rate,
        "cis_deduction_amount": deduction,
        "net_payment_due": total_net,
        "deduction_status": req.deduction_status,
        "text": "\n".join(lines),
    }


@app.get("/finance/vat")
async def vat_position() -> Dict[str, Any]:
    """VAT registration position vs HMRC threshold."""
    records = _load_records()
    today = date.today()
    # HMRC uses rolling 12-month lookback for VAT, not tax year
    from datetime import timedelta
    lookback_start = today - timedelta(days=365)
    rolling_income = round(sum(
        float(r.get("gross_amount", 0))
        for r in records
        if _in_rolling_year(r.get("date", ""), lookback_start)
    ), 2)

    below_threshold = rolling_income < VAT_THRESHOLD
    distance = round(VAT_THRESHOLD - rolling_income, 2)
    pct = round(rolling_income / VAT_THRESHOLD * 100, 1)

    guidance: str
    if rolling_income >= VAT_THRESHOLD:
        guidance = (
            f"MANDATORY REGISTRATION: rolling 12-month income £{rolling_income:,.2f} "
            f"exceeds VAT threshold (£{VAT_THRESHOLD:,.2f}). You must register for VAT "
            "with HMRC within 30 days."
        )
    elif distance < 5000:
        guidance = (
            f"WARNING: £{distance:,.2f} below VAT threshold. Monitor monthly — "
            "register voluntarily now to avoid a surprise mandatory registration penalty."
        )
    else:
        guidance = (
            f"Below VAT threshold by £{distance:,.2f} ({pct:.1f}% used). "
            "No registration required. Review again when income approaches £80,000."
        )

    return {
        "rolling_12m_income": rolling_income,
        "vat_threshold": VAT_THRESHOLD,
        "below_threshold": below_threshold,
        "distance_to_threshold": max(distance, 0.0),
        "threshold_used_pct": pct,
        "guidance": guidance,
    }


def _in_rolling_year(date_str: str, lookback_start: date) -> bool:
    try:
        return date.fromisoformat(date_str) >= lookback_start
    except Exception:
        return False


@app.get("/finance/tax")
async def tax_estimate() -> Dict[str, Any]:
    """Estimated Income Tax + NI liability for current tax year."""
    records = _load_records()
    gross_ytd    = _ytd_income_from_records(records)
    deducted_ytd = _ytd_deductions_from_records(records)
    inc_tax      = _income_tax(gross_ytd)
    class4_ni    = _class4_ni(gross_ytd)
    total_bill   = round(inc_tax + class4_ni, 2)
    net_due      = max(round(total_bill - deducted_ytd, 2), 0.0)

    return {
        "gross_income_ytd": gross_ytd,
        "personal_allowance": PERSONAL_ALLOWANCE,
        "taxable_income": max(round(gross_ytd - PERSONAL_ALLOWANCE, 2), 0.0),
        "income_tax_estimate": inc_tax,
        "class4_ni_estimate": class4_ni,
        "total_tax_ni_estimate": total_bill,
        "cis_already_deducted": deducted_ytd,
        "estimated_balance_due": net_due,
        "note": (
            "Estimates only — expenses not factored in. Deduct allowable business "
            "expenses (materials, tools, mileage, insurance, training) to reduce "
            "taxable profit before filing Self Assessment."
        ),
    }


@app.get("/finance/summary")
async def finance_summary() -> Dict[str, Any]:
    """Full financial snapshot: CIS, VAT position, tax estimate, alerts."""
    cis = await cis_summary()
    vat = await vat_position()
    tax = await tax_estimate()
    return {
        "cis": cis,
        "vat": vat,
        "tax": tax,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
