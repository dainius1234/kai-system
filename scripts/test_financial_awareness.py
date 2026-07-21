"""Unit tests for financial-awareness FastAPI service (P29 — CIS-aware finance)."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Path setup ────────────────────────────────────────────────────────────────
_svc_dir = os.path.join(os.path.dirname(__file__), "..", "financial-awareness")
if _svc_dir not in sys.path:
    sys.path.insert(0, _svc_dir)

from fastapi.testclient import TestClient  # noqa: E402

# Import after path setup; reload so FINANCE_ROOT env override takes effect per test
import app as _app_module  # noqa: E402


def _make_client(tmp_dir: str) -> TestClient:
    """Return a TestClient with FINANCE_ROOT pointed at a temp directory."""
    _app_module.FINANCE_ROOT = Path(tmp_dir)
    _app_module.CIS_RECORDS_FILE = Path(tmp_dir) / "cis_records.json"
    return TestClient(_app_module.app)


class TestHealth(unittest.TestCase):
    def test_health_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = _make_client(tmp)
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["service"], "financial-awareness")
            self.assertEqual(data["cis_records"], 0)


class TestCISRecord(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _make_client(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_registered_20pct(self):
        resp = self.client.post("/finance/cis/record", json={
            "contractor_name": "Acme Build Ltd",
            "gross_amount": 1000.00,
            "materials_amount": 0.0,
            "deduction_status": "registered",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["deduction_amount"], 200.0)
        self.assertAlmostEqual(data["net_payment"], 800.0)
        self.assertAlmostEqual(data["deduction_rate"], 0.20)

    def test_unregistered_30pct(self):
        resp = self.client.post("/finance/cis/record", json={
            "contractor_name": "Big Contractor Ltd",
            "gross_amount": 2000.00,
            "deduction_status": "unregistered",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["deduction_amount"], 600.0)
        self.assertAlmostEqual(data["net_payment"], 1400.0)

    def test_gross_payment_no_deduction(self):
        resp = self.client.post("/finance/cis/record", json={
            "contractor_name": "Trusted Co",
            "gross_amount": 5000.00,
            "deduction_status": "gross",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["deduction_amount"], 0.0)
        self.assertAlmostEqual(data["net_payment"], 5000.0)

    def test_materials_exempt_from_cis(self):
        # £1000 gross = £600 labour + £400 materials; CIS 20% on £600 only = £120
        resp = self.client.post("/finance/cis/record", json={
            "contractor_name": "Builder Ltd",
            "gross_amount": 1000.00,
            "materials_amount": 400.00,
            "deduction_status": "registered",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["deduction_amount"], 120.0)
        self.assertAlmostEqual(data["net_payment"], 880.0)

    def test_record_persisted(self):
        self.client.post("/finance/cis/record", json={
            "contractor_name": "Persisted Co",
            "gross_amount": 500.0,
            "deduction_status": "registered",
        })
        records_file = Path(self._tmp.name) / "cis_records.json"
        self.assertTrue(records_file.exists())
        records = json.loads(records_file.read_text())
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["contractor_name"], "Persisted Co")

    def test_negative_amount_rejected(self):
        resp = self.client.post("/finance/cis/record", json={
            "contractor_name": "Bad Input",
            "gross_amount": -100.0,
        })
        self.assertEqual(resp.status_code, 422)


class TestCISSummary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _make_client(self._tmp.name)
        # Seed two records dated today
        self.client.post("/finance/cis/record", json={
            "contractor_name": "Co A", "gross_amount": 3000.0, "deduction_status": "registered",
        })
        self.client.post("/finance/cis/record", json={
            "contractor_name": "Co B", "gross_amount": 2000.0, "deduction_status": "unregistered",
        })

    def tearDown(self):
        self._tmp.cleanup()

    def test_summary_totals(self):
        resp = self.client.get("/finance/cis/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        # Co A: 3000 @ 20% = 600 deducted; Co B: 2000 @ 30% = 600 deducted
        self.assertAlmostEqual(data["gross_income_ytd"], 5000.0)
        self.assertAlmostEqual(data["cis_deducted_ytd"], 1200.0)
        self.assertAlmostEqual(data["net_received_ytd"], 3800.0)

    def test_summary_has_tax_estimates(self):
        resp = self.client.get("/finance/cis/summary")
        data = resp.json()
        self.assertIn("estimated_income_tax", data)
        self.assertIn("estimated_class4_ni", data)
        self.assertIn("estimated_net_tax_due", data)

    def test_summary_alerts_list(self):
        resp = self.client.get("/finance/cis/summary")
        data = resp.json()
        self.assertIsInstance(data["alerts"], list)


class TestInvoiceGenerate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _make_client(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_invoice_registered(self):
        resp = self.client.post("/finance/invoice/generate", json={
            "contractor_name": "Big Builder Ltd",
            "subcontractor_name": "D Smith",
            "subcontractor_utr": "1234567890",
            "work_description": "Brickwork — block A",
            "labour_amount": 800.0,
            "materials_amount": 200.0,
            "deduction_status": "registered",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["cis_deduction_amount"], 160.0)   # 20% of £800 labour
        self.assertAlmostEqual(data["gross_total"], 1000.0)
        self.assertAlmostEqual(data["net_payment_due"], 840.0)
        self.assertIn("INVOICE", data["text"])
        self.assertIn("CIS deduction", data["text"])

    def test_invoice_gross_status_no_deduction(self):
        resp = self.client.post("/finance/invoice/generate", json={
            "contractor_name": "Trusted Contractor",
            "subcontractor_name": "J Bloggs",
            "work_description": "Roofing",
            "labour_amount": 1500.0,
            "deduction_status": "gross",
        })
        data = resp.json()
        self.assertAlmostEqual(data["cis_deduction_amount"], 0.0)
        self.assertAlmostEqual(data["net_payment_due"], 1500.0)


class TestVATPosition(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _make_client(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_below_threshold(self):
        resp = self.client.get("/finance/vat")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["below_threshold"])
        self.assertAlmostEqual(data["rolling_12m_income"], 0.0)

    def test_vat_guidance_present(self):
        resp = self.client.get("/finance/vat")
        data = resp.json()
        self.assertIn("guidance", data)
        self.assertIn("threshold", data["guidance"].lower())


class TestTaxEstimate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _make_client(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_zero_income(self):
        resp = self.client.get("/finance/tax")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertAlmostEqual(data["income_tax_estimate"], 0.0)
        self.assertAlmostEqual(data["class4_ni_estimate"], 0.0)
        self.assertAlmostEqual(data["estimated_balance_due"], 0.0)

    def test_basic_rate_income(self):
        # Seed £30,000 gross income
        self.client.post("/finance/cis/record", json={
            "contractor_name": "Co X", "gross_amount": 30000.0, "deduction_status": "gross",
        })
        resp = self.client.get("/finance/tax")
        data = resp.json()
        # Taxable: 30000 - 12570 = 17430 @ 20% = 3486
        self.assertAlmostEqual(data["income_tax_estimate"], 3486.0)
        self.assertGreater(data["class4_ni_estimate"], 0)

    def test_note_field_present(self):
        resp = self.client.get("/finance/tax")
        data = resp.json()
        self.assertIn("note", data)


class TestFinanceSummary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.client = _make_client(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_summary_has_all_sections(self):
        resp = self.client.get("/finance/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("cis", data)
        self.assertIn("vat", data)
        self.assertIn("tax", data)


if __name__ == "__main__":
    unittest.main()
