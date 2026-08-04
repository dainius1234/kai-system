"""Tests for scripts/hse_rams.py — RAMS.docx generator."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `hse_rams` lives in scripts/, not at the repo root. Because scripts/ is
# a package, pytest imports this file as `scripts.test_hse_rams` and does
# *not* put scripts/ on sys.path, so inserting ROOT alone raised
# ModuleNotFoundError at collection.
sys.path.insert(0, str(ROOT / "scripts"))

from hse_rams import (  # noqa: E402
    DEFAULT_CSV,
    generate_rams,
    load_activities,
    _risk_label_from_score,
    _risk_score,
)


class TestRiskHelpers(unittest.TestCase):
    def test_risk_score(self):
        self.assertEqual(_risk_score(5, 3), 15)
        self.assertEqual(_risk_score(1, 1), 1)

    def test_risk_label_boundaries(self):
        self.assertEqual(_risk_label_from_score(4), "Very Low")
        self.assertEqual(_risk_label_from_score(5), "Low")
        self.assertEqual(_risk_label_from_score(8), "Low")
        self.assertEqual(_risk_label_from_score(9), "Medium")
        self.assertEqual(_risk_label_from_score(12), "Medium")
        self.assertEqual(_risk_label_from_score(13), "High")
        self.assertEqual(_risk_label_from_score(16), "High")
        self.assertEqual(_risk_label_from_score(17), "Very High")
        self.assertEqual(_risk_label_from_score(25), "Very High")


class TestLoadActivities(unittest.TestCase):
    def test_loads_default_csv(self):
        rows = load_activities(DEFAULT_CSV)
        self.assertGreater(len(rows), 0)
        for row in rows:
            self.assertIn("activity", row)
            self.assertIn("hazard", row)
            self.assertIn("controls", row)
            self.assertIsInstance(row["severity"], int)
            self.assertIsInstance(row["likelihood"], int)

    def test_missing_csv_raises(self):
        with self.assertRaises(FileNotFoundError):
            generate_rams(csv_path=Path("/nonexistent/file.csv"), out_path=Path("/tmp/x.docx"))


class TestGenerateRams(unittest.TestCase):
    def test_generates_docx(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            out = Path(f.name)

        try:
            result = generate_rams(
                csv_path=DEFAULT_CSV,
                out_path=out,
                project_name="Test Project",
                site_address="123 Test St",
            )
            self.assertEqual(result, out)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)  # non-trivial file
        finally:
            out.unlink(missing_ok=True)

    def test_custom_project_name(self):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            out = Path(f.name)
        try:
            generate_rams(
                csv_path=DEFAULT_CSV,
                out_path=out,
                project_name="Drainage Grid B5",
                site_address="Unit 7, Industrial Estate",
                prepared_by="J. Smith",
            )
            # Verify it's a valid ZIP (docx = zip)
            import zipfile
            self.assertTrue(zipfile.is_zipfile(out))
        finally:
            out.unlink(missing_ok=True)

    def test_docx_contains_expected_content(self):
        """docx XML should include key text from the CSV."""
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            out = Path(f.name)
        try:
            generate_rams(csv_path=DEFAULT_CSV, out_path=out)
            import zipfile
            with zipfile.ZipFile(out) as z:
                xml = z.read("word/document.xml").decode("utf-8")
            self.assertIn("Excavation", xml)
            self.assertIn("Concrete", xml)
            self.assertIn("RAMS", xml)
        finally:
            out.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
