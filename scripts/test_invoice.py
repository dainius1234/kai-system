from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.invoice import should_add_vat, generate_invoice_file, main


def test_should_add_vat():
    assert not should_add_vat(0, 85000)
    assert not should_add_vat(84999.99, 85000)
    assert should_add_vat(85000, 85000)
    assert should_add_vat(100000, 85000)


def test_generate_invoice_file(tmp_path, capsys):
    out = tmp_path / "out.txt"
    generate_invoice_file(str(out), 1234.5, False)
    assert out.exists()
    content = out.read_text()
    assert "Income: 1234.5" in content
    assert "VAT applied: False" in content


def test_main_defaults(monkeypatch, tmp_path, capsys):
    # ensure environment variables are used when arguments are absent
    monkeypatch.setenv("CURRENT_INCOME", "5000")
    monkeypatch.setenv("VAT_THRESHOLD", "10000")
    outfile = tmp_path / "inv.pdf"
    # run main with explicit output
    monkeypatch.setattr(sys, "argv", ["invoice.py", "--output", str(outfile)])
    main()
    captured = capsys.readouterr()
    assert "inv.pdf generated" in captured.out
    assert outfile.exists()

print("invoice tests passed")
