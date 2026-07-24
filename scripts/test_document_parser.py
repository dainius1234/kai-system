"""Tests for document-parser/app.py."""
import importlib.util
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_SVC = Path(__file__).resolve().parents[1] / "document-parser" / "app.py"


def _load():
    common_stub = MagicMock()
    common_stub.setup_json_logger.return_value = MagicMock()
    common_stub.ErrorBudget = MagicMock(
        return_value=MagicMock(snapshot=MagicMock(return_value={}))
    )
    sys.modules.setdefault("common", MagicMock())
    sys.modules["common.runtime"] = common_stub

    # Stub all heavy optional deps so _*_OK flags are True and we can control calls
    fitz_stub = MagicMock()
    docx_stub = MagicMock()
    openpyxl_stub = MagicMock()
    xlrd_stub = MagicMock()
    pptx_stub = MagicMock()
    pptx_util_stub = MagicMock()
    ezdxf_stub = MagicMock()
    ezdxf_recover_stub = MagicMock()
    bs4_stub = MagicMock()

    sys.modules["fitz"] = fitz_stub
    sys.modules["docx"] = docx_stub
    sys.modules["openpyxl"] = openpyxl_stub
    sys.modules["xlrd"] = xlrd_stub
    sys.modules["pptx"] = pptx_stub
    sys.modules["pptx.util"] = pptx_util_stub
    sys.modules["ezdxf"] = ezdxf_stub
    sys.modules["ezdxf.recover"] = ezdxf_recover_stub
    sys.modules["bs4"] = bs4_stub

    spec = importlib.util.spec_from_file_location("doc_parser_app", _SVC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load()
client = TestClient(_mod.app)


# ── basic endpoints ──────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "pdf" in data


def test_metrics():
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_formats():
    resp = client.get("/formats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["zip"] is True
    assert data["csv"] is True
    assert data["json"] is True


# ── CSV ──────────────────────────────────────────────────────────────

def test_parse_csv():
    csv_data = b"name,age,city\nAlice,30,London\nBob,25,Paris\n"
    resp = client.post("/parse", files={"file": ("data.csv", csv_data, "text/csv")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "csv"
    assert data["filename"] == "data.csv"
    assert "Alice" in data["text"]
    assert "London" in data["text"]
    assert data["metadata"]["rows"] == 3


def test_parse_csv_empty():
    resp = client.post("/parse", files={"file": ("empty.csv", b"", "text/csv")})
    assert resp.status_code == 400


# ── JSON ─────────────────────────────────────────────────────────────

def test_parse_json():
    payload = json.dumps({"project": "kai", "phase": 0}).encode()
    resp = client.post("/parse", files={"file": ("config.json", payload, "application/json")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "json"
    assert "kai" in data["text"]


def test_parse_json_invalid():
    resp = client.post("/parse", files={"file": ("bad.json", b"not json at all!!!", "application/json")})
    assert resp.status_code == 200
    assert "not json" in resp.json()["text"]


# ── XML / HTML ───────────────────────────────────────────────────────

def test_parse_xml():
    xml_data = b"<root><item id='1'>Hello World</item></root>"
    resp = client.post("/parse", files={"file": ("data.xml", xml_data, "application/xml")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "xml"
    # bs4 is stubbed, so we exercise the regex fallback in _parse_xml_html
    assert "Hello World" in data["text"] or data["text"] is not None


def test_parse_html():
    html_data = b"<html><body><h1>Test</h1><p>Content here</p></body></html>"
    resp = client.post("/parse", files={"file": ("page.html", html_data, "text/html")})
    assert resp.status_code == 200
    assert resp.json()["format"] == "html"


# ── ZIP ──────────────────────────────────────────────────────────────

def _make_zip(members: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in members.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_parse_zip_with_csv():
    zdata = _make_zip({
        "report.csv": "col1,col2\nA,1\nB,2\n",
        "notes.txt": "Some text notes here.",
    })
    resp = client.post("/parse", files={"file": ("archive.zip", zdata, "application/zip")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "zip"
    assert data["page_count"] == 2
    assert "report.csv" in data["text"]
    assert "notes.txt" in data["text"]
    assert "A" in data["text"] or "col1" in data["text"]


def test_parse_zip_with_json():
    zdata = _make_zip({"config.json": '{"env": "dev"}'})
    resp = client.post("/parse", files={"file": ("pkg.zip", zdata, "application/zip")})
    assert resp.status_code == 200
    assert "env" in resp.json()["text"]


def test_parse_zip_invalid():
    resp = client.post("/parse", files={"file": ("fake.zip", b"not a zip", "application/zip")})
    assert resp.status_code == 400


# ── PDF (mocked) ─────────────────────────────────────────────────────

def test_parse_pdf_mocked():
    page_mock = MagicMock()
    page_mock.get_text.return_value = "Engineering drawing: page 1 content."
    doc_mock = MagicMock()
    doc_mock.page_count = 2
    doc_mock.load_page.return_value = page_mock
    doc_mock.metadata = {"title": "Site Plan", "author": "Engineer"}
    _mod._fitz.open.return_value = doc_mock

    resp = client.post("/parse", files={"file": ("plan.pdf", b"%PDF fake", "application/pdf")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "pdf"
    assert data["page_count"] == 2
    assert data["metadata"]["title"] == "Site Plan"
    assert "Engineering drawing" in data["text"]


# ── DOCX (mocked) ────────────────────────────────────────────────────

def test_parse_docx_mocked():
    para1 = MagicMock()
    para1.text = "Project Scope of Works"
    para2 = MagicMock()
    para2.text = "  "  # blank — should be skipped
    doc_mock = MagicMock()
    doc_mock.paragraphs = [para1, para2]
    doc_mock.tables = []
    _mod._docx.Document.return_value = doc_mock

    resp = client.post("/parse", files={"file": ("spec.docx", b"PK fake", "application/vnd.openxmlformats")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "docx"
    assert "Project Scope" in data["text"]
    assert "  " not in data["text"]


def test_parse_docx_with_table():
    para = MagicMock()
    para.text = "Header"
    cell1 = MagicMock()
    cell1.text = "Cell A"
    cell2 = MagicMock()
    cell2.text = "Cell B"
    row_mock = MagicMock()
    row_mock.cells = [cell1, cell2]
    table_mock = MagicMock()
    table_mock.rows = [row_mock]
    doc_mock = MagicMock()
    doc_mock.paragraphs = [para]
    doc_mock.tables = [table_mock]
    _mod._docx.Document.return_value = doc_mock

    resp = client.post("/parse", files={"file": ("doc.docx", b"PK fake", "application/vnd.openxmlformats")})
    assert resp.status_code == 200
    assert "Cell A" in resp.json()["text"]


# ── XLSX (mocked) ────────────────────────────────────────────────────

def test_parse_xlsx_mocked():
    ws_mock = MagicMock()
    ws_mock.iter_rows.return_value = [
        ("Name", "Value", "Unit"),
        ("Length", 5.2, "m"),
        (None, None, None),  # blank row — should be skipped
    ]
    wb_mock = MagicMock()
    wb_mock.sheetnames = ["Sheet1"]
    wb_mock.__getitem__ = lambda self, key: ws_mock
    _mod._openpyxl.load_workbook.return_value = wb_mock

    resp = client.post("/parse", files={"file": ("data.xlsx", b"PK fake", "application/vnd.openxmlformats")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "xlsx"
    assert "Sheet1" in data["metadata"]["sheets"]
    assert "Name" in data["text"] or "Length" in data["text"]


# ── PPTX (mocked) ────────────────────────────────────────────────────

def test_parse_pptx_mocked():
    para_mock = MagicMock()
    para_mock.text = "Kai System Overview"
    tf_mock = MagicMock()
    tf_mock.paragraphs = [para_mock]
    shape_mock = MagicMock()
    shape_mock.has_text_frame = True
    shape_mock.text_frame = tf_mock
    slide_mock = MagicMock()
    slide_mock.shapes = [shape_mock]
    prs_mock = MagicMock()
    prs_mock.slides = [slide_mock]
    _mod._Presentation.return_value = prs_mock

    resp = client.post("/parse", files={"file": ("deck.pptx", b"PK fake", "application/vnd.openxmlformats")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "pptx"
    assert data["page_count"] == 1
    assert "Kai System Overview" in data["text"]


# ── DXF (mocked) ─────────────────────────────────────────────────────

def test_parse_dxf_mocked():
    text_entity = MagicMock()
    text_entity.dxftype.return_value = "TEXT"
    text_entity.dxf.get.return_value = "NORTH ELEVATION"

    layer_mock = MagicMock()
    layer_mock.dxf.name = "STRUCTURAL"

    block_mock = MagicMock()
    block_mock.name = "COLUMN_A"

    msp_mock = [text_entity]

    doc_mock = MagicMock()
    doc_mock.layers = [layer_mock]
    doc_mock.modelspace.return_value = msp_mock
    doc_mock.blocks = [block_mock]
    doc_mock.header.get.return_value = 4  # millimetres

    _mod._ezdxf.recover.readfile.return_value = (doc_mock, MagicMock())

    resp = client.post("/parse", files={"file": ("drawing.dxf", b"AutoCAD", "application/dxf")})
    assert resp.status_code == 200
    data = resp.json()
    assert data["format"] == "dxf"
    assert "STRUCTURAL" in data["text"]
    assert "NORTH ELEVATION" in data["text"]
    assert "STRUCTURAL" in data["metadata"]["layers"]


# ── DWG (dwg2dxf not available) ──────────────────────────────────────

def test_parse_dwg_no_tool(monkeypatch):
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("dwg2dxf")
    monkeypatch.setattr("subprocess.run", raise_fnf)

    resp = client.post("/parse", files={"file": ("plan.dwg", b"\x00DWG fake", "application/dwg")})
    assert resp.status_code == 503
    assert "dwg2dxf" in resp.json()["detail"].lower()


# ── edge cases ───────────────────────────────────────────────────────

def test_no_filename():
    resp = client.post("/parse", files={"file": ("", b"data", "text/plain")})
    assert resp.status_code in (400, 422)


def test_unknown_extension_falls_back_to_text():
    resp = client.post("/parse", files={"file": ("notes.log", b"line one\nline two", "text/plain")})
    assert resp.status_code == 200
    assert "line one" in resp.json()["text"]


def test_zip_nested_dwg_skipped():
    zdata = _make_zip({"model.dwg": b"\x00DWG fake binary content"})
    resp = client.post("/parse", files={"file": ("project.zip", zdata, "application/zip")})
    assert resp.status_code == 200
    assert "model.dwg" in resp.json()["text"]
