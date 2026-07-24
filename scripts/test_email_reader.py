"""Email-reader service tests — IMAP is stubbed, no real mail server needed."""
from __future__ import annotations

import email as email_lib
import importlib.util
import imaplib
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw_message(subject="Test", sender="alice@example.com", body="Hello world"):
    import email.mime.text
    msg = email.mime.text.MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["Date"] = "Thu, 01 Jan 2026 12:00:00 +0000"
    return msg.as_bytes()


def _load_module(monkeypatch, with_creds=True):
    mod_name = "email_reader_app"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    if with_creds:
        monkeypatch.setenv("MAIL_HOST", "imap.example.com")
        monkeypatch.setenv("MAIL_USER", "user@example.com")
        monkeypatch.setenv("MAIL_PASS", "secret")
    else:
        monkeypatch.delenv("MAIL_HOST", raising=False)
        monkeypatch.delenv("MAIL_USER", raising=False)
        monkeypatch.delenv("MAIL_PASS", raising=False)

    monkeypatch.setenv("PORT", "8037")
    monkeypatch.setenv("EMAIL_POLL_INTERVAL_SECONDS", "9999")

    spec = importlib.util.spec_from_file_location(
        mod_name,
        Path(__file__).parent.parent / "email-reader" / "app.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client_nocreds(monkeypatch):
    mod = _load_module(monkeypatch, with_creds=False)
    return TestClient(mod.app), mod


@pytest.fixture()
def client(monkeypatch):
    mod = _load_module(monkeypatch, with_creds=True)
    return TestClient(mod.app), mod


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_unconfigured(client_nocreds):
    c, _ = client_nocreds
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["configured"] is False


def test_health_configured(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["configured"] is True


def test_metrics(client):
    c, _ = client
    r = c.get("/metrics")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_inbox_empty_initially(client):
    c, _ = client
    r = c.get("/inbox")
    assert r.status_code == 200
    body = r.json()
    assert body["messages"] == []
    assert body["total"] == 0


def test_unread_empty_initially(client):
    c, _ = client
    r = c.get("/unread")
    assert r.status_code == 200
    body = r.json()
    assert body["unread_count"] == 0
    assert body["sample"] == []


def test_inbox_populated(client):
    c, mod = client
    mod._inbox_cache = [
        {"id": "1", "subject": "Hello", "from": "a@b.com",
         "date": "2026-01-01", "snippet": "Hi there", "read": True},
        {"id": "2", "subject": "World", "from": "c@d.com",
         "date": "2026-01-02", "snippet": "Hey", "read": False},
    ]
    r = c.get("/inbox")
    body = r.json()
    assert body["total"] == 2
    assert len(body["messages"]) == 2
    assert body["messages"][0]["subject"] == "Hello"


def test_inbox_limit(client):
    c, mod = client
    mod._inbox_cache = [
        {"id": str(i), "subject": f"S{i}", "from": "x@y.com",
         "date": "2026-01-01", "snippet": "", "read": False}
        for i in range(10)
    ]
    r = c.get("/inbox?limit=3")
    assert len(r.json()["messages"]) == 3


def test_unread_count(client):
    c, mod = client
    mod._unread_count = 5
    mod._inbox_cache = [
        {"id": "1", "subject": "Hi", "from": "x@y.com", "date": "", "snippet": "", "read": False},
    ]
    r = c.get("/unread")
    body = r.json()
    assert body["unread_count"] == 5
    assert len(body["sample"]) == 1


def test_refresh_no_credentials(client_nocreds):
    c, _ = client_nocreds
    r = c.post("/refresh")
    assert r.status_code == 503


def test_refresh_with_mock_imap(client):
    c, mod = client

    raw = _make_raw_message("Test Subject", "alice@example.com", "Body text")

    fake_conn = MagicMock()
    fake_conn.login.return_value = ("OK", [])
    fake_conn.select.return_value = ("OK", [])
    fake_conn.search.side_effect = [
        ("OK", [b"1 2"]),   # ALL
        ("OK", [b"2"]),     # UNSEEN
    ]
    fake_conn.fetch.return_value = ("OK", [(b"1 (RFC822)", raw)])
    fake_conn.logout.return_value = ("OK", [])

    with patch("imaplib.IMAP4_SSL", return_value=fake_conn):
        r = c.post("/refresh")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["unread"] == 1


def test_decode_header_value(client):
    _, mod = client
    result = mod._decode_header_value("Hello World")
    assert result == "Hello World"


def test_decode_header_value_empty(client):
    _, mod = client
    result = mod._decode_header_value(None)
    assert result == ""


def test_get_body_plain_text(client):
    _, mod = client
    import email.mime.text
    msg = email.mime.text.MIMEText("Plain body text")
    body = mod._get_body(msg)
    assert "Plain body text" in body


def test_get_body_multipart(client):
    _, mod = client
    import email.mime.multipart
    import email.mime.text
    msg = email.mime.multipart.MIMEMultipart("alternative")
    msg.attach(email.mime.text.MIMEText("Text part", "plain"))
    msg.attach(email.mime.text.MIMEText("<p>HTML</p>", "html"))
    body = mod._get_body(msg)
    assert "Text part" in body
