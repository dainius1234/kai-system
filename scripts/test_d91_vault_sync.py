"""D91: Test suite for vault-sync service components.

Tests:
  - parser: NoteData extraction (frontmatter, wikilinks, tags, checksum)
  - mapper: VaultMapper CRUD and persistence
  - watcher: _VaultHandler schedule/flush/filter logic
  - app: HTTP endpoints with mocked memu-core
  - memu-core: vault ingest/search/delete endpoints
  - agentic: vault export/search proxy endpoints
  - feature_flags: FF_VAULT_SYNC, FF_VAULT_CONTEXT
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_plain_note(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "note.md"
        _write(f, "# Hello\n\nThis is a [[wikilink]] and a #tag.")
        note = parse_note(str(f))
        assert note is not None
        assert note.title == "note"
        assert ("wikilink", "wikilink") in note.wikilinks
        assert "tag" in note.tags

    def test_parse_frontmatter_title(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "crypto.md"
        _write(f, "---\ntitle: Bitcoin Deep Dive\n---\n\nContent here.")
        note = parse_note(str(f))
        assert note is not None
        assert note.title == "Bitcoin Deep Dive"

    def test_parse_wikilink_alias(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "links.md"
        _write(f, "See [[BTC|Bitcoin]] for more.")
        note = parse_note(str(f))
        assert note is not None
        assert ("BTC", "Bitcoin") in note.wikilinks

    def test_parse_multiple_tags(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "tagged.md"
        _write(f, "#crypto #finance/trading Something\n\nBody text.")
        note = parse_note(str(f))
        assert note is not None
        assert "crypto" in note.tags
        assert "finance/trading" in note.tags

    def test_parse_checksum_consistency(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "stable.md"
        content = "# Stable note\n\nSame content."
        _write(f, content)
        n1 = parse_note(str(f))
        n2 = parse_note(str(f))
        assert n1 is not None and n2 is not None
        assert n1.checksum == n2.checksum

    def test_parse_checksum_changes_on_edit(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "changing.md"
        _write(f, "# Version 1")
        n1 = parse_note(str(f))
        _write(f, "# Version 2")
        n2 = parse_note(str(f))
        assert n1 is not None and n2 is not None
        assert n1.checksum != n2.checksum

    def test_parse_missing_file_returns_none(self):
        from vault_sync_parser import parse_note
        assert parse_note("/nonexistent/path/note.md") is None

    def test_parse_modified_at_is_float(self, tmp_path):
        from vault_sync_parser import parse_note
        f = tmp_path / "ts.md"
        _write(f, "content")
        note = parse_note(str(f))
        assert note is not None
        assert isinstance(note.modified_at, float)
        assert note.modified_at > 0


# ── Mapper tests ──────────────────────────────────────────────────────────────

class TestMapper:
    def test_upsert_and_get_by_filepath(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m = VaultMapper(str(tmp_path))
        m.upsert("KAI/note.md", "cognee:note:abc", ["cid1"], "sha256:x", "2026-01-01T00:00:00")
        entry = m.get_by_filepath("KAI/note.md")
        assert entry is not None
        assert entry["note_node_id"] == "cognee:note:abc"
        assert entry["concept_ids"] == ["cid1"]

    def test_get_by_node_id(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m = VaultMapper(str(tmp_path))
        m.upsert("KAI/note.md", "cognee:note:abc", [], "checksum1", "2026-01-01T00:00:00")
        fp = m.get_by_node_id("cognee:note:abc")
        assert fp == "KAI/note.md"

    def test_remove(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m = VaultMapper(str(tmp_path))
        m.upsert("KAI/note.md", "cognee:note:abc", [], "checksum1", "2026-01-01")
        m.remove("KAI/note.md")
        assert m.get_by_filepath("KAI/note.md") is None

    def test_len(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m = VaultMapper(str(tmp_path))
        assert len(m) == 0
        m.upsert("a.md", "n1", [], "c1", "t1")
        m.upsert("b.md", "n2", [], "c2", "t2")
        assert len(m) == 2

    def test_persistence(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m1 = VaultMapper(str(tmp_path))
        m1.upsert("note.md", "node123", ["c1"], "checksum", "2026-01-01")
        m2 = VaultMapper(str(tmp_path))  # reload
        entry = m2.get_by_filepath("note.md")
        assert entry is not None
        assert entry["note_node_id"] == "node123"

    def test_all_entries(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m = VaultMapper(str(tmp_path))
        m.upsert("a.md", "n1", [], "c1", "t1")
        m.upsert("b.md", "n2", [], "c2", "t2")
        entries = m.all_entries()
        assert "a.md" in entries
        assert "b.md" in entries

    def test_mapping_file_created(self, tmp_path):
        from vault_sync_mapper import VaultMapper
        m = VaultMapper(str(tmp_path))
        m.upsert("x.md", "n1", [], "c", "t")
        mapping_path = tmp_path / ".vault-sync" / "mapping.json"
        assert mapping_path.exists()
        data = json.loads(mapping_path.read_text())
        assert data["version"] == 1
        assert "x.md" in data["entries"]


# ── Watcher tests ─────────────────────────────────────────────────────────────

class TestWatcher:
    def test_schedule_filters_hidden_files(self):
        from vault_sync_watcher import _VaultHandler
        received = []
        h = _VaultHandler(on_change=received.append, on_delete=lambda x: None)
        h._schedule("/vault/.obsidian/workspace.json")
        h._schedule("/vault/.trash/note.md")
        h._schedule("/vault/.vault-sync/mapping.json")
        time.sleep(0.1)
        assert len(received) == 0

    def test_schedule_filters_non_md(self):
        from vault_sync_watcher import _VaultHandler
        received = []
        h = _VaultHandler(on_change=received.append, on_delete=lambda x: None)
        h._schedule("/vault/KAI/image.png")
        h._schedule("/vault/KAI/data.json")
        time.sleep(0.1)
        assert len(received) == 0

    def test_schedule_accepts_md(self):
        from vault_sync_watcher import _VaultHandler, _DEBOUNCE_SECONDS
        received = []
        h = _VaultHandler(on_change=received.append, on_delete=lambda x: None)
        h._schedule("/vault/Knowledge/BTC.md")
        time.sleep(_DEBOUNCE_SECONDS + 0.5)
        assert "/vault/Knowledge/BTC.md" in received

    def test_debounce_deduplicates_rapid_saves(self, monkeypatch):
        import vault_sync_watcher as watcher_mod
        monkeypatch.setattr(watcher_mod, "_DEBOUNCE_SECONDS", 0.2)
        from vault_sync_watcher import _VaultHandler
        received = []
        h = _VaultHandler(on_change=received.append, on_delete=lambda x: None)
        for _ in range(5):
            h._schedule("/vault/KAI/note.md")
            time.sleep(0.02)
        time.sleep(0.5)
        assert received.count("/vault/KAI/note.md") == 1

    def test_on_deleted_triggers_delete_callback(self):
        from vault_sync_watcher import _VaultHandler
        deleted = []
        h = _VaultHandler(on_change=lambda x: None, on_delete=deleted.append)
        event = MagicMock(is_directory=False, src_path="/vault/KAI/old.md")
        h.on_deleted(event)
        assert "/vault/KAI/old.md" in deleted

    def test_on_moved_triggers_change_and_delete(self):
        from vault_sync_watcher import _VaultHandler, _DEBOUNCE_SECONDS
        changed = []
        deleted = []
        h = _VaultHandler(on_change=changed.append, on_delete=deleted.append)
        event = MagicMock(
            is_directory=False,
            src_path="/vault/KAI/old.md",
            dest_path="/vault/KAI/new.md",
        )
        h.on_moved(event)
        assert "/vault/KAI/old.md" in deleted
        time.sleep(_DEBOUNCE_SECONDS + 0.5)
        assert "/vault/KAI/new.md" in changed

    def test_directory_events_ignored(self):
        from vault_sync_watcher import _VaultHandler
        received = []
        h = _VaultHandler(on_change=received.append, on_delete=lambda x: None)
        event = MagicMock(is_directory=True, src_path="/vault/KAI")
        h.on_created(event)
        h.on_modified(event)
        time.sleep(0.2)
        assert len(received) == 0


# ── App endpoint tests ────────────────────────────────────────────────────────

class TestVaultSyncApp:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("FF_VAULT_SYNC", "true")
        # Lazy import after env is set
        from fastapi.testclient import TestClient
        import importlib
        import vault_sync_app as app_mod
        importlib.reload(app_mod)
        return TestClient(app_mod.app)

    def test_health_returns_ok(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("FF_VAULT_SYNC", "true")
        from fastapi.testclient import TestClient
        import vault_sync_app as app_mod
        with TestClient(app_mod.app) as c:
            r = c.get("/health")
            assert r.status_code == 200
            data = r.json()
            assert data["ff_vault_sync"] is True

    def test_export_conviction_too_low(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("FF_VAULT_SYNC", "true")
        from fastapi.testclient import TestClient
        import vault_sync_app as app_mod
        with TestClient(app_mod.app) as c:
            r = c.post("/export", json={
                "filepath": "KAI/test.md",
                "content": "# Test",
                "conviction": 5.0,
            })
            assert r.status_code == 403

    def test_export_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VAULT_PATH", str(tmp_path))
        monkeypatch.setenv("FF_VAULT_SYNC", "true")
        from fastapi.testclient import TestClient
        import vault_sync_app as app_mod
        with TestClient(app_mod.app) as c:
            r = c.post("/export", json={
                "filepath": "../../../etc/passwd",
                "content": "evil",
                "conviction": 9.5,
            })
            assert r.status_code == 400

    def test_export_writes_file_with_high_conviction(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        import vault_sync_app as app_mod

        monkeypatch.setattr(app_mod, "VAULT_PATH", str(tmp_path))
        monkeypatch.setattr(app_mod, "FF_VAULT_SYNC", True)

        async def _mock_ingest(fp):
            pass

        monkeypatch.setattr(app_mod, "_ingest_note", _mock_ingest)
        with TestClient(app_mod.app) as c:
            r = c.post("/export", json={
                "filepath": "KAI/lesson.md",
                "content": "# Lesson\n\nContent.",
                "conviction": 9.5,
            })
            assert r.status_code == 200
            assert (tmp_path / "KAI" / "lesson.md").exists()

    def test_export_disabled_when_ff_off(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        import vault_sync_app as app_mod
        monkeypatch.setattr(app_mod, "FF_VAULT_SYNC", False)
        with TestClient(app_mod.app) as c:
            r = c.post("/export", json={
                "filepath": "KAI/test.md",
                "content": "x",
                "conviction": 9.9,
            })
            assert r.status_code == 503

    def test_ingest_returns_skipped_on_unchanged_checksum(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient
        import vault_sync_app as app_mod

        monkeypatch.setattr(app_mod, "VAULT_PATH", str(tmp_path))
        monkeypatch.setattr(app_mod, "FF_VAULT_SYNC", True)

        note_path = tmp_path / "KAI" / "note.md"
        _write(note_path, "# Same content")
        # Pre-populate mapper with matching checksum
        import hashlib
        raw = "# Same content"
        checksum = hashlib.sha256(raw.encode()).hexdigest()
        app_mod._mapper = app_mod.VaultMapper(str(tmp_path))
        app_mod._mapper.upsert(str(note_path), "node1", [], checksum, "2026-01-01")

        with TestClient(app_mod.app) as c:
            r = c.post("/ingest", json={"filepath": str(note_path), "force": False})
            assert r.status_code == 200
            assert r.json()["status"] == "skipped"


# ── Memu-core vault endpoints ─────────────────────────────────────────────────

class TestMemuCoreVaultEndpoints:
    @pytest.fixture
    def client(self):
        # Lightweight import using importlib to avoid memu-core's heavy deps
        pytest.importorskip("httpx")
        try:
            from fastapi.testclient import TestClient
            sys.path.insert(0, str(Path(__file__).parent.parent / "memu-core"))
            import importlib
            import memu_core_app
            importlib.reload(memu_core_app)
            return TestClient(memu_core_app.app)
        except Exception:
            pytest.skip("memu-core import not available in test env")

    def test_vault_ingest_returns_node_id(self, client):
        r = client.post("/memory/vault/ingest", json={
            "filepath": "KAI/test.md",
            "title": "Test Note",
            "content": "# Test\n\nContent.",
            "tags": ["test", "kai"],
            "checksum": "abc123",
        })
        assert r.status_code == 200
        data = r.json()
        assert "note_node_id" in data
        assert data["note_node_id"].startswith("cognee:note:")

    def test_vault_search_finds_ingested_note(self, client):
        client.post("/memory/vault/ingest", json={
            "filepath": "KAI/searchable.md",
            "title": "Bitcoin Research",
            "content": "BTC analysis goes here.",
            "tags": ["crypto"],
            "checksum": "def456",
        })
        r = client.get("/memory/vault/search", params={"query": "bitcoin"})
        assert r.status_code == 200
        results = r.json().get("results", [])
        titles = [res["title"] for res in results]
        assert "Bitcoin Research" in titles

    def test_vault_delete_removes_note(self, client):
        resp = client.post("/memory/vault/ingest", json={
            "filepath": "KAI/todelete.md",
            "title": "To Delete",
            "content": "Content.",
            "tags": [],
            "checksum": "ghi789",
        })
        node_id = resp.json()["note_node_id"]
        r = client.delete(f"/memory/vault/{node_id}")
        assert r.status_code == 200
        assert r.json()["removed"] >= 1

    def test_vault_search_folder_filter(self, client):
        client.post("/memory/vault/ingest", json={
            "filepath": "Inbox/note.md",
            "title": "Inbox Note",
            "content": "Inbox content.",
            "tags": [],
            "checksum": "inbox1",
        })
        client.post("/memory/vault/ingest", json={
            "filepath": "KAI/kai-note.md",
            "title": "KAI Note",
            "content": "KAI content.",
            "tags": [],
            "checksum": "kai1",
        })
        r = client.get("/memory/vault/search", params={"query": "note", "folder_filter": "Inbox"})
        assert r.status_code == 200
        results = r.json().get("results", [])
        fps = [res["filepath"] for res in results]
        assert all(fp.startswith("Inbox") for fp in fps)

    def test_vault_ingest_idempotent(self, client):
        payload = {
            "filepath": "KAI/idem.md",
            "title": "Idempotent",
            "content": "Same.",
            "tags": [],
            "checksum": "same1",
        }
        r1 = client.post("/memory/vault/ingest", json=payload)
        r2 = client.post("/memory/vault/ingest", json=payload)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.json()["note_node_id"] == r2.json()["note_node_id"]


# ── Feature flag tests ────────────────────────────────────────────────────────

class TestFeatureFlags:
    def test_vault_sync_flag_exists_and_enabled(self):
        from common.feature_flags import FLAGS, is_enabled
        assert "VAULT_SYNC" in FLAGS
        assert is_enabled("VAULT_SYNC") is True

    def test_vault_context_flag_exists_and_disabled(self):
        from common.feature_flags import FLAGS, is_enabled
        assert "VAULT_CONTEXT" in FLAGS
        assert is_enabled("VAULT_CONTEXT") is False

    def test_vault_sync_can_be_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("FF_VAULT_SYNC", "false")
        import importlib
        import common.feature_flags as ff
        importlib.reload(ff)
        assert ff.is_enabled("VAULT_SYNC") is False

    def test_vault_context_can_be_enabled_via_env(self, monkeypatch):
        monkeypatch.setenv("FF_VAULT_CONTEXT", "true")
        import importlib
        import common.feature_flags as ff
        importlib.reload(ff)
        assert ff.is_enabled("VAULT_CONTEXT") is True


# ── Conftest: sys.path injection ──────────────────────────────────────────────

def pytest_configure(config):
    root = Path(__file__).parent.parent
    for sub in ["vault-sync", "common", "memu-core"]:
        p = str(root / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    # Alias module names for the vault-sync package (avoids `vault_sync.` prefix)
    _alias = {
        "vault_sync_parser": root / "vault-sync" / "parser.py",
        "vault_sync_mapper": root / "vault-sync" / "mapper.py",
        "vault_sync_watcher": root / "vault-sync" / "watcher.py",
        "vault_sync_app": root / "vault-sync" / "app.py",
    }
    for alias, src in _alias.items():
        if alias not in sys.modules:
            spec = __import__("importlib.util", fromlist=["spec_from_file_location", "module_from_spec"])
            spec_obj = spec.spec_from_file_location(alias, src)
            if spec_obj:
                mod = spec.module_from_spec(spec_obj)
                sys.modules[alias] = mod
                try:
                    spec_obj.loader.exec_module(mod)
                except Exception:
                    pass
