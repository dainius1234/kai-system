"""D91: Bidirectional filepath ↔ graph-node-id mapping for vault-sync.

Persists to .vault-sync/mapping.json inside the vault directory.
Thread-safe via a simple lock (single-process use only).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional


_MAPPING_VERSION = 1


class VaultMapper:
    def __init__(self, vault_path: str) -> None:
        self._lock = threading.Lock()
        self._path = Path(vault_path) / ".vault-sync" / "mapping.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Dict] = {}
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                raw = json.loads(self._path.read_text())
                self._data = raw.get("entries", {})
        except Exception:
            self._data = {}

    def _save(self) -> None:
        try:
            payload = {"version": _MAPPING_VERSION, "entries": self._data}
            self._path.write_text(json.dumps(payload, indent=2))
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────

    def get_by_filepath(self, filepath: str) -> Optional[Dict]:
        with self._lock:
            return self._data.get(str(filepath))

    def get_by_node_id(self, node_id: str) -> Optional[str]:
        with self._lock:
            for fp, entry in self._data.items():
                if entry.get("note_node_id") == node_id:
                    return fp
            return None

    def upsert(
        self,
        filepath: str,
        note_node_id: str,
        concept_ids: List[str],
        checksum: str,
        synced_at: str,
    ) -> None:
        with self._lock:
            self._data[str(filepath)] = {
                "note_node_id": note_node_id,
                "concept_ids": concept_ids,
                "last_synced_checksum": checksum,
                "last_synced_at": synced_at,
            }
            self._save()

    def remove(self, filepath: str) -> None:
        with self._lock:
            self._data.pop(str(filepath), None)
            self._save()

    def all_entries(self) -> Dict[str, Dict]:
        with self._lock:
            return dict(self._data)

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
