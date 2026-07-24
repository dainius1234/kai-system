"""D91: Vault file watcher with debounce.

Watches the Obsidian vault directory for .md file changes and feeds them
into an asyncio queue for the sync controller to process.
Ignores .vault-sync/ and hidden files/directories.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Set

logger = logging.getLogger("vault-sync")

_DEBOUNCE_SECONDS = 2.0


class _VaultHandler:
    """Watchdog-compatible event handler with debounce per filepath."""

    def __init__(self, on_change: Callable[[str], None], on_delete: Callable[[str], None]) -> None:
        self._on_change = on_change
        self._on_delete = on_delete
        self._pending: dict = {}   # filepath → scheduled_time
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    # watchdog calls these
    def dispatch(self, event) -> None:
        pass  # unused; we use on_modified/on_created/on_moved/on_deleted

    def on_created(self, event) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_modified(self, event) -> None:
        if not event.is_directory:
            self._schedule(event.src_path)

    def on_moved(self, event) -> None:
        if not event.is_directory:
            self._schedule(event.dest_path)
            self._on_delete(event.src_path)

    def on_deleted(self, event) -> None:
        if not event.is_directory:
            self._on_delete(event.src_path)

    def _schedule(self, filepath: str) -> None:
        path = Path(filepath)
        if not filepath.endswith(".md"):
            return
        # Skip hidden paths and .vault-sync directory
        if any(part.startswith(".") for part in path.parts):
            return
        with self._lock:
            deadline = time.monotonic() + _DEBOUNCE_SECONDS
            self._pending[filepath] = deadline
        if self._timer is None or not self._timer.is_alive():
            self._timer = threading.Timer(_DEBOUNCE_SECONDS + 0.1, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        now = time.monotonic()
        ready: list = []
        still_pending: dict = {}
        with self._lock:
            for fp, deadline in self._pending.items():
                if now >= deadline:
                    ready.append(fp)
                else:
                    still_pending[fp] = deadline
            self._pending = still_pending
        for fp in ready:
            try:
                self._on_change(fp)
            except Exception as exc:
                logger.error("Error processing %s: %s", fp, exc)
        if still_pending:
            self._timer = threading.Timer(_DEBOUNCE_SECONDS + 0.1, self._flush)
            self._timer.daemon = True
            self._timer.start()


class FileWatcher:
    def __init__(
        self,
        vault_path: str,
        on_change: Callable[[str], None],
        on_delete: Callable[[str], None],
    ) -> None:
        self._vault_path = vault_path
        self._handler = _VaultHandler(on_change, on_delete)
        self._observer = None

    def start(self) -> None:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class _Bridge(FileSystemEventHandler):
                def __init__(self, h):
                    self._h = h
                def on_created(self, e): self._h.on_created(e)
                def on_modified(self, e): self._h.on_modified(e)
                def on_moved(self, e): self._h.on_moved(e)
                def on_deleted(self, e): self._h.on_deleted(e)

            self._observer = Observer()
            self._observer.schedule(_Bridge(self._handler), self._vault_path, recursive=True)
            self._observer.start()
            logger.info("Vault file watcher started on %s", self._vault_path)
        except ImportError:
            logger.warning("watchdog not installed — file watching disabled")
        except Exception as exc:
            logger.error("Failed to start file watcher: %s", exc)

    def stop(self) -> None:
        if self._observer:
            try:
                self._observer.stop()
                self._observer.join(timeout=3)
            except Exception:
                pass

    @property
    def running(self) -> bool:
        return self._observer is not None and self._observer.is_alive()
