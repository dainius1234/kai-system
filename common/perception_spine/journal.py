"""Durable append-only event journal for the perception spine.

The journal persists PerceptionEvents as one-JSON-per-line to a local file.
Entries are fsynced on write for crash safety.  The journal supports:

  - append: write a validated event
  - replay: yield events in insertion order (optionally from an offset)
  - offset tracking: monotonic sequence number per entry
  - digest verification on replay
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from common.contracts.perception import PerceptionEvent


class JournalEntry:
    __slots__ = ("offset", "event", "journaled_at")

    def __init__(self, offset: int, event: PerceptionEvent, journaled_at: str):
        self.offset = offset
        self.event = event
        self.journaled_at = journaled_at


class EventJournal:
    """Append-only, file-backed event journal.

    Each line is a JSON object: ``{"offset": N, "journaled_at": ..., "event": {...}}``.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._next_offset = self._recover_offset()

    def _recover_offset(self) -> int:
        if not self._path.exists() or self._path.stat().st_size == 0:
            return 0
        last_offset = 0
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    last_offset = rec.get("offset", last_offset)
                except json.JSONDecodeError:
                    continue
        return last_offset + 1

    @property
    def next_offset(self) -> int:
        return self._next_offset

    def _ends_with_newline(self) -> bool:
        """Whether the journal's last byte is a newline.

        A crash mid-append leaves a partial line with no terminator.
        Appending straight onto it would concatenate the two records and
        corrupt the *new* one as well as the torn one.
        """
        if not self._path.exists():
            return True
        size = self._path.stat().st_size
        if size == 0:
            return True
        with open(self._path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            return fh.read(1) == b"\n"

    def append(self, event: PerceptionEvent) -> int:
        with self._lock:
            offset = self._next_offset
            record = {
                "offset": offset,
                "journaled_at": datetime.now(timezone.utc).isoformat(),
                "event": json.loads(event.model_dump_json()),
            }
            needs_terminator = not self._ends_with_newline()
            with open(self._path, "a", encoding="utf-8") as fh:
                if needs_terminator:
                    # Close off a torn line so this record starts clean.
                    # The torn line stays in the file and is skipped on
                    # replay — recoverable, and visible for audit.
                    fh.write("\n")
                fh.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False))
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            self._next_offset = offset + 1
            return offset

    def replay(
        self, from_offset: int = 0, verify_digests: bool = False
    ) -> Iterator[JournalEntry]:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                off = rec.get("offset", -1)
                if off < from_offset:
                    continue
                event = PerceptionEvent.model_validate(rec["event"])
                if verify_digests and not event.verify_digest():
                    raise ValueError(
                        f"Digest verification failed at offset {off}, "
                        f"event {event.id}"
                    )
                yield JournalEntry(
                    offset=off,
                    event=event,
                    journaled_at=rec.get("journaled_at", ""),
                )

    def erase_subject(self, principal_identity: str) -> int:
        """Remove every event belonging to one principal (roadmap §16.30).

        Rewrites the journal without the subject's events.  Offsets of
        surviving records are preserved so downstream references stay
        valid — an erasure must not silently renumber the audit trail.

        Returns the number of events removed.
        """
        if not self._path.exists():
            return 0

        with self._lock:
            surviving: list[str] = []
            removed = 0
            with open(self._path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Torn line — drop it, it is unreadable anyway.
                        continue
                    identity = (
                        rec.get("event", {})
                        .get("principal", {})
                        .get("identity")
                    )
                    if identity == principal_identity:
                        removed += 1
                        continue
                    surviving.append(json.dumps(
                        rec, separators=(",", ":"), ensure_ascii=False
                    ))

            tmp = self._path.with_suffix(self._path.suffix + ".erasing")
            with open(tmp, "w", encoding="utf-8") as fh:
                for line in surviving:
                    fh.write(line)
                    fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._path)

            return removed

    def count(self) -> int:
        return self._next_offset

    def truncate(self) -> None:
        with self._lock:
            with open(self._path, "w", encoding="utf-8") as fh:
                fh.truncate(0)
            self._next_offset = 0
