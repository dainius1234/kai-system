#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from fastapi import HTTPException


def load_module():
    root = Path(__file__).resolve().parents[1]
    # add workspace root for importing common and other packages
    sys.path.insert(0, str(root))
    # also ensure memu-core itself is on path
    sys.path.insert(0, str(root / "memu-core"))
    module_path = root / "memu-core" / "app.py"
    spec = importlib.util.spec_from_file_location("memu_phase_b_app", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    os.environ["MAX_MEMORY_RECORDS"] = "2"
    os.environ["MAX_STATE_KEY_SIZE"] = "8"
    os.environ["MAX_STATE_VALUE_SIZE"] = "16"
    mod = load_module()

    mod.store._records.clear()
    r1 = mod.MemoryRecord(id="1", timestamp="2026-01-01T00:00:00", event_type="a", content={"user_id": "keeper"}, embedding=[0.1], relevance=0.1)
    r2 = mod.MemoryRecord(id="2", timestamp="2026-01-01T00:00:00", event_type="b", content={"user_id": "keeper"}, embedding=[0.2], relevance=0.2)
    r3 = mod.MemoryRecord(id="3", timestamp="2026-01-01T00:00:00", event_type="c", content={"user_id": "keeper"}, embedding=[0.3], relevance=0.3)
    mod.store.insert(r1)
    mod.store.insert(r2)
    mod.store.insert(r3)
    assert mod.store.count() == 2
    assert [x.id for x in mod.store.search(10)] == ["3", "2"]

    mod._validate_state_delta_size({"ok": "tiny"})
    try:
        mod._validate_state_delta_size({"toolongkey": "x"})
        raise AssertionError("expected key size violation")
    except HTTPException as exc:
        assert exc.status_code == 400

    try:
        mod._validate_state_delta_size({"small": "x" * 100})
        raise AssertionError("expected value size violation")
    except HTTPException as exc:
        assert exc.status_code == 400

    print("test_phase_b_memu_core: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
