#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
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

    class FakeRedis:
        def __init__(self):
            self.data = {}

        def ping(self):
            return True

        def set(self, key, value):
            self.data[key] = value

        def get(self, key):
            return self.data.get(key)

        def expire(self, _key, _ttl):
            return True

    fake_redis = FakeRedis()
    original_get_redis_client = mod._get_redis_client
    mod._get_redis_client = lambda: fake_redis
    try:
        assert mod._persist_to_redis("kai:test:key", {"ok": True}) is True
        assert mod._load_from_redis("kai:test:key", {}) == {"ok": True}

        # P20 (formed_values, conscience_log, loyalty_ledger, gratitude_journal)
        # is intentionally excluded from this periodic snapshot/restore cycle —
        # it now reads/writes live against Redis's own hash/list keys via the
        # _p20_* helpers instead (see DECISIONS.md D22), so there's no global
        # to round-trip here anymore.
        mod._emotional_timeline = [{"emotion": "focused"}]
        persist_results = mod._persist_p17_p22_to_redis()
        assert persist_results["emotional_timeline"] is True
        assert "formed_values" not in persist_results

        fake_redis.data["kai:p17:emotional_timeline"] = json.dumps([{"emotion": "calm"}])
        restored = mod._restore_p17_p22_from_redis()
        assert restored["emotional_timeline"] is True
        assert "formed_values" not in restored
        assert mod._emotional_timeline[0]["emotion"] == "calm"
    finally:
        mod._get_redis_client = original_get_redis_client

    # H2.7 reconnection backoff: failed connection attempts should be rate-limited
    class FailingRedisModule:
        def __init__(self):
            self.calls = 0

        def from_url(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("redis unavailable")

    failing_mod = FailingRedisModule()
    original_redis_module = sys.modules.get("redis")
    original_time = mod.time.time
    now = [1000.0]
    mod._redis_client = None
    mod._redis_last_attempt = 0.0
    mod._redis_retry_delay = 1.0
    mod.time.time = lambda: now[0]
    sys.modules["redis"] = types.SimpleNamespace(from_url=failing_mod.from_url)
    try:
        assert mod._get_redis_client() is None
        assert failing_mod.calls == 1
        current_delay = mod._redis_retry_delay
        now[0] += 0.5
        assert mod._get_redis_client() is None
        assert failing_mod.calls == 1
        now[0] += current_delay
        assert mod._get_redis_client() is None
        assert failing_mod.calls == 2
    finally:
        mod.time.time = original_time
        if original_redis_module is not None:
            sys.modules["redis"] = original_redis_module
        else:
            sys.modules.pop("redis", None)

    print("test_phase_b_memu_core: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
