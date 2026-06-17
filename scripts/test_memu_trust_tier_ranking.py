"""Verify memu-core's retrieve_ranked() actually weighs trust_tier (Phase 0, Step D).

Two otherwise-identical memories, differing only in trust_tier, must rank
in trust order: PASS > unverified > REPAIR > FAIL_CLOSED.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "memu-core"))
module_path = root / "memu-core" / "app.py"
spec = importlib.util.spec_from_file_location("memu_trust_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.store._records.clear()

emb = mod.generate_embedding("identical content for trust ranking test")


def make(record_id: str, trust_tier: str) -> "mod.MemoryRecord":
    return mod.MemoryRecord(
        id=record_id,
        timestamp="2026-01-01T00:00:00",
        event_type="note",
        content={"user_id": "tester", "result": "identical content for trust ranking test"},
        embedding=emb,
        relevance=0.5,
        importance=0.5,
        pinned=False,
        trust_tier=trust_tier,
    )


r_pass = make("pass-1", "PASS")
r_unverified = make("unverified-1", "unverified")
r_repair = make("repair-1", "REPAIR")
r_fail = make("fail-1", "FAIL_CLOSED")
mod.store._records.extend([r_fail, r_repair, r_unverified, r_pass])

hits = mod.retrieve_ranked("identical content for trust ranking test", "tester", top_k=4)
order = [h.id for h in hits]
assert order == ["pass-1", "unverified-1", "repair-1", "fail-1"], order

print("memu trust_tier ranking test passed")
