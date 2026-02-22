from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "memu-core"))
module_path = root / "memu-core" / "app.py"
spec = importlib.util.spec_from_file_location("memu_app", module_path)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

mod.store._records.clear()

r1 = mod.MemoryRecord(id="1", timestamp="2026-01-01T00:00:00", event_type="note", content={"user_id": "keeper", "result": "risk plan"}, embedding=mod.generate_embedding("risk plan"), relevance=0.3, pinned=False)
r2 = mod.MemoryRecord(id="2", timestamp="2026-01-01T00:00:00", event_type="note", content={"user_id": "other", "result": "camera frame"}, embedding=mod.generate_embedding("camera"), relevance=0.9, pinned=False)
mod.store._records.extend([r1, r2])

hits = mod.retrieve_ranked("risk plan", "keeper", top_k=5)
assert hits and hits[0].id == "1"
assert all(h.content.get("user_id") == "keeper" for h in hits)

print("memu retrieval tests passed")
