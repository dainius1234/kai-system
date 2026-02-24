from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def load_module():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "memu-core"))
    module_path = root / "memu-core" / "app.py"
    spec = importlib.util.spec_from_file_location("memu_pg_app", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    # only run if a Postgres URI is configured
    uri = os.getenv("PG_URI")
    if not uri:
        print("PG_URI not set; skipping pgvector tests")
        return 0

    mod = load_module()
    os.environ["VECTOR_STORE"] = "postgres"
    # reinitialize store, but catch connection errors
    try:
        mod.store = mod.PGVectorStore()
    except Exception as exc:  # pragma: no cover - DB may not be available
        print(f"could not connect to Postgres, skipping: {exc}")
        return 0
    # verify correct implementation
    assert type(mod.store).__name__ == "PGVectorStore"

    # insert a few records
    mod.store.conn.autocommit = True
    mod.store.conn.cursor().execute("DELETE FROM memories;")
    r1 = mod.MemoryRecord(id="1", timestamp="2026-01-01T00:00:00", event_type="a", content={"user_id": "keeper"}, embedding=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1], relevance=0.1)
    r2 = mod.MemoryRecord(id="2", timestamp="2026-01-02T00:00:00", event_type="b", content={"user_id": "keeper"}, embedding=[0.2]*8, relevance=0.2)
    r3 = mod.MemoryRecord(id="3", timestamp="2026-01-03T00:00:00", event_type="c", content={"user_id": "keeper"}, embedding=[0.3]*8, relevance=0.3)
    mod.store.insert(r1)
    mod.store.insert(r2)
    mod.store.insert(r3)

    count = mod.store.count()
    assert count >= 3

    results = mod.store.search(top_k=2)
    assert len(results) == 2
    # check that results are ordered by timestamp desc
    assert results[0].id == "3"

    # test similarity query
    results2 = mod.store.search(top_k=1, query="test")
    assert len(results2) == 1

    print("pgvector tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
