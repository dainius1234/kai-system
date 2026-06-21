from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def load_module():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "memu-core"))
    module_path = root / "memu-core" / "app.py"
    spec = importlib.util.spec_from_file_location("memu_turbovec_app", module_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    # only run if both a Postgres URI and the turbovec package are available
    uri = os.getenv("PG_URI")
    if not uri:
        print("PG_URI not set; skipping turbovec tests")
        return 0
    try:
        import turbovec  # noqa: F401
    except ImportError:
        print("turbovec not installed; skipping turbovec tests")
        return 0

    mod = load_module()
    os.environ["VECTOR_STORE"] = "turbovec"
    os.environ["TURBOVEC_INDEX_PATH"] = str(Path(tempfile.mkdtemp()) / "test.tv")
    try:
        mod.store = mod.TurboVecStore()
    except Exception as exc:  # pragma: no cover - DB may not be available
        print(f"could not connect to Postgres, skipping: {exc}")
        return 0
    assert type(mod.store).__name__ == "TurboVecStore"

    conn = mod.store._get_conn()
    with conn.cursor() as cur:
        cur.execute("DELETE FROM memories;")
    conn.commit()
    mod.store._put_conn(conn)

    dim = mod.store._tv_dim
    r1 = mod.MemoryRecord(id="t1", timestamp="2026-01-01T00:00:00", event_type="a", content={"user_id": "keeper"}, embedding=[0.1] * dim, relevance=0.1)
    r2 = mod.MemoryRecord(id="t2", timestamp="2026-01-02T00:00:00", event_type="b", content={"user_id": "keeper"}, embedding=[0.2] * dim, relevance=0.2)
    r3 = mod.MemoryRecord(id="t3", timestamp="2026-01-03T00:00:00", event_type="c", content={"user_id": "keeper"}, embedding=[0.3] * dim, relevance=0.3)
    mod.store.insert(r1)
    mod.store.insert(r2)
    mod.store.insert(r3)

    count = mod.store.count()
    assert count >= 3

    results = mod.store.search(top_k=2)
    assert len(results) == 2
    assert results[0].id == "t3"

    results2 = mod.store.search(top_k=1, query="test")
    assert len(results2) == 1

    assert mod.store.delete_record("t1") is True
    assert mod.store.count() == 2

    print("turbovec tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
