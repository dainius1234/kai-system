from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# import agentic app
ROOT = Path(__file__).resolve().parents[1]
# include both workspace root and agentic directory so config module can be imported
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agentic"))

sys.path.insert(0, str(ROOT))

from scripts.module_stubs import fake_redis, stubbed  # noqa: E402
spec = importlib.util.spec_from_file_location("agentic_app", ROOT / "agentic" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
# redis is optional; the stub lasts only as long as the import.
with stubbed({} if "redis" in sys.modules else {"redis": fake_redis()}):
    spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"


if __name__ == "__main__":
    test_health()
    print("agentic service tests passed")
