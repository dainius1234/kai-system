from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("memu_app", ROOT / "memu-core" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

client = TestClient(mod.app)


def test_cross_session_context():
    # Memorize in session A
    session_a = "session-a"
    session_b = "session-b"
    text = "Kai learned about the site inspection on Friday."
    resp = client.post("/session/{}/append".format(session_a), json={"role": "user", "content": text})
    assert resp.status_code == 200

    # Memorize in session B
    resp = client.post("/session/{}/append".format(session_b), json={"role": "user", "content": "What events are coming up?"})
    assert resp.status_code == 200

    # Retrieve context in session B
    resp = client.get(f"/session/{session_b}/context", params={"query": "site inspection"})
    assert resp.status_code == 200
    data = resp.json()
    found = any("site inspection" in m for m in data.get("long_term_memories", []))
    assert found, "Session B should see session A's memory"

if __name__ == "__main__":
    test_cross_session_context()
    print("cross-session context test passed")
