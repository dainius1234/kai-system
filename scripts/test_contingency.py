from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("heartbeat_app", ROOT / "heartbeat" / "app.py")
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

lib = mod.CONTINGENCIES
assert "intrusion_detected" in lib
assert isinstance(lib["intrusion_detected"].get("checklist"), list)

# dry-check action handler determinism for unknown action
import asyncio
res = asyncio.run(mod._run_action("unknown_action", "intrusion_detected"))
assert res["status"] == "skipped"

print("contingency library test passed")
