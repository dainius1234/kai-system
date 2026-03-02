"""Telegram-bot service — structural smoke test.

Loads the module and verifies:
  - app.py compiles and has a FastAPI ``app`` object
  - /health endpoint is registered
  - /metrics endpoint is registered
  - /alert endpoint is registered
  - helper functions exist
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load telegram-bot/app.py as a module (hyphenated dir can't be a normal import)
spec = importlib.util.spec_from_file_location("telegram_bot_app", ROOT / "telegram-bot" / "app.py")
assert spec and spec.loader, "Could not locate telegram-bot/app.py"
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

# --- assertions -------------------------------------------------------
app = getattr(mod, "app", None)
assert app is not None, "No 'app' FastAPI instance found"

route_paths = {r.path for r in app.routes}
assert "/health" in route_paths, "/health endpoint missing"
assert "/metrics" in route_paths, "/metrics endpoint missing"
assert "/alert" in route_paths, "/alert endpoint missing"

# verify key helpers exist
assert callable(getattr(mod, "_is_allowed", None)), "_is_allowed helper missing"
assert callable(getattr(mod, "_tg", None)), "_tg helper missing"
assert callable(getattr(mod, "_send_text", None)), "_send_text helper missing"

print("telegram-bot smoke test: PASS")
