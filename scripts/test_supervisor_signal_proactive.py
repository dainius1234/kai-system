"""Verify supervisor's signal-driven proactive check (Phase 0, Step E).

_signal_proactive_check() asks perception-camera's auto-speak logic on
every loop tick instead of waiting for the 15-minute memory-nudge poll.
It must not duplicate camera's own cooldown logic, and the old
_proactive_check() 15-minute poll must remain untouched.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("sup_signal_app", ROOT / "supervisor" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _run(coro):
    return asyncio.run(coro)


def test_signal_check_calls_camera_proactive_auto():
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {"should_speak": True, "gate_approved": True, "reason": "high_stress_detected"}
    client_instance = AsyncMock()
    client_instance.post.return_value = fake_resp

    with patch("httpx.AsyncClient") as client_cls:
        client_cls.return_value.__aenter__.return_value = client_instance
        _run(mod._signal_proactive_check())
        client_instance.post.assert_awaited_once()
        called_url = client_instance.post.call_args.args[0]
        assert called_url.endswith("/proactive/auto")


def test_signal_check_fails_closed_on_unreachable_camera():
    with patch("httpx.AsyncClient") as client_cls:
        client_cls.return_value.__aenter__.side_effect = Exception("connection refused")
        # must not raise
        _run(mod._signal_proactive_check())


def test_old_fifteen_minute_poll_untouched():
    """_proactive_check still exists with its own interval gate, unchanged by Step E."""
    assert hasattr(mod, "_proactive_check")
    assert hasattr(mod, "PROACTIVE_INTERVAL")
    src = (ROOT / "supervisor" / "app.py").read_text()
    assert "await _proactive_check()" in src
    assert "await _signal_proactive_check()" in src


if __name__ == "__main__":
    test_signal_check_calls_camera_proactive_auto()
    test_signal_check_fails_closed_on_unreachable_camera()
    test_old_fifteen_minute_poll_untouched()
    print("supervisor signal proactive tests passed")
