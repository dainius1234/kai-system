"""Verify perception-camera's auto-speak routes through tool-gate (Phase 0, Step E).

The old code fired TTS directly from the urgency heuristic, skipping the
gate and mode entirely. /proactive/auto must consult _gate_allows_speak()
before calling TTS, and TTS must only fire when the gate approves.
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("camera_gate_app", ROOT / "perception" / "camera" / "app.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _run(coro):
    return asyncio.run(coro)


def _force_high_urgency():
    """Stub out capture/analysis so /proactive/auto deterministically wants to speak."""
    mod._last_proactive_ts = 0.0
    return (
        patch.object(mod, "_capture_screen", return_value=None),
        patch.object(mod, "_analyse_frame", return_value={
            "brightness": 128, "motion_level": 0, "motion_detected": True,
        }),
    )


def test_speak_blocked_when_gate_denies():
    p1, p2 = _force_high_urgency()
    with p1, p2, \
         patch.object(mod, "_speak_or_not", return_value={
             "should_speak": True, "suggested_message": "test message", "urgency": 0.9,
         }), \
         patch.object(mod, "_gate_allows_speak", new=AsyncMock(return_value=False)) as gate_mock, \
         patch("httpx.AsyncClient") as client_cls:
        client_instance = AsyncMock()
        client_cls.return_value.__aenter__.return_value = client_instance
        result = _run(mod.proactive_auto())
        gate_mock.assert_awaited_once()
        assert result["gate_approved"] is False
        # the audio-fetch call may happen, but /synthesize must never be posted
        synthesize_calls = [
            c for c in client_instance.post.call_args_list
            if c.args and c.args[0].endswith("/synthesize")
        ]
        assert synthesize_calls == []


def test_speak_allowed_when_gate_approves():
    p1, p2 = _force_high_urgency()
    with p1, p2, \
         patch.object(mod, "_speak_or_not", return_value={
             "should_speak": True, "suggested_message": "test message", "urgency": 0.9,
         }), \
         patch.object(mod, "_gate_allows_speak", new=AsyncMock(return_value=True)) as gate_mock, \
         patch("httpx.AsyncClient") as client_cls:
        client_instance = AsyncMock()
        client_cls.return_value.__aenter__.return_value = client_instance
        result = _run(mod.proactive_auto())
        gate_mock.assert_awaited_once()
        assert result["gate_approved"] is True
        client_instance.post.assert_awaited_once()
        assert client_instance.post.call_args.args[0].endswith("/synthesize")


if __name__ == "__main__":
    test_speak_blocked_when_gate_denies()
    test_speak_allowed_when_gate_approves()
    print("camera proactive gate tests passed")
