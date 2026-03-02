from __future__ import annotations

import importlib.util
import os
import tempfile
import time
from pathlib import Path

# Import our local kai_config, avoiding the installed langgraph package
_mod_path = Path(__file__).resolve().parents[1] / "langgraph" / "kai_config.py"
_spec = importlib.util.spec_from_file_location("kai_config", _mod_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ChecksummedSpoolSaver = _mod.ChecksummedSpoolSaver
build_saver = _mod.build_saver

with tempfile.TemporaryDirectory() as td:
    os.environ["EPISODE_STORE"] = "redis"
    os.environ["REDIS_URL"] = "redis://nonexistent-host:6379"
    os.environ["EPISODE_SPOOL_PATH"] = f"{td}/episodes.log"

    saver = build_saver()
    assert isinstance(saver, ChecksummedSpoolSaver)
    now = time.time()
    saver.save_episode({"user_id": "keeper", "ts": now, "outcome_score": 1.0})
    assert len(saver.recall("keeper", days=1)) == 1

    moved = saver.decay("keeper", days=30, score_threshold=0.2)
    assert moved == 0

print("episode saver fallback tests passed")
