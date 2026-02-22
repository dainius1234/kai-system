from __future__ import annotations

import os
import tempfile
import time

with tempfile.TemporaryDirectory() as td:
    os.environ["EPISODE_STORE"] = "redis"
    os.environ["REDIS_URL"] = "redis://nonexistent-host:6379"
    os.environ["EPISODE_SPOOL_PATH"] = f"{td}/episodes.log"

    from langgraph.config import ChecksummedSpoolSaver, build_saver  # noqa: E402

    saver = build_saver()
    assert isinstance(saver, ChecksummedSpoolSaver)

    now = time.time()
    saver.save_episode({"user_id": "keeper", "ts": now, "outcome_score": 1.0})
    assert len(saver.recall("keeper", days=1)) == 1

    moved = saver.decay("keeper", days=30, score_threshold=0.2)
    assert moved == 0

print("episode saver fallback tests passed")
