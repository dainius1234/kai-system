from __future__ import annotations

import os
import time

os.environ["EPISODE_STORE"] = "redis"
os.environ["REDIS_URL"] = "redis://nonexistent-host:6379"

from langgraph.config import InMemorySaver, build_saver  # noqa: E402


saver = build_saver()
assert isinstance(saver, InMemorySaver)

now = time.time()
saver.save_episode({"user_id": "keeper", "ts": now, "outcome_score": 1.0})
assert len(saver.recall("keeper", days=1)) == 1

# decay should keep recent good episodes
moved = saver.decay("keeper", days=30, score_threshold=0.2)
assert moved == 0

print("episode saver fallback tests passed")
