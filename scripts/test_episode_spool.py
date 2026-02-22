from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from langgraph.config import ChecksummedSpoolSaver

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "episodes.log"
    saver = ChecksummedSpoolSaver(str(p))
    saver.save_episode({"user_id": "keeper", "ts": time.time(), "outcome_score": 1.0, "input": "a"})

    # append corrupted line
    bad = json.dumps({"checksum": "deadbeef", "payload": {"user_id": "keeper", "ts": time.time(), "outcome_score": 0.1}})
    p.write_text(p.read_text() + bad + "\n", encoding="utf-8")

    reloaded = ChecksummedSpoolSaver(str(p))
    episodes = reloaded.recall("keeper", days=1)
    assert len(episodes) == 1

print("episode spool tests passed")
