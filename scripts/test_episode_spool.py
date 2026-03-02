from __future__ import annotations

import importlib.util
import json
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
