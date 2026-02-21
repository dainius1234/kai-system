from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from tools.medic_scan import run_medic_scan

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    data = root / "data"
    skills = data / "skills"
    logs = root / "logs"
    quarantine = root / "quarantine"
    skills.mkdir(parents=True)
    (skills / "payload.exe").write_text("fake", encoding="utf-8")

    os.environ["KAI_DATA_ROOT"] = str(data)
    os.environ["KAI_SKILLS_DIR"] = str(skills)
    os.environ["KAI_LOG_DIR"] = str(logs)
    os.environ["KAI_QUARANTINE_DIR"] = str(quarantine)
    os.environ["KAI_MEDIC_STATUS_PATH"] = str(root / "status.json")

    status = run_medic_scan()
    assert status["unsafe"] is True
    assert (quarantine / "payload.exe").exists()

    loaded = json.loads((root / "status.json").read_text(encoding="utf-8"))
    assert loaded["conviction_limit"] <= 8.4

print("medic scan tests passed")
