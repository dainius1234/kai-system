from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List


def _which(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _append_hash_chain(log_path: Path, payload: Dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prev = "genesis"
    if log_path.exists():
        last = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if last:
            try:
                prev = json.loads(last[-1]).get("hash", "genesis")
            except Exception:
                prev = "genesis"
    base = {"ts": time.time(), **payload, "prev": prev}
    digest = hashlib.sha256(json.dumps(base, sort_keys=True).encode("utf-8")).hexdigest()
    line = {**base, "hash": digest}
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")


def run_medic_scan() -> Dict[str, object]:
    data_root = Path(os.getenv("KAI_DATA_ROOT", "/data"))
    quarantine = Path(os.getenv("KAI_QUARANTINE_DIR", "/quarantine"))
    logs = Path(os.getenv("KAI_LOG_DIR", "/logs"))
    skills_dir = Path(os.getenv("KAI_SKILLS_DIR", str(data_root / "skills")))
    score_threshold = int(os.getenv("KAI_LYNIS_SCORE_THRESHOLD", "80"))

    quarantine.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    clam_log = logs / "clamav.log"
    lynis_log = logs / "lynis-report.txt"
    medic_chain = logs / "medic_audit.log"
    status_path = Path(os.getenv("KAI_MEDIC_STATUS_PATH", "/tmp/kai_medic_status.json"))

    suspicious: List[str] = []
    infected_files = 0
    lynis_score = 100
    lynis_high = False
    actions: List[str] = []

    if _which("clamscan") and data_root.exists():
        cmd = ["clamscan", "-r", str(data_root), f"--move={quarantine}", f"--log={clam_log}"]
        subprocess.run(cmd, check=False)
        text = _read_text(clam_log)
        m = re.search(r"Infected files:\s*(\d+)", text)
        infected_files = int(m.group(1)) if m else 0
    else:
        clam_log.write_text("clamscan unavailable; ran heuristic scan only\n", encoding="utf-8")

    if skills_dir.exists():
        for p in skills_dir.rglob("*.exe"):
            target = quarantine / p.name
            p.replace(target)
            suspicious.append(str(target))
            actions.append(f"quarantined suspicious executable: {target}")

    if _which("lynis"):
        with lynis_log.open("w", encoding="utf-8") as out:
            subprocess.run(["lynis", "audit", "system", "--quick"], stdout=out, stderr=subprocess.STDOUT, check=False)
        report = _read_text(lynis_log)
        m = re.search(r"Hardening index\s*:\s*\[(\d+)\]", report)
        lynis_score = int(m.group(1)) if m else 70
        lynis_high = "warning" in report.lower() or "high" in report.lower()
    else:
        lynis_log.write_text("lynis unavailable; using conservative fallback score=79\n", encoding="utf-8")
        lynis_score = 79
        lynis_high = True

    unsafe = infected_files > 0 or bool(suspicious) or lynis_score < score_threshold or lynis_high
    conviction_limit = 8.4 if unsafe else 10.0

    status = {
        "ts": time.time(),
        "infected_files": infected_files,
        "suspicious_quarantined": suspicious,
        "lynis_score": lynis_score,
        "lynis_high": lynis_high,
        "unsafe": unsafe,
        "conviction_limit": conviction_limit,
        "actions": actions,
    }
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    _append_hash_chain(medic_chain, status)
    return status


if __name__ == "__main__":
    print(json.dumps(run_medic_scan()))
