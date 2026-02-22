from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

CHECKS = [
    ["make", "go_no_go"],
    ["make", "test-conviction"],
    ["make", "test-self-emp"],
    ["make", "kai-control-selftest"],
    ["make", "hardening_smoke"],
    ["make", "kai-drill-test"],
    ["python", "scripts/test_episode_saver.py"],
    ["python", "scripts/test_tool_gate_security.py"],
    ["python", "scripts/test_memu_retrieval.py"],
]


def run_check(cmd: list[str]) -> dict[str, object]:
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "ok": proc.returncode == 0,
        "duration_s": round(time.time() - started, 2),
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-5:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-5:]),
    }


def main() -> None:
    results = [run_check(c) for c in CHECKS]
    passed = sum(1 for r in results if r["ok"])
    score = round((passed / len(results)) * 100, 1)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_checks": len(results),
        "passed_checks": passed,
        "score_percent": score,
        "results": results,
    }
    out = Path("output/gameday_scorecard.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"score_percent": score, "output": str(out)}, indent=2))
    if score < 100.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
