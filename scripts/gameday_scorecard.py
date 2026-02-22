from __future__ import annotations

import json
import os
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
    ["python", "scripts/test_episode_spool.py"],
    ["python", "scripts/test_error_budget_breaker.py"],
    ["python", "scripts/test_tool_gate_security.py"],
    ["python", "scripts/test_memu_retrieval.py"],
]

MIN_PASS_PERCENT = float(os.getenv("GAMEDAY_MIN_PASS_PERCENT", "100"))
MAX_TOTAL_DURATION_S = float(os.getenv("GAMEDAY_MAX_TOTAL_DURATION_S", "120"))


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
    overall_started = time.time()
    results = [run_check(c) for c in CHECKS]
    total_duration = round(time.time() - overall_started, 2)
    passed = sum(1 for r in results if r["ok"])
    score = round((passed / len(results)) * 100, 1)

    slo = {
        "min_pass_percent": MIN_PASS_PERCENT,
        "max_total_duration_s": MAX_TOTAL_DURATION_S,
        "pass_percent_ok": score >= MIN_PASS_PERCENT,
        "duration_ok": total_duration <= MAX_TOTAL_DURATION_S,
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_checks": len(results),
        "passed_checks": passed,
        "score_percent": score,
        "total_duration_s": total_duration,
        "slo": slo,
        "results": results,
    }
    out = Path("output/gameday_scorecard.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"score_percent": score, "total_duration_s": total_duration, "slo": slo, "output": str(out)}, indent=2))
    if not (slo["pass_percent_ok"] and slo["duration_ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
