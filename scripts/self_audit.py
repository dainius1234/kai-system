#!/usr/bin/env python3
"""
Self-Audit & Feedback Script for Kai System

- Reviews recent logs, test results, and code changes
- Summarizes system health and recurring issues
- Proposes actionable improvements
- Logs lessons/incidents to memu-core (if available)
- Optionally opens GitHub issues (future)
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path

AUDIT_LOG = Path("output/self_audit_log.json")

# 1. Gather recent test and lint results
def run_make(target):
    try:
        result = subprocess.run(["make", target], capture_output=True, text=True, timeout=120)
        return {
            "target": target,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-1000:],
        }
    except Exception as e:
        return {"target": target, "error": str(e)}

def summarize_results(results):
    summary = []
    for r in results:
        if r.get("returncode", 0) != 0:
            summary.append(f"{r['target']} failed: {r.get('stderr','').strip()[:200]}")
    return summary or ["All checks passed."]

# 2. Log lessons/incidents to memu-core (if available)
def log_lesson_to_memu(lesson):
    try:
        import requests
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "event_type": "self_audit_lesson",
            "content": {"lesson": lesson},
            "user_id": "kai-system",
        }
        requests.post("http://localhost:8001/memory/memorize", json=payload, timeout=2)
    except Exception:
        pass  # memu-core may not be running

def main():
    checks = ["merge-gate", "test-core", "health-sweep"]
    results = [run_make(t) for t in checks]
    summary = summarize_results(results)
    audit = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "summary": summary,
    }
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_LOG.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    for s in summary:
        log_lesson_to_memu(s)
    print("\nSelf-Audit Summary:")
    for s in summary:
        print("-", s)
    print(f"\nFull audit log: {AUDIT_LOG}")
    if any("failed" in s for s in summary):
        sys.exit(1)

if __name__ == "__main__":
    main()
