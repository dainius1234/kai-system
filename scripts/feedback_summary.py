#!/usr/bin/env python3
"""
Feedback Summary Tool for Kai System

- Retrieves and summarizes recent lessons, feedback, and actions from memu-core memory
- Falls back to local logs if memu-core is unavailable
- Surfaces actionable insights for both operator and AI review
"""
import time
import requests
from pathlib import Path

MEMU_URL = "http://localhost:8001/memory/query"
LOCAL_FEEDBACK = Path("output/operator_feedback_fallback.log")
LOCAL_AUDIT = Path("output/self_audit_log.json")

print("\n=== Feedback & Memory Summary ===\n")

try:
    # Query last 20 memory events of relevant types
    q = {
        "event_types": ["operator_feedback", "self_audit_lesson", "system_action"],
        "limit": 20,
        "order": "desc",
    }
    r = requests.post(MEMU_URL, json=q, timeout=3)
    if r.status_code == 200:
        events = r.json().get("results", [])
        for e in events:
            ts = e.get("timestamp", "?")
            et = e.get("event_type", "?")
            content = e.get("content", {})
            print(f"[{ts}] {et}: {content}")
    else:
        print(f"memu-core query failed: {r.status_code} {r.text}")
except Exception as e:
    print(f"Could not reach memu-core: {e}")
    # Fallback: print local logs
    if LOCAL_FEEDBACK.exists():
        print("\nOperator Feedback (local fallback):")
        print(LOCAL_FEEDBACK.read_text(encoding="utf-8"))
    if LOCAL_AUDIT.exists():
        print("\nSelf-Audit Log (local fallback):")
        print(LOCAL_AUDIT.read_text(encoding="utf-8"))

print("\n=== End of Summary ===\n")
