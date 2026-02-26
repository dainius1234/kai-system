#!/usr/bin/env python3
"""
Operator Feedback Channel for Kai System

- Allows operator to submit feedback, suggestions, or goals via CLI
- Logs each entry as a structured event in memu-core (if available)
- Builds persistent memory of operator guidance for future self-improvement
"""
import sys
import time
import requests

if len(sys.argv) < 2:
    print("Usage: operator_feedback.py <your feedback message>")
    sys.exit(1)

feedback = " ".join(sys.argv[1:]).strip()
if not feedback:
    print("Feedback message cannot be empty.")
    sys.exit(1)

payload = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "event_type": "operator_feedback",
    "content": {"feedback": feedback},
    "user_id": "operator",
}

try:
    r = requests.post("http://localhost:8001/memory/memorize", json=payload, timeout=2)
    if r.status_code == 200:
        print("Feedback logged to memu-core.")
    else:
        print(f"Failed to log feedback (status {r.status_code}): {r.text}")
except Exception as e:
    print(f"Could not reach memu-core: {e}\nFeedback will be retained for manual review.")
    # Optionally, write to a local file for later ingestion
    with open("output/operator_feedback_fallback.log", "a", encoding="utf-8") as f:
        f.write(f"{payload['timestamp']} | {feedback}\n")
    print("Feedback saved locally.")
