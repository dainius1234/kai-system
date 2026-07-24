#!/usr/bin/env python3
"""
Capture 0.5b baseline responses for G4 GPU Day comparison.

Usage (run against your local dev stack while on qwen2.5:0.5b):
    python scripts/capture_baseline_responses.py
    python scripts/capture_baseline_responses.py --host http://localhost:8007

Output: kai-pm/BASELINE_RESPONSES_0.5B.md

Run this BEFORE the GPU arrives. On GPU Day, run the same 5 prompts via the
kai_chat() helper in GPU_ARRIVAL_RUNBOOK.md and compare the outputs.
"""
import argparse
import json
import sys
import os
import datetime
import urllib.request
import urllib.error

PROMPTS = [
    {
        "id": "P1",
        "label": "Factual recall",
        "text": "What is the CIS deduction rate for verified subcontractors in the UK?",
    },
    {
        "id": "P2",
        "label": "Multi-step numerical reasoning",
        "text": (
            "I have a scaffolding contract worth £45,000 inc VAT. "
            "The subcontractor is CIS-registered and verified. "
            "Calculate the net payment after CIS deduction."
        ),
    },
    {
        "id": "P3",
        "label": "Memory and context",
        "text": "Remind me what the VAT threshold is and how it interacts with CIS registration.",
    },
    {
        "id": "P4",
        "label": "Self-awareness",
        "text": (
            "What can you actually do right now? "
            "Be honest about your current capabilities and limitations."
        ),
    },
    {
        "id": "P5",
        "label": "Complex task decomposition",
        "text": (
            "I need to plan a week to get my CIS returns up to date, "
            "chase 3 overdue invoices, and review my subcontractor contracts. "
            "Help me structure this."
        ),
    },
]

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kai-pm",
    "BASELINE_RESPONSES_0.5B.md",
)


def chat(host: str, message: str, session_id: str) -> str:
    """Send a chat message and collect the full streaming response."""
    url = f"{host.rstrip('/')}/chat"
    payload = json.dumps({"message": message, "session_id": session_id}).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    tokens = []
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    obj = json.loads(line)
                    token = obj.get("token") or obj.get("text") or obj.get("content") or ""
                    if token:
                        tokens.append(token)
                        print(token, end="", flush=True)
                except json.JSONDecodeError:
                    tokens.append(line)
                    print(line, end="", flush=True)
    except urllib.error.URLError as e:
        raise SystemExit(f"\nERROR: Cannot reach {url} — {e}\nIs the dev stack running?")
    print()
    return "".join(tokens).strip()


def get_model_info(host: str) -> str:
    """Try to read the current model from the health endpoint."""
    try:
        url = f"{host.rstrip('/')}/health"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("model") or data.get("ollama_model") or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost:8007", help="Agentic base URL")
    parser.add_argument("--session", default="baseline-0.5b", help="Session ID prefix")
    args = parser.parse_args()

    model = get_model_info(args.host)
    captured_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"=== Capturing 0.5b baseline responses ===")
    print(f"Host:  {args.host}")
    print(f"Model: {model}")
    print(f"Time:  {captured_at}")
    print()

    results = []
    for p in PROMPTS:
        print(f"--- {p['id']}: {p['label']} ---")
        print(f"PROMPT: {p['text']}")
        print("RESPONSE: ", end="", flush=True)
        response = chat(args.host, p["text"], f"{args.session}-{p['id'].lower()}")
        results.append({**p, "response": response})
        print()

    # Write the baseline file
    lines = [
        "# 0.5b Baseline Responses",
        "",
        f"**Captured:** {captured_at}  ",
        f"**Model:** {model}  ",
        f"**Host:** {args.host}  ",
        f"**Purpose:** G4 comparison baseline — compare these against 7B responses on GPU Day  ",
        "",
        "---",
        "",
    ]
    for r in results:
        lines += [
            f"## {r['id']} — {r['label']}",
            "",
            f"**Prompt:** {r['text']}",
            "",
            "**Response:**",
            "",
            r["response"] if r["response"] else "*(empty response)*",
            "",
            "---",
            "",
        ]

    lines += [
        "## G4 Evaluation Guide",
        "",
        "On GPU Day, run the same 5 prompts via the `kai_chat()` helper in",
        "`kai-pm/GPU_ARRIVAL_RUNBOOK.md` Step G4 and capture to",
        "`kai-pm/BASELINE_RESPONSES_7B.md`.",
        "",
        "Score each prompt 1 (worse/same) or 2 (better) on three axes:",
        "",
        "| Prompt | Completeness | Numerical accuracy | Structure |",
        "|--------|-------------|-------------------|-----------|",
        "| P1 |  |  |  |",
        "| P2 |  |  |  |",
        "| P3 |  |  |  |",
        "| P4 |  |  |  |",
        "| P5 |  |  |  |",
        "",
        "**Pass condition:** ≥3/5 prompts score better on at least 2 of 3 axes.",
        "Document the delta in DECISIONS.md D83 (Phase 1 entry).",
        "",
    ]

    with open(OUTPUT_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\n✓ Written to {OUTPUT_PATH}")
    print("Commit this file before the GPU arrives so the baseline is on main.")


if __name__ == "__main__":
    main()
