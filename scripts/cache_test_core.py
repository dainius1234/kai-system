"""
Cache and compare test-core results for Kai System.
- On run: executes all test-core targets, records results and timestamp.
- On subsequent runs: only re-runs targets that changed since last run.
- Stores results in scripts/test_core_results.json
"""
import subprocess
import json
import os
import time

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "test_core_results.json")

# List of test targets (should match Makefile)
TEST_TARGETS = [
    "test-phase-b-memu", "test-memu-pg", "test-dashboard-ui", "test-dashboard", "test-thinking-pathways", "test-tool-gate", "test-tool-gate-security", "test-telegram", "test-conviction", "test-audio", "test-camera", "test-executor", "test-langgraph", "test-kai-advisor", "test-tts", "test-avatar", "test-heartbeat", "test-episode-saver", "test-episode-spool", "test-error-budget", "test-invoice", "test-memu-retrieval", "test-router", "test-planner", "test-adversary", "test-failure-taxonomy", "test-selaur", "test-self-emp", "test-auth-hmac", "test-agentic", "test-v7", "test-contradiction", "test-mars-consolidation", "test-p3-organic", "test-p4-personality", "test-p16-operational", "test-p17-emotional-intelligence", "test-p18-narrative-identity", "test-p19-imagination-engine", "test-p20-conscience-values", "test-p21-proactive-agent", "test-p22-operator-model", "test-h1-hardening", "test-h2-self-healing", "test-mars-consolidation", "test-sage-critique", "test-agent-evolver", "test-checkpoint", "test-error-codes", "test-feature-flags"
]

def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {"last_run": None, "results": []}
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)

def save_results(data):
    with open(RESULTS_PATH, "w") as f:
        json.dump(data, f, indent=2)

def run_target(target):
    print(f"Running {target}...")
    try:
        result = subprocess.run(["make", target], capture_output=True, text=True, timeout=300)
        return {
            "target": target,
            "returncode": result.returncode,
            "stdout": result.stdout[-1000:],
            "stderr": result.stderr[-1000:],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
    except Exception as e:
        return {
            "target": target,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }

def main():
    results = load_results()
    new_results = []
    for target in TEST_TARGETS:
        res = run_target(target)
        new_results.append(res)
        save_results({"last_run": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": new_results})
    print("All test-core targets run. Results cached.")

if __name__ == "__main__":
    main()
