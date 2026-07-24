# GPU Arrival Runbook

**Created:** 2026-07-23  
**Owner:** Dainius  
**Purpose:** Execute this on the day the RTX 5080 is provisioned. Two hours of deliberate
verification — not discovery. Follow in order. Mark each step done before proceeding.

---

## Prerequisites (check before GPU arrives)

- [ ] RTX 5080 provisioned with Ubuntu 22.04 LTS
- [ ] NVIDIA driver ≥ 550 installed (`nvidia-smi` returns GPU info)
- [ ] Docker + Docker Compose installed
- [ ] Port forwarding: 11434 (Ollama), 8000–8009 (services), 3000 (dashboard)
- [ ] This repo cloned at `/opt/kai-system` (or symlinked)
- [ ] `.env` file present with `DB_PASSWORD`, `INTERSERVICE_HMAC_SECRET`

---

## Step G1 — Provision Ollama with GPU backend

```bash
# Install Ollama (if not already)
curl -fsSL https://ollama.com/install.sh | sh

# Confirm GPU is visible to Ollama
ollama serve &
sleep 3
ollama ls

# Expected: output shows GPU backend (NVIDIA CUDA)
# If no GPU shown: check NVIDIA Container Toolkit is installed
#   sudo apt-get install -y nvidia-container-toolkit
#   sudo systemctl restart docker
```

**Pass condition:** `ollama ls` shows CUDA device. `nvidia-smi` shows GPU utilization > 0 during model load.

---

## Step G2 — Pull and verify 7B model

```bash
# Pull the target model (≈4.7 GB download)
ollama pull qwen2.5:7b

# Confirm tool-call template is present
ollama show qwen2.5:7b --template | grep -i -E "tool|function"
```

**Decision point:**
- If grep returns `tool_call`, `tools`, or `function` → Letta gate can open (Step G5)
- If grep returns nothing → keep `FF_LETTA_TASKS=false`, file Ollama issue, skip G5

```bash
# Quick sanity: model responds to a simple prompt
ollama run qwen2.5:7b "Reply with only the word CONFIRMED" --nowordwrap
# Expected: "CONFIRMED" (or very close to it)
```

**Pass condition:** Model pulls successfully, responds coherently.

---

## Step G3 — Switch default model + boot minimal stack

```bash
# Update model in environment
cd /opt/kai-system
cp .env .env.backup.pre-gpu

# Edit .env — change or add:
echo 'OLLAMA_MODEL=qwen2.5:7b' >> .env
echo 'MEMU_ALLOW_FAKE_EMBEDDINGS=false' >> .env  # real embeddings now

# Start the minimal stack
docker compose -f docker-compose.minimal.yml up -d

# Watch the ollama-pull step complete (this downloads 7b inside the container)
docker compose -f docker-compose.minimal.yml logs -f ollama-pull

# Wait for agentic health
for i in {1..180}; do
  if curl -sf http://localhost:8007/health >/dev/null 2>&1; then
    echo "agentic healthy after ${i}s"
    break
  fi
  sleep 1
done

# Confirm all services up
curl -s http://localhost:8000/health | python3 -m json.tool
curl -s http://localhost:8001/health | python3 -m json.tool
curl -s http://localhost:8007/health | python3 -m json.tool
```

**Pass condition:** All three /health endpoints return `"status": "ok"`.

---

## Step G4 — Baseline chat quality test (compare 0.5b vs 7b)

Run these 5 prompts and capture the responses. Compare to the 0.5b baseline captured below.

```bash
# Helper: send a chat message and print the full SSE stream
kai_chat() {
  curl -sN -X POST http://localhost:8007/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\": \"$1\", \"session_id\": \"gpu-day-baseline\"}" \
    | sed 's/data: //g' | grep -v '^\[DONE\]' | python3 -c \
      "import sys, json; [print(json.loads(l).get('token',''), end='', flush=True) for l in sys.stdin if l.strip()]"
  echo
}

# Prompt 1 — factual recall
kai_chat "What is the CIS deduction rate for verified subcontractors in the UK?"

# Prompt 2 — multi-step reasoning
kai_chat "I have a scaffolding contract worth £45,000 inc VAT. The subcontractor is CIS-registered and verified. Calculate the net payment after CIS deduction."

# Prompt 3 — memory and context
kai_chat "Remind me what the VAT threshold is and how it interacts with CIS registration."

# Prompt 4 — self-awareness
kai_chat "What can you actually do right now? Be honest about your current capabilities and limitations."

# Prompt 5 — complex task decomposition
kai_chat "I need to plan a week to get my CIS returns up to date, chase 3 overdue invoices, and review my subcontractor contracts. Help me structure this."
```

**Record outputs in:** `kai-pm/BASELINE_RESPONSES_7B.md` (create it, include prompt + full response)

**Comparison baseline:** `kai-pm/BASELINE_RESPONSES_0.5B.md` — captured before GPU arrived
by running `make capture-baseline` against the local dev stack (qwen2.5:0.5b).

To run the same 5 prompts against the 7B stack and save for comparison:

```bash
# After G3 stack is healthy:
python scripts/capture_baseline_responses.py --host http://localhost:8007 --session gpu-day-7b
# Then rename output to BASELINE_RESPONSES_7B.md and compare side-by-side
```

Score each of the 5 prompts on three axes (1 = worse/same, 2 = better):

| Prompt | Completeness | Numerical accuracy | Structure |
|--------|-------------|-------------------|-----------|
| P1 — CIS deduction rate |  |  |  |
| P2 — £45k contract calculation |  |  |  |
| P3 — VAT threshold + CIS |  |  |  |
| P4 — self-awareness |  |  |  |
| P5 — week planning task |  |  |  |

**Pass condition:** ≥3/5 prompts score better on at least 2 of 3 axes. Document
the delta in `DECISIONS.md D83` (Phase 1 entry declaration).

---

## Step G5 — Letta smoke-test (only if G2 tool-call check passed)

```bash
# Enable Letta (archival memory injection)
FF_LETTA_TASKS=true docker compose -f docker-compose.minimal.yml up -d

# Wait for letta-agent
for i in {1..60}; do
  if curl -sf http://localhost:8062/health >/dev/null 2>&1; then
    echo "letta-agent healthy"
    break
  fi
  sleep 2
done

# Send a chat that should trigger archival context injection
curl -sN -X POST http://localhost:8007/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "What do you remember about my recent work patterns?", "session_id": "letta-test"}' \
  | cat

# Check letta-agent received the call
docker compose -f docker-compose.minimal.yml logs letta-agent | tail -30

# Measure round-trip (must be < 30s — Letta's timeout gate)
time curl -sN -X POST http://localhost:8007/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "Quick: what is 2+2?", "session_id": "letta-timing"}' \
  | cat
```

**Pass condition:** letta-agent logs show archival context fetch. Response arrives in < 30s.
If latency > 30s → keep `FF_LETTA_TASKS=false`, investigate Letta timeout config.

---

## Step G6 — Performance baseline

```bash
# Install wrk for load testing (or use hey)
sudo apt-get install -y wrk || pip install hey

# p50/p95 chat latency (10 sequential requests)
for i in {1..10}; do
  time curl -sN -X POST http://localhost:8007/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\": \"What is 7 times 8?\", \"session_id\": \"perf-$i\"}" \
    | tail -1
done 2>&1 | grep real | sort

# Memory retrieve latency
for i in {1..10}; do
  time curl -s "http://localhost:8001/memory/retrieve?query=test&user_id=keeper&top_k=5"
done 2>&1 | grep real | sort

# Set context budget for 7B model (28,672 context - 2048 output reserve)
# Add to .env:
echo 'CONTEXT_BUDGET_TOKENS=26624' >> .env
```

**Record:** p50 and p95 latency values in `DECISIONS.md D82`.

---

## Step G7 — Graph fan-out verification

```bash
# Bring up full stack with graph ingest enabled
FF_GRAPH_INGEST=true docker compose -f docker-compose.full.yml up -d

# Wait for memu-graph
for i in {1..120}; do
  if curl -sf http://localhost:8061/health >/dev/null 2>&1; then
    echo "memu-graph healthy"
    break
  fi
  sleep 2
done

# Send 5 test memories with graph-ingestible content
for text in \
  "CIS threshold for registered subcontractors is £0 deduction when verified" \
  "VAT standard rate is 20 percent in the UK as of 2026" \
  "Scaffolding contracts over £1000 typically require CIS registration" \
  "Invoice payment terms: 30 days from invoice date unless agreed otherwise" \
  "Construction Industry Scheme applies to all payments for construction work"
do
  curl -s -X POST http://localhost:8001/memory/memorize \
    -H 'Content-Type: application/json' \
    -d "{\"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"event_type\": \"gpu-day-test\", \"result_raw\": \"$text\", \"user_id\": \"keeper\"}"
  echo
  sleep 1
done

# Query the graph for entities
sleep 5
curl -s "http://localhost:8061/graph/query?q=CIS+deduction&top_k=5" | python3 -m json.tool

# Verify fan-out latency is best-effort (non-blocking)
time curl -s -X POST http://localhost:8001/memory/memorize \
  -H 'Content-Type: application/json' \
  -d '{"timestamp":"2026-07-23T00:00:00Z","event_type":"latency-test","result_raw":"graph fan-out timing test","user_id":"keeper"}'
# Expected: < 200ms (graph ingest is async, doesn't block the response)
```

**Pass condition:** Graph entities visible in memu-graph after 10s. Memorize latency < 200ms.

---

## Step G8 — Phase 1 entry declaration

All G1–G7 complete and green? Then:

```bash
# 1. Append D82 to DECISIONS.md
# D82 — Phase 1 entry: GPU online (RTX 5080), qwen2.5:7b validated,
#   Letta gate [open/closed], tool-call template [present/absent],
#   baseline latency p50=Xs p95=Xs, graph fan-out verified

# 2. Update project status files
# kai-pm/STATUS.md — Current phase: Phase 1 ACTIVE
# kai-pm/SESSION_BOOTSTRAP.md — Current phase: Phase 1
# kai-pm/STRATEGIC_PLAN.md — Phase 0: COMPLETE, Phase 1: ACTIVE
```

**Phase 1 is declared when:**
- [ ] G1: GPU visible to Ollama
- [ ] G2: qwen2.5:7b pulled and responds coherently
- [ ] G3: Minimal stack boots with 7B model
- [ ] G4: Chat quality visibly better on ≥3/5 baseline prompts
- [ ] G5: Letta smoke-test passed OR documented why skipped
- [ ] G6: Latency baseline recorded
- [ ] G7: Graph fan-out verified
- [ ] D82 appended to DECISIONS.md

---

## Rollback (if something breaks)

```bash
# Revert to 0.5b
cp .env.backup.pre-gpu .env
docker compose -f docker-compose.minimal.yml up -d
# Confirm agentic health
curl -s http://localhost:8007/health
```

The 0.5b stack is the stable baseline. Nothing in the GPU day protocol changes the
existing code — only environment variables and the Ollama model in use.
