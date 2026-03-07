# Sovereign AI — kai-system

> *Not a chatbot. Not an agent framework. A sovereign intelligence that grows, reflects, and earns the right to act.*

A self-sovereign, air-gapped personal intelligence platform. Kai runs fully offline by default — all network access, data sharing, and external actions are controlled by the operator. Built with soul, not just syntax.

## What Makes Kai Different

| Capability | What It Means | Commercial AI? |
|---|---|---|
| **Emotional Memory** | Detects 8 emotions in conversation, tracks mood arcs over time | Never |
| **Self-Reflection** | Analyzes its own mistakes, builds strengths/weaknesses journal | Never |
| **Epistemic Humility** | Knows what it doesn't know — warns when confidence is low | Never |
| **Confession Engine** | Proactively admits past mistakes without being asked | Never |
| **Relationship Timeline** | Tracks days together, milestones, emotional journey | Never |
| **Dream State** | Offline consolidation — clusters failures, recalibrates boundaries | Never |
| **Struggle Detection** | Detects operator frustration (5 signals) and offers help | Rarely |
| **Proactive Conversation** | Talks first — greetings, check-ins, goal nudges, drift alerts | Never |
| **Correction Learning** | Learns from mistakes, corrections always surface in evidence | Rarely |
| **Security Self-Hacking** | Fuzzes its own APIs with 34 payloads, scores risk | Never |
| **Operator Sovereignty** | Zero telemetry, zero corporate control, zero resets | Never |

**52 test targets. 532 tests. Zero failures. 25 Docker services. All real.**

---

## System Overview

Kai is modular, secure, and designed for real-world use:
- **Operator Control:** You decide when Kai can access the internet, update, or interact externally. All activity is logged.
- **Growth & Learning:** Kai improves through conversation — correction learning, spaced repetition, proactive memory surfacing.
- **Safety:** Every action is checked by Tool-Gate and Orchestrator. HMAC-signed inter-service calls. Sandboxed execution.
- **Personality:** Two modes — **WORK** (professional, focused) and **PUB** (uncensored friend, real talk like a mate at the pub).

**Main Components:**
- **Tool-Gate & Orchestrator:** Policy enforcement and final risk authority.
- **Memu-Core:** Memory engine — vector search, emotional memory, self-reflection, epistemic humility, goals, relationship tracking.
- **LangGraph:** Agentic brain — routing, planning, conviction scoring, tree search, adversary challenges, EQ context injection.
- **Dashboard:** 8-view operator console — Chat, Dashboard, Thinking, Goals, Memory, Logs, Settings, Soul.
- **Perception:** Telegram bot (voice + text), audio capture (faster-whisper), camera, screen capture.
- **Executor & Sandboxes:** Isolated execution of approved actions.

**How it works:**
1. Operator sends a message (Telegram, Dashboard, or API).
2. LangGraph enriches context: memories + session + goals + topics + emotional intelligence (5-way parallel fetch).
3. Specialist router classifies the domain. Planner builds a gap-aware plan.
4. Adversary challenges the plan (6 challenge types). Conviction scoring gates execution.
5. Tool-Gate checks policy (HMAC, rate limit, co-sign). Executor runs in sandbox.
6. Post-mortem: episode saved, corrections learned, emotion recorded, memory updated.

---

## Repo Structure

```
orchestrator/        # Final risk authority before execution
supervisor/          # Watchdog and circuit-breaker control loop
fusion-engine/       # Multi-signal consensus and conviction gating
verifier/            # Fact-checking and signal cross-validation
executor/            # Execution bridge and order-routing stubs
dashboard/           # Operator console (FastAPI + Starlette)
memu-core/           # Memory engine, vector search, reflection, quarantine
tool-gate/           # Tool access policy, HMAC auth, ledger
langgraph/           # Graph/runtime app integration (router.py, planner.py, adversary.py, conviction.py, kai_config.py)
kai-advisor/         # Self-employment advisor (offline, UK-focused)
telegram-bot/        # Telegram bot interface (text/voice pipeline)
heartbeat/           # System pulse and auto-sleep controller
memory-compressor/   # Memory compression and summarisation
ledger-worker/       # Ledger persistence worker
metrics-gateway/     # Prometheus metrics aggregator
screen-capture/      # Screen capture and OCR service
backup-service/      # Backup and restore service
calendar-sync/       # Calendar synchronisation service
workspace-manager/   # Workspace lifecycle manager
perception/          # Audio and camera capture services
  audio/             # Audio capture and transcription
  camera/            # Camera capture and vision
output/              # Output services
  tts/               # Text-to-speech (edge-tts)
  avatar/            # Avatar generation
sandboxes/           # Ephemeral sandbox environments
  shell/             # Shell sandbox
common/              # Shared utilities (auth, llm, policy, rate_limit)
security/            # HMAC/auth hardening helpers
scripts/             # Operational scripts, tests, and validation
data/                # Seed datasets and local advisor inputs
docs/                # Implementation plans and hardening runbooks
```

---

## Minimal Core Stack

`docker-compose.minimal.yml` starts the 8 core services:

| # | Service    | Purpose                 |
|---|-----------|-------------------------|
| 1 | postgres  | Ledger and vector store |
| 2 | redis     | Session buffer          |
| 3 | tool-gate | Execution choke point   |
| 4 | memu-core | Memory engine           |
| 5 | heartbeat | System pulse            |
| 6 | dashboard | Health UI               |
| 7 | supervisor| Watchdog                |
| 8 | verifier  | Fact-checking           |

The full stack (`docker-compose.full.yml`) adds: `fusion-engine`, `langgraph`, `executor`, `orchestrator`, `memory-compressor`, `ledger-worker`, `metrics-gateway`, `audio-service`, `camera-service`, `kai-advisor`, `tts-service`, `avatar-service`, `screen-capture`, `telegram-bot`, `backup-service`, `calendar-sync`, `workspace-manager`, and Ollama (local LLM).

---

## Quick Start

```bash
make setup            # check deps, create .env, build images
# Edit .env with your API keys
make core-up          # start core services
# Open http://localhost:8050/app
```

---

## Build & Run

```bash
# Build the minimal stack
docker compose -f docker-compose.minimal.yml build

# Build the full stack
docker compose -f docker-compose.full.yml build

# Start/stop
make core-up          # minimal stack
make core-down
make full-up          # full stack
make full-down
```

---

## Test

```bash
# Run ALL core unit/smoke tests (52 targets, ~532 tests)
make test-core

# Individual service tests
make test-phase-b-memu       # memu-core unit tests
make test-memu-pg            # memu pgvector tests (requires PG_URI)
make test-dashboard-ui       # dashboard UI tests
make test-dashboard          # dashboard structural tests
make test-tool-gate          # tool-gate API tests
make test-tool-gate-security # tool-gate HMAC/nonce security tests
make test-telegram           # telegram-bot smoke test
make test-audio              # audio service smoke test
make test-camera             # camera service smoke test
make test-executor           # executor service smoke test
make test-langgraph          # langgraph service smoke test
make test-kai-advisor        # kai-advisor unit tests
make test-tts                # TTS service smoke test
make test-avatar             # avatar service smoke test
make test-heartbeat          # heartbeat service tests
make test-conviction         # conviction scoring tests
make test-self-emp           # self-employment advisor tests
make test-auth-hmac          # HMAC auth hardening tests
make test-agentic            # agentic framework integration tests
make test-episode-saver      # episode saver fallback tests
make test-episode-spool      # episode spool integrity tests
make test-error-budget       # error budget breaker tests
make test-invoice            # invoice tests
make test-memu-retrieval     # memu retrieval tests
make test-router             # specialist router classification tests
make test-planner            # memory-driven planner tests
make test-adversary          # adversary challenge engine tests (6 challenges)
make test-dream-state        # P15 dream state consolidation tests
make test-security-audit     # P9 security self-hacking audit tests
make test-gaps-sprint        # JSON logging, vector cleanup, ledger stats tests
make test-tree-search        # HP4 CoT tree search tests
make test-priority-queue      # HP5 priority queue tests
make test-model-selector      # HP2 MoE model selector tests
make test-prod-hardening     # production hardening (secrets, pubsub, backup, HMAC)
make test-hmac-rotation-drill # HMAC rotation lifecycle drill
make test-p3-organic          # P3 organic memory (goals, drift, decay, proactive)
make test-p4-personality       # P4 personality & proactive (prompts, anti-annoyance, topics, modes)
make test-p16-operational      # P16 operational intelligence (struggle, feedback, logs, goals UI, memory browser)
make test-p17-emotional-intelligence # P17 emotional intelligence (emotional memory, self-reflection, relationship, epistemic humility, confession)

# v7 feature tests
make test-v7                 # verifier, quarantine, policy, idempotency, integration-chain

# Integration (requires running stack)
make core-smoke
make test-integration

# Full merge-gate check
make merge-gate
```

---

## Lint / Validate

```bash
# Syntax check all service entry points
make go_no_go

# Flake8
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

Run `make go_no_go` before committing to catch syntax errors early.

---

## Operator Console

Open **http://localhost:8050/app** for the unified operator interface.

| View | Shortcut | Description |
|------|----------|-------------|
| **Chat** | `Ctrl+1` | Conversational interface — streaming SSE, markdown, PUB/WORK toggle, feedback ratings, struggle detection |
| **Dashboard** | `Ctrl+2` | Service health grid, pipeline status, fusion metrics |
| **Thinking** | `Ctrl+3` | Live thinking-pathway trace — conviction pipeline, tempo gauge, boundary map, silence signals, dream state, security audit |
| **Settings** | `Ctrl+4` | Mode, notifications, markdown, keyboard shortcuts, PWA install |
| **Goals** | `Ctrl+5` | Ohana goal tracker — create/update goals, drift alerts, progress bars, feedback stats |
| **Memory** | `Ctrl+6` | Memory browser — search by query or category, stats overview, results with scores |
| **Logs** | `Ctrl+7` | Log aggregator — level filter, monospace viewer, time/level/service/msg columns |
| **💎 Soul** | `Ctrl+8` | Emotional intelligence — mood cards, emotion timeline, domain confidence, self-reflection journal, relationship milestones |

**Additional shortcuts:**

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Focus chat input |
| `Ctrl+Shift+M` | Toggle PUB/WORK mode |
| `Escape` | Close dropdown / stop generation |

**PWA:** Kai can be installed as a standalone app on desktop and mobile via Chrome/Edge. The manifest and icons are served from `/static/`.

---

## Personality Modes: PUB & WORK

| Mode | Personality | Use Case |
|------|------------|----------|
| **WORK** | Professional, focused, precise. Proactive but concise. Never lies, never sugarcoats. | Tasks, research, planning |
| **PUB** | Genuine mate — casual, witty, opinionated. All topics welcome. Not a service, a companion. | Open discussion, philosophy, banter |

Toggled from the chat UI header or via `Ctrl+Shift+M`. Stored in `localStorage`. Time-of-day auto-schedule: WORK 08-18 Mon-Fri, PUB otherwise. Manual override lasts 4h. Mode from tool-gate `/gate/mode` for cross-service consistency.

---

## Key Guidelines

- Always add or update `requirements.txt` when introducing new Python dependencies.
- When modifying a service `app.py`, run `make go_no_go` to catch syntax errors.
- Every service must expose a `/health` HTTP endpoint returning `{"status": "ok"}`.
- Prefer small, focused pull requests. Use `make merge-gate` to confirm checks pass.
- HMAC auth: Set `TOOL_GATE_DUAL_SIGN=true` and `INTERSERVICE_HMAC_STRICT_KEY_ID=true` after overlap stabilises.
- Never commit real credentials; use `.env` files and environment variables.
- The `langgraph/` directory contains the local orchestrator service; the installed `langgraph` pip package is separate.

---

## Session Continuation Guide

> **For AI assistants resuming work on this codebase. Updated 7 March 2026.**

### Target Hardware
- **Dev/staging:** GitHub Codespace (CPU only, no TPM)
- **Production:** Lenovo laptop, **RTX 5080 GPU**, **TPM 2.0**
- GPU arrival = local LLM inference, real STT/TTS, multi-model consensus
- All code must work in BOTH environments (stubs in codespace, live on laptop)

### Current State
- **52 test-core targets pass, ~532 individual tests.** Run `make test-core` to confirm.
- **P0-P5 COMPLETE:** All 25 services built, running, tested. CI/CD, secrets, backup, HMAC rotation.
- **P7-P15 COMPLETE:** Agentic patterns, thinking pathways, security self-hacking, dream state.
- **P3 Organic Memory COMPLETE:** Correction learning, category boost, spaced repetition, proactive engine, Ohana goals, drift detection.
- **P4 Personality COMPLETE:** Deep PUB/WORK prompts, anti-annoyance, conversation holding, mode transitions, greeting/check-in.
- **P16 Operational Intelligence COMPLETE:** Struggle detection (5-signal), feedback loop (1-5 stars), log aggregation, Goals/Memory/Logs views.
- **P17 Emotional Intelligence COMPLETE:** Emotional memory (8 emotions), self-reflection journal, relationship timeline, epistemic humility, confession engine, Soul dashboard.
- **Dashboard:** 8 views (Chat, Dashboard, Thinking, Settings, Goals, Memory, Logs, Soul).
- **LLM:** Ollama with qwen2:0.5b on CPU. GPU arrival = 3 env vars changed.

### Key Docs (read in this order)
1. `docs/PROJECT_BACKLOG.md` — Living backlog with session notes and all completed phases
2. `docs/unfair_advantages.md` — Strategic edge analysis + hardware performance track
3. `docs/agentic_patterns_spec.md` — Technical spec for the agentic architecture
4. This README for repo structure and test commands

### Agentic Pipeline (how /run works)
```
Request → injection filter → specialist selection → session buffer
  → gather_context() (memory + episodes + corrections + nudges + preferences)
  → build_enriched_plan() (history + conviction modifiers + preference constraints)
  → challenge_plan() (6 adversary challenges in parallel)
  → conviction scoring (5-signal + planner modifier + adversary modifier)
  → rethink loop (if conviction < 8.0, max 3 retries)
  → tool-gate policy check (HMAC, rate limit, co-sign)
  → executor (sandboxed)
  → post-mortem (episode save, correction learning, preference extraction, auto-memorize)
```

### Build Order (merged action plan)

| # | Feature | Status |
|---|---|---|
| P0-P2 | Stack, Perception, Voice | ✅ DONE |
| P3 | Organic Memory (spaced rep, proactive, goals, drift) | ✅ DONE |
| P4 | Full-Stack Personality & Proactive Conversation | ✅ DONE |
| P5 | Production Hardening (CI, secrets, backup, HMAC) | ✅ DONE |
| P7 | Agentic Patterns (episodes, error budget, router, planner) | ✅ DONE |
| P8 | Dashboard Thinking Pathways (6 visualization cards) | ✅ DONE |
| P9 | Security Self-Hacking (34 payloads, 6 adversary challenges) | ✅ DONE |
| P10-P14 | Adaptive Intelligence (predictive, tempo, self-deception, improvement gate, temporal self) | ✅ DONE |
| P15 | Dream State (6-phase offline consolidation) | ✅ DONE |
| P16 | Operational Intelligence (struggle, feedback, logs, 3 views) | ✅ DONE |
| P17 | Emotional Intelligence (emotional memory, self-reflection, epistemic humility, confession, Soul) | ✅ DONE |
| P6 | Nice-to-have (calendar sync, workspace manager, avatar, Prometheus) | Queued |
| HP1-HP6 | Hardware Performance Track (speculative decoding, VRAM watchdog, NVMe offload) | Awaiting GPU |

Full details in `docs/PROJECT_BACKLOG.md` and `docs/unfair_advantages.md`.

### Cross-Check: What's Been Done
- [x] All 25 services built, running, health-checked
- [x] Real LLM (Ollama qwen2:0.5b), real persistence (pgvector + Redis)
- [x] Real input (Telegram voice + text), real output (edge-tts British Ryan)
- [x] Specialist router (8 categories), memory-driven planner, adversary engine (6 challenges)
- [x] Conviction scoring (5-signal), tree search (CoT pruning), priority queue
- [x] Deep personality (PUB/WORK), proactive conversation (greetings, check-ins, nudges)
- [x] Anti-annoyance (cooldowns, DND, dismissal escalation)
- [x] Dream state (6-phase consolidation), security self-hacking (34 payloads)
- [x] Struggle detection (5-signal), feedback loop (1-5 stars), log aggregation
- [x] Emotional memory (8 emotions), self-reflection journal, relationship timeline
- [x] Epistemic humility (domain confidence), confession engine
- [x] 8 dashboard views (Chat, Dashboard, Thinking, Settings, Goals, Memory, Logs, Soul)
- [x] HMAC auth, Ed25519 signing, episode saver, error budget breaker
- [x] 52 test targets, 532 tests, zero failures
