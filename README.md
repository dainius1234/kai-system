# Sovereign AI -- kai-system

A secure, operator-controlled personal intelligence platform. Kai can run fully offline for privacy, or safely connect to the internet when you allow it. All network access, data sharing, and external actions are controlled by you -- the operator.

## System Overview

Kai is a modular, secure AI system designed for real-world use:
- **Operator Control:** You decide when Kai can access the internet, update, or interact externally. All network activity is logged and can be blocked or reviewed.
- **Growth & Learning:** Kai can develop and improve when you enable online mode, but defaults to privacy-first offline operation.
- **Safety:** Every action, data flow, and external request is checked by Tool-Gate and Orchestrator. Sandboxes and dashboards provide transparency and control.

**Main Components:**
- **Tool-Gate & Orchestrator:** Final authority and policy enforcement. You set internet and execution rules.
- **Memu-Core:** Memory and context engine. Stores history, feedback, and lessons.
- **LangGraph:** Primary agentic graph/runtime for planning and orchestration.
- **AutoGen, CrewAI, OpenAgents:** Additional agentic frameworks (smoke-tested, available for future multi-agent workflows).
- **Executor & Sandboxes:** Safe execution of approved actions.
- **Dashboard & Output:** Operator interface (FastAPI), TTS, and avatar feedback.

**How it works:**
1. Operator or service makes a request.
2. Tool-Gate and Orchestrator check for safety, policy, and internet rules.
3. Agentic planners use Memu-Core for context and decide next steps.
4. Executor runs approved actions in a sandbox.
5. Results and logs are visible in the Dashboard.

**Internet Access Policy:**
- Default: Offline for privacy and safety.
- Operator can enable online mode for updates, learning, or external tasks.
- All network activity is logged, reviewable, and can be blocked at any time.

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
# Run ALL core unit/smoke tests (~30 tests across all services)
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

## Architecture & Controls

- **Tool-Gate:** Central policy engine for all tool and network access. HMAC-signed inter-service calls.
- **Orchestrator:** Final risk authority before any execution.
- **Memu-Core:** Memory engine with vector search, session buffer, reflection, proactive nudges, and quarantine.
- **LangGraph service:** Graph-based orchestrator with conviction scoring and episode saving (`kai_config.py`).
- **Executor & Sandboxes:** Safe, isolated execution of approved actions.
- **Dashboard:** FastAPI operator console for monitoring, approval, and override.

---

## Personality Modes: PUB & WORK

Kai operates in two distinct personality modes, toggled from the chat UI header:

| Mode | Personality | Use Case |
|------|------------|----------|
| **WORK** | Professional, focused, precise. Proactive but concise. | Tasks, research, planning |
| **PUB** | Casual, witty, opinionated. Real talk like a mate at the pub. | Open discussion, philosophy, banter |

**How it works:**
- The chat UI (`dashboard/static/chat.html`) has WORK/PUB toggle buttons in the header.
- Selected mode is stored in the browser's `localStorage` as `kai-mode`.
- Each chat request sends `{ "message": "...", "mode": "WORK" }` (or `"PUB"`) to `/chat`.
- The langgraph service selects a system prompt from `_SYSTEM_PROMPTS` dict in `langgraph/app.py`.
- If no mode is specified, defaults to **PUB**.
- Mode can also be fetched from tool-gate's `/mode` endpoint for cross-service consistency.

**WORK mode prompt** (summary): Sovereign AI assistant — professional, focused, proactive, concise but thorough. Never lies, never sugarcoats.

**PUB mode prompt** (summary): Genuine mate — casual, witty, opinionated. All topics welcome (politics, science, philosophy, dark humour). Not a service, a companion.

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

## Quickstart

1. Clone the repo.
2. Start core stack: `make core-up`
3. Run tests: `make test-core`
4. Monitor: open dashboard at `http://localhost:8080`
5. Full stack: `make full-up` for all services including LLM, perception, and Telegram.

For architecture details see `docs/sovereign_ai_spec.md`.

---

## Session Continuation Guide

> **For AI assistants resuming work on this codebase. Updated 4 March 2026.**

### Target Hardware
- **Dev/staging:** GitHub Codespace (CPU only, no TPM)
- **Production:** Lenovo laptop, **RTX 5080 GPU**, **TPM 2.0**
- GPU arrival = local LLM inference, real STT/TTS, OMAR self-play
- TPM 2.0 = hardware-anchored Soulbound Identity
- All code must work in BOTH environments (stubs in codespace, live on laptop)

### Current State
- **All 47 test-core targets pass.** Run `make test-core` to confirm.
- **Phase 2 COMPLETE:** Specialist Router, Memory-Driven Planner, Adversary Challenge Engine — all built, tested, wired into `/chat` and `/run`.
- **P1+P2 COMPLETE:** Failure Taxonomy, Metacognitive Rules, SELAUR learning value — all generating real episode metadata.
- **P4+P5+P6 COMPLETE:** Contradiction Memory, GEM Cognitive Alignment, Knowledge Boundary Mapping — full engines in memu-core + kai_config + planner + app.py.
- **P7+P12+P14 COMPLETE:** Silence-as-Signal, Self-Deception Detection, Temporal Self-Model — cognitive self-awareness layer.
- **P10+P11+P13 COMPLETE:** Predictive Pre-Computation, Operator Tempo Modeling, Recursive Self-Improvement Gate — adaptive intelligence layer.
- **P8 COMPLETE:** Dashboard Thinking Pathways — cognitive transparency visualization (conviction pipeline, tempo gauge, boundary map, silence signals, self-assessment).
- **P9 COMPLETE:** Security Self-Hacking — automated security audit engine (injection fuzzing, sanitization, HMAC boundary, policy governance). 6th adversary challenge. Risk scoring.
- **P15 COMPLETE:** Dream State — offline consolidation engine (failure clustering, rule deduplication, pattern synthesis, contradiction detection, boundary recalibration).
- **Merged Action Plan:** Operator's 2026 paper research + AI-native blueprints → 15 prioritised advantages in `docs/unfair_advantages.md`.
- **LLM strategy:** ALL local models via Ollama. Kimi K2 (1T MoE, Apache 2.0) pending addition.

### Key Docs (read in this order)
1. `docs/unfair_advantages.md` — Strategic edge analysis + merged build order + continuation notes
2. `docs/agentic_patterns_spec.md` — Technical spec for the 3-pattern agentic architecture
3. `docs/PROJECT_BACKLOG.md` — Living backlog with hardware constraints
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

### Build Order (merged action plan — tackle in this sequence)

| # | Feature | Files to Touch | Status |
|---|---|---|---|
| P1 | Failure Taxonomy + Metacognitive Rules | kai_config.py, app.py, adversary.py | ✅ DONE |
| P2 | SELAUR (Uncertainty-Aware Evolution) | app.py, kai_config.py | ✅ DONE |
| P3 | Soulbound Identity (HMAC now → TPM on laptop) | pending/soulbound.py, memu-core | Skipped (HMAC works; one env flip on hardware) |
| P4 | TMC + Contradiction Memory | memu-core/app.py, kai_config.py | ✅ DONE |
| P5 | GEM (Cognitive Alignment) | kai_config.py, planner.py, memu-core, app.py | ✅ DONE |
| P6 | Knowledge Boundary + Active Probing | kai_config.py, memu-core/app.py | ✅ DONE |
| P7 | Silence-as-Signal | memu-core/app.py | ✅ DONE |
| P8 | Dashboard: Thinking Pathways | dashboard/app.py, static/thinking.html | ✅ DONE |
| P9 | Security Self-Hacking | langgraph/security_audit.py, adversary.py, app.py | ✅ DONE |
| P15 | Dream State (Offline Consolidation) | langgraph/kai_config.py, app.py, dashboard | ✅ DONE |

Full details + parked items in `docs/unfair_advantages.md`.

P10 (Predictive Pre-Computation), P11 (Operator Tempo), P12 (Self-Deception Detection), P13 (Self-Improvement Gate), and P14 (Temporal Self-Model) also complete — see unfair_advantages.md.

### Cross-Check: What's Been Done
- [x] Phase 2a: router.py (8 routes, 27 classification tests)
- [x] Phase 2b: planner.py (gather_context, build_enriched_plan, preference injection)
- [x] Phase 2c: adversary.py (5 challenges, 7 test groups)
- [x] Merged action plan documented (operator research + AI-native)
- [x] Kimi K2 research noted in LLM strategy
- [x] P1: Failure Taxonomy + Metacognitive Rules (27 tests)
- [x] P2: SELAUR Uncertainty-Aware Evolution (13 tests)
- [x] P4: TMC Contradiction Memory (13 tests)
- [x] P5: GEM Cognitive Alignment (13 + 8 = 21 tests)
- [x] P6: Knowledge Boundary Mapping (6 tests in gem suite)
- [x] P7: Silence-as-Signal (8 tests)
- [x] P10: Predictive Pre-Computation (15 tests)
- [x] P11: Operator Tempo Modeling (12 tests)
- [x] P12: Self-Deception Detection (12 tests)
- [x] P13: Recursive Self-Improvement Gate (20 tests)
- [x] P14: Temporal Self-Model (19 tests)
- [x] P8: Dashboard Thinking Pathways (21 tests)
- [x] P9: Security Self-Hacking (23 tests)
- [x] P15: Dream State (26 tests)
