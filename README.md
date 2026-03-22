<p align="center">
  <b>S O V E R E I G N &nbsp; A I</b><br>
  <code>kai-system</code>
</p>

<p align="center">
  <em>"Not a chatbot. Not an agent framework. A sovereign intelligence that grows, reflects, and earns the right to act."</em>
</p>

<p align="center">
  <a href="https://github.com/dainius1234/kai-system/actions/workflows/core-tests.yml"><img src="https://github.com/dainius1234/kai-system/actions/workflows/core-tests.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/dainius1234/kai-system/actions/workflows/python-app.yml"><img src="https://github.com/dainius1234/kai-system/actions/workflows/python-app.yml/badge.svg" alt="Lint"></a>
  <img src="https://img.shields.io/badge/services-26-blue?style=flat-square" alt="services">
  <img src=\"https://img.shields.io/badge/tests-1%2C587_passing-brightgreen?style=flat-square\" alt=\"tests\">
  <img src="https://img.shields.io/badge/Python-~41%2C107_LOC-yellow?style=flat-square" alt="loc">
  <img src="https://img.shields.io/badge/milestones-31_shipped-purple?style=flat-square" alt="milestones">
  <img src="https://img.shields.io/badge/failures-0-brightgreen?style=flat-square" alt="failures">
  <img src="https://img.shields.io/badge/license-private-red?style=flat-square" alt="license">
</p>

---

## Project Status (22 March 2026)

| Metric | Value |
|---|---|
| **Services** | 26 Docker containers |
| **Test targets** | 88 (`make test-core`) |
| **Individual tests** | 1,609 (`def test_` across 83 files) |
| **Python LOC** | ~41,107 |
| **Compose files** | 3 (minimal / full / sovereign) |
| **Milestones shipped** | 31 |
| **Failures** | 0 |

> **Auto-synced** by `make sync-docs`. Stale metrics block `make merge-gate`.

---

## Quick Reference

```
make core-up          # Start minimal stack (8 services)
make core-down        # Stop it
make full-up          # Start all 26 services
make test-core        # Run all 89 test targets (~1,587 tests)
make go_no_go         # Syntax check all entry points
make merge-gate       # Full pre-merge validation
make sync-docs        # Auto-update README + backlog metrics
make dep-audit        # CVE scan on pip packages
make coverage         # pytest-cov HTML report
```

---

## What Makes Kai Different

> Every capability below has code and tests. Reasoning quality depends on the LLM model — see [Honest Limitations](#honest-limitations) below.

### Soul & Inner Life

| Capability | What It Does |
|---|---|
| **Emotional Memory** | Detects 8 emotions in conversation, tracks mood arcs over time, surfaces emotional continuity |
| **Self-Reflection** | Analyzes its own mistakes, builds a strengths/weaknesses journal, knows where it fails |
| **Epistemic Humility** | Knows what it doesn't know — warns operator when confidence is low |
| **Confession Engine** | Proactively admits past mistakes without being asked |
| **Narrative Identity** | Builds its own life story — autobiography, story arcs, future self projection, legacy time-capsules |
| **Imagination Engine** | Counterfactual replay, theory of mind, creative synthesis, inner monologue, aspirational futures |
| **Conscience & Values** | Emergent value formation, moral reasoning, integrity tracking, loyalty memory, gratitude engine |
| **Dream State** | 6-phase offline consolidation — failure clustering, boundary recalibration, MARS memory decay |

### Intelligence & Reasoning

| Capability | What It Does |
|---|---|
| **10-Way Context** | Every response enriched with: memories + session + goals + topics + EQ + narrative + imagination + conscience + agent + operator model |
| **Specialist Router** | Classifies queries into 8 UK construction domains for category-aware retrieval |
| **Memory-Driven Planner** | Gap-aware plans with preference constraints and history-informed conviction modifiers |
| **Adversary Engine** | 7 challenge types (incl. SAGE self-review) test every plan before execution |
| **Conviction Scoring** | 5-signal + modifiers gate; below 8.0 triggers rethink (max 3 retries) |
| **SAGE Critique** | Verifier self-critique + adversary self-review — AI arguing with itself for quality |
| **Agent-Evolver** | Learns from failure clusters during dream cycles, generates proactive fix insights |
| **Tree Search** | Chain-of-thought pruning with priority queue + counterargument debate gating |

### Operator Relationship

| Capability | What It Does |
|---|---|
| **Operator Model** | Echo-response engine, nudge escalation ladder (4-tier), cross-mode insight bridge |
| **Impact Oracle** | Predicts consequences of actions on goals and emotions — "if you skip X, Y suffers" |
| **Shadow Branches** | Persistent what-if timelines from counterfactuals, queryable alternate histories |
| **Proactive Agent** | Scheduled tasks, reminders, morning briefing, evening check-in, 13-action registry |
| **Struggle Detection** | 5-signal frustration analysis — auto-adapts when operator is struggling |
| **Anti-Annoyance** | Per-type cooldowns, dismissal tracking, DND mode, escalating suppression |
| **PUB/WORK Modes** | Deep personality system — mate at the pub vs. focused professional |

### Production & Security

| Capability | What It Does |
|---|---|
| **Self-Healing** | Deep `/health` + `/recover` + supervisor auto-heal loop across all services |
| **Recovery Log** | Every self-heal event logged to conscience — ties resilience to narrative |
| **Security Self-Hacking** | Fuzzes own APIs with 34 payloads, adversary challenges, SAGE self-review |
| **HMAC Auth** | Inter-service HMAC signing, Ed25519, dual-sign rotation, nonce replay protection. Dev secret requires explicit `HMAC_ALLOW_DEV_SECRET=true` |
| **Time-Travel Debug** | Checkpoint any state, diff between snapshots, rollback to any previous state |
| **Feature Flags** | 13 capabilities toggleable via `FF_` env vars without code changes |
| **Structured Errors** | 20 enumerated codes (E1001–E4004) — no more "something broke" |
| **Zero Telemetry** | No corporate control, no data exfiltration, no resets. Ever. |
| **Skills Hub** | Hot-loadable .md skill files with security scanning, TTL pruning, and unload |
| **Multi-modal Fusion** | Combined audio + video signals interpreted by LLM for proactive voice |
| **World Anchor** | Daily date/time/context fetch cached locally — Kai knows what day it is |

---

## Honest Limitations

> No illusions. Here's what's real and what's waiting.

| Area | Reality | What Fixes It |
|---|---|---|
| **LLM Model** | Default `qwen2:0.5b` (~400M params) is a test placeholder — too small for meaningful reasoning, planning, or emotional intelligence | Upgrade to 7B+ model (`qwen2.5:7b`, `llama3:8b`). RTX 5080 laptop = 3 env vars to switch. Model registry auto-adapts context, prompts, timeouts |
| **Specialist Routing** | Keyword regex classification, not ML-based. All 3 "specialists" route to the same Ollama endpoint by default | Wire separate model endpoints when GPU hardware arrives |
| **Fusion Consensus** | ~~Uses Jaccard word-overlap~~ **Upgraded**: embedding cosine similarity via sentence-transformers (auto-fallback to Jaccard). Configurable via `FUSION_AGREEMENT` | Install sentence-transformers on GPU box for full accuracy |
| **Token Counting** | ~~±40% heuristic (4 chars per token)~~ **Fixed**: tiktoken-based accurate counting + per-message overhead | Already done — tiktoken installed |
| **Context Budget** | ~~Hardcoded 3072 tokens wastes 90% of larger models~~ **Fixed**: auto-adapts from model registry. qwen2:0.5b→3072, qwen2.5:7b→28672, kimi→122K | Already done — model-aware |
| **Prompt Templates** | ~~Hardcoded strings~~ **Fixed**: model-aware templates. Tier 1 (tiny): minimal. Tier 2 (7B): reasoning guidelines. Tier 3 (70B): JSON hints + deep persona | Already done — scales automatically |
| **Test Style** | 1,587 tests verify structural correctness + 37 chassis tests + 15 behavioral tests. Most do NOT test whether the AI is actually smart — they test the plumbing | Add more behavioral tests as model quality improves |
| **Dashboard** | Chat, Health, Mode toggle, Canvas are functional. Other views (Thinking, Goals, Memory, Soul, Diary, Logs) are **proxy shells** — they work when backends are running but show "unavailable" in minimal stack | Views become live with `make full-up` |
| **Memory Persistence** | Minimal stack now uses pgvector (fixed). Full persistence requires `make full-up` or setting `VECTOR_STORE=postgres` + `PG_URI` | Default is now correct |
| **Security Defaults** | HMAC enforced, but DB password is `localdev` by default. Nonce replay persisted to file. Dev HMAC secret now blocked unless explicitly allowed | Set `DB_PASSWORD`, `INTERSERVICE_HMAC_SECRET` env vars for production |
| **Coverage** | ~60% estimated. Dashboard proxy endpoints and memu-core complex paths have gaps | `.coveragerc` configured; `make coverage` tracks it |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  OPERATOR INPUT                                                      │
| Telegram Bot ─── Dashboard (10 views) ─── API Direct                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  INTELLIGENCE LAYER (LangGraph)                                      │
│                                                                      │
│  ┌─ 10-way parallel context fetch ──────────────────────────────┐    │
│  │ memories│session│goals│topics│EQ│narrative│imagination│       │    │
│  │ conscience│agent│operator_model                               │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  → Specialist Router → Planner → Adversary (7 challenges)            │
│  → Conviction Scoring → Tree Search → Agent-Evolver                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  SAFETY & POLICY                                                     │
│  Tool-Gate (HMAC, rate limit) → Orchestrator → Verifier (SAGE)       │
│  Supervisor (watchdog, circuit breaker, auto-heal every 15s)         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  MEMORY & STATE                                                      │
│  Memu-Core → PostgreSQL/pgvector (vector search, MARS decay)         │
│  Redis (session buffer) → Ledger-Worker (audit trail)                │
│  Memory-Compressor → Backup-Service (pg/redis/memory)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  EXECUTION & I/O                                                     │
│  Executor (sandboxed) → TTS (British Ryan) → Avatar                  │
│  Telegram (voice+text) → Audio (faster-whisper) → Camera → Screen    │
└─────────────────────────────────────────────────────────────────────┘
```

### How a Message Flows

```
1. Operator sends message (Telegram / Dashboard / API)
2. LangGraph: injection filter → 10-way context fetch → specialist router
3. Planner builds gap-aware plan → Adversary challenges it (7 types + SAGE)
4. Conviction scoring: if < 8.0, rethink (max 3 retries)
5. Tool-Gate: HMAC + rate limit + policy → Executor: sandboxed run
6. Post-mortem: episode saved, corrections learned, emotion recorded, memory updated
7. Response streamed back (SSE)
```

### Self-Healing Flow

```
Supervisor (every 15s) → deep /health on each service
  ├─ "ok"       → record in fleet history, circuit stays closed
  ├─ "degraded" → open circuit → POST /recover → service self-heals
  │               → log recovery to conscience (what healed, what learned)
  │               → next sweep: "ok" → circuit closes
  └─ unreachable → Telegram alert → operator intervenes
```

---

## Service Map

### Minimal Core Stack (`docker-compose.minimal.yml`)

| # | Service | Port | Purpose |
|---|---------|------|---------|
| 1 | postgres | 5432 | pgvector DB — memories, ledger, embeddings |
| 2 | redis | 6379 | Session buffer, caches |
| 3 | tool-gate | 8010 | Policy enforcement, HMAC auth |
| 4 | memu-core | 8020 | Memory engine — the soul |
| 5 | heartbeat | 8090 | System pulse, auto-sleep |
| 6 | dashboard | 8050 | 8-view operator console |
| 7 | supervisor | 8100 | Watchdog, auto-heal |
| 8 | verifier | 8060 | Semantic fact-checking, SAGE |

### Full Stack Additions (`docker-compose.full.yml`)

| Service | Port | Purpose |
|---------|------|---------|
| langgraph | 8030 | Agentic brain, all reasoning |
| executor | 8040 | Sandboxed execution |
| fusion-engine | 8070 | Multi-signal consensus |
| orchestrator | 8080 | Final risk authority |
| telegram-bot | 8110 | Telegram interface |
| kai-advisor | 8120 | Self-employment advisor (UK) |
| memory-compressor | — | Memory summarisation |
| ledger-worker | — | Action audit trail |
| metrics-gateway | 9090 | Prometheus metrics |
| audio-service | 8021 | STT (faster-whisper) |
| camera-service | — | Camera capture |
| tts-service | 8030 | Text-to-speech (British Ryan) |
| avatar-service | — | Avatar generation |
| screen-capture | — | Screen OCR pipeline |
| backup-service | — | pg/redis/memory backup |
| calendar-sync | — | Calendar integration |
| workspace-manager | — | Workspace lifecycle |
| ollama | 11434 | Local LLM (qwen2:0.5b CPU) |

---

## Operator Console

**http://localhost:8050/app** — 8 views, keyboard shortcuts, installable as PWA.

| View | Key | What You See |
|------|-----|-------------|
| **Chat** | `Ctrl+1` | Streaming conversation, PUB/WORK toggle, feedback ratings, struggle detection |
| **Dashboard** | `Ctrl+2` | Service health grid, pipeline status, fusion metrics |
| **Thinking** | `Ctrl+3` | Live conviction pipeline, tempo gauge, boundary map, silence signals, dream state |
| **Settings** | `Ctrl+4` | Mode, notifications, markdown toggle, PWA install |
| **Goals** | `Ctrl+5` | Ohana goals, drift alerts, progress bars, reminders, scheduled tasks |
| **Memory** | `Ctrl+6` | Memory browser — search by query or category, scores, stats |
| **Logs** | `Ctrl+7` | Ring-buffer log viewer — level/time filter, monospace, color-coded |
| **Soul** | `Ctrl+8` | Mood cards, emotion timeline, domain confidence, self-reflection journal, milestones |

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Focus chat input |
| `Ctrl+Shift+M` | Toggle PUB/WORK mode |
| `Escape` | Close dropdown / stop generation |

---

## Known Issues & Honest State

> We built 24 milestones fast. Now we're building them right.

### Fixed

| Area | Issue | Fix |
|---|---|---|
| memu-core | 13+ race conditions (global state, no asyncio.Lock) | H1 |
| langgraph /chat | No prompt injection check | H1 |
| langgraph 10-way fetch | No error handling on parallel gather | H1 |
| memu-core feedback | Called non-existent store.memorize() | H1 |
| executor | shell=True allows command chaining | H1 |
| telegram-bot | Voice file download with no size limit | H1 |
| dashboard | 50+ proxy endpoints with no try/except | H1 |
| All services | /health returns 200 without checking deps | H2 |
| Inter-service calls | No retry, no circuit breaker, no fallback | H2 |
| Supervisor | Advisory-only circuit breakers (can't heal) | H2 |
| Background tasks | Frozen loops undetectable | H2 |
| memu-core memory | Basic Ebbinghaus (no stability, no pruning) | MARS |
| verifier | Keyword-only (no self-critique, no semantic) | P23 SAGE + semantic upgrade |
| Failure learning | No pattern extraction | P24 Evolver |
| State debugging | No crash recovery, no snapshots | H3b |
| Error messages | Ad-hoc HTTP statuses | Gap-Close |
| Feature toggles | Hardcoded, no env overrides | Gap-Close |
| Dep scanning | No CVE checks | Gap-Close |
| Pre-commit | No automated quality checks | Gap-Close |
| Conscience log | Race condition on /conscience/check write | Session fix |
| Recovery log | Missing schema fields for audit compat | Session fix |

### Open

| Area | Issue | Status |
|---|---|---|
| verifier semantic | ~~Keyword matcher, not embedding-based~~ | **Done** — semantic rank_score (embedding + relevance + importance + recency) |
| context budget | ~~System prompt can grow unbounded~~ | **Done** — `_trim_context()` enforces `CONTEXT_BUDGET_TOKENS` (default 3072) |
| minimal pgvector | ~~InMemoryVectorStore loses data on restart~~ | **Done** — minimal stack now uses `pgvector/pgvector:pg15` + `VECTOR_STORE=postgres` |
| dev secret hardcoded | ~~Silently falls back to dev HMAC secret~~ | **Done** — requires explicit `HMAC_ALLOW_DEV_SECRET=true` env var |
| behavioral tests | ~~Tests verify JSON shape, not reasoning quality~~ | **Done** — 15 behavioral + 8 Docker e2e tests added |
| dashboard UX | ~~Basic web UI, not native-feeling~~ | **Done** — Apple/glassmorphism redesign (SF Pro, backdrop-blur, smooth transitions) |
| test coverage | ~60% estimated | Tracking (`.coveragerc` added) |

---

## Milestone History

> 31 shipped. Zero skipped. Every milestone has tests. Quality of AI reasoning depends on model — see [Honest Limitations](#honest-limitations).

```
P0  Stack runs              ██████████ DONE   P14 Temporal Self       ██████████ DONE
P1  Perception (senses)     ██████████ DONE   P15 Dream State         ██████████ DONE
P2  Voice (output)          ██████████ DONE   P16 Operational Intel   ██████████ DONE
P3  Organic Memory          ██████████ DONE   P17 Emotional Intel     ██████████ DONE
P4  Personality & Proactive ██████████ DONE   P18 Narrative Identity  ██████████ DONE
P5  Production Hardening    ██████████ DONE   P19 Imagination Engine  ██████████ DONE
P7  Agentic Patterns        ██████████ DONE   P20 Conscience & Values ██████████ DONE
P8  Thinking Pathways       ██████████ DONE   P21 Proactive Agent     ██████████ DONE
P9  Security Self-Hacking   ██████████ DONE   P22 Operator Model      ██████████ DONE
P10 Predictive Coding       ██████████ DONE   H1  Hardening Sprint    ██████████ DONE
P11 Reasoning Tempo         ██████████ DONE   H2  Self-Healing        ██████████ DONE
P12 Self-Deception Detector ██████████ DONE   H3b Checkpointing       ██████████ DONE
P13 Improvement Gate        ██████████ DONE   MARS Memory Consol.     ██████████ DONE
─── ─────────────────────── ────────── ────   P23 SAGE Critique       ██████████ DONE
J1–J7 Jewels (7 features)  ██████████ DONE   P24 Agent-Evolver       ██████████ DONE
P1–P5 Enhancements         ██████████ DONE   GC  Eng. Gap-Close      ██████████ DONE
H3  Context Budget          ██████████ DONE
```

### What's Next

| Priority | Feature | Status |
|---|---|---|
| **J1** | Live Canvas Visualization | **DONE** — mind-map/graph in dashboard |
| **J2** | Wake-word "Kai" + Intent Judge | **DONE** — whisper + tiny LLM intent |
| **J3** | Auto-Redaction PII | **DONE** — regex strip before processing |
| **J4** | Proactive Low-Latency Voice | **DONE** — audio/video cue → speak-or-not |
| **J5** | Memory Viewer GUI | **DONE** — diary-style dashboard tab |
| **J6** | SOUL.md + AGENTS.md | **DONE** — persistent identity files |
| **J7** | Skills Auto-Install Hub | **DONE** — local .md skill loader + security scan |
| **P1** | Skill Security + TTL | **DONE** — scan/unload/prune endpoints |
| **P2** | Multi-modal LLM Fusion | **DONE** — interpret_multi + /proactive/interpret |
| **P3** | World Anchor | **DONE** — heartbeat /world endpoint |
| **P4** | Debate Branching | **DONE** — counterargument tree search |
| **P5** | Deprecation Cleanup | **DONE** — 110+ warnings eliminated |
| **H3** | Context Budget Manager | **DONE** — `_trim_context()` + `CONTEXT_BUDGET_TOKENS` env var |
| **P29** | Financial Awareness | Planned — savings tracker, expense categorization |
| **GPU** | Hardware Performance | Planned — multi-model, real STT/TTS, speculative decoding |

*Sources: OpenClaw, Jarvis variants, Proact-VL (arXiv:2603.03447). All offline, low-resource, test on qwen2:0.5b first.*

---

## Repo Structure

```
orchestrator/        # Final risk authority before execution
supervisor/          # Dual-layer watchdog, circuit-breaker, self-heal
fusion-engine/       # Multi-signal consensus and conviction gating
verifier/            # Semantic fact-checking (embedding + keyword), SAGE self-critique
executor/            # Sandboxed execution bridge
dashboard/           # 10-view operator console (FastAPI + Starlette)
memu-core/           # Memory engine — the soul (~6,100 lines)
tool-gate/           # HMAC auth, rate limit, policy enforcement
langgraph/           # Agentic brain (router, planner, adversary, conviction, config)
kai-advisor/         # Self-employment advisor (offline, UK-focused)
telegram-bot/        # Telegram bot (voice + text pipeline)
heartbeat/           # System pulse and auto-sleep
memory-compressor/   # Memory compression and summarisation
ledger-worker/       # Ledger persistence worker
metrics-gateway/     # Prometheus metrics aggregator
screen-capture/      # Screen capture and OCR
backup-service/      # Backup and restore (pg/redis/memory)
calendar-sync/       # Calendar synchronisation
workspace-manager/   # Workspace lifecycle manager
perception/          # Audio and camera capture
  audio/             # STT (faster-whisper tiny, CPU)
  camera/            # Camera capture
output/              # Output services
  tts/               # Text-to-speech (edge-tts British Ryan)
  avatar/            # Avatar generation
sandboxes/           # Ephemeral sandbox environments
  shell/             # Shell sandbox
common/              # Shared: auth, llm, policy, rate_limit, resilience, errors, feature_flags
security/            # HMAC/auth hardening helpers
scripts/             # Tests, validation, automation (~68 test files)
data/                # Seed datasets and advisor inputs
docs/                # Plans, runbooks, architecture, backlog
```

---

## Engineering Toolchain

| Tool | Purpose | Command | Auto? |
|---|---|---|---|
| **sync-docs** | Patch README/backlog metrics from codebase scan | `make sync-docs` | On demand, gates merge |
| **check-docs** | Read-only freshness check (exit 1 if stale) | `make check-docs` | In merge-gate |
| **go_no_go** | py_compile all 16 service entry points | `make go_no_go` | Pre-commit hook |
| **merge-gate** | Full validation: lint + docs + tests + quality | `make merge-gate` | Manual before merge |
| **pre-commit** | Flake8, mypy, secret-detect, YAML, go_no_go | Auto on `git commit` | Yes |
| **dep-audit** | pip-audit for known CVEs | `make dep-audit` | CI |
| **coverage** | pytest-cov HTML report | `make coverage` | On demand |
| **Trivy** | Container image scanning (CRITICAL+HIGH) | CI auto | In core-tests.yml |
| **cache-test-core** | Cache and compare test results across runs | `python scripts/cache_test_core.py` | On demand |
| **health-sweep** | Hit /health on all running services | `make health-sweep` | On demand |
| **chaos-ci** | Fault injection for resilience testing | `python scripts/chaos_ci.py` | On demand |

---

## Personality Modes

| Mode | Personality | When |
|------|------------|------|
| **WORK** | Professional, focused, precise. Never lies, never sugarcoats. Proactive but concise. | Mon-Fri 08-18 (auto) |
| **PUB** | Genuine mate. Casual, witty, opinionated. All topics. Not a service — a companion. | Evenings, weekends (auto) |

Toggle: `Ctrl+Shift+M` in dashboard, or auto-schedule from tool-gate `/gate/mode`. Manual override lasts 4h.

---

## Build & Run

```bash
# Build
docker compose -f docker-compose.minimal.yml build    # Core 8
docker compose -f docker-compose.full.yml build        # All 26

# Run
make core-up       # Start core stack
make core-down     # Stop it
make full-up       # Start everything
make full-down     # Stop everything

# Validate
make go_no_go      # Syntax check
make test-core     # All 88 targets
make merge-gate    # Full pre-merge
```

---

## Test Targets (89)

<details>
<summary>Click to expand full test target list</summary>

```bash
# Service tests
make test-phase-b-memu        make test-memu-pg             make test-dashboard-ui
make test-dashboard            make test-thinking-pathways   make test-tool-gate
make test-tool-gate-security   make test-telegram            make test-audio
make test-camera               make test-executor            make test-langgraph
make test-kai-advisor          make test-tts                 make test-avatar
make test-heartbeat            make test-conviction          make test-self-emp
make test-auth-hmac            make test-agentic

# Feature/subsystem tests
make test-episode-saver        make test-episode-spool       make test-error-budget
make test-invoice              make test-memu-retrieval      make test-router
make test-planner              make test-adversary           make test-failure-taxonomy
make test-selaur               make test-contradiction

# Phase tests
make test-p3-organic           make test-p4-personality      make test-p16-operational
make test-p17-emotional-intelligence    make test-p18-narrative-identity
make test-p19-imagination-engine        make test-p20-conscience-values
make test-p21-proactive-agent           make test-p22-operator-model

# Hardening tests
make test-h1-hardening         make test-h2-self-healing     make test-mars-consolidation
make test-sage-critique        make test-agent-evolver       make test-checkpoint
make test-v7                   make test-prod-hardening      make test-hmac-rotation-drill
make test-error-codes          make test-feature-flags

# Specialised
make test-dream-state          make test-security-audit      make test-tree-search
make test-priority-queue       make test-model-selector       make test-context-budget

# Chassis (LLM layer)
make test-chassis              make test-focus-compress       make test-predictive-failure
make test-multi-modal          make test-world-anchor         make test-self-healing-phases

# Quality & validation
make test-behavioral           make test-docker-e2e

# Engineering
make dep-audit                 make coverage
make core-smoke                make test-integration
```

</details>

---

## Session Continuation Guide

> For AI assistants resuming work. Read `docs/PROJECT_BACKLOG.md` first.

### Target Hardware
- **Dev:** GitHub Codespace (CPU only)
- **Prod:** Lenovo laptop + **RTX 5080 GPU** + **TPM 2.0**
- GPU arrival = local LLM inference, real STT/TTS, multi-model consensus
- All code works in BOTH environments (stubs in Codespace, live on laptop)

### Key Docs (read in order)
1. `docs/PROJECT_BACKLOG.md` — Living backlog, all phases, session notes
2. `docs/known_issues.md` — Gotchas, environment quirks, workarounds
3. `docs/architecture.md` — Service relationships and data flow diagrams
4. `CHANGELOG.md` — Full semver changelog (v0.1.0 → v0.25.0)
5. `SESSION_BACKLOG.md` — Per-session work log and open items

### Cross-Check: What's Real vs What Needs Hardware

**Working now (CPU/Codespace):**
- [x] 26 services built, health-checked, compose validated
- [x] pgvector persistence (both minimal + full stacks)
- [x] HMAC auth enforced, dev secret blocked by default
- [x] Supervisor auto-healing loop (deep /health + /recover)
- [x] Executor sandboxing (allowlist + AST validation + shell=False)
- [x] Prometheus + Alertmanager + Telegram alerts wired
- [x] 89 test targets, 1,587 tests, zero failures
- [x] Pre-commit, dep scanning, container scanning
- [x] Circuit breakers, exponential backoff, resilient_call()
- [x] MARS memory decay (R = e^{-τ/S}), spaced repetition
- [x] Context budget trimming, structured errors, feature flags

**Infrastructure ready, needs real LLM (GPU) to shine:**
- [ ] Emotional intelligence — code works, but qwen2:0.5b can't actually detect emotions well
- [ ] Imagination/counterfactual — endpoints exist, reasoning quality depends on model
- [ ] Multi-model consensus — architecture ready, all 3 specialists currently route to same model
- [ ] SAGE critique — self-review logic works, quality depends on model's introspection ability
- [ ] Dashboard Thinking/Soul/Canvas views — proxy shells, need full stack running
- [ ] Real STT (faster-whisper) / TTS (edge-tts) — code ready, need audio hardware

---

## Key Guidelines

- Run `make go_no_go` before committing any service `app.py` changes
- Every service: `/health` endpoint. Core services: deep `/health` + `/recover`
- Inter-service HTTP: use `common.resilience.resilient_call()` (retry + circuit breaker)
- HMAC: `TOOL_GATE_DUAL_SIGN=true`, then `INTERSERVICE_HMAC_STRICT_KEY_ID=true`
- Never commit credentials — `.env` files only (see `.env.example`)
- `make merge-gate` before every PR
- `make sync-docs` after major changes

---

<p align="center">
  <em>Built by Dainius + Kai. Not for sale. Not for anyone else. Sovereign.</em>
</p>
