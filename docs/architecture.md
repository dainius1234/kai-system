# Kai System — Architecture

> How the 26 services connect, what data flows where, and what depends on what.

## Service Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  L0: OPERATOR INPUT                                              │
│  Telegram Bot ─── Dashboard (8 views) ─── API Direct            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  L1: INTELLIGENCE (LangGraph Brain)                              │
│                                                                  │
│  Request → Injection Filter → Specialist Router                  │
│    → 10-way Parallel Context Fetch:                              │
│      ┌─ memories ─┬─ session ─┬─ goals ──┬─ topics ─┐           │
│      ├─ EQ ───────┼─ narrative┼─ imagination─┤       │           │
│      └─ conscience┴─ agent ───┴─ operator_model ─────┘           │
│    → Planner (gap-aware, preference-constrained)                 │
│    → Adversary (6 challenges + SAGE self-review)                 │
│    → Conviction Scoring (5-signal + modifiers)                   │
│    → Tree Search (CoT pruning, priority queue)                   │
│    → Agent-Evolver (dream-cycle failure insights)                │
│    → Checkpoint Engine (state snapshot/rollback)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  L2: SAFETY & POLICY                                             │
│  Orchestrator ──► Tool-Gate (HMAC, rate limit, co-sign)          │
│  Verifier (fact-check, SAGE critique) ◄── Fusion Engine          │
│  Supervisor (watchdog, circuit breaker, auto-heal)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  L3: MEMORY & STATE                                              │
│  Memu-Core ──► PostgreSQL/pgvector                               │
│    episodic memory, emotional memory, goals, EQ, narrative,      │
│    imagination, conscience, values, operator model, agent loop    │
│  Redis ──► session buffer, caches                                │
│  Ledger-Worker ──► action audit trail                            │
│  Memory-Compressor ──► summarisation                             │
│  Backup-Service ──► pg/redis/memory backup+restore               │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  L4: EXECUTION                                                   │
│  Executor ──► Sandboxes (shell)                                  │
│    sandboxed command execution, shell=False, size limits          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  L5: PERCEPTION & OUTPUT                                         │
│  Audio (faster-whisper) ── Camera ── Screen-Capture              │
│  TTS (edge-tts British Ryan) ── Avatar                           │
│  Telegram Bot (voice+text pipeline)                              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  L6: INFRASTRUCTURE                                              │
│  Heartbeat (system pulse) ── Calendar-Sync                       │
│  Metrics-Gateway (Prometheus) ── Workspace-Manager               │
│  Ollama (qwen2:0.5b CPU, GPU arrival = 3 env vars)              │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow: Chat Message

```
User types in Dashboard Chat or sends Telegram message
  │
  ▼
LangGraph /chat endpoint
  │
  ├─ Injection filter (block prompt injection)
  ├─ Session buffer (Redis)
  │
  ▼
10-way parallel context fetch (all from memu-core):
  memories, session, goals, topics, EQ, narrative,
  imagination, conscience, agent, operator_model
  │
  ▼
Specialist Router → classify domain (8 categories)
  │
  ▼
Planner → build gap-aware plan with preference constraints
  │
  ▼
Adversary → 6 challenges + SAGE self-review in parallel
  │
  ▼
Conviction Scoring → 5-signal score + modifiers
  │ if < 8.0 → rethink loop (max 3 retries)
  │
  ▼
LLM call (Ollama) → generate response
  │
  ▼
Post-mortem:
  ├─ Episode saved (ledger-worker)
  ├─ Corrections learned (memu-core)
  ├─ Emotion detected & recorded (memu-core)
  ├─ Preferences extracted (memu-core)
  └─ Auto-memorize if important (memu-core)
  │
  ▼
Response streamed back to user (SSE)
```

## Data Flow: Self-Healing

```
Supervisor (every 15s sweep)
  │
  ▼
Call /health on each core service (deep health)
  │
  ├─ {"status": "ok"} → all good, record in fleet history
  │
  ├─ {"status": "degraded", "checks": {"redis": "fail"}}
  │     │
  │     ▼
  │   Supervisor circuit breaker opens
  │     │
  │     ▼
  │   POST /recover on the degraded service
  │     │
  │     ▼
  │   Service reconnects to dependency
  │     │
  │     ▼
  │   memu-core logs recovery event: what was healed and what was learned to conscience/narrative system
  │     │
  │     ▼
  │   Next sweep: /health returns "ok" → circuit closes
  │
  └─ Unreachable → alert via Telegram, log failure
```

## Data Flow: Dream Cycle

```
POST /dream (langgraph)
  │
  ▼
Checkpoint current state (auto-checkpoint)
  │
  ▼
Phase 1: Failure clustering (Agent-Evolver)
Phase 2: Boundary recalibration
Phase 3: MARS consolidation (R = e^{-τ/S})
Phase 4: Conscience-filtered pruning
Phase 5: Insight generation
Phase 6: State summary
  │
  ▼
Checkpoint post-dream state
```

## Service Dependency Map

| Service | Depends On | Called By |
|---|---|---|
| **langgraph** | memu-core, ollama, verifier, tool-gate | dashboard, telegram-bot |
| **memu-core** | postgres, redis | langgraph, dashboard, supervisor |
| **tool-gate** | redis | langgraph, orchestrator |
| **verifier** | memu-core | langgraph, fusion-engine |
| **supervisor** | all services (health checks) | heartbeat, dashboard |
| **dashboard** | all services (proxy) | operator (browser) |
| **executor** | tool-gate | langgraph, orchestrator |
| **fusion-engine** | verifier, memu-core | langgraph |
| **orchestrator** | tool-gate, executor | langgraph |
| **heartbeat** | supervisor | — |
| **telegram-bot** | langgraph, memu-core | Telegram API |
| **backup-service** | postgres, redis, memu-core | operator (manual) |
| **ledger-worker** | postgres | langgraph, tool-gate |
| **metrics-gateway** | all services (scrape) | Prometheus |
| **memory-compressor** | memu-core | supervisor (scheduled) |

## Docker Compose Stacks

| File | Services | Purpose |
|---|---|---|
| `docker-compose.minimal.yml` | 8 (postgres, redis, tool-gate, memu-core, heartbeat, dashboard, supervisor, verifier) | Dev core loop |
| `docker-compose.full.yml` | All 26 | Full deployment |
| `docker-compose.sovereign.yml` | Selected | Sovereign-only config |

## Port Map

| Service | Port | Notes |
|---|---|---|
| dashboard | 8050 | Operator console |
| tool-gate | 8010 | Policy enforcement |
| memu-core | 8020 | Memory engine |
| langgraph | 8030 | Brain / LLM |
| executor | 8040 | Sandboxed execution |
| verifier | 8060 | Fact checking |
| fusion-engine | 8070 | Signal consensus |
| orchestrator | 8080 | Risk authority |
| heartbeat | 8090 | System pulse |
| supervisor | 8100 | Watchdog |
| telegram-bot | 8110 | Telegram interface |
| kai-advisor | 8120 | Self-emp advisor |
| metrics-gateway | 9090 | Prometheus metrics |
| postgres | 5432 | pgvector DB |
| redis | 6379 | Session/cache |
| ollama | 11434 | LLM inference |
