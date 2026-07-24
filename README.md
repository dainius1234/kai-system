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
  <img src="https://img.shields.io/badge/services-60-blue?style=flat-square" alt="services">
  <img src="https://img.shields.io/badge/tests-2%2C671_passing-brightgreen?style=flat-square" alt="tests">
  <img src="https://img.shields.io/badge/GPU_Phase0-DONE-success?style=flat-square" alt="gpu-phase0">
  <img src="https://img.shields.io/badge/Python-~65%2C940_LOC-yellow?style=flat-square" alt="loc">
  <img src="https://img.shields.io/badge/milestones-39_shipped-purple?style=flat-square" alt="milestones">
  <img src="https://img.shields.io/badge/failures-0-brightgreen?style=flat-square" alt="failures">
  <img src="https://img.shields.io/badge/license-private-red?style=flat-square" alt="license">
</p>

---

## Project Status (24 July 2026)

| Metric | Value |
|---|---|
| **Services** | 60 unique across 3 compose profiles |
| **Test targets** | 82 (`make test-core`) |
| **Individual tests** | 2,671 (`def test_` across 128+ files) |
| **Python LOC** | ~65,940 |
| **Feature flags** | 35 (`FF_*` env vars) |
| **Dockerfiles** | 47 |
| **Compose profiles** | 3 (minimal / full / sovereign) |
| **Milestones shipped** | 39 |
| **Failures** | 0 |

> **Auto-synced** by `make sync-docs`. Stale metrics block `make merge-gate`.

---

## Quick Reference

```bash
make core-up          # Start minimal stack (33 services)
make core-down        # Stop it
make full-up          # Start full stack (35 services)
make test-core        # Run all 82 test targets (2,671 tests)
make go_no_go         # Syntax-check all service entry points
make merge-gate       # Full pre-merge validation
make sync-docs        # Auto-update README + backlog metrics
make dep-audit        # CVE scan on all pip packages
make coverage         # pytest-cov HTML report
make health-sweep     # Hit /health on all running services
```

GPU integration status: **Phase 0 complete** — see [`docs/gpu_integration_phase0.md`](docs/gpu_integration_phase0.md).

## Project Management

All PM operations live in [`kai-pm/`](kai-pm). Entry point for fast session re-hydration:
[`kai-pm/SESSION_BOOTSTRAP.md`](kai-pm/SESSION_BOOTSTRAP.md). Decision log (append-only, D1–D89):
[`kai-pm/DECISIONS.md`](kai-pm/DECISIONS.md). Known stubs and placeholders:
[`kai-pm/STUBS_AND_PLACEHOLDERS.md`](kai-pm/STUBS_AND_PLACEHOLDERS.md).

---

## What Makes Kai Different

> Every capability below has code and tests. Reasoning quality depends on the LLM model — see [Honest Limitations](#honest-limitations).

### Soul & Inner Life

| Capability | What It Does |
|---|---|
| **Emotional Memory** | Detects 8 emotions in conversation, tracks mood arcs over time, surfaces emotional continuity |
| **Self-Reflection** | Analyses its own mistakes, builds a strengths/weaknesses journal, knows where it fails |
| **Epistemic Humility** | Knows what it doesn't know — warns operator when confidence is low |
| **Confession Engine** | Proactively admits past mistakes without being asked |
| **Narrative Identity** | Builds its own life story — autobiography, story arcs, future-self projection, legacy time-capsules |
| **Imagination Engine** | Counterfactual replay, theory of mind, creative synthesis, inner monologue, aspirational futures |
| **Conscience & Values** | Emergent value formation, moral reasoning, integrity tracking, loyalty memory, gratitude engine |
| **Dream State** | 6-phase offline consolidation — failure clustering, boundary recalibration, MARS memory decay |

### Intelligence & Reasoning

| Capability | What It Does |
|---|---|
| **14-Way Context Gather** | Every response enriched with: memories + session + goals + topics + EQ + narrative + imagination + conscience + agent + operator model + world state + financial context + Letta archival + graph memory |
| **Specialist Router** | Classifies queries into 8 UK construction domains for category-aware retrieval |
| **Memory-Driven Planner** | Gap-aware plans with preference constraints and history-informed conviction modifiers |
| **Adversary Engine** | 7 challenge types (incl. SAGE self-review) test every plan before execution |
| **Conviction Scoring** | 5-signal + modifiers gate; below 8.0 triggers rethink (max 3 retries) |
| **SAGE Critique** | Verifier self-critique + adversary self-review — AI arguing with itself for quality |
| **Agent-Evolver** | Learns from failure clusters during dream cycles, generates proactive fix insights |
| **Tree Search** | Chain-of-thought pruning with priority queue + counterargument debate gating |

### Cognitive Depth (D89)

| Capability | What It Does |
|---|---|
| **System FSM** | Kai tracks its own operational state: IDLE → ACTIVE → FOCUSED / DEGRADED → RECOVERING. Thread-safe asyncio singleton. 5 states, 9 events, 16 transitions. Curiosity and autonomy only fire from IDLE. |
| **Cognitive Reasoning FSM** | Deterministic reasoning pipeline: GATHER → DEBATE → FACT_CHECK → CAUSAL_CHECK → CONVICTION_GATE → PRESENT. Bounded retry loops: RETHINK (max 3) → ESCALATE_LOOP (max 3) → HALT. Per-swarm configs (trading / research / skill_forge / default). Schema-validated `AgentHandoff` handoffs — never parses free text. |
| **Persistent Teammates** | 4 named cognitive personas invokable via `POST /chat/teammate/{name}`: **Scout** (skill discovery + package evaluation), **Doctor** (system health differential diagnosis), **Sage** (reflection + counterargument, steward of quality thinking), **Oracle** (probabilistic trend extrapolation from world model). Each has a specialty, system prompt, and world-state injection. |
| **House Doctor** | Continuous differential diagnosis service (port 8046). 9 rules (D001–D009) map symptom-tag constellations (cpu_high, ram_high, docker_unhealthy, aq_degraded, sensor_anomaly, calendar_soon) to severity/diagnosis/treatment. Fires `medical_report` memories to memu-core; notifies on WARNING/CRITICAL. |
| **World Model Provenance** | Every field in world_state now carries `{value, source, timestamp, confidence}`. Origin service, read time, and reliability score travel with every sensor reading. Enables temporal queries and trust-weighted reasoning. |
| **Emergent Ritual Discovery** | When a sensor pattern recurs in ≥7 of the last 10 cycles, Kai proposes a standing ritual and appends it to `RITUALS.md`. One-time notification to operator for co-authorship: "I've noticed X — shall we make this a ritual?" |
| **Capability Gap Logging** | Each missed skill match increments a counter. Reactive skill acquisition fires only after `GAP_HUNT_THRESHOLD` (default 3) consecutive misses for the same intent — prevents wasted hunts on one-off requests. |
| **Skill Provenance** | Every auto-hunted skill file carries YAML front-matter (`hunted_at`, `pypi_package`, `pypi_verified`, `probationary: true`). Sidecar `.meta.json` tracks runtime error count. Auto-disables at ≥3 errors. |
| **GPU-era Foundations** | Four slots wired and architected now, activating on GPU: **Counterfactual Rehearsal** (decision branch simulation, `can_rehearse()→False` until Phase 1), **Trust Negotiation** (`/gate/autonomy/request`, currently `pending_approval`), **Predictive Empathy** (`emotional_context` in world model, activates when emotional memory accumulates), **Resource-Aware Curiosity** (idle tick no-ops on CPU, `CURIOSITY.md` seed questions ready). |

### Sensory & World Awareness (D87 / D88)

| Capability | What It Does |
|---|---|
| **World Context Injection** | Every `/chat` call triggers a parallel 9-service probe with 2s per-service timeout. Result injected as "World State" system block into LLM prompt. Kai knows the environment it's operating in. |
| **Proactive Observer** | Background asyncio task runs every 5 minutes. Reads Docker health, email, AQ, git, calendar, sysmetrics, weather, news. Detects notable changes. Writes `proactive_observation` memories to memu-core for spontaneous awareness. |
| **Anomaly Detection** | Rolling 48-reading (≈4 hour) z-score baselines per sensor. Alerts when |z| > 2.0 after ≥6 readings warm-up. Moves Kai from snapshot to trend awareness. |
| **Cross-Sensor Correlation** | After each observation cycle, reasons across the full sensor set: cpu_high + docker_unhealthy → resource cascade; RAM + docker → memory leak; git dirty + email backlog → operator mid-task (tread lightly). |
| **Sensory Pattern Learning** | Tracks 10 recent observation cycles. Recurring types (≥3/10) written as `sensor_pattern` memories — future context retrieval surfaces predictable recurrences before they escalate. |
| **Proactive Scheduling** | Fuses calendar events (within 30 min) with current sensor state into `proactive_schedule` memories: "Meeting in 20 min + AQ poor → consider indoor location." Surfaces naturally via memory retrieval. |
| **Autonomous Skill Growth** | When `/chat` has no skill match and confidence < 0.4, fires `_hunt_skill_for_gap()` in background. Skill Hunter queries PyPI, generates a `.md` skill file, hot-reloads — Kai grows its own capabilities during conversation. |
| **Self-Capability Map** | `GET /introspect/capabilities` exposes: live service reachability, loaded skills, feature flag states, anomaly baseline keys, active FSM state, teammate registry, gap log top-5, counterfactual availability. |

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
| **Wake & Intent** | Wake-word detection (`/wake/detect`) + tiny-model intent judge (`/wake/intent`) with safe heuristic fallback |
| **Trust Negotiation** | Kai can formally request elevated autonomy via `POST /gate/autonomy/request`. All requests currently require human approval and are ledger-logged. |

### Production & Security

| Capability | What It Does |
|---|---|
| **Self-Healing** | Deep `/health` + `/recover` + supervisor auto-heal loop across all services every 15s |
| **Recovery Log** | Every self-heal event logged to conscience — ties resilience to narrative |
| **House Doctor** | Continuous differential diagnosis (9 rules) — CPU+RAM+docker constellations mapped to diagnosis + treatment |
| **Security Self-Hacking** | Fuzzes own APIs with 34 payloads, adversary challenges, SAGE self-review |
| **HMAC Auth** | Inter-service HMAC signing, Ed25519, dual-sign rotation, nonce replay protection. Dev secret requires explicit `HMAC_ALLOW_DEV_SECRET=true` |
| **Time-Travel Debug** | Checkpoint any state, diff between snapshots, rollback to any previous state |
| **35 Feature Flags** | All capabilities independently toggleable via `FF_*` env vars — see [Feature Flags](#feature-flags) |
| **Structured Errors** | 20 enumerated codes (E1001–E4004) — no more "something broke" |
| **Zero Telemetry** | No corporate control, no data exfiltration, no resets. Ever. |
| **Skills Hub** | Hot-loadable .md skill files with security scanning, TTL pruning, unload, and provenance front-matter |
| **Multi-modal Fusion** | Combined audio + video signals interpreted by LLM for proactive voice |
| **World Anchor** | Daily date/time/context fetch cached locally — Kai knows what day it is |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  OPERATOR INPUT                                                      │
│  Telegram Bot ─── Dashboard (10 views) ─── API Direct               │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  COGNITIVE REASONING PIPELINE (D89)                                  │
│                                                                      │
│  GATHER → DEBATE → FACT_CHECK → CAUSAL_CHECK → CONVICTION_GATE      │
│  → PRESENT                                                           │
│                                                                      │
│  RETHINK (max 3) → ESCALATE_LOOP (max 3) → HALT → ask operator      │
│  Per-swarm config: trading (≥8.0) / research (≥6.5) /               │
│  skill_forge (≥6.0) / default                                        │
│  Schema-validated AgentHandoff — never parses free text              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  INTELLIGENCE LAYER (D87 / D88 / D89)                                │
│                                                                      │
│  ┌─ 14-way parallel context gather ────────────────────────────┐    │
│  │ memories│session│goals│topics│EQ│narrative│imagination│      │    │
│  │ conscience│agent│operator_model│world_state│financial│Letta  │    │
│  └──────────────────────────────────────────────────────────────┘    │
│  World Context: 9 sensory services probed per /chat (2s timeout)    │
│  Anomaly detection: 2σ rolling baselines, 48-reading windows         │
│  Cross-sensor correlation → world_state (provenance per field)       │
│  Pattern learning → ritual discovery → proactive_schedule memories  │
│  Skill matching → gap logging → autonomous skill acquisition         │
│  Teammate routing: Scout / Doctor / Sage / Oracle                   │
│  System FSM: IDLE / ACTIVE / FOCUSED / DEGRADED / RECOVERING        │
│                                                                      │
│  → Specialist Router → Planner → Adversary (7 challenges)            │
│  → Conviction Scoring (≥8.0) → Tree Search → Agent-Evolver          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  SAFETY & POLICY                                                     │
│  Tool-Gate (HMAC + rate limit + autonomy request gate)               │
│  Verifier (SAGE semantic fact-checking)                              │
│  House Doctor (differential diagnosis, 9 rules D001–D009)           │
│  Supervisor (watchdog + circuit breaker + auto-heal every 15s)      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  MEMORY & STATE                                                      │
│  Memu-Core → TurboVec ANN index + PostgreSQL (hot path)             │
│  Memu-Core-Introspect → compress / decay / quarantine (cold path)   │
│  Memu-Graph → Cognee/Kuzu knowledge graph (entities + relations)    │
│  Letta-Agent → archival memory controller (gated FF_LETTA_TASKS)    │
│  Redis (session buffer) → Ledger-Worker (audit trail)                │
│  Memory-Compressor → Backup-Service (pg/redis/memory)                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│  SENSORY LAYER (33 probed services)                                  │
│  sysmetrics│calendar│email│news│weather│airquality│docker-watcher   │
│  git-watcher│broker-bridge│screen-watcher│vision│audio│clipboard    │
│  files│document-parser│notify│browser-agent│monitor                 │
│  Proactive observer: every 5 min → anomaly detect → correlate       │
│  → world_state persist → pattern check → ritual propose             │
│  Skill Hunter: autonomous PyPI discovery + hot-reload (port 8045)   │
│  House Doctor: differential diagnosis (port 8046)                   │
└─────────────────────────────────────────────────────────────────────┘
```

### How a Message Flows

```
1. Operator sends message (Telegram / Dashboard / API)
2. System FSM: fire USER_MESSAGE (IDLE → ACTIVE)
3. Agentic: 14-way parallel context gather + world context injection
4. Skill matching → if no match + conf < 0.4: async gap log → hunt after 3 misses
5. Cognitive pipeline: GATHER → DEBATE → FACT_CHECK → CAUSAL_CHECK
6. CONVICTION_GATE: ≥ threshold → PRESENT; else RETHINK (max 3) → ESCALATE → HALT
7. Tool-Gate: HMAC + rate limit + policy → Executor: sandboxed run
8. Post-mortem: episode saved, corrections learned, emotion recorded, memory updated
9. Response streamed back (SSE)
```

### Self-Healing Flow

```
Supervisor (every 15s) → deep /health on each service
  ├─ "ok"       → record in fleet history, circuit stays closed
  ├─ "degraded" → open circuit → POST /recover → service self-heals
  │               → log recovery to conscience (what healed, what learned)
  │               → House Doctor: re-diagnose after heal
  │               → System FSM: fire SERVICE_RESTORED → RECOVERING
  │               → next sweep: "ok" → circuit closes → FSM: HEAL_COMPLETE
  └─ unreachable → Telegram alert → operator intervenes
                 → System FSM: fire SERVICE_DOWN → DEGRADED
```

### Proactive Observer Cycle (every 5 min)

```
_proactive_observer()
  1. Probe all sensory services in parallel
  2. Anomaly detection: z-score against 48-reading baselines
  3. Cross-sensor correlation: _correlate_observations()
  4. House Doctor: POST /diagnose with observations + world_state
  5. World model persistence: write {value,source,timestamp,confidence} to memu-core
  6. Pattern learning: check 10-cycle history → sensor_pattern memories
  7. Ritual discovery: ≥7/10 cycles → _propose_ritual() → RITUALS.md
  8. Proactive scheduling: calendar ≤30 min → fuse with sensor state → memory
  9. Curiosity tick: if IDLE + GPU available → idle_curiosity_tick()
```

---

## Service Map

### Minimal Stack (`docker-compose.minimal.yml`) — 33 services

The default daily driver. All sensory, perception, memory, and cognitive services. No heavy AI extras (graph memory, Letta, telegram, avatar).

| # | Service | Port | IP | Purpose |
|---|---------|------|-----|---------|
| 1 | postgres | internal | .2 | pgvector DB — memories, ledger, embeddings |
| 2 | redis | internal | .3 | Session buffer, caches |
| 3 | ollama | 11434 | .4 | Local LLM (`qwen2.5:0.5b` CPU default) |
| 4 | ollama-pull* | — | — | One-shot init container — pulls model on first boot |
| 5 | tool-gate | 8000 | .5 | Policy enforcement, HMAC auth, autonomy gate |
| 6 | memu-core | 8001 | .6 | Memory engine hot path: memorize / retrieve / rank |
| 7 | memu-core-introspect | 8009 | .7 | Store maintenance cold path: compress / decay / quarantine |
| 8 | agentic | 8007 | .8 | Reasoning brain: chat / conviction / teammates / skills |
| 9 | heartbeat | 8010 | .9 | System pulse, world anchor, auto-sleep |
| 10 | dashboard | 8080 | .11 | 10-view operator console |
| 11 | audio-service | 8021 | .15 | STT (faster-whisper) |
| 12 | tts-service | 8030 | .16 | Text-to-speech (edge-tts `en-GB-RyanNeural`) |
| 13 | browser-agent | 8040 | .17 | Playwright Chromium — navigate / scrape / click |
| 14 | vision-service | 8023 | .18 | OpenCV face detect + DeepFace emotion |
| 15 | clipboard-service | 8024 | .19 | Clipboard read/write |
| 16 | files-service | 8025 | .20 | File system read (allowlisted paths) |
| 17 | notify-service | 8031 | .21 | Push notifications |
| 18 | document-parser | 8032 | .22 | PDF / DOCX / XLSX / PPTX / DXF / ZIP / CSV |
| 19 | monitor-service | 8033 | .23 | Background rule engine — HTTP/scrape + notify + TTS |
| 20 | broker-bridge | 8034 | .24 | Binance REST (spot/futures), HMAC-signed |
| 21 | sysmetrics | 8035 | .25 | CPU / RAM / disk / network / processes via psutil |
| 22 | screen-watcher | 8036 | .26 | Periodic screenshot diff + change alert |
| 23 | email-reader | 8037 | .27 | IMAP read-only polling |
| 24 | news-feed | 8038 | .28 | RSS aggregation + keyword search |
| 25 | weather-service | 8039 | .29 | Weather data (mocked httpx in CI) |
| 26 | docker-watcher | 8041 | .30 | Docker container health monitoring |
| 27 | airquality-service | 8042 | .31 | Air quality index |
| 28 | calendar-service | 8043 | .32 | CalDAV calendar — events, summaries |
| 29 | git-watcher | 8044 | .33 | Git repo dirty/stash/branch status |
| 30 | skill-hunter | 8045 | .34 | Autonomous skill discovery via PyPI |
| 31 | house-doctor | 8046 | .35 | Differential system diagnosis (9 rules D001–D009) |
| 32 | wake-service | 8022 | .10 | Wake-word + intent routing |
| 33 | supervisor | 8051 | .12 | Watchdog, auto-heal, proactive checks |
| 34 | verifier | 8052 | .13 | Semantic fact-checking (embedding + keyword), SAGE |

*ollama-pull is a one-shot init container — not counted as a long-running service.

### Full Stack Additions (`docker-compose.full.yml`)

Full stack includes all minimal services plus the heavy AI stack: introspect, graph memory, Letta, financial, executor, fusion, telegram, avatar, camera, and ops tooling.

| Service | Port | Purpose |
|---------|------|---------|
| agentic-introspect | 8023 | Dream / evolve / security-audit — cold path split off agentic |
| executor | 8002 | Sandboxed code execution (AST validation + allowlist) |
| fusion-engine | 8053 | Multi-signal consensus (embedding cosine + Jaccard fallback) |
| memu-graph | 8061 | Cognee/Kuzu knowledge graph — entity/relation ingest→query→forget |
| letta-agent | 8062 | Letta archival memory controller (`FF_LETTA_TASKS=false` default) |
| financial-awareness | 8063 | CIS/VAT/tax arithmetic (offline, UK-focused) |
| ledger-worker | 8056 | Action audit trail persistence |
| metrics-gateway | 8058 | Prometheus metrics aggregation |
| memory-compressor | 8057 | Memory summarisation |
| camera-service | 8020 | Camera capture |
| audio-service | 8021 | STT (faster-whisper) |
| tts-service | 8030 | TTS (edge-tts British Ryan) |
| avatar-service | 8081 | Avatar generation |
| screen-capture | 8059 | Screen OCR pipeline |
| backup-service | 8054 | pg / redis / memory backup |
| calendar-sync | 8055 | Calendar synchronisation |
| kai-advisor | 8090 | Self-employment advisor (UK, offline) |
| telegram-bot | 8025 | Telegram voice + text interface |
| workspace-manager | 8060 | Workspace lifecycle manager |
| parakeet-server | 8080 | Optional CPU ASR sidecar — opt-in (`--profile parakeet`) |
| skill-hunter | 8045 | Autonomous skill acquisition |
| house-doctor | 8046 | System differential diagnosis |

### Sovereign Stack (`docker-compose.sovereign.yml`)

Production-hardened subset with full security stack. Uses pgvector (not TurboVec), Vault secrets, Tailscale overlay, and Prometheus/Grafana observability.

| Service | Purpose |
|---------|---------|
| postgres | pgvector production DB |
| tailscale | Overlay network (private mesh) |
| vault + vault-rotator | Secret management + key rotation |
| prometheus + alertmanager + grafana | Observability stack |
| tool-gate | Policy + HMAC (production keys via Vault) |
| memu-core + memu-core-introspect | Memory hot + cold paths |
| executor | Sandboxed execution |
| dashboard | Operator console |
| agentic + agentic-introspect | Reasoning brain + cold-path split |
| redis | Session buffer |
| perception-telegram | Telegram perception |
| heartbeat | System pulse |
| camera-service + audio-service | Sensory perception |
| skill-hunter + house-doctor | Skill growth + system diagnosis |

---

## Cognitive Reasoning Pipeline (D89)

The pipeline is a deterministic state machine — not an ad-hoc chain of LLM calls. Every stage produces a schema-validated `AgentHandoff`. Stuck stages HALT and escalate to the operator — never silently degrade.

```
AgentHandoff schema
  from_stage:    str
  to_stage:      str
  status:        COMPLETE | PARTIAL | FAILED | NEEDS_INPUT |
                 CONSENSUS | NO_CONSENSUS | PASS | FAIL
  confidence:    float (0.0–10.0)
  payload:       dict (stage output)
  claims:        List[str]
  loop_count:    int
  elapsed_ms:    float
  halt_reason:   Optional[str]

Pipeline stages
  GATHER          → collects evidence, world state, memory context
  DEBATE          → devil's advocate challenges claims
  FACT_CHECK      → verifies claims against memory + world state
  CAUSAL_CHECK    → traces consequence chains
  CONVICTION_GATE → scores conviction against swarm threshold
  PRESENT         → formats response

Retry loops (bounded)
  RETHINK         → re-run from DEBATE (max 3 per swarm config)
  ESCALATE_LOOP   → re-run from GATHER (max 3 per swarm config)
  HALT            → log halt_reason → notify operator

Per-swarm configs
  trading         conviction_threshold=8.0, debate_retries=2, rethink_retries=2
  research        conviction_threshold=6.5, debate_retries=5, rethink_retries=3
  skill_forge     conviction_threshold=6.0, debate_retries=3, rethink_retries=3
  default         conviction_threshold=7.0, debate_retries=3, rethink_retries=3

Stage function assignments (D90)
  GATHER          → Scout    (parallel memory + world fetch; LLM extracts claims)
  DEBATE          → Sage     (build_plan + score_conviction; LLM counterargument)
  FACT_CHECK      → Doctor   (claim→verdict JSON dict; falls back to uncertain)
  CAUSAL_CHECK    → Oracle   (consequence chain JSON array per supported claim)
  CONVICTION_GATE → Sage     (adversary challenge + resolve_conflict())

Conflict resolution signal weights (D90)
  evidence weight         0.30  (evidence_count × 1.5, cap 10)
  causal chain quality    0.25  (chain_count × 2.0, cap 10)
  verdict fraction        0.20  (supported / total × 10; neutral 5.0 when empty)
  reputation-weighted vote 0.15 (weight = reliability × avg_confidence/10)
  adversary modifier      0.10  (0..10 centred at 5; modifier range −3..+1)

API (D90)
  POST /agentic:8007/chat/swarm    → run full pipeline; returns conviction_score, transition_log
  GET  /agentic:8007/swarm/reputation → per-teammate weights from data/teammate_reputation.json
```

---

## System FSM (D89)

Tracks Kai's operational state. Wired into proactive observer, `/chat`, and downstream autonomy/curiosity gating.

```
States:     IDLE → ACTIVE → FOCUSED
                         ↕
                      DEGRADED ↔ RECOVERING

Events:     USER_MESSAGE, SESSION_END, SERVICE_DOWN, SERVICE_RESTORED,
            ANOMALY_CRITICAL, HEAL_STARTED, HEAL_COMPLETE,
            FOCUS_ENTER, FOCUS_EXIT

Transitions (16 total):
  IDLE + USER_MESSAGE       → ACTIVE
  ACTIVE + SESSION_END      → IDLE
  * + SERVICE_DOWN          → DEGRADED
  DEGRADED + HEAL_STARTED   → RECOVERING
  RECOVERING + HEAL_COMPLETE→ IDLE
  ACTIVE + FOCUS_ENTER      → FOCUSED
  FOCUSED + FOCUS_EXIT      → ACTIVE
  ... (see agentic/system_fsm.py for full table)

API:  fire(event) → Optional[KaiState]
      current_state() → KaiState
      fsm_snapshot() → {state, history[-10], event_count}
```

---

## Persistent Teammates (D89)

Named cognitive personas loaded from `data/teammates/*.md`. Each defines specialty, description, and system prompt. Invoked via API with world-state injection.

```
Endpoint:   POST /agentic:8007/chat/teammate/{name}
            GET  /agentic:8007/teammates

Teammates:
  scout     Skill discovery — evaluates packages for maintenance, popularity, safety.
            Returns: recommended_package, risk_note, alternatives.

  doctor    System health differential diagnosis — mirrors House Doctor but conversational.
            Returns: diagnosis, evidence, treatment, differential, severity.

  sage      Reflection + counterargument. Steward of quality thinking. Max 200 words.
            Returns: key_assumption, challenge, affirmation, open_question.

  oracle    Probabilistic trend extrapolation from world model + sensory history.
            Returns: prediction, confidence (HIGH/MEDIUM/LOW), key_assumption,
                     early_indicator, time_horizon.

Data format (data/teammates/{slug}.md):
  # Name
  **Specialty:** <domain>
  **Description:** <one-liner>
  ## System Prompt
  <full persona instructions>
```

---

## House Doctor Service (D89)

Differential diagnosis of Kai's system health. Runs as a standalone service (port 8046) called by the proactive observer after each correlation pass.

```
Endpoint:   POST /house-doctor:8046/diagnose
            GET  /house-doctor:8046/rules
            GET  /house-doctor:8046/health

Request:    {observations: List[str], world_state: dict}
Response:   {diagnoses: [{rule_id, severity, diagnosis, treatment, differential}],
             top_severity: INFO|WARNING|CRITICAL,
             observation_tags: List[str]}

Diagnostic rules (D001–D009):
  D001  cpu_high + docker_unhealthy   WARNING   Resource-pressure cascade
  D002  ram_high + docker_unhealthy   WARNING   Memory pressure → container OOM
  D003  cpu_high + ram_high           WARNING   Runaway process suspected
  D004  sensor_anomaly + docker       WARNING   Anomaly + instability
  D005  aq_degraded + calendar_soon   INFO      Environment alert before meeting
  D006  docker_unhealthy alone        INFO      Isolated container issue
  D007  cpu_high alone                INFO      Elevated CPU load
  D008  ram_high alone                INFO      Elevated memory usage
  D009  sensor_anomaly + cpu + ram    CRITICAL  Multi-system failure cascade

Side effects: writes medical_report memory to memu-core;
              calls notify-service for WARNING/CRITICAL.
```

---

## Key Service APIs

### Agentic (`agentic:8007`)

```
POST /chat                    Streaming LLM conversation with full 14-way context
POST /run                     Graph pipeline execution (GraphResponse)
GET  /health                  Deep health check
POST /recover                 Self-heal agentic layer
GET  /soul                    Load SOUL.md identity file
POST /soul                    Update SOUL.md
GET  /agents-registry         Load AGENTS.md specialist registry
POST /agents-registry         Update AGENTS.md
GET  /skills                  List loaded skill files
POST /skills/reload           Hot-reload skill files from disk
POST /skills/match            Test skill matching for an input
POST /skills/unload           Remove a named skill
POST /skills/scan             Security scan a skill file
POST /skills/prune            Remove expired/stale skills
GET  /introspect/capabilities Full self-model: services, skills, flags, FSM, teammates, gaps
GET  /teammates               List teammate registry (slug, name, specialty, description)
POST /chat/teammate/{name}    Route query to named teammate with world-state injection
GET  /models                  Model registry info
POST /episodes/recall         Retrieve recent episodes
POST /checkpoint              Create named state checkpoint
GET  /checkpoints             List all checkpoints
GET  /checkpoint/{id}         Retrieve checkpoint
POST /checkpoint/{id}/restore Restore state to checkpoint
GET  /checkpoint/diff/{a}/{b} Diff two checkpoints
DELETE /checkpoint/{id}       Delete checkpoint
GET  /metrics                 Prometheus-format metrics
GET  /queue/stats             Priority queue statistics
GET  /logs                    Log ring buffer
```

### Tool-Gate (`tool-gate:8000`)

```
POST /gate/request            HMAC-validated gate decision
POST /gate/mode               Set PUB/WORK mode
GET  /gate/mode               Current mode
GET  /gate/pending            Pending gate decisions
POST /gate/cosign             Co-sign a pending decision
POST /gate/autonomy/request   Kai requests elevated autonomy (always pending_approval)
GET  /ledger/tail             Last N ledger entries
GET  /ledger/stats            Ledger statistics
GET  /ledger/verify           Merkle integrity check
GET  /ledger/merkle           Merkle root
```

### Memu-Core (`memu-core:8001`)

```
POST /memory/memorize         Store a memory (embedding + metadata)
POST /memory/retrieve         ANN vector search
POST /memory/retrieve_ranked  Ranked retrieval (embedding + recency + importance)
POST /memory/forget           Delete by id or query
GET  /memory/stats            Store statistics
POST /memory/consolidate      MARS consolidation pass
POST /memory/self-reflect     Reflection pass
GET  /health                  Deep health (DB + vector store + embedding backend)
POST /recover                 Self-heal memory layer
POST /api/embed               Generate embedding for text
```

### Skill Hunter (`skill-hunter:8045`)

```
POST /hunt                    Hunt a skill: gap → keyword table → PyPI → .md file
GET  /skills                  List auto-generated skills (with full metadata)
GET  /skill/{name}/health     Skill metadata: error_count, disabled, provenance
POST /skill/{name}/error      Report runtime error (auto-disable at ≥3 errors)
GET  /health
```

### House Doctor (`house-doctor:8046`)

```
POST /diagnose                Differential diagnosis from observations + world_state
GET  /rules                   List all 9 diagnostic rules
GET  /health
```

---

## Feature Flags

All 35 flags are toggleable via `FF_{NAME}=true|false` env var. Defaults shown. Changing a flag requires a service restart.

| Flag | Default | What It Gates |
|------|---------|---------------|
| `FF_CHECKPOINT_AUTO` | ✓ | Auto-checkpoint on /recover and /dream |
| `FF_TREE_SEARCH` | ✓ | CoT tree search with conviction pruning |
| `FF_PRIORITY_QUEUE` | ✓ | Latency-sensitive priority queue |
| `FF_SAGE_CRITIQUE` | ✓ | Verifier self-critique + adversary self-review |
| `FF_IMAGINATION_ENGINE` | ✓ | P19 imagination / scenario simulation |
| `FF_PROACTIVE_AGENT` | ✓ | Background proactive observer (anomaly + world model, every 5 min) |
| `FF_OPERATOR_MODEL` | ✓ | P22 operator preference learning |
| `FF_NARRATIVE_IDENTITY` | ✓ | P18 narrative identity context injection |
| `FF_CONSCIENCE_FILTER` | ✓ | P20 conscience value-gate on actions |
| `FF_MARS_CONSOLIDATION` | ✓ | MARS memory decay + consolidation |
| `FF_SELF_ASSESSMENT` | ✓ | P14 temporal self-model |
| `FF_SECURITY_AUDIT` | ✓ | P9 automated security self-hacking |
| `FF_FINANCIAL_CONTEXT` | ✓ | Inject CIS/VAT/tax summary on finance queries |
| `FF_CONTEXT_ENRICHMENT` | ✓ | Master toggle: 14-way context gather + world state injection |
| `FF_ANOMALY_DETECTION` | ✓ | D88/M1: 2σ rolling baselines per sensor |
| `FF_WORLD_MODEL_PERSISTENCE` | ✓ | D88/M4: write world_state to memu-core each cycle |
| `FF_SENSORY_LEARNING` | ✓ | D88/M5: detect recurring patterns across 10 cycles |
| `FF_SKILL_HUNTER` | ✓ | D88/M6+M8: skill-hunter integration + reactive acquisition |
| `FF_PROACTIVE_SCHEDULING` | ✓ | D88/M7: calendar + sensor fusion → schedule memories |
| `FF_FSM` | ✓ | D89: system FSM (IDLE/ACTIVE/FOCUSED/DEGRADED/RECOVERING) |
| `FF_PERSISTENT_TEAMMATES` | ✓ | D89: load teammate personas + /chat/teammate/{name} |
| `FF_HOUSE_DOCTOR` | ✓ | D89/E: House Doctor differential diagnosis calls |
| `FF_RITUAL_DISCOVERY` | ✓ | D89/C: ritual proposals at ≥7/10 pattern cycles |
| `FF_GAP_LOGGING` | ✓ | D89/C1: gap counter, hunt only after 3 misses |
| `FF_TRUST_NEGOTIATION` | ✓ | D89/B: autonomy request endpoint (always pending_approval) |
| `FF_PREDICTIVE_EMPATHY` | ✓ | D89/D: emotional_context key in world model (stub) |
| `FF_CURIOSITY` | ✓ | D89/F: idle curiosity tick (GPU-gated no-op in Phase 0) |
| `FF_WAKE_INTENT_ROUTING` | ✗ | Pre-classify chat intent via wake-intent before routing |
| `FF_GRAPH_INGEST` | ✗ | Fan-out memorize/forget to memu-graph |
| `FF_LETTA_TASKS` | ✗ | Delegate tasks to letta-agent memory controller |
| `FF_LETTA_MEMORY_SYNC` | ✗ | Sync Letta archival memories back to memu-core |
| `FF_DREAM_ENABLED` | ✗ | 6-phase dream cycle consolidation |
| `FF_EVOLVER_ENABLED` | ✗ | Agent-Evolver: cluster failures → proactive insights |
| `FF_SAGE_SELF_REVIEW` | ✗ | SAGE critique on all plans (not just high-stakes) |

---

## Operator Console

**http://localhost:8080/app** — 10 views, keyboard shortcuts, installable as PWA.

| View | Key | What You See |
|------|-----|-------------|
| **Chat** | `Ctrl+1` | Streaming conversation, PUB/WORK toggle, feedback ratings, struggle detection, 🔊 speak button |
| **Dashboard** | `Ctrl+2` | Service health grid, pipeline status, fusion metrics |
| **Thinking** | `Ctrl+3` | Live conviction pipeline, tempo gauge, boundary map, silence signals, dream state |
| **Settings** | `Ctrl+4` | Mode, notifications, markdown toggle, PWA install |
| **Goals** | `Ctrl+5` | Ohana goals, drift alerts, progress bars, reminders, scheduled tasks |
| **Memory** | `Ctrl+6` | Memory browser — search by query or category, scores, stats |
| **Logs** | `Ctrl+7` | Ring-buffer log viewer — level/time filter, monospace, colour-coded |
| **Soul** | `Ctrl+8` | Mood cards, emotion timeline, domain confidence, self-reflection journal, milestones; SOUL.md + AGENTS.md live editor |
| **Canvas** | — | D3 v7 SVG: force-simulation mind-map (drag+zoom), emotion timeline area chart, plan-flow with arrows, memory graph (trust-tier colours, category hubs) |
| **Diary** | — | Memory diary — browse recent, date groups, rich cards with emotion/pin/trust badges |
| **System** | — | Sysmetrics gauges (CPU/RAM/disk), process table, screen-watcher controls |
| **Feeds** | — | Email inbox, RSS articles + search, weather, air quality, docker containers, calendar widget |
| **Broker** | — | Binance tickers, balance, positions, orders, PnL, Quick Watch → monitor rules |
| **Monitor** | — | Background rule engine — add/delete rules, live alert feed |

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Focus chat input |
| `Ctrl+Shift+M` | Toggle PUB/WORK mode |
| `Escape` | Close dropdown / stop generation |
| `browse: <url>` in chat | Navigate browser-agent to URL |
| `search: <query>` in chat | Browser search shortcut |

---

## Personality Modes

| Mode | Personality | When |
|------|------------|------|
| **WORK** | Professional, focused, precise. Never lies, never sugarcoats. Proactive but concise. | Mon-Fri 08–18 (auto) |
| **PUB** | Genuine mate. Casual, witty, opinionated. All topics. Not a service — a companion. | Evenings, weekends (auto) |

Toggle: `Ctrl+Shift+M` in dashboard, or `POST /gate/mode`. Manual override lasts 4h.

---

## Honest Limitations

> No illusions. Here is what is real and what is waiting.

| Area | Reality | What Fixes It |
|---|---|---|
| **LLM Model** | Default `qwen2.5:0.5b` (~500M params) is a test placeholder — too small for meaningful reasoning, planning, or emotional intelligence | Upgrade to 7B+ model (`qwen2.5:7b`, `llama3:8b`). RTX 5080 = 3 env vars to switch. Model registry auto-adapts context, prompts, timeouts |
| **Cognitive FSM wiring** | ~~Not wired~~ **DONE (D90)** — `POST /chat/swarm` runs the full pipeline. Scout→GATHER, Sage→DEBATE+CONVICTION_GATE, Doctor→FACT_CHECK, Oracle→CAUSAL_CHECK. All 5 stage functions are real implementations calling live memory/LLM/adversary dependencies. | — |
| **Teammate reputation** | ~~No reputation tracking~~ **DONE (D90)** — per-teammate `TeammateRep` (total_calls, successful_handoffs, avg_confidence, reliability). Weights applied in `resolve_conflict()`. Persisted to `data/teammate_reputation.json`. | Add per-teammate memory slice (top-k retrieval scoped to teammate specialty) |
| **Counterfactual Rehearsal** | `can_rehearse()→False`, `rehearse()→stub_pending_gpu`. Interface and schema are correct. | Activate in Phase 1 when GPU enables real decision-branch simulation |
| **Predictive Empathy** | `emotional_context` key exists in world model with the right schema shape. No inference until emotional memory accumulates. | Implement after 30+ days of real emotional memory history |
| **Resource-Aware Curiosity** | `idle_curiosity_tick()` is a no-op on CPU. CURIOSITY.md seed questions are ready. | Activate on GPU: pick open question → research → append finding |
| **Specialist Routing** | Keyword regex classification, not ML-based. All 3 specialists currently route to the same Ollama endpoint | Wire separate model endpoints when GPU arrives |
| **Trust Negotiation** | `POST /gate/autonomy/request` always returns `pending_approval`. Human-approval flow not built yet. | Build approval UI + operator feedback loop in Phase 2 |
| **Coverage** | ~63% combined across 5 modules. `agentic/app.py` (34%) and `memu-core/app.py` (53%) anchor the total down — service-route-heavy files unreachable offline | CI gate: 60% (`--cov-fail-under=60`). `make coverage` enforces locally |
| **Memory Persistence** | Default: TurboVec ANN index + Postgres metadata (dev/CI). Sovereign: pgvector. Graph memory (`memu-graph`) is feature-flagged off by default (`FF_GRAPH_INGEST=false`) | Set `VECTOR_STORE`, `TURBOVEC_INDEX_PATH` in `.env` |
| **Security Defaults** | HMAC enforced. DB password is `localdev` by default. Dev HMAC secret blocked unless `HMAC_ALLOW_DEV_SECRET=true` | Set `DB_PASSWORD`, `INTERSERVICE_HMAC_SECRET` for production |

---

## Milestone History

> 39 shipped. Zero skipped. Every milestone has tests.

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
P12 Self-Deception Detect.  ██████████ DONE   H3b Checkpointing       ██████████ DONE
P13 Improvement Gate        ██████████ DONE   MARS Memory Consol.     ██████████ DONE
─── ─────────────────────── ────────── ────   P23 SAGE Critique       ██████████ DONE
J1–J7 Jewels (7 features)  ██████████ DONE   P24 Agent-Evolver       ██████████ DONE
P1–P5 Enhancements         ██████████ DONE   GC  Eng. Gap-Close      ██████████ DONE
H3  Context Budget          ██████████ DONE   H4  Hardening Sprint    ██████████ DONE
STT Whisper Audio Input     ██████████ DONE   TTS Voice Synthesis     ██████████ DONE
Browser Navigation          ██████████ DONE   Vision / Camera         ██████████ DONE
D87 Cognitive Architecture  ██████████ DONE   (world context + proactive observer + skill match)
D88 Advanced Cognition      ██████████ DONE   (anomaly baselines, correlation, skill-hunter, scheduling)
D89 Cognitive Depth         ██████████ DONE   (FSM, cognitive pipeline, teammates, house-doctor, foundations)
```

### Milestone Summary

| Sprint | Key Deliverables |
|--------|-----------------|
| **D87** | `_get_world_context()` (9-service parallel probe per /chat); `_proactive_observer()` background loop; skill matching wired into /chat; ghost flag fixes (FF_CONTEXT_ENRICHMENT, FF_PROACTIVE_AGENT); 14-way context gather; 17 tests |
| **D88** | M1 rolling 2σ anomaly baselines (48-reading window); M2 `/introspect/capabilities` self-map; M3 cross-sensor correlation; M4 world_state JSON persistence; M5 10-cycle pattern learning; M6 skill-hunter service (port 8045); M7 calendar+sensor scheduling; M8 reactive gap-triggered skill acquisition; 5 flags; 44 tests |
| **D89** | System FSM (5 states, 9 events, 16 transitions); Cognitive reasoning FSM (GATHER→DEBATE→FACT_CHECK→CAUSAL_CHECK→CONVICTION_GATE→PRESENT, HALT/ESCALATE/RETHINK, per-swarm configs, schema-validated handoffs); persistent teammates Scout/Doctor/Sage/Oracle; house-doctor service (9 rules D001–D009); skill provenance (YAML front-matter + `.meta.json` sidecars, auto-disable at 3 errors); gap logging (GAP_HUNT_THRESHOLD=3); world model provenance ({value,source,timestamp,confidence}); emergent ritual discovery (≥7/10 cycles); GPU-era foundations (counterfactual, trust negotiation, predictive empathy, curiosity); 8 flags; 47 tests |

---

## Roadmap & End Goal

**Where we are:** Phase 0 complete (as of 2026-07-24). All CPU-safe backlog is shipped and on `main`. The conviction/trust loop, minimal stack live CI boot, process-level failure isolation (hot vs cold paths), sensory world-context injection, proactive observer, 8 intelligence mechanisms, and D89 cognitive depth (FSM, teammates, house-doctor, GPU-era foundations) are all complete.

**The single unlock condition:** GPU hardware arrival (RTX 5080).

When it arrives:
1. `OLLAMA_MODEL=qwen2.5:7b` — real reasoning quality
2. `FF_LETTA_TASKS=true` — archival memory activation
3. `FF_GRAPH_INGEST=true` — production graph memory
4. Counterfactual rehearsal + curiosity tick activate automatically
5. Multi-specialist routing with separate model endpoints
6. Real STT (faster-whisper large), TTS with voice quality, avatar

**Phase 1 priorities after GPU:**
- ~~Wire cognitive FSM stage functions~~ Done (D90) — `POST /chat/swarm` live
- ~~Build teammate reputation scores~~ Done (D90) — per-teammate TeammateRep + resolve_conflict()
- Add per-teammate memory slices (top-k retrieval scoped to specialty)
- Implement predictive empathy from accumulated emotional memory
- Build trust negotiation approval UI (human-in-the-loop autonomy gate)
- Real multi-model consensus (3 specialist endpoints, once GPU enables 7B+ models)

**End-goal:** a fully offline, self-hosted sovereign AI companion — chat, memory, perception, voice, avatar — all gated by the conviction/trust loop and circuit-breaker infrastructure already built. No cloud dependency. No single point of failure. The architectural principles established here (process-level isolation, reused circuit breakers, no split without checking shared state) are the template for the rest of the stack.

---

## Repo Structure

```
agentic/               # Reasoning brain
  app.py               # 30 API endpoints — chat, skills, teammates, checkpoints, introspect
  system_fsm.py        # D89: operational state machine (IDLE/ACTIVE/FOCUSED/DEGRADED/RECOVERING)
  cognitive_fsm.py     # D89: reasoning pipeline FSM (GATHER→…→PRESENT, HALT/ESCALATE/RETHINK)
  teammates.py         # D89: named cognitive personas (Scout/Doctor/Sage/Oracle)
  counterfactual.py    # D89/A: counterfactual rehearsal stub (GPU-era)
  curiosity.py         # D89/F: resource-aware curiosity (GPU-gated)
  conviction.py        # Conviction scoring (5-signal + modifiers)
  adversary.py         # Adversary challenge engine (7 types)
  tree_search.py       # CoT tree search with counterargument debate
  introspect_app.py    # Dream / evolve / security-audit (cold-path split)
  config.py            # Model registry, swarm configs, context budgets
tool-gate/             # HMAC auth, rate limit, policy, autonomy request gate
memu-core/             # Memory hot path: memorize / retrieve / rank / embed
  introspect_app.py    # Store maintenance cold path: compress / decay / quarantine
memu-graph/            # Cognee/Kuzu knowledge graph — entity/relation ingest→query→forget
house-doctor/          # D89: system health differential diagnosis (port 8046)
skill-hunter/          # D88/D89: autonomous skill acquisition via PyPI (port 8045)
supervisor/            # Dual-layer watchdog, circuit breaker, self-heal
verifier/              # Semantic fact-checking (embedding + keyword), SAGE self-critique
executor/              # Sandboxed code execution (AST validation + allowlist)
fusion-engine/         # Multi-signal consensus (embedding cosine + Jaccard fallback)
dashboard/             # 10-view operator console (FastAPI + Starlette)
heartbeat/             # System pulse, world anchor, auto-sleep
data/
  teammates/           # Teammate persona definitions (*.md)
    scout.md           # Skill discovery specialist
    doctor.md          # System health differential diagnosis
    sage.md            # Reflection + counterargument
    oracle.md          # Probabilistic trend extrapolation
  skills/              # Hot-loadable skill files (*.md with YAML front-matter)
  RITUALS.md           # Operator-co-authored rituals (proposed by ritual discovery)
  CURIOSITY.md         # Open questions log (seed questions from D89, GPU fills it)
common/                # Shared libraries
  feature_flags.py     # 35 FF_* flags — runtime toggle without code changes
  llm.py               # LLM chassis with retry/backoff (LLM_MAX_RETRIES=3)
  resilience.py        # resilient_call() — circuit breaker + exponential backoff
  auth.py              # HMAC signing + verification
  errors.py            # 20 structured error codes (E1001–E4004)
  gpu_utils.py         # GPU detection + model tier selection
  ab_log.py            # A/B query logger
perception/            # Perception services
  audio/               # STT (faster-whisper, port 8021)
  camera/              # Camera capture (port 8020)
  vision/              # OpenCV face detect + DeepFace emotion (port 8023)
  clipboard/           # Clipboard read/write (port 8024)
  files/               # File system access (port 8025)
  wake/                # Wake-word + intent judge (port 8022)
output/                # Output services
  tts/                 # edge-tts voice synthesis (port 8030, en-GB-RyanNeural)
  notify/              # Push notifications (port 8031)
  avatar/              # Avatar generation (port 8081)
browser-agent/         # Playwright Chromium navigation (port 8040)
document-parser/       # Multi-format text extraction (port 8032)
monitor-service/       # Background rule engine with notify + TTS (port 8033)
broker-bridge/         # Binance REST + HMAC-signed (port 8034)
sysmetrics/            # System health snapshot via psutil (port 8035)
screen-watcher/        # Screenshot diff + change alert (port 8036)
email-reader/          # IMAP read-only polling (port 8037)
news-feed/             # RSS aggregation + keyword search (port 8038)
weather-service/       # Weather data (port 8039)
docker-watcher/        # Docker container health (port 8041)
airquality-service/    # Air quality index (port 8042)
calendar-service/      # CalDAV events + summaries (port 8043)
git-watcher/           # Git dirty/stash/branch status (port 8044)
sandboxes/shell/       # Read-only shell sandbox (allowlist, 64 KB cap)
letta-agent/           # Letta archival memory controller (port 8062, FF_LETTA_TASKS gated)
financial-awareness/   # CIS/VAT/tax arithmetic, offline UK (port 8063)
kai-advisor/           # Self-employment advisor (port 8090)
telegram-bot/          # Telegram voice + text (port 8025)
screen-capture/        # Screen OCR pipeline (port 8059)
orchestrator/          # Final risk authority before execution
memory-compressor/     # Memory summarisation (port 8057)
ledger-worker/         # Audit trail persistence (port 8056)
metrics-gateway/       # Prometheus aggregation (port 8058)
backup-service/        # pg / redis / memory backup (port 8054)
calendar-sync/         # Calendar synchronisation (port 8055)
workspace-manager/     # Workspace lifecycle (port 8060)
scripts/               # 128+ test files, automation, integration scripts
kai-pm/                # PM brain: DECISIONS.md, SESSION_BOOTSTRAP.md, roadmap, tech watch
docs/                  # Plans, runbooks, architecture, backlog
security/              # HMAC/auth hardening helpers
```

---

## Engineering Toolchain

| Tool | Purpose | Command | Auto? |
|---|---|---|---|
| **sync-docs** | Patch README/backlog metrics from codebase scan | `make sync-docs` | On demand, gates merge |
| **check-docs** | Read-only freshness check (exit 1 if stale) | `make check-docs` | In merge-gate + CI |
| **go_no_go** | py_compile all service entry points | `make go_no_go` | Pre-commit hook + CI |
| **merge-gate** | Full validation: lint + docs + tests + quality | `make merge-gate` | Manual before merge |
| **pre-commit** | Flake8, mypy, secret-detect, YAML, go_no_go | Auto on `git commit` | Yes |
| **dep-audit** | pip-audit for known CVEs | `make dep-audit` | CI |
| **coverage** | pytest-cov HTML report (60% floor) | `make coverage` | On demand |
| **coverage-floors** | Per-module coverage gates (agentic ≥45%, memu-core ≥60%) | `make coverage-floors` | CI |
| **Trivy** | Container image scanning (CRITICAL+HIGH, exit 1) | CI auto | core-tests.yml |
| **health-sweep** | Hit /health on all running services | `make health-sweep` | On demand |
| **chaos-ci** | Fault injection for resilience testing | `python scripts/chaos_ci.py` | On demand |
| **test-github-models** | GitHub Models CI LLM smoke test (best-effort) | `make test-github-models` | CI (non-breaking) |

---

## Test Targets (82)

`make test-core` runs all 82 targets. Each target maps to a `scripts/test_*.py` file.

<details>
<summary>Click to expand full test target list</summary>

```bash
# Memory & storage
make test-phase-b-memu          make test-memu-pg              make test-memu-turbovec
make test-letta                 make test-financial             make test-memu-retrieval

# Service integration
make test-dashboard-ui          make test-dashboard             make test-thinking-pathways
make test-tool-gate             make test-tool-gate-security    make test-telegram
make test-conviction            make test-audio                 make test-camera
make test-executor              make test-agentic-service       make test-agentic-introspect
make test-kai-advisor           make test-tts                   make test-avatar
make test-heartbeat             make test-auth-hmac             make test-agentic
make test-self-emp

# Sensory services
make test-upload-fuzz           make test-audio-transcribe      make test-browser-agent
make test-vision-service        make test-clipboard-service     make test-files-service
make test-notify-service        make test-document-parser       make test-monitor-service
make test-broker-bridge         make test-sysmetrics            make test-screen-watcher
make test-email-reader          make test-news-feed             make test-weather-service
make test-docker-watcher        make test-airquality            make test-calendar-service
make test-git-watcher           make test-broker-bridge-yfinance

# Reasoning & cognition
make test-episode-saver         make test-episode-spool         make test-error-budget
make test-invoice               make test-router                make test-planner
make test-adversary             make test-failure-taxonomy      make test-selaur
make test-contradiction         make test-gem                   make test-planner-prefs
make test-silence               make test-self-deception        make test-temporal-self
make test-predictive            make test-improvement-gate      make test-tree-search
make test-priority-queue        make test-model-selector

# Personality / P-series
make test-p3-organic            make test-p4-personality        make test-p16-operational
make test-p17-emotional-intelligence    make test-p18-narrative-identity
make test-p19-imagination-engine        make test-p20-conscience-values
make test-p21-proactive-agent           make test-p22-operator-model

# Hardening & quality
make test-h1-hardening          make test-h2-self-healing       make test-mars-consolidation
make test-sage-critique         make test-agent-evolver         make test-checkpoint
make test-v7                    make test-prod-hardening        make test-error-codes
make test-feature-flags         make test-dream-state           make test-security-audit
make test-gaps-sprint

# Chassis & LLM layer
make test-chassis               make test-chassis-runtime       make test-predictive-failure
make test-multi-modal           make test-world-anchor          make test-self-healing-phases

# Validation & e2e
make test-j-series              make test-wake                  make test-behavioral
make test-docker-e2e

# Intelligence layers (D87–D89)
make test-kai-intelligence      make test-cognitive-mechanisms  make test-d89-cognitive-depth
```

</details>

---

## Build & Run

```bash
# Build
docker compose -f docker-compose.minimal.yml build    # Core 33 services
docker compose -f docker-compose.full.yml build        # Full 35 services

# Run
make core-up       # Start minimal stack (33 services)
make core-down     # Stop minimal stack
make full-up       # Start full stack
make full-down     # Stop full stack

# Validate
make go_no_go      # Syntax check all entry points
make test-core     # All 82 test targets (2,671 tests)
make merge-gate    # Full pre-merge validation
```

### Environment Variables (Key Ones)

```bash
# Model
OLLAMA_MODEL=qwen2.5:0.5b        # Default (CPU safe). Change to 7b on GPU.
OLLAMA_BASE_URL=http://ollama:11434

# Memory
VECTOR_STORE=turbovec            # turbovec (default) | postgres | memory
TURBOVEC_INDEX_PATH=/data/turbovec.tv
MEMU_ALLOW_FAKE_EMBEDDINGS=true  # Required for offline CI runs

# Security
HMAC_ALLOW_DEV_SECRET=true       # Dev only — never in production
DB_PASSWORD=localdev             # Override for production
INTERSERVICE_HMAC_SECRET=<secret>

# Broker (stays inside broker-bridge only — never exposed to dashboard)
BINANCE_API_KEY=<key>
BINANCE_API_SECRET=<secret>
BINANCE_MODE=spot                # spot | futures

# Email
MAIL_HOST=imap.example.com
MAIL_USER=<user>
MAIL_PASS=<pass>

# Feature flags (all default shown — override any individually)
FF_GRAPH_INGEST=false            # Enable for production graph memory
FF_LETTA_TASKS=false             # Enable after GPU + live Ollama verified
FF_DREAM_ENABLED=false           # Enable for offline memory consolidation
FF_SKILL_HUNTER=true             # Autonomous skill acquisition
FF_PROACTIVE_AGENT=true          # Background sensory observer
FF_CONTEXT_ENRICHMENT=true       # 14-way context gather
FF_HOUSE_DOCTOR=true             # System differential diagnosis
FF_FSM=true                      # Operational state machine
FF_PERSISTENT_TEAMMATES=true     # Named cognitive teammates
```

---

## Session Continuation Guide

> For AI assistants resuming work. Read `kai-pm/SESSION_BOOTSTRAP.md` first.

### Target Hardware
- **Dev:** GitHub Codespace (CPU only, qwen2.5:0.5b)
- **Prod:** Lenovo laptop + **RTX 5080 GPU** + **TPM 2.0**
- GPU arrival = real LLM inference, GPU-era feature activation, multi-model consensus
- All code works in both environments (stubs in Codespace, live on laptop)

### Key Docs (read in order)
1. [`kai-pm/SESSION_BOOTSTRAP.md`](kai-pm/SESSION_BOOTSTRAP.md) — fast re-hydration, current state, next move
2. [`kai-pm/DECISIONS.md`](kai-pm/DECISIONS.md) — append-only decision log (D1–D89)
3. [`kai-pm/STATUS.md`](kai-pm/STATUS.md) — sprint health, open PRs, blocked items
4. [`CHANGELOG.md`](CHANGELOG.md) — full semver changelog
5. [`docs/PROJECT_BACKLOG.md`](docs/PROJECT_BACKLOG.md) — living backlog

### Cross-Check: What's Real vs What Needs Hardware

**Working now (CPU / Codespace):**
- [x] 33 services in minimal stack, all health-checked, compose validated
- [x] TurboVec ANN persistence (default; no pgvector extension needed)
- [x] pgvector persistence (sovereign stack; opt-in via `VECTOR_STORE=postgres`)
- [x] HMAC auth enforced, dev secret blocked by default
- [x] System FSM (IDLE/ACTIVE/FOCUSED/DEGRADED/RECOVERING) — wired into /chat + observer
- [x] Cognitive reasoning FSM framework (GATHER→PRESENT, HALT, per-swarm configs, schema-validated)
- [x] Swarm Assembly (D90) — real stage functions wired; `POST /chat/swarm` live; reputation tracking
- [x] Persistent teammates (Scout/Doctor/Sage/Oracle) — loaded and invokable
- [x] House Doctor service (port 8046) — 9 diagnostic rules, wired into proactive observer
- [x] World model provenance ({value, source, timestamp, confidence} per field)
- [x] Gap logging + threshold-gated reactive skill acquisition
- [x] Emergent ritual discovery (≥7/10 cycles → RITUALS.md)
- [x] Skill provenance (YAML front-matter + .meta.json sidecars, auto-disable at 3 errors)
- [x] 14-way context gather, world context injection, proactive observer
- [x] Anomaly detection with rolling 2σ baselines
- [x] Cross-sensor correlation reasoning
- [x] Supervisor auto-healing loop (deep /health + /recover every 15s)
- [x] Executor sandboxing (allowlist + AST validation + shell=False)
- [x] memu-graph (Cognee/Kuzu) — CI-verified against live Ollama stack
- [x] 82 test targets, 2,671 tests, zero failures
- [x] Pre-commit, dep scanning, container scanning (Trivy)
- [x] Circuit breakers, exponential backoff, resilient_call()
- [x] MARS memory decay, spaced repetition
- [x] Context budget trimming, structured errors, 35 feature flags
- [x] Debate engine (`tree_search.py`, `conviction.py`, `adversary.py`)

**Infrastructure ready, needs GPU to activate:**
- [x] Cognitive FSM stage functions wired — `POST /chat/swarm` live (D90)
- [ ] Counterfactual rehearsal — `can_rehearse()→False` until Phase 1
- [ ] Predictive empathy — `emotional_context` stub; needs emotional memory history
- [ ] Resource-aware curiosity — `idle_curiosity_tick()` no-ops on CPU
- [ ] Teammate reputation scores + memory slices — not yet built
- [ ] `FF_LETTA_TASKS=true` — pending live Ollama verification
- [ ] `FF_GRAPH_INGEST=true` in production — pending GPU workload validation
- [ ] Emotional intelligence quality — code works, qwen2.5:0.5b too small to detect emotions well
- [ ] Multi-model consensus — architecture ready, all 3 specialists route to same model

---

## Key Guidelines

- Run `make go_no_go` before committing any service `app.py` changes
- Every service: `/health` endpoint. Core services: deep `/health` + `/recover`
- Inter-service HTTP: use `common.resilience.resilient_call()` (retry + circuit breaker)
- HMAC: `TOOL_GATE_DUAL_SIGN=true`, then `INTERSERVICE_HMAC_STRICT_KEY_ID=true`
- `BINANCE_API_KEY` / `BINANCE_API_SECRET` stay inside `broker-bridge` only — never in dashboard or agentic
- `kai-pm/DECISIONS.md` is append-only — never edit past entries, supersede with new numbered entry
- Never commit credentials — `.env` files only (see `.env.example`)
- `make merge-gate` before every PR
- `make sync-docs` after major changes

---

<p align="center">
  <em>Built by Dainius + Kai. Not for sale. Not for anyone else. Sovereign.</em>
</p>
