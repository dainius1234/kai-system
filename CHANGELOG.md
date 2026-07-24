# Changelog

All notable changes to the Kai System project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed (PR #92, 2026-07-24)

- **flake8 F821 in model_registry.py** (PR #92): `List` and `Any` were used in type annotations but missing from `from typing import ...`. Caused CI `Python application` check to fail after PR #91 merge.

### Added (PR #91, D82–D83, 2026-07-24)

- **C10 A/B query logger** (D83): `common/ab_log.py` — JSONL log per `LLMRouter.query()` call recording model name, latency, and response quality metrics (lexical diversity, uncertainty penalty). Enabled by default; disable via `AB_LOG_ENABLED=false`.
- **P1 screen-capture headless tests** (PR #91): 20 pytest tests for `screen-capture/app.py` — `/health`, `/metrics`, `/capture/file` (PNG/JPEG upload, 413, 422), `/capture` watchdir fallback. No X11/Tesseract required.
- **C4 classify_semantic fallback tests** (PR #91): 4 test groups verifying `classify_semantic()` falls back to keyword routing when `sentence-transformers` absent or `min_quality_tier` too high.
- **F4/F6 feature flag tests** (PR #91): Offline verification that CONTEXT_ENRICHMENT defaults True and DREAM_ENABLED/EVOLVER_ENABLED/SAGE_SELF_REVIEW default False; env override round-trip tested.
- **5 new Makefile targets** (PR #91): `test-memu-routes`, `test-agentic-routes`, `test-context-enrichment-ab`, `test-ab-log`, `test-screen-capture`.
- **GPU Arrival Runbook** (D82, PR #91): `kai-pm/GPU_ARRIVAL_RUNBOOK.md` with G1–G8 verified shell commands for RTX 5080 onboarding day.

### Fixed (PR #91, 2026-07-24)

- **morning/evening briefing crash** (PR #91): `(t.get("fire_at") or "").startswith(...)` — `.get(key, "")` returns `None` when key exists with value `None`, causing `AttributeError`. Fixed in `memu-core/app.py`.
- **identity narrative float/str timestamp crash** (PR #91): `get_identity_narrative()` compared autobiography `timestamp` (float from `time.time()`) with ISO string using `<`, raising `TypeError`. Fixed by normalising float to ISO before comparison.

### Added (Phase 0 backlog sweep — PRs #86–#87 + branch, D60–D70, 2026-07-21)

- **SOUL.md / AGENTS.md identity editor** (D60, PR #86): `GET /soul` + `PUT /soul` in `agentic/app.py`; `GET /agents` + `PUT /agents` mirrored. Dashboard Soul tab renders both documents in side-by-side editors with live save. `security/SOUL.md` and `security/AGENTS.md` are the canonical on-disk files.
- **Live Canvas D3 v7** (D61, PR #86): Canvas tab (`Ctrl+2`) renders a real D3 v7 force-directed memory graph — nodes are memories, edges are temporal/emotional links, hover shows full text. Replaces the placeholder SVG.
- **Memory Diary tab** (D62, PR #86): Diary tab (`Ctrl+6`) shows full memu-core memory history grouped by date with rich cards (emotion badge, source tag, confidence bar, PII-redaction indicator). Lazy-loads 50-record pages.
- **PII auto-redaction in write path** (D63, PR #86): `memu-core/app.py` `memorize_event` detects PII (email, phone, NI/UTR, postcode patterns) via regex before storing; redacts matches with `[REDACTED-<type>]`. Dashboard Scanner panel shows per-category counts. `pii_redacted` field returned in memorize response.
- **H3: coverage gate** (D64, PR #86): `--cov-fail-under=65` added to `python-app.yml` pytest step and `make coverage` target. Closes RISKS.md R3.
- **Real ed25519 keypair generation** (D68): `scripts/auto_rotate_ed25519.py` `_new_keypair()` now uses `Ed25519PrivateKey.generate()` from the `cryptography` library — produces a mathematically-related keypair instead of two random blobs.
- **Makefile.archive** (D70): 10 dead/duplicate targets (`test-tempo`, `test-hmac-rotation-drill`, J-series filtered aliases, `cache-test-core`) deleted from main Makefile; preserved in `Makefile.archive` for reference.

### Fixed (D65–D67, D69, 2026-07-21)

- **CI: pii_redacted type error** (D65, PR #87): `memu-core/app.py` `memorize_event` returned `int` for `pii_redacted` inside a `Dict[str, str]` endpoint — `ResponseValidationError`. Fixed with `str(sum(...))`.
- **CI: chassis httpx mock** (D65, PR #87): `FakeResponse` in `test_chassis.py` lacked `status_code = 200`, causing `AttributeError` caught by broad `except Exception` which flipped `response.source` to `"error"`. Fixed by adding `status_code`, `headers=None`, and mock exception classes to the `fake_httpx` namespace.
- **CI: financial-awareness sys.modules collision** (D65, PR #87): bare `import app` in `test_financial_awareness.py` got `kai-advisor/app.py` due to alphabetical pytest discovery order. Fixed by loading via `importlib.util.spec_from_file_location` with a unique module name.
- **Hardcoded credentials in compose files** (D66): `docker-compose.full.yml` PG_URI (3 occurrences) and `docker-compose.sovereign.yml` `GF_SECURITY_ADMIN_USER` now use `${DB_PASSWORD:-localdev}` / `${GRAFANA_ADMIN_USER:-admin}` respectively. Makefile `init-memu-db` fallback updated to match.
- **OPENAI_API_KEY placeholder** (D67): `scripts/agentic_integration_test.py` placeholder changed from `"sk-test-placeholder-not-real"` to `""`.
- **Dashboard `alert()` in briefing** (D69): `triggerBriefing()` replaced `alert(msg)` with `_showBriefingModal()` — lightweight overlay with close button, click-outside dismissal, and dark/light theme CSS variables.
- **`test-tempo` orphan** (D70): `scripts/test_tempo.py` deleted; was testing a removed service and had no live references.

### Added (Phase 0.5 completion — PRs #82–#85, D55–D59, 2026-07-21)

- **Letta agent memory controller** (D55, PR #82): `letta-agent/` service (port 8062), feature flags `FF_LETTA_TASKS` / `FF_LETTA_MEMORY_SYNC`, agentic 12-way context gather with letta archival injection, `docker-compose.full.yml` wiring.
- **FF_GRAPH_INGEST=true** (D56, PR #83): graph fan-out from memu-core now defaults on in `docker-compose.full.yml`. `memu-graph` receives every memorize/forget event best-effort.
- **P29 CIS Financial Awareness service** (D57, PR #83): `financial-awareness/` service (port 8063). Endpoints: `/finance/cis/record`, `/finance/cis/summary`, `/finance/invoice`, `/finance/vat`, `/finance/summary`. UK CIS deduction rules (20%/30%/0%), MTD VAT threshold, Class 4 NI, income tax 2024/25 rates. 18 unit tests (`scripts/test_financial_awareness.py`). `finance_data` named volume for persistence.
- **Automation infrastructure** (D58, PR #84): `friday-cleanup.yml` (weekly lint/pip-audit/stale-branch/doc-sync), `weekly-report-card.yml` (Monday go/no-go + fast pytest), `scripts/backup_offsite.sh` (GPG-encrypted offsite backup), `docs/DEMO.md`, `docs/operator-journal/_template.md`, starter skill files.
- **Cloud LLM fallback backends** (D58, PR #84): Groq (`llama-3.3-70b-versatile`) and OpenRouter (`meta-llama/llama-3.3-70b-instruct:free`) added to `common/llm.py`. Opt-in via `GROQ_API_KEY` / `OPENROUTER_API_KEY`. `Authorization: Bearer` + `HTTP-Referer` headers handled. `.env.example` updated.
- **PWA service worker** (D58, PR #84): `dashboard/static/sw.js` — `kai-shell-v1` cache, 8 shell assets cached on install, network-first navigation, cache-first static, never intercepts `/api/`/`/stream`/`/health`. `index.html` wired with manifest link and SW registration.
- **Agentic financial context injection** (D58, PR #84): `FF_FINANCIAL_CONTEXT` flag (default on), `_FINANCE_KEYWORDS` frozenset, `_get_financial_context()` keyword-gated fetch from `financial-awareness`, 12-way gather expanded to 13-way, CIS/VAT/tax summary injected as system message on finance queries.
- **C3 — LLM retry/backoff** (D59, PR #85): `LLM_MAX_RETRIES` (default 3), `LLM_RETRY_BACKOFF` (default 1.0s), `_RETRY_STATUS_CODES = {429, 503}`. Exponential back-off loop in `_live_query()` for HTTP 429/503 and `ConnectError`/`TimeoutException`. Non-retriable exceptions break immediately.
- **Behavioral scoreboard** (D59, PR #85): `scripts/behavioral_scoreboard.py` — 5 test prompts, 0–100 score, A–F grade, always exits 0, offline-safe. Wired into `weekly-report-card.yml` as advisory CI step.
- **Finance dashboard tab** (D59, PR #85): `dashboard/app.py` proxy endpoints (`/api/finance/summary`, `/api/finance/cis`, `/api/finance/cis/record`). `app.html` Finance view (CIS stat cards, VAT/tax breakdown, Log CIS Payment form, recent records table, `refreshFinance()` + `logCisPayment()` JS).
- **PHONE_SETUP.md** (D59, PR #85): Android (Chrome) + iOS (Safari) PWA install walkthrough, feature table, troubleshooting.

### Fixed
- **memu-core Postgres extension race**: `PGVectorStore._init_schema()`'s `CREATE EXTENSION IF NOT EXISTS vector;` could raise `UniqueViolation` when `memu-core` and `memu-core-introspect` race against a freshly-initialized database — both can pass the existence check before either commits. Now caught and treated as success (the extension exists either way).
- **memu-graph startup crash**: Cognee's `LLMConfig` requires `LLM_API_KEY` to be non-empty when `LLM_PROVIDER=ollama` (pydantic all-or-nothing validator); added a placeholder value since Ollama performs no real auth.
- **memu-graph model-not-found 404s**: Cognee's `OllamaAPIAdapter` sends the configured model string to Ollama's API as-is, with no `ollama/` prefix-stripping. `LLM_MODEL`/`EMBEDDING_MODEL` in `docker-compose.full.yml`'s `memu-graph` block now use the bare model tag instead of `ollama/<tag>`.
- flake8 E999 (f-string backslash) blocking CI on main
- Removed stray file `pulls/48/review_comments`
- **TTS test de-flaked**: `scripts/test_tts_service.py` now mocks `edge_tts.Communicate` so the test runs offline and is deterministic — no more Bing WebSocket 403 failures in CI sandboxes
- **TTS app network errors → 503**: `output/tts/app.py` `_synthesize_edge` now maps aiohttp/OSError network failures to HTTP 503 (upstream unavailable) instead of 500
- **CVE-2026-40192**: Bumped `Pillow` from `10.4.0` to `>=12.2.0` in `screen-capture/requirements.txt`
- **CVE-2024-53981 / CVE-2026-24486 / CVE-2026-40347**: Bumped `python-multipart` from `0.0.12` to `>=0.0.26` in `perception/audio/requirements.txt` and `screen-capture/requirements.txt`
- **CVE-2024-47874 / CVE-2025-54121**: Bumped `fastapi` from `==0.115.0` to `>=0.116.2` and added explicit `starlette>=0.47.2` constraint across all 25 per-service `requirements.txt` files

### Added
- **memu-graph — Cognee/Kuzu graph memory layer** (D28-D32): new `memu-graph/` service wraps Cognee's knowledge-graph pipeline (entity extraction → KuzuDB graph storage → semantic search). Exposes `/graph/ingest`, `/graph/query`, `/graph/forget`. `memu-core` write-side fan-out (`FF_GRAPH_INGEST=false` by default) and read-side `/memory/graph/query` proxy wired into `agentic`'s parallel context fetch. MARS prune hook calls `/graph/forget`. Full live CI verification added (PR #79 — real Ollama/Cognee/KuzuDB container boot in `core-tests.yml`).
- **memu-core-introspect process split** (D21): 13 functions / 14 cold-path routes (compress, focus-compress, reflect, decay, quarantine, evidence-pack, and more) moved from `memu-core/app.py` into `memu-core/introspect_app.py` running as a separate container (`memu-core-introspect`, port 8009). Hot-path memorize/retrieve traffic can no longer be taken down by a bug in periodic store-maintenance code.
- **TurboVec ANN vector search** (D54): `VECTOR_STORE=turbovec` activated as the default in `docker-compose.full.yml` and `docker-compose.minimal.yml`. `TurboVecStore` (already fully implemented in `memu-core/app.py`) stores metadata + raw embeddings in Postgres and keeps an in-process `IdMapIndex` for compressed ANN similarity search, persisted to a `.tv` file on a named volume. Eliminates the `pgvector` extension requirement for development/CI. Sovereign production stack stays on `VECTOR_STORE=postgres` (pgvector).
- **C2 — Stream heartbeat / stall detection**: `STREAM_HEARTBEAT_TIMEOUT` env var (default 30 s).
  `LLMRouter.stream()` wraps per-token reads in `asyncio.wait_for`; emits a structured stall
  message and closes cleanly if no token arrives within the window.
- **C5 — Ollama pre-flight check**: `ensure_model_available()` in `common/llm.py` queries
  `/api/tags`, does prefix-aware name matching, caches results for `MODEL_TAGS_CACHE_TTL`
  (default 60 s), and falls back to `OLLAMA_MODEL` in `_live_query()` when the requested
  model is absent.  Ollama-unreachable → fail-open (returns `True`).
- **C9 — Model warm-up on startup**: `llm_warmup()` in `common/llm.py` fires via
  `asyncio.create_task` from a FastAPI `startup` hook in `langgraph/app.py`.  Checks model
  availability, optionally pulls it (`OLLAMA_AUTO_PULL=true`, default `false`), sends a
  single warm prompt, and logs completion time.  Controlled by `LLM_WARMUP_ENABLED`
  (default `true`).
- New env vars: `STREAM_HEARTBEAT_TIMEOUT=30`, `MODEL_TAGS_CACHE_TTL=60`,
  `LLM_WARMUP_ENABLED=true`, `OLLAMA_AUTO_PULL=false` (all documented in `.env.example`).
- Test suite `scripts/test_chassis_runtime.py` (C2/C5/C9 unit tests, 10 cases, all mocked).
- PM: introduced `kai-pm/` brain + `.github` automation (PM System v2)
- Context budget management (`CONTEXT_BUDGET_TOKENS` env, default 3072) — `_trim_context()` in agentic prevents system prompt from exceeding model context window
- Context budget test suite (`scripts/test_context_budget.py`, 11 tests)
- `make test-context-budget` Makefile target
- GPU Phase 0 hardware utilities in `common/gpu_utils.py` (`has_cuda`, `get_gpu_info`, `should_use_speculative_decoding`, `get_recommended_model`)
- Phase 0 implementation report: `docs/gpu_integration_phase0.md`
- memu-core manual persistence endpoint: `POST /memory/persist`
- New env controls for GPU/scaling and retrieval cap in `.env.example` (including `MEMU_MAX_CANDIDATES`)
- J2 wake-intent service (`perception/wake/app.py`) with:
  - `POST /wake/detect` (text/audio_b64 wake-word detect + confidence + debounce)
  - `POST /wake/intent` (tiny-model intent taxonomy classifier with strict JSON validation)
  - `POST /wake/process` (detect + intent combined)
  - `GET /health` (dependency-aware status)
- Wake-intent test suite (`scripts/test_wake_intent.py`) and `make test-wake` target (wired into `test-core`)
- Wake-intent dashboard proxy endpoints (`/api/wake/detect`, `/api/wake/intent`, `/api/wake/process`)
- New feature flag: `FF_WAKE_INTENT_ROUTING` (default off)
- **H2.2**: `MEMU_MAX_CANDIDATES` env var (default 500) caps `retrieve_ranked()` candidate fetch, preventing unbounded 10k-record loads on every retrieval call

### Changed
- Default Ollama model swapped `qwen2:0.5b` → `qwen2.5:0.5b` (compose defaults, `common/llm.py`, `common/gpu_utils.py`, `.env.example`) — `qwen2:0.5b`'s chat template very likely lacks tool-call markup, which would make Letta's model-discovery filter silently drop it; `qwen2.5:0.5b` is the generation Qwen's own tooling scopes native tool-call support to. `common/model_registry.py` gained a matching spec entry for the new tag.
- Replaced fabricated PM v2 content (DECISIONS, SEQUENCE, STATUS, RISKS) with honest, repo-grounded versions
- Added `kai-pm/STRATEGIC_PLAN.md` as canonical 5-phase roadmap pointer
- Verifier upgraded to semantic verification (v0.6.0) — uses memu-core `rank_score` (embedding similarity + relevance + importance + recency) instead of keyword-only matching
- Verifier now calls `/memory/evidence-pack` for richer evidence scoring, with fallback to `/memory/retrieve`
- Updated all stale documentation (README metrics, PROJECT_BACKLOG, Known Issues table)
- memu-core Redis handling hardened with reconnection backoff and persisted P17–P22 restore/sync lifecycle
- `common.llm.LLMRouter` live query path now uses model-aware timeout from `common.model_registry.model_timeout()`
- `common.model_registry` expanded with `deepseek-coder-v2:6.7b`, `qwen2.5-math:7b`, and `yi:34b`
- Langgraph `/chat` can now optionally pre-classify intent through wake-intent service before route selection when `FF_WAKE_INTENT_ROUTING=true`

### Previously Added
- Pre-commit hooks (flake8, trailing whitespace, secret detection, syntax check)
- CODEOWNERS file for automated review assignments
- Structured error codes (`common/errors.py`) with enumerated codes and HTTP status mapping
- Feature flags system (`common/feature_flags.py`) with env-based toggles
- Code coverage configuration (`.coveragerc`)
- Dependency vulnerability scanning (`pip-audit` in CI)
- Container image scanning (Trivy in CI)
- CHANGELOG.md (this file)

## [0.28.0] — 2026-03-22 — J-Series Jewels Roadmap

### Added
- 7 new planned features (J1–J7) from 2026 research (OpenClaw, Jarvis, Proact-VL)
- J1: Live Canvas Visualization (dashboard mind-map/graph)
- J2: Wake-word "Kai" + Intent Judge (whisper + tiny LLM)
- J3: Auto-Redaction PII (regex + OCR strip)
- J4: Proactive Low-Latency Voice (audio/video cue → speak-or-not)
- J5: Memory Viewer GUI (diary-style dashboard tab)
- J6: SOUL.md + AGENTS.md (persistent identity files)
- J7: Skills Auto-Install Hub (local skill loader)
- Full documentation update across PROJECT_BACKLOG, README, SESSION_BACKLOG,
  personality_and_proactive, unfair_advantages, next_level_roadmap, copilot-instructions

## [0.27.0] — 2026-03-22 — Close All Research Gaps

### Added
- Predictive Failure Modeling: `_linear_regression()`, `_forecast_failures()`, `/predict` endpoints (supervisor/app.py)
- Multi-Modal Sensory Input: voice emotion analysis (perception/audio/app.py), OpenCV frame analysis (perception/camera/app.py)
- External World Anchor: date/time context, local news/events feeds (calendar-sync/app.py)
- Bio-inspired Self-Healing: 4-phase ReCiSt engine in common/resilience.py
- 123 tests across 4 test files (test_predictive_failure, test_multi_modal, test_world_anchor, test_self_healing_phases)
- 4 new Makefile targets

## [0.26.0] — 2026-03-22 — Active Context Compression Loop

### Added
- Focus-compress endpoint: POST `/memory/focus-compress` with MARS-ranked focus zone
- Jaccard keyword clustering and memory merge logic
- Token budget meter: GET `/memory/token-budget` (configurable 50K default)
- 44 tests (`scripts/test_focus_compress.py`)

## [0.25.0] — 2026-03-22 — H3b LangGraph Checkpointing

### Added
- Checkpoint engine: create / list / load / diff / delete / restore full operational state
- Auto-checkpoint before `/recover` and after `/dream` cycle
- Manual save-points via `POST /checkpoint`
- Time-travel rollback via `POST /checkpoint/{id}/restore`
- Checkpoint diff via `GET /checkpoint/diff/{id_a}/{id_b}`
- 32 tests (`scripts/test_checkpoint.py`)

## [0.24.0] — 2026-03-22 — P24 Agent-Evolver Insight Engine

### Added
- `analyze_failures()` groups episodes by FailureClass, generates EvolutionSuggestions
- 8 fix templates mapped to failure classes with concrete remediation actions
- Priority assignment (critical/high/medium/low) based on frequency x severity
- Dream Phase 7 integration (`evolver_dream_phase()`)
- `POST /evolve/analyze` and `GET /evolve/suggestions` endpoints
- 34 tests (`scripts/test_agent_evolver.py`)

## [0.23.0] — 2026-03-22 — P23 SAGE Multi-Agent Critique

### Added
- Verifier self-critique (strategy 5): groupthink, thin-evidence, unsupported claims, contradictions
- Adversary self-review (challenge 7): false consensus, degraded groupthink, conflicting findings, over-optimism
- Both fire automatically before any action proposal
- 30 tests (`scripts/test_sage_critique.py`)

## [0.22.0] — 2026-03-22 — MARS Memory Consolidation

### Added
- Ebbinghaus stability parameter and retention formula R = e^{-τ/S}
- Conscience-filtered pruning during consolidation
- Nightly decay cycle with stability growth on rehearsal
- 35 tests (`scripts/test_mars_consolidation.py`)

## [0.21.0] — 2026-03-22 — H2 Self-Healing & Resilience

### Added
- Deep `/health` endpoints checking real dependencies on all core services
- `/recover` endpoints for self-healing without restart
- `resilient_call()` wrapper with retry, backoff, circuit breaker, and fallback
- TaskWatchdog for frozen async task detection
- Supervisor auto-recovery loop with cooldown
- 38 tests (`scripts/test_h2_self_healing.py`)

## [0.20.0] — 2026-03-22 — H1 Critical Hardening

### Fixed
- 7 security and stability issues across core services
- Injection detection hardening, error budget tuning, audit stream integrity

## [0.19.0] — 2026-03-21 — P22 Operator Model

### Added
- Echo engine, nudge escalation, cross-mode bridge, impact oracle, shadow branches
- 57 targets, 960 tests

## [0.18.0] — 2026-03-21 — P21 Proactive Agent Loop

### Added
- Scheduled tasks, reminders, briefings, action registry, agent summary
- Supervisor firing for proactive actions

## [0.17.0] — 2026-03-21 — P20 Conscience & Values Engine

### Added
- Emergent values, moral reasoning, loyalty memory, gratitude system

## [0.16.0] — 2026-03-21 — P19 Imagination Engine

### Added
- Counterfactual replay, theory of mind, creative synthesis, inner monologue

## [0.15.0] — 2026-03-21 — P18 Narrative Identity

### Added
- Life story engine, narrative context, identity coherence

## [0.14.0] — 2026-03-20 — P17 Emotional Intelligence

### Added
- Emotional memory, self-reflection, relationship timeline, epistemic humility, Soul dashboard

## [0.13.0] — 2026-03-20 — P16 Operational Intelligence

### Added
- Struggle detection, feedback loop, log aggregation, goals UI, memory browser

## [0.12.0] — 2026-03-20 — P4 Personality & Proactive

### Added
- Deep prompts, anti-annoyance, conversation holding, mode transitions

## [0.11.0] — 2026-03-20 — P3 Organic Memory

### Added
- Correction learning, category boost, spaced repetition, proactive engine, Ohana goals

## [0.10.0] — 2026-03-19 — Production Hardening

### Added
- Redis pubsub SSE, Docker secrets, backup service, HMAC rotation drill

## [0.9.0] — 2026-03-19 — HP2+HP4+HP5 Hardware Performance

### Added
- MoE model selector, CoT tree search with conviction pruning, priority queue

## [0.8.0] — 2026-03-18 — P8+P9+P15 Thinking & Security

### Added
- Dashboard thinking pathways, security self-hacking audit, Dream State consolidation

## [0.7.0] — 2026-03-18 — P7+P10+P11+P12+P13+P14

### Added
- Silence-as-signal, predictive pre-computation, operator tempo, self-deception detection
- Temporal self-model, self-improvement gate

## [0.6.0] — 2026-03-17 — P4+P5+P6 Cognitive Alignment

### Added
- Contradiction memory, GEM cognitive alignment, knowledge boundary awareness

## [0.5.0] — 2026-03-17 — Failure Taxonomy + SELAUR

### Added
- 9-class failure taxonomy, SELAUR scoring, metacognitive rule extraction

## [0.4.0] — 2026-03-16 — Agentic Framework Phase 2

### Added
- Specialist router, memory-driven planner, adversary challenge engine
- LangGraph, AutoGen, CrewAI, OpenAgents integration stubs

## [0.3.0] — 2026-03-15 — Quality Hardening

### Added
- Proactive nudges, correction learning, cross-session context, backup service
- Chat UI improvements (markdown, persist, stop, copy)

## [0.2.0] — 2026-03-14 — v7 Core Hardening

### Added
- Policy-as-code, verifier PASS/REPAIR/FAIL_CLOSED, evidence packs
- Circuit breakers, quarantine, rate limits, idempotency

## [0.1.0] — 2026-03-12 — Foundation

### Added
- 22 Docker services, initial compose stack
- Tool-gate, memu-core, LangGraph orchestration, heartbeat, supervisor
- HMAC inter-service auth, audit streams, error budgets
- GitHub Actions CI, flake8 lint, pytest suite
