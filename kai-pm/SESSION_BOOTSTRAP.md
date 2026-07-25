# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

---

## 1) Project one-liner

Kai is a self-sovereign, local-first personal AI system — cooperating Docker services
with tiered memory (vector + graph + archival agent), a conviction/trust loop, and full
process-level failure isolation between hot and cold paths.

---

## 2) Current phase + current focus (25 July 2026)

**Phase: Phase 0 — COMPLETE. Blocked on GPU hardware (RTX 5080) to enter Phase 1.**

All Phase 0 / 0.5 CPU-safe backlog items are shipped and on `main`. Last merged: D92–D100
Intelligence Sprint (2026-07-25) — 9 new capabilities in one push. CPU-safe live: D92 Socratic
Questioning (SocraticQuestioner, pre-GATHER 3-5 decomposition questions, enriched_query wired into
`build_swarm_pipeline()`), D93 Hypothesis Engine (idle-cycle "If X then Y" formation + LLM test +
SUPPORTED/REFUTED/INCONCLUSIVE verdict → CURIOSITY.md, wired into `idle_curiosity_tick()`), D94
Temporal Projection (ForecastFan 4-branch probability fan: base/optimistic/pessimistic/wild_card,
causal-chain input, robust JSON parse with static fallback). D98 Cognitive Fingerprint collecting
InteractionSample records NOW to `/data/cognitive_fingerprint.jsonl`; infers at 90 samples.
GPU-era stubs with fixed interfaces: D95 Dialectical Synthesis (`dialectic.py`), D96 Analogical
Reasoning (`analogy.py`), D97 Concept Blending (`concept_blend.py`), D99 Synthetic Experience
(`synthetic_experience.py`), D100 Transitive Reasoning (`memu-graph/transitive.py`, PageRank +
community detection + shortest-path + rule mining, MIN_EDGES=500). SwarmContext gained
`decomposition_questions: List[str]` + `enriched_query: str`. 9 new FF_* flags (47 total). 4
new test targets (88 total). 75 new tests (~2,829 total). No open PRs.

### What has shipped to `main` (merged PRs, in order)

| PR | What | Key decisions |
|----|------|--------------|
| #77 | Phase 0.5: minimal-stack real spine (ollama+agentic wired); live Docker boot-test | D37 |
| #78 | Default model `qwen2:0.5b` → `qwen2.5:0.5b`; memu-core Postgres extension race fixed | D38, D39 |
| #79 | memu-graph (Cognee/Kuzu) live CI verification | D41–D53 |
| #81 | TurboVec activated as default VECTOR_STORE in dev/CI | D54 |
| #82 | Letta agent memory controller — service, feature flags, agentic 12-way gather | D55 |
| #83 | FF_GRAPH_INGEST=true; P29 CIS Financial Awareness service | D56, D57 |
| #84 | Automation infra, cloud LLM backends, PWA service worker, agentic financial wiring | D58 |
| #85 | C3 LLM retry/backoff, behavioral scoreboard, Finance dashboard tab, PHONE_SETUP.md | D59 |
| #86 | Phase 0 backlog: SOUL.md, Live Canvas D3 v7, Memory Diary, PII auto-redaction, coverage gate | D60–D64 |
| #87 | CI fix: pii_redacted type, chassis httpx mock, financial-awareness sys.modules collision | D65 |
| #88 | Cleanup sprint: merge-gate, redis stub, MAKEFILE_TARGETS, CI isolation fixes, 5-module coverage gate (60%), env-specific skips; TurboVecStore BIGSERIAL race + generate_embedding ordering fix | D71–D78 |
| #89 | COMPOSE_DRIFT fixes D1/D2/D6/D9/D10; README sync; SESSION_BOOTSTRAP + DECISIONS.md D77–D80 housekeeping | D79, D80 |
| #91 | Phase 1 readiness S1–S5: langgraph shim removed, agentic/memu-core route tests, sovereign CI boot, GPU runbook; C4/C10/P1 screen-capture; F4/F6 feature-flag tests; 2 bug fixes | D82, D83 |
| #92 | CI test-isolation: tesseract binary probe, lakefs_client importlib isolation; 30 failures resolved; 2,243 tests passing | D84 |
| #93 | PM housekeeping; S7 shell sandbox (`sandboxes/shell/`); T3 RAMS generator (`scripts/hse_rams.py`) | — |
| #94 | U4 file upload: dashboard text-inject + image OCR via screen-capture; `/api/upload`; drag-and-drop + paste | — |
| #95 | Simplify sprint: unified `_RISK_LEVELS`, explicit 400 on oversized shell input, `raise_for_status()` split, JS closure hoist; sovereign compose `:?` → `:-`; CI disk-cleanup step | D85 |
| #96 | PM docs: D85 + STATUS.md update for PRs #93–95 | — |
| #97 | README metrics sync + SESSION_BOOTSTRAP update to PRs #93–96 | — |
| #98 | Hardening sprint: shell sandbox `SAFE_DIRS` path restriction (11 tests); kill-isolation CI step; Trivy exit-code `'1'` + `ignore-unfixed`; per-module coverage floors (`agentic ≥ 45%`, `memu-core ≥ 60%`); `go_no_go` + `check-docs` early CI gates; restart-persistence smoke test; upload fuzz (14 tests) | D86 |
| #99 | Memory Graph tab (D3 v7 force-directed, category hubs, trust-tier colours, zoom/pan/drag, hover, filter); Whisper audio-service in minimal stack (`172.20.0.15`); `WHISPER_BACKEND=stub`; `/api/audio/transcribe` proxy; MediaRecorder fallback in `toggleVoice()` | D86 |
| #100 | TTS service in minimal stack (`172.20.0.16`, edge-tts, `en-GB-RyanNeural`); `/api/tts/synthesize` proxy; 🔊 speak button on all assistant messages; `speakMsg()` with ObjectURL playback; audio transcribe fuzz suite (13 tests); `make test-audio-transcribe` | D86 |
| #101 | Doc sweep: SESSION_BOOTSTRAP + CHANGELOG + README for PRs #97–#100 + D86 | — |
| #102 | Browser agent (`browser-agent/`, Playwright Chromium, port 8040, `172.20.0.17`): `/navigate`, `/click`, `/type`, `/scrape`, `/screenshot`, `/run`; `/api/browser/*` dashboard proxies; `browse: <url>` chat shortcut; 13 tests. Vision service (`perception/vision/`, OpenCV+DeepFace, port 8023, `172.20.0.18`): face detection + emotion; `/api/vision/{analyze,presence}` proxies; 📷 camera panel (5 s frame sampling, presence+emotion overlay); 12 tests | — |
| #103 | Doc sweep: STATUS, SESSION_BOOTSTRAP, CHANGELOG, README for PR #102 (services 35→37) | — |
| #104 | clipboard-service (8024/.19), files-service (8025/.20), notify-service (8031/.21); browser `/search` + `search:` chat shortcut; 📋 clipboard button; notify pending poll; 52 tests; services 37→40 | — |
| #105 | document-parser service (8032, `172.20.0.22`): PDF, DOCX, XLSX, XLS, PPTX, DXF, DWG, ZIP, CSV, JSON, XML, HTML; `/api/upload` extension-based routing; `_DOC_EXTS` frontend; 22 tests; services 40→41 | — |
| #106 | Doc sweep: STATUS + SESSION_BOOTSTRAP for PR #105 | — |
| #107 | monitor-service (8033, `172.20.0.23`): background rule engine; HTTP/scrape sources; 11 condition ops; notify + TTS actions; interval + cooldown; Monitor tab; RULES_FILE persistence; 34 tests; services 41→42 | — |
| #109 | broker-bridge (8034, `172.20.0.24`): Binance REST wrapper; HMAC-SHA256 signing; spot + futures (USDM) modes via `BINANCE_MODE`; `/ticker`, `/balance`, `/positions`, `/orders`, `/pnl/summary`, `/templates`; Broker tab (status, tickers, balance, positions, orders, Quick Watch, template browser); 20 tests; services 43 | — |
| #111 | CI fixes: flake8 F824 in browser-agent/app.py; stale README metrics synced | — |
| #112 | Sensory expansion: sysmetrics (8035, `172.20.0.25`), screen-watcher (8036, `172.20.0.26`), email-reader (8037, `172.20.0.27`), news-feed (8038, `172.20.0.28`); broker `/depth`, `/stats/24hr`, `/trades`, `/futures/funding`, `/futures/openinterest`; System + Feeds dashboard tabs; 61 tests; services 43→47 | — |
| #114 | — | Sensory expansion wave 2: weather-service (8039, `.29`), docker-watcher (8041, `.30`), airquality-service (8042, `.31`), calendar-service (8043, `.32`); sysmetrics `/temperature` + `/battery`; dashboard weather/AQ/docker/calendar widgets; 63 tests; services 47→51 | — |
| #115 | — | Doc sweep for PR #114: CHANGELOG, SESSION_BOOTSTRAP, STATUS, README badges (services 33→51, tests 2,328→2,567, LOC ~62,099) | — |
| #116 | — | yfinance stocks/forex endpoints on broker-bridge; git-watcher service (8044, `.33`); 19 tests; services 51→52 | — |
| D87 | D87 | Cognitive architecture: `_get_world_context()` (9 sensory services per /chat), `_proactive_observer()` (background anomaly loop → memu-core), skill matching wired into /chat, ghost flag fixes (FF_CONTEXT_ENRICHMENT + FF_PROACTIVE_AGENT), 14-way gather; 17 tests | — |
| D88 | D88 | 8 advanced cognitive mechanisms: M1 anomaly detection with rolling baselines (2σ z-score); M2 `/introspect/capabilities` self-capability map; M3 cross-service correlation (`_correlate_observations()`); M4 world model persistence (JSON `world_state` to memu-core each cycle); M5 sensory learning (pattern history deque, `sensor_pattern` memories); M6 skill-hunter service (port 8045, `.34`); M7 proactive scheduling (calendar + sensors → `proactive_schedule` memories); M8 reactive skill acquisition (capability gap → skill hunter on-demand). 5 new feature flags. 44 tests. Services: 57 → 58 | — |
| D89 | D89 | Cognitive Depth sprint: `agentic/system_fsm.py` (KaiFSM — 5 states, 9 events, 16 transitions, asyncio.Lock singleton); `agentic/cognitive_fsm.py` (CognitiveFSM reasoning pipeline — GATHER→DEBATE→FACT_CHECK→CAUSAL_CHECK→CONVICTION_GATE→PRESENT, HALT/ESCALATE_LOOP/RETHINK with bounded retries, schema-validated `AgentHandoff`, per-swarm `SwarmConfig` for trading/research/skill_forge/default); `agentic/teammates.py` (TeammateDef, markdown-based registry, `GET /teammates`, `POST /chat/teammate/{name}`); `data/teammates/` (Scout/Doctor/Sage/Oracle personas); `agentic/counterfactual.py` (stub: `can_rehearse()→False`, `rehearse()→stub_pending_gpu`); `agentic/curiosity.py` (idle curiosity tick, CURIOSITY.md, GPU-gated); `house-doctor/` service (port 8046, `.35`, 9 differential-diagnosis rules D001–D009, `/diagnose` + `/rules`, writes `medical_report` to memu-core); skill-hunter v0.2 (YAML front-matter provenance, `.meta.json` sidecars, auto-disable at ≥3 errors, `/skill/{name}/health` + `/skill/{name}/error`); tool-gate `POST /gate/autonomy/request` (trust negotiation stub); world model provenance layer (`{value,source,timestamp,confidence}` per field); capability gap logging (`GAP_HUNT_THRESHOLD=3`); emergent ritual discovery (≥7/10 pattern cycles → RITUALS.md); `emotional_context` world model key (predictive empathy foundation); 8 new feature flags; `scripts/test_d89_cognitive_depth.py` (47 tests). Services: 58 → 59 | — |
| D90 | D90 | Swarm Assembly: `agentic/swarm.py` (SwarmContext shared state, TeammateRep reputation tracking, `resolve_conflict()` 5-signal weighted average: evidence 0.30 + causal 0.25 + verdict 0.20 + reputation_vote 0.15 + adversary_mod 0.10); `agentic/swarm_stages.py` (5 real stage function factories: make_gather_stage, make_debate_stage, make_fact_check_stage, make_causal_check_stage, make_conviction_gate_stage + `build_swarm_pipeline()`); `POST /chat/swarm` live endpoint; `GET /swarm/reputation` → per-teammate weights; `FF_SWARM=true` (default); `data/teammate_reputation.json` persists reputation across sessions; 38 tests. | — |
| D91 | D91 | Obsidian Brain: `vault-sync/` service (port 8047, 172.20.0.36) — `parser.py` (NoteData dataclass: filepath, title, frontmatter, wikilinks, tags, SHA256 checksum; python-frontmatter + wikilink regex); `mapper.py` (VaultMapper: filepath↔node-id, thread-safe Lock, persists to .vault-sync/mapping.json); `watcher.py` (FileWatcher: watchdog Observer, _VaultHandler, 2s per-filepath debounce via threading.Timer, thread→asyncio bridge via loop.call_soon_threadsafe); `app.py` (FastAPI: GET /health, POST /ingest manual trigger, POST /export conviction≥9.0 gate + path-traversal block, GET /search, GET /mapping; asyncio ingest+delete queue workers; FF_VAULT_SYNC toggle). 3 new memu-core vault endpoints (POST /memory/vault/ingest, DELETE /memory/vault/{id}, GET /memory/vault/search). Agentic vault proxy (POST /vault/export, GET /vault/search) + FF_VAULT_CONTEXT world-context injection. FF_VAULT_SYNC=True (default) + FF_VAULT_CONTEXT=False (default). 4 Jinja2 templates (daily-note, lesson-learned, kai-inbox, soul-mirror). Dockerfile + compose wiring (172.20.0.36, vault_data volume). ~45 tests. Services: 59 → 60. | — |
| D92–D100 | D92–D100 | Intelligence Sprint: D92 `agentic/questioner.py` — SocraticQuestioner: `decompose(query)→SocraticResult`, 3-5 decomposition questions + enriched_query, LLM-driven with 5-question FALLBACK, `can_question()` checks FF_SOCRATIC; non-fatal `questioner_stage` wired into `build_swarm_pipeline()` (optional kwarg, backwards-compatible); SwarmContext gained `decomposition_questions: List[str]` + `enriched_query: str`; 25 tests. D93 `agentic/hypothesis.py` — HypothesisEngine: idle-cycle "If X then Y" gap formation from memory evidence, LLM tests hypotheses → SUPPORTED/REFUTED/INCONCLUSIVE verdict, appends to CURIOSITY_LOG (`/data/CURIOSITY.md`); wired into `idle_curiosity_tick()` (CPU-safe, before GPU check); MIN_MEMORIES_TO_SCAN=3, MAX_HYPOTHESES_PER_CYCLE=3; 20 tests. D94 `agentic/forecaster.py` — TemporalForecaster: ForecastFan with 4 ScenarioBranch objects (base/optimistic/pessimistic/wild_card), probability-weighted, causal_chains input, `consensus_probability` property, robust JSON extraction, `_FALLBACK_BRANCHES` static fallback; 15 tests. GPU-era stubs (all `can_*()→False` in Phase 0, interfaces frozen): D95 `dialectic.py` (DialecticalTriad: thesis/antithesis/synthesis, `resolution_level`, stub pending dual-model GPU); D96 `analogy.py` (AnalogyMapping + Analogy, `can_find()→False`, stub pending ≥1000 concept graph nodes); D97 `concept_blend.py` (BlendedConcept: emergent_properties/novelty_score/suppressed, `can_blend()→False`, stub pending graph+GPU); D98 `cognitive_fingerprint.py` (InteractionSample + CognitiveFingerprintCollector collecting NOW to `/data/cognitive_fingerprint.jsonl`; `can_infer()→False` until 90 samples; module-level `collector` singleton + `quick_sample()` heuristics); D99 `synthetic_experience.py` (SyntheticScenario + SyntheticExperienceGenerator, `can_generate()→False`, pending GPU dream cycles); D100 `memu-graph/transitive.py` (TransitiveReasoner: PageRank, community detection, shortest_path, mine_rules; `can_reason()→False` until MIN_EDGES_FOR_REASONING=500). 9 new FF_* flags (total: 47). 4 new test targets (total: 88). 75 new tests (total: ~2,829). | — |

### In-flight work

None. All work is on `main`. No open PRs.

---

## 3) Next priorities (in order)

1. **S1–S5 — DONE** (PR #91, D82) — langgraph shim removed, route tests, sovereign CI boot, GPU runbook
2. **S7 — DONE** (PR #93) — shell sandbox service (`sandboxes/shell/`)
3. **T3 — DONE** (PR #93) — HSE RAMS generator (`scripts/hse_rams.py`)
4. **U4 — DONE** (PR #94) — file upload + OCR in dashboard
5. **Simplify sprint — DONE** (PR #95, D85) — quality cleanup across 4 files + CI fixes
6. **Hardening sprint — DONE** (PR #98, D86) — path restriction, CI gates, coverage floors, fuzz tests
7. **Memory Graph tab — DONE** (PR #99, D86) — D3 v7 force-directed, trust-tier colours, hover/filter
8. **Whisper STT — DONE** (PR #99, D86) — audio-service in minimal stack, MediaRecorder fallback
9. **TTS voice synthesis — DONE** (PR #100, D86) — tts-service, edge-tts, 🔊 speak button
10. **Browser agent — DONE** (PR #102) — Playwright Chromium navigation, scrape, click, type
11. **Vision/camera — DONE** (PR #102) — OpenCV face detect + DeepFace emotion, webcam panel
12. **Document parser — DONE** (PR #105) — PDF, Word, Excel, PowerPoint, DXF, DWG, ZIP, all formats
13. **Monitor service — DONE** (PR #107) — background rule engine, HTTP/scrape sources, notify + TTS alerts
14. **Broker bridge — DONE** (PR #109) — Binance REST, spot/futures, Broker tab, Quick Watch → monitor rules
15. **Sensory expansion — DONE** (PR #112) — sysmetrics, screen-watcher, email-reader, news-feed; broker market depth; System + Feeds tabs
16. **Swarm Assembly — DONE** (D90) — real stage functions wired, SwarmContext, reputation, `POST /chat/swarm` live
17. **Obsidian Brain — DONE** (D91) — vault-sync service, SHA256 dedup, conviction gate, memu-core vault API, 4 templates
18. **Intelligence Sprint D92–D100 — DONE** — Socratic Questioning, Hypothesis Engine, Temporal Projection live; D98 Cognitive Fingerprint collecting; D95/D96/D97/D99/D100 GPU-era stubs with fixed interfaces; 9 flags, 88 test targets, ~2,829 tests
19. **GPU hardware arrival** — RTX 5080: execute GPU Day protocol (G1–G8 in GPU_ARRIVAL_RUNBOOK.md), declare Phase 1

Full plan: [`kai-pm/PHASE1_READINESS.md`](PHASE1_READINESS.md)

---

## 4) Blocked items + unlock conditions

| Blocked | Unlock |
|---------|--------|
| Letta live smoke-test | Live Ollama instance |
| `FF_LETTA_TASKS=true` in production | GPU + live Ollama verified |
| Phase 1 — real multi-model routing (`qwen2.5:7b`) | RTX 5080 provisioned |
| Phase 2 — Multi-Specialist Routing | Phase 1 complete |
| Phase 4 — Avatar / Voice / Multimodal | GPU provisioned |
| Phase 5 — Production Hardening & Self-Improvement | Phase 4 complete |

---

## 5) Key architecture facts (don't re-derive these)

- **Memory layers:**
  - `memu-core` — vector store (TurboVec ANN by default in dev/CI; pgvector in sovereign)
  - `memu-graph` — Cognee/Kuzu knowledge graph, port 8061. Fan-out active (`FF_GRAPH_INGEST=true`)
  - `letta-agent` — Letta archival memory controller, port 8062. Gated by `FF_LETTA_TASKS=false` (default)
  - `vault-sync` — Obsidian Brain, port 8047, IP `172.20.0.36`. Watchdog watcher + 2s debounce, SHA256 dedup, conviction ≥9.0 export gate, VaultMapper (filepath↔node-id), 4 Jinja2 templates. `FF_VAULT_SYNC=true` (default), `FF_VAULT_CONTEXT=false` (enable once vault is populated). Memu-core vault endpoints: `POST /memory/vault/ingest`, `DELETE /memory/vault/{id}`, `GET /memory/vault/search`.
  - `financial-awareness` — CIS/VAT/tax arithmetic service, port 8063. Pure Python, no LLM.
- **Perception / output stack (minimal stack):**
  - `audio-service` — Whisper STT (`perception/audio/`), port 8021, IP `172.20.0.15`. `WHISPER_BACKEND=stub` (CI/dev); `local` for real transcription. Dashboard proxy: `POST /api/audio/transcribe`.
  - `tts-service` — edge-tts voice synthesis (`output/tts/`), port 8030, IP `172.20.0.16`. `en-GB-RyanNeural` default voice. Dashboard proxy: `POST /api/tts/synthesize`; returns `audio/mpeg`.
  - `wake-service` — wake-word + intent detection (`perception/wake/`), port 8022, IP `172.20.0.10`.
  - `browser-agent` — Playwright Chromium (`browser-agent/`), port 8040, IP `172.20.0.17`. Headless browser; `/navigate`, `/click`, `/type`, `/scrape`, `/screenshot`, `/run`. Dashboard proxies `/api/browser/*`. Chat shortcut `browse: <url>`.
  - `document-parser` — multi-format text extraction (`document-parser/`), port 8032, IP `172.20.0.22`. PDF (PyMuPDF), DOCX (python-docx), XLSX (openpyxl), XLS (xlrd), PPTX (python-pptx), DXF/DWG (ezdxf + LibreDWG `dwg2dxf`), ZIP (recursive), CSV/JSON/XML/HTML. Dashboard routes `/api/upload` by extension: images→OCR, documents→doc-parser.
  - `monitor-service` — background rule-based alerting (`monitor-service/`), port 8033, IP `172.20.0.23`. Rules: HTTP/scrape source + condition (gt/lt/gte/lte/eq/ne/contains/changed/+%/-%) + actions (notify-service, TTS). Per-rule interval + cooldown. RULES_FILE env for persistence. Monitor tab in dashboard with Add Rule form, live table, alert feed.
  - `broker-bridge` — Binance REST wrapper (`broker-bridge/`), port 8034, IP `172.20.0.24`. `BINANCE_MODE=spot|futures`. HMAC-SHA256 signing for authenticated endpoints. `/ticker`, `/balance`, `/positions`, `/orders`, `/pnl/summary` (futures), `/templates`, `/depth/{symbol}`, `/stats/24hr/{symbol}`, `/trades/{symbol}`, `/futures/funding/{symbol}`, `/futures/openinterest/{symbol}`. Dashboard proxies `/api/broker/*`; `/api/broker/watch` creates monitor rule per position. Credentials via `BINANCE_API_KEY` + `BINANCE_API_SECRET` env vars.
  - `sysmetrics` — system health snapshot (`sysmetrics/`), port 8035, IP `172.20.0.25`. CPU/RAM/disk/network/processes via psutil. `/snapshot`, `/processes`. Dashboard proxies `/api/sysmetrics/*`; System tab shows gauges + process table + screen-watcher controls.
  - `screen-watcher` — periodic screenshot diff + alert (`screen-watcher/`), port 8036, IP `172.20.0.26`. Captures from screen-capture service, MD5-samples for change detection, fires notify+TTS when diff ≥ threshold. `POST /watch/start|stop`. `CHANGE_THRESHOLD` env (default 0.05).
  - `email-reader` — IMAP read-only polling (`email-reader/`), port 8037, IP `172.20.0.27`. Polls every `EMAIL_POLL_INTERVAL_SECONDS` (default 120s). Starts in stub mode when `MAIL_HOST`/`MAIL_USER`/`MAIL_PASS` unset. `/inbox`, `/unread`, `/refresh`. Dashboard proxies `/api/email/*`; Feeds tab.
  - `news-feed` — RSS aggregation + keyword search (`news-feed/`), port 8038, IP `172.20.0.28`. Default feeds: BBC News, NYT Tech, Hacker News. `/feeds` CRUD, `/articles`, `/search`, `/refresh`. `SEED_FEEDS` env (comma-separated URLs). Dashboard proxies `/api/news/*`; Feeds tab.
  - `vision-service` — webcam frame analysis (`perception/vision/`), port 8023, IP `172.20.0.18`. OpenCV haar cascade (face detect) + DeepFace emotion (CPU-safe). Dashboard proxies `/api/vision/{analyze,presence}`. 📷 camera panel in frontend.
- **VECTOR_STORE env var** in `memu-core`: `turbovec` (default dev/CI) → TurboVec; `postgres` → pgvector; else → ephemeral InMemory. Sovereign uses `postgres`.
- **`FF_GRAPH_INGEST=true`** (default in full compose): every memorize/forget fans out to memu-graph. Best-effort — never blocks memu-core.
- **`FF_LETTA_TASKS=false`** (default): when `true`, each `/chat` fires a 30s POST to letta-agent and injects archival context into the system prompt.
- **`FF_FINANCIAL_CONTEXT=true`** (default): keyword-gated CIS/VAT/tax summary injected into agentic context on finance queries.
- **LLM retry**: `LLM_MAX_RETRIES=3`, `LLM_RETRY_BACKOFF=1.0s`, exponential — handles 429/503 and connection errors.
- **Cloud LLM backends**: Groq (`GROQ_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`) available as fallback when env key is set.
- **Process boundaries:** `agentic` (hot: chat/run/checkpoints/skills) vs `agentic-introspect` (cold: dream/evolve/security-audit); `memu-core` (hot) vs `memu-core-introspect` (compress/decay).
- **HMAC**: inter-service signing enforced. Dev secret opt-in only (`HMAC_ALLOW_DEV_SECRET=true`).
- **Model**: `qwen2.5:0.5b` (default). Embedding: `all-MiniLM-L6-v2` (384-dim).
- **Embedding endpoint**: `/api/embed` (not deprecated `/api/embeddings`). Confirmed D47.
- **TurboVecStore startup**: embedding backend (`_embedding_backend` / `generate_embedding`) must be defined before store selection block in `memu-core/app.py` — see D78.
- **Intelligence Sprint D92–D100 (live + stubs):**
  - `agentic/questioner.py` — SocraticQuestioner: `decompose(query)→SocraticResult(questions, enriched_query)`. Wired into `build_swarm_pipeline(questioner=...)` as non-fatal pre-GATHER stage. SwarmContext fields: `decomposition_questions`, `enriched_query`.
  - `agentic/hypothesis.py` — HypothesisEngine: idle-cycle "If X then Y" formation from memories, LLM tests → SUPPORTED/REFUTED/INCONCLUSIVE → appends to `/data/CURIOSITY.md`. Wired into `idle_curiosity_tick()`.
  - `agentic/forecaster.py` — TemporalForecaster → ForecastFan(4×ScenarioBranch: base/optimistic/pessimistic/wild_card). `consensus_probability` property. Causal-chain input. Static fallback if LLM unavailable.
  - `agentic/cognitive_fingerprint.py` — `collector` singleton, `record(InteractionSample)` appending to `/data/cognitive_fingerprint.jsonl` NOW. `quick_sample(query)` helper. `can_infer()→False` until 90 samples.
  - GPU-era stubs (interface frozen, `can_*()→False`): `dialectic.py` (D95), `analogy.py` (D96), `concept_blend.py` (D97), `synthetic_experience.py` (D99), `memu-graph/transitive.py` (D100, MIN_EDGES=500).
- **Tests**: ~2,829 across 134+ files (88 test targets in `make test-core`); per-module floors: `agentic ≥ 45%`, `memu-core ≥ 60%`. `MEMU_ALLOW_FAKE_EMBEDDINGS=true` required for offline runs. `scripts/conftest.py` redis stub required for collection.
- **Coverage**: 5 modules (`common`, `agentic`, `memu-core`, `letta-agent`, `financial-awareness`), 60% gate.

---

## 6) PM operating rules

- **`kai-pm/DECISIONS.md`** is append-only — never edit past entries, supersede with new numbered entry. Last entry: **D100**.
- Reality checks → new file `REALITY_CHECK_<date>.md`, not silent rewrites.
- No drift between docs, status, and delivered code.
- `make sync-docs` after major changes; `make merge-gate` before every PR.

---

## 7) How to resume after a context loss

1. Open this file.
2. Check open PRs: `https://github.com/dainius1234/kai-system/pulls`
3. Read tail of `kai-pm/DECISIONS.md` for the last 3–5 entries.
4. Say: *"Resume — read SESSION_BOOTSTRAP and tell me the next move."*

---

## 8) Pointer index

| File | What |
|------|------|
| `kai-pm/DECISIONS.md` | Append-only decision log (D1–D100) |
| `kai-pm/STATUS.md` | Sprint health + open PRs |
| `kai-pm/CLEANUP_TODO.md` | Cleanup sprint tracker (all items done except §2.1 merge-order decision) |
| `kai-pm/COMPOSE_DRIFT.md` | Docker compose divergence audit (§2.2 shared-block extraction deferred) |
| `kai-pm/MAKEFILE_TARGETS.md` | Full ~110-target catalogue with pass/fail per environment |
| `kai-pm/PHASE1_READINESS.md` | Pre-GPU sprint (S1–S5) + GPU Day protocol (G1–G7) + Phase 1 activation (F1–F6) |
| `kai-pm/LETTA_INTEGRATION_PLAN.md` | Letta integration plan (Steps 1–5 done; Step 0 live-verify pending GPU) |
| `kai-pm/STRATEGIC_PLAN.md` | Canonical 5-phase roadmap |
| `kai-pm/SEQUENCE.md` | Phase sequencing |
| `kai-pm/TECH_WATCH.md` | External tool evaluations |
| `docs/PROJECT_BACKLOG.md` | Living backlog |
| `docs/PHONE_SETUP.md` | PWA phone install guide (Android + iOS) |
| `perception/audio/` | Whisper STT service (`audio-service`, port 8021) |
| `output/tts/` | edge-tts voice synthesis service (`tts-service`, port 8030) |
| `browser-agent/` | Playwright Chromium navigation service (port 8040) |
| `perception/vision/` | Webcam face/emotion analysis service (port 8023) |
| `CHANGELOG.md` | Full semver changelog |
