# Kai PM Status Dashboard

- **Last updated (UTC):** 2026-07-24
- **Current phase:** Phase 0 — COMPLETE. Awaiting GPU hardware (RTX 5080) to enter Phase 1.
- **Current focus:** PRs #98–#102 merged. Full perception stack on main: audio-service (STT), tts-service (TTS), browser-agent (Playwright navigation), vision-service (webcam/face/emotion). All CPU-safe pre-GPU work done. Blocked on GPU.

## What's landed on main

| PR | Decisions | What |
|---|---|---|
| #77 | D37 | Phase 0.5 minimal-stack real spine — Ollama + agentic wired, live Docker boot-test |
| #78 | D38, D39 | Default model `qwen2:0.5b` → `qwen2.5:0.5b`; memu-core Postgres extension race fixed |
| #79 | D41–D53 | memu-graph (Cognee/Kuzu) live CI verification |
| #81 | D54 | TurboVec activated as default VECTOR_STORE in dev/CI |
| #82 | D55 | Letta agent memory controller — service, flags, agentic 12-way gather |
| #83 | D56, D57 | FF_GRAPH_INGEST=true; P29 CIS Financial Awareness service |
| #84 | D58 | Automation infra, cloud LLM backends, PWA service worker, agentic financial wiring |
| #85 | D59 | C3 LLM retry/backoff, behavioral scoreboard, Finance dashboard tab, PHONE_SETUP.md |
| #86 | D60–D64 | Phase 0 backlog: SOUL.md, Live Canvas D3 v7, Memory Diary, PII auto-redaction, coverage gate |
| #87 | D65 | CI fix: pii_redacted type, chassis httpx mock, financial-awareness sys.modules collision |
| #88 | D71–D76 | Cleanup sprint: merge-gate, redis stub, MAKEFILE_TARGETS, 10 CI isolation fixes, 5-module coverage gate (60%), 3 env-specific skip fixes; also fixed TurboVecStore BIGSERIAL race + generate_embedding ordering bug |
| #89 | D79, D80 | COMPOSE_DRIFT fixes: sovereign pgvector image + PG_URI env var (D79); full/minimal OLLAMA_MODEL param + embedding model pull + service_healthy conditions (D80); README sync; SESSION_BOOTSTRAP + DECISIONS.md D77/D78 |
| #91 | D82, D83 | Phase 1 readiness: S1–S5 pre-GPU sprint; agentic/app.py 91% (169 tests), memu-core/app.py 65% (230 tests); C4 classify_semantic fallback tests; C10 A/B query logger (common/ab_log.py); P1 screen-capture headless tests (20); F4/F6 feature flag tests; GPU Arrival Runbook; 5 new Makefile targets; 2 bug fixes (fire_at None crash, float/str timestamp) |
| #92 | D84 | CI test-isolation sprint: screen-capture tesseract binary detection (probe binary via `get_tesseract_version()`, not just Python package import); lakefs_client importlib isolation in integration chain test; 30 CI failures resolved; full suite 2243 tests passing |
| #93 | — | PM housekeeping sweep; S7 shell sandbox service (`sandboxes/shell/app.py`, allowlisted read-only commands, 64 KB output cap); T3 RAMS generator (`scripts/hse_rams.py`, Word docx from site_data.csv with risk matrix colouring and sign-off table) |
| #94 | — | U4 file upload: dashboard text-inject + image OCR via screen-capture service; `/api/upload` endpoint; drag-and-drop + paste in frontend |
| #95 | D85 | Simplify sprint: unified `_RISK_LEVELS` table in hse_rams; explicit 400 on oversized shell commands; `raise_for_status()` split error handling in dashboard upload; `prefix` closure in JS; sovereign compose `:?` → `:-` CI fix; runner disk-cleanup step before heavy Docker builds |
| #98 | D86 | Hardening sprint: shell sandbox `SAFE_DIRS` path restriction (11 tests); kill-isolation CI step; Trivy exit-code `'1'` + `ignore-unfixed`; per-module coverage floors (agentic ≥45%, memu-core ≥60%); `go_no_go` + `check-docs` as early CI gates; restart-persistence smoke test; upload endpoint security fuzz (14 tests) |
| #99 | D86 | Memory Graph tab (D3 v7 force-directed, `/api/memory/graph-data`, category hubs, trust-tier colours, zoom/pan/drag, hover tooltip, detail card, filter); Whisper audio-service in minimal stack (172.20.0.15, WHISPER_BACKEND=stub); TTS service in minimal stack (172.20.0.16, edge-tts); `/api/audio/transcribe` + `/api/tts/synthesize` dashboard proxies; `toggleVoice()` MediaRecorder fallback; 🔊 speak button on all assistant messages |
| #101 | — | Doc sweep: SESSION_BOOTSTRAP + CHANGELOG + README updated for PRs #97–#100 + D86 (milestones 32→35, services 33→35, tests 2,279→2,303) |
| #102 | — | Browser agent (`browser-agent/`, Playwright, port 8040, 172.20.0.17): `/navigate`, `/click`, `/type`, `/scrape`, `/screenshot`, `/run`, dashboard proxies, `browse:` chat shortcut, 13 tests. Vision service (`perception/vision/`, OpenCV+DeepFace, port 8023, 172.20.0.18): face detection, emotion, presence, 📷 camera panel in dashboard, 5 s frame sampling, 12 tests. 25 new tests total. |
| #103 | — | Doc sweep: STATUS, SESSION_BOOTSTRAP, CHANGELOG, README for PR #102 (services 35→37) |
| #104 | — | clipboard-service (8024/.19), files-service (8025/.20), notify-service (8031/.21), browser `/search`; `search:` chat shortcut; 📋 clipboard button; 52 tests; services 37→40 |

## Open PRs

None.

## Blocked items (GPU)

- Phase 1 — Local LLM Integration (`OLLAMA_MODEL=qwen2.5:7b`, real multi-model routing)
- Phase 2 — Multi-Specialist Routing
- Phase 4 — Avatar / Voice / Multimodal
- Phase 5 — Production Hardening & Self-Improvement

Unlock condition: RTX 5080 procurement + provisioning + validation.

## Sprint health signals (2026-07-24)

- Weekly CI: behavioral scoreboard + go/no-go + fast pytest subset (weekly-report-card.yml)
- Friday cleanup: lint, pip-audit, stale-branch hygiene (friday-cleanup.yml)
- Stale branches: 0

## Source of truth pointers

- Resume layer: [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md)
- Decision log: [`DECISIONS.md`](DECISIONS.md) (last entry: D85)
- Latest reality check: [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md)
