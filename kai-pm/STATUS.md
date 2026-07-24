# Kai PM Status Dashboard

- **Last updated (UTC):** 2026-07-24
- **Current phase:** Phase 0 — COMPLETE. Awaiting GPU hardware (RTX 5080) to enter Phase 1.
- **Current focus:** PR #91 merged. agentic/app.py 91%, memu-core/app.py 65%, C4/C10/P1 done. Blocked on GPU.

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

## Open PRs

None.

## Blocked items (GPU)

- Phase 1 — Local LLM Integration (`OLLAMA_MODEL=qwen2.5:7b`, real multi-model routing)
- Phase 2 — Multi-Specialist Routing
- Phase 4 — Avatar / Voice / Multimodal
- Phase 5 — Production Hardening & Self-Improvement

Unlock condition: RTX 5080 procurement + provisioning + validation.

## Sprint health signals (2026-07-21)

- Weekly CI: behavioral scoreboard + go/no-go + fast pytest subset (weekly-report-card.yml)
- Friday cleanup: lint, pip-audit, stale-branch hygiene (friday-cleanup.yml)
- Stale branches: 0

## Source of truth pointers

- Resume layer: [`SESSION_BOOTSTRAP.md`](SESSION_BOOTSTRAP.md)
- Decision log: [`DECISIONS.md`](DECISIONS.md) (last entry: D82)
- Latest reality check: [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md)
