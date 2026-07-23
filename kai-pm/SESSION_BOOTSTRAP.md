# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

---

## 1) Project one-liner

Kai is a self-sovereign, local-first personal AI system — cooperating Docker services
with tiered memory (vector + graph + archival agent), a conviction/trust loop, and full
process-level failure isolation between hot and cold paths.

---

## 2) Current phase + current focus (23 July 2026)

**Phase: Phase 0 — COMPLETE. Blocked on GPU hardware (RTX 5080) to enter Phase 1.**

All Phase 0 / 0.5 CPU-safe backlog items are shipped and on `main`. The cleanup sprint
(D71–D78) is fully merged (PR #88, 2026-07-23). No open PRs.

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

### In-flight work

None. All work is on `main`. No open PRs.

---

## 3) Next priorities (in order)

1. **GPU hardware arrival** — RTX 5080: enable `OLLAMA_MODEL=qwen2.5:7b`, STT/TTS/avatar, `FF_LETTA_TASKS=true`, real multi-model routing
2. **Letta live smoke-test** — blocked on reachable Ollama with tool-call support: `ollama show qwen2.5:0.5b --template | grep -i tools`
3. **Phase 1 planning** — once GPU provisioned: multi-model consensus, real STT, full graph quality

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
  - `financial-awareness` — CIS/VAT/tax arithmetic service, port 8063. Pure Python, no LLM.
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
- **Tests**: 1825 collected, 0 errors. `MEMU_ALLOW_FAKE_EMBEDDINGS=true` required for offline runs. `scripts/conftest.py` redis stub required for collection.
- **Coverage**: 5 modules (`common`, `agentic`, `memu-core`, `letta-agent`, `financial-awareness`), 62.67% measured, 60% gate.

---

## 6) PM operating rules

- **`kai-pm/DECISIONS.md`** is append-only — never edit past entries, supersede with new numbered entry. Last entry: **D80**.
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
| `kai-pm/DECISIONS.md` | Append-only decision log (D1–D78) |
| `kai-pm/STATUS.md` | Sprint health + open PRs |
| `kai-pm/CLEANUP_TODO.md` | Cleanup sprint tracker (all items done except §2.1 merge-order decision) |
| `kai-pm/COMPOSE_DRIFT.md` | Docker compose divergence audit (§2.2 shared-block extraction deferred) |
| `kai-pm/MAKEFILE_TARGETS.md` | Full ~110-target catalogue with pass/fail per environment |
| `kai-pm/LETTA_INTEGRATION_PLAN.md` | Letta integration plan (Steps 1–5 done; Step 0 live-verify pending GPU) |
| `kai-pm/STRATEGIC_PLAN.md` | Canonical 5-phase roadmap |
| `kai-pm/SEQUENCE.md` | Phase sequencing |
| `kai-pm/TECH_WATCH.md` | External tool evaluations |
| `docs/PROJECT_BACKLOG.md` | Living backlog |
| `docs/PHONE_SETUP.md` | PWA phone install guide (Android + iOS) |
| `CHANGELOG.md` | Full semver changelog |
