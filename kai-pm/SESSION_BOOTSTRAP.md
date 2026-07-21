# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

---

## 1) Project one-liner

Kai is a self-sovereign, local-first personal AI system — cooperating Docker services
with tiered memory (vector + graph + archival agent), a conviction/trust loop, and full
process-level failure isolation between hot and cold paths.

---

## 2) Current phase + current focus (21 July 2026)

**Phase: Phase 0.5 — Complete (all CPU-safe backlog items shipped)**

Phase 0.5 is fully merged. The core memory architecture is live:
- TurboVec ANN vector search (default in dev/CI)
- Cognee/Kuzu knowledge graph (CI-verified, graph fan-out live)
- Letta agent memory controller (feature-flagged, wired and ready)
- P29 CIS Financial Awareness service (port 8063, full arithmetic, no LLM dependency)
- Automation infra: Friday cleanup + weekly report-card GitHub Actions
- Cloud LLM fallback: Groq + OpenRouter support in `common/llm.py`
- LLM retry/backoff: exponential back-off on 429/503 in `_live_query()`
- PWA service worker + phone install guide
- Finance dashboard tab in Kai UI (CIS stat cards, VAT/tax breakdown, log payments)
- Behavioral scoreboard: weekly LLM quality advisory in CI

Next focus: **GPU hardware arrival** — when RTX 5080 lands: `OLLAMA_MODEL=qwen2.5:7b`,
STT/TTS/avatar, `FF_LETTA_TASKS=true`, real multi-model consensus.

### What has shipped to `main` (merged PRs, in order)

| PR | What | Key decisions |
|----|------|--------------|
| #77 | Phase 0.5: minimal-stack real spine (ollama+agentic wired in); live Docker boot-test | D37 |
| #78 | Default model `qwen2:0.5b` → `qwen2.5:0.5b`; memu-core Postgres extension race fixed | D38, D39 |
| #79 | memu-graph (Cognee/Kuzu) live CI verification | D41–D53 |
| #81 | TurboVec activated as default VECTOR_STORE in dev/CI | D54 |
| #82 | Letta agent memory controller — service, feature flags, agentic 12-way gather | D55 |
| #83 | FF_GRAPH_INGEST=true (D56); P29 CIS Financial Awareness service (D57) | D56, D57 |
| #84 | Automation infra, cloud LLM backends, PWA service worker, agentic financial wiring | D58 |

### In-flight work

| Branch | What | Status |
|--------|------|--------|
| `claude/project-rework-plan-pgvp35` | C3 retry, scoreboard, Finance tab, PHONE_SETUP.md | Pending merge (D59) |

---

## 3) Next priorities (in order)

1. **Merge D59 batch** — C3 retry, behavioral scoreboard, Finance tab, PHONE_SETUP.md
2. **GPU hardware arrival** — RTX 5080: enable `OLLAMA_MODEL=qwen2.5:7b`, STT/TTS/avatar, `FF_LETTA_TASKS=true`
3. **Letta live smoke-test** — blocked on reachable Ollama: `ollama show qwen2.5:0.5b --template | grep -i tools`
4. **Phase 1 planning** — once GPU is provisioned: multi-model consensus, real STT, full graph quality

---

## 4) Blocked items + unlock conditions

| Blocked | Unlock |
|---------|--------|
| Letta live smoke-test (Step 0) | Live Ollama instance |
| `FF_LETTA_TASKS=true` in production | GPU + live Ollama verified |
| GPU-dependent phases (real STT, multi-model consensus, full graph quality) | RTX 5080 procurement + provisioning |

---

## 5) Key architecture facts (don't re-derive these)

- **Memory layers:**
  - `memu-core` — vector store (TurboVec ANN by default in dev/CI; pgvector in sovereign)
  - `memu-graph` — Cognee/Kuzu knowledge graph, port 8061. Fan-out from memu-core active (`FF_GRAPH_INGEST=true`)
  - `letta-agent` — Letta archival memory controller, port 8062. Gated by `FF_LETTA_TASKS=false` (default)
  - `financial-awareness` — CIS/VAT/tax arithmetic service, port 8063. Pure Python, no LLM.
- **VECTOR_STORE env var** in `memu-core`: `turbovec` (default dev/CI) → TurboVec; `postgres` → pgvector; else → ephemeral InMemory. Sovereign uses `postgres`.
- **`FF_GRAPH_INGEST=true`** (default in full compose): every memorize/forget fans out to memu-graph. Best-effort — never blocks memu-core.
- **`FF_LETTA_TASKS=false`** (default): when `true`, each `/chat` fires a 30s POST to letta-agent and injects archival context into the system prompt.
- **`FF_FINANCIAL_CONTEXT=true`** (default): keyword-gated CIS/VAT/tax summary injected into agentic context on finance queries.
- **LLM retry**: `LLM_MAX_RETRIES=3`, `LLM_RETRY_BACKOFF=1.0s`, exponential — handles 429/503 and connection errors.
- **Cloud LLM backends**: Groq (`GROQ_API_KEY`) and OpenRouter (`OPENROUTER_API_KEY`) available as fallback when env key is set.
- **Process boundaries:** `agentic` (hot) vs `agentic-introspect` (dream/evolve); `memu-core` (hot) vs `memu-core-introspect` (compress/decay).
- **HMAC**: inter-service signing enforced. Dev secret opt-in only.
- **Model**: `qwen2.5:0.5b` (default). Embedding: `all-minilm` (384-dim, separate model).
- **Embedding endpoint**: `/api/embed` (not deprecated `/api/embeddings`). Confirmed D47.

---

## 6) PM operating rules

- **`kai-pm/DECISIONS.md`** is append-only — never edit past entries, supersede with new numbered entry. Last entry: **D59**.
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
| `kai-pm/DECISIONS.md` | Append-only decision log (D1–D59) |
| `kai-pm/LETTA_INTEGRATION_PLAN.md` | Letta integration plan (Steps 1–5 done; Step 0 live-verify pending) |
| `kai-pm/PHASE_0_5_BACKLOG.md` | Phase 0.5 CPU-safe backlog (all items shipped) |
| `kai-pm/STRATEGIC_PLAN.md` | Canonical 5-phase roadmap |
| `kai-pm/SEQUENCE.md` | Phase sequencing |
| `kai-pm/TECH_WATCH.md` | External tool evaluations |
| `docs/PROJECT_BACKLOG.md` | Living backlog |
| `docs/PHONE_SETUP.md` | PWA phone install guide (Android + iOS) |
| `CHANGELOG.md` | Full semver changelog |
