# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

---

## 1) Project one-liner

Kai is a self-sovereign, local-first personal AI system — cooperating Docker services
with tiered memory (vector + graph + archival agent), a conviction/trust loop, and full
process-level failure isolation between hot and cold paths.

---

## 2) Current phase + current focus (21 July 2026)

**Phase: Phase 0 — Pre-GPU Hardening (Memory layer complete; moving to user-facing features)**

The core memory architecture is fully landed and activated:
- TurboVec ANN vector search (default in dev/CI)
- Cognee/Kuzu knowledge graph (CI-verified, graph fan-out now live)
- Letta agent memory controller (feature-flagged, wired and ready)

Next focus: **P29 CIS Financial Awareness** — UK construction-subcontractor finance service.

### What has shipped to `main` (merged PRs, in order)

| PR | What | Key decisions |
|----|------|--------------|
| #77 | Phase 0.5: minimal-stack real spine (ollama+agentic wired in); live Docker boot-test | D37 |
| #78 | Default model `qwen2:0.5b` → `qwen2.5:0.5b`; memu-core Postgres extension race fixed | D38, D39 |
| #79 | memu-graph (Cognee/Kuzu) live CI verification — real container boot, full ingest→query→forget cycle | D41–D53 |
| #81 | TurboVec activated as default VECTOR_STORE in dev/CI stacks; sovereign.yml pgvector bug fixed | D54 |
| #82 | Letta agent memory controller — `letta-agent/` service, feature flags, agentic 12-way gather, compose wiring | D55 |

### In-flight work

| Branch | What | Status |
|--------|------|--------|
| `claude/project-rework-plan-pgvp35` | FF_GRAPH_INGEST=true (D56) + P29 CIS finance (in progress) | Active |

---

## 3) Next priorities (in order)

1. **P29 CIS Financial Awareness** — UK construction subcontractor finance service. Endpoints: `/finance/cis`, `/finance/invoice`, `/finance/vat`, `/finance/summary`. See `PHASE_0_5_BACKLOG.md` item 3 for full scope.
2. **Automation infrastructure (0a)** — Friday cleanup workflow, weekly report card cron, off-site backup.
3. **Letta live smoke-test** — blocked on reachable Ollama: `ollama show qwen2.5:0.5b --template | grep -i tools`.
4. **GPU hardware arrival** — when RTX 5080 lands: `OLLAMA_MODEL=qwen2.5:7b`, STT/TTS/avatar, `FF_LETTA_TASKS=true`.

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
- **VECTOR_STORE env var** in `memu-core`: `turbovec` (default dev/CI) → TurboVec; `postgres` → pgvector; else → ephemeral InMemory. Sovereign uses `postgres`.
- **`FF_GRAPH_INGEST=true`** (default in full compose): every memorize/forget fans out to memu-graph. Best-effort — never blocks memu-core.
- **`FF_LETTA_TASKS=false`** (default): when `true`, each `/chat` fires a 30s POST to letta-agent and injects archival context into the system prompt.
- **`FF_LETTA_MEMORY_SYNC=false`** (default): when `true`, `memories_updated=true` from letta triggers a sync back to memu-core with `category="letta_archival"`.
- **Process boundaries:** `agentic` (hot) vs `agentic-introspect` (dream/evolve); `memu-core` (hot) vs `memu-core-introspect` (compress/decay).
- **HMAC**: inter-service signing enforced. Dev secret opt-in only.
- **Model**: `qwen2.5:0.5b` (default). Embedding: `all-minilm` (384-dim, separate model).
- **Embedding endpoint**: `/api/embed` (not deprecated `/api/embeddings`). Confirmed D47.

---

## 6) PM operating rules

- **`kai-pm/DECISIONS.md`** is append-only — never edit past entries, supersede with new numbered entry. Last entry: **D56**.
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
| `kai-pm/DECISIONS.md` | Append-only decision log (D1–D56) |
| `kai-pm/LETTA_INTEGRATION_PLAN.md` | Letta integration plan (Steps 1–5 done; Step 0 live-verify pending) |
| `kai-pm/PHASE_0_5_BACKLOG.md` | Phase 0.5 CPU-safe backlog (P29 CIS finance is next) |
| `kai-pm/STRATEGIC_PLAN.md` | Canonical 5-phase roadmap |
| `kai-pm/SEQUENCE.md` | Phase sequencing |
| `kai-pm/TECH_WATCH.md` | External tool evaluations |
| `docs/PROJECT_BACKLOG.md` | Living backlog |
| `CHANGELOG.md` | Full semver changelog |
