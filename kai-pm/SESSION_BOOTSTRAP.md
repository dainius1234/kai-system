# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

---

## 1) Project one-liner

Kai is a self-sovereign, local-first personal AI system — cooperating Docker services
with tiered memory (vector + graph), a conviction/trust loop, and full process-level
failure isolation between hot and cold paths.

---

## 2) Current phase + current focus (21 July 2026)

**Phase: Memory Architecture** — the core spine is live and battle-tested; the
focus is now deepening the memory layer (graph memory CI-verified, TurboVec
activated, Letta integration next) before the GPU hardware arrives.

### What has shipped to `main` (merged PRs, in order)

| PR | What | Key decisions |
|----|------|--------------|
| #77 | Phase 0.5: minimal-stack real spine (ollama+agentic wired in); live Docker boot-test confirmed in CI | D37 |
| #78 | Default model `qwen2:0.5b` → `qwen2.5:0.5b`; memu-core Postgres extension race fixed | D38, D39 |

### What is in flight (open branches/PRs)

| Branch | PR | What | Status |
|--------|----|------|--------|
| `claude/graph-live-verify` | #79 | memu-graph (Cognee/Kuzu) live CI verification — real container boot, full ingest→query→forget cycle | CI running (D52 fix in flight: `COGNEE_SKIP_CONNECTION_TEST=true`) |
| `claude/project-rework-plan-pgvp35` | (no PR yet) | TurboVec activated as default VECTOR_STORE; README refresh; D40 | Pushed, awaiting PR |

---

## 3) Next priorities (in order)

1. **Merge PR #79** once CI goes green — read the actual job log (`mcp__github__get_job_logs`), confirm ingest/query/forget cycle passes, then merge.
2. **Merge `claude/project-rework-plan-pgvp35`** — TurboVec + README. Open a PR after #79 is closed to avoid compose-file conflicts.
3. **Letta integration** — see `kai-pm/LETTA_INTEGRATION_PLAN.md` for the full scoped plan. First concrete step: confirm `qwen2.5:0.5b` has `"tools"` in its Ollama chat template on a live instance (`ollama show qwen2.5:0.5b --template | grep -i tools`).
4. **GPU hardware arrival** — when the RTX 5080 lands: `OLLAMA_MODEL=qwen2.5:7b`, wire STT/TTS/avatar, enable `FF_GRAPH_INGEST=true` in production, run multi-model consensus.

---

## 4) Blocked items + unlock conditions

| Blocked | Unlock |
|---------|--------|
| Letta live smoke-test | Live Ollama instance (not reachable in this sandbox) |
| GPU-dependent phases (real STT, multi-model consensus, full Cognee graph quality) | RTX 5080 procurement + provisioning |
| memu-core P17-P22 personality engine split | Backing-store rework first (currently 5-min Redis lag in-process) |

---

## 5) Key architecture facts (don't re-derive these)

- **VECTOR_STORE env var** in `memu-core`: `turbovec` (default dev/CI) → TurboVec ANN index; `postgres` → pgvector extension; anything else → ephemeral InMemoryVectorStore. Sovereign uses `postgres`.
- **`FF_GRAPH_INGEST`**: memu-core's write-side fan-out to memu-graph. Default `false`. Safe to flip to `true` when memu-graph is healthy.
- **Process boundaries already in place**: `agentic` (hot chat/run) vs `agentic-introspect` (dream/evolve/security-audit); `memu-core` (hot memorize/retrieve) vs `memu-core-introspect` (compress/decay/quarantine etc.).
- **HMAC**: inter-service signing enforced. Dev secret (`HMAC_ALLOW_DEV_SECRET=true`) is explicit opt-in only.
- **Model registry** (`common/model_registry.py`): `qwen2.5:0.5b` entry present (context 32768, supports_json True). Active default: `qwen2.5:0.5b`.
- **memu-graph** runs on port 8061. Cognee version: `1.1.3`. Embedding model: `all-minilm` (384-dim, separate from the chat model).

---

## 6) PM operating rules

- **`kai-pm/DECISIONS.md`** is append-only — never edit past entries, supersede with a new numbered entry. Last entry: **D52**.
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
| `kai-pm/DECISIONS.md` | Append-only decision log (D1–D52) |
| `kai-pm/LETTA_INTEGRATION_PLAN.md` | Letta scoped integration plan (next priority) |
| `kai-pm/STRATEGIC_PLAN.md` | Canonical 5-phase roadmap |
| `kai-pm/TECH_WATCH.md` | External tool evaluations (Letta, TurboVec, Cognee, etc.) |
| `kai-pm/SHOPPING_LIST_PLAN.md` | Tool-to-phase mapping with architecture diagrams |
| `kai-pm/SEQUENCE.md` | Phase sequencing |
| `docs/PROJECT_BACKLOG.md` | Living backlog |
| `docs/known_issues.md` | Gotchas, environment quirks |
| `CHANGELOG.md` | Full semver changelog |
