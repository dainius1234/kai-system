# Letta Integration Plan

> **Status:** Scoped, not yet implemented. Next step: live smoke-test (blocked on reachable Ollama instance).
> **Decisions:** D33 (spike), D34 (tool-call template sourced), D38 (model swap unblocking prerequisite)

---

## What Letta is (in one paragraph)

Letta (formerly MemGPT) is a local, self-hostable agent framework that gives an LLM
a structured, tiered memory system: in-context memory (always present in the prompt),
external memory (retrieved via search on each turn), and archival memory (long-term,
compressed, bulk-searchable). It manages its own SQLite or Postgres-backed memory store
and exposes a REST API for creating/running agents and inspecting their memory state.
Pinned version for this project: `letta==0.16.8` (confirmed clean install on Python
3.11.15 in D33 spike; well past the buggy 0.7.21-0.7.29 Ollama range from issues #2388/#2668).

---

## Why integrate it

Kai's existing memory architecture (memu-core vector search + memu-graph knowledge graph)
is strong for *retrieval* — finding what was said/known before — but weaker for *agent
memory management*: deciding what to remember, when to compress, how to maintain a
coherent in-context representation of the conversation state across many turns. Letta's
memory controller handles exactly this, and its architecture is designed to layer on top
of an existing LLM backend (Ollama in our case) without replacing it.

The integration is additive, not a replacement: LangGraph in `agentic/` remains the
routing/planning/conviction engine. Letta adds a memory-management controller that keeps
context coherent across long sessions.

---

## Pre-condition (still needed)

**Confirm `qwen2.5:0.5b` has `"tools"` in its Ollama chat template:**

```bash
# On any machine with Ollama running:
ollama pull qwen2.5:0.5b
ollama show qwen2.5:0.5b --template | grep -i "tools\|tool_call"
```

- **If it matches** → Letta's `OllamaProvider.list_llm_models_async()` will auto-discover it. Proceed.
- **If it doesn't match** → Hand-construct `LLMConfig` directly (bypasses discovery; still works, slightly more config). Proceed anyway.

This is the only remaining unknown. The D33/D34 research established that `qwen2.5:0.5b`
*very likely* has the template text (Qwen2.5 series is where Qwen scopes native tool-call
support), but it has not been confirmed against a live Ollama instance.

---

## Integration architecture

```
┌─────────────────────────────────────────┐
│  agentic/ (LangGraph)                   │
│  chat → planner → adversary → conviction│
│                     │                   │
│         delegate long-running tasks     │
│                     ▼                   │
│  ┌──────────────────────────────────┐   │
│  │  letta-agent/ (new service)      │   │
│  │  POST /agent/run  {task, ctx}    │   │
│  │  GET  /agent/memory/export       │   │
│  │  Letta agent instance            │   │
│  │  in-ctx + external + archival    │   │
│  │  SQLite by default (no new dep)  │   │
│  └──────────────────────────────────┘   │
│                     │                   │
│         sync memories back              │
│                     ▼                   │
│  memu-core (vector) + memu-graph (KG)   │
└─────────────────────────────────────────┘
```

**What `letta-agent/` is NOT:**
- Not a replacement for LangGraph
- Not a replacement for memu-core
- Not used for every request — only for tasks that benefit from multi-turn agent memory management (research tasks, multi-day planning, long financial queries for kai-advisor)

---

## Implementation steps

### Step 0 — Verify (unblocked by GPU, blocked only on live Ollama)
```bash
ollama show qwen2.5:0.5b --template | grep -i tools
```
Document result as D53.

### Step 1 — `letta-agent/` service skeleton
```
letta-agent/
  app.py           # FastAPI wrapper: /health, /agent/run, /agent/memory/export
  requirements.txt # letta==0.16.8, fastapi, uvicorn
  Dockerfile
```

`app.py` init: create a Letta client pointing at Ollama (`OLLAMA_BASE_URL=http://ollama:11434`),
create a default agent with `qwen2.5:0.5b`. Expose:
- `POST /agent/run` — accept `{task: str, context: dict}`, run the Letta agent, return `{response: str, memories_updated: bool}`
- `GET /agent/memory/export` — return the agent's current archival memory as a list of strings (for memu-core sync)
- `GET /health`

### Step 2 — Wire into `agentic/app.py`
Add feature flag `FF_LETTA_TASKS=false`. When `true` and the task classifier marks a
request as `long_running` or `research`, delegate to letta-agent via
`resilient_call(LETTA_URL + "/agent/run", ...)` instead of running it through the
LangGraph conviction loop directly.

### Step 3 — Memory sync
After each `/agent/run`, if `memories_updated=true`, call `/agent/memory/export` and
POST each exported memory to `memu-core /memory/memorize` with `category="letta_archival"`.
This keeps the memu-core retrieval index aware of what Letta learned. Feature-flagged
under `FF_LETTA_MEMORY_SYNC=false` initially.

### Step 4 — Add to compose stacks
- `docker-compose.full.yml`: add `letta-agent` service on port 8062, depends on `ollama` and `memu-core`
- `docker-compose.minimal.yml`: do NOT add (minimal stack is for the spine only)
- `.env.example`: document `LETTA_URL`, `FF_LETTA_TASKS`, `FF_LETTA_MEMORY_SYNC`

### Step 5 — Tests
- `scripts/test_letta_agent.py`: unit tests for the FastAPI wrapper (mocked Letta client)
- `make test-letta` target wired into `make test-core`

---

## What NOT to do

- Do not replace LangGraph with Letta — LangGraph is load-bearing in `agentic/` and has
  the conviction/adversary/tree-search pipeline baked in. These are complementary, not competing.
- Do not add Letta to the minimal stack — it's a full-stack enhancement.
- Do not set Letta's storage to Postgres by default — SQLite is fine for the agent's own
  working memory; memu-core is the authoritative long-term store.
- Do not let Letta write directly to memu-core — sync through the export endpoint only,
  so the memory controller stays auditable and reversible.

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `qwen2.5:0.5b` doesn't have `"tools"` template | Low (sourced evidence says it does) | Hand-construct `LLMConfig`, bypass discovery |
| Letta's Ollama integration regresses again | Medium (known history in 0.7.x) | Pin `letta==0.16.8`, test on upgrade |
| Memory sync creates noise in memu-core | Medium | `category="letta_archival"` keeps it filterable; `FF_LETTA_MEMORY_SYNC=false` default |
| Adds latency to `/chat` for flagged tasks | Certain (Letta adds a round-trip) | Feature-flagged; only fires for explicitly long-running task class |

---

## Estimated scope

| Phase | Work | Decision |
|-------|------|---------|
| Verify (Step 0) | 1 shell command on live Ollama | D53 |
| letta-agent/ skeleton (Steps 1–2) | ~150 lines Python + Dockerfile | D54 |
| Memory sync (Step 3) | ~50 lines in agentic/app.py | D55 |
| Compose wiring + tests (Steps 4–5) | ~100 lines | D55 |

Total: ~1 session of focused work after Step 0 is confirmed.
