# `agentic/app.py` Responsibility Map

> Prep doc for Cleanup Sprint Week 2.1. This is a map of the current file so future split PRs can stay mechanical and low-risk.

## Current truth

- `agentic/app.py` is still the service god-object.
- The first low-risk split target already landed: prompt definitions and SOUL/AGENTS loading live in `/tmp/workspace/dainius1234/kai-system/agentic/prompts.py`.
- `agentic/app.py` now consumes that module and re-exports the legacy globals needed by older imports and tests.

## Responsibility slices

| Area | Approx. location | What lives there today | Suggested destination |
|---|---|---|---|
| Runtime bootstrap + breakers + config | top of file through middleware | FastAPI app setup, breaker restore/persist, env wiring, token-budget config | `runtime.py` / `state.py` |
| Context-budget helpers | `_estimate_tokens`, `_trim_context` | prompt-size trimming and token estimation | `context_budget.py` |
| Session + auto-memorize helpers | `_append_session_turn`, `_fetch_session_context`, `_auto_memorize` | memu-core integration and write-back helpers | `memory_io.py` |
| Ops/health endpoints | `/health`, `/recover`, `/metrics`, `/queue/stats`, `/models`, `/logs` | service status and operator observability | `routes_ops.py` |
| SOUL / AGENTS endpoints | `/soul`, `/agents-registry` | editable identity documents backed by `agentic/prompts.py` | `routes_identity.py` |
| Skills endpoints | `/skills*` | skill loading, scan, unload, prune | `routes_skills.py` |
| Chat context assembly | `_get_mode` through `_preclassify_wake_intent` | 10-way context fetch, wake intent preclassification, operator model fetches | `chat_context.py` |
| `/chat` stream | `chat_stream` | main streaming conversation path, planner + memory + LLM assembly | `routes_chat.py` |
| `/run` graph flow | `run_graph` | planner/adversary/conviction/tool-gate orchestration path | `routes_run.py` |
| Episode + dream + checkpoint routes | `/episodes/recall`, `/dream`, `/checkpoint*` | recall, consolidation, time-travel/debugging | `routes_memory_ops.py` |
| Evolver + security routes | `/evolve/*`, `/security/audit` | offline improvement and self-audit entry points | `routes_meta.py` |

## Dependency pressure points

- `chat_stream` and `run_graph` both pull from the same global runtime state, breakers, prompt catalog, memu helpers, and router/planner imports.
- The safest splits are the route groups that do not own the main streaming loop:
  1. SOUL / AGENTS endpoints
  2. Skills endpoints
  3. Metrics / queue / model / log endpoints
  4. Checkpoint + evolve + security routes
- The riskiest split is the `/chat` path because it mixes context collection, prompt assembly, streaming, memory writes, and alert side effects.

## Recommended next PR sequence

1. **Already done:** prompts extraction to `agentic/prompts.py`.
2. **Next safest leaf:** move SOUL/AGENTS + skills + metrics/models/logs endpoints out of `agentic/app.py`.
3. Split checkpoint / evolve / security routes.
4. Extract chat-context helper functions.
5. Split `/chat` and `/run` only after the helper layers above are isolated.

## Guardrails for follow-up splits

- Keep behavior-preserving moves only.
- Keep old imports working while compatibility shims are still needed.
- Re-run `make go_no_go` after every split touching `agentic/app.py`.
- Treat `agentic/prompts.py` as the precedent: move pure data / pure leaf code first, then route groups, then orchestration.
