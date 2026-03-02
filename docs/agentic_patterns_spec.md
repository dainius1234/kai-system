# Agentic Patterns Spec — kai-system Phase 2

> Version: 1.0 — 2 Mar 2026
> Status: IN PROGRESS
> Author: Copilot (architect session)

## Executive Summary

Kai-system already **is** a multi-agent system — each microservice (Tool-Gate,
Verifier, Fusion-Engine, Memu-Core, Supervisor) acts as a specialised agent
with authority boundaries and structured communication. The goal of Phase 2 is
NOT to bolt on another framework but to add **intelligent wiring** between the
services we already have.

Three patterns, in priority order:

| # | Pattern | Complexity | Value | Status |
|---|---------|-----------|-------|--------|
| 1 | Specialist Router | Low | High — reduces LLM waste | **Building** |
| 2 | Memory-Driven Planner | Medium | High — leverages Memu-Core | **Building** |
| 3 | Proposer-Adversary Loop | Medium-High | Medium — safer exec | Designed |

---

## Architecture Principles

1. **No LLM where vector search or rules suffice.**
   Most queries don't need an LLM. Memory recall = vector search. Tax = rules.
   Only genuine reasoning tasks should burn LLM tokens.

2. **Every "agent" is a LangGraph node backed by a real service.**
   No prompt-chain agents. Each node calls a service with real policy, auth,
   circuit breakers. This is what makes Kai different from generic agent demos.

3. **Conviction gating is non-negotiable.**
   Nothing executes below MIN_CONVICTION. The planner must produce a plan that
   passes the 5-signal score before Executor touches it.

4. **Memory-first: check history before acting.**
   Before any planning or LLM call, the system queries Memu-Core for similar
   past interactions. If the same question was asked before, reuse the outcome.
   If a similar action failed before, adjust the plan.

5. **Operator remains in the loop.**
   Dashboard shows which route was chosen and why. Operator can override the
   router, force a specialist, or block execution.

---

## Pattern 1: Specialist Router

### Problem
Currently, `infer_specialist_fallback()` uses naive keyword matching to pick a
specialist. The `/chat` endpoint always routes to `_DEFAULT_SPECIALIST` (Ollama).
Many queries don't need an LLM at all.

### Design

```
User request → classify(input) → Route
                                    ├── MEMORY_RECALL → Memu-Core /memory/retrieve (no LLM)
                                    ├── TAX_ADVISORY  → kai-advisor (no LLM)
                                    ├── FACT_CHECK    → Verifier pipeline (no LLM)
                                    ├── EXECUTE_ACTION→ Tool-Gate → Executor (LLM for plan only)
                                    ├── GENERAL_CHAT  → Ollama (LLM)
                                    └── MULTI_SIGNAL  → Fusion-Engine (multi-LLM consensus)
```

### Route Definitions

| Route | Trigger Keywords/Patterns | Target Service | LLM Cost |
|-------|--------------------------|----------------|----------|
| `MEMORY_RECALL` | "remember", "last time", "what did I", "find my notes", "search for" | Memu-Core `/memory/retrieve` + `/memory/search-by-category` | **Zero** |
| `TAX_ADVISORY` | "tax", "VAT", "MTD", "self-employment", "HMRC", "expense", "invoice" | kai-advisor via `common.self_emp_advisor` | **Zero** |
| `FACT_CHECK` | "is it true", "verify", "check if", "fact check" | Verifier `/verify` | **Zero** |
| `PROACTIVE_REVIEW` | "what should I know", "any reminders", "what's pending" | Memu-Core `/memory/proactive` | **Zero** |
| `EXECUTE_ACTION` | "run", "execute", "deploy", "build", "create file" | Tool-Gate + Executor (needs plan + conviction) | **Low** |
| `REFLECT` | "what have I been working on", "summarize my week", "consolidate" | Memu-Core `/memory/reflect` | **Zero** |
| `GENERAL_CHAT` | Everything else, opinions, creative, conversation | Ollama via LLMRouter | **Normal** |
| `MULTI_SIGNAL` | "compare", "get multiple opinions", "consensus" | Fusion-Engine `/fuse` | **High** |

### Implementation

New module: `langgraph/router.py`

```python
class RouteDecision:
    route: str           # one of the route names above
    confidence: float    # 0-1 how sure we are about the classification
    reason: str          # human-readable explanation
    bypass_llm: bool     # True if this route doesn't need LLM

def classify(user_input: str, session_context: dict) -> RouteDecision:
    """Rule-based classifier with keyword + pattern matching.
    
    NOT an LLM call — this must be instant and deterministic.
    Falls back to GENERAL_CHAT if no strong signal.
    """
```

The `/chat` endpoint in `app.py` will call `classify()` first, then dispatch
to the appropriate handler. Each handler returns a `ChatResponse` that the
streaming SSE pipeline can send to the client.

### Token Savings Estimate
Based on typical usage, ~40-60% of queries are memory recall, tax, or proactive
reviews — all zero-LLM-cost routes. This saves significant Ollama processing time
and context window pollution.

---

## Pattern 2: Memory-Driven Planner

### Problem
The `/run` endpoint builds a plan *then* fetches memory context to score
conviction. It never checks: "Have I seen this exact pattern before? Did it
succeed or fail?"

### Design

```
User request
    ↓
[1] Episode Lookup — query saver.recall() for similar past inputs
    ↓
[2] Memory Retrieval — Memu-Core /memory/retrieve for context
    ↓
[3] Past Outcome Analysis
    ├── Similar past SUCCESS (score > 7) → reuse plan, boost conviction
    ├── Similar past FAILURE (score < 4) → warn, adjust, or block
    └── No history → proceed normally
    ↓
[4] Plan Construction — build_plan() with enriched context
    ↓
[5] Conviction Gate — score_conviction() with history bonus/penalty
    ↓
[6] Execute or Rethink
```

### Implementation

New module: `langgraph/planner.py`

```python
class PlanContext:
    """Enriched context for plan construction."""
    user_input: str
    session_id: str
    memory_chunks: list      # from Memu-Core
    episode_history: list    # from episode saver
    past_outcomes: list      # filtered similar past interactions
    proactive_nudges: list   # relevant reminders
    correction_memories: list # past corrections for similar queries

class PlanDecision:
    plan: dict
    conviction: float
    history_influence: str   # "boosted", "penalised", "neutral"
    reuse_episode_id: str | None  # if reusing a past plan

async def memory_driven_plan(user_input: str, session_id: str) -> PlanDecision:
    """Build a plan that learns from history."""
```

### History Matching
Uses a combination of:
- **Keyword overlap** between current input and past episode inputs
- **Vector similarity** via Memu-Core retrieve (semantic match)
- **Category match** using Memu-Core's category classification

### Conviction Modifiers
- Past success with similar input: +1.0 to conviction
- Past failure with similar input: -1.5 to conviction (and warning)
- Past correction memory exists: -1.0 and inject correction into plan
- Proactive nudge relevant: +0.5 (the system already flagged this)

---

## Pattern 3: Proposer-Adversary Loop

### Problem
The conviction rethink loop currently just re-queries Memu-Core for more chunks.
It doesn't challenge the plan's assumptions or check for known failure modes.

### Design

```
User request + plan
    ↓
[Proposer] — builds or refines the action plan
    ↓
[Adversary] — tries to break it:
    ├── Query Verifier: "Is the core claim factual?"
    ├── Query Memu-Core: "Did a similar plan fail before?"
    ├── Check Policy: "Does this violate any operator rules?"
    └── Check Fusion: "Do multiple signals agree?"
    ↓
[Gate] — conviction score with adversary findings
    ↓
If conviction < threshold AND rethink_count < max:
    → back to Proposer with adversary feedback
    ↓
If conviction >= threshold:
    → Executor (sandboxed)
```

### LangGraph Graph Definition

```python
from langgraph.graph import StateGraph, END

class AdversaryState(TypedDict):
    user_input: str
    plan: dict
    adversary_findings: list
    conviction: float
    rethink_count: int

graph = StateGraph(AdversaryState)
graph.add_node("propose", propose_node)
graph.add_node("challenge", adversary_node)
graph.add_node("gate", conviction_gate_node)
graph.add_node("execute", execute_node)

graph.set_entry_point("propose")
graph.add_edge("propose", "challenge")
graph.add_edge("challenge", "gate")
graph.add_conditional_edges("gate", should_rethink, {
    "rethink": "propose",
    "execute": "execute",
    "reject": END,
})
graph.add_edge("execute", END)
```

### Implementation Priority
This is Phase 2c — build after Router and Planner are working. The
Adversary pattern is most valuable for `EXECUTE_ACTION` route where
real-world side effects are at stake.

---

## File Layout

```
langgraph/
├── app.py            # existing — add router dispatch to /chat
├── kai_config.py     # existing — episode saver
├── conviction.py     # existing — conviction scoring (add history modifiers)
├── router.py         # NEW — specialist router + classify()
├── planner.py        # NEW — memory-driven planning
└── adversary.py      # FUTURE — proposer-adversary graph
```

---

## Test Plan

| Test | File | What it validates |
|------|------|-------------------|
| Router classification | `scripts/test_router.py` | All 8 routes classify correctly |
| Router edge cases | `scripts/test_router.py` | Ambiguous inputs fall to GENERAL_CHAT |
| Memory planner history | `scripts/test_planner.py` | Past success boosts, past failure penalises |
| Planner with empty history | `scripts/test_planner.py` | Works normally when no past episodes |
| Conviction history modifiers | `scripts/test_conviction.py` | Existing + new modifier signals |
| Integration: /chat routes | `scripts/test_langgraph_service.py` | Extend existing test |

---

## Progress Tracker

- [x] Deep-dive: audited all services (LangGraph, Memu-Core, Verifier, Fusion, Executor)
- [x] Identified existing hooks: `/memory/retrieve`, `/memory/proactive`, `/memory/reflect`, `/verify`, `/fuse`, episode saver
- [x] Designed Router (8 routes, rule-based, zero-LLM for 5 of 8)
- [x] Designed Memory-Driven Planner (history lookup, conviction modifiers)
- [x] Designed Proposer-Adversary (LangGraph graph, deferred to Phase 2c)
- [x] Build `langgraph/router.py` — classify() + 5 dispatch functions
- [x] Build `langgraph/planner.py` — gather_context() + build_enriched_plan()
- [x] Wire router into `/chat` endpoint (Step 0: classify, Step 1: zero-LLM dispatch)
- [x] Wire planner into `/run` endpoint (history lookup → enriched plan → conviction modifier)
- [x] Write tests — 27 router tests + 4 planner test groups, all pass
- [x] Add Makefile targets (test-router, test-planner) — test-core now 26 targets
- [x] All 32 test targets pass as of commit da677f1
- [x] `/chat` returns `X-Kai-Route` header in SSE responses
- [x] Build adversary.py (Phase 2c) — 5 challenge strategies, orchestrator, plan metadata
- [x] Wire adversary into /run (between enriched plan and conviction scoring)
- [x] 7 adversary test groups passing, test-core now 27 targets
- [ ] Dashboard: show route decisions + adversary findings in UI
- [ ] Update conviction.py with configurable history modifier weights
- [ ] Calibration dashboard (predicted vs actual conviction)
- [ ] Token budget tracking per route

---

## Continuation Notes (for next session)

If picking up from here, check:
1. `git log --oneline -5` to see where we stopped
2. `make test-core` to confirm baseline is green (expect 33+ passes)
3. Read this doc for technical spec
4. Read `docs/unfair_advantages.md` for strategic context — Kai's competitive edge analysis
5. Phases 2a+2b+2c all complete. Next: Phase 3 (dashboard, calibration UI, token tracking)
6. All agentic modules: router.py, planner.py, adversary.py, conviction.py, kai_config.py in langgraph/
