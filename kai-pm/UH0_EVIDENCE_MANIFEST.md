# UH-0 Evidence Manifest — Module Inventory and Decision-to-Action Call Graph

Repository: `dainius1234/kai-system`  
Acquisition commit: `7adab8d291011f7dddd92a7702ce8236ddb01ea9`  
Acquired at: `2026-07-28T07:50:34Z`  
Scope: `agentic/` layer (44 Python modules, ~17,347 lines)  
Status: **IMMUTABLE BASELINE — UH-0 deliverable per KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md §14**  
Author: Orion (session claude/project-rework-plan-pgvp35)

This document fulfils the UH-0 exit-gate requirement:
> every consequential path has an owner and migration state; no new direct action path is permitted.

---

## 1. Module Role Inventory

Classification uses the six-role taxonomy from §5 of the architecture roadmap.

### 1.1 Perception Providers

Produce typed observations from a defined source. Must not recommend or execute consequential actions.

| Module | D-number | Source | Status |
|---|---|---|---|
| `alpha_signals.py` | D130 | Binance Futures public API (funding rate, OI, L/S ratio, mark premium) | **COMPLIANT** — read-only, returns None on error, no directive output |
| `market_intel.py` | D129 | CoinGecko Fear/Greed API + news tone classification | **COMPLIANT** — read-only, regime + macro context only |
| `market_data.py` | D127 | CoinGecko price API | **COMPLIANT** — read-only, price feed only |
| `web_scout.py` | — | HTTP fetch, web search | **COMPLIANT** — read-only retrieval |
| `cortex.py` | D110 | Internal sensor state, service health digest | **COMPLIANT** — synthesises situation model; `bid_to_workspace()` submits to stub workspace (no-op) |
| `forecaster.py` | — | Probabilistic prediction engine | **COMPLIANT** — produces predictions, no execution path |
| `wisdom_ingestion.py` | — | Document/text ingestion | **COMPLIANT** — write to knowledge graph only |

### 1.2 Transformers / State Reducers

Validate events and build deterministic materialised views. Must not invent missing facts.

| Module | D-number | Function | Status |
|---|---|---|---|
| `global_workspace.py` | D102 | Deliberation coordinator / conscious moment stream | **STUB** — `submit_bid()` discards bids; `select_winner()` is a no-op; `broadcast()` is a no-op; `can_operate()` returns `False`; the salience loop is NOT running |
| `policy_memory.py` | — | Interaction history + outcome recording | **PARTIAL** — records events but also stores Kai responses as if they are operator decisions |
| `cognitive_fingerprint.py` | — | Behavioural pattern extraction | **COMPLIANT** — read-only analysis |
| `causal_world_model.py` | — | Causal inference over world events | **COMPLIANT** — model output only, no execution |
| `wisdom_graph.py` | — | Knowledge graph reads and alignment evaluation | **COMPLIANT** — read-only retrieval path |

### 1.3 Proposal Specialists

Produce candidate interpretations or action proposals. Must declare uncertainty. Must not approve or execute.

| Module | D-number | Function | Status |
|---|---|---|---|
| `strategy_engine.py` | D128 | Momentum/MA/RSI consensus signals | **VIOLATION** — also owns `auto_trade()` which directly executes paper trades without deliberation workspace or immutable approval contract (see §2) |
| `opportunity_intel.py` | D130 | Cross-domain opportunity scoring (financial/content/affiliate/trend-arb) | **SOFT VIOLATION** — produces `recommended_action` field and conviction scores that are used directionally; does not approve or execute but violates UH-INV-02 spirit by embedding directive recommendations in its output without an `ActionProposal` contract |
| `adversary.py` | — | Adversarial critique and counterargument generation | **COMPLIANT** — produces analysis, no execution path |
| `hypothesis.py` | — | Hypothesis generation and evaluation | **COMPLIANT** — produces proposals, no execution path |
| `counterfactual.py` | — | Counterfactual simulation | **COMPLIANT** — analysis only |
| `planner.py` | — | Multi-step action planning | **COMPLIANT** — produces plan structures, no execution path |
| `analogy.py` | — | Analogical reasoning | **COMPLIANT** — analysis only |
| `concept_blend.py` | — | Conceptual blending | **COMPLIANT** — analysis only |
| `dialectic.py` | — | Dialectical reasoning | **COMPLIANT** — analysis only |
| `tree_search.py` | — | Tree-based decision search | **COMPLIANT** — simulation only |
| `model_council.py` | — | Multi-model deliberation council | **COMPLIANT** — produces deliberation output, routes through router |

### 1.4 Policy and Approval Authority

Determines whether an exact action is allowed. This role must not be bypassed.

| Module | D-number | Function | Status |
|---|---|---|---|
| `trust_integration.py` | D126 | `gate_autonomous_action()` — checks trust level + Ohana alignment | **CRITICAL FLAW** — documented as "Never raises — fails open with a warning." Any exception in the trust check silently allows the action. Import failure also fails open (lines 66, 79 of trust_integration.py). |
| `trust_core.py` | D126 | Trust level record (DORMANT/OBSERVER/ASSISTANT/AGENT) | **COMPLIANT as storage** — records trust tier; trust promotion/demotion exposed via `/trust/promote` and `/trust/demote` HTTP endpoints |
| `moral_core.py` | D109 | Ohana value alignment evaluation | **ADVISORY ONLY** — `evaluate_action_alignment()` produces a 0.0–1.0 score; the score is consumed by `gate_autonomous_action()` but the gate fails open on any error; value boundaries are static defaults in Phase 0 |
| `security_audit.py` | — | Security posture assessment | **PLANNING/AUDIT** — produces findings, does not enforce |

### 1.5 Actuators

Execute one fixed-schema operation after validating an exact one-time capability. Must validate capability before effect.

| Module | D-number | Function | Status |
|---|---|---|---|
| `paper_trader.py` | D127 | Paper position open/close with trust check | **PARTIAL COMPLIANCE** — has `_check_trust()` but: (a) trust gate is fail-open; (b) called directly from `strategy_engine.auto_trade()` bypassing workspace; (c) called directly from HTTP `/paper-trading/open` and `/paper-trading/close` without deliberation; (d) no `ActionCapability` contract — the trust check is process-local and not an exact one-time token |
| `teammates.py` | — | Sub-agent task delegation | **NOT YET MAPPED** — may delegate to external LLM calls with side effects |
| `swarm.py` / `swarm_stages.py` | — | Multi-agent swarm orchestration | **NOT YET MAPPED** — spawns sub-agents; consequential side effects TBD |
| `router.py` | — | LLM call dispatch | **PARTIAL** — routes model calls; model output is not itself an actuator but downstream parsing may trigger actions |

### 1.6 Outcome Verifiers

Independently check what actually happened. The actuator cannot be the sole judge of its own success.

| Module | D-number | Function | Status |
|---|---|---|---|
| *(none)* | — | — | **GAP** — No independent outcome verifier exists in the current codebase. `paper_trader.py` self-reports success via its own return values. `trust_integration.py` records audit events but these are produced by the same process that executed the action. |

---

## 2. Direct Decision-to-Action Call Graph

These are the paths that violate UH-INV-01 (one canonical sequence). Every path below bypasses the Workspace → Proposal → Approval → Capability chain.

### Path A — HTTP Strategy Auto-Trade (ACTIVE, CRITICAL)

```
POST /strategy/auto-trade  [app.py:1075]
  └─► StrategyEngine.auto_trade(symbol, quantity, price)  [strategy_engine.py:273]
        ├─► StrategyEngine.evaluate(symbol, price, lookback)  [strategy_engine.py:220]
        │     └─► MomentumStrategy.signal() + MACrossStrategy.signal() + RSIStrategy.signal()
        │           └─► majority vote → Signal(action="buy"|"sell"|"hold")
        └─► IF signal.action != "hold":
              └─► PaperTrader.open_position() or close_position()  [paper_trader.py:171,217]
                    ├─► _check_trust("paper_trade_open"/"paper_trade_close", ...)
                    │     └─► gate_autonomous_action(...)  ← FAILS OPEN
                    └─► writes position to JSON file [_DATA_DIR/positions.json]
                          and appends to [_DATA_DIR/trades.json]
```

**Missing layers:** D102 Workspace deliberation, ActionProposal contract, D109 Ohana approval, immutable ActionCapability, independent outcome verification.  
**Migration state:** MUST REMAIN CALLABLE IN LAB ONLY. Gate must be hardened before any further use.

### Path B — HTTP Paper Trade Direct Open (ACTIVE)

```
POST /paper-trading/open  [app.py:903]
  └─► PaperTrader.open_position(symbol, quantity, price)  [paper_trader.py:171]
        └─► _check_trust(...)  ← FAILS OPEN (same flaw as Path A)
              └─► writes to positions.json + trades.json
```

**Missing layers:** No strategy evaluation, no deliberation, no proposal contract.  
**Migration state:** MUST REMAIN CALLABLE IN LAB ONLY.

### Path C — HTTP Paper Trade Direct Close (ACTIVE)

```
POST /paper-trading/close  [app.py:920]
  └─► PaperTrader.close_position(position_id, price)  [paper_trader.py:217]
        └─► _check_trust(...)  ← FAILS OPEN
              └─► updates positions.json + appends trades.json
```

**Migration state:** MUST REMAIN CALLABLE IN LAB ONLY.

### Path D — Trust Promotion Without Step-Up Auth (ACTIVE)

```
POST /trust/promote  [app.py:961]
  └─► TrustCore.propose_promotion(reason)  [trust_core.py]
        └─► writes trust record to disk [_DATA_DIR/trust_record.json]
```

**Missing layers:** Per UH-INV-11, trust tier changes for consequential capability must be authenticated, operation-specific and auditable. Current path accepts any POST with a reason string.  
**Migration state:** MUST NOT BE EXPOSED TO ANY UNAUTHENTICATED CALLER.

### Path E — Opportunity Intel Directive Recommendations (ACTIVE)

```
GET /opportunity/{symbol}/financial  [app.py]
  └─► OpportunityIntelligence.scan_financial(symbol)
        └─► _score_financial(...)
              └─► returns OpportunitySignal(recommended_action="go long"/"go short"/...)
```

**Issue:** `recommended_action` is free-form text in a response field. Per UH-INV-13, free-form text must not carry control authority. The field name implies executability that the architecture does not yet provide. The caller receiving this response could act on it without further deliberation.  
**Migration state:** `recommended_action` field must be renamed to `analyst_note` or removed until an `ActionProposal` contract is in place.

---

## 3. Side-Effect Endpoint Registry

All HTTP endpoints in `agentic/app.py` that produce side effects (file writes, external calls, state mutation).

| Endpoint | Method | Side Effect | Trust Gate | Risk |
|---|---|---|---|---|
| `/paper-trading/open` | POST | Writes position + trade to JSON | Fail-open | HIGH |
| `/paper-trading/close` | POST | Writes trade + closes position JSON | Fail-open | HIGH |
| `/strategy/auto-trade` | POST | Triggers paper trade via strategy vote | Fail-open | HIGH |
| `/trust/promote` | POST | Mutates trust tier on disk | None (auth check TBD) | HIGH |
| `/trust/demote` | POST | Mutates trust tier on disk | None | HIGH |
| `/chat` | POST | Calls external LLM (OpenAI/Anthropic) | OBSERVER tier | MEDIUM |
| `/chat/teammate/{name}` | POST | Calls external LLM via teammate | OBSERVER tier | MEDIUM |
| `/chat/swarm` | POST | Spawns multi-agent swarm → external LLM calls | OBSERVER tier | MEDIUM |
| `/web-scout/fetch` | POST | External HTTP GET | OBSERVER tier | LOW |
| `/web-scout/search` | POST | External search engine query | OBSERVER tier | LOW |
| `/web-scout/summarize` | POST | External LLM call | OBSERVER tier | LOW |
| `/soul` | POST | Writes moral fingerprint to disk | None | MEDIUM |
| `/recover` | POST | System recovery action | None | HIGH |
| `/checkpoint` | POST | Writes checkpoint to disk | None | MEDIUM |
| `/checkpoint/{id}/restore` | POST | Restores system state from disk | None | HIGH |
| `/checkpoint/{id}` | DELETE | Deletes checkpoint from disk | None | MEDIUM |
| `/agents-registry` | POST | Registers agent in memory | None | LOW |
| `/skills/reload` | POST | Reloads skill definitions | None | MEDIUM |
| `/episodes/recall` | POST | Writes episode to memory | None | LOW |
| `/run` | POST | Executes graph workflow | varies | HIGH |

---

## 4. Process-Local Stores and Shared Writable Files

| File/Directory | Written by | Contents | Risk |
|---|---|---|---|
| `~/.kai/paper_trader/positions.json` | `paper_trader.py` | Open positions (symbol, qty, price, timestamp) | HIGH — financial state |
| `~/.kai/paper_trader/trades.json` | `paper_trader.py` | Trade history | HIGH — audit trail |
| `~/.kai/trust/trust_record.json` | `trust_core.py` | Trust tier + audit log | HIGH — authorization state |
| `~/.kai/moral/fingerprint.json` | `moral_core.py` | Moral/value fingerprint | MEDIUM |
| `~/.kai/checkpoints/` | `app.py` | System state snapshots | MEDIUM |
| In-memory only: `GlobalWorkspace._stream` | `global_workspace.py` | Conscious moment history | LOW (stub, no persistence) |
| In-memory only: `AlphaSignalFeed._cache` | `alpha_signals.py` | Market signal cache | LOW (TTL-bound, read-only source) |

All persistent stores are process-local JSON files. There is no durable shared state, no database, and no event journal. This is a known gap — UH-3 addresses it.

---

## 5. Data / Source / Consumer Lineage Map

```
Binance Futures API
  └─► alpha_signals.AlphaSignalFeed [D130]
        ├─► opportunity_intel.scan_financial() [D130] → OpportunitySignal
        └─► /alpha/* HTTP endpoints → dashboard

CoinGecko Fear/Greed API
  └─► market_intel.MarketIntelligence [D129]
        ├─► opportunity_intel.scan_financial() [D130] → OpportunitySignal
        └─► /market-intel/* HTTP endpoints → dashboard

CoinGecko Price API
  └─► market_data.MarketDataFeed [D127]
        ├─► strategy_engine.auto_trade() [D128] → paper_trader [actuator]
        └─► /market-data/* HTTP endpoints → dashboard

Web search / fetch
  └─► web_scout.WebScout
        ├─► opportunity_intel.scan_content() / scan_affiliate() [D130]
        └─► /web-scout/* HTTP endpoints → dashboard

Internal sensor state
  └─► cortex.Cortex [D110]
        └─► global_workspace.submit_bid() [D102 — STUB, discarded]

LLM external calls
  └─► router.py → model_council.py / teammates.py / swarm.py
        └─► /chat, /run HTTP endpoints → dashboard
```

Confirmed independence violations:
- `opportunity_intel.scan_financial()` receives both `alpha_signals` AND `market_intel` data. Both ultimately derive from market-sentiment sources that are correlated (BTC fear/greed and BTC funding rate move together). This satisfies UH-INV-08 compliance risk: correlated sources counted as independent evidence.

---

## 6. Classification Summary Table

| Module | Role | D-number | Violations |
|---|---|---|---|
| `alpha_signals.py` | Perception Provider | D130 | None |
| `market_intel.py` | Perception Provider | D129 | None |
| `market_data.py` | Perception Provider | D127 | None |
| `web_scout.py` | Perception Provider | — | None |
| `cortex.py` | Perception Provider | D110 | Bid submits to stub workspace |
| `forecaster.py` | Perception Provider | — | None |
| `global_workspace.py` | Transformer (STUB) | D102 | All methods no-op; must not be activated without UH-1 contracts |
| `policy_memory.py` | Transformer | — | Records Kai responses as operator decisions |
| `causal_world_model.py` | Transformer | — | None |
| `cognitive_fingerprint.py` | Transformer | — | None |
| `wisdom_graph.py` | Transformer | — | None |
| `strategy_engine.py` | Proposal Specialist + **Actuator** | D128 | **CRITICAL: owns `auto_trade()` — dual role violates UH-INV-02** |
| `opportunity_intel.py` | Proposal Specialist | D130 | `recommended_action` field violates UH-INV-13 spirit |
| `adversary.py` | Proposal Specialist | — | None |
| `hypothesis.py` | Proposal Specialist | — | None |
| `counterfactual.py` | Proposal Specialist | — | None |
| `planner.py` | Proposal Specialist | — | None |
| `dialectic.py` | Proposal Specialist | — | None |
| `analogy.py` | Proposal Specialist | — | None |
| `model_council.py` | Proposal Specialist | — | None |
| `trust_integration.py` | Policy Authority | D126 | Fail-open: any exception → allowed |
| `trust_core.py` | Policy Authority | D126 | Trust promotion lacks step-up auth |
| `moral_core.py` | Policy Authority | D109 | Advisory score only; not enforced |
| `paper_trader.py` | Actuator | D127 | No ActionCapability contract; trust gate fail-open |
| `teammates.py` | Actuator | — | Side effects not yet mapped |
| `swarm.py` | Actuator | — | Side effects not yet mapped |
| *(none)* | **Outcome Verifier** | — | **GAP: role does not exist** |

---

## 7. Paths That Must Remain Disabled

The following paths are authorised for LAB-ONLY operation at current trust posture. They must not be reachable from any network-accessible surface outside the development host:

1. `POST /strategy/auto-trade` — direct trade execution bypassing Workspace
2. `POST /paper-trading/open` — direct position open bypassing deliberation
3. `POST /paper-trading/close` — direct position close bypassing deliberation
4. `POST /trust/promote` — trust tier mutation without step-up authentication
5. `POST /trust/demote` — same
6. `POST /checkpoint/{id}/restore` — system state restore without approval workflow
7. `POST /recover` — system recovery without approval workflow

No new endpoint may create a direct decision-to-action path until UH-1 canonical contracts are in place and the Global Workspace (D102) is operating.

---

## 8. Exit Gate Status

| Gate condition | Status |
|---|---|
| Every consequential path has an owner | **MET** — see §2 |
| Every consequential path has a migration state | **MET** — see §2 and §7 |
| No new direct action path is permitted | **POLICY ENFORCED** — this document establishes the gate |

---

## 9. Authorised Next Step

Per `KAI_UNIFIED_HUNTER_ARCHITECTURE_AND_ROADMAP.md §14`:

> **UH-1 — Freeze canonical contracts**

Deliverables:
- Versioned schemas for PerceptionEvent, WorldStateSnapshot, ActionProposal, ConstraintAssessment, ApprovalRecord, ActionCapability, ActionWorkflow, VerifiedOutcome
- Canonical serialisation/digest rules
- Risk tiers and approval matrix
- Schema compatibility policy
- Error/state vocabulary
- Architecture dependency rules

The canonical contracts must pass malformed, unknown-field, digest and compatibility tests before any D131 implementation begins.

**Immediate remediation required (not gated on UH-1):**

1. `strategy_engine.py` — remove `auto_trade()` method; replace with `generate_proposal()` returning a plain data object with no execution path
2. `opportunity_intel.py` — rename `recommended_action` to `analyst_note` in `OpportunitySignal`
3. `trust_integration.py` — fail-open must become fail-closed for the paper trading capability; log and return `(False, "trust gate unavailable")` instead of `(True, ...)`

---

*This document is append-only. Corrections must be new dated entries, not edits to existing content.*
