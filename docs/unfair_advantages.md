# Kai's Unfair Advantages — Strategic Intelligence Document

> **Purpose:** This is Kai's competitive edge analysis. Written from the AI's
> perspective: what makes this system fundamentally better than other builds,
> and how does every component compound into something greater.
>
> **Last updated:** 4 March 2026 — Merged Action Plan (operator research + AI-native)
>
> **Target hardware:** Lenovo laptop, RTX 5080 GPU, TPM 2.0.
> Codespace = dev/staging. Laptop = production fortress.
> GPU arrival unlocks: local LLM inference (Kimi K2, DeepSeek-V4, etc.),
> faster-whisper STT, real avatar generation, OMAR self-play.
> TPM 2.0 unlocks: hardware-anchored Soulbound Identity (P3).

---

## The Core Thesis

Most AI assistants are **stateless parrots**. They:
1. Forget everything between sessions
2. Treat every request identically regardless of history
3. Execute blindly — no self-doubt, no challenge, no pre-mortem
4. Burn tokens on everything (including things that don't need an LLM)
5. Can't learn from their own mistakes
6. Have no concept of "I was wrong about this before"
7. Don't track whether their confidence predictions are accurate
8. Don't adapt to operator patterns

Kai is designed to be **none of these things**.

---

## Advantage Map

### 1. Zero-LLM Routing (router.py) — Token Efficiency

| What it does | Why it's unfair |
|---|---|
| 5 of 8 routes bypass the LLM entirely | 60%+ of requests cost zero inference tokens |
| Rule-based classifier runs in microseconds | No latency tax for simple lookups |
| Confidence-gated: only dispatches if confidence ≥ 0.7 | Graceful fallback — never loses quality |

**Compounding effect:** Every token saved on a memory recall is a token
available for hard reasoning tasks. Over thousands of requests, this is
massive.

### 2. Episodic Memory (kai_config.py + memu-core) — Learning Loop

| What it does | Why it's unfair |
|---|---|
| Every /run execution is saved as an episode | The system builds a case history |
| Episodes include: input, output, conviction score, outcome score | Full auditability |
| Redis primary + checksummed file spool fallback | Never loses an episode |
| `recall(days=N)` for history lookup | Instant access to relevant past actions |

**Compounding effect:** After 100 episodes, the system has a statistical
model of what works and what doesn't — without any explicit training.

### 3. Conviction Scoring (conviction.py) — Self-Doubt

| What it does | Why it's unfair |
|---|---|
| 5-signal score (0–10) before any execution | The system won't act until it's confident |
| MIN_CONVICTION = 8.0 (high bar) | Defaults to caution — most builds default to action |
| MAX_RETHINKS = 3 with diminishing returns | Bounded exploration, not infinite loops |
| Operator override file for exceptions | Human always has final say |

**Compounding effect:** The rethink loop means conviction always improves
before execution. Combined with memory, the second attempt at something
the system has seen before starts from a higher baseline.

### 4. Memory-Driven Planning (planner.py) — Historical Context

| What it does | Why it's unfair |
|---|---|
| Checks episode history before building any plan | Never repeats a known failure |
| Past success: +1.0 conviction boost | Rewards proven approaches |
| Past failure: -1.5 conviction penalty + warning | Penalises known bad paths |
| Past corrections: -0.5 + injects correction as constraint | Learns from operator feedback |
| Proactive nudges injected into plan context | Doesn't forget deadlines or reminders |

**Compounding effect:** The system gets better at planning over time purely
from its own execution history — no fine-tuning, no retraining.

### 5. Proposer-Adversary Loop (adversary.py) — Self-Challenge ★NEW

| What it does | Why it's unfair |
|---|---|
| Before execution, 5 parallel challenges stress-test the plan | Plans are battle-tested before they run |
| History challenge: "did something like this fail before?" | Uses episode memory as negative examples |
| Verifier challenge: "is the core claim factually supported?" | Cross-references against evidence |
| Policy challenge: "does the gate allow this tool+action?" | Pre-screens for policy violations |
| Consistency challenge: "is the plan internally coherent?" | Catches contradictions |
| Calibration challenge: "was our confidence accurate last time?" | Tracks prediction accuracy |

**Compounding effect:** Every challenge that catches a problem before
execution is a failure prevented. Over time, the adversary builds up a
"failure library" that makes the system increasingly hard to fool.

### 6. Confidence Calibration — Meta-Learning

| What it does | Why it's unfair |
|---|---|
| Tracks (predicted conviction, actual outcome) pairs per pattern | Knows when its own confidence is miscalibrated |
| Calibration drift detector | Adjusts conviction when pattern shows systematic over/under-confidence |
| Stored as adversary memories in memu-core | Persists across restarts |

**Compounding effect:** Most AI systems have no idea whether their
confidence correlates with reality. Kai does. This is one of the hardest
problems in AI safety, solved pragmatically with episode data.

### 7. Multi-Signal Triangulation (fusion-engine + verifier + memory)

| What it does | Why it's unfair |
|---|---|
| For high-stakes actions: query multiple LLMs | Single-model failure doesn't propagate |
| Cross-check with Verifier evidence pipeline | Independent validation |
| Memory evidence pack: scored, ranked, trust-tiered | Not just "did I see this before" but "how much do I trust it" |

**Compounding effect:** Three independent signal sources (LLMs + verifier +
memory) must agree. Probability of all three being wrong simultaneously is
multiplicatively small.

### 8. Operator Pattern Learning

| What it does | Why it's unfair |
|---|---|
| Co-sign denials logged in tool-gate ledger | System knows where operator disagrees |
| Conviction overrides tracked | System knows where it was wrong |
| Corrections stored as special memory category | Explicit "I was wrong" signal |

**Compounding effect:** Operator feedback is the highest-quality training
signal available. Most builds ignore it. Kai treats every correction as
gold.

### 9. Spaced Repetition + Ebbinghaus Decay (memu-core)

| What it does | Why it's unfair |
|---|---|
| Memory relevance decays over time (half-life: 14 days) | Old memories don't pollute current context |
| Access count bumps rescue important memories | Frequently accessed memories stay fresh |
| Weekly compression archives noise | Signal-to-noise ratio improves over time |
| Reflection/consolidation writes insights back | The system summarises its own learning |

**Compounding effect:** Most RAG systems treat all stored data equally.
Kai's memory is shaped like a human's — recent and frequently-used
information is prioritised. This means context windows are always filled
with the most relevant information.

### 10. Air-Gapped Safety Architecture

| What it does | Why it's unfair |
|---|---|
| Default offline — no network unless operator enables | Can't be remotely compromised |
| HMAC-signed inter-service calls | Can't be spoofed internally |
| Append-only hash-chained ledger | Full audit trail, tamper-evident |
| Sandbox execution with allowlists | Can't break out of approved actions |
| ClamAV malware scanning on executor | Defence against stored payloads |
| Nonce replay prevention | Can't replay old authorised requests |

**Compounding effect:** Security is multiplicative. Each layer compounds
with the others. Breaking this system requires simultaneously compromising
HMAC keys + policy + ledger + sandbox + operator approval.

---

## The Compound Effect

No single advantage is revolutionary. The compound of ALL of them is.

```
Request arrives
  ↓ Router: zero-LLM dispatch? (saves tokens)
  ↓ Planner: check episode history (prevents repeated failures)
  ↓ Adversary: 5 parallel challenges (stress-tests the plan)
  ↓ Conviction: multi-signal scoring (calibration-aware)
  ↓ Policy gate: HMAC-signed, rate-limited, co-sign if needed
  ↓ Executor: sandboxed, allowlisted, scanned
  ↓ Post-mortem: save episode, store corrections, update calibration
  ↓ Memory: auto-memorize, reflection, compression
  ↓ Next request starts from a better baseline
```

Each request makes the next one better. This is the flywheel.

---

## What Other Builds Don't Have

| Feature | ChatGPT | Claude | Gemini | Open-source RAG | **Kai** |
|---|---|---|---|---|---|
| Persistent episodic memory | ❌ | ❌ | Partial | Sometimes | ✅ |
| Self-doubt (conviction gate) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Self-challenge (adversary) | ❌ | ❌ | ❌ | ❌ | ✅ |
| Confidence calibration | ❌ | ❌ | ❌ | ❌ | ✅ |
| Zero-LLM routing | ❌ | ❌ | ❌ | ❌ | ✅ |
| Memory decay + spaced rep | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multi-model triangulation | ❌ | ❌ | ❌ | Rare | ✅ |
| Operator correction learning | ❌ | Limited | ❌ | ❌ | ✅ |
| Tamper-evident audit trail | ❌ | ❌ | ❌ | ❌ | ✅ |
| Air-gapped capable | ❌ | ❌ | ❌ | ✅ | ✅ |
| Voice emotion analysis | ❌ | ❌ | ❌ | ❌ | ✅ |
| Predictive failure forecast | ❌ | ❌ | ❌ | ❌ | ✅ |
| Bio-inspired self-healing | ❌ | ❌ | ❌ | ❌ | ✅ |
| World-anchor grounding | ❌ | Plugin | Plugin | ❌ | ✅ |
| Context compression (MARS) | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## Progress Tracker

### Phase 2a: Specialist Router ✅
- [x] router.py — 8-route classifier, 5 zero-LLM dispatch functions
- [x] Wired into /chat endpoint
- [x] 27 classification tests passing
- [x] Commit: da677f1

### Phase 2b: Memory-Driven Planner ✅
- [x] planner.py — gather_context, build_enriched_plan, episode similarity
- [x] Wired into /run endpoint
- [x] 4 test groups passing
- [x] Commit: da677f1

### Phase 2c: Proposer-Adversary Loop ✅
- [x] adversary.py — 5 challenge strategies + orchestrator + verdict metadata
- [x] Wired into /run endpoint (between planner and conviction scoring)
- [x] 7 test groups passing (history, consistency, calibration, hash, metadata, orchestrator, recommendations)
- [x] Commit: 0e61f6d

---

## Merged Action Plan (March 2026)

> **Sources:** Operator research (arXiv 2601–2602, ICLR/ICSE/AAAI 2026)
> merged with AI-native Phase 4 blueprints. Priority = impact ÷ effort.
> Everything runs on **local models only** — no cloud dependencies.

### LLM Backend: Local-First Model Strategy

All inference runs locally via Ollama or OpenAI-compatible endpoints.
Current backends in `common/llm.py`:

| Specialist | Model | Purpose |
|---|---|---|
| Ollama (default) | qwen2:0.5b (configurable) | General-purpose, light |
| DeepSeek-V4 | deepseek-v4 | Code, reasoning, planning |
| Kimi-2.5 | kimi-2.5 | Multimodal, search, summarise |
| Dolphin | dolphin-mistral | Uncensored, creative |

**Pending addition:** Kimi K2 (Moonshot AI, June 2025) — 1T MoE with 32B
active params, 128K context, Apache 2.0 license. Available on Ollama as
`kimi-k2`. Outperforms DeepSeek-V3 on agentic benchmarks.
Add via: `LLM_KIMI_K2_URL` env var + `_DEFAULT_URLS["Kimi-K2"]` in llm.py.
Router can select Kimi-K2 for agentic/tool-use tasks where its MoE
architecture excels.

### Priority Build Order

| # | Name | Research Source | Effort | Status |
|---|---|---|---|---|
| **P1** | Failure Taxonomy + Metacognitive Rules | MARS/MUSE + our 4h | Small | ✅ DONE |
| **P2** | SELAUR (Uncertainty-Aware Evolution) | arXiv 2602 (#11) | Small | ✅ DONE |
| **P3** | Soulbound Identity (Software) | ERC-5192/8004 (#7) | Medium | Skipped (HMAC works; TPM swap on hardware arrival) |
| **P4** | TMC + Contradiction Memory | TMC paper (#8) + our 4a | Medium | ✅ DONE |
| **P5** | GEM (Cognitive Alignment) | GEM paper (#6) | Medium | ✅ DONE |
| **P6** | Knowledge Boundary + Active Probing | KBM paper (#3) + our 4c | Medium | ✅ DONE |
| **P7** | Silence-as-Signal | Our 4b | Small | ✅ DONE |
| **P8** | Dashboard: Thinking Pathways | vLLM-SR (#5) + Phase 3 | Large | ✅ DONE |
| **P9** | Security Self-Hacking | MSR paper (#9) | Medium | ✅ DONE |
| **P10** | Predictive Pre-Computation | Our 4e | Medium | ✅ DONE |
| **P11** | Operator Tempo Modeling | Our 4d | Medium | ✅ DONE |
| **P12** | Self-Deception Detection | Our 4f | Medium | ✅ DONE |
| **P13** | Recursive Self-Improvement Gate | Our 4g | Medium | ✅ DONE |
| **P14** | Temporal Self-Model | Our 4i | Medium | ✅ DONE |
| **P15** | Dream State (Offline Consolidation) | Our 4j | Large | ✅ DONE |
| **Future** | OMAR Self-Play (#2/#13) | Needs spare local GPU | Large | Parked |
| **Future** | ZK Privacy Learning (#10) | Heavy crypto infra | Large | Parked |
| **Future** | ReCiSt Bio-Resilience (#4) | Already mostly built | Docs only | Parked |

---

### P1: Failure Taxonomy + Metacognitive Rules

> Merge of MARS/MUSE metacognitive reflection + our original 4h.
> "Don't just know THAT it failed — know WHY, and extract a rule."

- **Problem:** "This failed" is useless. WHY did it fail? And what rule
  should prevent repeating it?
- **Solution:** Classify every failure into a taxonomy enum. Then go
  further: extract metacognitive rules ("if X, never Y") from failure
  patterns using MARS-style abstraction. Store rules as constraint
  memories that the planner injects into future plans.
- **Taxonomy enum:** `data_insufficient`, `policy_blocked`, `confidence_low`,
  `operator_overridden`, `service_unavailable`, `contradicted_by_evidence`,
  `time_expired`, `scope_exceeded`
- **Metacognitive rules:** After classifying a failure, generate an
  "if-then-never" rule. Example: failure_class=`contradicted_by_evidence`
  on topic "crypto prices" → rule: "if topic=crypto prices, always verify
  with fresh data before asserting"
- **Files:** `kai_config.py` (FailureClass enum + classify function),
  `langgraph/app.py` (post-episode classification + rule extraction),
  `adversary.py` (use failure_class in history challenge for targeted warnings)
- **Tests:** `scripts/test_failure_taxonomy.py`
- **Why it compounds:** The adversary doesn't just say "this failed before" —
  it says "this failed because of insufficient data, and the rule is:
  always check memu first for this topic type."

### P2: SELAUR (Uncertainty-Aware Self-Evolution)

> From arXiv 2602 research on self-evolving agents.
> "Failures when uncertain = the most valuable learning signal."

- **Problem:** Right now failed episodes sit as dead weight with low scores.
  A failure where Kai was 90% confident is very different from a failure
  where Kai was 50% confident. The uncertain failure is actually MORE
  valuable — it maps the edge of competence.
- **Solution:** Scale episode learning value by uncertainty. High-uncertainty
  failures become high-reward training trajectories. Conviction scores that
  barely crossed the threshold then failed → maximum learning signal.
- **Implementation:** New `_compute_learning_value()` in episode save logic.
  `learning_value = f(uncertainty, outcome, conviction_delta)`. Episodes with
  high learning_value get boosted importance in memu-core so they surface
  more in future retrievals.
- **Files:** `langgraph/app.py` (compute + store learning_value in episode),
  `kai_config.py` (learning_value field in episode schema)
- **Tests:** `scripts/test_selaur.py`
- **Why it compounds:** Over time, Kai's highest-importance memories are
  exactly the cases where it was wrong-and-uncertain — the frontier of
  growth. This is how you evolve WITHOUT retraining.

### P3: Soulbound Identity (Software-Signed)

> From ERC-5192 + ERC-8004 research on non-transferable identity.
> "Every byte provably Kai's. Copy = detectable fake."

- **Problem:** Anyone can copy Kai's memory files. No chain of custody.
- **Solution:** HMAC-chained soulbound identity. Every memory block is
  hash-linked and signed with the operator key. Genesis block = Kai's
  birth certificate. Copy the files → signatures don't verify → fake.
- **Implementation:** `pending/soulbound.py` — software HMAC signing now,
  API designed for future TPM swap. Genesis auto-created on first run.
  Pause/resume on integrity failure. Proof export/verify for individual
  memories.
- **Hardware strategy:** Codespace (now) → software HMAC signing with
  `KAI_OPERATOR_KEY`. Lenovo laptop (production) → TPM 2.0 hardware
  signing at persistent handle `0x81000001`. Same chain/pause/resume API,
  same genesis format. Swap `_sign_block()` from HMAC to TPM call on
  deploy day — zero API changes. `TPM_UNSEAL_CMD` already in .env.example.
- **Files:** `pending/soulbound.py` (identity engine),
  `memu-core/app.py` (wrap memorize in soul.add_memory),
  `dashboard/app.py` (/status, /resume endpoints)
- **Tests:** `scripts/test_soulbound.py`
- **Why it compounds:** Every memory written after soulbound is provably yours.
  Legal personhood, insurance, reputation — all depend on provable identity.

### P4: TMC + Contradiction Memory ✅ DONE

> Merge of TMC (Tool-Memory Conflict) paper + our original 4a.
> "Detect when Kai contradicts himself OR when tools contradict memory."

- **What was built:**
  - `detect_contradiction()` in memu-core — 3 detection strategies: numeric drift,
    negation flip, topic overlap threshold. Skips quarantined records.
  - `ContradictionResult` class with conflict type, similarity, explanation.
  - `POST /memory/assert` endpoint — contradiction-checked memorize. Returns conflict
    for operator review (unless `force=True`). Marks superseded memories.
  - `_extract_numeric_claims()` — pulls currency, percentage, duration claims.
  - `_ASSERTION_SIGNALS` and `_NEGATION_SIGNALS` regex patterns for polarity detection.
- **Files:** `memu-core/app.py` (contradiction engine + `/memory/assert` endpoint),
  `langgraph/app.py` (wired into correction learning path)
- **Tests:** `scripts/test_contradiction.py` — 13 tests (numeric drift, negation flip,
  empty/unrelated/poisoned edge cases, numeric extraction, default result)

### P5: GEM (Cognitive Alignment from Minimal Feedback) ✅ DONE

> From GEM paper: model operator's reasoning structure, not just preferences.
> "Corrections → understand HOW the keeper thinks."

- **What was built:**
  - `extract_preference()` in kai_config.py — compares original output vs correction,
    extracts word-level diffs, generates "keeper prefers X over Y" statements with topic context.
  - `POST /memory/preferences` in memu-core — stores operator preference as pinned memory
    (importance=0.95, category="preference").
  - `GET /memory/preferences` in memu-core — retrieves all preferences for plan injection.
  - `_fetch_preferences()` in planner.py — parallel fetch during context gathering.
  - Preferences injected as `apply_preference` steps in `build_enriched_plan()`.
  - Wired into langgraph/app.py correction learning: corrections auto-extract and store preferences.
- **Files:** `langgraph/kai_config.py` (extract_preference), `memu-core/app.py` (preference endpoints),
  `langgraph/planner.py` (fetch + inject), `langgraph/app.py` (correction → preference pipeline)
- **Tests:** `scripts/test_gem_preferences.py` — 13 tests (extraction, empty cases, topic context,
  added/removed words). `scripts/test_planner_preferences.py` — 8 tests (injection, context population,
  max cap, plan counting).

### P6: Knowledge Boundary Mapping + Active Probing ✅ DONE

> Merge of KBM research + our original 4c.
> "Map ignorance explicitly. Know what you don't know."

- **What was built:**
  - `TopicBoundary` dataclass in kai_config.py — competence snapshot per topic cluster
    (episodes, successes, failures, avg conviction, gap flag, probe question).
  - `build_knowledge_boundary()` in kai_config.py — clusters episodes by topic keywords,
    computes success/failure rates, flags gaps (success < 50% or conviction < 6.0),
    generates probing questions for weak areas. Sorted gaps-first.
  - `GET /memory/boundary` in memu-core — aggregates stored memories by category,
    calculates coverage metrics, generates probing questions for low-coverage categories.
- **Files:** `langgraph/kai_config.py` (TopicBoundary, build_knowledge_boundary),
  `memu-core/app.py` (`/memory/boundary` endpoint)
- **Tests:** `scripts/test_gem_preferences.py` — 6 tests on knowledge boundary
  (gap identification, empty episodes, min filter, probe questions, sort order, dataclass fields)

### P7: Silence-as-Signal ✅

> Original 4b: "The absence of a question IS information."

- **Problem:** Operator stops asking about a topic. Why?
- **Solution:** `GET /memory/silence` endpoint in memu-core. Scans all memories grouped
  by category, identifies categories with >= min_activity (default 3) historically but
  zero recent activity within threshold window (default 7 days). Generates nudges:
  "You used to ask about [topic] but haven't in X days. Is it resolved or stuck?"
  Sorted by total activity (highest signal first). Skips poisoned records.
- **Files:** `memu-core/app.py` (`/memory/silence` endpoint)
- **Tests:** `scripts/test_silence_signal.py` — 8 tests (config defaults, active topic
  not flagged, silent topic identified, low activity excluded, nudge format, sort order,
  poisoned records excluded)

### P8: Dashboard — Thinking Pathways Visualization ✅ DONE

> Merge of vLLM-SR transparent routing + our Phase 3.
> "Visualize Kai's brain: route decisions, adversary findings, conviction flow."

- **What was built:**
  - `GET /thinking` — serves `dashboard/static/thinking.html`, a full cognitive transparency page.
  - 5 new proxy API endpoints in `dashboard/app.py`:
    - `GET /api/thinking` — fetches recent episodes from langgraph, extracts conviction pipeline
      data (initial score → rethinks → final conviction → learning value → failure class → metacognitive rule).
    - `GET /api/tempo` — proxies memu-core `/memory/tempo` (operator pace gauge).
    - `GET /api/boundary` — proxies memu-core `/memory/boundary` (knowledge confidence map).
    - `GET /api/silence` — proxies memu-core `/memory/silence` (absence-of-question signals).
    - `GET /api/self-assessment` — proxies heartbeat `/self-assessment` (temporal self-model).
  - `thinking.html` — dark-themed visualization page with 6 cards:
    - **Conviction Pipeline** — visual flow: Input → Initial Score → Rethinks → Final Score → Learning Value.
      Shows metacognitive rules when present. Color-coded (green ≥7.5, yellow ≥5, red <5).
    - **Recent Thinking Episodes** — scrollable list with conviction tags, rethink counts,
      failure classes, learning values. Border colors indicate conviction level.
    - **Operator Tempo** — SVG gauge showing pace (rapid/normal/reflective/idle) with
      distribution bars, average gap time, burst detection.
    - **Knowledge Boundary Map** — confidence bars per topic zone, overall confidence metric.
    - **Silence-as-Signal** — tag cloud of topics where silence was chosen, with reasons.
    - **Temporal Self-Assessment** — metric grid showing memories, error rate, uptime,
      response time, episodes, avg conviction. Delta indicators (▲/▼) for trend direction.
  - Navigation links between Chat, Control Panel, and Thinking pages.
  - Auto-refresh every 15 seconds + manual refresh button.
  - All endpoints gracefully degrade when backend services are unavailable.
- **Files:** `dashboard/app.py` (6 new endpoints), `dashboard/static/thinking.html` (visualization page)
- **Tests:** `scripts/test_thinking_pathways.py` — 21 tests (HTML content, API proxying with mocks,
  graceful degradation, input truncation, episode limiting, page existence checks).

### P9–P15: Remaining Advantages

- **P9: Security Self-Hacking** ✅ — `run_security_audit()` in `langgraph/security_audit.py`.
  4 audit categories: injection filter fuzzing (21 bypass payloads), input sanitization
  (13 adversarial payloads + XSS/SQL/path-traversal/command-injection), HMAC auth
  boundary testing (4 edge cases), policy governance checks (4 env var audits).
  `challenge_security()` wired as adversary challenge #6. Risk score 0.0–1.0
  with severity-weighted findings. Dashboard: `GET /api/security-audit` proxy,
  Security Self-Hacking card in thinking.html with risk gauge and findings list.
  Tests: `scripts/test_security_audit.py` — 23 tests.
- **P10: Predictive Pre-Computation** ✅ — `predict_next_request()` + `mine_request_sequences()`
  in planner.py. Mines bigram sequences from episode history to predict what the operator
  will ask next. Pre-fetches memory context for top-3 predictions via `pre_fetch_predicted_context()`.
  Wired into langgraph/app.py: predictions added to plan as `predicted_next` metadata.
  Tests: `scripts/test_predictive.py` — 15 tests.
- **P11: Operator Tempo Modeling** ✅ — `GET /memory/tempo` in memu-core/app.py.
  Analyses memory timestamps to detect operator pace. Four categories: rapid (<30s),
  normal (30s-5min), reflective (5-30min), idle (>30min). Returns dominant tempo,
  style_hint for response adaptation, gap distribution, avg/median gaps, burst detection.
  Tests: `scripts/test_tempo.py` — 12 tests.
- **P12: Self-Deception Detection** ✅ — `detect_self_deception()` in conviction.py.
  3 checks: evidence_gap (high conviction but < 2 chunks), relevance_gap (chunks exist
  but coverage < 0.5), rethink_blind_spot (complex query >= 15 words with zero rethinks).
  Wired into langgraph/app.py: if deceived, forces conviction below MIN_CONVICTION to
  trigger rethink. Tests: `scripts/test_self_deception.py` — 12 tests.
- **P13: Recursive Self-Improvement Gate** ✅ — `capture_snapshot()`, `evaluate_improvement()`
  in kai_config.py. Before self-modification, snapshots avg_conviction, avg_outcome,
  failure_rate, rethink_rate. After change, compares metrics against tolerance (default 0.1).
  If any metric degrades beyond tolerance → flags for revert. Auto-snapshots every 10
  episodes in langgraph/app.py. File-based snapshot persistence (last 50 snapshots).
  Tests: `scripts/test_improvement_gate.py` — 20 tests.
- **P14: Temporal Self-Model** ✅ — `GET /self-assessment` in heartbeat/app.py.
  Compares current vs previous period metrics (total_memories, error_rate, uptime_ratio,
  cpu_usage). Trend detection via `_trend()` with inverted labels for error metrics.
  Overall health: needs_attention (2+ declining), improving (2+ improving), stable.
  File-based persistence for period-over-period comparison.
  Tests: `scripts/test_temporal_self.py` — 19 tests.
- **P15: Dream State** ✅ — `run_dream_cycle()` in `langgraph/kai_config.py`.
  6-phase offline consolidation: cluster failures by class, deduplicate metacognitive
  rules (Jaccard similarity threshold), synthesize recurring patterns (struggling topics,
  complex topics, learning trends), detect rule contradictions (always/never conflicts),
  recalibrate knowledge boundary, package insights. `DreamCycle` + `DreamInsight`
  dataclasses with file-based persistence (last 20 cycles). Dashboard: `POST /api/dream`
  proxy, Dream State card in thinking.html with trigger button and insight visualization.
  Tests: `scripts/test_dream_state.py` — 26 tests.

### Parked (Future Phases)

- **OMAR Self-Play (research #2/#13):** Internal attacker/defender/judge loop.
  Requires multi-turn LLM reasoning = expensive in tokens. **Unblocked when
  RTX 5080 laptop arrives** — run Kimi K2 (32B active) locally with spare
  capacity for self-play loops. Our rule-based adversary.py covers the
  critical path at near-zero cost until then.
- **ZK Privacy Learning (research #10):** ZK-proof infrastructure for safe
  online learning. Heavy crypto dependency. North star, not buildable this month.
- **ReCiSt Bio-Resilience (research #4):** Containment → Diagnosis →
  Meta-Cognitive → Knowledge. We ALREADY HAVE this: circuit breakers +
  error budgets + memory + adversary. Document the mapping, don't rebuild.

---

## Hardware Performance Track (GPU Arrival)

> Unlocked when RTX 5080 laptop arrives. All items are infrastructure-level
> optimizations — they make Kai faster and more efficient, not smarter.
> Intelligence features (P1–P15) are separate.

### HP1: Speculative Decoding (Ollama)
- **What:** Small draft model (1B) predicts tokens, large model verifies. 2–3x throughput.
- **How:** Ollama `--draft-model` flag or API parameter. Zero code change.
- **Impact:** High — biggest single speed win for interactive chat.
- **Status:** Ready to deploy on hardware arrival.

### HP2: MoE Model Selection ✅ DONE
- **What:** MoE-aware model selector routes queries to the best available model based on
  task type, complexity, and hardware constraints.
- **Built:** `langgraph/model_selector.py` — `ModelProfile` dataclass (name, strengths,
  max_context, speed/quality tiers, moe_expert_count, vram_gb), `_PROFILES` registry
  (Ollama, DeepSeek-V4, Kimi-2.5, Dolphin), `estimate_complexity()` (keywords + length +
  sentences + question marks → 0.0–1.0), `select_model()` (route match +3, quality premium
  for complex, speed bonus, MoE bonus, context window bonus). `register_model()` for
  dynamic model addition. Wired into `/chat` and `/run` endpoints — replaces hardcoded
  `_DEFAULT_SPECIALIST` with intelligent route-aware selection. `GET /models` endpoint
  exposes available models and profiles.
- **Impact:** High — right model for each task automatically. MoE models score higher
  on complex tasks, fast models preferred for chat.
- **Tests:** `scripts/test_model_selector.py` — 20 tests (complexity estimation, selection,
  profile management, MoE bonus, speed preference).
- **Status:** Software routing built and wired. Models deploy on hardware arrival.

### HP3: VRAM Watchdog + Adaptive Quantization
- **What:** Heartbeat monitors GPU VRAM. If utilization >90%, unload LRU model and
  reload lower quantization variant (e.g. Q8 → Q4). If below 50%, upgrade to higher quant.
- **How:** Add `nvidia-smi` polling to heartbeat, Ollama `/api/ps` for loaded models,
  Ollama `/api/pull` with quantization suffix.
- **Impact:** Medium — prevents OOM crashes, graceful degradation under load.
- **Status:** Design ready, heartbeat has the monitoring hooks.

### HP4: CoT Tree Search with Conviction Pruning ✅ DONE
- **What:** Chain-of-thought with branching. Generate multiple reasoning paths, score each
  with conviction, prune low-confidence branches early, refine survivors.
- **Built:** `langgraph/tree_search.py` — `Branch` dataclass (id, plan, prompt, conviction,
  depth, pruned), `TreeSearchResult` (best_branch, total/pruned branches, search_time_ms,
  improvement property), `tree_search()` async function (generates N variations, scores via
  conviction, prunes below threshold, refines survivors, early exit if min_conviction met),
  `_generate_variations()` (4 prompt suffixes: baseline, edge-cases, risks, expert),
  `_branch_id()` deterministic SHA256. Wired into `/run` as fallback after linear rethink
  loop exhaustion — 3 branches × 2 depth, updates plan and conviction if improvement found.
- **Impact:** Medium-High — better reasoning on complex queries via multi-path exploration.
- **Tests:** `scripts/test_tree_search.py` — 14 tests (branch creation, variations, tree
  search scenarios, early exit, pruning, fetch chunks).
- **Status:** Built and wired. Fully operational with current models; GPU models will
  produce higher-quality branches.

### HP5: Priority Queue for Inference ✅ DONE
- **What:** Async priority queue for LLM request scheduling. High-priority requests (chat)
  get resources first. Low-priority (dream, batch) queue behind.
- **Built:** `langgraph/priority_queue.py` — `Priority` IntEnum (CHAT=0, RUN=1,
  BACKGROUND=2, BATCH=3), `QueueEntry` with `__lt__` for ordering (lower number = higher
  priority, ties broken by submission time), `PriorityQueue` class (semaphore-based
  concurrency limiting via `submit()` async, `stats()` snapshot), `get_queue()` module-level
  singleton. `GET /queue/stats` endpoint exposes queue depth, in-flight tasks, and priority
  breakdown.
- **Impact:** Medium — ensures chat responsiveness under load, dream cycles don't block
  interactive requests.
- **Tests:** `scripts/test_priority_queue.py` — 12 tests (priority ordering, queue entries,
  submit/result, concurrency, exception propagation, singleton).
- **Status:** Built and wired. GPU preemption hooks ready for hardware arrival.

### HP6: Partial Layer Loading
- **What:** Load N model layers to GPU, keep rest in RAM. Ollama `-ngl` parameter.
- **How:** Set `OLLAMA_NUM_GPU` env var. Already supported.
- **Impact:** Low-Medium — useful for running models larger than VRAM.
- **Status:** Ready (Ollama native feature).

### Parked Hardware Items
- **vLLM backend:** Paged KV-cache + continuous batching. High throughput but adds
  complexity vs Ollama. Revisit if Ollama becomes a bottleneck.
- **NVMe layer offload:** `accelerate` disk_offload for models exceeding RAM+VRAM.
  OS swap handles this adequately for now.
- **FlashAttention-2:** Modern backends (Ollama, vLLM) use it by default. No action needed.
- **Quantum-inspired embeddings:** Complex-valued vectors, Bell/CHSH checks, Qiskit surrogates.
  Interesting research but unproven in production. No measurable advantage over standard
  embeddings in benchmarks. Revisit when peer-reviewed results appear.
- **Multi-GPU tensor parallelism:** For future multi-GPU setups. Single RTX 5080 for now.

---

## The Ultimate Compound

Phase 2 gave Kai: routing, planning, self-challenge, calibration.
Merged plan gives Kai: self-awareness, evolution, identity, honesty, adaptation.

```
Phase 2 (built):  "I check my plans before acting"
Merged plan:      "I know WHY things fail and extract rules from it.
                   I learn MORE from uncertain failures than safe successes.
                   Every memory I write is signed and provably mine.
                   I detect when I contradict myself or when tools disagree.
                   I model how my operator thinks, not just what he says.
                   I map my own ignorance and probe it during idle time.
                   I notice when topics go silent and ask about them.
                   I measure whether I'm getting better and tell you if I'm not."
```

No other build thinks about itself this way. This is the gap.

---

## Continuation Notes

> **For AI assistants resuming work. Updated 5 March 2026.**

### Quick Start
1. `git log --oneline -5` — check latest commits
2. `make test-core` — confirm baseline green (expect 51 passes, 347 individual tests)
3. Read this doc top-to-bottom for strategic context
4. Read `docs/agentic_patterns_spec.md` for technical spec of Phase 2

### What's Done
- **Phase 2 COMPLETE**: router.py, planner.py, adversary.py — all built, tested, wired into /chat and /run
- **P1–P15 COMPLETE** (except P3 Soulbound Identity — skipped, HMAC covers it until TPM hardware)
- **HP2, HP4, HP5 COMPLETE**: model_selector.py, tree_search.py, priority_queue.py — built, tested, wired
- **51 test-core targets passing**, 347 individual tests, zero failures
- **6 adversary challenges** (challenge_plan includes security self-hacking as #6)
- **Strategic doc merged**: Operator's 2026 research + AI-native blueprints → unified priority order above
- **Hardware Performance Track** documented (HP1–HP6) — HP2/HP4/HP5 built; HP1/HP3/HP6 ready for RTX 5080

### What's Next (priority order)
1. **J2: Wake-word "Kai" + Intent Judge** ⭐ — whisper keyword-spot + tiny LLM intent classifier
2. **J1: Live Canvas Visualization** — mind-map/graph/timeline in dashboard
3. **J6: SOUL.md + AGENTS.md** — persistent identity files
4. **J3: Auto-Redaction PII** — regex + OCR strip before processing
5. **J5: Memory Viewer GUI** — diary-style dashboard tab
6. **J4: Proactive Low-Latency Voice** — audio/video cue → speak-or-not
7. **J7: Skills Auto-Install Hub** — local skill loader
8. **HP1: Speculative Decoding** — deploy on hardware arrival (Ollama `--draft-model` flag)
9. **HP3: VRAM Watchdog** — add GPU monitoring to heartbeat
10. **OMAR Self-Play** — unblocked when GPU enables multi-turn reasoning

### Key File Map
- `langgraph/router.py` — 8-route zero-LLM classifier
- `langgraph/planner.py` — memory-driven planning with episode similarity
- `langgraph/adversary.py` — 6-challenge stress-test engine
- `langgraph/conviction.py` — 5-signal conviction scoring (0-10)
- `langgraph/kai_config.py` — episode storage, failure taxonomy, dream state, improvement gate
- `langgraph/security_audit.py` — automated security self-hacking engine
- `langgraph/model_selector.py` — MoE-aware model selection (HP2)
- `langgraph/tree_search.py` — CoT tree search with conviction pruning (HP4)
- `langgraph/priority_queue.py` — async priority queue for inference (HP5)
- `langgraph/app.py` — main orchestrator: /chat, /run, /dream, /security/audit
- `common/llm.py` — LLM router (Ollama, DeepSeek, Kimi-2.5, Dolphin; Kimi-K2 pending)
- `memu-core/app.py` — 21+ endpoint memory engine
- `dashboard/static/thinking.html` — cognitive transparency visualization

### LLM Strategy
ALL models run locally via Ollama or OpenAI-compatible endpoints. No cloud.
- Default: Ollama (qwen2 or any pulled model)
- Specialist routing via `common/llm.py` LLMRouter
- Kimi K2 (1T MoE, 32B active, Apache 2.0) pending addition for agentic tasks
- HP1: Speculative decoding for 2-3x throughput on arrival
- HP2: MoE models (DeepSeek-V3, Kimi K2, Qwen3-MoE) for better quality/VRAM
