# Kai Code Audit — Conviction, Planning and Forecasting Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-DECIDE-001 | CRITICAL | Conviction can exceed the execution threshold with zero supporting evidence |
| KAI-DECIDE-002 | HIGH | The stock two-step plan receives near-maximum specificity credit |
| KAI-DECIDE-003 | HIGH | Query length, punctuation and generic keywords inflate conviction without increasing certainty |
| KAI-DECIDE-004 | HIGH | Rethink count increases conviction without verifying that the plan improved |
| KAI-DECIDE-005 | HIGH | Context support is based on any three-character keyword overlap |
| KAI-DECIDE-006 | HIGH | Duplicate and poisoned memory chunks increase context coverage |
| KAI-DECIDE-007 | HIGH | Specialist fit is manipulable through substring keyword stuffing |
| KAI-DECIDE-008 | HIGH | Domain confidence is one global mutable value shared across requests and users |
| KAI-DECIDE-009 | MEDIUM | Conviction inputs and intermediate values lack finite/range validation |
| KAI-DECIDE-010 | MEDIUM | Empty or failed model output can leave an already-high conviction effectively intact |
| KAI-DECIDE-011 | HIGH | Planner hard-codes the `keeper` memory identity and ignores session isolation |
| KAI-DECIDE-012 | HIGH | Caller-provided episode scores are trusted as historical outcomes |
| KAI-DECIDE-013 | HIGH | Fabricated past success can boost conviction and inject a previous output into the new plan |
| KAI-DECIDE-014 | HIGH | Retrieved corrections and preferences become trusted plan constraints without provenance boundaries |
| KAI-DECIDE-015 | HIGH | Correction status is inferred from a substring in general-category memory text |
| KAI-DECIDE-016 | HIGH | Dependency failure is indistinguishable from an evidence-free clean history |
| KAI-DECIDE-017 | MEDIUM | Episode similarity rewards shared common words rather than semantic or causal relevance |
| KAI-DECIDE-018 | MEDIUM | Future timestamps are treated as more recent than current history |
| KAI-DECIDE-019 | MEDIUM | Invalid episode timestamps and scores can abort planning |
| KAI-DECIDE-020 | MEDIUM | Planner input, history and downstream response complexity are unbounded |
| KAI-DECIDE-021 | MEDIUM | New HTTP clients are created for each planning dependency call |
| KAI-DECIDE-022 | MEDIUM | Predictive topic keys discard word order and retain only three alphabetically sorted words |
| KAI-DECIDE-023 | HIGH | Cross-session episode history can create behavioural predictions and memory prefetches for the wrong user |
| KAI-DECIDE-024 | MEDIUM | Sequence thresholds and prefetch parameters are not validated |
| KAI-DECIDE-025 | HIGH | Forecasting silently substitutes fixed canned scenarios for failed or absent inference |
| KAI-DECIDE-026 | HIGH | Four duplicate scenario labels can satisfy the forecast parser |
| KAI-DECIDE-027 | HIGH | Forecast probabilities are not checked for range, finiteness or total probability |
| KAI-DECIDE-028 | HIGH | Claims, causal chains and query text are inserted directly into the forecasting prompt |
| KAI-DECIDE-029 | MEDIUM | `consensus_probability` is merely the caller/model-provided base-scenario probability |
| KAI-DECIDE-030 | MEDIUM | Fallback scenario objects are globally shared mutable instances |
| KAI-DECIDE-031 | MEDIUM | Forecast model calls have no enforced timeout or cancellation contract |
| KAI-DECIDE-032 | HIGH | Counterfactual rehearsal is a non-functional stub for a claimed high-stakes safety step |
| KAI-DECIDE-033 | MEDIUM | Counterfactual input size and simulation step count are unbounded |
| KAI-DECIDE-034 | MEDIUM | Counterfactual output exposes world-state key names despite performing no simulation |

---

## Conviction scoring: `agentic/conviction.py`

### KAI-DECIDE-001 — CRITICAL — Evidence-free execution-threshold bypass
**Issue:** With no context chunks, the scoring formula can still exceed `MIN_CONVICTION = 8.0`. The generic plan can score about 1.7, a long keyword-rich query 2.0, three rethinks 1.5, specialist fit 2.0 and neutral domain confidence 1.0, producing approximately 8.2 while context coverage is zero.  
**Risk:** A consequential plan can pass the execution conviction threshold without any supporting memory or external evidence. The gate measures presentation features and iteration count rather than justified belief.  
**Recommendation:** Require a non-bypassable evidence/provenance floor for consequential actions and calibrate conviction from verified outcomes rather than additive stylistic heuristics.  
**Status:** OPEN — immediate remediation required

### KAI-DECIDE-002 — HIGH — Generic plan structure is over-rewarded
**Issue:** `build_plan` always emits `analyze` and `propose` steps plus a summary and specialist name. `_plan_specificity` awards 0.5 for two steps, 0.5 for all steps having actions, 0.4 for the summary and 0.3 for the specialist: approximately 1.7/2 before the plan contains any task-specific method, constraint, validation or rollback.  
**Risk:** Boilerplate plan formatting creates nearly maximum specificity and materially lifts execution confidence.  
**Recommendation:** Score concrete verified parameters, dependencies, checks, safety controls and task-specific actions rather than schema presence.  
**Status:** OPEN

### KAI-DECIDE-003 — HIGH — Verbosity is treated as clarity
**Issue:** `_query_clarity` increases score mainly from word count. Long text receives 1.5 and gains bonuses for a question mark and broad words such as `file`, `run`, `build` or `error`.  
**Risk:** Padding or keyword stuffing raises conviction even when the request is ambiguous, contradictory or malicious.  
**Recommendation:** Evaluate typed completeness, contradictions, missing prerequisites and ambiguity; never use length as a confidence proxy.  
**Status:** OPEN

### KAI-DECIDE-004 — HIGH — Repetition is treated as improvement
**Issue:** `_rethink_improvement` adds up to 1.5 points solely from `rethink_count`. No before/after plan comparison, adversarial correction or evidence gain is required.  
**Risk:** Repeating an unchanged or worse plan can push it over the execution threshold.  
**Recommendation:** Grant improvement credit only from measured defect resolution or independently verified evidence gain.  
**Status:** OPEN

### KAI-DECIDE-005 — HIGH — Weak overlap becomes evidence coverage
**Issue:** A memory chunk is relevant when any token of three or more characters overlaps with the request. Source, proposition, contradiction and relationship to the planned action are ignored.  
**Risk:** Generic words make unrelated memories appear supportive and increase conviction.  
**Recommendation:** Score claim-level entailment with provenance, independence and contradiction handling.  
**Status:** OPEN

### KAI-DECIDE-006 — HIGH — Duplicate evidence is counted repeatedly
**Issue:** Coverage is the fraction of chunks with any overlap. Chunks are not deduplicated or weighted by source independence/integrity.  
**Risk:** Repeated or poisoned memory records create apparent broad support and can maximise coverage.  
**Recommendation:** Semantically deduplicate evidence and require independent trusted sources.  
**Status:** OPEN

### KAI-DECIDE-007 — HIGH — Specialist score is keyword-gameable
**Issue:** `_specialist_fit` counts raw substrings from a short specialist-domain dictionary. Three matching words produce the maximum score regardless of the actual specialist, task or model capability.  
**Risk:** Crafted phrasing can add two conviction points and steer the decision path.  
**Recommendation:** Bind specialist selection to validated capabilities and risk-aware intent classification.  
**Status:** OPEN

### KAI-DECIDE-008 — HIGH — Cross-request domain-confidence contamination
**Issue:** `_active_domain_confidence` is a module-global float. `update_domain_confidence` changes it for every concurrent and subsequent request, with no user/domain/session key or lock.  
**Risk:** One request’s domain history changes another user or task’s conviction; concurrent calls race and worker processes diverge.  
**Recommendation:** Pass immutable domain-confidence evidence explicitly per request.  
**Status:** OPEN

### KAI-DECIDE-009 — MEDIUM — Numerical states are weakly validated
**Issue:** Plans, rethink counts, chunk values and confidence inputs are ordinary dictionaries/numbers. Non-finite and extreme values are not consistently rejected at this boundary.  
**Risk:** NaN, infinity, negative counts or malformed plans can distort comparisons and produce non-portable JSON/state.  
**Recommendation:** Validate typed finite ranges and legal plan schemas.  
**Status:** OPEN

### KAI-DECIDE-010 — MEDIUM — Output failure has limited effect on high base confidence
**Issue:** An empty LLM response receives an uncertainty penalty of 0.5; response refinement is bounded to at most a one-point reduction.  
**Risk:** A plan already above the threshold can remain high-confidence despite no usable model output.  
**Recommendation:** Treat empty/error output as a hard failed decision stage, not a small stylistic adjustment.  
**Status:** OPEN

---

## Memory-driven planner: `agentic/planner.py`

### KAI-DECIDE-011 — HIGH — Memory identity and session boundaries are collapsed
**Issue:** Memory retrieval always uses `user_id="keeper"`; preferences and nudges are global. `session_id` is stored in `PlanContext` but does not scope any retrieval or episode comparison.  
**Risk:** Memories, preferences and behavioural history can cross sessions/users and alter plans for the wrong principal.  
**Recommendation:** Bind every planning source to an authenticated principal and session/purpose boundary.  
**Status:** OPEN

### KAI-DECIDE-012 — HIGH — Historical outcome data is accepted as truth
**Issue:** The planner accepts arbitrary episode dictionaries and directly converts `outcome_score`, conviction, timestamps, inputs and outputs. No signed source or verified outcome is required.  
**Risk:** Poisoned episode history becomes evidence of success/failure and changes future conviction.  
**Recommendation:** Accept only immutable outcome records produced by an authenticated evaluator.  
**Status:** OPEN

### KAI-DECIDE-013 — HIGH — Fabricated success changes the new plan
**Issue:** A similar episode with outcome at least 0.7 and conviction at least 7 adds up to +1.0 conviction and, at similarity 0.5, inserts its prior output into a `reference_past` plan step.  
**Risk:** One poisoned record can both raise the gate score and inject attacker-controlled instructions/content into the current plan.  
**Recommendation:** Verify applicability, provenance and outcome before reuse; quote prior output as untrusted evidence, never as an instruction.  
**Status:** OPEN

### KAI-DECIDE-014 — HIGH — Retrieved text becomes privileged planning constraints
**Issue:** Correction memories and operator preferences are copied directly into `apply_correction` and `apply_preference` steps. No provenance, trust level, instruction/data separation or current-user validation exists.  
**Risk:** Stored prompt injection or stale preferences directly modify the action plan.  
**Recommendation:** Use signed typed preference/correction records and enforce policy-authorised fields only.  
**Status:** OPEN

### KAI-DECIDE-015 — HIGH — Correction classification is a substring test
**Issue:** General-category memory is treated as a correction when the result text contains the word `correction`, or when a caller-provided event type equals correction.  
**Risk:** Ordinary or malicious text can become a planning constraint and conviction penalty.  
**Recommendation:** Require a dedicated authenticated correction schema linked to the corrected outcome.  
**Status:** OPEN

### KAI-DECIDE-016 — HIGH — Dependency outage looks like clean evidence
**Issue:** Memory, correction, nudge and preference helpers catch every exception and return empty lists. `gather_context` exposes no unavailable/stale/partial state.  
**Risk:** Plans proceed as though no relevant warning or correction exists precisely when the memory authority failed.  
**Recommendation:** Represent source failure explicitly and fail/escalate consequential planning when required evidence is unavailable.  
**Status:** OPEN

### KAI-DECIDE-017 — MEDIUM — Similarity is semantically weak
**Issue:** Episode matching uses Jaccard overlap of unique tokens of length at least three. It ignores order, negation, entities, action direction and duplicate/common-word effects.  
**Risk:** Unrelated requests can inherit past success, failure and outputs.  
**Recommendation:** Use verified semantic/task identity and explicit applicability checks.  
**Status:** OPEN

### KAI-DECIDE-018 — MEDIUM — Future history is ranked as newest
**Issue:** `age_days` is calculated from unvalidated timestamps. A future timestamp becomes negative and sorts ahead of legitimate recent outcomes.  
**Risk:** Fabricated/faulty future events dominate reuse and conviction decisions.  
**Recommendation:** Validate event time, source clock and acceptable skew.  
**Status:** OPEN

### KAI-DECIDE-019 — MEDIUM — Malformed episode numerics abort planning
**Issue:** `float()` conversions for timestamps, conviction and outcome scores are not protected per episode.  
**Risk:** One malformed historical record can terminate context planning rather than being quarantined and surfaced as an integrity error.  
**Recommendation:** Validate records at ingestion and fail the history source explicitly on corruption.  
**Status:** OPEN

### KAI-DECIDE-020 — MEDIUM — Planning data is unbounded
**Issue:** User input, session IDs, episode arrays, nested episode fields and downstream JSON responses have no aggregate size/depth limits in this module.  
**Risk:** Large history or service payloads consume CPU/memory during matching and plan construction.  
**Recommendation:** Enforce strict typed body, history, response and text limits.  
**Status:** OPEN

### KAI-DECIDE-021 — MEDIUM — Planning recreates HTTP clients
**Issue:** Each memory, correction, nudge and preference fetch creates a separate `AsyncClient`.  
**Risk:** Every plan causes avoidable connection-pool/socket churn.  
**Recommendation:** Reuse lifecycle-managed bounded clients.  
**Status:** OPEN

### KAI-DECIDE-022 — MEDIUM — Predictive topic identity destroys meaning
**Issue:** `_extract_topic_key` takes unique words, sorts them alphabetically and retains the first three. Sequence and salience are discarded.  
**Risk:** Distinct requests collapse to the same topic while meaningful later words disappear, producing spurious next-request predictions.  
**Recommendation:** Use a validated per-user task taxonomy or semantic representation.  
**Status:** OPEN

### KAI-DECIDE-023 — HIGH — Behavioural prediction crosses user/session boundaries
**Issue:** Sequence mining operates on the supplied global episode list and prefetches `keeper` memories for predicted topics. No authenticated principal partition exists.  
**Risk:** One user’s interaction sequence can predict and preload private context into another user’s planning flow.  
**Recommendation:** Partition sequence models and prefetches by principal, session and consent.  
**Status:** OPEN

### KAI-DECIDE-024 — MEDIUM — Predictive configuration and prefetch are weakly governed
**Issue:** Minimum support, probability threshold and prefetch `top_k` are not validated. Predicted memory calls run sequentially, accept unbounded JSON and suppress every error.  
**Risk:** Invalid thresholds create noisy/uncontrolled prediction, while prefetch failure is invisible.  
**Recommendation:** Validate bounded policy and publish prefetch freshness/error/provenance.  
**Status:** OPEN

---

## Temporal forecasting: `agentic/forecaster.py`

### KAI-DECIDE-025 — HIGH — Canned output is substituted for forecasting
**Issue:** If no LLM is supplied, parsing fails or the LLM call raises, every query receives the same four narratives and probabilities (0.50/0.25/0.20/0.05). The result remains a normal `ForecastFan`, differentiated only by `used_llm=False`.  
**Risk:** Generic constants can be consumed/displayed as evidence-derived future probabilities.  
**Recommendation:** Return an explicit unavailable/not-forecasted state; never attach fabricated probabilities to the query.  
**Status:** OPEN

### KAI-DECIDE-026 — HIGH — Scenario completeness check accepts duplicates
**Issue:** The parser accepts any four valid-label objects. It does not require exactly one base, optimistic, pessimistic and wild-card branch.  
**Risk:** Four duplicate base branches pass as a complete fan; missing risk branches remain undetected.  
**Recommendation:** Validate a unique exact label set.  
**Status:** OPEN

### KAI-DECIDE-027 — HIGH — Probabilities are not probabilities
**Issue:** Parsed values are converted to float but not checked for finite range or sum. NaN, infinity, negative values, values above one and totals far from one are accepted.  
**Risk:** Invalid numerical forecasts can enter downstream conviction/synthesis and JSON.  
**Recommendation:** Reject non-finite/out-of-range values and normalise only after an explicit calibrated model contract.  
**Status:** OPEN

### KAI-DECIDE-028 — HIGH — Forecast prompt injection
**Issue:** Query, supported claims and causal chains are concatenated directly into the user prompt. No provenance or instruction/data boundary exists.  
**Risk:** Poisoned claims can change the requested schema, labels or scenario content.  
**Recommendation:** Use structured bounded fields and independently validate every output proposition.  
**Status:** OPEN

### KAI-DECIDE-029 — MEDIUM — Base probability is mislabeled consensus
**Issue:** `consensus_probability` simply returns the first branch labelled base, or 0.5. No consensus process or evidence aggregation occurs.  
**Risk:** Downstream consumers may treat one generated number as multi-source agreement.  
**Recommendation:** Rename it base-scenario probability and provide calibration/provenance.  
**Status:** OPEN

### KAI-DECIDE-030 — MEDIUM — Global fallback objects are mutable
**Issue:** `list(_FALLBACK_BRANCHES)` copies only the list; each `ScenarioBranch` and its assumptions list remain shared objects.  
**Risk:** A caller modifying one fallback forecast changes later forecasts process-wide.  
**Recommendation:** construct fresh immutable objects per result.  
**Status:** OPEN

### KAI-DECIDE-031 — MEDIUM — Injected model execution is unbounded
**Issue:** The forecaster awaits the supplied LLM callback directly with no timeout, token limit or cancellation policy.  
**Risk:** A hung backend stalls the caller and can consume worker capacity.  
**Recommendation:** enforce a bounded model contract and deadline.  
**Status:** OPEN

---

## Counterfactual rehearsal: `agentic/counterfactual.py`

### KAI-DECIDE-032 — HIGH — Claimed high-stakes safety mechanism is a stub
**Issue:** `rehearse` always returns no scenarios, no recommendation and zero confidence; `can_rehearse` always returns false. The module is imported by the active agentic application and described as a pre-decision high-stakes rehearsal foundation.  
**Risk:** Architecture/documentation can imply a counterfactual safety layer exists while it performs no analysis. Any path that treats interface presence as completion receives no protection.  
**Recommendation:** Keep the control explicitly unavailable and ensure every high-stakes caller fails/escalates when rehearsal is required.  
**Status:** OPEN

### KAI-DECIDE-033 — MEDIUM — Rehearsal inputs are unbounded
**Issue:** Decision text, nested world state and `steps` have no type depth, length or range enforcement beyond Python hints.  
**Risk:** Large or malformed state can consume memory and produce misleading requested-step metadata even though no simulation occurs.  
**Recommendation:** define strict schemas and safe simulation limits before implementation.  
**Status:** OPEN

### KAI-DECIDE-034 — MEDIUM — Stub leaks world-model structure
**Issue:** The response returns every top-level `world_state` key despite using none of the values.  
**Risk:** Callers receive internal world-model field names without any analytical benefit.  
**Recommendation:** return only an unavailable status and opaque trace ID.  
**Status:** OPEN

---

## Batch totals

- Findings: **34**
- Critical: **1**
- High: **17**
- Medium: **16**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **857**
- Critical: **87**
- High: **316**
- Medium: **451**
- Low: **3**

## Files materially reviewed in this batch

`agentic/conviction.py`, `agentic/planner.py`, `agentic/forecaster.py`, `agentic/counterfactual.py`, with active-path confirmation against `agentic/app.py`.
