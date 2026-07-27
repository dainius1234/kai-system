# Kai Code Audit — Model and Decision Controls Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MODELCTRL-001 | HIGH | An explicitly empty available-model list expands to every registered model |
| KAI-MODELCTRL-002 | HIGH | A single available candidate is returned without registry or readiness validation |
| KAI-MODELCTRL-003 | HIGH | Unknown/unscored candidates silently fall back to Ollama regardless of availability |
| KAI-MODELCTRL-004 | HIGH | Multiple conflicting model registries create non-authoritative capability decisions |
| KAI-MODELCTRL-005 | MEDIUM | Complexity scoring rewards verbosity, punctuation and keyword stuffing |
| KAI-MODELCTRL-006 | MEDIUM | Long-context selection uses character length rather than model token count |
| KAI-MODELCTRL-007 | HIGH | Runtime code can overwrite model profiles without governance |
| KAI-MODELCTRL-008 | MEDIUM | Model profiles lack range, type and cross-field validation |
| KAI-MODELCTRL-009 | MEDIUM | Ranking ties are resolved by mutable registry insertion order |
| KAI-MODELCTRL-010 | MEDIUM | Model selection does not verify a live backend or exact artefact |
| KAI-MODELCTRL-011 | HIGH | Model Council trust enforcement fails open |
| KAI-MODELCTRL-012 | HIGH | Built-in models start as available without discovery or credentials |
| KAI-MODELCTRL-013 | HIGH | Static heuristics are persisted as successful benchmark results |
| KAI-MODELCTRL-014 | HIGH | Injected benchmark scores are unbounded and unauthenticated |
| KAI-MODELCTRL-015 | HIGH | `set_primary` performs no trust or operator-authorisation check |
| KAI-MODELCTRL-016 | HIGH | Any internal caller can reset model availability through success/failure recording |
| KAI-MODELCTRL-017 | HIGH | Model Council reports successful mutation after persistence failure |
| KAI-MODELCTRL-018 | HIGH | Profile and primary persistence is concurrency-unsafe |
| KAI-MODELCTRL-019 | HIGH | Persisted primary identity can refer to an unavailable or invalid model |
| KAI-MODELCTRL-020 | MEDIUM | Custom persisted profiles are accepted without semantic validation |
| KAI-MODELCTRL-021 | MEDIUM | Built-in profile objects are shared across Council resets |
| KAI-MODELCTRL-022 | MEDIUM | Unknown task types silently become chat rankings |
| KAI-MODELCTRL-023 | MEDIUM | Failover mutates the caller’s exclusion set |
| KAI-MODELCTRL-024 | MEDIUM | Failure counters are global and not scoped by task, endpoint or failure class |
| KAI-MODELCTRL-025 | MEDIUM | Provider, cost and model capability metadata is static and unverified |
| KAI-MODELCTRL-026 | HIGH | Verifier unavailability is recorded as a passed adversarial challenge |
| KAI-MODELCTRL-027 | HIGH | Any unknown verifier verdict is interpreted as PASS and adds conviction |
| KAI-MODELCTRL-028 | HIGH | Tool Gate unavailability is recorded as a passed policy challenge |
| KAI-MODELCTRL-029 | HIGH | Missing Tool Gate mode defaults to WORK |
| KAI-MODELCTRL-030 | HIGH | Security-audit failure is recorded as a passed challenge |
| KAI-MODELCTRL-031 | HIGH | The security challenge disappears entirely when no injection regex is supplied |
| KAI-MODELCTRL-032 | HIGH | Superficial plan structure receives a positive consistency modifier |
| KAI-MODELCTRL-033 | HIGH | History and calibration challenges trust caller-controlled episode outcomes |
| KAI-MODELCTRL-034 | MEDIUM | History challenge mutates caller-owned episode dictionaries |
| KAI-MODELCTRL-035 | MEDIUM | Policy challenge recognises only three hard-coded risky tool names |
| KAI-MODELCTRL-036 | MEDIUM | Claimed under-confidence increases conviction from weak historical matching |
| KAI-MODELCTRL-037 | MEDIUM | Adversary scores and episode values lack finite/range validation |
| KAI-MODELCTRL-038 | MEDIUM | Internal verifier, policy and security errors are exposed in findings |
| KAI-MODELCTRL-039 | HIGH | Claimed independent challenges share the same memory/history assumptions |
| KAI-MODELCTRL-040 | HIGH | Tree search generates prompt suffixes, not alternative reasoned outputs |
| KAI-MODELCTRL-041 | HIGH | Tree search returns the first branch reaching threshold instead of the best branch |
| KAI-MODELCTRL-042 | HIGH | Branch scores are inflated by added prompt wording rather than reasoning quality |
| KAI-MODELCTRL-043 | HIGH | Extra context fetched for one survivor contaminates later branches |
| KAI-MODELCTRL-044 | HIGH | When every branch is pruned, one pruned branch is reinstated anyway |
| KAI-MODELCTRL-045 | HIGH | Failed debate only adds metadata and does not block or replace the branch |
| KAI-MODELCTRL-046 | MEDIUM | Debate evaluates only the already-selected best branch |
| KAI-MODELCTRL-047 | MEDIUM | Counterargument is scored using an unknown `debate` specialist, biasing the margin |
| KAI-MODELCTRL-048 | MEDIUM | Recycled prompt variations create duplicate branch IDs and pseudo-diversity |
| KAI-MODELCTRL-049 | MEDIUM | Search and debate thresholds lack finite and relationship validation |
| KAI-MODELCTRL-050 | MEDIUM | Context-fetch failures are silently ignored during refinement |
| KAI-MODELCTRL-051 | HIGH | PriorityQueue never enqueues or orders requests by priority |
| KAI-MODELCTRL-052 | HIGH | Advertised preemption is not implemented |
| KAI-MODELCTRL-053 | HIGH | The submitted priority value has no effect on scheduling |
| KAI-MODELCTRL-054 | HIGH | Waiting submissions are unbounded and create unrestricted suspended coroutines |
| KAI-MODELCTRL-055 | MEDIUM | The `_pending` queue is never populated or consumed |
| KAI-MODELCTRL-056 | MEDIUM | Reported pending count is actually occupied semaphore capacity |
| KAI-MODELCTRL-057 | MEDIUM | Per-priority statistics are always empty |
| KAI-MODELCTRL-058 | MEDIUM | First singleton construction permanently fixes concurrency configuration |
| KAI-MODELCTRL-059 | MEDIUM | Queue waits and task execution have no deadline or cancellation policy |
| KAI-MODELCTRL-060 | MEDIUM | Task IDs are collision-prone and unused for control or observability |

---

## Model selector: `agentic/model_selector.py`

### KAI-MODELCTRL-001 — HIGH — Empty availability expands to all models
**Issue:** `candidates = available if available else list(_PROFILES.keys())`. An explicitly empty list, meaning no live model is available, is treated the same as `None` and expands to every registered profile.  
**Risk:** The selector recommends models precisely when discovery reported zero availability.  
**Recommendation:** distinguish unknown availability from a verified empty set and return a no-model state.  
**Status:** OPEN

### KAI-MODELCTRL-002 — HIGH — Single candidate bypasses validation
**Issue:** When the candidate list has one element, it is returned immediately before checking whether it exists in `_PROFILES`, is live or supports the route.  
**Risk:** An arbitrary/unavailable model string is accepted as the best model.  
**Recommendation:** validate exact approved identity and readiness before every return path.  
**Status:** OPEN

### KAI-MODELCTRL-003 — HIGH — Unscored selection silently chooses Ollama
**Issue:** Unknown candidates are skipped; if no candidate receives a score, the function returns `Ollama` without checking that Ollama was offered or available.  
**Risk:** Routing silently leaves the caller’s availability set and can target a nonexistent backend.  
**Recommendation:** return an explicit no-valid-candidate result.  
**Status:** OPEN

### KAI-MODELCTRL-004 — HIGH — Conflicting model authorities
**Issue:** `model_selector`, `common/model_registry` and `model_council` maintain separate hard-coded profiles with different names, context windows, costs and quality claims. No shared version or consistency check exists.  
**Risk:** Token budgets, routing, health and recommendations can describe different capabilities for the same model.  
**Recommendation:** use one immutable model/artefact registry as the only authority.  
**Status:** OPEN

### KAI-MODELCTRL-005 — MEDIUM — Complexity is style-based
**Issue:** Complexity increases from word count, question marks, sentence count and matching broad substrings.  
**Risk:** Padding or keyword injection steers traffic toward costly/high-tier models without reflecting actual reasoning requirements.  
**Recommendation:** use typed task requirements and measured resource needs.  
**Status:** OPEN

### KAI-MODELCTRL-006 — MEDIUM — Long input is measured in characters
**Issue:** Context-window bonus triggers at `len(user_input) > 2000`, not provider/model token count or total message context.  
**Risk:** Multilingual/code inputs are misclassified and historical/system context is ignored.  
**Recommendation:** use exact model-token accounting for the complete prompt.  
**Status:** OPEN

### KAI-MODELCTRL-007 — HIGH — Registry mutation is unrestricted
**Issue:** `register_model` replaces any profile in the process-global dictionary with no authorisation, duplicate policy, audit or lock.  
**Risk:** Imported code can alter capability/quality metadata and redirect later routing.  
**Recommendation:** freeze a signed registry during startup.  
**Status:** OPEN

### KAI-MODELCTRL-008 — MEDIUM — Invalid profiles are accepted
**Issue:** Context, speed/quality tiers, expert count and VRAM are ordinary dataclass fields without finite/range/cross-field validation.  
**Risk:** negative/extreme/NaN values distort ranking and output.  
**Recommendation:** enforce a strict versioned profile schema.  
**Status:** OPEN

### KAI-MODELCTRL-009 — MEDIUM — Ties depend on insertion order
**Issue:** Sorting uses score only and Python’s stable order; mutable dictionary insertion order becomes the implicit tie-break.  
**Risk:** Registration order silently selects the model when scores tie.  
**Recommendation:** define an explicit deterministic policy with readiness/cost constraints.  
**Status:** OPEN

### KAI-MODELCTRL-010 — MEDIUM — Selection has no backend proof
**Issue:** Static profiles are considered candidates without probing endpoint, model digest, loaded context or current capacity.  
**Risk:** selection metadata is mistaken for operational availability.  
**Recommendation:** bind candidates to fresh signed backend readiness.  
**Status:** OPEN

---

## Model Council: `agentic/model_council.py`

### KAI-MODELCTRL-011 — HIGH — Council trust fails open
**Issue:** Unexpected governance/import errors are debug logged and all operations continue.  
**Risk:** compute-cost and model-selection mutations occur when the trust authority is broken.  
**Recommendation:** fail closed for benchmark/recommend/switch mutations.  
**Status:** OPEN

### KAI-MODELCTRL-012 — HIGH — Built-ins are assumed available
**Issue:** Every built-in profile defaults `available=True`, including remote models requiring credentials and potentially nonexistent/future identifiers. No discovery occurs before ranking/recommendation.  
**Risk:** unavailable models are recommended and failover targets from process start.  
**Recommendation:** default unknown/unprobed models unavailable.  
**Status:** OPEN

### KAI-MODELCTRL-013 — HIGH — Heuristic benchmark certifies availability
**Issue:** Without a supplied probe, `_default_probe` returns a score derived only from static quality/affinity metadata. A positive score marks the model available and is persisted as a benchmark result.  
**Risk:** no inference or connectivity occurs, yet the Council records a successful benchmark and live availability.  
**Recommendation:** benchmark the exact endpoint/artefact against validated tasks and distinguish static estimates.  
**Status:** OPEN

### KAI-MODELCTRL-014 — HIGH — Injected benchmark result is trusted
**Issue:** `probe_fn` may return any float. No authenticated producer, finite/range validation, evidence or benchmark identity is required before persistence.  
**Risk:** one internal caller can assign arbitrary quality and availability.  
**Recommendation:** accept signed benchmark records with bounded scores and reproducible evidence.  
**Status:** OPEN

### KAI-MODELCTRL-015 — HIGH — Primary model mutation is ungated
**Issue:** `set_primary` claims AGENT trust is required but performs only an existence check and writes the new primary.  
**Risk:** any internal caller can redirect the primary model.  
**Recommendation:** enforce authenticated operator/governance approval at this method.  
**Status:** OPEN

### KAI-MODELCTRL-016 — HIGH — Availability is caller-mutable
**Issue:** `record_success` immediately marks a model available and clears failures; `record_failure` can disable it after three calls. Neither validates caller identity, request outcome or model artefact.  
**Risk:** compromised/misbehaving callers manipulate model routing and denial of service.  
**Recommendation:** accept signed telemetry from the authoritative router with deduplication and scoped health logic.  
**Status:** OPEN

### KAI-MODELCTRL-017 — HIGH — Mutation success survives storage failure
**Issue:** `_save_profiles` suppresses every error. Benchmark, primary and health mutations return normally even if persistence failed.  
**Risk:** callers believe durable state changed but restart restores a different model configuration.  
**Recommendation:** acknowledge only verified durable commits.  
**Status:** OPEN

### KAI-MODELCTRL-018 — HIGH — Council persistence races
**Issue:** Complete profile state is rewritten without locks/version checks; singleton state is process-local and multiple workers overwrite each other.  
**Risk:** benchmarks, failures and primary changes are lost or corrupted.  
**Recommendation:** use transactional shared storage with monotonic revisions.  
**Status:** OPEN

### KAI-MODELCTRL-019 — HIGH — Invalid primary persists
**Issue:** persisted `primary` is assigned without checking that it exists, is available or is approved. Failover then indexes `self._profiles[self._primary]`, which can raise.  
**Risk:** corrupted/tampered state breaks failover and exposes invalid identity as primary.  
**Recommendation:** validate the complete profile snapshot before activation.  
**Status:** OPEN

### KAI-MODELCTRL-020 — MEDIUM — Custom profiles are weakly deserialised
**Issue:** unknown persisted entries are passed into the dataclass after filtering names only; capability lists, numerical values and identifiers are not validated.  
**Risk:** malformed profiles enter rankings or trigger runtime errors.  
**Recommendation:** validate a strict schema and approved provider/model IDs.  
**Status:** OPEN

### KAI-MODELCTRL-021 — MEDIUM — Built-in state leaks across resets
**Issue:** `_BUILTIN_PROFILES` contains mutable objects inserted directly into each Council. Benchmark/availability changes mutate those global objects; resetting the singleton does not restore pristine defaults.  
**Risk:** tests/restarts-in-process inherit hidden prior state.  
**Recommendation:** construct immutable fresh profiles from canonical data.  
**Status:** OPEN

### KAI-MODELCTRL-022 — MEDIUM — Invalid task silently becomes chat
**Issue:** `rank` replaces every unknown task type with `chat`; `recommend` does not validate task type and uses heuristic scores directly.  
**Risk:** misspelled/high-risk tasks receive unrelated chat recommendations.  
**Recommendation:** reject unknown task types.  
**Status:** OPEN

### KAI-MODELCTRL-023 — MEDIUM — Failover mutates caller state
**Issue:** `excluded = excluded or set()` then `excluded.add(self._primary)` modifies a non-empty set supplied by the caller.  
**Risk:** hidden side effects alter later caller decisions.  
**Recommendation:** copy caller collections.  
**Status:** OPEN

### KAI-MODELCTRL-024 — MEDIUM — Failure counters collapse distinct conditions
**Issue:** one count combines tasks, endpoints, credentials, rate limits and transient/permanent failures. Any success clears the full count.  
**Risk:** a chat success masks code failures; unrelated failures disable a model globally.  
**Recommendation:** track typed rolling health per artefact/capability/endpoint.  
**Status:** OPEN

### KAI-MODELCTRL-025 — MEDIUM — Cost and capability data is unverified
**Issue:** provider, token cost, context and quality are static source constants with no effective date, currency/billing basis or artefact validation.  
**Risk:** ranking and cost decisions use stale/fictional metadata.  
**Recommendation:** ingest versioned provider manifests and actual billing/usage telemetry.  
**Status:** OPEN

---

## Adversary: `agentic/adversary.py`

### KAI-MODELCTRL-026 — HIGH — Missing verifier passes
**Issue:** Any verifier transport, status, timeout or parsing exception returns `passed=True`, zero modifier and “challenge skipped”.  
**Risk:** a core factual control disappears without blocking the plan.  
**Recommendation:** return governance-unavailable and block/escalate consequential plans.  
**Status:** OPEN

### KAI-MODELCTRL-027 — HIGH — Unknown verdict becomes PASS
**Issue:** Every verdict other than exact `FAIL_CLOSED` or `REPAIR` enters the `else: # PASS` branch and adds +0.3 conviction.  
**Risk:** `UNKNOWN`, malformed, attacker-controlled or future verdicts positively certify the plan.  
**Recommendation:** accept a strict enum and fail closed on unknown values.  
**Status:** OPEN

### KAI-MODELCTRL-028 — HIGH — Missing Tool Gate passes
**Issue:** Tool Gate errors return a passed policy challenge with zero penalty.  
**Risk:** the plan survives when authoritative execution policy cannot be checked.  
**Recommendation:** block execution planning until fresh policy is available.  
**Status:** OPEN

### KAI-MODELCTRL-029 — HIGH — Missing mode defaults WORK
**Issue:** A successful health response without a `mode` field is interpreted as `WORK`, the execution-permissive mode.  
**Risk:** malformed/false health responses bypass PUB restrictions.  
**Recommendation:** require a signed strict mode enum and default restricted.  
**Status:** OPEN

### KAI-MODELCTRL-030 — HIGH — Security audit failure passes
**Issue:** Import or execution errors in `challenge_security` return `passed=True`.  
**Risk:** broken security self-testing is represented as a survived challenge.  
**Recommendation:** return failed/degraded and block security-relevant actions.  
**Status:** OPEN

### KAI-MODELCTRL-031 — HIGH — Security challenge is optional by argument omission
**Issue:** Security testing runs only when `injection_re` is non-null. No required-control policy validates its presence.  
**Risk:** callers silently remove the entire security challenge.  
**Recommendation:** make required challenges server-controlled and immutable by risk tier.  
**Status:** OPEN

### KAI-MODELCTRL-032 — HIGH — Boilerplate earns consistency credit
**Issue:** consistency checks only require steps, summary, specialist and a tiny contradictory-action list. A generic stock plan passes at confidence 0.85 and gains +0.2.  
**Risk:** formatting is mistaken for coherent safe reasoning and increases conviction.  
**Recommendation:** validate task dependencies, parameters, effects, rollback and evidence-linked steps.  
**Status:** OPEN

### KAI-MODELCTRL-033 — HIGH — Historical challenge data is untrusted
**Issue:** episode inputs, outcomes, conviction, failure class and metacognitive rules are accepted directly. Similarity uses token overlap.  
**Risk:** poisoned history creates positive/negative challenge modifiers and fabricated warnings.  
**Recommendation:** use immutable authenticated outcome evidence.  
**Status:** OPEN

### KAI-MODELCTRL-034 — MEDIUM — Caller episode objects are mutated
**Issue:** `challenge_history` writes `_similarity` into each matching episode dictionary.  
**Risk:** adversarial review changes shared history objects and contaminates later processing/concurrent calls.  
**Recommendation:** keep computed metadata local/immutable.  
**Status:** OPEN

### KAI-MODELCTRL-035 — MEDIUM — Risky tool classification is incomplete
**Issue:** only `shell`, `script` and `python` receive a weak warning. File mutation, trading, recovery, browser, database and other consequential tools are omitted.  
**Risk:** most high-impact plans receive “policy pre-check OK”.  
**Recommendation:** query the actual signed Tool Gate policy for the exact immutable request.  
**Status:** OPEN

### KAI-MODELCTRL-036 — MEDIUM — Weak historical under-confidence boosts execution
**Issue:** two token-overlap episodes whose outcomes exceed self-reported conviction add up to +0.5, regardless of source quality, sample size or task equivalence.  
**Risk:** poisoned/noisy history increases current conviction.  
**Recommendation:** use calibrated statistically valid outcome models and never boost consequential execution from tiny samples.  
**Status:** OPEN

### KAI-MODELCTRL-037 — MEDIUM — Adversary numbers are unvalidated
**Issue:** episode conviction/outcome/timestamps and verifier confidence/strong-chunk counts are converted directly without finite/range checks. Challenge modifiers are ordinary floats.  
**Risk:** NaN/extreme values break aggregation and recommendation comparisons.  
**Recommendation:** enforce strict typed ranges.  
**Status:** OPEN

### KAI-MODELCTRL-038 — MEDIUM — Diagnostics enter plan metadata
**Issue:** verifier evidence summaries, governance/security exception strings and historical text become findings/warnings copied into plan/episode output.  
**Risk:** internal errors and untrusted text are propagated broadly.  
**Recommendation:** use stable codes and protected traces; quote bounded evidence.  
**Status:** OPEN

### KAI-MODELCTRL-039 — HIGH — “Independent” challenges are correlated
**Issue:** history and calibration use the same episode set and similarity heuristic; verifier uses the same supplied memory; consistency scores the same boilerplate structure. Their modifiers are summed as separate scrutiny.  
**Risk:** one poisoned/shared source produces multiple reinforcing challenge results and false confidence.  
**Recommendation:** model challenge dependencies and require independent evidence/control authorities.  
**Status:** OPEN

---

## Tree search: `agentic/tree_search.py`

### KAI-MODELCTRL-040 — HIGH — No alternative reasoning is generated
**Issue:** branches are only the original query plus fixed suffixes. `build_plan_fn` receives each prompt; no LLM answer/reasoning path is generated before scoring.  
**Risk:** the system claims reasoning-tree search while comparing prompt wording and generic plan metadata.  
**Recommendation:** evaluate independently generated bounded candidate solutions with evidence and verifier results.  
**Status:** OPEN

### KAI-MODELCTRL-041 — HIGH — First threshold hit wins
**Issue:** branches are evaluated sequentially and the function immediately returns the first score meeting `min_conviction`. Later potentially safer/better branches are never evaluated.  
**Risk:** ordering and fixed suffix position determine the selected plan, contradicting best-branch search.  
**Recommendation:** evaluate the complete bounded candidate set or apply a justified admissible stopping rule.  
**Status:** OPEN

### KAI-MODELCTRL-042 — HIGH — Prompt suffixes inflate the scoring heuristics
**Issue:** the conviction scorer rewards word count, question/keywords, rethink depth and plan structure. Suffixes such as “risk”, “expert”, “step-by-step” change those inputs without improving an actual answer.  
**Risk:** branch conviction rises from prompt padding and can cross the execution threshold.  
**Recommendation:** score generated output evidence/correctness, not prompt surface form.  
**Status:** OPEN

### KAI-MODELCTRL-043 — HIGH — Cross-branch context contamination
**Issue:** extra chunks fetched for one survivor are appended to the shared `chunk_dicts`, affecting subsequent survivors/depths rather than remaining branch-local.  
**Risk:** evidence retrieved for one hypothesis boosts/changes unrelated branches and ordering influences results.  
**Recommendation:** maintain immutable context per branch with provenance.  
**Status:** OPEN

### KAI-MODELCTRL-044 — HIGH — Pruning does not eliminate all low branches
**Issue:** if every branch is below `prune_threshold`, the highest-scoring pruned branch is reinstated and search continues/returns it.  
**Risk:** a result is produced even when the search’s own acceptance criterion rejected every option.  
**Recommendation:** return no viable branch and escalate.  
**Status:** OPEN

### KAI-MODELCTRL-045 — HIGH — Debate failure is non-enforcing
**Issue:** `tree_search_with_debate` only sets `debate_survived=False` metadata. It does not lower conviction, choose the counterargument, block the branch or return a failed search.  
**Risk:** a plan that loses its explicit adversarial debate remains the best executable branch.  
**Recommendation:** make debate outcome a hard selection/gating rule.  
**Status:** OPEN

### KAI-MODELCTRL-046 — MEDIUM — Debate is not a branch comparison
**Issue:** only `result.best_branch` is placed into candidates; other survivors are unavailable because tree search does not retain them in the result.  
**Risk:** alternatives that might survive opposition are never considered.  
**Recommendation:** retain/evaluate all bounded survivors.  
**Status:** OPEN

### KAI-MODELCTRL-047 — MEDIUM — Counterargument scoring is structurally biased
**Issue:** the counter-plan uses specialist `debate`, which the conviction specialist-fit map does not recognise, while the original uses a selected known specialist.  
**Risk:** the counterargument begins with a built-in scoring disadvantage unrelated to merit.  
**Recommendation:** compare solutions under identical scoring conditions.  
**Status:** OPEN

### KAI-MODELCTRL-048 — MEDIUM — Branch diversity cycles after four templates
**Issue:** `n_branches` allows eight but suffixes cycle modulo four. Duplicate prompts at the same depth produce identical branch IDs and scores.  
**Risk:** pseudo-diversity inflates branch counts and can overwrite/collapse external tracking.  
**Recommendation:** enforce unique candidate generation and IDs.  
**Status:** OPEN

### KAI-MODELCTRL-049 — MEDIUM — Threshold relationships are unvalidated
**Issue:** branch/depth are clamped, but prune/min-conviction/debate margin accept NaN, infinity, negative or contradictory values.  
**Risk:** pruning/early-exit/debate logic can be disabled or inverted.  
**Recommendation:** validate finite ordered policy ranges.  
**Status:** OPEN

### KAI-MODELCTRL-050 — MEDIUM — Retrieval failure is silent
**Issue:** every refinement-context exception is ignored.  
**Risk:** branches proceed as though no additional evidence exists, with no degraded marker.  
**Recommendation:** preserve source availability and fail/escalate evidence-dependent search.  
**Status:** OPEN

---

## Priority queue: `agentic/priority_queue.py`

### KAI-MODELCTRL-051 — HIGH — No priority queue exists
**Issue:** `submit` never inserts a `QueueEntry` into `_pending` or selects the lowest-priority-number entry. It immediately waits on a semaphore.  
**Risk:** coroutine scheduling order, not declared priority, decides service order.  
**Recommendation:** implement a real bounded `asyncio.PriorityQueue` with workers.  
**Status:** OPEN

### KAI-MODELCTRL-052 — HIGH — Preemption claim is false
**Issue:** active tasks are never interrupted and waiting tasks cannot jump ahead in a semaphore wait queue under application control.  
**Risk:** urgent chat requests remain blocked behind batch work despite documented preemption.  
**Recommendation:** support cooperative cancellation/checkpointing or accurately remove the claim.  
**Status:** OPEN

### KAI-MODELCTRL-053 — HIGH — Priority is operationally unused
**Issue:** priority affects only the generated unused `task_id`; it never influences acquisition/execution.  
**Risk:** callers believe latency classes are enforced when all classes are equivalent.  
**Recommendation:** use priority as the scheduling key and test ordering.  
**Status:** OPEN

### KAI-MODELCTRL-054 — HIGH — Waiting work is unbounded
**Issue:** every submission creates/retains a coroutine waiting on the semaphore; no queue length, admission control or per-principal quota exists.  
**Risk:** load spikes exhaust memory and caller tasks while pretending to be queued safely.  
**Recommendation:** enforce bounded admission and reject/defer overload.  
**Status:** OPEN

### KAI-MODELCTRL-055 — MEDIUM — Pending structure is dead code
**Issue:** `_pending` and `QueueEntry.future` are never used.  
**Risk:** implementation and documentation diverge and hide the absence of scheduling.  
**Recommendation:** implement or remove the unused abstraction.  
**Status:** OPEN

### KAI-MODELCTRL-056 — MEDIUM — Pending metric is incorrect
**Issue:** `pending` is computed as maximum concurrency minus private semaphore value, which measures occupied permits, not waiting submissions. It duplicates active state.  
**Risk:** monitoring cannot detect backlog.  
**Recommendation:** report actual admitted/waiting/running counts.  
**Status:** OPEN

### KAI-MODELCTRL-057 — MEDIUM — Priority metrics are empty
**Issue:** `by_priority` is always `{}`.  
**Risk:** operators cannot observe class starvation/load despite the advertised scheduler.  
**Recommendation:** maintain accurate bounded per-class counters.  
**Status:** OPEN

### KAI-MODELCTRL-058 — MEDIUM — Singleton configuration depends on call order
**Issue:** the first `get_queue(max_concurrent)` call fixes concurrency for the process; later values are ignored without warning.  
**Risk:** tests/internal modules silently receive unexpected capacity.  
**Recommendation:** initialise one validated immutable scheduler at startup.  
**Status:** OPEN

### KAI-MODELCTRL-059 — MEDIUM — No execution/wait deadline
**Issue:** submissions can wait indefinitely and `fn` can run indefinitely.  
**Risk:** abandoned callers/tasks consume queue capacity and block all later work.  
**Recommendation:** use separate bounded queue-wait and execution deadlines with cancellation.  
**Status:** OPEN

### KAI-MODELCTRL-060 — MEDIUM — Task identity has no authority
**Issue:** default IDs use priority plus rounded monotonic time, may collide, and are never indexed, returned, cancelled or logged by the queue.  
**Risk:** work cannot be reliably traced or controlled.  
**Recommendation:** use collision-resistant job IDs and durable lifecycle state.  
**Status:** OPEN

---

## Batch totals

- Findings: **60**
- Critical: **0**
- High: **33**
- Medium: **27**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **964**
- Critical: **87**
- High: **372**
- Medium: **502**
- Low: **3**

## Files materially reviewed in this batch

`agentic/model_selector.py`, `agentic/model_council.py`, `agentic/adversary.py`, `agentic/tree_search.py`, `agentic/priority_queue.py`, with active-path confirmation against `agentic/app.py`.
