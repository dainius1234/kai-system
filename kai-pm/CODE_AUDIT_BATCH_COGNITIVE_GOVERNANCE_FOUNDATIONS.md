# Kai Code Audit — Cognitive Governance Foundations Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already captured in `CODE_AUDIT_REGISTER_CONTINUED.md` or `CODE_AUDIT_BATCH_AUTONOMOUS_STATE.md`. Existing Trust Core, Trust Integration and Wisdom Graph findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-COGOV-001 | HIGH | Enabled Global Workspace paths silently discard every submitted bid |
| KAI-COGOV-002 | HIGH | Global Workspace interfaces are wired as capabilities despite being permanently non-operational |
| KAI-COGOV-003 | MEDIUM | Workspace bids accept unbounded and invalid salience values/content |
| KAI-COGOV-004 | MEDIUM | Any imported module can overwrite workspace subscribers |
| KAI-COGOV-005 | MEDIUM | Workspace singleton state is concurrency-unsafe and worker-local |
| KAI-COGOV-006 | MEDIUM | Stream and cycle configuration is unvalidated and stream APIs ignore their contracts |
| KAI-COGOV-007 | HIGH | Active causal-edge ingestion always fails because the constructor call is incompatible |
| KAI-COGOV-008 | HIGH | Active surprise detection always fails because the application calls a nonexistent method |
| KAI-COGOV-009 | HIGH | Repeated causal observations overwrite rather than strengthen evidence |
| KAI-COGOV-010 | HIGH | Causal observations are volatile process-local state with no persistence or reconciliation |
| KAI-COGOV-011 | HIGH | Causal mutation remains callable while the capability gate reports unavailable |
| KAI-COGOV-012 | MEDIUM | Causal edge values and provenance fields are not validated |
| KAI-COGOV-013 | MEDIUM | Raw source/target strings create ambiguous and collision-prone edge identities |
| KAI-COGOV-014 | MEDIUM | Causal query APIs always return empty despite stored edge counts |
| KAI-COGOV-015 | MEDIUM | In-module policy memory accepts policies but never retrieves or updates them |
| KAI-COGOV-016 | MEDIUM | Two incompatible Policy and PolicyMemory implementations coexist |
| KAI-COGOV-017 | MEDIUM | Simulation scenarios and state payloads are unbounded and unvalidated |
| KAI-COGOV-018 | MEDIUM | Causal singletons are unsynchronised and diverge across workers |
| KAI-COGOV-019 | HIGH | Ohana moral learning mutates state even though `can_operate()` is always false |
| KAI-COGOV-020 | HIGH | The application records Kai’s response as the operator’s moral decision |
| KAI-COGOV-021 | HIGH | Every chat response overwrites the same generic situational stance |
| KAI-COGOV-022 | HIGH | Moral decisions have no authenticated operator provenance |
| KAI-COGOV-023 | HIGH | Moral fingerprint state is an unauthenticated mutable local file |
| KAI-COGOV-024 | HIGH | Corrupt moral state silently resets to hard-coded defaults |
| KAI-COGOV-025 | HIGH | Permissive loyalty/rule-flexibility defaults can govern later actions without learning evidence |
| KAI-COGOV-026 | MEDIUM | Harm boundaries use first-three-word substring matching |
| KAI-COGOV-027 | MEDIUM | Loyalty alignment uses manipulable substring matches |
| KAI-COGOV-028 | MEDIUM | Unmatched actions receive neutral approval and fingerprint scoring cannot express ordinary misalignment |
| KAI-COGOV-029 | MEDIUM | Wisdom Graph failures are silently removed from moral evaluation |
| KAI-COGOV-030 | MEDIUM | Fingerprint writes are non-atomic and concurrency-unsafe |
| KAI-COGOV-031 | MEDIUM | Moral state is exposed as mutable objects and the in-memory decision log is unbounded |
| KAI-COGOV-032 | HIGH | Moral Imagination fails open to `proceed` whenever its dependencies or code fail |
| KAI-COGOV-033 | HIGH | A `halt` moral recommendation never halts the cognitive pipeline |
| KAI-COGOV-034 | HIGH | Value alignment directly increases factual/action conviction |
| KAI-COGOV-035 | HIGH | Moral Imagination iterates a private edge dictionary as edge objects and fails when populated |
| KAI-COGOV-036 | HIGH | Moral action extraction ignores normal plan-step actions and effects |
| KAI-COGOV-037 | MEDIUM | Moral harm detection is substring-based and linguistically bypassable |
| KAI-COGOV-038 | MEDIUM | Missing moral infrastructure is represented as neutral alignment |
| KAI-COGOV-039 | MEDIUM | Alignment and handoff confidence values lack finite/range validation |
| KAI-COGOV-040 | MEDIUM | Sensitive values and boundaries are copied into downstream handoff payloads |
| KAI-COGOV-041 | MEDIUM | “Projected consequences” are labels and graph identifiers, not outcome simulation |
| KAI-COGOV-042 | HIGH | Cognitive fingerprinting collects query/session data in plaintext by default |
| KAI-COGOV-043 | HIGH | All users and sessions are merged into one operator fingerprint |
| KAI-COGOV-044 | HIGH | Any file line counts toward the 90-sample readiness threshold |
| KAI-COGOV-045 | HIGH | Readiness becomes true while inference remains an explicit zero-confidence stub |
| KAI-COGOV-046 | HIGH | Feature-flag import failure bypasses disablement and falls through to count-based readiness |
| KAI-COGOV-047 | MEDIUM | Fingerprint appends are synchronous and multi-process unsafe |
| KAI-COGOV-048 | MEDIUM | Fingerprint write failures and data loss are silent |
| KAI-COGOV-049 | MEDIUM | Cached sample counts become stale across workers and external writes |
| KAI-COGOV-050 | MEDIUM | Fingerprint storage grows without retention or size bounds |
| KAI-COGOV-051 | MEDIUM | Direct samples and arbitrary `extra` metadata are not schema/range validated |
| KAI-COGOV-052 | MEDIUM | Surface-text heuristics falsely infer decisions, risk tolerance and response preference |
| KAI-COGOV-053 | HIGH | Policy storage remains active while policy distillation is disabled |
| KAI-COGOV-054 | HIGH | Arbitrary manually supplied policies are accepted without authority or semantic validation |
| KAI-COGOV-055 | HIGH | Policy IDs are collision-prone, process-randomised and unstable |
| KAI-COGOV-056 | HIGH | Policy writes fail silently with no acknowledgement contract |
| KAI-COGOV-057 | HIGH | Policy log integrity and concurrent append are unprotected |
| KAI-COGOV-058 | HIGH | One-word overlap can surface poisoned policies as relevant guidance |
| KAI-COGOV-059 | MEDIUM | Policy confidence, utility, counts, versions, domains and sources are unvalidated |
| KAI-COGOV-060 | MEDIUM | Retrieval limits and confidence thresholds accept invalid values |
| KAI-COGOV-061 | MEDIUM | Corrupt policies are silently skipped, creating invisible evidence gaps |
| KAI-COGOV-062 | MEDIUM | `all_policies()` loads the complete unbounded log into memory |
| KAI-COGOV-063 | MEDIUM | Policy counts include invalid lines and cache stale cross-process state |
| KAI-COGOV-064 | MEDIUM | Policy validation is a no-op and records no real-world outcome |
| KAI-COGOV-065 | HIGH | A role string, not authenticated identity, determines who is the operator |
| KAI-COGOV-066 | HIGH | Concatenating separate messages can synthesize wisdom statements never made |
| KAI-COGOV-067 | HIGH | Wisdom regexes ignore quotation, negation, hypotheticals and attribution |
| KAI-COGOV-068 | HIGH | `confirm_all()` bypasses the documented individual operator-review gate |
| KAI-COGOV-069 | HIGH | Wisdom confirmation and rejection have no authenticated actor check |
| KAI-COGOV-070 | HIGH | Wisdom confirmation spans multiple stores without a transaction |
| KAI-COGOV-071 | HIGH | Partial Ohana, ledger or graph failure still returns confirmation success |
| KAI-COGOV-072 | HIGH | Pending/confirmed/rejected wisdom is tamperable and concurrency-unsafe local JSON |
| KAI-COGOV-073 | HIGH | Wisdom ingestion mutates private Ohana state directly, bypassing governance APIs |
| KAI-COGOV-074 | HIGH | Pattern confidence is written as alignment evidence and can inflate the Trust Ledger score |
| KAI-COGOV-075 | MEDIUM | Bulk-confirm thresholds are not validated |
| KAI-COGOV-076 | MEDIUM | Corrupt wisdom files silently become empty state without quarantine |
| KAI-COGOV-077 | MEDIUM | Sensitive source quotes and inferred moral statements are stored in plaintext |
| KAI-COGOV-078 | MEDIUM | Wisdom queues and interaction counters are unbounded, process-local and first-caller configured |

---

## Global Workspace: `agentic/global_workspace.py`

### KAI-COGOV-001 — HIGH — Submitted bids are silently discarded
`agentic/app.py` submits proactive and Cortex bids whenever `FF_GLOBAL_WORKSPACE` is enabled. `GlobalWorkspace.submit_bid()` only logs and discards the bid; `_bid_queue` is never populated. Callers receive no unavailable/error state. This makes an enabled cognitive-control path silently non-functional.

### KAI-COGOV-002 — HIGH — Wired capability is permanently unavailable
The singleton, subscriber API, progress API and application integration exist as live interfaces while `select_winner()` always returns `None`, `broadcast()` is a no-op and `can_operate()` always returns `False`. Interface presence and subscriber counts can therefore be mistaken for an operating global-awareness control.

### KAI-COGOV-003 — MEDIUM — Bid data is unvalidated
`WorkspaceBid` accepts arbitrary module/content strings, non-finite or out-of-range urgency, relevance, surprise, confidence and emotional salience. No content or aggregate bounds exist.

### KAI-COGOV-004 — MEDIUM — Subscriber replacement is unrestricted
`subscribe()` replaces any existing callback under a caller-selected module name. There is no ownership, duplicate rejection, frozen registry or audit trail.

### KAI-COGOV-005 — MEDIUM — Worker-local unsynchronised state
Subscriber, stream and bid collections are mutable process-local structures without locks. Multiple workers would hold different cognitive state, and concurrent registration/removal is unsafe.

### KAI-COGOV-006 — MEDIUM — Configuration and stream contracts are not enforced
`max_stream_length` and `cycle_ms` accept invalid values. `get_stream(limit)` ignores `limit` and always returns an empty list; `get_latest_moment()` always returns `None` regardless of internal state.

---

## Causal World Model: `agentic/causal_world_model.py`, `agentic/app.py`

### KAI-COGOV-007 — HIGH — Active edge ingestion cannot construct an edge
The application calls `CausalEdge(source=..., target=..., strength=..., source_type=..., note=...)`. The dataclass requires `confidence` and has no `note` field. Every active correlation write raises `TypeError`, and the application suppresses the exception. No causal observations are stored.

### KAI-COGOV-008 — HIGH — Surprise detector method mismatch
The proactive observer calls `get_surprise_detector().check(...)`; the class only defines `check_surprise(...)`. The resulting `AttributeError` is swallowed, so surprise detection never executes.

### KAI-COGOV-009 — HIGH — Repeated evidence is overwritten
`add_edge()` keys only by source and target and replaces the complete edge object. It does not increment `evidence_count`, merge provenance or update confidence, despite the application comment claiming each co-occurrence strengthens an edge.

### KAI-COGOV-010 — HIGH — Learned causal evidence is volatile
Edges exist only in one process dictionary. Restart loses all observations and replicas diverge, so accumulated evidence cannot support later calibration or forensic review.

### KAI-COGOV-011 — HIGH — Mutation bypasses capability readiness
`add_edge()` has no feature/readiness check and remains callable even though `can_reason()` is permanently false. Disabled/unready causal infrastructure can accumulate caller-controlled state that may later become authoritative.

### KAI-COGOV-012 — MEDIUM — Edge semantics are unvalidated
Strength, confidence, lag, evidence count, direction, source type and context modifiers have no finite/range/enum/depth validation.

### KAI-COGOV-013 — MEDIUM — Edge identity is ambiguous
Raw source and target strings are concatenated into `causal:{source}->{target}` without escaping, canonical identity or namespace. Delimiter-containing names and cross-domain concepts can collide or be confused.

### KAI-COGOV-014 — MEDIUM — Stored edges cannot be queried
Path, upstream, downstream and outcome methods always return empty values while `edge_count()` reports stored edges. Consumers cannot distinguish “no relationship” from “query engine unavailable”.

### KAI-COGOV-015 — MEDIUM — In-module PolicyMemory is non-functional
The in-memory policy store accepts policies, but `get_relevant_policies()` always returns empty and `update_policy_success()` does nothing. Stored state appears available through IDs/counts without influencing decisions or learning.

### KAI-COGOV-016 — MEDIUM — Competing policy models
`causal_world_model.py` defines `Policy`/`PolicyMemory`, while `policy_memory.py` defines a different `Policy`/`PolicyLibrary` with different fields and persistence. There is no canonical conversion or authority.

### KAI-COGOV-017 — MEDIUM — Simulation inputs are unbounded
Goals, state dictionaries, actions, horizon and variation counts have no size/range validation, even though they are intended for expensive future simulation.

### KAI-COGOV-018 — MEDIUM — Singleton state races and diverges
Causal graph, simulator, policy memory and detector factories are unlocked module globals. Concurrent first use can race and worker processes hold independent models.

---

## Ohana Core: `agentic/moral_core.py`, `agentic/app.py`

### KAI-COGOV-019 — HIGH — Moral learning operates while capability is disabled
`can_operate()` always returns false, but `record_decision()`, fingerprint persistence and `evaluate_action_alignment()` are fully active and do not check `FF_OHANA_CORE`.

### KAI-COGOV-020 — HIGH — Kai’s answer is learned as the operator’s decision
After every streamed exchange, the application passes the assistant response to `record_decision(..., decision=response[:300])`. The resulting moral fingerprint therefore learns what Kai said, not what the operator decided or endorsed.

### KAI-COGOV-021 — HIGH — One generic stance is repeatedly overwritten
The application supplies `query`, `mode` and `session_id`, but no `type` or `domain`. `record_decision()` therefore uses `general` as the stance key, overwriting it on every chat turn with the newest assistant response.

### KAI-COGOV-022 — HIGH — No operator provenance exists
Direct callers can submit arbitrary situation/decision/outcome values. No authenticated principal, consent, confirmation, source message ID or operator acknowledgement is required.

### KAI-COGOV-023 — HIGH — Fingerprint is tamperable state
Core loyalties, harm boundaries, stances and override weights are loaded from and written to ordinary JSON without integrity protection, permission validation, monotonic revision or external root of trust.

### KAI-COGOV-024 — HIGH — Corruption silently restores defaults
Any parse/deserialisation failure logs a warning and returns a new default fingerprint. The damaged file is not quarantined and governance does not enter a locked state.

### KAI-COGOV-025 — HIGH — Permissive values precede evidence
Defaults include `rule_flexibility=0.9` and `loyalty_override=1.0`. These are available to progress/alignment logic before authenticated learning has occurred and are not tied to policy revision or evidence.

### KAI-COGOV-026 — MEDIUM — Boundary matching is linguistically unsound
A boundary blocks when any of its first three whitespace words appears as a substring anywhere in JSON-serialised action text. Common words cause false blocks; paraphrases evade the boundary.

### KAI-COGOV-027 — MEDIUM — Loyalty alignment is keyword-gameable
A loyalty counts as matched when any of its first two words appears as a substring. Crafted text can increase alignment without serving the value.

### KAI-COGOV-028 — MEDIUM — Neutral is the minimum ordinary score
Without an exact boundary match, fingerprint scoring starts at 0.5 and can only increase. Ordinary conflicts, trade-offs and absence of evidence cannot produce a sub-neutral score.

### KAI-COGOV-029 — MEDIUM — Graph failure disappears
All Wisdom Graph import/evaluation errors are ignored and alignment silently falls back to fingerprint-only scoring.

### KAI-COGOV-030 — MEDIUM — Persistence is non-atomic
The complete fingerprint is rewritten directly without a temporary file, fsync, lock or compare-and-swap. Concurrent decisions can lose updates or corrupt the file.

### KAI-COGOV-031 — MEDIUM — Mutable/unbounded moral state
`get_fingerprint_snapshot()` returns the live mutable object. `_decision_log` grows without bound and is not durably reconciled with the persisted fingerprint.

---

## Moral Imagination: `agentic/moral_imagination.py`

### KAI-COGOV-032 — HIGH — Failure defaults to approval
The default result is alignment 0.5, zero penalty and recommendation `proceed`. Every dependency or programming exception retains this approval state.

### KAI-COGOV-033 — HIGH — Halt is metadata only
Even when `_recommendation()` returns `halt`, the stage always creates a `COMPLETE` handoff to the conviction gate. It never sets a halted status or enforces a boundary.

### KAI-COGOV-034 — HIGH — Values inflate epistemic conviction
Alignment at least 0.75 adds +0.8 to handoff confidence. Moral affinity is therefore treated as evidence that a factual plan is correct or operationally reliable.

### KAI-COGOV-035 — HIGH — Private-edge iteration bug
`_query_moral_context()` returns `graph._edges`, a dictionary. `_project_goods()` iterates it as if each item were an edge object and accesses `.relation`. Once the dictionary is non-empty, this raises and the stage falls open to `proceed`.

### KAI-COGOV-036 — HIGH — Normal plan actions are omitted
`_extract_action_text()` reads query, plan summary and a singular `plan.action`; the normal plan representation uses a list of steps. Concrete tools, parameters, targets and side effects are therefore absent from moral review.

### KAI-COGOV-037 — MEDIUM — Harm projection uses weak substrings
Boundary detection repeats first-three-word substring matching and ignores negation, role, target, intent and semantic equivalence.

### KAI-COGOV-038 — MEDIUM — Unavailability equals neutrality
Graph and Ohana failures become empty context and alignment 0.5 rather than an unavailable or review-required result.

### KAI-COGOV-039 — MEDIUM — Numerical inputs are not validated
Alignment and handoff confidence can be non-finite/out-of-range; the stage relies on comparisons and clamping without a strict input contract.

### KAI-COGOV-040 — MEDIUM — Sensitive moral data is propagated
Relevant values, imagined harms and boundary content are copied into the handoff payload and may enter logs, episode storage or API output.

### KAI-COGOV-041 — MEDIUM — No consequence model exists
“Goods” merely restate relevant values or abbreviated target IDs. No causal outcome, affected party, likelihood, reversibility or alternative is evaluated.

---

## Cognitive Fingerprint: `agentic/cognitive_fingerprint.py`

### KAI-COGOV-042 — HIGH — Default-on behavioural profiling
The feature is described as default true and records query text, session ID and inferred behavioural labels to `/data/cognitive_fingerprint.jsonl`. There is no consent state, minimisation policy, encryption, access hardening, retention or deletion mechanism.

### KAI-COGOV-043 — HIGH — Cross-principal profiling
One global file and singleton aggregate all sessions and users. Sample count/inference has no authenticated operator partition.

### KAI-COGOV-044 — HIGH — Readiness counts arbitrary lines
`sample_count()` counts every line without parsing or validating it. Blank, corrupt, duplicated or attacker-appended lines satisfy the 90-sample threshold.

### KAI-COGOV-045 — HIGH — Ready state contradicts inference state
`progress()` reports `ready_for_inference=True` at 90 lines, but `infer()` still returns `stub_pending_gpu_clustering` with confidence 0.0.

### KAI-COGOV-046 — HIGH — Flag import failure is permissive
`can_infer()` imports `feature_flags` from a top-level module. On `ImportError` it continues and permits readiness based solely on sample count.

### KAI-COGOV-047 — MEDIUM — Unsafe synchronous append
Each record performs inline filesystem append without a multi-process lock, fsync or durable queue.

### KAI-COGOV-048 — MEDIUM — Silent evidence loss
Record failures are debug-only and do not affect health, readiness or caller success.

### KAI-COGOV-049 — MEDIUM — Count cache is incoherent
The cached count is invalidated only by this process’s `record()`. Other workers/external writes leave stale counts, while each worker may report different readiness.

### KAI-COGOV-050 — MEDIUM — Unbounded retention
The JSONL log has no size, age, sample or cardinality cap; count requires a complete line scan after invalidation.

### KAI-COGOV-051 — MEDIUM — Sample schema is unenforced
Direct `InteractionSample` objects can carry arbitrary categorical strings, timestamps, session IDs and nested `extra` metadata.

### KAI-COGOV-052 — MEDIUM — Heuristics do not measure claimed traits
Query length is labelled response-length preference; substrings such as `yes`, `ok`, `risk`, `try` and `plan` are treated as decisions, risk tolerance and time horizon without outcome/follow-up evidence.

---

## Policy Memory: `agentic/policy_memory.py`

### KAI-COGOV-053 — HIGH — Storage bypasses feature readiness
`store()` works unconditionally even while `can_distill()` is false and the feature is disabled. Unreviewed seed policies can accumulate before policy infrastructure is ready.

### KAI-COGOV-054 — HIGH — Arbitrary policy authority
No authenticated producer, approval state, evidence reference or schema validation is required before storing conditions and recommended actions.

### KAI-COGOV-055 — HIGH — Weak unstable identity
IDs combine second-resolution creation time with only 16 bits of Python’s process-randomised `hash(condition)`. Same-second collisions are feasible and the hash changes across processes/runs.

### KAI-COGOV-056 — HIGH — Failed storage is indistinguishable from success
All exceptions are swallowed and `store()` returns no status. The caller cannot know whether a policy became durable.

### KAI-COGOV-057 — HIGH — Log integrity is unprotected
Policies are plaintext JSONL appended without locking, fsync, signature, ownership validation or retention.

### KAI-COGOV-058 — HIGH — Weak retrieval promotes poisoned guidance
Any one whitespace-token overlap qualifies. Score is raw overlap plus caller-controlled confidence, so generic/high-confidence malicious policies rank highly and may later be injected into system context.

### KAI-COGOV-059 — MEDIUM — Policy fields are unvalidated
Confidence, utility, counts, version, timestamps, domains, sources and text sizes are not checked for finite/range/enum validity.

### KAI-COGOV-060 — MEDIUM — Retrieval controls are invalidatable
Negative/huge `top_k`, non-finite/negative confidence thresholds and unbounded context strings are accepted.

### KAI-COGOV-061 — MEDIUM — Corruption is hidden
Malformed lines are silently skipped, so policy evidence can disappear while retrieval appears healthy.

### KAI-COGOV-062 — MEDIUM — Full-log materialisation
`all_policies()` reads and deserialises the entire unbounded file with no pagination or byte limit.

### KAI-COGOV-063 — MEDIUM — Count is not a valid-policy count
`policy_count()` counts all file lines, including blank/corrupt records, and caches a process-local value that becomes stale.

### KAI-COGOV-064 — MEDIUM — Validation does nothing
`validate()` only emits a debug message; it does not persist the outcome, increase validation count, update confidence or link evidence.

---

## Wisdom Ingestion: `agentic/wisdom_ingestion.py`, `trust-ledger/score.py`

### KAI-COGOV-065 — HIGH — Role label substitutes for identity
`extract_from_messages()` treats every message whose caller-controlled role equals `user` as Dainius’s statement. No authenticated principal or message provenance exists.

### KAI-COGOV-066 — HIGH — Cross-message synthesis
All operator messages are joined with spaces before regex extraction. Words from separate times/contexts can form a pattern that was never stated in any message.

### KAI-COGOV-067 — HIGH — Pattern extraction ignores meaning
Regexes do not distinguish a belief from a quotation, denial, joke, hypothetical, criticism or report of someone else’s view, yet assign confidence as high as 1.0.

### KAI-COGOV-068 — HIGH — Bulk confirmation bypasses review
`confirm_all()` automatically confirms every extract above a threshold, contradicting the documented `operator approves` gate.

### KAI-COGOV-069 — HIGH — Confirmation has no actor authority
`confirm`, `reject` and `confirm_all` accept IDs/notes only. Any internal caller can assert operator approval or rejection.

### KAI-COGOV-070 — HIGH — Confirmation is non-transactional
Pending removal, confirmed append, Ohana mutation, Trust Ledger append and Wisdom Graph mutation are separate writes with no transaction or idempotency key.

### KAI-COGOV-071 — HIGH — Partial commit still succeeds
Failures in Ohana, ledger or graph writes are swallowed. The extract remains confirmed and the method returns true even when downstream governance state did not update consistently.

### KAI-COGOV-072 — HIGH — Wisdom records are tamperable
Pending, confirmed and rejected states are ordinary complete-file JSON rewrites without locks, atomic replacement, integrity protection or trusted ownership.

### KAI-COGOV-073 — HIGH — Governance API is bypassed
`_write_to_ohana()` directly mutates `core.fingerprint`, increments the private interaction counter and invokes the private save method instead of an authenticated/reviewed transition API.

### KAI-COGOV-074 — HIGH — Extraction confidence becomes Trust score evidence
Confirmation writes `ohana_alignment=min(1.0, extract.confidence)` as an `ALIGNMENT_AUDIT`. `trust-ledger/score.py` makes average `ohana_alignment` worth 25% of the Trust score. Regex confidence is therefore treated as demonstrated value alignment and can raise autonomy scoring.

### KAI-COGOV-075 — MEDIUM — Bulk threshold is unvalidated
Negative thresholds confirm every pending item; non-finite/extreme values produce inconsistent behaviour.

### KAI-COGOV-076 — MEDIUM — Corrupt state resets silently
A malformed file logs a warning and loads as an empty list. The original is not quarantined and no locked/recovery state is exposed.

### KAI-COGOV-077 — MEDIUM — Sensitive text is stored openly
Source quotes, inferred values/boundaries and operator notes are persisted in plaintext without minimisation, retention or deletion controls.

### KAI-COGOV-078 — MEDIUM — Unbounded and process-local lifecycle
Pending/confirmed/rejected lists grow without limits; singleton first-call configuration wins; multiple workers diverge; each confirmation artificially increments Ohana interaction count.

---

## Batch totals

- Findings: **78**
- Critical: **0**
- High: **39**
- Medium: **39**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,042**
- Critical: **87**
- High: **411**
- Medium: **541**
- Low: **3**

## Files materially reviewed

`agentic/global_workspace.py`, `agentic/causal_world_model.py`, `agentic/moral_core.py`, `agentic/moral_imagination.py`, `agentic/cognitive_fingerprint.py`, `agentic/policy_memory.py`, `agentic/wisdom_ingestion.py`, `trust-ledger/score.py`, with active-path confirmation against `agentic/app.py`.
