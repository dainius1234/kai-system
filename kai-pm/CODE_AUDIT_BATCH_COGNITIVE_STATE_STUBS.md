# Kai Code Audit — Cognitive State and Stubbed Governance Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-COGSTATE-001 | HIGH | Global Workspace bids are accepted and discarded while callers may assume participation |
| KAI-COGSTATE-002 | HIGH | Global Workspace subscription state is mutable despite the broadcast mechanism being non-functional |
| KAI-COGSTATE-003 | HIGH | Global Workspace capability checks fail open on feature-flag import failure before returning a hard-coded false |
| KAI-COGSTATE-004 | MEDIUM | Workspace configuration accepts invalid stream and cycle values without validation |
| KAI-COGSTATE-005 | MEDIUM | Workspace bid and moment fields are unbounded and unvalidated |
| KAI-COGSTATE-006 | MEDIUM | Workspace singleton and subscriber registry are concurrency-unsafe |
| KAI-COGSTATE-007 | HIGH | Causal edges are stored from observations without validating causality, range or provenance |
| KAI-COGSTATE-008 | HIGH | Repeated edge writes overwrite prior evidence rather than accumulating or reconciling it |
| KAI-COGSTATE-009 | HIGH | Causal graph persistence and recovery are absent while active code records observations into it |
| KAI-COGSTATE-010 | HIGH | Causal query and prediction APIs return empty normal-looking results instead of unavailable states |
| KAI-COGSTATE-011 | HIGH | Policy Memory accepts arbitrary learned policies although retrieval and success tracking are stubs |
| KAI-COGSTATE-012 | MEDIUM | Causal model identifiers are collision-prone and caller-controlled |
| KAI-COGSTATE-013 | MEDIUM | Causal model state is process-local and concurrency-unsafe |
| KAI-COGSTATE-014 | MEDIUM | Simulation and surprise thresholds/inputs lack range and size validation |
| KAI-COGSTATE-015 | HIGH | Ohana moral fingerprint persistence is unsigned and writable by ordinary process code |
| KAI-COGSTATE-016 | HIGH | A single recorded decision permanently becomes a moral stance without outcome verification |
| KAI-COGSTATE-017 | HIGH | Caller-controlled situation keys can overwrite prior moral stances |
| KAI-COGSTATE-018 | HIGH | Moral harm boundaries use substring matching and are bypassable or over-broad |
| KAI-COGSTATE-019 | HIGH | Loyalty keyword matches can only increase alignment and never penalise conflicting actions |
| KAI-COGSTATE-020 | HIGH | Missing or failed Wisdom Graph evaluation silently falls back to permissive fingerprint scoring |
| KAI-COGSTATE-021 | HIGH | Moral fingerprint corruption silently resets to hard-coded defaults |
| KAI-COGSTATE-022 | HIGH | Moral fingerprint saves are non-atomic, unsynchronised and not durability-checked |
| KAI-COGSTATE-023 | MEDIUM | Moral fingerprint fields and numerical weights are unvalidated |
| KAI-COGSTATE-024 | MEDIUM | Moral context generation ignores the supplied situation |
| KAI-COGSTATE-025 | MEDIUM | Moral decision log is unbounded, process-local and not persisted |
| KAI-COGSTATE-026 | MEDIUM | Ohana can-operate status remains false while active learning and alignment methods mutate state |
| KAI-COGSTATE-027 | HIGH | Cognitive fingerprint logging defaults on and stores raw operator queries in plaintext |
| KAI-COGSTATE-028 | HIGH | Fingerprint records lack consent, retention, encryption and principal isolation |
| KAI-COGSTATE-029 | HIGH | Any line in the JSONL file counts as a valid behavioural sample |
| KAI-COGSTATE-030 | HIGH | Corrupt or forged sample logs can satisfy the 90-sample readiness threshold |
| KAI-COGSTATE-031 | MEDIUM | Fingerprint writes are non-atomic and unsafe across workers |
| KAI-COGSTATE-032 | MEDIUM | Fingerprint logging failures are silently suppressed |
| KAI-COGSTATE-033 | MEDIUM | Query heuristics misclassify decisions, risk tolerance and response preferences by substring |
| KAI-COGSTATE-034 | MEDIUM | Feature-flag import failure permits readiness evaluation to continue |
| KAI-COGSTATE-035 | MEDIUM | Fingerprint sample-count cache becomes stale across processes and external file changes |
| KAI-COGSTATE-036 | MEDIUM | Fingerprint sample fields and extra metadata are not schema/range bounded |

---

## Global Workspace: `agentic/global_workspace.py`

### KAI-COGSTATE-001 — HIGH — Submitted bids are discarded
**Issue:** `submit_bid` only logs the bid and never appends it to `_bid_queue`; `select_winner` always returns `None`. Active callers in `agentic/app.py` can submit anomaly and Cortex bids without receiving a failure or unavailable state.  
**Risk:** modules and dashboards can believe information entered a shared cognition layer when it was silently discarded.  
**Recommendation:** return an explicit unavailable result until the workspace is implemented, and prevent callers from claiming successful submission.  
**Status:** OPEN

### KAI-COGSTATE-002 — HIGH — Subscriber registration creates false readiness
**Issue:** `subscribe` stores callbacks even though `broadcast` never invokes them. `subscriber_count` therefore rises and appears in progress metrics despite no operational broadcast path.  
**Risk:** activation/readiness logic can be satisfied by inert registrations.  
**Recommendation:** distinguish registered, active and successfully invoked subscribers.  
**Status:** OPEN

### KAI-COGSTATE-003 — HIGH — Feature-flag import failure is tolerated
**Issue:** `can_operate` catches `ImportError` and continues. It currently returns false later, but the pattern makes missing governance infrastructure non-fatal at the capability boundary and is unsafe for future activation changes.  
**Risk:** later implementation changes can unintentionally activate the workspace when feature-flag enforcement is unavailable.  
**Recommendation:** fail closed and surface a configuration error.  
**Status:** OPEN

### KAI-COGSTATE-004 — MEDIUM — Invalid workspace configuration
**Issue:** `max_stream_length` and `cycle_ms` are accepted without finite/range validation.  
**Risk:** negative, zero or extreme values can break future scheduling and retention logic.  
**Recommendation:** validate bounded positive configuration at startup.  
**Status:** OPEN

### KAI-COGSTATE-005 — MEDIUM — Unbounded bid and moment fields
**Issue:** module names, content, context and all salience dimensions are unvalidated.  
**Risk:** oversized or non-finite values can distort future selection, exhaust memory and leak arbitrary content into broadcasts.  
**Recommendation:** enforce strict bounded schemas and finite ranges.  
**Status:** OPEN

### KAI-COGSTATE-006 — MEDIUM — Shared mutable workspace state is unsynchronised
**Issue:** subscriber, stream and bid collections are mutable process-global structures without locks.  
**Risk:** concurrent registration, broadcast and selection can race when the stub is activated.  
**Recommendation:** use one event-loop-owned scheduler or concurrency-safe state machine.  
**Status:** OPEN

---

## Causal World Model: `agentic/causal_world_model.py`

### KAI-COGSTATE-007 — HIGH — Observation is promoted to causality
**Issue:** `add_edge` accepts arbitrary `CausalEdge` objects without validating strength, confidence, evidence count, source type or whether evidence demonstrates causation. Active observer code records co-occurrences as causal edges.  
**Risk:** correlations and heuristic guesses become causal knowledge used by future simulation and policy learning.  
**Recommendation:** require causal-identification provenance, bounded values and independent evidence review.  
**Status:** OPEN

### KAI-COGSTATE-008 — HIGH — Edge evidence is overwritten
**Issue:** edge ID is only `source->target`; each new edge replaces the prior object. No merge, versioning, uncertainty update or conflicting-direction handling exists.  
**Risk:** the latest caller silently rewrites causal belief and evidence history.  
**Recommendation:** use immutable versioned observations and an explicit aggregation model.  
**Status:** OPEN

### KAI-COGSTATE-009 — HIGH — Active causal observations are not durable
**Issue:** edges are held only in an in-memory singleton, despite active code adding observations.  
**Risk:** restart, worker routing or process crash loses causal state, while workers disagree.  
**Recommendation:** persist an append-only provenance log and build validated snapshots.  
**Status:** OPEN

### KAI-COGSTATE-010 — HIGH — Stub queries look like valid empty answers
**Issue:** path, upstream/downstream and prediction APIs return `[]` or `{}` rather than an unavailable/error state.  
**Risk:** callers cannot distinguish “no causal relationship” from “causal reasoning not implemented”.  
**Recommendation:** return typed capability-unavailable results.  
**Status:** OPEN

### KAI-COGSTATE-011 — HIGH — Arbitrary policy storage precedes policy validation
**Issue:** `PolicyMemory.add_policy` accepts and stores any policy, while retrieval, application and success-rate update are non-functional stubs.  
**Risk:** untrusted strategies can accumulate and later become active when retrieval is implemented, without provenance or migration controls.  
**Recommendation:** block policy ingestion until a strict signed schema and review lifecycle exist.  
**Status:** OPEN

### KAI-COGSTATE-012 — MEDIUM — Caller-controlled IDs collide
**Issue:** causal and policy IDs are derived directly from source, target or policy name.  
**Risk:** name collisions overwrite unrelated objects and special characters create ambiguous identifiers.  
**Recommendation:** use immutable generated IDs plus canonical typed keys.  
**Status:** OPEN

### KAI-COGSTATE-013 — MEDIUM — Process-local causal singletons are unsynchronised
**Issue:** graph, simulator, policy memory and detector instances are global mutable objects without locks or cross-worker authority.  
**Risk:** concurrent writes race and each worker develops a different world model.  
**Recommendation:** centralise state in a transactional service.  
**Status:** OPEN

### KAI-COGSTATE-014 — MEDIUM — Simulation configuration is unbounded
**Issue:** scenario horizon, variations, nested state/actions and surprise threshold are not validated.  
**Risk:** future activation can trigger excessive work or inverted surprise behaviour.  
**Recommendation:** define strict resource and numerical limits before activation.  
**Status:** OPEN

---

## Ohana Core: `agentic/moral_core.py`

### KAI-COGSTATE-015 — HIGH — Moral policy file is unauthenticated
**Issue:** the fingerprint is loaded from a normal relative JSON file with no signature, ownership, permission, symlink or trusted-revision verification.  
**Risk:** filesystem modification changes the moral alignment model.  
**Recommendation:** store signed immutable operator-approved policy in a protected authority.  
**Status:** OPEN

### KAI-COGSTATE-016 — HIGH — One decision becomes a durable moral stance
**Issue:** `record_decision` immediately writes the caller’s decision into `situational_stances`, regardless of whether the decision was actually made by the operator, successful, corrected or harmful.  
**Risk:** one poisoned/misinterpreted event changes future moral alignment.  
**Recommendation:** require authenticated operator confirmation and outcome review before learning a stance.  
**Status:** OPEN

### KAI-COGSTATE-017 — HIGH — Situation keys overwrite prior values
**Issue:** a caller controls `type`/`domain`; the resulting string is the dictionary key and replaces any existing stance.  
**Risk:** an arbitrary request can overwrite a high-value moral category such as `family` or `safety`.  
**Recommendation:** use an approved ontology and append immutable reviewed decisions.  
**Status:** OPEN

### KAI-COGSTATE-018 — HIGH — Boundary matching is linguistically unsafe
**Issue:** a boundary blocks if any of its first three words appears as a substring anywhere in JSON-serialised action text.  
**Risk:** paraphrased harmful actions bypass the gate, while harmless actions containing common words are blocked.  
**Recommendation:** enforce structured policy predicates and semantic review.  
**Status:** OPEN

### KAI-COGSTATE-019 — HIGH — Loyalty scoring is one-directional
**Issue:** each loyalty can only add to a baseline 0.5 score; there is no conflict, trade-off or negative evidence analysis.  
**Risk:** merely mentioning loyalty words makes an action appear more aligned, including actions that harm those values.  
**Recommendation:** evaluate consequences and conflicts, not keyword presence.  
**Status:** OPEN

### KAI-COGSTATE-020 — HIGH — Graph failure falls back permissively
**Issue:** any Wisdom Graph import/evaluation exception is silently ignored; fingerprint scoring then proceeds from the permissive 0.5 baseline.  
**Risk:** alignment checks become less restrictive when the richer governance authority fails.  
**Recommendation:** return alignment-unavailable and block/escalate consequential actions.  
**Status:** OPEN

### KAI-COGSTATE-021 — HIGH — Corruption silently resets morality
**Issue:** malformed fingerprint files are warning-logged and replaced with hard-coded defaults.  
**Risk:** learned/operator-approved boundaries disappear without blocking readiness.  
**Recommendation:** quarantine corruption and fail closed until restored from a verified revision.  
**Status:** OPEN

### KAI-COGSTATE-022 — HIGH — Moral persistence is unsafe
**Issue:** `_save_fingerprint` rewrites the file directly without atomic replacement, locking, fsync, version checks or exception handling at the public mutation boundary.  
**Risk:** concurrent writes or interruption corrupt/lose moral state and can crash callers after partial mutation.  
**Recommendation:** use transactional versioned storage and acknowledge only durable commits.  
**Status:** OPEN

### KAI-COGSTATE-023 — MEDIUM — Fingerprint values are unvalidated
**Issue:** lists, stance dictionaries, flexibility, fairness and override weights are accepted directly from JSON.  
**Risk:** NaN, negative/extreme values and oversized content distort progress and future alignment.  
**Recommendation:** validate a strict versioned schema.  
**Status:** OPEN

### KAI-COGSTATE-024 — MEDIUM — Situation is ignored in context building
**Issue:** `build_moral_context` does not use the supplied situation to select relevant stances; it takes the first five insertion-ordered stances.  
**Risk:** unrelated or stale moral statements shape the current decision.  
**Recommendation:** retrieve only provenance-backed context applicable to the exact situation.  
**Status:** OPEN

### KAI-COGSTATE-025 — MEDIUM — Decision history is volatile and unbounded
**Issue:** `_decision_log` grows without a cap and is not persisted; entries may contain arbitrary nested situation data.  
**Risk:** memory grows while forensic history disappears on restart and differs by worker.  
**Recommendation:** use bounded governed event storage.  
**Status:** OPEN

### KAI-COGSTATE-026 — MEDIUM — Stub status conflicts with active mutation
**Issue:** `can_operate` always returns false and `inject_into_prompt` is a no-op, but `record_decision` and `evaluate_action_alignment` actively persist and score state.  
**Risk:** operators may believe the subsystem is inactive while it changes future alignment outputs.  
**Recommendation:** expose separate collecting, mutating, evaluating and enforcing states.  
**Status:** OPEN

---

## Cognitive Fingerprint: `agentic/cognitive_fingerprint.py`

### KAI-COGSTATE-027 — HIGH — Raw behavioural logging defaults on
**Issue:** the feature is documented as enabled by default and `record` writes raw query text, session ID and inferred behavioural labels to `/data/cognitive_fingerprint.jsonl`.  
**Risk:** sensitive operator conversations and behavioural profiling data accumulate in plaintext by default.  
**Recommendation:** require explicit consent, minimise content and encrypt governed telemetry.  
**Status:** OPEN

### KAI-COGSTATE-028 — HIGH — No privacy or principal boundary
**Issue:** records have no authenticated user ID, tenant separation, retention, deletion, purpose, consent or access-control metadata.  
**Risk:** multiple users/sessions can be combined into one behavioural model and retained indefinitely.  
**Recommendation:** partition by principal and implement purpose-limited retention/deletion controls.  
**Status:** OPEN

### KAI-COGSTATE-029 — HIGH — Line count is treated as sample count
**Issue:** `sample_count` counts every line without parsing or schema validation. Empty, corrupt, duplicate or attacker-appended lines all count.  
**Risk:** readiness is based on file length rather than valid independent interactions.  
**Recommendation:** validate signed unique sample records and count accepted records only.  
**Status:** OPEN

### KAI-COGSTATE-030 — HIGH — Readiness threshold is forgeable
**Issue:** appending 90 arbitrary lines makes `can_infer` true, even though `infer` remains a stub.  
**Risk:** dependent activation conditions can be satisfied by tampering or corruption.  
**Recommendation:** derive readiness from validated sample quality, diversity and principal provenance.  
**Status:** OPEN

### KAI-COGSTATE-031 — MEDIUM — JSONL append races
**Issue:** multiple workers append without locks or durable flush/fsync; initial directory/file handling is not coordinated.  
**Risk:** records interleave, disappear or corrupt.  
**Recommendation:** use a transactional telemetry store.  
**Status:** OPEN

### KAI-COGSTATE-032 — MEDIUM — Collection failure is invisible
**Issue:** every write exception is reduced to a debug log and callers receive no failure state.  
**Risk:** sample completeness and readiness are unknowable.  
**Recommendation:** expose dropped-record metrics and durable delivery status.  
**Status:** OPEN

### KAI-COGSTATE-033 — MEDIUM — Behaviour inference is substring-based
**Issue:** phrases such as `yes`, `ok`, `try`, `risk`, `plan` and query length directly determine decisions, risk tolerance, horizon and response preference without linguistic context.  
**Risk:** ordinary text, negation and quoted words produce incorrect psychological labels.  
**Recommendation:** treat these as unverified observations and require calibrated inference/human review.  
**Status:** OPEN

### KAI-COGSTATE-034 — MEDIUM — Missing flag authority does not block readiness
**Issue:** `can_infer` catches `ImportError` and continues to the sample threshold.  
**Risk:** inference readiness can be reported when feature governance is unavailable.  
**Recommendation:** fail closed on missing control infrastructure.  
**Status:** OPEN

### KAI-COGSTATE-035 — MEDIUM — Cached count is cross-process stale
**Issue:** `_sample_count` is invalidated only when this collector instance writes. Other workers or external file changes are not observed.  
**Risk:** progress/readiness differs across processes.  
**Recommendation:** use authoritative storage/count queries with revisions.  
**Status:** OPEN

### KAI-COGSTATE-036 — MEDIUM — Sample schema is unbounded
**Issue:** enum-like fields, session ID and `extra` accept arbitrary values and nested size; only query is truncated.  
**Risk:** malformed records and large metadata pollute future inference and storage.  
**Recommendation:** enforce strict enums, finite bounds and allowed metadata fields.  
**Status:** OPEN

---

## Batch totals

- Findings: **36**
- Critical: **0**
- High: **19**
- Medium: **17**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,000**
- Critical: **87**
- High: **391**
- Medium: **519**
- Low: **3**

## Files materially reviewed in this batch

`agentic/global_workspace.py`, `agentic/causal_world_model.py`, `agentic/moral_core.py`, `agentic/cognitive_fingerprint.py`, with active-path confirmation against `agentic/app.py`. Existing Wisdom Graph findings remain in `CODE_AUDIT_BATCH_AUTONOMOUS_STATE.md` and were not duplicated.
