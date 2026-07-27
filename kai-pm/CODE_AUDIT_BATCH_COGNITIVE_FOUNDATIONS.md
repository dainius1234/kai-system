# Kai Code Audit — Cognitive Foundations Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-COGFOUND-001 | HIGH | Active agent modules submit Global Workspace bids that are silently discarded |
| KAI-COGFOUND-002 | HIGH | Workspace interfaces create false operational awareness despite permanent no-op behaviour |
| KAI-COGFOUND-003 | MEDIUM | The documented active-bidder activation condition cannot be measured |
| KAI-COGFOUND-004 | MEDIUM | Workspace bid and moment fields lack schema/range/size validation |
| KAI-COGFOUND-005 | MEDIUM | Workspace subscribers and state are unsynchronised process globals |
| KAI-COGFOUND-006 | MEDIUM | Stream access always returns empty despite submitted activity |
| KAI-COGFOUND-007 | MEDIUM | Subscriber registration is unbounded and unauthorised |
| KAI-COGFOUND-008 | MEDIUM | Workspace capacity and cycle configuration are unvalidated and unused |
| KAI-COGFOUND-009 | HIGH | Proactive surprise detection calls a nonexistent detector method and fails silently |
| KAI-COGFOUND-010 | HIGH | Heuristic sensor correlations are stored as causal edges without validation |
| KAI-COGFOUND-011 | HIGH | Repeated causal observations overwrite rather than accumulate evidence |
| KAI-COGFOUND-012 | HIGH | Causal queries and predictions remain empty while edge counts imply a functioning model |
| KAI-COGFOUND-013 | HIGH | Policy Memory accepts policies but can never retrieve or update them |
| KAI-COGFOUND-014 | HIGH | Causal/policy mutation methods do not enforce feature or readiness gates |
| KAI-COGFOUND-015 | MEDIUM | Causal state is volatile and lost on restart |
| KAI-COGFOUND-016 | MEDIUM | Causal graph and policy singletons are concurrency-unsafe |
| KAI-COGFOUND-017 | MEDIUM | Edge identity collapses context, direction and multiple observations |
| KAI-COGFOUND-018 | MEDIUM | Causal numerical fields and direction/source enums are unvalidated |
| KAI-COGFOUND-019 | MEDIUM | Causal provenance and evidence counts are caller assertions |
| KAI-COGFOUND-020 | MEDIUM | Simulation sizes, horizons and utility values are unvalidated |
| KAI-COGFOUND-021 | MEDIUM | Simulation APIs return normal empty results rather than unavailable states |
| KAI-COGFOUND-022 | MEDIUM | Two separate policy-memory implementations create conflicting authorities |
| KAI-COGFOUND-023 | HIGH | Ohana records and persists moral decisions despite claiming Phase-0 no-op operation |
| KAI-COGFOUND-024 | HIGH | Plain local fingerprint data directly influences governance alignment |
| KAI-COGFOUND-025 | HIGH | Arbitrary decisions overwrite situational moral stances without source authentication |
| KAI-COGFOUND-026 | HIGH | Harm boundaries use common-word substring matching and cause false blocks |
| KAI-COGFOUND-027 | HIGH | Paraphrased harmful actions evade boundary matching |
| KAI-COGFOUND-028 | HIGH | Loyalty keywords can only increase alignment, never represent conflict or uncertainty |
| KAI-COGFOUND-029 | HIGH | Unknown actions default to neutral approval rather than requiring review |
| KAI-COGFOUND-030 | HIGH | Wisdom-graph failures are silently removed from alignment decisions |
| KAI-COGFOUND-031 | HIGH | Moral learning and alignment ignore `can_operate` and feature-state checks |
| KAI-COGFOUND-032 | MEDIUM | One global moral fingerprint is shared across users and sessions |
| KAI-COGFOUND-033 | MEDIUM | Situation input is ignored when building moral context |
| KAI-COGFOUND-034 | MEDIUM | Stored stances can become future system-prompt injection content |
| KAI-COGFOUND-035 | MEDIUM | Moral fingerprint persistence is non-atomic and concurrency-unsafe |
| KAI-COGFOUND-036 | MEDIUM | Corrupt moral state silently resets to permissive defaults |
| KAI-COGFOUND-037 | MEDIUM | Moral decision, stance and situation content is unbounded |
| KAI-COGFOUND-038 | HIGH | Cognitive fingerprint collection defaults on and stores query/session data in plaintext |
| KAI-COGFOUND-039 | HIGH | Blank or malformed lines count toward the inference threshold |
| KAI-COGFOUND-040 | HIGH | Progress reports readiness even though inference remains a stub |
| KAI-COGFOUND-041 | HIGH | Inference always returns zero-confidence placeholder output after threshold |
| KAI-COGFOUND-042 | MEDIUM | Fingerprint append operations are not multi-process safe or durable |
| KAI-COGFOUND-043 | MEDIUM | Collection failures are silently suppressed |
| KAI-COGFOUND-044 | MEDIUM | Cached sample counts race writes and external file changes |
| KAI-COGFOUND-045 | MEDIUM | Feature-flag import failure defaults collection/inference gating to enabled |
| KAI-COGFOUND-046 | MEDIUM | Substring heuristics misclassify decisions, risk and intent |
| KAI-COGFOUND-047 | MEDIUM | Response-length preference is inferred from query length, not response feedback |
| KAI-COGFOUND-048 | MEDIUM | Fingerprint data has no encryption, permission or retention controls |
| KAI-COGFOUND-049 | MEDIUM | All sessions contribute to one global behavioural fingerprint |
| KAI-COGFOUND-050 | MEDIUM | Interaction samples and extra metadata are weakly bounded |

---

## Global Workspace: `agentic/global_workspace.py`

### KAI-COGFOUND-001 — HIGH — Live bids are discarded
**Issue:** Active agentic code submits anomaly and Cortex bids to `get_global_workspace().submit_bid`. The method only logs and never appends to `_bid_queue`.  
**Risk:** The system behaves and documents itself as broadcasting situational awareness while every contribution is lost. Downstream safety/design assumptions about unified awareness are false.  
**Recommendation:** keep the feature explicitly unavailable and prohibit callers from reporting successful bid submission until an operational, tested queue exists.  
**Status:** OPEN

### KAI-COGFOUND-002 — HIGH — No-op interfaces impersonate capability
**Issue:** subscribe, submit, select, broadcast and stream interfaces are callable; subscribers are stored, but selection/broadcast/stream always do nothing.  
**Risk:** integration tests and introspection can infer that modules are wired while no conscious-content or safety information propagates.  
**Recommendation:** return explicit disabled/not-implemented results and fail readiness for consumers requiring the workspace.  
**Status:** OPEN

### KAI-COGFOUND-003 — MEDIUM — Active bidder condition is unimplementable
**Issue:** documentation requires at least three active bidder modules, but the class has no bidder registry; discarded submissions cannot establish activity.  
**Risk:** activation progress cannot be truthfully evaluated.  
**Recommendation:** define authenticated bidder registration and freshness before activation.  
**Status:** OPEN

### KAI-COGFOUND-004 — MEDIUM — Workspace objects are unvalidated
**Issue:** urgency, relevance, surprise, confidence, emotional salience, valence, timestamps, content and context are plain dataclass fields with no finite/range/size checks.  
**Risk:** invalid future inputs can distort salience, memory and output.  
**Recommendation:** use strict immutable schemas.  
**Status:** OPEN

### KAI-COGFOUND-005 — MEDIUM — Shared state is unsafe
**Issue:** subscribers, bids and stream lists are unsynchronised process memory.  
**Risk:** concurrent registration/broadcast would race and workers would maintain separate awareness streams.  
**Recommendation:** use one authoritative event loop/store with atomic snapshots.  
**Status:** OPEN

### KAI-COGFOUND-006 — MEDIUM — Activity is invisible
**Issue:** `get_stream` and `get_latest_moment` always return empty/None while `stream_length` reports the private list length, which never changes.  
**Risk:** monitoring cannot distinguish inactivity from nonimplementation.  
**Recommendation:** expose an explicit disabled state.  
**Status:** OPEN

### KAI-COGFOUND-007 — MEDIUM — Subscriber registry is ungoverned
**Issue:** any internal caller can register/replace arbitrary callback names; no maximum, identity or lifecycle cleanup exists.  
**Risk:** future activation permits callback hijacking, leaks and unbounded fan-out.  
**Recommendation:** use an approved immutable module registry.  
**Status:** OPEN

### KAI-COGFOUND-008 — MEDIUM — Configuration is cosmetic
**Issue:** max stream length and cycle milliseconds are accepted without validation but unused.  
**Risk:** introspection reports settings that do not constrain operation and unsafe values await activation.  
**Recommendation:** validate and test configuration only when implementation exists.  
**Status:** OPEN

---

## Causal world model: `agentic/causal_world_model.py`

### KAI-COGFOUND-009 — HIGH — Surprise detection method mismatch
**Issue:** the active proactive observer calls `detector.check(...)`; `CausalSurpriseDetector` implements only `check_surprise(...)`. The call raises `AttributeError`, which the observer catches and suppresses.  
**Risk:** the advertised prediction-error feedback loop never runs while remaining operationally invisible.  
**Recommendation:** enforce interface tests and expose causal-surprise readiness/failure.  
**Status:** OPEN

### KAI-COGFOUND-010 — HIGH — Heuristics become causal facts
**Issue:** active sensor-correlation code creates `CausalEdge` records from simultaneous keyword observations and stores them directly through `add_edge`. No statistical test, provenance signature or human validation is required.  
**Risk:** co-occurrence and poisoned sensor text become causal relationships used by future reasoning.  
**Recommendation:** store them as unverified hypotheses with evidence and validation requirements.  
**Status:** OPEN

### KAI-COGFOUND-011 — HIGH — Evidence is overwritten
**Issue:** edge ID is solely `source->target`. A later observation replaces the entire edge instead of incrementing evidence or preserving history.  
**Risk:** confidence/source/context can be silently rewritten by the latest caller and evidence count does not reflect observations.  
**Recommendation:** append immutable observations and derive versioned edges transactionally.  
**Status:** OPEN

### KAI-COGFOUND-012 — HIGH — Edge count implies nonexistent reasoning
**Issue:** edges can be added and counted, but path queries, upstream/downstream queries and outcome prediction always return empty.  
**Risk:** introspection can show a growing causal model while no causal inference is possible.  
**Recommendation:** distinguish observation-store size from operational reasoning readiness.  
**Status:** OPEN

### KAI-COGFOUND-013 — HIGH — Policy Memory is write-only
**Issue:** policies can be added, but `get_relevant_policies` always returns empty and `update_policy_success` is a no-op.  
**Risk:** callers believe strategies are learned/stored while none can be retrieved or evaluated.  
**Recommendation:** mark the component unavailable until full lifecycle exists.  
**Status:** OPEN

### KAI-COGFOUND-014 — HIGH — Mutation bypasses readiness
**Issue:** `add_edge` and `add_policy` do not check feature flags, GPU/data thresholds or `can_*` gates.  
**Risk:** unvalidated state accumulates while the system says the model cannot operate, then may be consumed after later activation.  
**Recommendation:** quarantine pre-activation observations separately with explicit provenance.  
**Status:** OPEN

### KAI-COGFOUND-015 — MEDIUM — Causal state is volatile
**Issue:** edges and policies exist only in process memory.  
**Risk:** restart erases the causal model and different workers disagree.  
**Recommendation:** use durable versioned storage before presenting learned state.  
**Status:** OPEN

### KAI-COGFOUND-016 — MEDIUM — Singleton mutation races
**Issue:** dictionaries and singleton construction have no locking.  
**Risk:** concurrent updates are lost and reads observe inconsistent objects.  
**Recommendation:** use transactional single-writer state.  
**Status:** OPEN

### KAI-COGFOUND-017 — MEDIUM — Edge identity omits semantics
**Issue:** direction, context modifiers, temporal lag and source type do not participate in the ID.  
**Risk:** direct/inverse or context-specific relationships overwrite one another.  
**Recommendation:** model immutable qualified relation identity.  
**Status:** OPEN

### KAI-COGFOUND-018 — MEDIUM — Causal schemas are unconstrained
**Issue:** strength, confidence, lag, success rate and utility can be negative, non-finite or extreme; direction/source/evidence types accept arbitrary strings.  
**Risk:** invalid future inference and non-standard serialization.  
**Recommendation:** enforce strict finite enums/ranges.  
**Status:** OPEN

### KAI-COGFOUND-019 — MEDIUM — Provenance is self-declared
**Issue:** source type, evidence count, update time and confidence are caller-provided fields with no identity or evidence link.  
**Risk:** inferred/simulated relations can be represented as observed high-confidence facts.  
**Recommendation:** generate provenance server-side from authenticated evidence records.  
**Status:** OPEN

### KAI-COGFOUND-020 — MEDIUM — Simulation complexity is unbounded
**Issue:** horizon steps, variations per action, action/state sizes and utility/confidence lack limits.  
**Risk:** future activation can create uncontrolled GPU/CPU work and invalid results.  
**Recommendation:** define bounded simulation budgets and admission policy.  
**Status:** OPEN

### KAI-COGFOUND-021 — MEDIUM — Stubs return success-like emptiness
**Issue:** simulator and surprise methods return empty lists/zero/None rather than an unavailable result.  
**Risk:** callers cannot distinguish no scenario/surprise from no implementation.  
**Recommendation:** use typed disabled states.  
**Status:** OPEN

### KAI-COGFOUND-022 — MEDIUM — Duplicate policy authorities
**Issue:** this module defines in-memory `PolicyMemory`, while `agentic/policy_memory.py` is described as the persisted production version. No ownership or synchronisation contract exists.  
**Risk:** callers store/read different policy sets depending on import path.  
**Recommendation:** designate one authoritative implementation.  
**Status:** OPEN

---

## Ohana moral core: `agentic/moral_core.py`

### KAI-COGFOUND-023 — HIGH — Active mutation contradicts no-op claim
**Issue:** documentation says Phase 0 operations are no-ops and `can_operate` always returns false, yet `record_decision` changes the fingerprint and writes it to disk; `evaluate_action_alignment` actively returns governance scores.  
**Risk:** unready experimental moral state already affects trust decisions without activation controls.  
**Recommendation:** enforce readiness at every mutation/evaluation boundary.  
**Status:** OPEN

### KAI-COGFOUND-024 — HIGH — Tamperable file controls alignment
**Issue:** the fingerprint is ordinary JSON at a relative local path, without authentication, permissions, signature, revision or protected trust root.  
**Risk:** filesystem writers can alter loyalties, boundaries, flexibility and stances that influence autonomous-action alignment.  
**Recommendation:** store signed policy state in protected transactional governance storage.  
**Status:** OPEN

### KAI-COGFOUND-025 — HIGH — Stances are unauthenticated overwrites
**Issue:** `record_decision` derives a key from caller-controlled `type/domain` and stores the caller’s decision directly, overwriting the prior stance.  
**Risk:** any internal caller can teach/replace moral policy and persistent prompt content.  
**Recommendation:** accept only authenticated operator-confirmed decisions with history/versioning.  
**Status:** OPEN

### KAI-COGFOUND-026 — HIGH — Common words trigger hard blocks
**Issue:** a boundary blocks if any of its first three whitespace words is a substring of JSON action text. Default “no violence against innocents” therefore tests words such as `no` and `against`.  
**Risk:** unrelated actions containing common substrings can be assigned alignment 0.0.  
**Recommendation:** use structured action predicates and explicit harm taxonomy.  
**Status:** OPEN

### KAI-COGFOUND-027 — HIGH — Harm is easy to paraphrase
**Issue:** boundary checks depend on literal substrings and inspect only up to three words.  
**Risk:** synonymous/obfuscated harmful actions avoid all boundaries and receive neutral/positive alignment.  
**Recommendation:** combine strict policy predicates with verified semantic classification and human escalation.  
**Status:** OPEN

### KAI-COGFOUND-028 — HIGH — Loyalty scoring is one-directional
**Issue:** matching loyalty words only raises the baseline from 0.5 toward 1.0; conflicting loyalties, trade-offs, uncertainty and negative evidence are not represented.  
**Risk:** mentioning family/safety/autonomy can inflate alignment regardless of action consequences.  
**Recommendation:** evaluate structured positive/negative impacts with provenance.  
**Status:** OPEN

### KAI-COGFOUND-029 — HIGH — Unknown consequential actions receive neutral approval
**Issue:** unmatched actions return at least 0.5. Existing trust integration blocks only exact 0.0 and warns for low-but-positive scores.  
**Risk:** unanalysed actions pass the moral gate rather than requiring review.  
**Recommendation:** treat unknown alignment as unavailable/needs approval.  
**Status:** OPEN

### KAI-COGFOUND-030 — HIGH — Graph control fails silently
**Issue:** every wisdom-graph exception is suppressed, leaving fingerprint keyword scoring as if no control failed.  
**Risk:** alignment becomes more permissive when the richer boundary authority is broken.  
**Recommendation:** fail closed/escalate on required governance dependency failure.  
**Status:** OPEN

### KAI-COGFOUND-031 — HIGH — Feature and readiness gates are ignored
**Issue:** `record_decision`, build context and evaluate alignment never call `can_operate` or `FF_OHANA_CORE`.  
**Risk:** route-level flags cannot disable direct internal use.  
**Recommendation:** enforce policy within the moral authority itself.  
**Status:** OPEN

### KAI-COGFOUND-032 — MEDIUM — Global operator identity
**Issue:** one singleton/file represents the operator and all sessions; no authenticated user key exists.  
**Risk:** multi-user/test data contaminates the same moral model.  
**Recommendation:** bind governance state to explicit authorised principal.  
**Status:** OPEN

### KAI-COGFOUND-033 — MEDIUM — Situation does not shape context
**Issue:** `build_moral_context` accepts `situation` but ignores it, returning the same first stances/loyalties.  
**Risk:** irrelevant stances are injected while relevant context is absent.  
**Recommendation:** select verified applicable policies from structured situation data.  
**Status:** OPEN

### KAI-COGFOUND-034 — MEDIUM — Persistent prompt-injection path
**Issue:** arbitrary decision strings are stored as stances and `MoralContext.to_prompt` renders them into a system-level block.  
**Risk:** a poisoned recorded decision can become privileged model instruction after activation.  
**Recommendation:** constrain stance schemas and render data as quoted untrusted evidence.  
**Status:** OPEN

### KAI-COGFOUND-035 — MEDIUM — Persistence is unsafe
**Issue:** the complete fingerprint is directly rewritten without lock, temporary file, fsync or compare-and-swap.  
**Risk:** concurrent decisions lose updates or corrupt governance state.  
**Recommendation:** use atomic transactional storage.  
**Status:** OPEN

### KAI-COGFOUND-036 — MEDIUM — Corruption restores permissive defaults
**Issue:** any load error logs a warning and returns default fingerprint values, including loyalty override 1.0 and rule flexibility 0.9.  
**Risk:** tampering/corruption removes learned boundaries and continues without forensic recovery.  
**Recommendation:** quarantine and lock governance on corruption.  
**Status:** OPEN

### KAI-COGFOUND-037 — MEDIUM — Moral records are unbounded
**Issue:** situation dictionaries, decision/outcome text, stance keys/values and lists have no size/depth/format limits.  
**Risk:** memory/disk/prompt growth and malformed state.  
**Recommendation:** enforce strict bounded schemas.  
**Status:** OPEN

---

## Cognitive fingerprint: `agentic/cognitive_fingerprint.py`

### KAI-COGFOUND-038 — HIGH — Default-on behavioural surveillance
**Issue:** collection is documented default enabled and writes raw query excerpts, session ID, inferred risk/decision/style and arbitrary extra metadata to `/data/cognitive_fingerprint.jsonl` without encryption or access control enforcement.  
**Risk:** sensitive interaction and behavioural profiling accumulates automatically.  
**Recommendation:** require explicit consent, minimise/pseudonymise data and protect storage.  
**Status:** OPEN

### KAI-COGFOUND-039 — HIGH — Junk reaches the readiness threshold
**Issue:** `sample_count` counts every line without parsing/validating it. Blank, corrupt or attacker-added lines count toward 90 samples.  
**Risk:** readiness can be reached with no valid interaction evidence.  
**Recommendation:** count only authenticated validated unique records.  
**Status:** OPEN

### KAI-COGFOUND-040 — HIGH — Progress claims readiness for nonexistent inference
**Issue:** `progress.ready_for_inference` is only `count >= 90`, ignoring the feature flag and the fact that `infer` remains an unimplemented GPU placeholder.  
**Risk:** dashboards/callers believe behavioural inference is ready.  
**Recommendation:** separate data threshold from implemented/configured/operational readiness.  
**Status:** OPEN

### KAI-COGFOUND-041 — HIGH — Threshold output remains a stub
**Issue:** after 90 samples, `infer` returns `stub_pending_gpu_clustering`, confidence 0 and default dimensions.  
**Risk:** downstream code may consume a normal fingerprint object despite no inference.  
**Recommendation:** return an explicit unavailable result.  
**Status:** OPEN

### KAI-COGFOUND-042 — MEDIUM — Append is not durable/concurrent-safe
**Issue:** multiple processes append to one JSONL file without locking, flush/fsync, event IDs or integrity chain.  
**Risk:** lines interleave/disappear and the threshold/profile dataset is unauditable.  
**Recommendation:** use a transactional append-only event store.  
**Status:** OPEN

### KAI-COGFOUND-043 — MEDIUM — Collection failures disappear
**Issue:** write failures are debug logged and do not affect callers/readiness.  
**Risk:** missing biased samples go undetected.  
**Recommendation:** expose dropped-event metrics and dataset completeness.  
**Status:** OPEN

### KAI-COGFOUND-044 — MEDIUM — Count cache becomes stale
**Issue:** `_sample_count` is invalidated only by this collector’s successful record call; external/multi-worker writes are not seen and concurrent calls race.  
**Risk:** readiness differs by process and time.  
**Recommendation:** use authoritative indexed storage.  
**Status:** OPEN

### KAI-COGFOUND-045 — MEDIUM — Feature import failure is permissive
**Issue:** `can_infer` imports `feature_flags` from a non-package path; on `ImportError` it skips the flag check and relies only on count.  
**Risk:** broken flag integration enables inference readiness rather than disabling it.  
**Recommendation:** import the authoritative module and fail closed.  
**Status:** OPEN

### KAI-COGFOUND-046 — MEDIUM — Behaviour heuristics use raw substrings
**Issue:** terms such as `ok`, `yes`, `try`, `risk`, `plan` are matched anywhere in the lowercased query without boundaries or context.  
**Risk:** words containing those character sequences and negated statements create false decisions/risk profiles.  
**Recommendation:** use explicit feedback/outcome events and validated classification.  
**Status:** OPEN

### KAI-COGFOUND-047 — MEDIUM — Preference inference uses the wrong signal
**Issue:** response-length preference is inferred solely from query character length, not from requested length, corrections or user feedback on responses.  
**Risk:** the learned preference is systematically unrelated to the claimed attribute.  
**Recommendation:** learn from explicit preference and observed accepted response behaviour.  
**Status:** OPEN

### KAI-COGFOUND-048 — MEDIUM — No data lifecycle controls
**Issue:** the log has no retention, deletion, purpose limitation, permission hardening or encryption.  
**Risk:** behavioural records accumulate indefinitely.  
**Recommendation:** implement consent-linked retention and secure deletion.  
**Status:** OPEN

### KAI-COGFOUND-049 — MEDIUM — Cross-session behavioural mixing
**Issue:** session ID is stored but all lines contribute to one global count/future fingerprint; no user partition exists.  
**Risk:** tests, other users and unrelated contexts alter the operator model.  
**Recommendation:** partition and scope datasets explicitly.  
**Status:** OPEN

### KAI-COGFOUND-050 — MEDIUM — Sample content is weakly bounded
**Issue:** quick query is sliced to 200, but direct `InteractionSample` construction and `extra` nested data have no limits or enum validation.  
**Risk:** internal callers write oversized/malformed behavioural records.  
**Recommendation:** validate strict bounded sample schemas at `record`.  
**Status:** OPEN

---

## Batch totals

- Findings: **50**
- Critical: **0**
- High: **21**
- Medium: **29**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,014**
- Critical: **87**
- High: **393**
- Medium: **531**
- Low: **3**

## Files materially reviewed in this batch

`agentic/global_workspace.py`, `agentic/causal_world_model.py`, `agentic/moral_core.py`, `agentic/cognitive_fingerprint.py`, with integration confirmation against `agentic/app.py` and existing trust integration findings.
