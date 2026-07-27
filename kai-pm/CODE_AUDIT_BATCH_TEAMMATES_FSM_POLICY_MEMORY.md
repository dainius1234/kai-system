# Kai Code Audit — Teammates, FSM and Policy Memory Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-STATECTRL-001 | HIGH | Unsigned Markdown files become teammate system prompts |
| KAI-STATECTRL-002 | HIGH | Teammate prompt activation has no approval, digest or version control |
| KAI-STATECTRL-003 | HIGH | Registry reload is unsynchronised and replaces all active teammates globally |
| KAI-STATECTRL-004 | HIGH | Missing System Prompt heading promotes the entire Markdown file to system instruction |
| KAI-STATECTRL-005 | MEDIUM | Teammate file count and content size are unbounded |
| KAI-STATECTRL-006 | MEDIUM | Individual parse/read failures silently remove teammates from the registry |
| KAI-STATECTRL-007 | MEDIUM | One teammate registry and persona set is shared across every user/session |
| KAI-STATECTRL-008 | MEDIUM | Teammate names, specialties and descriptions are exposed through public integration routes |
| KAI-STATECTRL-009 | MEDIUM | Teammate loading performs synchronous filesystem work |
| KAI-STATECTRL-010 | MEDIUM | Teammate metadata and prompts lack schema/content validation |
| KAI-STATECTRL-011 | HIGH | Critical anomalies are silently ignored while the FSM is FOCUSED |
| KAI-STATECTRL-012 | HIGH | One service-restored event clears DEGRADED state for the entire fleet |
| KAI-STATECTRL-013 | HIGH | Healing completion always returns IDLE and loses the pre-failure state |
| KAI-STATECTRL-014 | HIGH | Focus exit always returns IDLE even when an active user session exists |
| KAI-STATECTRL-015 | HIGH | Any internal caller can fire operational state events without source authority |
| KAI-STATECTRL-016 | HIGH | FSM state is process-local and inconsistent across workers |
| KAI-STATECTRL-017 | MEDIUM | Undefined transitions are silently ignored without degraded evidence |
| KAI-STATECTRL-018 | MEDIUM | Additional failures during RECOVERING are ignored |
| KAI-STATECTRL-019 | MEDIUM | Snapshot reads are not protected by the FSM lock |
| KAI-STATECTRL-020 | MEDIUM | Events carry no service identity, incident ID, reason or severity |
| KAI-STATECTRL-021 | MEDIUM | Transition history lacks timestamps and durable audit evidence |
| KAI-STATECTRL-022 | MEDIUM | FSM state is not tied to real service readiness or freshness |
| KAI-STATECTRL-023 | HIGH | Policies can be stored while Policy Memory reports distillation disabled |
| KAI-STATECTRL-024 | HIGH | Plain local JSONL policies can become future privileged planning instructions |
| KAI-STATECTRL-025 | HIGH | Policy creation has no authenticated author or evidence provenance |
| KAI-STATECTRL-026 | HIGH | Policy IDs are collision-prone and non-reproducible across processes |
| KAI-STATECTRL-027 | HIGH | `validate` records no validation outcome or confidence update |
| KAI-STATECTRL-028 | HIGH | Retrieval defaults to accepting zero-confidence unvalidated policies |
| KAI-STATECTRL-029 | HIGH | One shared keyword can make an unrelated policy relevant |
| KAI-STATECTRL-030 | HIGH | Policy store reports normal completion after persistence failure |
| KAI-STATECTRL-031 | HIGH | Policies have no revocation, supersession or enforced version lifecycle |
| KAI-STATECTRL-032 | MEDIUM | JSONL appends are non-atomic and multi-process unsafe |
| KAI-STATECTRL-033 | MEDIUM | Storing a policy mutates the caller-owned object |
| KAI-STATECTRL-034 | MEDIUM | Policy confidence, utility, counts, enums and timestamps are unvalidated |
| KAI-STATECTRL-035 | MEDIUM | Malformed policy lines are silently skipped |
| KAI-STATECTRL-036 | MEDIUM | Full policy retrieval has no pagination or aggregate bound |
| KAI-STATECTRL-037 | MEDIUM | Negative or extreme `top_k` and thresholds have misleading behaviour |
| KAI-STATECTRL-038 | MEDIUM | Policy count includes blank and corrupt records |
| KAI-STATECTRL-039 | MEDIUM | Count cache becomes stale across workers and external writes |
| KAI-STATECTRL-040 | MEDIUM | Feature-flag import failure bypasses the flag check |

---

## Persistent teammates: `agentic/teammates.py`

### KAI-STATECTRL-001 — HIGH — Filesystem content becomes system authority
**Issue:** every `data/teammates/*.md` file is parsed and `system_prompt` is returned in a formatted teammate context used by the active teammate-chat path. No signature, source identity or instruction policy is checked.  
**Risk:** any process/user able to modify the data directory can persist privileged model instructions and impersonate a cognitive teammate.  
**Recommendation:** load only signed reviewed immutable persona artefacts from protected storage.  
**Status:** OPEN

### KAI-STATECTRL-002 — HIGH — Prompt activation is unaudited
**Issue:** no approval state, content digest, version, reviewer, activation timestamp or rollback reference is associated with loaded teammates.  
**Risk:** prompt changes silently become active and cannot be tied to a reviewed revision.  
**Recommendation:** require versioned signed manifests and explicit activation.  
**Status:** OPEN

### KAI-STATECTRL-003 — HIGH — Reload races and replaces the fleet
**Issue:** `load_teammates` builds and assigns a new process-global dictionary without a lock. Concurrent reads/reloads and multiple workers are not coordinated.  
**Risk:** requests see different persona sets, and one partial reload removes every omitted/failed teammate.  
**Recommendation:** atomically publish a validated immutable registry generation across workers.  
**Status:** OPEN

### KAI-STATECTRL-004 — HIGH — Entire file fallback becomes instruction
**Issue:** `system_prompt` initially equals the complete Markdown text. If the exact `## System Prompt` heading is missing/misspelled, metadata, examples and arbitrary content become the system prompt.  
**Risk:** malformed/unreviewed files receive more privileged instruction content than intended.  
**Recommendation:** require a strict schema and reject missing prompt sections.  
**Status:** OPEN

### KAI-STATECTRL-005 — MEDIUM — Corpus is unbounded
**Issue:** every Markdown file is read fully; file count, bytes, prompt tokens and metadata length are unrestricted.  
**Risk:** startup/reload memory and prompt context can be exhausted.  
**Recommendation:** enforce approved file and aggregate limits.  
**Status:** OPEN

### KAI-STATECTRL-006 — MEDIUM — Partial load failure is normalised
**Issue:** failed files are warning logged and omitted; the remaining registry is activated with no completeness/readiness state.  
**Risk:** required safety/domain teammates disappear while the service continues normally.  
**Recommendation:** validate required identities and fail registry activation atomically.  
**Status:** OPEN

### KAI-STATECTRL-007 — MEDIUM — Personas are global
**Issue:** one registry applies to every user and session with no purpose/tenant partition.  
**Risk:** test/custom personas affect unrelated users.  
**Recommendation:** bind persona availability to authenticated policy scopes.  
**Status:** OPEN

### KAI-STATECTRL-008 — MEDIUM — Persona metadata is broadly visible
**Issue:** `list_teammates` returns identity, specialty and descriptions; active agent endpoints expose this registry without a module-level access control.  
**Risk:** callers map internal cognitive roles and target specific prompts/behaviour.  
**Recommendation:** restrict operational persona inventory.  
**Status:** OPEN

### KAI-STATECTRL-009 — MEDIUM — Blocking file load
**Issue:** directory scans and full text reads are synchronous.  
**Risk:** runtime reloads block serving threads/event loops.  
**Recommendation:** load offline or through a bounded worker before atomic activation.  
**Status:** OPEN

### KAI-STATECTRL-010 — MEDIUM — Weak teammate schema
**Issue:** slug/name/specialty/description/system prompt accept arbitrary text; duplicate headings, control characters and prompt length are not validated.  
**Risk:** malformed identity and injection content enters prompt construction.  
**Recommendation:** validate strict bounded fields and content policy.  
**Status:** OPEN

---

## Operational FSM: `agentic/system_fsm.py`

### KAI-STATECTRL-011 — HIGH — Critical anomaly ignored in focus mode
**Issue:** `ANOMALY_CRITICAL` transitions exist only from IDLE and ACTIVE. In FOCUSED it is undefined and `fire` silently returns None.  
**Risk:** PUB/WORK focus mode can remain FOCUSED during a critical anomaly instead of entering DEGRADED.  
**Recommendation:** define safety-precedence transitions from every operational state.  
**Status:** OPEN

### KAI-STATECTRL-012 — HIGH — One restoration clears fleet degradation
**Issue:** `(DEGRADED, SERVICE_RESTORED) -> IDLE` contains no service identity/count or check that all critical dependencies recovered.  
**Risk:** restoration of one service falsely clears degraded state while others remain down.  
**Recommendation:** derive state from authoritative per-service incident sets.  
**Status:** OPEN

### KAI-STATECTRL-013 — HIGH — Recovery loses prior activity/focus
**Issue:** both HEAL_COMPLETE and SERVICE_RESTORED from RECOVERING transition to IDLE. The pre-failure state is not stored.  
**Risk:** an active or focused session is silently reset and state no longer represents actual operation.  
**Recommendation:** maintain a validated state stack/incident model and restore only when conditions hold.  
**Status:** OPEN

### KAI-STATECTRL-014 — HIGH — Focus exit loses active session
**Issue:** FOCUSED + FOCUS_EXIT always becomes IDLE.  
**Risk:** exiting focus during a live interaction incorrectly reports no active session.  
**Recommendation:** track focus as an orthogonal dimension or restore the previous state.  
**Status:** OPEN

### KAI-STATECTRL-015 — HIGH — Event source is untrusted
**Issue:** `fire` accepts any `KaiEvent` from any imported caller; no service identity, signature, authorisation or incident evidence is required.  
**Risk:** compromised modules can set/clear DEGRADED/RECOVERING/FOCUSED state and influence monitoring/behaviour.  
**Recommendation:** accept authenticated typed events from approved authorities.  
**Status:** OPEN

### KAI-STATECTRL-016 — HIGH — State diverges by worker
**Issue:** one singleton FSM lives in process memory.  
**Risk:** concurrent workers report different state/history and process restart resets to IDLE.  
**Recommendation:** use one durable authoritative state machine or derive state from durable facts.  
**Status:** OPEN

### KAI-STATECTRL-017 — MEDIUM — Undefined transitions disappear
**Issue:** missing state/event pairs return None without log, metric or rejected-event record.  
**Risk:** important control events vanish and callers cannot distinguish duplicate/invalid/unsafe transitions.  
**Recommendation:** record every rejected transition with reason and safety severity.  
**Status:** OPEN

### KAI-STATECTRL-018 — MEDIUM — New failure during recovery is ignored
**Issue:** RECOVERING + SERVICE_DOWN/ANOMALY_CRITICAL has no transition.  
**Risk:** escalating or separate incidents do not update state.  
**Recommendation:** maintain incident multiplicity and recovery failure/escalation states.  
**Status:** OPEN

### KAI-STATECTRL-019 — MEDIUM — Snapshot is not locked
**Issue:** state/history are read without acquiring `_lock` while `fire` mutates them.  
**Risk:** snapshots can combine new state with old transition history.  
**Recommendation:** return immutable snapshots under the same lock.  
**Status:** OPEN

### KAI-STATECTRL-020 — MEDIUM — Events are context-free
**Issue:** enum events carry no affected service, actor, incident, cause, evidence, severity or timestamp.  
**Risk:** different failures collapse into one Boolean-like state and cannot be reconciled safely.  
**Recommendation:** use structured immutable incident events.  
**Status:** OPEN

### KAI-STATECTRL-021 — MEDIUM — History is not forensic
**Issue:** recent tuples contain only event/from/to, remain in memory and cap at 100.  
**Risk:** ordering across restarts/hosts and event attribution cannot be reconstructed.  
**Recommendation:** persist timestamped signed transition events.  
**Status:** OPEN

### KAI-STATECTRL-022 — MEDIUM — State is assertion-driven
**Issue:** transitions do not independently verify service health, focus/session lifecycle or recovery outcome.  
**Risk:** false/no-op events produce authoritative-looking state.  
**Recommendation:** gate transitions on durable verified predicates.  
**Status:** OPEN

---

## Policy Memory: `agentic/policy_memory.py`

### KAI-STATECTRL-023 — HIGH — Writes bypass disabled readiness
**Issue:** `can_distill` always returns false, but `store` accepts and persists policies immediately without checking `FF_POLICY_MEMORY` or source type.  
**Risk:** unready/unapproved policies accumulate and may later be surfaced after activation.  
**Recommendation:** quarantine seed policies under an explicit authenticated import workflow.  
**Status:** OPEN

### KAI-STATECTRL-024 — HIGH — Tamperable policies can become prompt guidance
**Issue:** policies are plaintext JSONL at `/data/policies.jsonl`; documentation states relevant high-confidence policies will be prepended to chat system context. No signature or protected root exists.  
**Risk:** filesystem writers can inject future privileged actions/conditions/outcomes.  
**Recommendation:** store signed reviewed policy objects and render them as bounded data, not instructions.  
**Status:** OPEN

### KAI-STATECTRL-025 — HIGH — Policy authorship is absent
**Issue:** `store` accepts any `Policy` from any internal caller; `source` is caller-controlled and no evidence/simulation/outcome reference is required.  
**Risk:** arbitrary advice is represented as simulated, observed or human-labelled policy.  
**Recommendation:** generate source/provenance server-side from authenticated records.  
**Status:** OPEN

### KAI-STATECTRL-026 — HIGH — Weak policy identity
**Issue:** ID uses second-resolution creation time plus only 16 bits from Python’s process-randomised `hash(condition)`.  
**Risk:** same-second/collision overwrites are indistinguishable in consumers, and identical conditions receive different IDs across processes/restarts.  
**Recommendation:** use collision-resistant canonical content IDs and immutable event IDs.  
**Status:** OPEN

### KAI-STATECTRL-027 — HIGH — Validation is fictitious
**Issue:** `validate` only writes a debug log. It does not locate the policy, increment validation count, update confidence/outcome, persist evidence or reject unknown IDs.  
**Risk:** callers can believe real-world validation occurred while the policy remains unchanged.  
**Recommendation:** implement transactional evidence-linked validation or return not implemented.  
**Status:** OPEN

### KAI-STATECTRL-028 — HIGH — Unvalidated policies surface by default
**Issue:** `retrieve_relevant` defaults `min_confidence=0.0`, so a manually stored policy with no validation/confidence is eligible.  
**Risk:** arbitrary seed policies can influence planning immediately.  
**Recommendation:** require verified minimum confidence and validation provenance by risk tier.  
**Status:** OPEN

### KAI-STATECTRL-029 — HIGH — One word creates relevance
**Issue:** context/condition are whitespace token sets; any overlap produces a match, ranked by overlap count plus caller-provided confidence. Action/domain/negation and word boundaries/punctuation are ignored.  
**Risk:** unrelated policies are surfaced from common words and confidence stuffing.  
**Recommendation:** use structured predicates and verified semantic applicability.  
**Status:** OPEN

### KAI-STATECTRL-030 — HIGH — Failed persistence looks successful
**Issue:** all store errors are suppressed and the method returns None in both success/failure cases.  
**Risk:** callers assume policy was stored while no durable record exists.  
**Recommendation:** return a verified commit receipt or typed failure.  
**Status:** OPEN

### KAI-STATECTRL-031 — HIGH — No policy lifecycle
**Issue:** policies are append-only records but duplicate IDs/conditions, supersession, revocation, expiry and active-version selection are not enforced. `version` is only caller metadata.  
**Risk:** outdated/contradictory policies remain simultaneously retrievable.  
**Recommendation:** implement immutable revisions with one authoritative active state and revocation precedence.  
**Status:** OPEN

### KAI-STATECTRL-032 — MEDIUM — Append races
**Issue:** multi-process JSONL append has no lock, fsync, integrity chain or event deduplication.  
**Risk:** lines interleave/disappear and acknowledged policies are not durable.  
**Recommendation:** use transactional append-only storage.  
**Status:** OPEN

### KAI-STATECTRL-033 — MEDIUM — Caller object is mutated
**Issue:** `store` writes generated `policy_id` back into the supplied dataclass.  
**Risk:** hidden side effects race callers and can be mistaken for proof that persistence succeeded.  
**Recommendation:** return a new immutable stored record/receipt.  
**Status:** OPEN

### KAI-STATECTRL-034 — MEDIUM — Policy values are unconstrained
**Issue:** confidence, utility, counts, versions and timestamps accept non-finite/negative/extreme values; source/domain are not checked against declared constants; text is unbounded.  
**Risk:** ranking, JSON and future decisions become invalid.  
**Recommendation:** enforce strict bounded schemas.  
**Status:** OPEN

### KAI-STATECTRL-035 — MEDIUM — Corruption is hidden
**Issue:** malformed lines are skipped individually; file-level errors return empty results.  
**Risk:** policies disappear from planning without integrity failure or quarantine.  
**Recommendation:** preserve and expose corrupt records, fail closed for required policy sources.  
**Status:** OPEN

### KAI-STATECTRL-036 — MEDIUM — Full reads are unbounded
**Issue:** `all_policies` and every relevance query scan/parse the entire file; no pagination/byte/record limit exists.  
**Risk:** storage growth blocks requests and exhausts memory/CPU.  
**Recommendation:** index bounded transactional storage.  
**Status:** OPEN

### KAI-STATECTRL-037 — MEDIUM — Query parameters are weak
**Issue:** negative `top_k` uses Python slicing semantics and non-finite/extreme confidence thresholds are not rejected.  
**Risk:** callers retrieve surprising/large sets or disable filtering.  
**Recommendation:** validate finite bounded parameters.  
**Status:** OPEN

### KAI-STATECTRL-038 — MEDIUM — Count is not valid-record count
**Issue:** `policy_count` counts all lines, including blank/malformed/duplicate records.  
**Risk:** progress/readiness overstates usable policies.  
**Recommendation:** count validated active records from an index.  
**Status:** OPEN

### KAI-STATECTRL-039 — MEDIUM — Cached count diverges
**Issue:** cache invalidates only after this process’s successful-looking store; external/multi-worker changes and failed writes are not reconciled.  
**Risk:** workers expose different policy counts.  
**Recommendation:** use shared authoritative metadata.  
**Status:** OPEN

### KAI-STATECTRL-040 — MEDIUM — Flag integration fails open
**Issue:** `can_distill` imports `feature_flags` via a fragile non-package path and skips the flag check on ImportError.  
**Risk:** a broken flag import removes one activation prerequisite.  
**Recommendation:** use the authoritative package and fail closed.  
**Status:** OPEN

---

## Batch totals

- Findings: **40**
- Critical: **0**
- High: **19**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,054**
- Critical: **87**
- High: **412**
- Medium: **552**
- Low: **3**

## Files materially reviewed in this batch

`agentic/teammates.py`, `agentic/system_fsm.py`, `agentic/policy_memory.py`, with integration confirmation against active agentic routes and the earlier causal-policy duplicate-authority finding.
