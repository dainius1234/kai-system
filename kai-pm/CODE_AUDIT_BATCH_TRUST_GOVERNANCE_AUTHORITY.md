# Kai Code Audit — Trust Governance Authority Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records findings not already logged as `KAI-TRUST-001` through `KAI-TRUST-008` in `CODE_AUDIT_REGISTER_CONTINUED.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TRUSTAUTH-001 | CRITICAL | Trust audit-write failure converts a denied capability check into an allowed action |
| KAI-TRUSTAUTH-002 | CRITICAL | `revoke()` can increase trust to any higher level |
| KAI-TRUSTAUTH-003 | HIGH | `grant()` and `revoke()` have no authenticated operator boundary |
| KAI-TRUSTAUTH-004 | HIGH | Live capability names are absent from the authoritative capability map |
| KAI-TRUSTAUTH-005 | HIGH | Unknown capability attempts bypass audit and action counters |
| KAI-TRUSTAUTH-006 | HIGH | Action conviction is recorded but never enforced by the trust gate |
| KAI-TRUSTAUTH-007 | HIGH | Trust status merges two independent and potentially contradictory trust authorities |
| KAI-TRUSTAUTH-008 | HIGH | Score-computation failure is represented as score 50 / Journeyman |
| KAI-TRUSTAUTH-009 | HIGH | Arbitrary action context is copied into the trust ledger without redaction or bounds |
| KAI-TRUSTAUTH-010 | MEDIUM | Valid JSON containing an invalid trust level triggers fail-open integration behaviour |
| KAI-TRUSTAUTH-011 | MEDIUM | Trust mutations occur before audit and durable-state writes complete |
| KAI-TRUSTAUTH-012 | MEDIUM | `refused_actions` does not count refused capability attempts |
| KAI-TRUSTAUTH-013 | MEDIUM | Chat-response evidence is not linked to the actual response content |
| KAI-TRUSTAUTH-014 | MEDIUM | Trust audit-tail reads the full file and accepts invalid limits |
| KAI-TRUSTAUTH-015 | MEDIUM | Trust integration writes to a different relative ledger path than the service default |
| KAI-TRUSTAUTH-016 | MEDIUM | Trust integration mutates global import search order and imports generic module names |
| KAI-TRUSTAUTH-017 | MEDIUM | Trust events lack operation IDs and idempotency guarantees |
| KAI-TRUSTAUTH-018 | MEDIUM | Trust singleton configuration is first-caller controlled and worker-local |
| KAI-TRUSTAUTH-019 | CRITICAL | Trust Ledger write and acknowledgement APIs implement no authentication |
| KAI-TRUSTAUTH-020 | CRITICAL | Public acknowledgement can inflate operator-approval trust factors |
| KAI-TRUSTAUTH-021 | CRITICAL | A fresh ledger starts at approximately 54.5 / Journeyman rather than Neophyte |
| KAI-TRUSTAUTH-022 | CRITICAL | Ledger replay skips invalid records and then reports the surviving subset intact |
| KAI-TRUSTAUTH-023 | CRITICAL | Ledger HMAC defaults to the repository-known `trust-dev-secret` |
| KAI-TRUSTAUTH-024 | HIGH | Callers can impersonate operator, Kai or system when creating trust events |
| KAI-TRUSTAUTH-025 | HIGH | Public alignment-audit events can manipulate value, empathy and reliability scores |
| KAI-TRUSTAUTH-026 | HIGH | Trust events and score factors are publicly readable |
| KAI-TRUSTAUTH-027 | HIGH | HMAC signatures omit capability, trust tier and acknowledgement fields |
| KAI-TRUSTAUTH-028 | HIGH | Operator acknowledgements are neither persisted nor represented as chained events |
| KAI-TRUSTAUTH-029 | HIGH | Append publishes in-memory state before durable file persistence |
| KAI-TRUSTAUTH-030 | HIGH | Concurrent appends can fork the hash chain or interleave JSONL writes |
| KAI-TRUSTAUTH-031 | HIGH | The documented PostgreSQL production backend is not implemented |
| KAI-TRUSTAUTH-032 | HIGH | Merkle manifests are described as signed but contain no signature |
| KAI-TRUSTAUTH-033 | MEDIUM | Reported Merkle root covers only the most recent 100 events by default |
| KAI-TRUSTAUTH-034 | MEDIUM | Event payloads, notes and query limits are unbounded and weakly validated |
| KAI-TRUSTAUTH-035 | MEDIUM | Malformed numeric event fields can break trust-score computation |
| KAI-TRUSTAUTH-036 | MEDIUM | Trust Ledger performs synchronous full-file work inside async endpoints |
| KAI-TRUSTAUTH-037 | MEDIUM | Every event creation scans up to one million ledger entries for Merkle scheduling |
| KAI-TRUSTAUTH-038 | MEDIUM | Merkle interval, paths, score windows and port configuration are not validated |

---

## Trust Core and integration: `agentic/trust_core.py`, `agentic/trust_integration.py`

### KAI-TRUSTAUTH-001 — CRITICAL — Audit storage failure bypasses a denial
**Issue:** `TrustCore.can_do()` calculates a denial, then appends an audit event and rewrites the record before returning. If either write raises, `gate_autonomous_action()` catches the exception under its fail-open handler while its local `allowed` value is still `True`.  
**Risk:** Making the trust/audit filesystem unwritable converts an action that the current trust level denied into an allowed autonomous action.  
**Recommendation:** make authorisation a pure fail-closed decision; persist the audit separately and return governance-unavailable if mandatory audit storage fails.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-002 — CRITICAL — Revocation can promote
**Issue:** `revoke(level, ...)` assigns the supplied level directly and does not require it to be lower than the current level.  
**Risk:** a caller can invoke a method named revoke with `GUARDIAN` and raise trust to the maximum level while producing a misleading revocation event.  
**Recommendation:** enforce a monotonic decrease and authenticated expected-current-version checks.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-003 — HIGH — Trust transitions are unauthorised method calls
**Issue:** `grant()` and `revoke()` accept any `TrustLevel` and any caller-provided `by` string. No authenticated operator identity, signature, approval token or capability check exists at the mutation boundary.  
**Risk:** any imported/internal code can set the full trust level and self-assert that the operator authorised it.  
**Recommendation:** move transitions behind a separately authenticated operator authority with signed requests and immutable event IDs.  
**Status:** OPEN

### KAI-TRUSTAUTH-004 — HIGH — Capability vocabulary is disconnected from live callers
**Issue:** the map defines names such as `financial_micro`, `model_management` and `proactive_care`, while callers use names such as `strategy_auto_trade`, `paper_trade`, `model_council_benchmark`, `model_council_recommend`, `web_scout_search` and `proactive_observation`.  
**Risk:** the documented level model is not actually applied to live actions; most real capabilities fall into the unknown-GUARDIAN fallback instead of their intended tier.  
**Recommendation:** use one versioned capability registry referenced by all callers and fail build/tests on unmapped names.  
**Status:** OPEN

### KAI-TRUSTAUTH-005 — HIGH — Unknown actions disappear from the audit trail
**Issue:** `can_do()` returns immediately for unknown capabilities before incrementing `total_actions`, appending an audit event or saving the record.  
**Risk:** attempts against the most suspicious/unmapped capability names leave no TrustCore evidence despite the module promise that every attempt is recorded.  
**Recommendation:** audit every attempted capability before returning, including unknown/denied requests.  
**Status:** OPEN

### KAI-TRUSTAUTH-006 — HIGH — Conviction is decorative at the trust boundary
**Issue:** `gate_autonomous_action(..., conviction)` never uses conviction to allow, deny or escalate. It is copied only into ledger event data.  
**Risk:** a zero-confidence action is treated identically to a high-confidence action once level/alignment checks pass, while callers may believe conviction participates in governance.  
**Recommendation:** define risk-specific minimum verified conviction or remove the misleading parameter; do not accept caller-self-reported values as sufficient evidence.  
**Status:** OPEN

### KAI-TRUSTAUTH-007 — HIGH — Two incompatible trust states are presented together
**Issue:** `get_trust_status()` combines the Trust Ledger’s continuous score/tier with TrustCore’s discrete level, but defines no mapping, precedence or inconsistency state.  
**Risk:** status can simultaneously report DORMANT and Journeyman, or a high level with a low score, leaving downstream consumers to choose whichever is more permissive.  
**Recommendation:** designate one authoritative gate and expose explicit reconciled/inconsistent states.  
**Status:** OPEN

### KAI-TRUSTAUTH-008 — HIGH — Scoring failure produces elevated trust
**Issue:** `_get_score()` returns `{"score": 50.0, "tier": "Journeyman"}` on any import or computation error.  
**Risk:** broken scoring infrastructure is represented as a mid-tier level permitting paper trading, sandboxed skill hunting and draft posting under the score documentation.  
**Recommendation:** return unavailable/Neophyte and fail closed.  
**Status:** OPEN

### KAI-TRUSTAUTH-009 — HIGH — Sensitive context is blindly audited
**Issue:** `gate_autonomous_action` expands the complete caller-supplied context into persistent ledger data. No schema, secret filter, field allowlist, nesting or size limit exists.  
**Risk:** prompts, credentials, financial values, paths and private user context can be permanently copied into broadly readable ledger events.  
**Recommendation:** log a minimal typed redacted decision record.  
**Status:** OPEN

### KAI-TRUSTAUTH-010 — MEDIUM — Syntactically valid state can break the enum gate
**Issue:** `_load_record()` accepts `level` without validating it. A value outside 0–6 loads successfully, then `TrustLevel(self._record.level)` raises during checks/status; integration catches the gate failure and fails open.  
**Risk:** ordinary corruption or tampering can bypass rather than merely disable governance.  
**Recommendation:** validate the complete record before activation and enter locked recovery on invalid values.  
**Status:** OPEN

### KAI-TRUSTAUTH-011 — MEDIUM — Partial in-memory mutations precede persistence
**Issue:** evidence scores, action counters and trust levels are mutated before audit append and record write. Exceptions leave process memory changed while disk/audit state remains old.  
**Risk:** callers, workers and restart state disagree about the current authority.  
**Recommendation:** use one transactional compare-and-swap commit.  
**Status:** OPEN

### KAI-TRUSTAUTH-012 — MEDIUM — Refusal metric is semantically false
**Issue:** denied `can_do()` attempts never increment `refused_actions`. The counter is incremented whenever positive `values` evidence is recorded, regardless of whether an action was refused.  
**Risk:** progress/status misrepresents value-aligned refusal performance and can be gamed by generic evidence events.  
**Recommendation:** derive counters from immutable typed outcomes.  
**Status:** OPEN

### KAI-TRUSTAUTH-013 — MEDIUM — Chat evidence has no response linkage
**Issue:** `record_chat_response` accepts `response_summary` but never stores or hashes it. Positive consistency evidence records only conviction and specialist.  
**Risk:** there is no way to verify which answer supposedly earned evidence or whether it was correct.  
**Recommendation:** bind evidence to an immutable response/outcome record and operator evaluation.  
**Status:** OPEN

### KAI-TRUSTAUTH-014 — MEDIUM — Audit-tail is unbounded and fragile
**Issue:** `audit_tail(n)` reads and splits the complete audit file, accepts negative/extreme `n`, and lets one malformed JSON line raise.  
**Risk:** growing logs consume memory and corruption can disable status/inspection paths.  
**Recommendation:** use bounded indexed storage and validate limits/records.  
**Status:** OPEN

### KAI-TRUSTAUTH-015 — MEDIUM — Integration and service use different ledger paths
**Issue:** integration uses relative `data/trust-ledger/events.jsonl`; the Trust Ledger service defaults to `/data/trust-ledger/events.jsonl`. Working directory and mounts determine whether they refer to the same file.  
**Risk:** the agent can write one trust history while the service scores and verifies another.  
**Recommendation:** use one absolute configured authority accessed through a defined API.  
**Status:** OPEN

### KAI-TRUSTAUTH-016 — MEDIUM — Global import-path mutation
**Issue:** `_get_ledger()` inserts the trust-ledger directory at the start of `sys.path` and imports generic module names `ledger` and `score`.  
**Risk:** global import resolution changes at runtime and a conflicting/preloaded module can become the trust implementation.  
**Recommendation:** package and import the authority by an unambiguous verified module identity.  
**Status:** OPEN

### KAI-TRUSTAUTH-017 — MEDIUM — Trust events are replayable/duplicable
**Issue:** integration supplies no operation ID, source event ID or idempotency key. Repeated retries/calls append indistinguishable autonomous actions and audits.  
**Risk:** counts, averages and trust evidence can be distorted by duplicates.  
**Recommendation:** require immutable unique decision IDs and deduplicate.  
**Status:** OPEN

### KAI-TRUSTAUTH-018 — MEDIUM — Trust singleton authority depends on call order
**Issue:** the first `get_trust_core(data_dir)` fixes the process’s storage directory; later callers cannot change/verify it. Each worker owns a different singleton.  
**Risk:** tests, deployments and workers can use different trust records without warning.  
**Recommendation:** initialise one validated external authority during startup.  
**Status:** OPEN

---

## Trust Ledger service: `trust-ledger/app.py`, `trust-ledger/ledger.py`, `trust-ledger/score.py`

### KAI-TRUSTAUTH-019 — CRITICAL — Trust mutation is public
**Issue:** the module header claims an HMAC-authenticated internal write API, but `/trust/event`, `/trust/alignment-audit` and `/trust/events/{event_id}/ack` perform no authentication or authorisation.  
**Risk:** any reachable caller can create trust evidence and impersonate operator endorsements.  
**Recommendation:** require mutual service identity for writes and separately authenticated operator identity for acknowledgements.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-020 — CRITICAL — Public endorsement changes trust scoring
**Issue:** unauthenticated `PATCH /trust/events/{event_id}/ack` sets `operator_ack=True`. `compute_score` treats acknowledged autonomous actions as endorsed/successful and uses them in approval and conviction factors.  
**Risk:** a caller can endorse attacker-created or arbitrary actions and raise the trust score.  
**Recommendation:** cryptographically bind acknowledgements to an authenticated operator and append them as immutable signed events.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-021 — CRITICAL — Empty history starts at Journeyman
**Issue:** with no events, neutral defaults sum to 15 + 10 + 12.5 + 5 + 9.5 + 2.5 = **54.5**, which maps to Journeyman. The code comment says a new system starts around 20/Neophyte.  
**Risk:** a brand-new or erased ledger receives a tier documented as permitting paper trading, sandboxed skill hunting and draft posting without earning trust.  
**Recommendation:** start at zero/Neophyte and require verified evidence for every positive factor.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-022 — CRITICAL — Integrity failure is erased during replay
**Issue:** `_replay()` skips corrupt or signature-mismatched lines rather than retaining a broken-chain marker. `verify_chain()` later checks only the surviving in-memory events, so `/health` can report `intact: true` after damaged/tampered records were omitted.  
**Risk:** deletion, corruption and chain breaks can be presented as a valid shorter ledger.  
**Recommendation:** fail startup/readiness on the first invalid raw record and preserve the exact damaged stream for investigation.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-023 — CRITICAL — Known default signing secret
**Issue:** absent configuration uses the source-known HMAC key `trust-dev-secret`.  
**Risk:** any caller with repository access can forge valid event signatures and reconstructed chains for default deployments.  
**Recommendation:** fail startup without a high-entropy secret held in protected key management and support key versions/rotation.  
**Status:** OPEN — immediate remediation required

### KAI-TRUSTAUTH-024 — HIGH — Event identity is self-asserted
**Issue:** `initiator` and `trust_tier` are caller-controlled strings; no identity binding exists.  
**Risk:** attackers can create events apparently issued by `operator`, `system` or high-trust Kai.  
**Recommendation:** derive actor/tier from authenticated server-side identity and current authority.  
**Status:** OPEN

### KAI-TRUSTAUTH-025 — HIGH — Public self-audits manipulate three score factors
**Issue:** `/trust/alignment-audit` accepts arbitrary `ohana_alignment`, `empathy_accuracy` and `uptime_pct` inside event data. These averages contribute 45% of total score.  
**Risk:** callers can submit maximum values and materially elevate trust.  
**Recommendation:** accept signed measurements from independent approved evaluators with bounded schemas.  
**Status:** OPEN

### KAI-TRUSTAUTH-026 — HIGH — Full trust history is public
**Issue:** events, event data, operator notes/acknowledgements, scores, factors, tier and integrity state are returned without access control.  
**Risk:** sensitive autonomous-action context, behavioural history and security posture are exposed.  
**Recommendation:** require scoped audit-reader access and redact payloads.  
**Status:** OPEN

### KAI-TRUSTAUTH-027 — HIGH — Significant fields are outside the signature
**Issue:** `_sign` covers event ID, timestamp, type, initiator and event data, but omits capability, trust tier, previous hash, operator acknowledgement and operator note.  
**Risk:** these governance-significant fields can be changed without invalidating the event signature; acknowledgement changes are entirely outside chain integrity.  
**Recommendation:** sign a canonical complete immutable event envelope.  
**Status:** OPEN

### KAI-TRUSTAUTH-028 — HIGH — Acknowledgements are volatile mutable state
**Issue:** `ack()` edits the in-memory event object only. It does not rewrite the file or append an acknowledgement event. Restart removes every endorsement and note.  
**Risk:** trust scores change across restart and operator decisions have no durable audit evidence.  
**Recommendation:** append a signed immutable acknowledgement referencing the event.  
**Status:** OPEN

### KAI-TRUSTAUTH-029 — HIGH — In-memory append precedes disk commit
**Issue:** `append()` adds the event to `_events` before `_persist`. If persistence fails, the live process chains later events from a record absent on disk.  
**Risk:** API responses/score include non-durable events and restart produces a different/broken history.  
**Recommendation:** atomically persist/fsync before publishing the new revision in memory.  
**Status:** OPEN

### KAI-TRUSTAUTH-030 — HIGH — Concurrent chain append is unsafe
**Issue:** no lock protects reading the last signature, appending memory or writing JSONL. Multiple requests can derive the same previous hash and append competing next events.  
**Risk:** the ledger forks/interleaves and later replay silently discards records.  
**Recommendation:** serialise append through a transactional single writer.  
**Status:** OPEN

### KAI-TRUSTAUTH-031 — HIGH — Production backend claim is false
**Issue:** the ledger documentation says `TRUST_LEDGER_DB_URL` selects PostgreSQL, but only `FileLedger` exists and `app.py` always constructs it.  
**Risk:** operators can believe production uses transactional shared storage while it remains a local file.  
**Recommendation:** remove the claim or implement and verify the production backend.  
**Status:** OPEN

### KAI-TRUSTAUTH-032 — HIGH — “Signed” Merkle publication is unsigned
**Issue:** the published manifest contains only root, event count and timestamp and is written to an ordinary mutable JSON array. No signature, key ID or external anchoring exists.  
**Risk:** the checkpoint can be rewritten together with the ledger and provides no independent integrity proof.  
**Recommendation:** sign and externally anchor canonical append-only checkpoints.  
**Status:** OPEN

### KAI-TRUSTAUTH-033 — MEDIUM — Merkle root is a trailing-window root
**Issue:** `merkle_root()` defaults to the last 100 event signatures, while health/publishing present it as the ledger root.  
**Risk:** older events are outside the displayed checkpoint and callers can misinterpret coverage.  
**Recommendation:** publish explicit range/start/end IDs or a full append-only tree root.  
**Status:** OPEN

### KAI-TRUSTAUTH-034 — MEDIUM — Inputs and reads are unbounded
**Issue:** event type-adjacent strings, initiator, capability, tier, nested event data and acknowledgement notes have no size/depth limits. Event listing limits and time windows accept negative/extreme values.  
**Risk:** callers consume memory/disk and inject oversized sensitive content.  
**Recommendation:** use strict bounded schemas and positive capped query parameters.  
**Status:** OPEN

### KAI-TRUSTAUTH-035 — MEDIUM — Numeric event poisoning breaks scoring
**Issue:** `avg_field` sums arbitrary stored values without numeric/finite validation; score factors likewise accept out-of-range values.  
**Risk:** unauthenticated events containing strings, NaN or extreme values can crash scoring or create invalid trust outputs.  
**Recommendation:** validate at ingestion and recompute only from accepted typed events.  
**Status:** OPEN

### KAI-TRUSTAUTH-036 — MEDIUM — Async API performs synchronous ledger work
**Issue:** endpoints execute file replay-derived scans, append, full chain verification, Merkle calculation and file writes synchronously on the event-loop thread.  
**Risk:** large ledgers block all API traffic.  
**Recommendation:** use a transactional service/worker and bounded indexed queries.  
**Status:** OPEN

### KAI-TRUSTAUTH-037 — MEDIUM — Every write scans the ledger
**Issue:** after each event, `record_event` calls `ledger.events(limit=1_000_000)` solely to determine whether the count is divisible by the Merkle interval.  
**Risk:** write cost grows linearly and large event lists are repeatedly allocated.  
**Recommendation:** maintain an atomic durable event counter.  
**Status:** OPEN

### KAI-TRUSTAUTH-038 — MEDIUM — Configuration lacks validation
**Issue:** Merkle interval, publish path, ledger path, score `since_days`, query limits and port are accepted directly. A zero Merkle interval causes division by zero on event creation.  
**Risk:** misconfiguration disables trust recording or writes checkpoints to unintended locations.  
**Recommendation:** validate typed ranges and approved paths at startup.  
**Status:** OPEN

---

## Batch totals

- Findings: **38**
- Critical: **8**
- High: **18**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **1,038**
- Critical: **95**
- High: **409**
- Medium: **531**
- Low: **3**

## Files materially reviewed in this batch

`agentic/trust_core.py`, `agentic/trust_integration.py`, `trust-ledger/app.py`, `trust-ledger/ledger.py`, `trust-ledger/score.py`, with integration confirmation against live `gate_autonomous_action()` callers. Existing `KAI-TRUST-001` through `KAI-TRUST-008` were not duplicated.
