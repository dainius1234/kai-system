# Kai Code Audit — Trust Ledger Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_TRUST_GOVERNANCE_AUTHORITY.md` or the earlier Trust Core register entries.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TRUSTLX-001 | HIGH | Failed or disabled Merkle publication is still recorded as a successful `MERKLE_PUBLISH` trust event |
| KAI-TRUSTLX-002 | HIGH | A published Merkle manifest is stale immediately because publication appends a new ledger event after computing the root |
| KAI-TRUSTLX-003 | HIGH | Merkle publication tasks are fire-and-forget and can race each other and ordinary ledger appends |
| KAI-TRUSTLX-004 | HIGH | Merkle manifest persistence is an unlocked non-atomic full-file read-modify-write |
| KAI-TRUSTLX-005 | HIGH | Corrupt Merkle manifest JSON is silently treated as empty and overwritten, erasing prior checkpoints |
| KAI-TRUSTLX-006 | HIGH | Merkle manifest history grows without bounds and is completely read and rewritten for every checkpoint |
| KAI-TRUSTLX-007 | HIGH | The event-signing key is loaded once at import and events carry no key ID or rotation version |
| KAI-TRUSTLX-008 | HIGH | The complete unbounded ledger is retained in process memory for the lifetime of the service |
| KAI-TRUSTLX-009 | HIGH | Ledger storage has no retention, rotation, archival or legal-hold lifecycle |
| KAI-TRUSTLX-010 | HIGH | Ledger and Merkle files are trusted without ownership, permission, symlink or regular-file validation |
| KAI-TRUSTLX-011 | HIGH | Concurrent first access can construct multiple independent `FileLedger` instances for the same path |
| KAI-TRUSTLX-012 | HIGH | Multiple workers do not observe one another’s appended events and therefore compute different scores and roots |
| KAI-TRUSTLX-013 | HIGH | Public health performs full-chain verification and Merkle recomputation on every probe |
| KAI-TRUSTLX-014 | HIGH | No rate limit, caller quota or workload-admission policy protects append, scoring or integrity scans |
| KAI-TRUSTLX-015 | HIGH | Explicit zero value alignment is replaced with a favourable neutral factor score |
| KAI-TRUSTLX-016 | HIGH | Explicit zero empathy accuracy is replaced with a favourable neutral factor score |
| KAI-TRUSTLX-017 | HIGH | Explicit zero uptime is replaced with 95% reliability |
| KAI-TRUSTLX-018 | HIGH | Explicit zero conviction receives a positive default conviction factor |
| KAI-TRUSTLX-019 | HIGH | Operator acknowledgement is treated as execution success without any linked outcome evidence |
| KAI-TRUSTLX-020 | HIGH | Override penalties are global counts and are not linked to the autonomous actions they corrected |
| KAI-TRUSTLX-021 | MEDIUM | Boolean event-data values are accepted as numeric score measurements |
| KAI-TRUSTLX-022 | MEDIUM | Score factors are not individually bounded before aggregation and total clamping hides invalid evidence |
| KAI-TRUSTLX-023 | MEDIUM | Score responses contain no ledger generation, final event ID or Merkle revision |
| KAI-TRUSTLX-024 | MEDIUM | `/trust/score/tier` serialises the score as a string while `/trust/score` returns a number |
| KAI-TRUSTLX-025 | MEDIUM | Zero and negative `since_days` values have surprising cutoff semantics |
| KAI-TRUSTLX-026 | MEDIUM | Merkle manifests omit the previous checkpoint, covered event range, signing-key version and policy revision |
| KAI-TRUSTLX-027 | MEDIUM | Merkle publication records use pre-publication event counts and cannot describe the current ledger state |
| KAI-TRUSTLX-028 | MEDIUM | File appends do not explicitly flush, fsync or record a durable sequence number |
| KAI-TRUSTLX-029 | MEDIUM | Event timestamps use wall-clock floats without source-event time or a monotonic sequence |
| KAI-TRUSTLX-030 | MEDIUM | Score computation repeatedly performs multiple independent scans rather than one consistent snapshot |
| KAI-TRUSTLX-031 | MEDIUM | Public integrity and score reads have no immutable access audit |
| KAI-TRUSTLX-032 | MEDIUM | Deprecated startup hooks provide no graceful ledger flush, task drain or multi-worker safety contract |

---

## High-severity findings

### KAI-TRUSTLX-001 — HIGH — False publication evidence
**Issue:** `_publish_merkle()` appends a `MERKLE_PUBLISH` event after the file-write attempt. It does so when the publish path is unset and also after exceptions writing the manifest.  
**Risk:** The trust history states a checkpoint was published even when it was only logged locally or publication failed.  
**Recommendation:** append a success event only after a durable externally verifiable checkpoint commit; record failures as separate typed events.  
**Status:** OPEN

### KAI-TRUSTLX-002 — HIGH — Checkpoint is stale at creation
The root and event count are computed, the manifest is written, and then a new `MERKLE_PUBLISH` event is appended. The published root therefore never represents the ledger state immediately after publication.

### KAI-TRUSTLX-003 — HIGH — Publication-task races
Every interval boundary creates an untracked task. Concurrent appends can schedule overlapping publications against changing ledger state and the same manifest file.

### KAI-TRUSTLX-004 — HIGH — Unsafe manifest commit
Existing JSON is read, modified and overwritten without a lock, temporary file, compare-and-swap or fsync.

### KAI-TRUSTLX-005 — HIGH — Corruption becomes checkpoint loss
Any manifest parse error resets `existing=[]`; the next write permanently discards the damaged and all prior checkpoint history.

### KAI-TRUSTLX-006 — HIGH — Unbounded manifest rewrite
The complete ever-growing array is materialised and pretty-printed for each checkpoint.

### KAI-TRUSTLX-007 — HIGH — No signing-key lifecycle
`_HMAC_KEY` is fixed during module import. Events identify neither key version nor rotation epoch, so old/new-key verification and controlled rollover are impossible.

### KAI-TRUSTLX-008 — HIGH — Full ledger retained indefinitely
`FileLedger._events` holds every accepted event object; startup and runtime memory grow with the complete history.

### KAI-TRUSTLX-009 — HIGH — No evidence lifecycle
The file-backed authority implements no rotation, immutable archive generations, retention policy or protected incident/legal-hold state.

### KAI-TRUSTLX-010 — HIGH — Untrusted local file authority
The ledger and publication paths may point through symlinks or files with unsafe ownership/mode; these properties are never checked before replay or writes.

### KAI-TRUSTLX-011 — HIGH — Lazy singleton race
`get_ledger()` has no lock. Concurrent first calls can each replay and construct an instance before `_ledger` is assigned, then independently append to the same file.

### KAI-TRUSTLX-012 — HIGH — Worker-local trust universes
Each worker replays once and never tails/reloads external writes. One worker’s events and acknowledgements are absent from another worker’s score, root and event list.

### KAI-TRUSTLX-013 — HIGH — Public O(n) health workload
Every health probe verifies every in-memory event and calculates a Merkle tree. Docker, Supervisor and anonymous callers can repeatedly trigger this growing synchronous work.

### KAI-TRUSTLX-014 — HIGH — No workload admission
Event creation, event listing, full scoring, chain verification and Merkle calculation have no rate, concurrency or principal limits.

### KAI-TRUSTLX-015 — HIGH — Zero alignment is rewarded
`alignment_score = avg_alignment * 25 if avg_alignment else 12.5`; a measured value of `0.0` becomes 12.5 points rather than zero.

### KAI-TRUSTLX-016 — HIGH — Zero empathy is rewarded
`avg_empathy=0.0` becomes the default five points.

### KAI-TRUSTLX-017 — HIGH — Zero uptime becomes 95%
`reliability_score = (avg_uptime or 0.95) * 10`; an explicit uptime measurement of zero yields 9.5 points.

### KAI-TRUSTLX-018 — HIGH — Zero conviction receives positive credit
When average conviction is zero, the factor is set to 10.0 rather than reflecting the observed absence of conviction.

### KAI-TRUSTLX-019 — HIGH — Endorsement is mislabeled success
The conviction factor’s `successful` count is the number of acknowledged autonomous actions. No execution outcome, verifier result, harm state or actual success record is required.

### KAI-TRUSTLX-020 — HIGH — Unrelated overrides penalise all actions
Overrides are counted globally and divided by autonomous-action count; no override references the action it corrected or confirms it occurred within the same decision set.

---

## Medium-severity findings

### KAI-TRUSTLX-021 — MEDIUM — Booleans are score numbers
Python booleans participate in `sum()` and division; `True`/`False` measurements can silently become 1/0 alignment, empathy, uptime or conviction values.

### KAI-TRUSTLX-022 — MEDIUM — Invalid factors hidden by final clamp
Out-of-range negative or greater-than-one measurements can create negative/oversized individual factors; only the final total is clamped, obscuring invalid evidence in factor outputs.

### KAI-TRUSTLX-023 — MEDIUM — Score lacks evidence revision
The response does not state the final event ID, count, root or exact snapshot used.

### KAI-TRUSTLX-024 — MEDIUM — Score type inconsistency
The quick tier endpoint converts the numeric score to text, complicating strict consumers and comparisons.

### KAI-TRUSTLX-025 — MEDIUM — Time-window edge cases
`since_days=0` disables the cutoff entirely because zero is false; negative values create a future cutoff and generally return no evidence.

### KAI-TRUSTLX-026 — MEDIUM — Incomplete checkpoint envelope
The manifest cannot prove continuity, key identity, covered event IDs/range or policy/scoring revision.

### KAI-TRUSTLX-027 — MEDIUM — Event-count semantic mismatch
The manifest count refers to the ledger before its own publication event and cannot be used as the current ledger length.

### KAI-TRUSTLX-028 — MEDIUM — Weak append durability
JSONL writes rely on context-manager close only and contain no explicit fsync, sequence or durable-commit acknowledgement.

### KAI-TRUSTLX-029 — MEDIUM — Weak chronology
Events use `time.time()` only; source occurrence time, trusted clock and monotonic ordering are absent.

### KAI-TRUSTLX-030 — MEDIUM — Inconsistent score snapshot
`compute_score()` calls `count`, `avg_field` and `events` repeatedly. Concurrent mutation can make factors refer to different ledger moments.

### KAI-TRUSTLX-031 — MEDIUM — No read-access evidence
Sensitive score, event and integrity reads are not recorded with actor, purpose or returned revision.

### KAI-TRUSTLX-032 — MEDIUM — Missing lifecycle ownership
The service uses a deprecated startup hook and has no shutdown handling for pending Merkle tasks, ledger flush verification or worker reconciliation.

---

## Batch totals

- Findings: **32**
- Critical: **0**
- High: **20**
- Medium: **12**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,436**
- Critical: **189**
- High: **1,220**
- Medium: **1,024**
- Low: **3**

## Files materially reviewed

`trust-ledger/app.py`, `trust-ledger/ledger.py`, `trust-ledger/score.py` and the existing Trust Governance Authority audit.
