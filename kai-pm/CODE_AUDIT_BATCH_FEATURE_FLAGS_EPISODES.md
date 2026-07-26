# Kai Code Audit — Feature Flags and Episode Persistence Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CFG-001 | HIGH | Feature-flag implementation contradicts its documented safe-by-default policy |
| KAI-CFG-002 | HIGH | Numerous autonomous and identity-affecting capabilities are enabled by default |
| KAI-CFG-003 | MEDIUM | Invalid feature-flag values silently disable capabilities |
| KAI-CFG-004 | MEDIUM | Runtime feature registration is process-local and concurrency-unsafe |
| KAI-EPISODE-001 | CRITICAL | Redis decay deletes every episode beyond the first 1,000 records |
| KAI-EPISODE-002 | HIGH | Redis decay is non-transactional and loses concurrent writes |
| KAI-EPISODE-003 | HIGH | Redis failure silently creates process-local split-brain episode storage |
| KAI-EPISODE-004 | HIGH | Fallback episode spool defaults to ephemeral `/tmp` storage |
| KAI-EPISODE-005 | MEDIUM | Corrupt spool records are silently discarded |
| KAI-EPISODE-006 | MEDIUM | Spool rotation rewrites files non-atomically and without locking |
| KAI-EPISODE-007 | MEDIUM | Episode recall silently inspects only the newest 201 Redis entries |
| KAI-EPISODE-008 | MEDIUM | Spool checksum provides no authenticity against deliberate modification |

---

## Feature flags: `common/feature_flags.py`

### KAI-CFG-001 — HIGH — Implementation contradicts the documented default-off policy
**Issue:** The module documentation states that every flag defaults to OFF and new capabilities must be opted into. The registry sets many entries to `True`.  
**Risk:** Operators and reviewers can rely on a false safety assumption, while new deployments automatically activate broad autonomous behaviour.  
**Recommendation:** Make the documented policy true in code, or explicitly document and review every default-enabled capability.  
**Status:** OPEN

### KAI-CFG-002 — HIGH — High-impact capabilities are enabled by default
**Issue:** Defaults enable dream evolution, automatic checkpointing, tree search, proactive agents, narrative identity, conscience filtering, memory consolidation, security self-audit, financial context, sensory learning, skill acquisition, FSM, house doctor, ritual discovery, swarm, vault sync, Socratic decomposition and autonomous hypothesis testing.  
**Risk:** Fresh or partially configured deployments activate governance-, identity-, memory- and autonomy-affecting functions without an explicit operational acceptance step.  
**Recommendation:** Default consequential features off and enable them through signed environment profiles after dependency and security readiness checks.  
**Status:** OPEN

### KAI-CFG-003 — MEDIUM — Invalid values silently become false
**Issue:** Any environment value outside `1`, `true`, `yes` or `on` is interpreted as disabled, including misspellings and malformed values.  
**Risk:** Configuration errors silently alter system behaviour rather than failing validation.  
**Recommendation:** Accept an explicit boolean grammar and reject unknown values at startup.  
**Status:** OPEN

### KAI-CFG-004 — MEDIUM — Runtime registration is not shared or synchronised
**Issue:** `register_flag` mutates the module-level registry dictionary without locking or durable/shared propagation.  
**Risk:** Workers can expose different flag registries and defaults, producing inconsistent behaviour and diagnostics.  
**Recommendation:** Use immutable startup configuration or a versioned shared configuration authority.  
**Status:** OPEN

---

## Episode persistence: `agentic/kai_config.py`

### KAI-EPISODE-001 — CRITICAL — Decay deletes records beyond the first 1,000
**Issue:** `RedisSaver.decay` reads only `LRANGE key 0 1000`, classifies those records, then deletes the entire source key and rebuilds it only from the retained records in that limited range.  
**Risk:** Once a user has more than 1,001 episodes, every older record outside the fetched window is permanently deleted during decay, regardless of age, score or retention policy.  
**Recommendation:** Process the complete collection using cursor/range batches and an atomic migration protocol, or use a sorted-set/database retention operation.  
**Status:** OPEN — immediate remediation required

### KAI-EPISODE-002 — HIGH — Decay loses concurrent writes and can duplicate archives
**Issue:** Episodes are pushed to the archive individually before a later pipeline deletes and rebuilds the active list. The read, archive writes and replacement are not one transaction.  
**Risk:** Episodes added during decay can be deleted; retries can duplicate archive entries; readers can observe mixed states.  
**Recommendation:** Use an atomic server-side script or transactional database operation with immutable episode IDs and idempotency.  
**Status:** OPEN

### KAI-EPISODE-003 — HIGH — Redis outage creates split-brain storage
**Issue:** `build_saver` catches any Redis construction or ping failure and silently returns a local `ChecksummedSpoolSaver`. Each process can therefore maintain an independent episode history. There is no replay or reconciliation into Redis when it recovers.  
**Risk:** Learning, failure classification and self-improvement decisions differ by process, while accepted episodes disappear on restart or remain outside the authoritative store.  
**Recommendation:** Fail readiness or use a durable shared outbox with automatic reconciliation and an explicit degraded state.  
**Status:** OPEN

### KAI-EPISODE-004 — HIGH — Fallback spool is ephemeral by default
**Issue:** `EPISODE_SPOOL_PATH` defaults to `/tmp/langgraph_episode_spool.log`.  
**Risk:** The fallback history can be removed by restart, container replacement or temporary-directory cleanup.  
**Recommendation:** Require an explicitly mounted durable path and secure permissions whenever fallback persistence is allowed.  
**Status:** OPEN

### KAI-EPISODE-005 — MEDIUM — Corrupt spool records silently disappear
**Issue:** Checksum mismatch, malformed JSON and schema errors are skipped without quarantine, metrics or failure state.  
**Risk:** Episode evidence can vanish from learning history while the system continues as though the spool loaded correctly.  
**Recommendation:** Preserve malformed records, report integrity failure and require reconciliation.  
**Status:** OPEN

### KAI-EPISODE-006 — MEDIUM — Spool rotation is non-atomic and concurrency-unsafe
**Issue:** Rotation reads the entire spool, writes an archive directly and rewrites the active spool directly, without locks, temporary replacement or fsync of the archive and directory.  
**Risk:** Concurrent appends can be lost and interruption can corrupt both active and archived history. Timestamp-only archive names can also collide.  
**Recommendation:** Use locked atomic generations or a transactional append store.  
**Status:** OPEN

### KAI-EPISODE-007 — MEDIUM — Recall silently truncates history
**Issue:** Redis recall reads only indexes `0..200` before applying the requested day filter.  
**Risk:** A busy period with more than 201 recent episodes causes valid in-window history to be omitted, distorting learning and competence analysis.  
**Recommendation:** Page until the time boundary is reached or query a time-indexed sorted set.  
**Status:** OPEN

### KAI-EPISODE-008 — MEDIUM — Checksums do not authenticate spool data
**Issue:** Each line contains an ordinary SHA-256 checksum alongside the payload. Anyone able to modify the spool can recalculate it.  
**Risk:** Deliberate episode tampering is accepted as intact and can poison learning or self-improvement decisions.  
**Recommendation:** Use an HMAC/signature with protected keys and an append-chain or independently anchored log.  
**Status:** OPEN

---

## Batch totals

- Findings: **12**
- Critical: **1**
- High: **5**
- Medium: **6**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **224**
- Critical: **28**
- High: **101**
- Medium: **94**
- Low: **1**

## Files materially reviewed in this batch

`common/feature_flags.py`, `agentic/kai_config.py` feature, Redis saver and spool paths.
