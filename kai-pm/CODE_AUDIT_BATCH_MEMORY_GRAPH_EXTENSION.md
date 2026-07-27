# Kai Code Audit — Memory Graph Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_MEMORY_GRAPH.md`. The existing nine findings are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-GRAPHX-001 | CRITICAL | Re-ingesting an existing source ID overwrites its deletion mapping and permanently orphans the earlier graph data |
| KAI-GRAPHX-002 | CRITICAL | `cognee.add()` can commit data before `cognify()` fails, leaving partial graph state without deletion lineage |
| KAI-GRAPHX-003 | HIGH | Ingest has no idempotency key, content digest or duplicate-detection policy |
| KAI-GRAPHX-004 | HIGH | Only the first object returned by `cognee.add()` is indexed for deletion |
| KAI-GRAPHX-005 | HIGH | Caller-provided metadata is accepted but silently discarded |
| KAI-GRAPHX-006 | HIGH | Caller-controlled category and source ID are inserted directly into Cognee node sets and can cross-link unrelated memory domains |
| KAI-GRAPHX-007 | HIGH | Every ingest runs `cognify()` over the complete shared dataset |
| KAI-GRAPHX-008 | HIGH | Concurrent ingests can run overlapping whole-dataset cognification without a coordinator or generation lock |
| KAI-GRAPHX-009 | HIGH | Queries can run while cognification is mutating the same dataset and observe inconsistent generations |
| KAI-GRAPHX-010 | HIGH | Ingest and forget operations for the same source can race and produce orphaned or unexpectedly deleted state |
| KAI-GRAPHX-011 | HIGH | Cognee add, cognify, search and delete calls have no application-enforced deadline |
| KAI-GRAPHX-012 | HIGH | Client cancellation does not provide a transaction/cancellation contract for in-progress graph mutations |
| KAI-GRAPHX-013 | HIGH | Missing dataset IDs are replaced with the all-zero UUID during deletion |
| KAI-GRAPHX-014 | HIGH | Successful deletion is not verified against the graph backend before returning `forgotten` |
| KAI-GRAPHX-015 | HIGH | Backend deletion failures are returned with HTTP 200 and an error-shaped body |
| KAI-GRAPHX-016 | HIGH | Any Cognee `SearchType` enum exposed by the installed library can be selected by the caller |
| KAI-GRAPHX-017 | HIGH | Search results are returned without aggregate byte, depth or item-shape limits |
| KAI-GRAPHX-018 | HIGH | No rate limit, ingest/search concurrency bound or dataset-work budget exists |
| KAI-GRAPHX-019 | HIGH | Full query text and source identifiers are written to logs on failure |
| KAI-GRAPHX-020 | HIGH | All callers and memory categories share one global dataset namespace |
| KAI-GRAPHX-021 | MEDIUM | `not_found` deletion is returned as HTTP 200 rather than a typed absence result |
| KAI-GRAPHX-022 | MEDIUM | Dataset name and default `top_k` configuration are not validated at startup |
| KAI-GRAPHX-023 | MEDIUM | Query type matching is case-sensitive and leaks the caller-supplied enum name in errors |
| KAI-GRAPHX-024 | MEDIUM | Cognee is imported and initialised lazily inside request paths rather than validated during startup |
| KAI-GRAPHX-025 | MEDIUM | Search-result serialisation occurs outside the backend exception boundary |
| KAI-GRAPHX-026 | MEDIUM | The service has no ErrorBudget, request metrics or graph-operation telemetry |
| KAI-GRAPHX-027 | MEDIUM | No tamper-evident audit records ingest/query/forget actor, input digest, backend generation and outcome |
| KAI-GRAPHX-028 | MEDIUM | Health publicly exposes the dataset name and process-local indexed-source count |
| KAI-GRAPHX-029 | MEDIUM | No reconciliation detects graph records missing from `_source_index` or stale mappings pointing to absent data |
| KAI-GRAPHX-030 | MEDIUM | Ingested graph data has no stored source-content hash, version or supersession relation |
| KAI-GRAPHX-031 | MEDIUM | Docker build treats failure to install the required Ladybug/Kuzu JSON extension as successful |
| KAI-GRAPHX-032 | MEDIUM | A failed build-time extension install can defer an unpinned network download to runtime |
| KAI-GRAPHX-033 | MEDIUM | Several runtime dependencies use lower-bound version ranges rather than immutable hashes |
| KAI-GRAPHX-034 | MEDIUM | Lifespan performs no backend close, flush, mutation drain or graph-integrity checkpoint |
| KAI-GRAPHX-035 | MEDIUM | The current deployment inventory is inconsistent: source and historical port references exist while the latest full Compose file omits the service |

---

## Critical findings

### KAI-GRAPHX-001 — CRITICAL — Same-source re-ingest orphans old graph data
**Issue:** Every successful ingest assigns `_source_index[req.source_id]` to the newest returned `data_id`. It does not delete, supersede or retain the previous mapping.  
**Risk:** Reusing a source ID makes the previous graph record unreachable through `/graph/forget`, undermining correction, deletion and privacy erasure.  
**Recommendation:** make source ID/version a transactional unique key and atomically supersede or delete the previous generation.  
**Status:** OPEN — immediate remediation required

### KAI-GRAPHX-002 — CRITICAL — Add/cognify partial commit
**Issue:** `cognee.add()` is awaited before `cognee.cognify()`. If cognify fails, the endpoint returns 502 before the returned data ID is stored in `_source_index`. There is no rollback of the completed add.  
**Risk:** Data may persist in the graph system while the wrapper reports failure and loses the only deletion lineage it knows how to use. Retries can create duplicates.  
**Recommendation:** use a durable staged operation record, compensate failed cognification and acknowledge only an atomic verified generation.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-GRAPHX-003 — HIGH — No idempotent ingest identity
Repeated identical requests create repeated graph data; no body digest, operation ID or existing-content check is used.

### KAI-GRAPHX-004 — HIGH — Additional add results are orphaned
When `cognee.add()` returns multiple objects, only element zero contributes a data ID. Every additional object is unmanaged by the deletion index.

### KAI-GRAPHX-005 — HIGH — Metadata contract is false
`IngestRequest.metadata` accepts arbitrary provenance/context but is never passed to Cognee, stored locally or returned as ignored.

### KAI-GRAPHX-006 — HIGH — Untrusted node-set membership
Source ID and category are directly added to `node_set`, allowing caller strings to merge unrelated records into common graph groupings.

### KAI-GRAPHX-007 — HIGH — Whole-dataset work per write
Each individual ingest invokes `cognify(datasets=[DATASET_NAME])`, not a bounded per-record transformation.

### KAI-GRAPHX-008 — HIGH — Overlapping cognification
There is no single-flight lock, queue or generation coordinator around whole-dataset mutation.

### KAI-GRAPHX-009 — HIGH — Query/mutation generation race
Search can run while add/cognify updates the same dataset, yet responses expose no snapshot/generation identity.

### KAI-GRAPHX-010 — HIGH — Ingest/forget race
A forget can pop/delete one mapping while a concurrent ingest replaces it, leading to stale deletion, lost mapping or unexpected survival/removal.

### KAI-GRAPHX-011 — HIGH — Unbounded backend operations
No `asyncio.timeout`, cancellation deadline or backend-specific request budget wraps add, cognify, search or delete.

### KAI-GRAPHX-012 — HIGH — Cancellation is not transactional
A client disconnect/task cancellation may leave Cognee work continuing or partially committed without a durable operation state.

### KAI-GRAPHX-013 — HIGH — Zero dataset UUID fallback
When add results omit `dataset_id`, forget passes UUID zero rather than resolving the authoritative dataset or rejecting incomplete lineage.

### KAI-GRAPHX-014 — HIGH — Unverified deletion completion
The wrapper trusts the absence of an exception and does not confirm that the target data was removed from query results/backend storage.

### KAI-GRAPHX-015 — HIGH — Delete failure appears as HTTP success
Forget catches backend exceptions and returns `{status:"error"}` with HTTP 200, so clients and shared resilience layers may classify it as successful.

### KAI-GRAPHX-016 — HIGH — Library-wide query capability exposure
Caller text indexes `SearchType[query_type]`; every enum available in the installed Cognee version becomes remotely selectable rather than a server-approved subset.

### KAI-GRAPHX-017 — HIGH — Unbounded result materialisation
`results` is returned directly with no response byte/depth, per-item field or serialisable-type validation.

### KAI-GRAPHX-018 — HIGH — Missing workload admission
There is no per-caller quota, global semaphore, ingest/search queue, dataset-size budget or expensive-query policy.

### KAI-GRAPHX-019 — HIGH — Sensitive diagnostic logging
Failures log complete source IDs and `q=%r`, which may contain private memory text, credentials or personal queries.

### KAI-GRAPHX-020 — HIGH — Global dataset identity collapse
All callers, users, source types and categories are written into one environment-selected dataset without authenticated partition or purpose scope.

---

## Medium-severity findings

### KAI-GRAPHX-021 — MEDIUM — Absence uses success status
A missing source returns 200 `not_found`, making it difficult to distinguish an idempotent absence from a completed deletion using HTTP semantics alone.

### KAI-GRAPHX-022 — MEDIUM — Weak startup configuration
Dataset name and top-K accept empty, negative or extreme values without a typed startup report.

### KAI-GRAPHX-023 — MEDIUM — Query-type contract is brittle
Enum lookup is case-sensitive and error detail repeats the untrusted supplied value.

### KAI-GRAPHX-024 — MEDIUM — Lazy dependency initialisation
Cognee import/initialisation occurs on the first data request, shifting heavy failures/side effects from readiness into user traffic.

### KAI-GRAPHX-025 — MEDIUM — Serialisation outside try block
A backend may return objects FastAPI cannot encode; that failure occurs after the Cognee call and outside the explicit 502 handling.

### KAI-GRAPHX-026 — MEDIUM — No operation metrics
The service reports no latency, add/cognify/delete failures, orphan counts, generation age or query workload.

### KAI-GRAPHX-027 — MEDIUM — Missing audit evidence
No durable record links actor, request/content digest, source version, returned backend IDs, graph generation and delete verification.

### KAI-GRAPHX-028 — MEDIUM — Public topology disclosure
Health exposes dataset and local mapping count without authentication.

### KAI-GRAPHX-029 — MEDIUM — No reconciliation
The wrapper never scans for backend records without source mappings, mappings with invalid UUIDs or deleted backend records.

### KAI-GRAPHX-030 — MEDIUM — Missing source version
Graph data stores no wrapper-level source content digest, revision, superseded ID or event timestamp.

### KAI-GRAPHX-031 — MEDIUM — Required extension failure is build-success
The Dockerfile’s JSON-extension warm-up ends with `|| echo`, so the image build succeeds even when a documented query-time prerequisite is absent.

### KAI-GRAPHX-032 — MEDIUM — Runtime network dependency
The Dockerfile states that a failed warm-up may install the extension on first run, making production readiness depend on runtime network access and remote artefact availability.

### KAI-GRAPHX-033 — MEDIUM — Non-reproducible dependency ranges
FastAPI, Starlette, Pydantic and Transformers use `>=` without lockfile hashes, allowing materially different future dependency graphs.

### KAI-GRAPHX-034 — MEDIUM — Incomplete lifecycle
Lifespan logs startup only; it does not initialise/validate Cognee or drain/flush/close backend resources at shutdown.

### KAI-GRAPHX-035 — MEDIUM — Deployment inventory drift
The source, Dockerfile and historical `8061` references exist, but the latest fetched `docker-compose.full.yml` ends without a memu-graph service definition. Operational documentation and deployment state can therefore disagree about whether graph memory is active.

---

## Batch totals

- Findings: **35**
- Critical: **2**
- High: **18**
- Medium: **15**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,222**
- Critical: **191**
- High: **1,107**
- Medium: **921**
- Low: **3**

## Files materially reviewed

`memu-graph/app.py`, `memu-graph/Dockerfile`, `memu-graph/requirements.txt`, the existing Memory Graph audit, historical/current Compose definitions and memU integration references.
