# Kai Code Audit — Memory Graph Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-GRAPH-001 | CRITICAL | Graph ingest and forget endpoints are unauthenticated |
| KAI-GRAPH-002 | HIGH | Failed graph deletion permanently discards the retry mapping |
| KAI-GRAPH-003 | HIGH | Source-to-graph deletion index is process-local and lost on restart |
| KAI-GRAPH-004 | HIGH | Backend access control is explicitly disabled |
| KAI-GRAPH-005 | HIGH | Graph query exposes derived memory without access control |
| KAI-GRAPH-006 | MEDIUM | Graph request inputs and result limits lack bounded validation |
| KAI-GRAPH-007 | MEDIUM | Raw backend exception details are returned to callers |
| KAI-GRAPH-008 | MEDIUM | Health reports success without checking Cognee or graph-store readiness |
| KAI-GRAPH-009 | MEDIUM | Ingest completion can be reported without a durable deletion identifier |

---

## Memory graph: `memu-graph/app.py`

### KAI-GRAPH-001 — CRITICAL — Graph mutation endpoints are unauthenticated
**Issue:** `POST /graph/ingest` and `POST /graph/forget` accept caller-controlled memory content and source identifiers without authentication or authorisation.  
**Risk:** Any caller with network reachability can poison the knowledge graph or attempt deletion of indexed memories.  
**Recommendation:** Require authenticated service identity, scoped mutation capability and immutable provenance bound to each source record.  
**Status:** OPEN — immediate remediation required

### KAI-GRAPH-002 — HIGH — Failed deletion loses the retry mapping
**Issue:** `graph_forget` executes `_source_index.pop(req.source_id, None)` before calling `cognee.delete`. If deletion raises, the endpoint returns an error but the mapping has already been removed.  
**Risk:** The graph record remains while the service loses the `data_id` and `dataset_id` needed to retry, creating an orphan that ordinary forget calls can no longer remove.  
**Recommendation:** Retain mapping until confirmed idempotent deletion, and persist failure in a durable retry queue.  
**Status:** OPEN

### KAI-GRAPH-003 — HIGH — Deletion index is process-local
**Issue:** `_source_index` is an in-memory dictionary and is explicitly lost on restart.  
**Risk:** After any restart, existing graph records cannot be addressed by source ID for deletion, undermining erasure, reconciliation and MARS forget semantics.  
**Recommendation:** Persist source lineage transactionally in the graph store or a durable mapping database.  
**Status:** OPEN

### KAI-GRAPH-004 — HIGH — Backend access control is disabled
**Issue:** Startup sets `ENABLE_BACKEND_ACCESS_CONTROL=false`, while the FastAPI wrapper adds no replacement authentication layer.  
**Risk:** The graph backend and wrapper operate without an effective identity boundary, despite holding sensitive memory-derived relationships.  
**Recommendation:** Enable backend controls or enforce equivalent authenticated tenancy and service identity at the wrapper and network layers.  
**Status:** OPEN

### KAI-GRAPH-005 — HIGH — Graph query exposes memory-derived data
**Issue:** `GET /graph/query` returns Cognee search results to any reachable caller without access control or data classification.  
**Risk:** Personal, operational and relationship-derived entities and links can be enumerated or inferred.  
**Recommendation:** Require least-privilege read scopes, filter sensitive fields and apply query-rate and result controls.  
**Status:** OPEN

### KAI-GRAPH-006 — MEDIUM — Inputs and result limits are weakly bounded
**Issue:** Ingest text, source ID, category and metadata are not length/depth bounded. `top_k` accepts arbitrary integers without a finite range.  
**Risk:** Oversized ingest, metadata expansion and extreme searches can consume CPU, memory and storage.  
**Recommendation:** Add strict Pydantic bounds, body-size limits and capped pagination.  
**Status:** OPEN

### KAI-GRAPH-007 — MEDIUM — Backend errors leak to callers
**Issue:** Ingest and query raise HTTP errors containing raw exception text; forget returns `detail: str(exc)`.  
**Risk:** Callers can learn internal library, datastore, identifier and filesystem details.  
**Recommendation:** Return stable error codes and protected trace IDs only.  
**Status:** OPEN

### KAI-GRAPH-008 — MEDIUM — Health is not dependency-aware
**Issue:** `/health` returns `status: ok` based only on process state and the in-memory mapping count. It does not import or query Cognee or validate the underlying graph database.  
**Risk:** Orchestration can route traffic to an instance whose principal dependency is unavailable or corrupt.  
**Recommendation:** Separate liveness and readiness and perform bounded backend integrity checks.  
**Status:** OPEN

### KAI-GRAPH-009 — MEDIUM — Ingest may succeed without a usable deletion ID
**Issue:** After Cognee ingest and cognify, the endpoint returns `status: ingested` even when `add_result` does not yield a `data_id`; no source mapping is then stored.  
**Risk:** Data becomes queryable but cannot be deleted through the service’s stated source-ID interface.  
**Recommendation:** Require and durably persist deletion lineage before acknowledging ingest success, or mark the record unreconciled and retry.  
**Status:** OPEN

---

## Batch totals

- Findings: **9**
- Critical: **1**
- High: **4**
- Medium: **4**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **212**
- Critical: **27**
- High: **96**
- Medium: **88**
- Low: **1**

## Files materially reviewed in this batch

`memu-graph/app.py`.
