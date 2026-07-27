# Kai Code Audit — Fusion Engine Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-FUSION-001 | CRITICAL | Empty specialist lists are assigned 100% agreement and can return `consensus: true` |
| KAI-FUSION-002 | CRITICAL | A single specialist is automatically assigned 100% agreement |
| KAI-FUSION-003 | CRITICAL | Caller-controlled `min_agreement=0.0` makes any result pass the consensus gate |
| KAI-FUSION-004 | HIGH | Deterministic stub responses participate in consensus as if they were specialist evidence |
| KAI-FUSION-005 | HIGH | Duplicate specialist names can manufacture apparent multi-specialist agreement |
| KAI-FUSION-006 | HIGH | Caller-controlled context is injected directly as the system message to every LLM backend |
| KAI-FUSION-007 | HIGH | Verifier output does not affect the returned consensus verdict |
| KAI-FUSION-008 | HIGH | `require_consensus` controls only verifier invocation and does not require consensus |
| KAI-FUSION-009 | HIGH | Specialist count and outbound parallelism are unbounded |
| KAI-FUSION-010 | HIGH | Semantic model loading and encoding run synchronously inside the async request path |
| KAI-FUSION-011 | MEDIUM | Agreement measures textual similarity rather than factual correctness |
| KAI-FUSION-012 | MEDIUM | Merge logic selects the longest response rather than resolving conflicts |
| KAI-FUSION-013 | MEDIUM | Only the first 500 characters of the merged result are sent for verification |
| KAI-FUSION-014 | MEDIUM | LLM response size and JSON structure are not bounded or schema-validated |
| KAI-FUSION-015 | MEDIUM | Backend error strings are embedded in API responses |
| KAI-FUSION-016 | MEDIUM | One global LLM error guard aggregates unrelated backend failures |
| KAI-FUSION-017 | MEDIUM | Backend inventory and reachability are exposed without authentication |
| KAI-FUSION-018 | MEDIUM | Health reports ok while operating entirely on canned stubs |
| KAI-FUSION-019 | MEDIUM | Fusion hashes are non-reproducible timestamps, not integrity proofs |
| KAI-FUSION-020 | MEDIUM | Backend URLs, agreement mode and numeric configuration are not validated |

---

## Fusion engine: `fusion-engine/app.py`

### KAI-FUSION-001 — CRITICAL — Empty specialist set passes consensus
**Issue:** `_measure_agreement` returns `1.0` whenever `len(responses) < 2`. `FusionRequest.specialists` accepts an empty list, so no backend is queried, `_merge_responses` returns “No valid specialist responses available,” yet the default threshold produces `consensus: true`.  
**Risk:** A high-confidence decision gate can certify a result produced with zero evidence and zero specialist responses.  
**Recommendation:** Reject fewer than two distinct successful specialists and fail closed when evidence is absent.  
**Status:** OPEN — immediate remediation required

### KAI-FUSION-002 — CRITICAL — One specialist is treated as unanimous consensus
**Issue:** The same `len(responses) < 2` branch assigns agreement `1.0` to a one-specialist request.  
**Risk:** Any single model response is represented as full multi-signal convergence, defeating the stated purpose of independent specialist agreement and human co-sign escalation.  
**Recommendation:** Require a configured minimum number of distinct independent backends before calculating consensus.  
**Status:** OPEN — immediate remediation required

### KAI-FUSION-003 — CRITICAL — Consensus threshold can be set to zero
**Issue:** Callers may set `min_agreement` anywhere from `0.0` to `1.0`. The verdict is simply `agreement >= min_agreement`, so `0.0` makes even zero agreement pass.  
**Risk:** An unauthenticated caller can explicitly bypass the conviction gate and receive `consensus: true` for divergent, failed or empty evidence.  
**Recommendation:** Make the threshold server-controlled, policy-versioned and bounded to a safe minimum that callers cannot reduce.  
**Status:** OPEN — immediate remediation required

### KAI-FUSION-004 — HIGH — Stub text is treated as specialist evidence
**Issue:** When no backend URL exists, `_query_specialist` returns a canned deterministic paragraph with `source: stub`. Stub responses are included in agreement, merging and consensus exactly like live model responses.  
**Risk:** Similar boilerplate from multiple stubs can generate apparent agreement and a high-confidence result despite no model inference occurring.  
**Recommendation:** Mark stub mode non-operational and prohibit it from producing consensus or verification-ready output.  
**Status:** OPEN

### KAI-FUSION-005 — HIGH — Duplicate specialists manufacture multiplicity
**Issue:** `specialists` is an unrestricted list. Duplicate names create duplicate tasks against the same backend and are counted as separate responses in agreement and merge messaging.  
**Risk:** Repeating one model name can create artificial multi-specialist convergence without independent evidence.  
**Recommendation:** Deduplicate names and require independently configured backend identities.  
**Status:** OPEN

### KAI-FUSION-006 — HIGH — Caller data controls the system prompt
**Issue:** `req.context` is sent verbatim as the `system` message to every configured LLM backend. No trusted/untrusted separation, schema or policy prefix is applied.  
**Risk:** Any caller can redefine backend behaviour, impersonate policy or inject instructions into the highest-priority prompt channel.  
**Recommendation:** Keep system policy server-controlled and place caller context in a clearly delimited untrusted data channel.  
**Status:** OPEN

### KAI-FUSION-007 — HIGH — Verification cannot veto consensus
**Issue:** The consensus Boolean is calculated before the verifier request. Whatever the verifier returns is attached as metadata only; it is never evaluated to alter or block the verdict.  
**Risk:** A failed, negative or contradictory verification result can accompany `consensus: true`, while downstream consumers may trust the consensus field.  
**Recommendation:** Define typed verifier outcomes and fail/hold the verdict when verification rejects or is unavailable for required workflows.  
**Status:** OPEN

### KAI-FUSION-008 — HIGH — `require_consensus` is semantically false
**Issue:** The field named `require_consensus` only decides whether to call the verifier. It does not reject low agreement, request human co-sign or prevent a merged response from being returned.  
**Risk:** Callers and integrators can believe consensus is enforced when the endpoint always returns a result regardless of the verdict.  
**Recommendation:** Rename the field or enforce the documented gate with explicit blocked/needs-human states.  
**Status:** OPEN

### KAI-FUSION-009 — HIGH — Unbounded specialist fan-out
**Issue:** The specialist list has no item-count or aggregate-length limit. One task is created for every entry and all are launched concurrently.  
**Risk:** One unauthenticated request can create arbitrary outbound LLM traffic, model load and memory usage for up to the backend timeout.  
**Recommendation:** Enforce a small server-approved specialist set and bounded concurrency, quotas and rate limits.  
**Status:** OPEN

### KAI-FUSION-010 — HIGH — Embedding model work blocks the event loop
**Issue:** `_semantic_agreement` imports and constructs `SentenceTransformer(model_name)` and calls `model.encode` synchronously during `/fuse`. The model is recreated on every request and may load/download substantial assets.  
**Risk:** Unauthenticated calls can block the event-loop worker, exhaust memory/CPU and repeatedly initialise the embedding model.  
**Recommendation:** Load one approved local model during startup and run encoding in a bounded worker pool.  
**Status:** OPEN

### KAI-FUSION-011 — MEDIUM — Similar wording is equated with correctness
**Issue:** Agreement is Jaccard keyword overlap or embedding cosine similarity. Neither method checks factual accuracy, reasoning validity, independence or whether specialists share the same error.  
**Risk:** Fluent or templated responses can score highly while being jointly wrong; differently worded correct answers can score poorly.  
**Recommendation:** Label the metric textual similarity and combine it with evidence-backed verification and calibrated task-specific evaluation.  
**Status:** OPEN

### KAI-FUSION-012 — MEDIUM — Merge chooses length, not consensus content
**Issue:** `_merge_responses` selects the longest non-error response as primary and appends a statement that other specialists were queried. It does not identify agreements, contradictions or supported propositions.  
**Risk:** Verbosity becomes the selection criterion, and the resulting label can imply agreement that the merge never established.  
**Recommendation:** Use a structured proposition/evidence comparison and explicitly surface unresolved conflicts.  
**Status:** OPEN

### KAI-FUSION-013 — MEDIUM — Verification is truncated to 500 characters
**Issue:** The verifier receives only `merged[:500]`, while the complete merged response is returned to callers.  
**Risk:** Claims, qualifications or unsafe content after the first 500 characters are never verified, yet the attached verification may be interpreted as covering the whole response.  
**Recommendation:** Verify the complete bounded result or return precise verified-span metadata.  
**Status:** OPEN

### KAI-FUSION-014 — MEDIUM — LLM payloads are unbounded and weakly parsed
**Issue:** Prompt, context and specialist names lack length limits. Backend response bytes and JSON complexity are unrestricted, and the code extracts nested fields without a validated response schema.  
**Risk:** Oversized requests or backend responses consume memory and can produce malformed empty evidence that is still processed.  
**Recommendation:** Enforce body/token/response limits and validate a strict backend schema.  
**Status:** OPEN

### KAI-FUSION-015 — MEDIUM — Backend errors are returned as content
**Issue:** Exceptions become specialist response text such as `[error: ...]`, and all specialist responses are returned to the unauthenticated caller.  
**Risk:** Network, model and internal diagnostic details can leak through the API.  
**Recommendation:** Return stable error codes and keep detailed diagnostics in protected logs.  
**Status:** OPEN

### KAI-FUSION-016 — MEDIUM — One circuit state couples all LLMs
**Issue:** `LLM_ERROR_GUARD` is global for every configured specialist backend. Failures from one backend affect the allow/deny state for all others.  
**Risk:** One degraded or attacker-targeted backend can suppress healthy independent specialists, reducing evidence diversity and availability.  
**Recommendation:** Maintain per-backend breakers plus a separately governed aggregate policy.  
**Status:** OPEN

### KAI-FUSION-017 — MEDIUM — Backend topology is public
**Issue:** `/health` and `/backends` expose configured specialist names, counts, breaker states, stub mode and reachability without authentication.  
**Risk:** Callers can map model infrastructure and time attacks around degraded components.  
**Recommendation:** Restrict detailed operational metadata to authorised administrators.  
**Status:** OPEN

### KAI-FUSION-018 — MEDIUM — Health treats stubs as ready
**Issue:** `/health` always returns `status: ok`, including when no LLM backends are configured and every result will be canned stub text. It also does not test verifier readiness.  
**Risk:** Orchestration treats a non-inferential test implementation as a functioning high-confidence decision service.  
**Recommendation:** Separate liveness, live-backend readiness and verification readiness.  
**Status:** OPEN

### KAI-FUSION-019 — MEDIUM — Fusion hash is not evidence integrity
**Issue:** `fusion_hash` hashes prompt, agreement and current time only. It excludes specialist responses, context, backend identities, verification and configuration.  
**Risk:** The hash cannot reproduce, authenticate or prove the evaluated evidence despite appearing as a result identifier.  
**Recommendation:** Hash a canonical signed record containing all inputs, outputs, identities, policy versions and verifier result.  
**Status:** OPEN

### KAI-FUSION-020 — MEDIUM — Configuration lacks validation
**Issue:** Backend URLs and names, agreement strategy, embedding model and service port are accepted directly. Unknown agreement modes silently use semantic/auto behaviour; unsafe or malformed URLs are not rejected.  
**Risk:** Misconfiguration can route prompts to unintended destinations, trigger model downloads or create inconsistent scoring behaviour.  
**Recommendation:** Validate a typed allowlisted configuration at startup and pin local model artefacts.  
**Status:** OPEN

---

## Batch totals

- Findings: **20**
- Critical: **3**
- High: **7**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **678**
- Critical: **77**
- High: **235**
- Medium: **363**
- Low: **3**

## Files materially reviewed in this batch

`fusion-engine/app.py`.
