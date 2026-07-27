# Kai Code Audit — Letta Agent Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_LETTA_AGENT.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-LETTAX-001 | CRITICAL | Letta memory/tool operations execute without Tool Gate approval or a server-owned memory-write policy |
| KAI-LETTAX-002 | HIGH | A request can commit durable memory before response parsing fails and the endpoint returns HTTP 502 |
| KAI-LETTAX-003 | HIGH | Retrying the same task has no idempotency key and can duplicate conversation turns or archival memory |
| KAI-LETTAX-004 | HIGH | Client disconnect or request cancellation does not cancel the synchronous Letta/Ollama operation |
| KAI-LETTAX-005 | HIGH | Unlimited concurrent callers can interleave messages and memory-tool operations in one shared agent timeline |
| KAI-LETTAX-006 | HIGH | Letta client and SQLite/store thread-safety are not protected by a lock, queue or single-worker contract |
| KAI-LETTAX-007 | HIGH | Agent creation can succeed durably before globals are assigned, leaving an orphaned duplicate agent after failure |
| KAI-LETTAX-008 | HIGH | Repeated initialisation failures can create multiple partially configured or orphaned agents |
| KAI-LETTAX-009 | HIGH | The fixed agent name is not used to look up and reuse an existing persistent agent |
| KAI-LETTAX-010 | HIGH | Archival-memory export omits passage IDs, source, timestamps, trust, score and provenance |
| KAI-LETTAX-011 | HIGH | Memory export returns at most 200 passages without total count, pagination cursor or truncation marker |
| KAI-LETTAX-012 | HIGH | Export order is unspecified, so callers cannot establish stable chronology or completeness |
| KAI-LETTAX-013 | HIGH | Agent-run and memory-export responses lack `Cache-Control: no-store` |
| KAI-LETTAX-014 | HIGH | Generic `.text` extraction can expose tool, internal, system or intermediate message content as assistant output |
| KAI-LETTAX-015 | HIGH | Letta response messages, assistant output and exported passages have no service-side size bound |
| KAI-LETTAX-016 | HIGH | The service enforces no output-token or generated-message limit independent of the model/library |
| KAI-LETTAX-017 | HIGH | Archival memory has no contradiction, quarantine, trust-tier or correction-review boundary |
| KAI-LETTAX-018 | HIGH | There is no API or policy for deleting, superseding, correcting, expiring or legally retaining stored memories |
| KAI-LETTAX-019 | HIGH | Letta’s default agent persona, system prompt and enabled tools are library-controlled and not versioned or attested |
| KAI-LETTAX-020 | HIGH | Model and embedding identities are trusted as tags without verifying exact Ollama model digests/capabilities |
| KAI-LETTAX-021 | HIGH | Ollama endpoint identity is an arbitrary environment URL with no host allowlist, authentication or mTLS |
| KAI-LETTAX-022 | HIGH | Prompts, memories and embeddings travel over plain HTTP inside the service network |
| KAI-LETTAX-023 | HIGH | Embedding model/dimension compatibility is not probed before agent creation or health success |
| KAI-LETTAX-024 | HIGH | Chat-model context-window compatibility is not probed before agent creation or health success |
| KAI-LETTAX-025 | HIGH | Persistent Letta/SQLite memory is stored on an unencrypted volume without integrity protection |
| KAI-LETTAX-026 | HIGH | Data-directory ownership and mode are created but not verified on startup against an expected secure policy |
| KAI-LETTAX-027 | HIGH | The image unnecessarily copies the repository security directory into a public persistent-agent service |
| KAI-LETTAX-028 | HIGH | Trusted token and policy material becomes readable after any Letta/model/parser compromise |
| KAI-LETTAX-029 | HIGH | No PII, credential or secret classification is applied before writing prompts/context into Ollama or durable memory |
| KAI-LETTAX-030 | HIGH | Memory export provides no redaction or purpose-based field minimisation |
| KAI-LETTAX-031 | HIGH | No rate limit, concurrency cap, token budget, caller quota or admission queue protects model/memory operations |
| KAI-LETTAX-032 | HIGH | No tamper-evident audit links caller, message, model revision, tool calls and resulting memory changes |
| KAI-LETTAX-033 | HIGH | Agent responses have no request ID, message ID, conversation revision, model revision or processing timestamp |
| KAI-LETTAX-034 | MEDIUM | Context dictionary ordering becomes prompt semantics without a canonical schema or signed digest |
| KAI-LETTAX-035 | MEDIUM | Context values use arbitrary Python string conversion, creating unstable or misleading representations |
| KAI-LETTAX-036 | MEDIUM | Context keys can duplicate trusted-looking labels such as system, user, memory or policy |
| KAI-LETTAX-037 | MEDIUM | Function-call arguments, result, status and error are discarded when reporting memory updates |
| KAI-LETTAX-038 | MEDIUM | Multiple assistant/intermediate texts are concatenated without role, order or source labels |
| KAI-LETTAX-039 | MEDIUM | Empty response-message lists return HTTP 200 with an empty successful response |
| KAI-LETTAX-040 | MEDIUM | Agent-run and memory-export endpoints define no strict response models or versioned schema |
| KAI-LETTAX-041 | MEDIUM | Health exposes no Letta version, Ollama readiness, embedding readiness, store integrity or last successful operation |
| KAI-LETTAX-042 | MEDIUM | Health does not distinguish agent absent, initialising, failed, orphaned or ready states |
| KAI-LETTAX-043 | MEDIUM | Initialisation failure is not persisted into a cooldown/degraded readiness state |
| KAI-LETTAX-044 | MEDIUM | The writable HOME directory allows runtime caches/configuration to diverge from the persistent Letta volume |
| KAI-LETTAX-045 | MEDIUM | Upgrading pip, setuptools and wheel without version pins makes the image build non-reproducible |
| KAI-LETTAX-046 | MEDIUM | FastAPI dependencies and the Python base image are not reproducibly digest-pinned |
| KAI-LETTAX-047 | MEDIUM | The Ollama image/model pull chain uses mutable image tags and model tags rather than immutable digests |
| KAI-LETTAX-048 | MEDIUM | The service is deployed only in the full topology and has no equivalent minimal-stack state/migration contract |
| KAI-LETTAX-049 | MEDIUM | The test suite stubs the entire Letta package and never exercises SQLite, Ollama, embeddings or memory tools |
| KAI-LETTAX-050 | MEDIUM | Tests do not cover concurrent calls, duplicate-agent creation, request cancellation or side-effect-before-error behaviour |
| KAI-LETTAX-051 | MEDIUM | Tests explicitly validate the heuristic memory-write detector rather than a durable write acknowledgement |
| KAI-LETTAX-052 | MEDIUM | The service exposes no metrics for model latency, token use, memory writes, store size, queue depth or failures |
| KAI-LETTAX-053 | MEDIUM | Storage/model work has no explicit per-operation deadline separate from HTTP caller cancellation |
| KAI-LETTAX-054 | MEDIUM | No backup/snapshot revision is returned with exported memory, preventing consistent point-in-time interpretation |
| KAI-LETTAX-055 | MEDIUM | The service has no controlled admission-stop and in-flight drain before closing persistent resources during deployment replacement |

---

## Critical finding

### KAI-LETTAX-001 — CRITICAL — Durable memory tools bypass governance
**Issue:** `/agent/run` calls `lc.send_message()` directly. The Letta agent may invoke archival-memory tools, but the service obtains no Tool Gate decision and enforces no exact memory-write policy, source trust, review or operator approval.  
**Risk:** Anonymous or compromised callers can cause durable memory side effects through the agent’s tool layer without the repository’s stated execution-governance boundary.  
**Recommendation:** Bind every memory mutation to an authenticated, single-use policy grant for the exact content, principal, purpose and memory operation.  
**Status:** OPEN — immediate remediation required

---

## High-severity findings

### KAI-LETTAX-002 — HIGH — Side effect before failed response
A Letta tool may commit memory during `send_message`; later iteration or attribute access can raise, causing HTTP 502 while the durable side effect remains.

### KAI-LETTAX-003 — HIGH — Duplicate retry side effects
The endpoint has no operation ID or idempotency key. Client/proxy retries can append the same turn and memory repeatedly.

### KAI-LETTAX-004 — HIGH — Cancellation does not stop model/storage work
The synchronous call runs on the event-loop thread; disconnect/cancellation cannot reliably interrupt the underlying Ollama/SQLite operation.

### KAI-LETTAX-005 — HIGH — Concurrent global conversation interleaving
There is no lock or per-principal queue around `send_message` to the same agent ID.

### KAI-LETTAX-006 — HIGH — Uncontrolled client/store concurrency
The implementation assumes one shared Letta client and persistent store are safe under concurrent requests and multiple workers without proving or enforcing that contract.

### KAI-LETTAX-007 — HIGH — Orphan agent after partial initialisation
`create_agent()` can persist an agent before `_letta_client` and `_agent_id` are assigned; a later failure causes another creation attempt.

### KAI-LETTAX-008 — HIGH — Repeated partial initialisation
No transaction/idempotent registry binds client creation, agent creation and global activation.

### KAI-LETTAX-009 — HIGH — Fixed name does not prevent duplication
The service always creates `kai-memory-agent`; it never lists/resolves an existing agent by name or stable configured ID.

### KAI-LETTAX-010 — HIGH — Export strips evidence metadata
Only `p.text` is returned, making source, age, ownership, memory identity and trust impossible to evaluate.

### KAI-LETTAX-011 — HIGH — Hidden export truncation
The hard-coded 200 limit has no cursor, total available count or `truncated=true` state.

### KAI-LETTAX-012 — HIGH — Unstable memory chronology
No sort/order contract or timestamp is returned.

### KAI-LETTAX-013 — HIGH — Cacheable persistent memory
Private agent output and archival memory have no privacy cache controls.

### KAI-LETTAX-014 — HIGH — Internal message leakage
For each response message, the code accepts `assistant_message` or generic `text`; message type/role is not restricted.

### KAI-LETTAX-015 — HIGH — Unbounded output allocation
All message texts are converted and joined; all passage texts are materialised and returned.

### KAI-LETTAX-016 — HIGH — No service-owned generation cap
Context window configuration is not an output limit; the wrapper accepts whatever message volume the backend/library emits.

### KAI-LETTAX-017 — HIGH — No evidence quality boundary
Archival memories are not independently marked verified, contradicted, poisoned, corrected or synthetic.

### KAI-LETTAX-018 — HIGH — No memory lifecycle API/policy
The service supports write/use/export but no correction/deletion/retention/consent operation.

### KAI-LETTAX-019 — HIGH — Unattested agent behaviour
The wrapper does not specify or hash the system prompt, tool set, memory policies or agent configuration created by Letta defaults.

### KAI-LETTAX-020 — HIGH — Mutable model identity
Compose pulls Ollama model tags and the service records only tag strings, not exact artefact digests/capabilities.

### KAI-LETTAX-021 — HIGH — Arbitrary unverified Ollama destination
`OLLAMA_BASE_URL` is used directly, with no approved host, service identity or response attestation.

### KAI-LETTAX-022 — HIGH — Plaintext model/memory transport
The default Ollama endpoint is HTTP within the shared service network.

### KAI-LETTAX-023 — HIGH — Embedding compatibility unverified
The service trusts the configured model and dimension without a startup embedding-length/capability probe.

### KAI-LETTAX-024 — HIGH — Chat capability unverified
Health does not prove that the configured model supports the requested context window or successful generation.

### KAI-LETTAX-025 — HIGH — Unprotected persistent memory volume
Letta’s SQLite/store data is ordinary volume content without encryption, signatures, append integrity or trusted snapshot anchoring.

### KAI-LETTAX-026 — HIGH — Data permissions not asserted
Startup `makedirs(... exist_ok=True)` does not verify owner, group, symlink, mode or mount properties of an existing data path.

### KAI-LETTAX-027 — HIGH — Security bundle copied unnecessarily
The Dockerfile copies `security/`, although the service imports nothing from it.

### KAI-LETTAX-028 — HIGH — Additional compromise loot
Repository token/policy files unnecessarily expand the impact of a public model/memory service compromise.

### KAI-LETTAX-029 — HIGH — No sensitive-data filter
Task/context are sent to the model and may be retained by agent memory without PII/credential classification.

### KAI-LETTAX-030 — HIGH — No export minimisation
Every returned passage is exposed in full without caller purpose or redaction.

### KAI-LETTAX-031 — HIGH — No workload governance
Expensive model, embedding and database operations are unrestricted by caller or global capacity.

### KAI-LETTAX-032 — HIGH — Missing durable action audit
No immutable event links input, model/tool messages, exact memory writes and response.

### KAI-LETTAX-033 — HIGH — Missing result provenance
Responses cannot be correlated to a particular model artefact, agent revision or conversation state.

---

## Medium-severity findings

### KAI-LETTAX-034 — MEDIUM — Context order becomes meaning
Dictionary insertion order controls the flattened prompt sequence without a canonical signed representation.

### KAI-LETTAX-035 — MEDIUM — Unstable string conversion
Arbitrary nested values rely on Python `str()`, which may be ambiguous, huge or implementation-dependent.

### KAI-LETTAX-036 — MEDIUM — Trusted-looking context labels
Keys are unrestricted and can imitate system/policy/memory labels within the plaintext prefix.

### KAI-LETTAX-037 — MEDIUM — Tool evidence discarded
Function arguments/result/error/status are unavailable to the caller/auditor.

### KAI-LETTAX-038 — MEDIUM — Roleless output concatenation
Different messages are joined with newlines and lose type/role/tool provenance.

### KAI-LETTAX-039 — MEDIUM — Empty success
No reply text still produces HTTP 200 with `response=""`.

### KAI-LETTAX-040 — MEDIUM — Unversioned API response
No Pydantic response model or schema revision is enforced.

### KAI-LETTAX-041 — MEDIUM — Incomplete health identity
Health reports only status, agent ID and configured model string.

### KAI-LETTAX-042 — MEDIUM — No initialisation state machine
A null agent ID is health-green and failures/orphans are not represented.

### KAI-LETTAX-043 — MEDIUM — Retry storm after init failure
Each request may immediately repeat heavyweight/partial initialisation.

### KAI-LETTAX-044 — MEDIUM — Ephemeral HOME divergence
Libraries may write caches/config under `/data/home`, which is not mounted as the Letta persistent volume.

### KAI-LETTAX-045 — MEDIUM — Mutable build tooling
The image upgrades pip/setuptools/wheel to whatever versions are current during build.

### KAI-LETTAX-046 — MEDIUM — Mutable service dependencies
FastAPI lower bounds and the base image tag reduce build reproducibility.

### KAI-LETTAX-047 — MEDIUM — Mutable Ollama supply chain
Both Ollama image and model tags lack immutable digests.

### KAI-LETTAX-048 — MEDIUM — Topology-specific persistence contract
Only the full stack defines the service/volume; migration/absence behaviour in other profiles is not governed.

### KAI-LETTAX-049 — MEDIUM — Fully mocked tests
The test replaces the Letta package and all client/model/storage behaviour.

### KAI-LETTAX-050 — MEDIUM — Critical concurrency/partial-side-effect paths untested
No tests exercise actual persistence, races or cancellation.

### KAI-LETTAX-051 — MEDIUM — Heuristic behaviour is enshrined by tests
The suite asserts that a function name substring means a successful memory update, rather than testing a durable acknowledgement.

### KAI-LETTAX-052 — MEDIUM — No operational metrics
Model/store health and workload are invisible.

### KAI-LETTAX-053 — MEDIUM — No explicit operation deadline
The wrapper has no model/storage timeout independent of HTTP lifecycle.

### KAI-LETTAX-054 — MEDIUM — Export lacks snapshot consistency
No store revision or as-of marker accompanies the exported set.

### KAI-LETTAX-055 — MEDIUM — Incomplete deployment drain
The lifespan does not stop admission, wait for synchronous calls or prove store flush before replacement.

---

## Batch totals

- Findings: **55**
- Critical: **1**
- High: **32**
- Medium: **22**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,857**
- Critical: **194**
- High: **1,457**
- Medium: **1,203**
- Low: **3**

## Files materially reviewed

`letta-agent/app.py`, `letta-agent/Dockerfile`, `letta-agent/requirements.txt`, `scripts/test_letta_agent.py`, Letta/Ollama deployment in `docker-compose.full.yml`, persistent-volume/backup references and the existing Letta Agent audit.
