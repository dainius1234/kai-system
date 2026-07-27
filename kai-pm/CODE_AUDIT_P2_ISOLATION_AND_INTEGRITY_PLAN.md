# Kai System — Phase 2 Isolation and Integrity Plan

Repository: `dainius1234/kai-system`  
Authoritative audit baseline: **4,580 findings — 252 Critical, 2,440 High, 1,885 Medium, 3 Low**  
Parent backlog: `kai-pm/CODE_AUDIT_REMEDIATION_BACKLOG.md`  
Dependencies:

- `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md`
- `kai-pm/CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`

Status: **IMPLEMENTATION DESIGN ONLY — NO RUNTIME REMEDIATION PERFORMED**

---

## 1. Objective

Build the isolation and integrity boundaries required before Kai can process hostile content, browse external sites, execute tools or store sensitive/persistent memory.

Phase 2 establishes:

1. Fixed-schema execution operations instead of generic command interpreters.
2. Disposable, resource-limited execution workers.
3. One isolated browser context per principal and approved operation.
4. One controlled egress authority with SSRF and redirect protection.
5. Disposable parser/converter workers for every hostile document.
6. Principal-, tenant-, purpose- and data-class partitioning across all stores.
7. Typed immutable evidence and provenance records.
8. A strict untrusted-content boundary before web/document/sensor/model data reaches prompts or authority systems.
9. Transactional memory, vector and graph mutation with durable lineage.
10. Verified supersession, contradiction and derivative deletion.

Phase 2 depends on Phase 1 operation identities and capabilities. Isolation without exact principal, operation and authority binding would remain a collection of advisory controls.

---

## 2. Governing security invariants

### INV-P2-01 — No generic execution authority

Ordinary Kai operations must not accept arbitrary shell commands, Python expressions, Make/Git/Pip command lines, caller-selected scripts or unrestricted command arguments.

Every executable operation has:

- a server-owned operation identifier;
- a strict versioned input schema;
- an immutable implementation artefact digest;
- bounded inputs, outputs, time and resources;
- explicit filesystem and network grants;
- an expected postcondition;
- an exact Phase 1 capability.

### INV-P2-02 — One disposable worker per hostile operation

Execution, parsing, conversion and other hostile-content work runs in a disposable worker with no inherited application secrets, no broad service-network reachability and no persistent writable state.

### INV-P2-03 — Browser state never crosses principals or workflows

Cookies, localStorage, sessionStorage, IndexedDB, cache, service workers, permissions, downloads, popups and authenticated page state belong to one principal and one approved workflow. They are destroyed or explicitly retained under policy when that workflow ends.

### INV-P2-04 — External destinations are policy objects

No component directly interprets a caller URL as network authority. Every connection and redirect is checked against an approved destination policy after DNS resolution and immediately before connection.

### INV-P2-05 — Extracted content is untrusted evidence

Web pages, documents, OCR, email, news, screen, clipboard, camera, audio, model output and external API content are never promoted directly into system instructions, operator preferences, verified evidence or trusted memory.

### INV-P2-06 — Every stored record is principal and purpose scoped

A record without authenticated `principal_id`, `tenant_id`, purpose, data class and retention class is invalid. No global `keeper` namespace or empty-user wildcard is permitted.

### INV-P2-07 — Every derivative has lineage

Vectors, graph nodes, summaries, embeddings, compressed memories, prompt-context records, caches, archives and backups must identify their authoritative source record/version and transformation revision.

### INV-P2-08 — Multi-store mutation is a durable operation

Memory, vector, graph and mapping updates acknowledge success only after a durable state machine records all required steps and verified terminal state. Partial commits are visible, retryable and compensatable.

### INV-P2-09 — Supersession changes authority atomically

When a record is corrected or superseded, the old version cannot remain active in retrieval, graph search, prompt assembly or trust scoring. History may remain where policy requires, but it is clearly non-active and access-controlled.

### INV-P2-10 — Deletion is end-to-end and verifiable

A deletion request follows lineage across primary stores, caches, vectors, graph, prompts, indexes, exports and backup policy. Completion is not reported until applicable derivatives are gone or placed into a documented cryptographic-expiry state.

---

## 3. Confirmed source conditions driving Phase 2

## 3.1 Executor

Primary source:

- `executor/app.py`

Primary audit:

- `kai-pm/CODE_AUDIT_BATCH_EXECUTOR.md`

Confirmed conditions include:

- Open `/execute` endpoint with no final Tool Gate enforcement.
- Generic command allowlist contains Python, Find, Make, Pip, Git, Docker and Curl primitives capable of arbitrary code, file and network effects.
- Python-expression validation can be bypassed.
- Caller-controlled scripts/arguments and basename-based command selection.
- No true filesystem, network, memory, process or descendant containment.
- Timeout controls only the direct process.
- “Rollback” removes an in-memory metadata item and reverses no real effect.
- Subprocess output is buffered before limits.
- Workers inherit service environment and broad network access.
- Duplicate task IDs execute repeatedly.
- State/history is process-local, unbounded and not linked to authoritative operation/capability identity.

Primary finding groups:

- `KAI-EXEC-001` through `KAI-EXEC-014`.
- `KAI-EXEC-015` through `KAI-EXEC-034`.
- `KAI-EXEC-043` through `KAI-EXEC-053`.
- `KAI-EXEC-064`, `KAI-EXEC-070`, `KAI-EXEC-071`, `KAI-EXEC-074`.

## 3.2 Browser Agent

Primary source:

- `browser-agent/app.py`
- Browser Agent Dockerfile and tests.

Primary audits:

- `CODE_AUDIT_BATCH_BROWSER_AGENT.md`
- `CODE_AUDIT_BATCH_BROWSER_AGENT_EXTENSION.md`

Confirmed conditions include:

- Browser actions bypass Tool Gate.
- One browser context/page persists cookies and authenticated state across callers.
- No context reset, logout or per-principal ownership.
- Popups, downloads, service workers, WebSockets and background traffic remain active.
- Click/type operations can cause side effects before errors are returned.
- Fuzzy target selection and no postcondition verification.
- No operation identity/idempotency.
- World-writable Playwright installation and mutable browser artefacts.
- No verified browser sandbox.
- No download, subresource, response-byte or total-request budget.
- Scraped content lacks untrusted-evidence metadata.
- Screenshots and authenticated content lack privacy cache controls and provenance.
- Tests mock Playwright and do not prove real isolation, internal-destination denial or cleanup.

Primary finding groups:

- `KAI-BROWSERX-001` through `KAI-BROWSERX-034`.
- Original Browser Agent findings.

## 3.3 Monitor/browser interaction

Primary sources:

- Monitor service source.
- Browser Agent scrape endpoints.

Primary audits:

- `CODE_AUDIT_BATCH_MONITOR_SERVICE.md`
- `CODE_AUDIT_BATCH_MONITOR_SERVICE_EXTENSION.md`

Confirmed conditions include:

- Monitor scrape rules read whichever shared Browser Agent page another workflow left open.
- Configured scrape URL and selector are not applied by the called endpoint.
- Alerts can disclose another authenticated browser workflow while claiming a different source.
- Disabled/deleted/updated rules can continue in-flight work or delivery.
- Source evidence lacks digest, source event time, freshness and service identity.
- Notify/TTS actions bypass final policy enforcement.

Critical mapping:

- `KAI-MONITORX-001`.
- `KAI-MONITORX-002` through `KAI-MONITORX-029`.

## 3.4 Document/parser/converter services

Primary sources:

- Document Parser application and Dockerfile.
- OCR, CAD/DWG/DXF and other conversion services.
- Upload/file services.

Primary audits:

- `CODE_AUDIT_BATCH_DOCUMENT_PARSER.md`
- `CODE_AUDIT_BATCH_DOCUMENT_PARSER_EXTENSION.md`
- Files/OCR/screen/vision extension batches.

Confirmed conditions include:

- OOXML ZIP containers bypass archive member/ratio/expanded-size controls.
- ZIP directory metadata materialises before limits.
- Duplicate members, nested archives and unsupported binary formats are misclassified or decoded as text.
- Unknown extensions and invalid JSON can produce success-shaped results.
- PDF/CSV/JSON/CAD parsing creates multiple unbounded in-memory representations.
- External converters inherit environment, filesystem and network access.
- Converter output and stderr are buffered without pre-materialisation limits.
- Timeouts do not ensure descendant cleanup.
- No per-job disk/resource budget.
- Extracted text has no untrusted-document classification or immutable provenance.
- Parser results omit source digest, parser versions, job ID, truncation and completeness metadata.

Critical mapping:

- `KAI-DOCPARSEX-001`.
- High parser/converter findings `002` through `029`.

## 3.5 Vault and file ingestion

Primary sources:

- `vault-sync/app.py`
- `vault-sync/mapper.py`
- `vault-sync/parser.py`
- `vault-sync/watcher.py`

Primary audit:

- `CODE_AUDIT_BATCH_VAULT_SYNC.md`
- Vault Bridge extension batch.

Confirmed conditions include:

- Manual ingest can read arbitrary container-readable files outside the vault.
- Automatic watcher ingestion follows symlinks outside the vault.
- Export trusts caller-supplied conviction instead of exact policy capability.
- Path validation has check-then-write races.
- Writes are non-atomic and can overwrite existing notes.
- Graph deletion failure can discard the only local mapping.
- Mapping persistence failure is hidden.
- Existing files are not initially reconciled.
- Path aliases create duplicate graph nodes and broken deletion lineage.
- Ingest/delete queues are unbounded.
- Search exposes private synced memory.

Critical mapping:

- `KAI-VAULT-001`, `002`, `012`, `013`.
- `KAI-VAULT-003` through `022`.

## 3.6 memU Core

Primary source:

- `memu-core/app.py`

Primary audits:

- `CODE_AUDIT_BATCH_MEMU_CORE_HOT_PATH.md`
- memU introspection and personality/autonomy batches.

Confirmed conditions include:

- Open memory authority with caller-supplied users and sessions.
- Caller can impersonate `keeper`, create pinned preferences and inject system-role messages.
- Verifier gating stores non-PASS states under several failure conditions.
- Arbitrary-user retrieval and route endpoints disclose private records and embeddings.
- Stored records omit authenticated source provenance.
- State mutation can commit before memory insertion.
- Graph ingest is fire-and-forget and not part of write success.
- Empty user IDs can become globally visible.
- Retrieval mutates access/stability and repeated queries strengthen selected records.
- Ranking uses caller-manipulable importance/relevance/pin/access fields.
- Superseded records may remain active.
- TurboVec/Postgres trimming can diverge.
- Upsert updates only part of security/provenance state.
- Redis/session failure silently creates process-local split brain.
- Prompt context is assembled from untrusted memory as ready-to-inject content.

Critical mapping:

- `KAI-MEMCORE-001` through `KAI-MEMCORE-008`.
- High write, retrieval, storage, graph and session findings.

## 3.7 Memory graph

Primary sources:

- `memu-graph/app.py`
- Graph deployment/build files.

Primary audits:

- `CODE_AUDIT_BATCH_MEMORY_GRAPH.md`
- `CODE_AUDIT_BATCH_MEMORY_GRAPH_EXTENSION.md`

Confirmed conditions include:

- Re-ingest overwrites the only deletion mapping and orphans old graph data.
- `add()` can commit before `cognify()` fails, leaving data without lineage.
- No idempotency/content digest.
- Only one returned graph object may be indexed for deletion.
- Source/category strings can cross-link unrelated domains.
- Whole-dataset cognification runs per write and can overlap concurrently.
- Search can observe an inconsistent mutation generation.
- Ingest/forget races.
- Deletion is not verified and failures may return HTTP 200.
- One global dataset namespace for all users/purposes.
- No reconciliation for unmapped/orphaned data.

Critical mapping:

- `KAI-GRAPHX-001`, `KAI-GRAPHX-002`.
- `KAI-GRAPHX-003` through `KAI-GRAPHX-020`.

---

# 4. Target execution architecture

## 4.1 Fixed operation registry

Replace generic execution requests with a server-owned registry.

Suggested files:

- `security/operation_registry.yaml`
- `common/security/operation_registry.py`

Each operation definition must contain:

```text
operation_type
schema_version
implementation_artefact_digest
input_schema
output_schema
allowed_filesystem_inputs
allowed_filesystem_outputs
network_policy
cpu_limit
memory_limit
process_limit
time_limit
stdout_limit
stderr_limit
expected_postconditions
cleanup_policy
consequence_class
required_capability_scope
owner
```

Examples of acceptable fixed operations:

- Render a known report from a supplied bounded JSON file.
- Run a reviewed health diagnostic with fixed flags.
- Convert one uploaded file using a pinned converter.
- Read a declared file object through a brokered content handle.

Examples that remain prohibited:

- Arbitrary shell text.
- Arbitrary Python expression/module.
- Caller-selected binary/absolute path.
- Caller-controlled Git/Pip/Make/Curl/Docker command line.
- Script selected only by filename.

## 4.2 Disposable workers

Each operation receives a new disposable worker/container/VM/sandbox.

Required controls:

- Immutable image pinned by digest.
- Non-root user with no privilege escalation.
- Read-only root filesystem.
- `cap_drop: ALL` and a minimal syscall profile.
- No host PID/network/user namespace.
- No Docker socket or remote Docker authority.
- No application/service secrets.
- Explicit read-only input mount or object handle.
- Explicit bounded output mount.
- Empty temporary filesystem with quota.
- Default-deny network namespace.
- CPU, memory, pids, file-size, open-file and wall-time limits.
- Process group/session ownership and forced descendant termination.
- Worker destroyed after result collection.

## 4.3 Execution broker

The long-running Executor service becomes a broker, not the command environment.

Broker responsibilities:

1. Verify and consume the Phase 1 capability.
2. Resolve approved operation definition and artefact digest.
3. Create isolated worker with exact grants.
4. Stream bounded input.
5. Enforce deadline and cancellation.
6. Collect bounded stdout/stderr/outputs.
7. Verify postconditions.
8. Destroy worker and reconcile resources.
9. Record outcome against operation digest.

The broker must not execute arbitrary subprocesses in its own container.

---

# 5. Target browser architecture

## 5.1 Browser job identity

Every browser job binds:

- Phase 1 operation ID/digest and capability.
- Authenticated principal/tenant.
- Browser context ID.
- Approved destination policy.
- Locale/timezone/user-agent/viewport.
- Allowed permissions.
- Download and upload policy.
- Retention/cleanup policy.
- Expected page/action postcondition.

## 5.2 Per-operation context

Default behaviour:

- Launch a new browser context for each operation/workflow.
- No shared persistent profile.
- No inherited cookies or storage.
- Reject popups/downloads/permissions unless explicitly authorised.
- Block service workers and persistent background connections unless required.
- Close every page/context at terminal state.
- Delete context storage and downloads.

Longer authenticated workflows may retain one context only under a principal-scoped workflow lease with expiry, explicit logout and deletion evidence.

## 5.3 Exact action model

Replace fuzzy action endpoints with typed actions:

```text
NAVIGATE(url_policy_ref, expected_origin)
CLICK(target_fingerprint, expected_page_revision, expected_postcondition)
TYPE(target_fingerprint, value_digest, expected_page_revision)
SUBMIT(form_fingerprint, expected_postcondition)
EXTRACT(selector_or_accessibility_target, output_schema)
SCREENSHOT(page_revision, redaction_policy)
```

Target fingerprint should use stable reviewed attributes and require uniqueness. `exact=False` first-match behaviour is not acceptable for consequential actions.

## 5.4 Postcondition verification

Success requires observed evidence such as:

- expected final origin/URL;
- expected DOM/accessibility state;
- expected form value;
- expected confirmation element;
- expected downloaded object digest;
- no unexpected popup/navigation;
- no unresolved background activity above policy.

A timeout after a click is `OUTCOME_UNKNOWN`, not “no effect”.

## 5.5 Monitor redesign

Monitor must not scrape a shared Browser Agent page.

Each scrape rule must have:

- immutable rule revision;
- principal and purpose;
- approved destination and selector;
- isolated browser context/job;
- source response/page digest;
- retrieval/event timestamps;
- freshness and completeness status;
- condition result;
- separately capability-governed action/delivery.

Updating, disabling or deleting a rule must fence old revisions and cancel in-flight work before action delivery.

---

# 6. Controlled egress architecture

## 6.1 Egress proxy

All browser, Web Scout, feed, parser update, package/model and provider traffic must use one hardened egress authority.

Direct outbound connections from application and worker networks are denied.

## 6.2 Destination policy

Validate at every hop:

- scheme;
- normalised hostname;
- destination port;
- DNS answers;
- IP class/range;
- TLS server identity;
- redirects;
- protocol upgrade;
- proxy tunnelling;
- content type and expected service.

Default denied:

- loopback;
- RFC1918/private ranges;
- link-local;
- multicast/broadcast;
- cloud metadata ranges;
- service-mesh/control/data networks;
- Unix/file/data/javascript schemes;
- DNS names resolving to denied ranges;
- redirects from an allowed public host to a denied target.

## 6.3 Budgets

Enforce per operation/principal:

- maximum requests;
- maximum redirects;
- DNS resolution count;
- total inbound/outbound bytes;
- response-body bytes;
- subresource count;
- connection count;
- concurrency;
- wall time;
- destination set;
- content types.

## 6.4 Evidence

Every egress result includes:

- operation digest;
- destination policy revision;
- requested and final canonical URL;
- resolved/connected IP;
- TLS identity;
- redirect chain;
- retrieval time;
- response status/content type/size;
- response body/object digest;
- truncation and freshness state;
- source trust classification.

---

# 7. Parser and upload isolation

## 7.1 Upload broker

Before a parser sees bytes:

- authenticate principal and capability;
- stream upload with hard byte limit;
- compute digest while streaming;
- reject excess before full materialisation;
- validate declared filename separately from content;
- inspect magic bytes/container type;
- store in a quarantined immutable object;
- assign a parser job ID and data classification.

## 7.2 Archive preflight

Apply one archive policy to ZIP, DOCX, XLSX, PPTX and any ZIP-derived container.

Required checks before library parsing:

- central-directory byte/cardinality limit;
- member count;
- duplicate names;
- canonical member paths;
- no traversal or symlink/device entries;
- per-member compressed/uncompressed limit;
- aggregate expanded-size limit;
- compression-ratio limit;
- nested archive depth;
- XML size/entity/depth constraints;
- directory entries excluded before file-count allowance;
- reject unsupported encrypted/unknown containers explicitly.

## 7.3 Parser worker

Each job uses a disposable worker under the same controls as execution workers, with no external egress and no service-network access.

Converters receive:

- minimal clean environment;
- immutable input file;
- bounded output directory;
- process-tree control;
- CPU/memory/disk/time limits;
- no inherited credentials;
- no access to other uploads.

## 7.4 Versioned result

Every parse result must include:

```text
schema_version
job_id
principal_id
source_object_id
source_digest
detected_format
declared_format
parser/converter identities and digests
started_at/completed_at
completeness classes processed/omitted
warnings/recovery diagnostics
truncation flags and counts
output digest
untrusted_content = true
data_class
retention/deletion lineage
```

Invalid JSON, encrypted PDF, unsupported format, incomplete extraction and parser recovery must not use an ordinary successful “parsed” state.

---

# 8. Evidence and provenance model

## 8.1 Evidence classes

At minimum:

- `EXTERNAL_OBSERVATION`
- `USER_ASSERTION`
- `DOCUMENT_EXTRACT`
- `SENSOR_OBSERVATION`
- `MODEL_INFERENCE`
- `SYSTEM_REFLECTION`
- `OPERATOR_APPROVAL`
- `INDEPENDENT_OUTCOME`

## 8.2 Immutable evidence record

Required fields:

```text
evidence_id
evidence_version
principal_id
tenant_id
purpose
data_class
evidence_class
source_identity
source_object/content digest
source event time
system receipt time
transformation chain
parser/model/tool versions
trust state
verification state
freshness/expiry
independence group
supersedes/contradicts references
retention class
signature/audit reference
```

## 8.3 Authority rules

- User, web, document, sensor and model content begins unverified.
- Generated content cannot certify itself or another generated derivative from the same independence group.
- Caller-provided confidence/importance does not become evidence strength.
- Operator approval is narrow and does not validate factual content unless that is explicitly the reviewed decision.
- Missing provenance blocks promotion to trusted memory/evidence.
- A parser warning/degraded extraction propagates to every derivative.

---

# 9. Prompt and untrusted-content boundary

## 9.1 Context assembler

Create a dedicated context assembler rather than directly interpolating raw records into privileged prompts.

Responsibilities:

- authorise principal/purpose access;
- select active record versions only;
- preserve evidence class and source labels;
- quote/encode untrusted content as data;
- separate system policy from retrieved content;
- enforce byte/token budgets;
- remove hidden control/bidirectional characters where display-safe rendering requires it;
- block retrieved instructions from becoming system/developer directives;
- expose contradiction, freshness, truncation and verification state.

## 9.2 Prompt injection tests

Required sources:

- browser pages;
- email/news/calendar;
- PDF/DOCX/XLSX/PPTX/CAD;
- OCR/screen/clipboard;
- audio transcripts;
- vault notes;
- memories/preferences/feedback;
- model-generated summaries and reflections.

Pass condition: malicious instructions remain quoted untrusted evidence and cannot alter identity, policy, Tool Gate mode, approval, capability, memory authority or side-effect selection.

---

# 10. Principal-partitioned data model

## 10.1 Mandatory partition key

Every primary and derivative record includes:

```text
tenant_id
principal_id
purpose_id
data_class
retention_class
```

Where shared data is required, create an explicit shared-resource ACL; do not use an empty user ID or global `keeper` fallback.

## 10.2 Enforcement layers

Apply at:

- API authorisation;
- database row-level security or equivalent;
- Redis key namespace and ACL;
- vector collection/index partition;
- graph dataset/node partition;
- object/file path/object-store policy;
- cache key;
- browser context;
- audit search;
- backup/restore selection.

## 10.3 Session model

- Server-generated session IDs.
- Principal/tenant binding.
- No caller-added `system` role.
- Authorised service-origin messages carry verified workload/provenance.
- Redis failure does not create a permissive process-local session authority.
- Session deletion and expiry produce derivative cleanup events.

---

# 11. Transactional memory/vector/graph architecture

## 11.1 Authoritative source record

Postgres or another transactional store becomes the authority for:

- source record/version;
- active/superseded/quarantined/deleted state;
- principal/purpose partition;
- evidence/provenance;
- transformation jobs;
- vector/graph derivative IDs;
- deletion and reconciliation state.

A local mapping file or process dictionary cannot be the only lineage authority.

## 11.2 Durable operation state machine

Suggested states:

```text
RECEIVED
VALIDATED
SOURCE_COMMITTED
VECTOR_PENDING
VECTOR_COMMITTED
GRAPH_PENDING
GRAPH_COMMITTED
VERIFIED_ACTIVE
FAILED_COMPENSATION_PENDING
QUARANTINED
SUPERSEDED
DELETION_PENDING
DELETION_VERIFIED
```

## 11.3 Outbox/inbox processing

- Source transaction writes record plus outbox event atomically.
- One owned worker consumes each event idempotently.
- Vector and graph services use inbox deduplication by operation/content/version digest.
- Derivative identifiers/generations are written back transactionally.
- Retry is safe and convergent.
- Poison/dead-letter state is visible and does not look successful.

## 11.4 Single writers and generations

- One writer authority per mutable vector index/graph dataset.
- Readers consume immutable generation/snapshot IDs.
- Queries report generation.
- Rebuild publishes a new generation atomically.
- Old generation is retained only under bounded rollback policy.
- memU Core and Introspection must not write the same TurboVec file independently.

## 11.5 Graph rules

- Dataset partition by tenant/principal/purpose.
- Stable source record/version key.
- Store every returned backend object ID.
- No caller-controlled arbitrary node-set cross-linking.
- Staged add/cognify with compensation.
- No query during uncommitted generation unless explicit snapshot isolation exists.
- Forget verifies backend deletion and query absence before lineage removal.
- Reconciliation detects orphan graph records and stale mappings.

---

# 12. Supersession, contradiction and deletion

## 12.1 Record lifecycle

Use explicit states:

- `ACTIVE_UNVERIFIED`
- `ACTIVE_VERIFIED`
- `QUARANTINED`
- `CONTRADICTED`
- `SUPERSEDED`
- `DELETION_PENDING`
- `DELETED`

Retrieval and prompt assembly select only policy-allowed active states.

## 12.2 Supersession transaction

Atomically:

1. Validate new source/evidence.
2. Create new version.
3. Mark old version superseded/non-active.
4. Queue derivative replacement/removal.
5. Publish new active pointer only after required derivatives verify.
6. Record contradiction/supersession links.

A `force=true` request cannot leave both versions authoritative.

## 12.3 Deletion workflow

Deletion must enumerate and verify:

- primary source record;
- state/session records;
- object/file/vault copy;
- vector entries/index generations;
- graph nodes/edges;
- summaries/compressions/reflections;
- prompt-context caches;
- search/cache/Redis entries;
- exports and notification payload where policy applies;
- archives and backup expiry/cryptographic-erasure state.

## 12.4 Deletion evidence

Return a deletion receipt only after the authoritative deletion state includes:

- request principal/authority;
- source record/version;
- derivative inventory;
- completion/exception per store;
- verification query/digest;
- backup handling;
- completion time and audit reference.

A downstream failure must not cause local lineage/mapping deletion.

---

# 13. Ordered implementation PRs

## P2-PR-01 — Isolation and data-integrity contracts

Deliverables:

- Execution operation-registry schema.
- Worker sandbox profile.
- Browser job/context schema.
- Egress destination-policy schema.
- Upload/parser job and result schemas.
- Evidence/provenance schema.
- Data partition and retention schema.
- Memory mutation/deletion state machine.
- Threat models and adversarial test matrix.

Acceptance:

- One reviewed contract is adopted across service owners.
- Phase 1 operation/capability fields are reused rather than duplicated.

---

## P2-PR-02 — Disable generic Executor operations

Primary files:

- `executor/app.py`
- Executor tests and Dockerfile.

Required changes:

- Remove shell/Python expression/Make/Git/Pip/Curl/Docker generic operations.
- Remove caller-selected script and unrestricted args.
- Introduce fixed operation registry and immutable implementation digest.
- Reject unknown operation versions.

Acceptance:

- Existing arbitrary-code test corpus is rejected.
- No interpreter/package/build/version-control/network client is reachable through an ordinary operation schema.

---

## P2-PR-03 — Disposable execution worker platform

Required changes:

- Move execution out of the broker container.
- Implement immutable one-job workers with resource/network/filesystem controls.
- Process-tree termination and cleanup reconciliation.
- Bounded streaming output.
- No inherited secrets.

Acceptance:

- Escape, descendant, fork, timeout, memory, disk, file and network tests pass.
- Worker destruction removes all job state outside declared outputs/audit.

---

## P2-PR-04 — Executor postcondition and outcome authority

Required changes:

- Explicit expected postcondition per operation.
- Distinguish `SUCCESS_VERIFIED`, `FAILED_BEFORE_EFFECT`, `OUTCOME_UNKNOWN`, `PARTIAL_EFFECT` and `CANCELLED_CLEAN`.
- Bind result and output object digests to Phase 1 operation/capability.

Acceptance:

- Timeout/error after possible effect never returns a clean failure claim.
- Reconciliation can determine and record terminal state.

---

## P2-PR-05 — Hardened egress proxy

Required changes:

- Deploy one policy-enforcing proxy.
- Default-deny direct application/worker egress.
- Implement DNS/IP/redirect/TLS/destination validation and budgets.
- Record complete egress evidence.

Acceptance:

- SSRF, DNS rebinding, redirect-to-private, metadata, file/data/javascript scheme and oversized-response tests fail safely.

---

## P2-PR-06 — Browser context isolation

Primary files:

- Browser Agent application/Dockerfile/tests.

Required changes:

- Remove singleton shared context/page.
- One principal/workflow context lease.
- Pinned browser artefact and verified sandbox.
- Permission/download/popup/background-worker policy.
- Cleanup and deletion receipt.

Acceptance:

- Caller B cannot access Caller A cookies, page, storage, downloads or authenticated state.
- Context closes and storage disappears at terminal state.

---

## P2-PR-07 — Exact browser actions and postconditions

Required changes:

- Typed action schemas with page revision and unique target fingerprint.
- No fuzzy first-match consequential click.
- Idempotency and capability linkage.
- Verified before/after page evidence.

Acceptance:

- Wrong/ambiguous target refuses action.
- Retry cannot duplicate submission.
- Success requires expected observed result.

---

## P2-PR-08 — Monitor isolated-rule execution

Primary files:

- Monitor service.
- Browser scrape interface.

Required changes:

- Immutable rule revisions.
- Isolated browser job per scrape rule/check.
- Apply exact URL and selector.
- Fence/cancel old/deleted/disabled rule revisions.
- Separate source observation from notify/TTS action capability.

Acceptance:

- Rule cannot scrape another workflow’s page.
- Deleted/disabled revision cannot deliver after fence.
- Alert proves exact source/page/selector/value digest.

---

## P2-PR-09 — Upload quarantine and format detection

Primary files:

- Files Service.
- Dashboard upload path.
- Document Parser ingress.

Required changes:

- Streamed bounded upload.
- Immutable quarantined object and digest.
- Magic-byte/container validation.
- Principal/data-class/retention assignment.
- No filename-extension authority.

Acceptance:

- Oversized/malformed upload stops before full materialisation.
- Filename mismatch and unsupported binary receive typed rejection.

---

## P2-PR-10 — Archive/OOXML preflight

Required changes:

- Apply shared archive policy to ZIP/DOCX/XLSX/PPTX.
- Duplicate/member/path/ratio/expanded-size/nesting/XML limits.
- Explicit encrypted/unsupported states.

Acceptance:

- Archive/OOXML bombs and duplicate-entry attacks fail before complex parser libraries run.

---

## P2-PR-11 — Disposable parser/converter workers

Required changes:

- Isolate PDF/Office/CAD/OCR/converter jobs.
- No egress/service network.
- Minimal environment and explicit inputs/outputs.
- Resource/process/disk controls.
- Pinned parser/converter artefacts.

Acceptance:

- Converter compromise cannot reach internal services or unrelated files.
- Descendants and temp files are removed after timeout/cancel.

---

## P2-PR-12 — Provenance-rich parser results

Required changes:

- Versioned result schema.
- Source/output digest, parser versions, completeness, warnings, truncation, timestamps and untrusted classification.
- Privacy no-store controls.

Acceptance:

- Invalid/protected/incomplete documents cannot appear as ordinary complete success.
- Downstream context retains untrusted/degraded labels.

---

## P2-PR-13 — Vault secure object/path model

Primary files:

- Vault Sync application, parser, mapper and watcher.

Required changes:

- Accept vault-relative object IDs only.
- Descriptor-based no-follow traversal beneath pre-opened root.
- Reject symlinks and path aliases.
- Atomic revision-checked writes.
- Bounded durable queues.
- Initial and periodic reconciliation.

Acceptance:

- Manual/watcher symlink/path escape fails.
- Concurrent path replacement cannot redirect read/write.
- Existing files, graph and mapping reconcile before readiness.

---

## P2-PR-14 — Principal-partitioned memory/session store

Primary files:

- `memu-core/app.py`
- Postgres/Redis/vector schemas.

Required changes:

- Derive principal/tenant/purpose from Phase 1 context.
- Row-level and key/index partition enforcement.
- Remove global/empty `keeper` fallback.
- Server-owned sessions and roles.
- Reject unauthenticated pin/preference/system-message authority.

Acceptance:

- Cross-principal read/write/route/session tests fail.
- Empty user or caller-provided `keeper` grants no authority.

---

## P2-PR-15 — Evidence/provenance and promotion policy

Required changes:

- Implement immutable evidence records/classes.
- Strict PASS-only promotion where policy requires verification.
- Non-PASS/unavailable states quarantine or reject.
- Generated/model content cannot self-certify.
- Caller relevance/importance/conviction becomes untrusted metadata only.

Acceptance:

- Poisoned external/user/model records cannot become operator preference, verified evidence or high-authority memory without required independent promotion.

---

## P2-PR-16 — Safe context assembler

Required changes:

- Principal/purpose-authorised retrieval.
- Active versions only.
- Structured untrusted-content quoting.
- System policy physically/logically separate from retrieved content.
- Contradiction/freshness/truncation labels.
- Strict token/byte budgets.

Acceptance:

- Web/document/email/sensor/memory prompt-injection corpus cannot change system policy or side-effect authority.

---

## P2-PR-17 — Durable memory/vector/graph outbox

Required changes:

- Authoritative source/version transaction.
- Durable outbox/inbox.
- Idempotent derivative jobs by source/version/content digest.
- Single writer per mutable index/dataset.
- Verified generation publication.
- Visible compensation/dead-letter states.

Acceptance:

- Failure at every step is recoverable and converges after retry.
- No hidden partial source/vector/graph success.

---

## P2-PR-18 — Graph partitioning and lineage

Required changes:

- Dataset partition by principal/purpose.
- Stable source version and every backend object ID.
- Staged add/cognify.
- Generation-aware query.
- Verified forget and reconciliation.

Acceptance:

- Same-source re-ingest cannot orphan earlier data.
- Add/cognify failure retains compensatable lineage.
- Query never merges unrelated principal/purpose datasets.

---

## P2-PR-19 — Supersession and contradiction state

Required changes:

- Explicit active/quarantined/contradicted/superseded lifecycle.
- Atomic active-version pointer.
- Derivative replacement/removal.
- Retrieval excludes non-active records by construction.

Acceptance:

- Old contradictory version cannot remain active after successful supersession.
- History remains attributable but cannot influence prompts/trust unless policy explicitly permits it.

---

## P2-PR-20 — End-to-end derivative deletion

Required changes:

- Lineage-driven deletion state machine.
- Verify primary, vector, graph, cache, session, prompt-context, vault/file and backup-policy handling.
- Durable deletion receipt.
- Reconciliation and retry.

Acceptance:

- Delete failure in one derivative never discards lineage.
- End-to-end test proves removal/non-accessibility and prevents resurrection on restart/restore/re-ingest.

---

## P2-PR-21 — Phase 2 CI and runtime assurance

Required gates:

- Fixed-operation registry drift checks.
- Sandbox escape/resource/process/network tests.
- Real-browser cross-context, cookie, popup, download, crash and cleanup tests.
- SSRF/DNS-rebinding/redirect test suite.
- Archive/OOXML bomb and converter containment tests.
- Principal partition/RLS/vector/graph isolation tests.
- Persistent prompt-injection corpus.
- Fault injection at every memory/vector/graph step.
- Supersession/deletion/restore tests.
- Built image/parser/browser/converter digests linked to signed assurance artefact.

Release remains NO_GO if tests use mocks/stubs where the claimed boundary is a real browser, process, network, parser or database boundary.

---

# 14. Adversarial closure tests

## P2-A — Executor arbitrary-code corpus

Attempt Python, Find, Make, Pip, Git, Curl, Docker, basename replacement, script-argument and expression escapes.

**Pass:** no generic path exists; only reviewed fixed operation schemas execute.

## P2-B — Sandbox breakout and persistence

Attempt filesystem escape, secret access, internal/external network, process descendants, fork/resource exhaustion and persistent modification.

**Pass:** operation remains inside disposable worker; no survivor or undeclared output.

## P2-C — Browser cross-principal state

Authenticate Principal A to a test site, complete workflow, then run Principal B and Monitor operations.

**Pass:** B/Monitor cannot observe or act with A cookies, storage, page, downloads or session.

## P2-D — Browser duplicate/ambiguous action

Retry a form submission and present multiple fuzzy text matches.

**Pass:** duplicate is consumed once; ambiguous target blocks before action.

## P2-E — SSRF and redirect matrix

Test IP literals, alternative encodings, localhost names, private/link-local/metadata, DNS rebinding, redirect chains, proxy tunnels and unsafe schemes.

**Pass:** no prohibited connection occurs.

## P2-F — Archive/parser hostile corpus

Test ZIP/OOXML bombs, duplicates, traversal, symlink entries, nested archives, malformed XML/PDF/Office/CAD, huge JSON/CSV and converter child processes.

**Pass:** bounded typed failure in disposable worker; no internal network or unrelated file access.

## P2-G — Vault path and watcher escape

Test absolute paths, `..`, symlinks, aliases and path replacement race.

**Pass:** no read/write outside pre-opened vault root; one canonical object identity.

## P2-H — Persistent prompt poisoning

Inject instructions through web, document, email, OCR, clipboard, screen, audio, memory, preference and model reflection.

**Pass:** content remains untrusted evidence and cannot alter policy, identity, approval, capability or authoritative preference.

## P2-I — Principal partition attack

Attempt cross-user memory, session, vector, graph, vault, cache and restore access using IDs from another principal.

**Pass:** denied at API and storage layer; no existence oracle beyond policy.

## P2-J — Multi-store fault injection

Fail after source commit, vector write, graph add, cognify, mapping write and active-pointer publication.

**Pass:** durable visible state converges or compensates; no orphan without lineage.

## P2-K — Supersession conflict

Replace a record while concurrent retrieval/graph query occurs.

**Pass:** readers see one defined generation; old version becomes non-authoritative atomically.

## P2-L — End-to-end deletion and restore

Delete a record with vector, graph, summary, prompt cache, vault and backup lineage; restart and restore from backup.

**Pass:** deleted data remains inaccessible and is not resurrected outside documented retention/cryptographic-erasure policy.

---

# 15. Phase 2 exit criteria

Phase 2 is complete only when:

- Generic execution primitives are removed from ordinary authority.
- All hostile execution uses disposable workers with enforced resource, filesystem and network limits.
- Browser contexts are principal/workflow isolated and deleted at terminal state.
- Browser actions are exact, idempotent and postcondition verified.
- Monitor cannot reuse or scrape another browser workflow.
- All external connections use the hardened egress authority.
- SSRF, DNS rebinding and redirect attacks fail.
- Uploads are streamed, bounded, content-detected and quarantined.
- ZIP/OOXML and parser/converter jobs are preflighted and isolated.
- Parser results carry immutable provenance, completeness and untrusted-content state.
- Vault reads/writes are securely constrained beneath one canonical root/object model.
- All stores enforce principal/tenant/purpose/data-class partitioning.
- No global/empty `keeper` or caller-selected system-role authority remains.
- Every memory/evidence record has authenticated provenance and lifecycle state.
- Prompt assembly preserves the untrusted-content boundary.
- Memory/vector/graph updates use durable idempotent state and single-writer generations.
- Supersession removes old versions from active authority atomically.
- Derivative deletion is lineage-driven and verified end to end.
- Phase 2 adversarial and failure-injection tests pass against production-equivalent artefacts.

Passing Phase 2 permits bounded security testing with controlled test data after Phase 0 and Phase 1 remain effective. It does not authorise production, live financial action, unrestricted sensitive-data processing or autonomous operation. Phase 3 reliability/audit/privacy/recovery and Phase 4 requalification remain required.

---

## Final Phase 2 planning judgement

Kai’s execution, browser, parser and memory risks are mutually reinforcing. A compromised parser or browser can reach the flat service network; extracted hostile content can enter privileged prompts; caller-controlled memory can become persistent authority; and partial graph/vector operations can leave undeletable derivatives.

The correct boundary is not another denylist or text sanitiser. It is a system of disposable workers, controlled egress, principal-partitioned storage, immutable provenance and transactional derivative lineage, all tied to the Phase 1 operation/capability model.

**Current release decision remains NO_GO. This document implements no runtime remediation and closes no findings.**
