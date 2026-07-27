# Kai System — Full Code, Security, Trust and Architecture Audit Report

Repository: `dainius1234/kai-system`  
Audited snapshot: default branch through findings commit `2d830f25d569baa5ce955dd8d17e8f0744239876`  
Report completed: 27 July 2026  
Audit status: **SOURCE, DEPLOYMENT AND SYSTEM CONSOLIDATION COMPLETE FOR THE REVIEWED SNAPSHOT**  
Remediation status: **NO RUNTIME, INFRASTRUCTURE OR CONFIGURATION REMEDIATION PERFORMED**  
Overall release decision: **NO_GO**

This report is the complete explanatory and action-planning companion to:

- `kai-pm/CODE_AUDIT_MASTER.md` — authoritative numerical register.
- `kai-pm/CODE_AUDIT_REMEDIATION_BACKLOG.md` — prioritised remediation programme.
- `kai-pm/CODE_AUDIT_IMPLEMENTATION_SEQUENCE_AND_CLOSURE_MATRIX.md` — programme dependencies, attack-chain closure and release control.
- `kai-pm/CODE_AUDIT_P0_CONTAINMENT_PLAN.md`.
- `kai-pm/CODE_AUDIT_P1_SECURITY_FOUNDATION_PLAN.md`.
- `kai-pm/CODE_AUDIT_P2_ISOLATION_AND_INTEGRITY_PLAN.md`.
- `kai-pm/CODE_AUDIT_P3_RELIABILITY_AUDIT_PRIVACY_RECOVERY_PLAN.md`.
- `kai-pm/CODE_AUDIT_P4_CAPABILITY_REQUALIFICATION_PLAN.md`.
- `kai-pm/CODE_AUDIT_BATCH_*.md` — detailed source-confirmed evidence.

Historical working totals and batch-local provisional totals are retained for chronology only. The final numerical authority is `CODE_AUDIT_MASTER.md`.

---

# 1. Executive summary

## 1.1 Final conclusion

The reviewed Kai System is **not safe for production deployment, Internet exposure, shared-LAN exposure, autonomous execution, financial decision-making, operational recovery authority, external messaging, or storage of sensitive personal, credential, financial, biometric or operational data**.

The dominant risk is architectural. The system contains many individually serious defects, but the decisive problem is that insecure components reinforce one another:

- privileged services are reachable without trustworthy identity;
- policy decisions are not enforced at every final side-effect boundary;
- Dashboard can borrow internal privilege for anonymous callers;
- Executor, browser, parser and egress services provide compromise pivots;
- memory, evidence, confidence, trust and operator-state inputs are poisonable;
- verification and consensus can be manufactured from weak, duplicated or caller-controlled evidence;
- failures, stubs and degraded states frequently look successful;
- critical state is process-local, file-backed, unsigned, race-prone or non-transactional;
- audit, recovery and backup evidence cannot reliably prove what happened or restore a known-good state;
- CI and release tooling can certify shallow reachability, mocks, known development credentials or production-shaped stubs.

A reachable attacker, malicious browser payload, poisoned document, compromised internal service or unauthorised local caller can plausibly progress from reconnaissance or content injection to persistent memory poisoning, false identity/authority, policy-mode changes, sensitive-data extraction, external action, destructive recovery or arbitrary code execution. Several credible paths require no credentials.

## 1.2 Final confirmed findings

| Severity | Count | Approximate share |
|---|---:|---:|
| Critical | **252** | **5.5%** |
| High | **2,440** | **53.3%** |
| Medium | **1,885** | **41.2%** |
| Low | **3** | **0.1%** |
| **Total** | **4,580** | **100%** |

Arithmetic:

`252 + 2,440 + 1,885 + 3 = 4,580`

The earlier **2,529** total is an intermediate baseline through Wake Service commit `3112c21f8258d5749e632b7cbf45d12b970b0eaf`. It is not the final repository total.

## 1.3 Final release decision

# **NO_GO**

The reviewed snapshot must remain an **isolated disposable development laboratory**. No report, plan, code comment, green unit test, screenshot, operator statement or planning commit changes that decision.

No finding is fixed by this report. No finding count may be reduced until implementation, adversarial verification, downstream integration testing and independent closure review are complete.

## 1.4 Immediate management meaning

The system must not be treated as “mostly working with a long bug list.” It must be treated as an **unqualified high-consequence platform whose trust foundations are not yet established**.

The safe programme is not to patch 4,580 items in arbitrary order. The safe programme is to remove the shared root causes in strict dependency order:

`Evidence preservation and containment → identity and final enforcement → isolation and data integrity → distributed reliability, audit, privacy and recovery → model, trust and autonomy requalification`

---

# 2. Audit purpose, scope and methodology

## 2.1 Purpose

The audit assessed whether the repository’s source code, deployment configuration and cross-service architecture support safe operation of the capabilities they claim to provide.

The review did not accept service names, comments, health labels, test names or documentation claims as proof. Controls were evaluated at the point where they must actually prevent, authorise, record or recover a consequential action.

## 2.2 Methodology

The audit methodology was:

1. Review source files and deployed services, not documentation alone.
2. Record only source- or configuration-confirmed findings.
3. Cross-check service code against Compose, Dockerfiles, environment handling, networks, ports, volumes, health checks, startup order and service interactions.
4. Avoid duplicating an already captured finding unless a new source path, affected component or independent impact justified a separate record.
5. Analyse attack chains across service boundaries rather than treating files in isolation.
6. Preserve finding-level evidence in committed `CODE_AUDIT_BATCH_*.md` files.
7. Reconcile cumulative totals from one coherent baseline and every later findings-bearing delta exactly once.
8. Separate planning from remediation and closure.

## 2.3 Material scope reviewed

Material scope included:

- identified FastAPI services and host-published APIs;
- Dashboard backend, browser client, SSE streams and privileged proxy behaviour;
- Tool Gate, Executor, Verifier, Fusion, Trust Core and Trust Ledger;
- Agentic planning, conviction, model routing, adversary, forecasting and cognitive modules;
- memU Core, introspection, graph, compression, vault, sessions, operator memory and P17–P22 personality/autonomy systems;
- Browser Agent, Web Scout, Monitor and network-egress paths;
- file, document, OCR, clipboard, camera, audio, vision, screen, wake and sensor services;
- Financial Awareness, Broker Bridge, market intelligence, calendar, weather, news, email and advisory tools;
- Supervisor, Heartbeat, Metrics, backup, archival, ledger and recovery workers;
- Dockerfiles, Compose profiles, host ports, volumes, secrets, networks, health checks and startup ordering;
- CI workflows, test bootstrap, fake/stub paths, release checks, smoke tests, chaos drills, rotation scripts and host-hardening tooling;
- cross-service attack chains, orchestration behaviour and architecture-level trust invariants.

## 2.4 Limits

This was a source, deployment and architecture review. It does not claim:

- live exploitation of every path;
- runtime penetration testing of every provider, account, device or external network;
- hardware-specific behaviour on unavailable devices;
- enumeration of every third-party dependency CVE;
- equal exploitability in every possible deployment profile.

These limits do not change the **NO_GO** decision because multiple independent critical paths are directly present in source and deployment topology.

## 2.5 Confidence

Confidence is high for findings directly supported by committed source and configuration. The report distinguishes confirmed conditions from proposed future architecture. No proposed control is represented as implemented.

---

# 3. System-level risk model

## 3.1 What the architecture attempts to do

Kai is a distributed, service-oriented AI and automation platform combining:

- user/operator interaction;
- planning and model selection;
- memory and personality state;
- policy decisions;
- execution and browser operations;
- sensor and document ingestion;
- financial and external communication functions;
- health supervision, recovery, backup and audit.

This breadth creates a high-consequence system. A weakness in one service can become authority in another because data, identity, policy, memory, models and actions are linked.

## 3.2 Why service separation is not currently a security boundary

The repository uses multiple services and containers, but separation alone does not provide isolation when:

- privileged services publish host ports;
- services share a flat network;
- static addresses or shared secrets are treated as identity;
- internal APIs accept anonymous calls;
- a gateway proxies anonymous requests using reusable internal credentials;
- execution and egress services can reach control/data services;
- shared volumes permit multiple writers;
- browser state is shared across users or workflows;
- file and model outputs are trusted without provenance.

The result is distributed complexity without a dependable trust plane.

## 3.3 Primary assets at risk

The reviewed architecture may handle or influence:

- operator identity and preferences;
- credentials and service secrets;
- private communications and email;
- financial information and proposed transactions;
- browser sessions and authenticated pages;
- documents, screenshots, camera, audio and clipboard content;
- memories, trust scores, personality and autonomy state;
- source code, host files and internal service data;
- policy decisions, recovery actions and audit evidence.

Compromise therefore affects confidentiality, integrity, availability, accountability and physical/financial decision safety.

---

# 4. Dominant root causes

## 4.1 No authoritative identity plane

### Confirmed pattern

Many privileged services accept anonymous network callers. Where authentication-like controls exist, they frequently rely on shared HMAC material, body-supplied identifiers, reusable gateway credentials or development fallbacks.

Credentials and identity assertions are often not:

- unique to a human, workload or service;
- cryptographically bound to the complete operation;
- audience restricted;
- short lived and single use;
- revoked and rotated consistently;
- separated between runtime, approval and administration.

### Why it matters

A service cannot enforce authorisation or produce meaningful audit evidence unless it can prove who or what made the request, under whose authority and for what exact scope.

### Likely impact

- anonymous privileged access;
- service impersonation;
- privilege borrowing through Dashboard or another gateway;
- false audit attribution;
- inability to distinguish user intent from service automation;
- replay and confused-deputy attacks.

### Corrective principle

Establish verified human principal identity, workload identity, explicit delegation and scoped short-lived credentials before re-enabling privileged services.

### Programme priority

**P1 foundation**, after P0 containment.

---

## 4.2 Decision and enforcement are separated

### Confirmed pattern

Tool Gate can make policy decisions, but numerous final action services do not require a valid, exact, one-time Gate capability before executing a side effect.

Affected classes include:

- Executor and subprocess operations;
- browser and web actions;
- memory, preference and identity mutation;
- file and vault actions;
- notifications and external messaging;
- recovery and breaker reset;
- monitoring actions;
- financial mutations;
- sensor-triggered actions.

### Why it matters

A denied decision is meaningless if a caller can invoke the final action endpoint directly.

### Likely impact

- complete policy bypass;
- execution without operator approval;
- inconsistent enforcement between routes;
- replay of previously approved actions with changed parameters;
- inability to prove that the executed operation matched the approved operation.

### Corrective principle

Define one canonical operation envelope and digest. Issue a short-lived, audience-bound, single-use capability for that exact operation. Require atomic capability consumption at the service performing the side effect.

### Programme priority

**P1 critical dependency**.

---

## 4.3 Dashboard is a privileged confused deputy

### Confirmed pattern

Dashboard aggregates many internal services onto one host-published origin and can use internal authority for requests from callers that have not independently proved identity or scope.

The reviewed paths can expose or mutate:

- Tool Gate mode;
- Agentic identity and registry files;
- memory and operator models;
- finance and email data;
- logs and Redis streams;
- browser, monitor, file and notification functions;
- self-improvement and administrative operations.

The browser client also contains unsafe rendering paths and a deterministic JavaScript parse failure.

### Why it matters

Dashboard converts an external or local browser compromise into fleet-level internal authority. Stored XSS is particularly dangerous because same-origin script can call privileged proxy routes.

### Likely impact

- control-plane compromise;
- sensitive-data exposure;
- policy changes;
- arbitrary internal service invocation;
- persistent browser-based attack.

### Corrective principle

Reduce Dashboard to an authenticated, least-privilege client. Remove reusable server-held operator/admin credentials. Separate data display from administrative mutation. Require step-up approval and exact capabilities for consequential actions. Apply safe rendering and strict CSP.

### Programme priority

**P0 containment and P1 rebuild**.

---

## 4.4 Executor is not a security sandbox

### Confirmed pattern

Executor accepts direct requests and exposes command routes capable of arbitrary or near-arbitrary behaviour through Python, Make, Git, Pip, Curl, Find and related tooling. Timeout and rollback logic does not reliably terminate descendants or reverse external effects.

### Why it matters

An allowlist of powerful interpreters and package/build/network tools is not a sandbox. Once invoked, those tools can read files, execute code, alter dependencies, access networks or persist changes.

### Likely impact

- arbitrary code execution;
- host/container compromise;
- internal-service reconnaissance;
- credential theft;
- persistence;
- supply-chain modification;
- destructive or irreversible side effects.

### Corrective principle

Remove generic command authority from normal operations. Use fixed-schema operations executed in disposable, resource-limited workers with read-only roots, restricted mounts, controlled egress, descendant containment and independently verified postconditions.

### Programme priority

**P0 disable; P1 capability enforcement; P2 sandbox replacement**.

---

## 4.5 Memory, evidence and operator state are poisonable

### Confirmed pattern

Unauthenticated or weakly attributed inputs can influence:

- memories and pinned preferences;
- corrections, feedback and episode outcomes;
- operator values, loyalty and conscience state;
- historical and future-self records;
- model confidence and trust signals;
- world/calendar context;
- Verifier evidence packs.

Retrieval can mutate rank, access count or stability, allowing repeated reads to strengthen selected records. Generated assessments can be stored and reused as evidence.

### Why it matters

Persistent context becomes a control input. Poisoned records can survive the original request and influence later planning, verification, trust and autonomy.

### Likely impact

- persistent prompt injection;
- cross-user data leakage;
- operator impersonation;
- false confidence and trust inflation;
- autonomous action based on attacker-authored context;
- inability to delete all derivatives.

### Corrective principle

Partition all records by verified principal, tenant, purpose and data class. Store immutable provenance and lineage. Treat external/model/document/sensor text as untrusted data. Require explicit promotion rules before it can become memory, evidence or system context.

### Programme priority

**P2 data-integrity foundation**, dependent on P1 identity.

---

## 4.6 Verification and consensus can be manufactured

### Confirmed pattern

Verifier accepts caller-supplied evidence and relies on rank, overlap and formatting heuristics rather than proposition-level entailment, contradiction and source independence. Fusion can report consensus from one specialist, failed specialists, duplicates, correlated sources or deterministic stubs. Verifier rejection does not consistently block downstream output.

### Why it matters

A system that labels its own weak or duplicated inputs as independent verification creates false authority. Downstream automation may treat confidence formatting as proof.

### Likely impact

- false PASS results;
- manufactured consensus;
- inaccurate financial or operational recommendations;
- trust/autonomy inflation;
- self-certification loops;
- unsafe action under an apparently verified result.

### Corrective principle

Create immutable typed claims and evidence, authoritative source identity, independence groups, proposition-level entailment/contradiction, strict enforcement of verification failure and reproducible task-specific benchmarks.

### Programme priority

**P4**, only after P1–P3 foundations are verified.

---

## 4.7 Browser, egress and parser services are compromise pivots

### Confirmed pattern

Browser Agent, Web Scout, Monitor, Document Parser, OCR, Screen Capture, Vault Sync and Executor contain combinations of:

- arbitrary destinations or SSRF exposure;
- redirect and DNS-rebinding risk;
- shared authenticated browser state;
- unbounded response/archive processing;
- external parser/converter execution;
- unsafe files, paths or symlinks;
- prompt-injection propagation;
- broad internal-network access;
- missing destination-level egress policy.

### Why it matters

These services process hostile content and often possess network, file or authenticated-session access. A parser, browser or monitor compromise becomes an internal pivot.

### Likely impact

- internal metadata/service access;
- credential or session extraction;
- cross-user browsing leakage;
- host filesystem access;
- decompression/resource exhaustion;
- malicious content entering trusted prompts or memory.

### Corrective principle

Use one controlled egress authority, per-principal browser contexts, disposable parser/converter workers, upload quarantine, true format detection, archive preflight, strict resource limits and provenance-rich outputs.

### Programme priority

**P0 disable/segment; P2 rebuild**.

---

## 4.8 Failure and degraded states look successful

### Confirmed pattern

Examples include:

- HTTP 200 responses carrying error-shaped bodies;
- health reporting `ok` while required capabilities are absent;
- stubs represented as completed reasoning;
- missing dependencies represented as neutral evidence;
- recovery reported successful without verified postconditions;
- backup success without verifiable artefacts;
- release checks passing when dependencies are unavailable;
- CI using fake embeddings, known development secrets or mocked services while reporting green.

### Why it matters

Operators and dependent services cannot make safe decisions when success, failure, blocked, degraded, stubbed and unknown outcomes share similar contracts.

### Likely impact

- unsafe retries;
- duplicated actions;
- release of unready services;
- silent data loss;
- false recovery claims;
- false verification and autonomy signals.

### Corrective principle

Adopt one typed state/error/health vocabulary. Separate liveness, readiness and capability health. Make unknown outcome explicit. Fail closed in release tooling and prohibit production-shaped stubs.

### Programme priority

**P0 temporary containment; P3 systemic replacement**.

---

## 4.9 Security-critical state is mutable and process-local

### Confirmed pattern

Security, reliability and autonomy state is often:

- process-local and inconsistent across workers;
- stored in unsigned JSON/JSONL;
- rewritten non-atomically;
- shared through writable files or volumes;
- split across database, vector, graph and local cache;
- updated without durable operation state;
- restored from incomplete or unverified checkpoints.

### Why it matters

Multiple workers can disagree about identity, policy, health, trust, breaker state, ownership or operation outcome. A restart can erase control history or activate stale state.

### Likely impact

- double execution;
- race conditions and lost updates;
- stale leader writes;
- inconsistent policy decisions;
- corrupted indexes and ledgers;
- rollback to insecure state.

### Corrective principle

Move critical state to transactional shared stores. Use durable operation records, idempotency keys, outbox/inbox, leases, fencing tokens, explicit generations and single-writer ownership.

### Programme priority

**P2 for memory/index integrity; P3 for distributed operation**.

---

## 4.10 Audit, backup and recovery cannot prove correctness

### Confirmed pattern

Logs and ledgers often omit one or more of:

- authenticated actor and workload;
- exact operation digest;
- policy/configuration revision;
- before/after state revision;
- evidence identity and lineage;
- tool/model/backend digest;
- delivery/execution postcondition;
- durable signature or external integrity anchor.

Some ledgers retain credentials or signatures. Some acknowledge writes after persistence failure. Recovery may be generic, unauthenticated or triggered by shallow health. Backups may lack coherent manifests and isolated restore qualification.

### Why it matters

A high-consequence system must be able to prove what was requested, approved, executed, observed and restored. Without that chain, incident response and safe recovery are unreliable.

### Likely impact

- non-repudiation failure;
- undetected tampering;
- destructive or unauthorised recovery;
- incomplete restores;
- loss of legal/forensic evidence;
- false trust and performance scoring.

### Corrective principle

Build a signed append-only audit authority, externally anchored checkpoints, structured minimised logging, authorised service-specific recovery, coherent immutable backup manifests and isolated restore tests.

### Programme priority

**P0 freeze recovery; P3 rebuild**.

---

## 4.11 CI, assurance and release controls can create false green

### Confirmed pattern

Assurance paths may globally enable development secrets, fake embeddings, mocks, shared environments, shallow health checks or known stubs. Some go/no-go logic can pass despite missing or unavailable dependencies.

### Why it matters

A release gate that tests a different security profile from production can certify the very bypasses it is meant to detect.

### Likely impact

- deployment of known-unready services;
- credential and identity regressions;
- untested migration paths;
- false confidence in sandbox, backup, recovery or model capability;
- inability to reproduce the tested revision.

### Corrective principle

Use production-equivalent profiles, immutable artefact digests, negative security tests, policy-as-code, fail-closed dependency checks, signed tested-revision reports and explicit prohibition of stub/compatibility mode.

### Programme priority

**P0 deployment policy checks; integrated gates in P1–P4**.

---

# 5. Detailed assessment by domain

## 5.1 Authentication, authorisation and identity propagation

**Assessment: Critical failure.**

The architecture lacks one authoritative way to identify a human principal, workload and delegated authority. Body fields such as `user_id`, `actor_did`, `keeper`, `role` or `session_id` are data, not proof of identity. Shared secrets and gateway-held credentials allow one component to impersonate another or act for an unauthenticated caller.

Required outcome:

- authenticated human principal;
- unique workload identity;
- explicit narrow delegation;
- audience and scope restrictions;
- short expiry and revocation;
- separate runtime, operator approval and administrative credentials;
- end-to-end identity propagation bound to the operation digest.

## 5.2 Tool Gate and side-effect governance

**Assessment: Critical failure.**

Tool Gate cannot protect actions that bypass it. The system requires a complete machine-readable side-effect registry and final-boundary enforcement at every consequential route.

Required outcome:

- canonical operation schema;
- deterministic serialisation and digest;
- exact approval object;
- one-time capability;
- atomic consumption at the final service;
- no legacy HMAC/body-token/cosign path in protected profiles;
- immutable decision/execution/outcome linkage.

## 5.3 Dashboard and browser client

**Assessment: Critical failure.**

Dashboard combines broad data access, administrative mutation and internal service proxying on one origin. Unsafe rendering and server-held privilege make content injection a control-plane attack.

Required outcome:

- authenticated ingress;
- no reusable operator/admin token in Dashboard;
- data and administration separated;
- safe rendering without untrusted `innerHTML`;
- strict CSP and response hardening;
- step-up approval for administrative operations;
- capability enforcement at every proxied final service;
- no mode mutation during page load.

## 5.4 Executor and tool execution

**Assessment: Critical failure.**

Generic interpreters and powerful development/network tools defeat the intended allowlist. Cancellation does not prove that descendants or external effects stopped.

Required outcome:

- fixed-schema operation registry;
- no ordinary arbitrary shell/Python/Make/Git/Pip/Curl command authority;
- disposable worker per operation;
- restricted mounts and read-only root;
- process/namespace/resource containment;
- controlled egress;
- descendant kill and cleanup;
- independently verified postcondition and immutable outcome record.

## 5.5 Browser, Web Scout and Monitor

**Assessment: Critical failure.**

Shared browser state and weak destination control can expose authenticated pages or internal services to unrelated callers and monitoring rules.

Required outcome:

- isolated browser context per principal and workflow;
- exact URL/action fingerprints;
- destination allow policy after DNS resolution and redirects;
- private/link-local/metadata/internal ranges denied;
- postcondition verification;
- no Monitor access to an unrelated shared page;
- no untrusted page content promoted directly into authority context.

## 5.6 Files, Vault, documents, OCR and converters

**Assessment: High/Critical risk.**

Hostile files can exploit path, symlink, archive, format-detection, converter and resource-handling weaknesses. Parsed text can become prompt or memory authority without a strict trust boundary.

Required outcome:

- object-based storage rather than caller-controlled host paths;
- upload quarantine;
- content-based format detection;
- archive and OOXML preflight;
- recursion, size and file-count limits;
- disposable parser/converter workers;
- provenance-rich structured output;
- strict untrusted-content classification;
- lineage-driven derivative deletion.

## 5.7 Sensors, screen, clipboard, audio, camera and wake services

**Assessment: High/Critical privacy and injection risk.**

These services collect sensitive content and can feed untrusted text or events into prompts, memory or actions.

Required outcome:

- explicit principal, purpose and consent/basis;
- capability-scoped collection;
- minimised retention;
- sensor network isolation;
- no direct mutation of memory, trust, policy or execution;
- provenance and untrusted-content labels;
- visible collection state and reliable disable controls.

## 5.8 memU, graph, personality and autonomy state

**Assessment: Critical integrity failure.**

Personal and behavioural state is not consistently partitioned, provenance-bound or transactionally synchronised. Multiple writers and derivative stores can preserve poisoned or deleted records.

Required outcome:

- principal/tenant/purpose/data-class partitioning;
- one writer/generation authority per mutable index;
- durable memory/vector/graph state machine;
- immutable lineage;
- explicit supersession and contradiction;
- deletion across source and every derivative;
- no retrieval-driven trust inflation;
- existing records treated as untrusted until migrated or requalified.

## 5.9 Verifier, Fusion, model routing and confidence

**Assessment: Critical assurance failure.**

Verification, specialist independence, consensus and confidence are not sufficiently grounded in immutable external evidence. Style and heuristic scoring can increase apparent authority.

Required outcome:

- signed authoritative model/service/tool registry;
- exact backend and revision attestation;
- fresh readiness;
- reproducible benchmark authority;
- typed claims and immutable evidence;
- independence/correlation groups;
- proposition-level entailment and contradiction;
- strict Verifier enforcement;
- structured Fusion disagreement and abstention;
- calibrated task-specific uncertainty;
- no stub or fallback contribution to PASS, trust or GO.

## 5.10 Financial, broker and market actions

**Assessment: High/Critical risk.**

Weak signals, correlated sources, mutable state, uncertain execution outcomes and incomplete authority separation make financial mutation unsafe.

Required outcome:

- financial actions disabled until their separate domain gate;
- exact approved order/operation digest;
- final-boundary capability consumption;
- durable idempotent operation state;
- independent post-trade/position reconciliation;
- verified source freshness and independence;
- strict loss, exposure and budget limits;
- human step-up approval for high-consequence actions;
- complete audit and rollback/suspension control.

## 5.11 Supervisor, health, recovery and circuit breakers

**Assessment: High/Critical risk.**

Observation and mutation are mixed. Shallow or spoofed health can trigger generic recovery, and success may be assumed without postconditions.

Required outcome:

- standard liveness/readiness/capability health;
- Supervisor observation-only by default;
- incident authority separate from health observation;
- service-specific recovery operations;
- exact capability and policy approval;
- leases/fencing for recovery ownership;
- idempotent action and independent postcondition;
- recovery unable to weaken security or restore permissive defaults.

## 5.12 Audit, Trust Ledger and evidence retention

**Assessment: High/Critical risk.**

Current records cannot consistently prove actor, operation, evidence, decision, execution and outcome. Mutable file ledgers and optional writes are inadequate for protected effects.

Required outcome:

- authoritative transactional audit sequence;
- signed immutable segments;
- external checkpoints;
- protected effect fails if mandatory audit cannot persist;
- no secrets or reusable signatures in ordinary audit-reader output;
- exact linkage to operation, configuration, model/tool revision and outcome;
- retention, legal hold and privacy controls.

## 5.13 Backup and restore

**Assessment: High/Critical risk.**

Backups and checkpoints cannot be trusted until coherent state, manifest integrity and isolated restore have been demonstrated.

Required outcome:

- coordinated backup boundary;
- immutable manifest and artefact digests;
- encryption and key separation;
- retention/expiry rules;
- isolated restoration;
- application-level integrity checks;
- recovery point and recovery time evidence;
- regular requalification and tamper tests.

## 5.14 Deployment, networks, ports and secrets

**Assessment: Critical exposure failure.**

Broad host publication, a flat network, development credential fallbacks, inconsistent Compose inventories and permissive startup create reachable compromise paths.

Required outcome:

- one approved authenticated ingress;
- no direct publication of privileged/data-plane services;
- dangerous capabilities in explicit default-off profiles;
- fail-closed secret loading;
- service-specific least-privilege database users;
- Redis authentication/protected transport or strict isolation;
- trust-zone segmentation;
- policy-as-code checking of resolved deployment;
- immutable image references for qualified releases.

## 5.15 CI, tests and release assurance

**Assessment: Critical assurance failure.**

Tests and go/no-go cannot be authoritative when they use different identity, storage, model, dependency or security profiles from the release environment.

Required outcome:

- production-equivalent protected profile;
- negative tests for every critical bypass;
- no global dev-secret/fake mode;
- isolated test environments;
- dependency unavailability causes fail-closed result;
- immutable artefact and configuration revisions;
- signed release evidence;
- no qualification from stubs, compatibility paths or shallow health.

---

# 6. Cross-service attack-chain analysis

A local fix does not close a chain. Each chain closes only when every relevant trust boundary and final action path passes the required negative test.

## Chain 1 — Dashboard stored XSS to control-plane compromise

**Path:** hostile email/feed/finance/system/operator content → unsafe Dashboard rendering → same-origin script → privileged proxy routes → policy, memory, file, browser or execution action.

**Impact:** fleet-wide control and data compromise.

**Closure controls:** P0 isolation; authenticated ingress; safe rendering/CSP; removal of Dashboard-held admin authority; final-boundary capabilities.

**Required test:** inject hostile content through every displayed source and attempt every privileged proxy operation.

## Chain 2 — Anonymous Dashboard to Tool Gate mode change

**Path:** unauthenticated Dashboard access → mode API/proxy → permissive Tool Gate state.

**Impact:** removal or weakening of policy restrictions.

**Closure controls:** locked startup; no page-load mutation; authenticated step-up admin operation; no reusable admin credential in Dashboard.

**Required test:** anonymous, low-scope, modified-localStorage and XSS-driven mode-change attempts must fail.

## Chain 3 — Anonymous Agentic input to trusted action

**Path:** anonymous task/run input → body-supplied identity/delegation → planning/conviction → Gate or direct service → side effect.

**Impact:** attacker-authored instructions gain trusted execution authority.

**Closure controls:** authenticated ingress; explicit delegation; canonical operation; exact approval; final-boundary capability; P4 action qualification.

**Required test:** anonymous and identity-spoofed task escalation across all side-effect services.

## Chain 4 — Direct Executor to arbitrary code and fleet pivot

**Path:** direct Executor call → powerful allowlisted command/interpreter → filesystem/network/internal-service access → persistence or destructive action.

**Impact:** arbitrary code execution and broader system compromise.

**Closure controls:** P0 disable/isolate; P1 capability enforcement; P2 fixed operations, disposable sandbox and controlled egress.

**Required test:** direct bypass, argument smuggling, interpreter escape, descendant persistence and internal-network access suite.

## Chain 5 — memU poisoning to privileged prompt/action

**Path:** create preference/correction/episode/values record → retrieval strengthens record → Agentic promotes it into privileged context → confidence/trust rises → action.

**Impact:** persistent attacker influence over later decisions.

**Closure controls:** verified principal; provenance; untrusted-content boundary; partitioning; promotion policy; outcome-only trust.

**Required test:** cross-principal persistent injection, repeated-read ranking manipulation and derivative persistence after deletion.

## Chain 6 — Forged Verifier evidence to PASS

**Path:** caller supplies high-rank/duplicate evidence → heuristic scoring → PASS → downstream reliance.

**Impact:** false verification of unsupported or contradicted claims.

**Closure controls:** immutable evidence authority; source identity; independence groups; proposition-level entailment/contradiction; caller cannot set ranking or authority.

**Required test:** duplicate, correlated, stale, contradictory, caller-ranked and fabricated evidence packs.

## Chain 7 — Fusion manufactures consensus

**Path:** empty/one/failed/duplicate/stub specialists → aggregation → apparent consensus → downstream action.

**Impact:** false independent agreement and unsafe confidence.

**Closure controls:** signed specialist registry; live attestation; independence model; strict minimum evidence; structured disagreement; Verifier enforcement.

**Required test:** zero, one, duplicate, correlated, failed and stub specialist combinations must not produce qualifying consensus.

## Chain 8 — External content to Dashboard or prompt authority

**Path:** document/email/feed/broker/sensor payload → unsafe rendering or context assembly → stored XSS/prompt injection → privileged call or persistent memory.

**Impact:** browser compromise, memory poisoning and action manipulation.

**Closure controls:** safe schemas; output encoding; CSP; provenance; untrusted-content roles; explicit promotion and capability enforcement.

**Required test:** payload propagation from every external ingestion service through Dashboard, memory, Agentic and action paths.

## Chain 9 — Vault/path abuse to data exfiltration

**Path:** caller-controlled file/path/object mapping → ingest secret or host file → memory/index → retrieval or external egress.

**Impact:** credential, source, configuration or private-data disclosure.

**Closure controls:** object storage; canonical path policy; symlink denial; capability/principal scope; data classification; egress restrictions.

**Required test:** traversal, symlink, container secret, host path and cross-principal ingest/retrieval attempts.

## Chain 10 — Gate or ledger disclosure to lateral privilege

**Path:** low-scope reader accesses secrets, signatures, reusable decision artefacts or admin metadata → credential replay/forgery → privileged service calls.

**Impact:** lateral movement and policy bypass.

**Closure controls:** credential separation; minimised audit-reader views; one-time capabilities; no reusable secrets in logs/ledgers; encryption and access control.

**Required test:** low-scope and compromised-service attempts to extract and replay protected material.

## Chain 11 — Health manipulation to recovery reset

**Path:** spoof or repeatedly trigger shallow health/sweep → Supervisor generic recovery → containment/security state reset → unhealthy service reactivated.

**Impact:** bypass of containment and unsafe automated mutation.

**Closure controls:** recovery freeze; observation/action separation; incident authority; exact capability; fencing; independent postcondition.

**Required test:** spoofed health, repeated recovery, stale leader, timeout and partial-recovery scenarios.

## Chain 12 — Values/loyalty feedback to trust inflation

**Path:** anonymous or weakly attributed feedback/value/acknowledgement → persistent operator state → trust/conviction increase → broader autonomy.

**Impact:** attacker-driven authority escalation without verified outcomes.

**Closure controls:** principal scope; provenance; no self-certification; externally verified outcome-only trust; bounded expiring autonomy.

**Required test:** anonymous, duplicate, self-generated and cross-principal trust-credit attempts.

## Chain 13 — Weak market signal to financial mutation

**Path:** stale/correlated/invalid signal → manufactured confidence/consensus → broker operation → uncertain or duplicate execution.

**Impact:** financial loss, unintended exposure and false position state.

**Closure controls:** P1 final enforcement; durable financial operation; P4 domain qualification; independent data/source checks; post-trade reconciliation and limits.

**Required test:** one-source, correlated, stale, invalid, timeout, duplicate and partial-fill scenarios.

---

# 7. Trust-boundary review

## 7.1 User/browser to edge

**Current state:** host-published interfaces can expose privileged behaviour without a consistently authenticated principal.

**Required boundary:** one approved authenticated ingress, secure session management, CSRF protection, step-up authentication for administration, strict origin and content controls.

## 7.2 Dashboard to internal services

**Current state:** Dashboard can act as a reusable privileged proxy and confused deputy.

**Required boundary:** Dashboard carries verified principal/delegation context but no reusable fleet-wide authority. Final services independently enforce exact capabilities.

## 7.3 Service to service

**Current state:** network location, static IP, shared secret or body identity can substitute for workload identity.

**Required boundary:** unique workload identity, authenticated transport, narrow audience/scope, rotation and revocation, explicit delegation.

## 7.4 Tool Gate to side-effect service

**Current state:** advisory decision can be bypassed by direct final-service calls.

**Required boundary:** canonical digest and single-use audience-bound capability atomically consumed at the effect boundary.

## 7.5 External content to model/prompt/memory

**Current state:** hostile content can be promoted into privileged context, memory or evidence.

**Required boundary:** immutable provenance, untrusted data roles, safe context assembly, explicit promotion policy and no authority from formatting or source rank alone.

## 7.6 Principal to persistent data

**Current state:** stores and derivatives are not consistently principal/tenant/purpose partitioned.

**Required boundary:** authenticated partition key on every source and derivative, machine-enforced purpose/data class, lineage and deletion.

## 7.7 Container/service to host and network

**Current state:** powerful services may have broad file, process or network reach.

**Required boundary:** least-privilege container profile, disposable workers, restricted mounts, controlled egress and denied access to control/data zones.

## 7.8 Health observer to recovery authority

**Current state:** observation can lead directly to consequential mutation.

**Required boundary:** observer reports evidence; separate authorised incident/recovery service decides and executes exact actions with postconditions.

## 7.9 Backup/restore to live state

**Current state:** local files, mappings or incomplete checkpoints may be treated as authority.

**Required boundary:** signed coherent manifest, independent integrity check, isolated restore qualification and exact authorised promotion into live state.

## 7.10 Model output to action authority

**Current state:** confidence, consensus or verification labels can appear authoritative without qualifying evidence.

**Required boundary:** model output remains advisory until independently verified claims, calibrated uncertainty, exact human/policy approval and final-boundary capability are present.

---

# 8. Architecture assessment

## 8.1 Isolation

**Assessment: inadequate.** Containers and services are present, but broad ports, a flat network, powerful shared services, writable volumes and weak identity prevent them from acting as dependable security zones.

## 8.2 Resilience and availability

**Assessment: unreliable for high-consequence operation.** Process-local state, shallow health, unsafe retries, generic recovery and missing durable operation semantics create duplicated, lost or unknown outcomes.

## 8.3 Integrity

**Assessment: critical weakness.** Memory, evidence, audit, trust and state transitions lack consistent immutability, provenance, transactionality and independent verification.

## 8.4 Confidentiality and privacy

**Assessment: high/critical weakness.** Sensitive personal, financial, biometric, browser and operational data lacks uniform principal partitioning, purpose controls, retention, encryption and minimised exposure.

## 8.5 Accountability

**Assessment: insufficient.** Existing logs and ledgers cannot consistently prove the actor, exact approved operation, executing revision and verified outcome.

## 8.6 Operability and recovery

**Assessment: unsafe.** Health, recovery and backup controls can create false success and may restore or reactivate an insecure state.

## 8.7 Testability and release assurance

**Assessment: critical weakness.** Mocks, stubs, fake modes and shallow checks can produce green results that are not representative of the protected release profile.

## 8.8 Positive architectural potential

The repository demonstrates useful modular decomposition, extensive service coverage and an intention to implement policy, verification, memory, recovery and audit controls. Those elements provide a basis for reconstruction, but they are not presently reliable security boundaries.

The remediation programme should preserve useful modularity while replacing shared implicit trust with explicit, machine-verifiable contracts.

---

# 9. Prioritised action plan

## 9.1 Non-negotiable execution rules

1. **Preserve evidence before altering state.**
2. **Contain before constructing new capability.**
3. **Identity and exact operation enforcement precede side-effect re-enablement.**
4. **Isolation without identity is not sufficient.**
5. **A local patch does not close a systemic or attack-chain finding.**
6. **Legacy and replacement authority paths must not remain simultaneously usable.**
7. **Rollback may disable or restore the last verified secure revision; it may not weaken controls.**
8. **Finding counts remain unchanged until formal evidence-backed closure.**
9. **Each capability receives a revision-bound release decision; there is no blanket “Kai is safe.”**

## 9.2 Wave 0 / P0 — Evidence preservation and immediate containment

### Objective

Stop reachable compromise paths and establish a safe evidence-preserving development baseline. Wave 0 does not make the system production safe.

### Ordered work packages

1. **P0-PR-01 — Evidence freeze and deployment manifest capture**
   - capture Git revision/dirty state, resolved Compose, image digests, running configuration, environment-key names, network membership, volume/data snapshots and hashes;
   - keep secret values out of Git;
   - preserve an independently protected copy.

2. **P0-PR-02 — Edge lockdown and host-port removal**
   - remove direct publication from privileged/data services;
   - retain only an approved loopback or authenticated ingress;
   - add policy checks for disallowed ports.

3. **P0-PR-03 — Dangerous capability profiles and default-off fleet**
   - default startup excludes Executor, broad browser/egress, vault writes, introspection mutation, finance mutation, sensors, recovery and host watchers;
   - profile activation is explicit and machine-visible.

4. **P0-PR-04 — Tool Gate locked startup and Dashboard mode containment**
   - locked/restricted state on absent or invalid configuration;
   - no page-load or localStorage-driven server mode change;
   - no reusable Dashboard admin credential.

5. **P0-PR-05 — Fail-closed secrets and credential rotation support**
   - remove `localdev` and development-HMAC fallbacks from protected deployment;
   - fail startup when required secrets are absent;
   - separate service users and add rotation evidence.

6. **P0-PR-06 — Temporary trust-zone segmentation**
   - edge, control, data, agent, execution, egress, sensor and observability zones;
   - execution/egress denied direct control/data access;
   - observers denied recovery mutation.

7. **P0-PR-07 — Single-writer TurboVec containment**
   - one live writer;
   - introspection/readers use read-only snapshots or API/read models;
   - generation and ownership checks.

8. **P0-PR-08 — Restart, health and recovery containment**
   - freeze automatic recovery and generic mutation;
   - restrictive state survives restart;
   - health cannot directly re-enable unsafe capability.

9. **P0-PR-09 — Compose convergence and policy-as-code**
   - reconcile full/minimal inventories and security rules;
   - machine-check ports, profiles, secrets, networks, restart policy, health and image references.

### Wave 0 exit evidence

- evidence copies independently verified;
- no privileged service reachable from Internet/shared LAN;
- dangerous capabilities disabled by default;
- Tool Gate starts locked on invalid/missing configuration;
- no protected profile accepts known development credentials;
- resolved deployment policy check passes.

### Permitted state after Wave 0

**Isolated disposable development laboratory only.**

## 9.3 Wave 1 / P1 — Identity, canonical operation and final enforcement

### Objective

Create the security foundation every later control depends on.

### Ordered work packages

1. Security contracts and threat-model freeze.
2. Immutable identity/keyring runtime.
3. Human principal authentication.
4. Workload mTLS identity.
5. Canonical operation envelope and digest.
6. Explicit delegation authority.
7. Tool Gate decision-API rebuild.
8. Protected operator approval.
9. Single-use capability issue and transactional consumption.
10. Executor enforcement pilot.
11. Dashboard confused-deputy removal.
12. Agentic caller/delegation migration.
13. Remaining side-effect service migration.
14. Legacy HMAC/body-token/cosign removal.
15. Integrated security CI and release evidence.

### Required architecture

- one canonical operation serialisation;
- immutable operation digest shared by request, approval, capability, execution, outcome, idempotency and audit;
- asymmetric or equivalent audience-bound capabilities;
- explicit workload delegation;
- atomic single-use consumption;
- complete side-effect route registry;
- machine-readable migration state;
- legacy authority rejected in protected profiles.

### Wave 1 exit evidence

- anonymous privileged requests rejected;
- workload cannot impersonate another service/operator;
- changing any consequential field invalidates approval and capability;
- direct action-service bypass fails before side effect;
- Dashboard holds no reusable admin credential;
- Agentic cannot promote anonymous input into trusted Gate identity;
- every registered side-effect route enforces exact capabilities;
- protected release tests observe zero legacy protocol use.

### Permitted state after Wave 1

Privileged internal testing only. No hostile-content, sensitive-data, generic-execution or broad external-action qualification.

## 9.4 Wave 2 / P2 — Isolation, egress and persistent-data integrity

### Objective

Make hostile content, execution, browsing and persistent data processing safe enough for controlled testing.

### Ordered work packages

1. Isolation and data-integrity contracts.
2. Removal of generic Executor operations.
3. Disposable execution workers.
4. Execution postcondition/outcome authority.
5. Hardened egress proxy.
6. Browser context isolation.
7. Exact browser actions and postconditions.
8. Monitor isolated-rule execution.
9. Upload quarantine and format detection.
10. Archive/OOXML preflight.
11. Disposable parser/converter workers.
12. Provenance-rich parser results.
13. Vault secure object/path model.
14. Principal-partitioned memory/session storage.
15. Evidence/provenance and promotion policy.
16. Safe context assembly and prompt-injection boundary.
17. Durable memory/vector/graph outbox.
18. Graph partitioning and lineage.
19. Atomic supersession/contradiction state.
20. End-to-end derivative deletion.
21. Integrated Phase 2 CI and runtime assurance.

### Wave 2 exit evidence

- no arbitrary-code route in approved operations;
- worker escape, descendant, filesystem, network and resource tests pass;
- browser authenticated state cannot cross principal/workflow;
- SSRF, redirect and DNS-rebinding attacks fail safely;
- hostile archives/parsers/converters remain contained;
- every persistent record and derivative is principal/purpose/class scoped;
- external/model/document/sensor data cannot directly enter trusted prompt/evidence roles;
- memory/vector/graph partial failure is visible and recoverable;
- superseded/deleted records cannot remain active through derivatives.

### Permitted state after Wave 2

Controlled hostile-content and sensitive-data testing under W0–W2 constraints. No production or consequential-autonomy qualification.

## 9.5 Wave 3 / P3 — Distributed reliability, audit, privacy, recovery and backup

### Objective

Make distributed state and operational controls dependable enough for formal production qualification of explicitly non-autonomous capabilities.

### Ordered work packages

1. Reliability and lifecycle contract freeze.
2. Retry/fallback semantic replacement.
3. Standard liveness/readiness/capability health library.
4. Durable operation and idempotency authority.
5. Transactional outbox/inbox foundation.
6. Shared breaker and dependency-state authority.
7. Leader election and scheduler fencing.
8. Supervisor observation-only rebuild.
9. Recovery policy and incident authority.
10. Removal of fabricated healing knowledge.
11. Authoritative audit sequencer.
12. Signed audit segments and external checkpoints.
13. Tool Gate and Trust Ledger migration.
14. Data classification and schema annotations.
15. Encryption and key-management foundation.
16. Retention, deletion and legal-hold engine.
17. Structured operational logging.
18. Backup job and immutable manifest rebuild.
19. Isolated restore and qualification pipeline.
20. Incident-response and evidence-preservation workflow.
21. Integrated chaos, restore and operational release gate.

### Wave 3 exit evidence

- error, stub, degraded and fallback states cannot look successful;
- distributed mutation executes once logically through timeout/failover;
- multi-worker control state converges;
- stale leaders cannot commit;
- health observation cannot directly mutate recovery state;
- recovery requires exact authority and independent postcondition;
- audit append is linear, signed, segmented and externally anchored;
- protected effects cannot succeed without required audit;
- sensitive data is classified, encrypted and retention governed;
- logs are structured, minimised and injection resistant;
- backups are immutable, manifest-bound and regularly restore-qualified;
- chaos, race, clock, privacy and restore tests pass.

### Permitted state after Wave 3

Formal production qualification of individually released, non-autonomous capabilities. Model judgement, trust and autonomy remain unqualified.

## 9.6 Wave 4 / P4 — Model, evidence, trust and autonomy requalification

### Objective

Requalify cognitive and autonomous capabilities only after the P0–P3 foundations are independently verified.

### Ordered work packages

1. Capability and autonomy contract freeze.
2. Authoritative signed registry.
3. Model/backend attestation.
4. Reproducible benchmark authority.
5. Model selection and failover rebuild.
6. Removal of heuristic execution conviction.
7. Immutable claim/evidence service.
8. Proposition-level Verifier rebuild.
9. Verifier enforcement integration.
10. Specialist and Fusion registry.
11. Structured fusion and contradiction handling.
12. Prediction/outcome separation.
13. Calibration service.
14. Trust Ledger/scoring replacement.
15. Staged autonomy authority.
16. Financial-domain qualification.
17. Public-communication qualification.
18. Destructive/admin/recovery qualification.
19. Self-modification review pipeline.
20. Stub/fallback truthfulness migration.
21. Integrated capability requalification gate.

### Required architecture

- one signed service/model/tool/capability registry;
- exact model/backend attestation and fresh readiness;
- signed reproducible benchmark records;
- typed claims, immutable evidence and independence groups;
- claim-level entailment and contradiction;
- qualified specialist identity and structured Fusion;
- task-specific calibration and abstention;
- strict separation of prediction, proposed action, execution, observation and verified outcome;
- trust based only on linked independently verified outcomes;
- scoped, budgeted, expiring, revision-bound A0–A4 autonomy;
- separate qualification for financial, public, destructive, recovery and self-modifying domains.

### Wave 4 exit evidence

- all models/backends/tools resolve from the signed registry;
- no model selected without exact attestation, readiness and task qualification;
- stub/fake/fallback cannot create benchmark, consensus, trust or GO state;
- caller cannot fabricate evidence, ranking or PASS;
- contradiction and source independence are enforced;
- empty/one/duplicate/correlated specialists cannot create consensus;
- Verifier blocks consequential output/action when required;
- style, wording, hedging or formatting cannot increase authority;
- trust credit requires independently verified outcomes;
- autonomy is scoped, budgeted, expiring and revision-bound;
- high-consequence domains pass separate attack-chain qualification;
- every released capability has a signed evidence bundle and suspension/rollback plan.

### Permitted state after Wave 4

Only individually qualified capabilities may be enabled at their approved release state. Every other capability remains disabled, test-only or advisory.

---

# 10. Programme ownership and execution model

Dates and staffing durations are not assigned by this report because they depend on the delivery team, environment and independent assurance capacity. The dependency order must not be compressed by running unsafe downstream work early.

## 10.1 Required ownership functions

- **Programme security authority:** owns invariants, threat model, release decision and exceptions.
- **Platform/infrastructure:** exposure, Compose, networks, images, secrets and worker isolation.
- **Identity/security engineering:** principal/workload identity, delegation, capabilities and key management.
- **Service owners:** final-boundary enforcement, operation schemas and postconditions.
- **Data/privacy engineering:** partitioning, classification, lineage, retention and deletion.
- **SRE/reliability:** durable operations, health, leadership, recovery, chaos and backup/restore.
- **Model/evidence assurance:** registry, benchmarks, Verifier, Fusion, calibration and trust.
- **Independent reviewer:** adversarial validation and finding closure approval.

## 10.2 Safe parallelisation

Parallel work is allowed only where dependencies remain intact. Examples:

- P0 evidence tooling can be developed while P1 contracts are drafted, but no state-changing containment action occurs before evidence capture.
- Identity, canonical operation and side-effect inventory can progress in parallel within P1.
- Executor, browser and parser worker designs can be prepared during P1, but release testing waits for exact capability enforcement.
- Data classification and retention design can begin early, but machine enforcement depends on verified principal and lineage.
- P4 benchmark and claim schemas may be designed early, but no autonomy qualification occurs before P0–P3 gates pass.

## 10.3 Prohibited shortcuts

- patching Dashboard while leaving direct services reachable;
- adding authentication at a gateway but not the final service;
- treating feature flags as security boundaries;
- keeping legacy shared-secret and new capability paths simultaneously active;
- calling a container a sandbox without escape/egress/descendant tests;
- treating unit tests or screenshots as closure evidence;
- granting trust from self-generated outcomes;
- re-enabling a capability because one local issue was fixed.

---

# 11. Finding closure and evidence standard

A finding may be proposed for closure only when the evidence package contains:

1. finding ID and owning batch;
2. exact affected source/configuration paths;
3. root-cause statement and governing invariant;
4. immutable remediation commit(s);
5. built image/artefact digests;
6. configuration/registry/policy revisions;
7. positive functional tests;
8. negative/adversarial tests matching the exploit condition;
9. integration test across downstream consumers;
10. multi-worker/restart/failure test where applicable;
11. audit and verified outcome evidence;
12. residual risk and known exclusions;
13. independent reviewer and approval date;
14. qualification expiry or retest trigger where applicable.

Permitted closure states:

- `OPEN`
- `IMPLEMENTATION_IN_PROGRESS`
- `IMPLEMENTED_NOT_VERIFIED`
- `VERIFICATION_FAILED`
- `VERIFIED_PENDING_REVIEW`
- `CLOSED`
- `RISK_ACCEPTED` — explicit authority required; does not mean fixed.
- `NOT_APPLICABLE_AFTER_ARCHITECTURE_REMOVAL` — proof required that the affected capability/path no longer exists.

No planning document, including this report, changes a finding from `OPEN`.

---

# 12. Capability release model

There is no valid system-wide statement that “Kai is safe.” Release must be capability-specific, revision-bound and evidence-backed.

Recommended release states:

- `DISABLED`
- `ISOLATED_TEST_ONLY`
- `ADVISORY_ONLY`
- `SUPERVISED_INTERNAL`
- `SUPERVISED_PRODUCTION`
- `NARROW_AUTONOMOUS`
- `SUSPENDED`
- `REVOKED`

A released capability does not release unrelated services, data classes, domains, tools or autonomy scopes.

Every release bundle must identify:

- exact source and image revision;
- configuration and registry revisions;
- principal/delegation model;
- operation and side-effect routes;
- allowed data classes and purpose;
- tests and attack chains passed;
- budgets and limits;
- expiry and retest trigger;
- suspension and rollback procedure;
- residual risk and exclusions.

---

# 13. Immediate operating restrictions

Until applicable gates are implemented and independently verified:

1. Treat the stack as an isolated disposable development laboratory.
2. Do not expose it to the Internet or a shared LAN.
3. Do not load real sensitive personal, financial, credential, biometric or operational data.
4. Do not permit autonomous execution, browser actions, recovery, financial mutation or external messaging.
5. Do not treat Dashboard, Agentic, Verifier, Fusion, Tool Gate, Trust, self-audit or health output as authoritative evidence.
6. Treat existing memory, preference, feedback, confidence, trust, personality and world-context records as untrusted.
7. Treat current backups, checkpoints and ledgers as unverified until independently validated through isolated restoration.
8. Preserve audit files, logs, volumes, indexes and ledgers before destructive cleanup or credential/network changes.
9. Do not interpret the completed planning package as remediation.

---

# 14. Numerical reconciliation

The exact arithmetic is maintained in `CODE_AUDIT_MASTER.md`.

- Coherent pre-extension baseline: **2,529 findings**.
- Later findings-bearing batch delta: **2,051 findings**.
- Final total: **4,580 findings**.

Severity reconciliation:

- Critical: `221 + 31 = 252`.
- High: `1,284 + 1,156 = 2,440`.
- Medium: `1,021 + 864 = 1,885`.
- Low: `3 + 0 = 3`.

Every batch-local provisional repository total is historical only.

---

# 15. Final programme judgement

The repository audit is complete for the reviewed findings snapshot, and the explanatory report plus P0–P4 action plan are now consolidated.

The system is not responsibly remediated as a flat queue of 4,580 independent tickets. The findings are dominated by shared architectural causes and end-to-end compromise paths. The only defensible order is:

1. evidence preservation and containment;
2. identity, delegation, canonical operations and final enforcement;
3. execution/browser/parser isolation and persistent-data integrity;
4. distributed reliability, immutable audit, privacy, recovery and verified backup;
5. model, evidence, trust and autonomy requalification.

Current authoritative status:

- **4,580 confirmed findings**.
- **252 Critical**.
- **2,440 High**.
- **1,885 Medium**.
- **3 Low**.
- **Runtime remediation: none**.
- **Formally verified closed findings: zero**.
- **Overall release decision: NO_GO**.

The first implementation action, only when separately authorised, is `P0-PR-01`: preserve evidence and create the immutable acquisition manifest before secrets, volumes, networks, indexes, logs, ledgers or deployment behaviour are altered.
