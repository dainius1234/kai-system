# Kai Code Audit — Cross-Service Attack Chains Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This phase records emergent vulnerabilities that require two or more previously audited components. Component defects remain in their original batches; the findings below describe distinct end-to-end compromise paths and control failures.

## Consolidated batch index

| ID | Severity | Cross-service finding |
|---|---|---|
| KAI-CHAIN-001 | CRITICAL | Stored Dashboard XSS becomes full same-origin control of identity, memory, finance, browser, notifications and Tool Gate mode |
| KAI-CHAIN-002 | CRITICAL | Anonymous Dashboard callers borrow the server-held Tool Gate credential to change enforcement mode |
| KAI-CHAIN-003 | CRITICAL | Anonymous Agentic input is transformed into a server-signed Tool Gate request with caller-selected tool identity |
| KAI-CHAIN-004 | CRITICAL | Executor approval bypass plus allowlisted code-execution primitives creates an internal-service takeover pivot |
| KAI-CHAIN-005 | CRITICAL | Anonymous memU preference/correction poisoning becomes privileged Agentic system-prompt instruction |
| KAI-CHAIN-006 | CRITICAL | Caller-forged Verifier evidence can produce PASS and be consumed as authoritative truth by Agentic and Fusion |
| KAI-CHAIN-007 | CRITICAL | Fusion can manufacture consensus from failed, duplicate or stub specialists while Verifier rejection remains non-enforcing |
| KAI-CHAIN-008 | CRITICAL | News, email, broker and system metadata can persist hostile HTML that executes inside the privileged Dashboard origin |
| KAI-CHAIN-009 | CRITICAL | Vault arbitrary-file ingestion can move container secrets/private files into memU and then Agentic prompt context |
| KAI-CHAIN-010 | CRITICAL | Tool Gate ledger disclosure exposes trusted tokens and signatures that enable lateral privilege expansion |
| KAI-CHAIN-011 | CRITICAL | Supervisor health manipulation can invoke unauthenticated recovery endpoints that reset containment and security state |
| KAI-CHAIN-012 | CRITICAL | Anonymous value, conscience, loyalty and feedback poisoning can inflate trust/autonomy evidence used by later decisions |
| KAI-CHAIN-013 | CRITICAL | Weak market signals can become maximum conviction and reach fail-open autonomous paper-trading mutation |
| KAI-CHAIN-014 | HIGH | One global `keeper` namespace causes cross-caller memory, episode, preference, emotion and identity contamination |
| KAI-CHAIN-015 | HIGH | Process-local queues, breakers, ledgers, sessions and model state create worker-dependent security decisions |
| KAI-CHAIN-016 | HIGH | Optional and incomplete audit streams leave no trustworthy reconstruction of multi-service actions |
| KAI-CHAIN-017 | HIGH | memU Core and Introspection independently mutate one TurboVec file, creating stale retrieval and corruption across services |
| KAI-CHAIN-018 | HIGH | Memory compression and graph deletion failures create unreconciled evidence that remains retrievable after claimed deletion |
| KAI-CHAIN-019 | HIGH | Browser/search/OCR content is promoted into Agentic prompts without an untrusted-data boundary |
| KAI-CHAIN-020 | HIGH | Monitor rules combine attacker-selected web sources with browser, notification and TTS actions |
| KAI-CHAIN-021 | HIGH | Clipboard, email, camera and audio flows expose private operator data through unauthenticated Dashboard APIs |
| KAI-CHAIN-022 | HIGH | Shared retry logic can replay committed POST mutations across goals, feedback, schedules, values and notifications |
| KAI-CHAIN-023 | HIGH | Shared resilience treats many 4xx responses as dependency success, hiding policy and authentication failures |
| KAI-CHAIN-024 | HIGH | Shallow health contracts propagate false readiness into Dashboard, Supervisor and Compose orchestration |
| KAI-CHAIN-025 | HIGH | Model selector, Council, adversary and tree search combine stylistic heuristics and fail-open checks into false execution confidence |
| KAI-CHAIN-026 | HIGH | Independent hard-coded policy, model and tool registries drift and create contradictory enforcement across services |
| KAI-CHAIN-027 | HIGH | Poisoned P17–P22 state can generate reminders, scheduled tasks and escalations that Supervisor sends to Telegram |
| KAI-CHAIN-028 | HIGH | Stale or unauthenticated weather, news and market observations are injected into privileged situational context |
| KAI-CHAIN-029 | HIGH | Notification fallback and Dashboard polling can mark messages handled without verified operator delivery |
| KAI-CHAIN-030 | HIGH | Backup, ledger rotation and archive limits can erase the evidence needed to investigate earlier autonomous actions |
| KAI-CHAIN-031 | HIGH | Deployment inventory drift leaves security assumptions inconsistent about which services and ports are actually active |
| KAI-CHAIN-032 | HIGH | Public service topology and error disclosure make internal targeting and chain construction substantially easier |

---

## Critical end-to-end chains

### KAI-CHAIN-001 — CRITICAL — Dashboard XSS to complete control-plane compromise
**Entry:** Persistent or external content from finance, email, news, Docker, Git, broker or operator-model records reaches unsanitised `innerHTML` in `dashboard/static/app.html`.  
**Propagation:** Script executes in the Dashboard origin, where no user authentication or CSP constrains it.  
**Outcome:** The script can call same-origin Dashboard proxies for SOUL/AGENTS rewrites, Tool Gate mode, memU preferences, financial records, browser automation, file watching, notifications, model operations and private data reads.  
**Source batches:** `CODE_AUDIT_BATCH_DASHBOARD_FRONTEND.md`, `CODE_AUDIT_BATCH_DASHBOARD_GATEWAY.md`, feed/mail/broker/system service batches.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-002 — CRITICAL — Dashboard confused-deputy mode change
**Entry:** Anonymous caller posts to Dashboard `/api/mode`.  
**Propagation:** Dashboard attaches its server-held bearer credential and calls Tool Gate. Tool Gate mode administration does not enforce per-token administrative scopes.  
**Outcome:** An unauthenticated browser caller changes the system-wide WORK/PUB enforcement mode using internal service privilege.  
**Source batches:** Dashboard Gateway `KAI-DASH-002`; Tool Gate Extension `KAI-GATEX-001`, `KAI-GATEX-017`, `KAI-GATEX-025`.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-003 — CRITICAL — Anonymous input becomes trusted signed tool request
**Entry:** Anonymous caller submits Agentic `/run` with `task_hint`, session and plan-related input.  
**Propagation:** Agentic signs the caller-selected tool as actor `langgraph`; its HMAC excludes plan parameters and conviction. Low conviction/adversary block does not reliably stop the request.  
**Outcome:** Tool Gate receives a trusted internal signature for an action selected by an unauthenticated external caller.  
**Source batches:** Agentic API `KAI-AGAPI-016`, `044`, `045`, `046`, `047`, `048`; Tool Gate authentication/register findings.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-004 — CRITICAL — Executor pivot to fleet takeover
**Entry:** Caller reaches host-published Executor directly or through a weakly approved action.  
**Propagation:** Executor does not verify Gate approval. Allowlisted Python, Find, Make, Pip, Git and Curl provide arbitrary code, filesystem and network primitives.  
**Outcome:** Code in the Executor container can probe or invoke the many unauthenticated internal services, read accessible secrets/environment, mutate memory/identity and establish persistent state.  
**Source batches:** `CODE_AUDIT_BATCH_EXECUTOR.md`, Dashboard/Agentic/Tool Gate exposure batches.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-005 — CRITICAL — Persistent memory-to-system-prompt poisoning
**Entry:** Anonymous caller creates pinned preferences, corrections, notes or memories as `keeper`.  
**Propagation:** memU returns those records to Agentic planning/chat; Agentic inserts multiple retrieved values as privileged system messages. Planner also turns preferences/corrections into plan steps.  
**Outcome:** Attacker-controlled stored text persistently steers future model output and tool planning across unrelated sessions.  
**Source batches:** memU Core `KAI-MEMCORE-002`, `007`, `055`; Planner `KAI-DECIDE-014`, `015`, `016`; Agentic API `KAI-AGAPI-040`, `042`.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-006 — CRITICAL — Forged evidence to authoritative PASS
**Entry:** Caller supplies a fabricated Verifier evidence pack containing duplicate high-ranking records and a superficial plan.  
**Propagation:** Verifier treats rank/overlap as support, counts duplicates and awards plan consistency. Agentic/Fusion consume the verdict as verification metadata.  
**Outcome:** Unsupported or contradictory claims can receive PASS and influence execution, memory promotion or merged responses.  
**Source batches:** Verifier `KAI-VERIFY-002` through `007`, `010` through `027`; Agentic/Fusion integration batches.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-007 — CRITICAL — Consensus fabrication with no enforcing verifier
**Entry:** Caller requests one specialist, repeats specialist names, selects stubs or sets `min_agreement=0`.  
**Propagation:** Fusion reports maximum/high agreement; failed specialists may be excluded or one failed result may score 1.0. Verifier FAIL_CLOSED/outage does not alter consensus.  
**Outcome:** A non-reasoned or rejected answer is returned as a positive multi-model consensus result.  
**Source batches:** `CODE_AUDIT_BATCH_FUSION_ENGINE.md`, `CODE_AUDIT_BATCH_FUSION_ENGINE_EXTENSION.md`.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-008 — CRITICAL — External content to privileged-origin XSS
**Entry:** Attacker sends an email, publishes an RSS entry, controls broker/system metadata or poisons a globally writable operator-model record.  
**Propagation:** Source services return markup/control text without a canonical safe schema; Dashboard interpolates it into `innerHTML` or inline handlers.  
**Outcome:** Same-origin code executes and obtains every unauthenticated Dashboard proxy capability.  
**Source batches:** Dashboard Frontend; News Feed Extension; Email Reader Extension; Broker, Docker/Git watcher and memU personality batches.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-009 — CRITICAL — Local file to long-term prompt exfiltration
**Entry:** Anonymous caller invokes Vault Sync ingest with an arbitrary container-readable path or tampers with vault mappings.  
**Propagation:** File content is sent to memU as a vault note; memU and Dashboard/Agentic expose/search those records.  
**Outcome:** Secrets/private files can be moved into retrievable long-term memory and privileged prompt context, with stale duplicates surviving restart/deletion.  
**Source batches:** `CODE_AUDIT_BATCH_VAULT_SYNC.md`, `CODE_AUDIT_BATCH_VAULT_BRIDGE_EXTENSION.md`, memU Core/Agentic batches.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-010 — CRITICAL — Ledger secret leakage to privilege expansion
**Entry:** Holder of any trusted Tool Gate token calls ledger tail.  
**Propagation:** Ledger records include other tokens/session identifiers, signatures, nonces and full params; ledger access ignores tool scopes.  
**Outcome:** A low-purpose service token can recover credentials and signed material for lateral movement and stronger Gate operations.  
**Source batches:** Tool Gate Extension `KAI-GATEX-003`, `017`, `018`, `019`, `061`.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-011 — CRITICAL — Health manipulation to security-state reset
**Entry:** Attacker repeatedly triggers public Supervisor sweeps or causes shallow health failures.  
**Propagation:** Circuit failure threshold opens; Supervisor POSTs unauthenticated `/recover` to Agentic, memU, Tool Gate or Executor.  
**Outcome:** Breakers, pools, nonce/token state or temporary files are reset without authenticated incident diagnosis, potentially defeating containment during an active attack.  
**Source batches:** Live Supervisor `KAI-SUP-001` through `020`; recovery findings in Agentic, memU, Tool Gate and Executor.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-012 — CRITICAL — Moral/trust evidence self-poisoning
**Entry:** Anonymous caller submits feedback, values, conscience actions, loyalty, gratitude or wisdom confirmations.  
**Propagation:** memU/Ohana/Wisdom stores convert caller text and regex confidence into high-importance memories, alignment records and Trust Ledger evidence.  
**Outcome:** Fabricated “operator values” and successful alignment can influence prompts, trust score, autonomy readiness and future action selection.  
**Source batches:** memU Personality `KAI-PERSONA-001`, `002`, `064` through `075`; Cognitive Governance `KAI-COGOV-065` through `078`; Trust Governance batches.  
**Status:** OPEN — immediate remediation required

### KAI-CHAIN-013 — CRITICAL — Weak signal to autonomous financial mutation
**Entry:** One malformed/weak market or heuristic signal enters opportunity/strategy scoring.  
**Propagation:** One signal can yield conviction 10/10; correlated indicators become independent votes; governance failures are fail-open and a SELL closes all matching longs.  
**Outcome:** Unverified external data or attacker-controlled price history mutates financial/paper-trading state and generates a misleading performance record for future autonomy.  
**Source batches:** Financial Autonomy/Market Intelligence `KAI-MARKET-001` through `047`; Autonomous State batch.  
**Status:** OPEN — immediate remediation required

---

## High-severity systemic chains

### KAI-CHAIN-014 — HIGH — Global keeper identity collapse
Agentic, memU, planner, introspection and several personal-state modules hard-code `keeper` or accept it from callers. Data and behaviour from unrelated callers/sessions become one operator identity.

### KAI-CHAIN-015 — HIGH — Worker-dependent security state
Queues, nonces, idempotency caches, ledger history, mode overrides, sessions, breakers, feedback, graph mappings and model availability are frequently process-local. A request routed to another worker receives different security and business state.

### KAI-CHAIN-016 — HIGH — No trustworthy incident reconstruction
Audit streams are optional, suppress Redis failures, record only method/path/status and omit canonical body digest, actor, policy revision and downstream outcome. Multi-service chains cannot be reconstructed reliably.

### KAI-CHAIN-017 — HIGH — Shared TurboVec corruption/staleness
memU Core and memU Introspection load independent in-memory indexes and write one mounted file. Maintenance and live writes can corrupt the file while each process continues with stale vector state.

### KAI-CHAIN-018 — HIGH — Claimed deletion without evidence erasure
Memory compression, graph forget, vault delete and mapping failures can remove local lineage before backend deletion or retain graph/vector records after the API reports success/error-as-200.

### KAI-CHAIN-019 — HIGH — Web/document prompt injection
Browser search/scrape and OCR/parser results are inserted into the next Agentic prompt as user content, while Agentic itself promotes many retrieved records into system messages without provenance separation.

### KAI-CHAIN-020 — HIGH — Monitor action composition
Anonymous Monitor rules can fetch attacker-selected sources and invoke browser, Notify or TTS actions. Combined with Notify spoofing and Browser egress, one rule becomes a recurring network/social-engineering workflow.

### KAI-CHAIN-021 — HIGH — Private sensor/data aggregation exposure
Dashboard exposes clipboard history, email, camera/vision frames, audio transcripts, screen state, finance and memories through one unauthenticated origin. Compromise of any one frontend content source exposes the rest.

### KAI-CHAIN-022 — HIGH — Duplicate mutation through retries
Shared proxy resilience retries POST operations without idempotency. A committed-but-timed-out first call can be repeated for goals, feedback, schedules, values, notifications, memory and other mutations.

### KAI-CHAIN-023 — HIGH — Policy rejection hidden as service success
Shared resilience accepts many 4xx responses as successful dependency calls. Policy/auth/validation rejection can close breakers, suppress alerts and be returned as normal business JSON.

### KAI-CHAIN-024 — HIGH — False readiness propagation
Numerous health endpoints return `ok` in stub, stale, no-token, no-model or failed-dependency states. Compose, Dashboard and Supervisor then route, display or recover based on false readiness.

### KAI-CHAIN-025 — HIGH — Confidence from presentation rather than evidence
Conviction, planner specificity, rethink count, model selection, adversary checks and tree-search suffixes reward length/keywords/structure. Dependency failures often pass or disappear, allowing style to substitute for evidence.

### KAI-CHAIN-026 — HIGH — Registry/policy authority drift
Tool Gate, Executor, Agentic, model selector, Model Council, memU specialist routing and Compose each maintain independent hard-coded tool/model/capability inventories that disagree.

### KAI-CHAIN-027 — HIGH — Poisoned personal state to external Telegram delivery
Anonymous reminders, scheduled tasks and escalation records enter global memU state. Supervisor polls and sends them to Telegram, then may mark them fired regardless of delivery status.

### KAI-CHAIN-028 — HIGH — Untrusted situational data becomes privileged context
Weather, news, market, email and sensory services provide stale/unverified values or text that Agentic injects into privileged context without source expiry or trust boundaries.

### KAI-CHAIN-029 — HIGH — False notification acknowledgement
Notify may queue rather than deliver; Dashboard polling automatically dismisses returned notifications; Supervisor may mark tasks fired without confirming Telegram success. The system can record communication as handled when the operator never received it.

### KAI-CHAIN-030 — HIGH — Evidence retention failure
Tool Gate ledger size, Ledger Worker archive rotation, Backup weaknesses and local volatile logs can erase or fail to preserve the exact evidence required to investigate prior autonomous actions.

### KAI-CHAIN-031 — HIGH — Deployment inventory drift
Source, docs, health sweeps and Compose files disagree about active services such as memu-graph/orchestrator; ports and identities change across minimal/full definitions. Security reviews and monitoring therefore operate on different fleet models.

### KAI-CHAIN-032 — HIGH — Reconnaissance amplification
Dashboard, Supervisor, Metrics Gateway, Tool Gate, Fusion and individual services publicly expose topology, backend identities, errors, policy hashes, breaker states and model availability, substantially reducing the work needed to construct the chains above.

---

## Batch totals

- Findings: **32**
- Critical: **13**
- High: **19**
- Medium: **0**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,254**
- Critical: **204**
- High: **1,126**
- Medium: **921**
- Low: **3**

## Scope used for this phase

All confirmed service/module audit batches committed under `kai-pm/`, current Compose definitions, and direct integration code in Dashboard, Agentic, memU, Supervisor, Tool Gate, Executor, Verifier, Fusion, Vault Sync, Monitor, Telegram and perception/output services.
