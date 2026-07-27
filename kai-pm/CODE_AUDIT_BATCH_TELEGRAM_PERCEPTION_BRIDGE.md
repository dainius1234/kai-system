# Kai Code Audit — Telegram Perception Bridge

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation and duplicate reconciliation  
Reviewed: 27 July 2026

This batch covers the standalone `perception/telegram.py` bridge and its sovereign Compose deployment. Findings already logged for `telegram-bot/app.py`, Agentic’s unauthenticated API, and `common.runtime` are not duplicated.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-TGBRIDGE-001 | CRITICAL | Any bridge-delivered message is forwarded into Agentic `/run` without sender or chat authorisation |
| KAI-TGBRIDGE-002 | CRITICAL | The bridge secret is sent to an arbitrary configured Telegram API destination |
| KAI-TGBRIDGE-003 | HIGH | An empty bridge secret silently disables bridge authentication |
| KAI-TGBRIDGE-004 | HIGH | The Agentic forwarding request carries no inbound authentication or channel-specific authority |
| KAI-TGBRIDGE-005 | HIGH | Polling has no durable cursor, acknowledgement or message identity |
| KAI-TGBRIDGE-006 | HIGH | Agentic delivery status is ignored and failed messages are neither retried safely nor dead-lettered |
| KAI-TGBRIDGE-007 | HIGH | Repeated bridge responses can replay the same Agentic task every five seconds |
| KAI-TGBRIDGE-008 | HIGH | Message fan-out and response payload size are unbounded |
| KAI-TGBRIDGE-009 | HIGH | One malformed message aborts processing of the complete polled batch |
| KAI-TGBRIDGE-010 | HIGH | One poison message can fail every subsequent polling cycle indefinitely |
| KAI-TGBRIDGE-011 | HIGH | Sequential forwarding lets one slow Agentic request block the complete message batch |
| KAI-TGBRIDGE-012 | HIGH | No rate limit, backpressure or workload quota protects Agentic from bridge traffic |
| KAI-TGBRIDGE-013 | HIGH | Caller-supplied session IDs enter Agentic without ownership or namespace validation |
| KAI-TGBRIDGE-014 | HIGH | Telegram channel provenance and sender identity are discarded before Agentic processing |
| KAI-TGBRIDGE-015 | HIGH | The configured `telegram-bridge` dependency is absent from the sovereign Compose stack |
| KAI-TGBRIDGE-016 | HIGH | Runtime startup installs unpinned `httpx` from the package index |
| KAI-TGBRIDGE-017 | HIGH | Runtime package installation is incompatible with the internal-only sovereign network |
| KAI-TGBRIDGE-018 | HIGH | The Telegram bridge container omits sovereign hardening defaults |
| KAI-TGBRIDGE-019 | HIGH | The bridge container has no healthcheck or readiness contract |
| KAI-TGBRIDGE-020 | HIGH | Startup failure is not automatically restarted by the sovereign Compose definition |
| KAI-TGBRIDGE-021 | MEDIUM | Telegram and Agentic traffic uses plaintext HTTP by default |
| KAI-TGBRIDGE-022 | MEDIUM | A new HTTP client and connection pool is created on every poll cycle |
| KAI-TGBRIDGE-023 | MEDIUM | The poll loop uses fixed-delay retry without exponential backoff or jitter |
| KAI-TGBRIDGE-024 | MEDIUM | Polling failures lose all diagnostic class, endpoint and response information |
| KAI-TGBRIDGE-025 | MEDIUM | The bridge exposes no metrics, last-success time, backlog or delivery counters |
| KAI-TGBRIDGE-026 | MEDIUM | The bridge has no graceful shutdown or owned client lifecycle |
| KAI-TGBRIDGE-027 | MEDIUM | Server device type is attributed to the externally originated task |
| KAI-TGBRIDGE-028 | MEDIUM | Sanitisation truncates content but does not establish a trusted Telegram input policy |
| KAI-TGBRIDGE-029 | MEDIUM | Forwarded tasks lack an idempotency key, Telegram update ID or trace correlation |
| KAI-TGBRIDGE-030 | MEDIUM | Compose dependency ordering waits only for Agentic container start, not readiness |

---

## Telegram perception bridge: `perception/telegram.py`

### KAI-TGBRIDGE-001 — CRITICAL — All returned messages become Agentic runs
**Issue:** every object under `payload["messages"]` is forwarded to Agentic `/run`. The bridge does not inspect or validate Telegram user ID, chat ID, membership, role, bot command, allowlist or operator approval.  
**Risk:** compromise or permissive configuration of the upstream bridge lets arbitrary external users initiate the full Agentic planning, memory and Tool Gate workflow.  
**Recommendation:** accept only authenticated, allowlisted, provenance-signed updates and apply a Telegram-specific restricted capability policy.  
**Status:** OPEN — immediate remediation required

### KAI-TGBRIDGE-002 — CRITICAL — Shared-secret exfiltration through destination configuration
**Issue:** when configured, `BRIDGE_SHARED_SECRET` is attached as `x-bridge-secret` to `TELEGRAM_API`, whose full base URL is environment-controlled and not restricted by scheme or host.  
**Risk:** a compromised or mistaken URL sends the bridge credential to an attacker-controlled endpoint.  
**Recommendation:** pin an authenticated HTTPS/mTLS bridge identity and bind the secret to that exact destination.  
**Status:** OPEN — immediate remediation required

### KAI-TGBRIDGE-003 — HIGH — Missing secret fails open
**Issue:** an empty `BRIDGE_SHARED_SECRET` simply produces an empty header dictionary and polling continues. The Compose default is empty.  
**Risk:** omitted secret configuration turns the bridge into an unauthenticated message source without failing startup/readiness.  
**Recommendation:** fail startup unless a valid service identity is configured.  
**Status:** OPEN

### KAI-TGBRIDGE-004 — HIGH — Anonymous forwarding into Agentic
**Issue:** the POST to `LANGGRAPH_URL` contains only `user_input`, `session_id` and `device`; no HMAC, bearer credential, sender identity, nonce or channel scope is supplied.  
**Risk:** Agentic cannot distinguish the bridge from an anonymous network caller or restrict Telegram-originated capabilities.  
**Recommendation:** use an authenticated service principal and bind the original actor/channel/update ID to a signed request.  
**Status:** OPEN

### KAI-TGBRIDGE-005 — HIGH — No committed message cursor
**Issue:** polling requests `/messages` without offset, cursor, since-time or acknowledgement token. The response’s message/update identity is not retained.  
**Risk:** restart and repeated polling can replay already processed messages; there is no exactly-once or at-least-once delivery contract.  
**Recommendation:** persist a durable cursor and commit it only after an idempotent Agentic acknowledgement.  
**Status:** OPEN

### KAI-TGBRIDGE-006 — HIGH — Delivery outcome is discarded
**Issue:** the Agentic POST response is not assigned, checked or parsed. HTTP 4xx/5xx and body-level denial are treated exactly like successful delivery.  
**Risk:** messages disappear without retry/dead-letter evidence, while upstream polling may also replay them unpredictably.  
**Recommendation:** validate a typed accepted operation ID and store durable delivery status.  
**Status:** OPEN

### KAI-TGBRIDGE-007 — HIGH — Unlimited replay amplification
**Issue:** because no message is acknowledged or deduplicated, an upstream bridge that returns a retained queue/list causes every message to be resubmitted on every five-second poll.  
**Risk:** one Telegram update can generate repeated LLM, memory and tool activity indefinitely.  
**Recommendation:** deduplicate by signed immutable update ID and use idempotency-bound Agentic operations.  
**Status:** OPEN

### KAI-TGBRIDGE-008 — HIGH — Unbounded batch and body allocation
**Issue:** the complete `/messages` response is materialised and parsed, and every `messages` item is iterated. No byte, item-count, nesting or aggregate text limit exists at the transport/schema boundary.  
**Risk:** a large or hostile bridge response exhausts memory and generates arbitrary Agentic request volume.  
**Recommendation:** stream/page a bounded typed queue.  
**Status:** OPEN

### KAI-TGBRIDGE-009 — HIGH — One malformed item aborts all later messages
**Issue:** the loop assumes each item has `.get`. A string, integer or `None` raises, leaving all later valid messages in the batch unprocessed.  
**Risk:** one malformed item creates denial of service and ordering loss.  
**Recommendation:** validate every message independently and quarantine invalid records.  
**Status:** OPEN

### KAI-TGBRIDGE-010 — HIGH — Persistent poison-loop
**Issue:** the broad outer exception restarts the same poll after five seconds. If the upstream list retains the malformed item, the bridge repeatedly fails at the same position forever.  
**Risk:** all subsequent Telegram delivery remains blocked with only a generic warning.  
**Recommendation:** use per-message failure tracking, dead-lettering and cursor advancement policy.  
**Status:** OPEN

### KAI-TGBRIDGE-011 — HIGH — Serial head-of-line blocking
**Issue:** each Agentic POST is performed synchronously inside the message loop. A five-second timeout or slow response delays every subsequent message.  
**Risk:** one slow task blocks the complete queue and causes repeated polling/backlog growth.  
**Recommendation:** use a bounded per-chat/actor worker queue with global concurrency and ordering guarantees.  
**Status:** OPEN

### KAI-TGBRIDGE-012 — HIGH — No admission control
The bridge has no message-rate limit, per-sender quota, bounded queue or Agentic concurrency budget.

### KAI-TGBRIDGE-013 — HIGH — Untrusted session namespace
`session_id` is taken directly from each message and only character-truncated by the generic sanitiser. It is not bound to a Telegram actor/chat or an authenticated Agentic session.

### KAI-TGBRIDGE-014 — HIGH — Source identity is erased
Only text and session ID survive forwarding. Telegram update ID, sender, chat, timestamp, command type and bridge identity are not included, so downstream audit and policy cannot establish provenance.

### KAI-TGBRIDGE-021 — MEDIUM — Plaintext internal transport
Both default service URLs use HTTP; the shared bridge secret, messages and Agentic tasks have no transport-level confidentiality or authenticated server identity.

### KAI-TGBRIDGE-022 — MEDIUM — Connection churn
A new synchronous `httpx.Client` is created and destroyed every five seconds, losing persistent pools and repeatedly allocating sockets.

### KAI-TGBRIDGE-023 — MEDIUM — Fixed retry cadence
Every failure retries after exactly five seconds; there is no exponential backoff, jitter, failure ceiling or circuit breaker.

### KAI-TGBRIDGE-024 — MEDIUM — Error evidence is discarded
All exception classes and details become the same `telegram poll failed` warning, preventing distinction between authentication, DNS, schema, Agentic denial and malformed messages.

### KAI-TGBRIDGE-025 — MEDIUM — No operational telemetry
The module exposes no health/readiness endpoint, processed/failed/replayed counts, last bridge/Agentic success or queue age.

### KAI-TGBRIDGE-026 — MEDIUM — No lifecycle ownership
The infinite loop owns no persistent client and does not handle cancellation/signals or close in-flight work gracefully.

### KAI-TGBRIDGE-027 — MEDIUM — Incorrect device attribution
`DEVICE` reflects the bridge server’s CPU/CUDA state and is submitted as the task device, not the Telegram client/source or requested execution device.

### KAI-TGBRIDGE-028 — MEDIUM — Generic truncation is not channel security
`sanitize_string` only strips three punctuation characters and truncates to 1,024 characters; it does not validate commands, provenance, role, prompt injection or allowed Agentic intent.

### KAI-TGBRIDGE-029 — MEDIUM — No traceable operation identity
Forwarded tasks contain no update ID, request ID, idempotency key or delivery-attempt number.

---

## Sovereign deployment: `docker-compose.sovereign.yml`

### KAI-TGBRIDGE-015 — HIGH — Referenced bridge service does not exist
**Issue:** `TELEGRAM_API` defaults to `http://telegram-bridge:9000`, but the sovereign Compose file defines no `telegram-bridge` service or external-network declaration for that hostname.  
**Risk:** the deployed perception bridge continuously fails DNS/connection and never receives messages.  
**Recommendation:** define and health-gate the authenticated bridge or remove the non-functional service.  
**Status:** OPEN

### KAI-TGBRIDGE-016 — HIGH — Unpinned runtime dependency installation
**Issue:** container command runs `pip install --no-cache-dir httpx` on every start, without version, hashes, lockfile or approved index.  
**Risk:** restarts execute changing supply-chain code and depend on package-index availability.  
**Recommendation:** build a pinned, hashed image artefact.  
**Status:** OPEN

### KAI-TGBRIDGE-017 — HIGH — Internal network prevents reliable package installation
**Issue:** the service is attached only to `sovereign-net`, declared `internal: true`, while startup requires public package-index access.  
**Risk:** a clean deployment cannot install `httpx`, so the container exits before running the bridge.  
**Recommendation:** install dependencies at image-build time and run offline.  
**Status:** OPEN

### KAI-TGBRIDGE-018 — HIGH — Hardened defaults are omitted
**Issue:** unlike core services, `perception-telegram` does not inherit `x-service-defaults`; no non-root user, read-only root filesystem, dropped capabilities, no-new-privileges, tmpfs or resource limits are configured.  
**Risk:** compromise of externally influenced message parsing/forwarding has a materially broader container impact.  
**Recommendation:** apply the hardened service baseline and least privilege.  
**Status:** OPEN

### KAI-TGBRIDGE-019 — HIGH — No health or readiness check
Compose has no healthcheck, and the Python module has no HTTP health endpoint.  
**Risk:** orchestration cannot distinguish running, dependency-failed, poison-loop or exited operation.  
**Recommendation:** expose readiness tied to authenticated bridge reachability and recent successful delivery.  
**Status:** OPEN

### KAI-TGBRIDGE-020 — HIGH — No restart policy
The service does not inherit `restart: unless-stopped` and defines no restart. Startup pip/DNS/import failure leaves it stopped.

### KAI-TGBRIDGE-030 — MEDIUM — Dependency is start-order only
`depends_on: agentic` has no `condition: service_healthy`; the poller can start while Agentic is not ready and silently lose/duplicate early deliveries.

---

## Batch totals

- Findings: **30**
- Critical: **2**
- High: **18**
- Medium: **10**
- Low: **0**

Repository-wide cumulative totals are intentionally omitted until duplicate reconciliation is completed.

## Files materially reviewed

`perception/telegram.py`, the `perception-telegram` service in `docker-compose.sovereign.yml`, and the Agentic `/run` request contract. Existing `telegram-bot/app.py` findings were not duplicated.
