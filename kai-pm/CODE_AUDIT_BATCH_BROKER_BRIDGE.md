# Kai Code Audit — Broker Bridge Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BROKER-001 | CRITICAL | Binance balances are exposed without authentication |
| KAI-BROKER-002 | CRITICAL | Trading positions and unrealised P&L are exposed without authentication |
| KAI-BROKER-003 | CRITICAL | Open orders and order identifiers are exposed without authentication |
| KAI-BROKER-004 | HIGH | Binance API error bodies are returned directly to callers |
| KAI-BROKER-005 | HIGH | Configurable Binance base URLs allow credential-bearing requests to arbitrary destinations if environment configuration is compromised |
| KAI-BROKER-006 | HIGH | Symbol-list requests create unbounded sequential upstream calls |
| KAI-BROKER-007 | HIGH | Public and signed response sizes and JSON complexity are unbounded |
| KAI-BROKER-008 | MEDIUM | A new HTTP client is created for every upstream request |
| KAI-BROKER-009 | MEDIUM | Per-symbol failures are silently suppressed in bulk ticker and spot-position enrichment |
| KAI-BROKER-010 | MEDIUM | Financial values are converted to binary floating point |
| KAI-BROKER-011 | MEDIUM | Health reports ok without checking credentials, mode validity or upstream reachability |
| KAI-BROKER-012 | MEDIUM | Request/error metrics are process-local and unsynchronised |
| KAI-BROKER-013 | MEDIUM | Stock and forex symbol lengths are unbounded |
| KAI-BROKER-014 | MEDIUM | yfinance errors are disclosed directly to callers |
| KAI-BROKER-015 | MEDIUM | Monitor-rule templates do not match the monitor-service schema |
| KAI-BROKER-016 | MEDIUM | Configuration values and mode are not validated at startup |

---

## Broker bridge: `broker-bridge/app.py`

### KAI-BROKER-001 — CRITICAL — Private account balances are public
**Issue:** `GET /balance` requires no authentication or authorisation. When Binance credentials are configured, it returns every non-zero spot or futures asset balance, including free, locked and total amounts.  
**Risk:** Any reachable caller can enumerate the operator’s exchange holdings and liquidity, exposing highly sensitive financial information and enabling targeted fraud or coercion.  
**Recommendation:** Require strong user authentication, least-privilege account scopes and explicit financial-data authorisation.  
**Status:** OPEN — immediate remediation required

### KAI-BROKER-002 — CRITICAL — Positions and P&L are exposed
**Issue:** `GET /positions` and futures `GET /pnl/summary` are unauthenticated. They disclose symbols, quantities, direction, entry and mark prices, leverage and unrealised profit/loss.  
**Risk:** Callers can reconstruct the user’s active strategy, exposure, liquidation sensitivity and current gains/losses.  
**Recommendation:** Restrict to authenticated, authorised principals and minimise returned fields.  
**Status:** OPEN — immediate remediation required

### KAI-BROKER-003 — CRITICAL — Open-order disclosure
**Issue:** `GET /orders` requires no authentication and returns open Binance orders, including order IDs, sides, types, prices, quantities, fills and status.  
**Risk:** Callers can inspect pending trading intentions and operational identifiers, front-run decisions or use the data for targeted social engineering.  
**Recommendation:** Protect the endpoint with strong authentication and audit every access.  
**Status:** OPEN — immediate remediation required

### KAI-BROKER-004 — HIGH — Binance error responses are disclosed
**Issue:** `_signed_get` returns `exc.response.text` directly in HTTP error details while preserving the upstream status code.  
**Risk:** Binance account, permission, signature, IP restriction, request and rate-limit diagnostics are exposed to unauthenticated callers.  
**Recommendation:** Return stable internal error codes and keep upstream bodies in protected logs with redaction.  
**Status:** OPEN

### KAI-BROKER-005 — HIGH — Credential-bearing destination is environment-controlled
**Issue:** `BASE_URL` and `FAPI_URL` are accepted directly from environment configuration. Signed requests send `X-MBX-APIKEY` and HMAC-signed query parameters to those destinations without host validation.  
**Risk:** Compromised deployment configuration can redirect credential-bearing requests to an attacker-controlled endpoint, exposing API keys and signed request material.  
**Recommendation:** Pin approved Binance hosts and TLS policy; reject arbitrary base URLs.  
**Status:** OPEN

### KAI-BROKER-006 — HIGH — Bulk ticker fan-out is unbounded
**Issue:** `/ticker?symbols=` accepts an unlimited comma-separated symbol list and performs one sequential upstream request per item.  
**Risk:** One unauthenticated request can generate an arbitrary number of Binance calls, hold the worker for an extended period and consume rate limits.  
**Recommendation:** Enforce a small maximum symbol count, deduplicate inputs and use bounded concurrency/cache.  
**Status:** OPEN

### KAI-BROKER-007 — HIGH — Upstream payloads are unbounded
**Issue:** Public and signed helpers call `resp.json()` without limiting response bytes or JSON complexity. Several endpoints process complete account, position, order-book and trade payloads.  
**Risk:** Unexpected or malicious upstream responses can consume excessive memory and CPU before validation.  
**Recommendation:** Stream with strict response limits and validate bounded schemas.  
**Status:** OPEN

### KAI-BROKER-008 — MEDIUM — HTTP connection pools are recreated
**Issue:** Every public or signed request creates a new `httpx.AsyncClient`.  
**Risk:** High-frequency market polling repeatedly creates TCP/TLS connections and pools, increasing latency and socket pressure.  
**Recommendation:** Reuse lifecycle-managed clients with bounded pools.  
**Status:** OPEN

### KAI-BROKER-009 — MEDIUM — Partial failures are hidden
**Issue:** Bulk ticker and spot-position price enrichment wrap each upstream call in `suppress(Exception)`. Failed symbols disappear or return `None` without an error list or completeness marker.  
**Risk:** Consumers can treat incomplete portfolio or market data as complete, producing incorrect valuations or decisions.  
**Recommendation:** Return explicit per-symbol success/error state and aggregate completeness.  
**Status:** OPEN

### KAI-BROKER-010 — MEDIUM — Monetary quantities use binary floats
**Issue:** Prices, balances, quantities, P&L and order values are converted using Python `float`.  
**Risk:** Binary floating-point rounding can alter exact financial quantities and comparisons, particularly for small assets and high precision markets.  
**Recommendation:** Preserve exchange decimal strings or use `Decimal` with defined precision.  
**Status:** OPEN

### KAI-BROKER-011 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always reports `status: ok`; it only indicates whether both credential strings are non-empty. It does not validate `MODE`, Binance connectivity, API permissions, clock synchronisation or credential acceptance.  
**Risk:** Orchestration treats the bridge as ready while all private calls may fail.  
**Recommendation:** Separate liveness, public-market readiness and authenticated-account readiness.  
**Status:** OPEN

### KAI-BROKER-012 — MEDIUM — Metrics are worker-local and race-prone
**Issue:** `_req_count` and `_err_count` are unsynchronised module-level integers.  
**Risk:** Multiple workers expose inconsistent totals; restarts erase history and concurrent updates can be lost.  
**Recommendation:** Use shared telemetry or explicitly label metrics as per-process.  
**Status:** OPEN

### KAI-BROKER-013 — MEDIUM — Equity/forex symbols lack bounds
**Issue:** `/stocks/{symbol}` and `/forex/{pair}` accept arbitrary path-string lengths and pass them to yfinance.  
**Risk:** Oversized or malformed values consume parsing, logging and downstream request capacity.  
**Recommendation:** Enforce strict symbol syntax and length allowlists.  
**Status:** OPEN

### KAI-BROKER-014 — MEDIUM — yfinance diagnostics are public
**Issue:** Stock and forex exceptions are interpolated directly into HTTP 502 details.  
**Risk:** Internal dependency, network and parsing details are exposed.  
**Recommendation:** Return stable errors and protected trace identifiers.  
**Status:** OPEN

### KAI-BROKER-015 — MEDIUM — Published monitor templates are incompatible
**Issue:** `/templates` emits rules using `source.field`, `condition.threshold`, action objects, `interval` and `cooldown`. The reviewed monitor service expects `source.extract`, `condition.value/percent`, string actions, `interval_seconds` and `cooldown_seconds`.  
**Risk:** Consumers applying the advertised templates receive invalid or semantically ineffective rules, including financial alerts that may never trigger as described.  
**Recommendation:** Generate templates from the monitor-service schema and test them end-to-end.  
**Status:** OPEN

### KAI-BROKER-016 — MEDIUM — Startup configuration is weakly validated
**Issue:** `MODE`, URLs and port are accepted directly. Unknown mode values fall into spot behaviour in several endpoints while health reports the unknown string.  
**Risk:** Misconfiguration causes inconsistent routing and misleading operational state.  
**Recommendation:** Validate mode as a strict enum and pin approved URLs and numeric ranges at startup.  
**Status:** OPEN

---

## Batch totals

- Findings: **16**
- Critical: **3**
- High: **4**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **523**
- Critical: **59**
- High: **188**
- Medium: **273**
- Low: **3**

## Files materially reviewed in this batch

`broker-bridge/app.py`.
