# Kai Code Audit — Broker Bridge Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_BROKER_BRIDGE.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BROKERX-001 | HIGH | Private balance, position, order and P&L responses lack `Cache-Control: no-store` |
| KAI-BROKERX-002 | HIGH | Financial endpoints define no strict response models or versioned schemas |
| KAI-BROKERX-003 | HIGH | Binance numerical fields are not checked for finiteness, legal ranges or cross-field consistency |
| KAI-BROKERX-004 | HIGH | Upstream response schemas are assumed and malformed/missing fields can fail requests after partial work |
| KAI-BROKERX-005 | HIGH | The service does not verify that configured Binance credentials are read-only or least privilege |
| KAI-BROKERX-006 | HIGH | Binance credentials are long-lived environment strings with no key ID, reload or rotation state |
| KAI-BROKERX-007 | HIGH | Configured Binance destinations are not required to use HTTPS |
| KAI-BROKERX-008 | HIGH | HTTPX environment-proxy settings can route API-key and signed requests through an inherited proxy |
| KAI-BROKERX-009 | HIGH | No explicit Binance hostname/certificate identity pin is bound to the credential |
| KAI-BROKERX-010 | HIGH | Signed requests use the local wall clock without synchronising against Binance server time |
| KAI-BROKERX-011 | HIGH | Fixed `recvWindow=5000` has no measured clock-skew or latency evidence |
| KAI-BROKERX-012 | HIGH | Binance request-weight and rate-limit response headers are ignored |
| KAI-BROKERX-013 | HIGH | Private account endpoints have no service-side rate limit or caller quota |
| KAI-BROKERX-014 | HIGH | Market and account data has no cache, freshness policy or stale-value state |
| KAI-BROKERX-015 | HIGH | Spot position enrichment makes one sequential public quote request per non-stable asset |
| KAI-BROKERX-016 | HIGH | Spot portfolio values combine quotes fetched at different times without snapshot coherence |
| KAI-BROKERX-017 | HIGH | Spot `/positions` silently excludes all assets in a hard-coded stablecoin list |
| KAI-BROKERX-018 | HIGH | The hard-coded stablecoin list can omit new/renamed stables and misclassify portfolio exposure |
| KAI-BROKERX-019 | HIGH | A legitimate zero price is treated as missing because valuation uses `if price` |
| KAI-BROKERX-020 | HIGH | Futures “PnL summary” omits realised P&L, fees, funding, currency and valuation time |
| KAI-BROKERX-021 | HIGH | Account and market responses contain no retrieval time, exchange event time or snapshot ID |
| KAI-BROKERX-022 | HIGH | Binance-provided event/update timestamps are omitted from most derived responses |
| KAI-BROKERX-023 | HIGH | Non-finite Binance/yfinance values can enter API output and financial arithmetic |
| KAI-BROKERX-024 | HIGH | yfinance operations have no explicit deadline and continue in executor threads after cancellation |
| KAI-BROKERX-025 | HIGH | Anonymous callers can create unbounded concurrent yfinance work in the default thread executor |
| KAI-BROKERX-026 | HIGH | Crypto ticker, depth, stats, trades, funding and open-interest symbols lack canonical syntax/length validation |
| KAI-BROKERX-027 | HIGH | Open-order symbol filtering also accepts arbitrary unbounded strings |
| KAI-BROKERX-028 | HIGH | Exchange-controlled asset, symbol, side, type and status strings lack a safe display/control-character contract |
| KAI-BROKERX-029 | HIGH | Private financial reads have no tamper-evident access audit or purpose record |
| KAI-BROKERX-030 | HIGH | The Docker build makes application source writable by the runtime service user |
| KAI-BROKERX-031 | MEDIUM | Public Binance 4xx/429 responses are converted into generic 502 “unreachable” errors |
| KAI-BROKERX-032 | MEDIUM | Ticker symbol lists are not deduplicated before upstream fan-out |
| KAI-BROKERX-033 | MEDIUM | Recent-trade `side` represents taker direction but the response does not define that semantic |
| KAI-BROKERX-034 | MEDIUM | Stock `volume` is populated from `three_month_average_volume`, not current trading volume |
| KAI-BROKERX-035 | MEDIUM | Forex normalisation accepts malformed/empty values and ambiguous repeated `=X` suffixes |
| KAI-BROKERX-036 | MEDIUM | yfinance results contain no provider event time, exchange time or freshness state |
| KAI-BROKERX-037 | MEDIUM | Upstream failures have no retry budget, exponential backoff or circuit breaker |
| KAI-BROKERX-038 | MEDIUM | Public market endpoints provide no ETag/conditional-request support and repeatedly refetch identical data |
| KAI-BROKERX-039 | MEDIUM | Public metrics expose request/error counts and mode without administrative authentication |
| KAI-BROKERX-040 | MEDIUM | Requests and responses have no operation/correlation ID linking downstream calls |
| KAI-BROKERX-041 | MEDIUM | Uptime and request chronology use wall-clock time without a monotonic sequence |
| KAI-BROKERX-042 | MEDIUM | Health exposes no Binance/yfinance versions, endpoint identity or last successful request time |
| KAI-BROKERX-043 | MEDIUM | The service has no structured middleware/audit for endpoint access and upstream outcomes |
| KAI-BROKERX-044 | MEDIUM | Browser/Dashboard consumers receive no explicit source-confidence or market-data quality metadata |
| KAI-BROKERX-045 | MEDIUM | yfinance attributes are returned without normalising types into a strict JSON/financial schema |
| KAI-BROKERX-046 | MEDIUM | The service is deployed in the minimal topology but omitted from the full topology |
| KAI-BROKERX-047 | MEDIUM | FastAPI, HTTPX, yfinance and the Python base image are not reproducibly digest-pinned |
| KAI-BROKERX-048 | MEDIUM | No dedicated Broker Bridge test suite was found for signing, schema, clock skew or financial completeness |
| KAI-BROKERX-049 | MEDIUM | The service has no lifespan-owned clients, bounded yfinance executor or graceful in-flight drain |
| KAI-BROKERX-050 | MEDIUM | No immutable record links caller, signed request digest, upstream response digest and returned financial snapshot |

---

## High-severity findings

### KAI-BROKERX-001 — HIGH — Private financial responses are cacheable
**Issue:** Sensitive account endpoints return ordinary JSON with no no-store/private cache policy.  
**Risk:** Browser/proxy caches may retain balances, open orders, positions and P&L after the request.  
**Recommendation:** Require authenticated access and emit strict private/no-store headers for account data.  
**Status:** OPEN

### KAI-BROKERX-002 — HIGH — Financial schema is not enforced
Endpoints return dictionaries constructed from assumed provider fields without Pydantic response models, precision contracts or API revisions.

### KAI-BROKERX-003 — HIGH — Invalid financial numerics
Direct `float()`/`int()` conversions do not reject NaN, infinity, negative prices where illegal, impossible fill quantities or inconsistent balance totals.

### KAI-BROKERX-004 — HIGH — Provider schema trust
Missing/wrong fields raise after the entire provider payload has already been accepted; no typed provider-contract failure or partial-completeness record exists.

### KAI-BROKERX-005 — HIGH — Credential permission scope unknown
Health checks only that strings exist. It does not prove trade/withdraw permissions are disabled or bind the key to required read endpoints/IP restrictions.

### KAI-BROKERX-006 — HIGH — No credential lifecycle
Keys have no key ID, source revision, activation/expiry time, reload endpoint or rotation readiness.

### KAI-BROKERX-007 — HIGH — Plain-HTTP credential destination allowed
`BASE_URL`/`FAPI_URL` may use `http://`; signed query material and API-key headers would be sent without TLS.

### KAI-BROKERX-008 — HIGH — Inherited proxy credential path
HTTPX clients use default `trust_env=True`; configured proxy environment can observe destination, API-key header and signed query.

### KAI-BROKERX-009 — HIGH — Credential not bound to approved identity
System CA validation alone is used; the service has no approved-host manifest or certificate/public-key pin for account credentials.

### KAI-BROKERX-010 — HIGH — Unsynchronised signed timestamp
The signature timestamp is derived solely from local `time.time()`.

### KAI-BROKERX-011 — HIGH — Fixed receive window
No current skew/round-trip measurement demonstrates that 5,000 ms is safe or explains rejection.

### KAI-BROKERX-012 — HIGH — Upstream quota state discarded
Used-weight/order-count/rate-limit headers are not recorded or exposed to admission control.

### KAI-BROKERX-013 — HIGH — Account-read amplification
Anonymous traffic can repeatedly invoke signed account/order/position endpoints and consume account/IP limits.

### KAI-BROKERX-014 — HIGH — No freshness contract
Every result looks current even when the upstream/provider supplied stale/cached data or a previous UI value remains displayed.

### KAI-BROKERX-015 — HIGH — Portfolio quote fan-out
One spot position call can produce one account request plus one ticker request for every non-stable asset.

### KAI-BROKERX-016 — HIGH — Non-coherent valuation
Assets are valued at serially different instants without one exchange snapshot time.

### KAI-BROKERX-017 — HIGH — Stable holdings omitted
The endpoint named positions excludes USDT/BUSD/USDC/FDUSD/TUSD without returning an excluded-assets section.

### KAI-BROKERX-018 — HIGH — Static asset taxonomy
New, depegged or renamed stable assets are not governed by provider metadata or a versioned policy.

### KAI-BROKERX-019 — HIGH — Zero-price ambiguity
`price=0.0` follows the missing-price branch and returns no value.

### KAI-BROKERX-020 — HIGH — Misleading P&L summary
Only current unrealised position values are summed while financially material components are omitted from the “summary” contract.

### KAI-BROKERX-021 — HIGH — Missing snapshot identity
Account/ticker/order responses lack a common retrieval/event timestamp and immutable snapshot ID.

### KAI-BROKERX-022 — HIGH — Source chronology discarded
Fields such as trade/funding/open-interest time are included only in selected endpoints; account/position/order/ticker derivations omit equivalent source time.

### KAI-BROKERX-023 — HIGH — Non-finite provider output
Provider values are not finite-checked before arithmetic, rounding and JSON serialisation.

### KAI-BROKERX-024 — HIGH — Unbounded yfinance task lifetime
`run_in_executor` calls are awaited without timeout and cannot reliably cancel the running blocking thread.

### KAI-BROKERX-025 — HIGH — Thread-pool saturation
No semaphore or per-symbol quota protects the default executor from concurrent anonymous stock/forex calls.

### KAI-BROKERX-026 — HIGH — Crypto symbol contract absent
Arbitrary path/query strings are forwarded to exchange endpoints after uppercase conversion.

### KAI-BROKERX-027 — HIGH — Order filter contract absent
The open-order symbol is not length/character/market validated.

### KAI-BROKERX-028 — HIGH — Unsafe provider strings
Upstream identifiers/status fields may contain unexpected Unicode/control text and are propagated to Dashboard rendering.

### KAI-BROKERX-029 — HIGH — No private-read accountability
There is no durable event recording who accessed which account snapshot and why.

### KAI-BROKERX-030 — HIGH — Writable runtime application
The Dockerfile runs `chown -R app:app /app`; compromise of the service user can modify its own source inside the running container.

---

## Medium-severity findings

### KAI-BROKERX-031 — MEDIUM — Public error semantics lost
`_public_get` catches all HTTP status errors as generic Binance-unreachable 502, hiding invalid symbols and rate limiting.

### KAI-BROKERX-032 — MEDIUM — Duplicate ticker work
Repeated symbols create repeated sequential calls/results.

### KAI-BROKERX-033 — MEDIUM — Ambiguous trade side
The BUY/SELL label is derived from `isBuyerMaker` but does not state whether it represents maker or taker direction.

### KAI-BROKERX-034 — MEDIUM — Mislabelled stock volume
The field called `volume` is three-month average volume.

### KAI-BROKERX-035 — MEDIUM — Weak forex normalisation
The pair is not restricted to two valid currency codes and suffix handling is string-based.

### KAI-BROKERX-036 — MEDIUM — Missing yfinance chronology
Snapshot values have no provider timestamp.

### KAI-BROKERX-037 — MEDIUM — No resilience policy
Transient failures are immediately returned; persistent failures are retried by callers without coordinated backoff.

### KAI-BROKERX-038 — MEDIUM — Repeated identical upstream work
No service cache or conditional request policy exists.

### KAI-BROKERX-039 — MEDIUM — Public telemetry
Metrics requires no administrative identity.

### KAI-BROKERX-040 — MEDIUM — Missing correlation
One portfolio response cannot be traced across its multiple upstream requests.

### KAI-BROKERX-041 — MEDIUM — Wall-clock uptime
Clock adjustments can alter reported uptime.

### KAI-BROKERX-042 — MEDIUM — Incomplete health provenance
No dependency version, approved host or last-success detail is supplied.

### KAI-BROKERX-043 — MEDIUM — Missing service audit
No structured access/upstream-result middleware exists.

### KAI-BROKERX-044 — MEDIUM — Missing quality metadata
Consumers cannot distinguish exchange account data, Binance public data and yfinance-derived values by confidence/freshness policy.

### KAI-BROKERX-045 — MEDIUM — yfinance type drift
Provider-specific scalar/object types are returned directly and may drift or fail JSON serialisation.

### KAI-BROKERX-046 — MEDIUM — Topology drift
Full Compose omits the Broker Bridge while full-stack code still contains financial integrations and URLs.

### KAI-BROKERX-047 — MEDIUM — Non-reproducible build
Dependencies/base are range/tag based.

### KAI-BROKERX-048 — MEDIUM — Missing financial integration tests
No dedicated source test was found for signatures, permissions, clock/rate headers, Decimal fidelity or portfolio completeness.

### KAI-BROKERX-049 — MEDIUM — Missing resource lifecycle
No shared clients/executor limits or shutdown drain exist.

### KAI-BROKERX-050 — MEDIUM — Missing signed-request audit chain
No immutable evidence binds requestor, exact signed query, provider response and returned snapshot.

---

## Batch totals

- Findings: **50**
- Critical: **0**
- High: **30**
- Medium: **20**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,802**
- Critical: **193**
- High: **1,425**
- Medium: **1,181**
- Low: **3**

## Files materially reviewed

`broker-bridge/app.py`, `broker-bridge/Dockerfile`, `broker-bridge/requirements.txt`, minimal/full deployment topology, Dashboard/Monitor integrations and the existing Broker Bridge audit.
