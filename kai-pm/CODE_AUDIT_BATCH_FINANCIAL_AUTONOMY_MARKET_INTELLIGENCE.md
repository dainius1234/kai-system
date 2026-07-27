# Kai Code Audit — Financial Autonomy and Market Intelligence Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-MARKET-001 | HIGH | Strategy auto-trading fails open when governance is unavailable |
| KAI-MARKET-002 | HIGH | The strategy trust request uses a fixed self-asserted conviction score |
| KAI-MARKET-003 | HIGH | Correlated technical indicators are treated as independent consensus voters |
| KAI-MARKET-004 | HIGH | Vote fraction is misrepresented as confidence |
| KAI-MARKET-005 | HIGH | Strategy errors are converted into HOLD signals and hidden from consensus quality |
| KAI-MARKET-006 | HIGH | Strategy calculations accept non-finite, negative and malformed price histories |
| KAI-MARKET-007 | HIGH | Flat price history produces RSI 100 and a false overbought SELL signal |
| KAI-MARKET-008 | HIGH | One SELL signal closes every long position for a symbol |
| KAI-MARKET-009 | HIGH | Multi-position closes are non-transactional and can partially execute |
| KAI-MARKET-010 | MEDIUM | Consensus price is taken from the last strategy response |
| KAI-MARKET-011 | MEDIUM | Strategy engine singleton and trading state are unsynchronised |
| KAI-MARKET-012 | MEDIUM | Module-level feature flags are documented but not enforced |
| KAI-MARKET-013 | MEDIUM | Strategy and governance errors are returned as normal result dictionaries |
| KAI-MARKET-014 | HIGH | One unauthenticated public price source directly marks financial state |
| KAI-MARKET-015 | HIGH | Market prices are cached without finite or positive-value validation |
| KAI-MARKET-016 | HIGH | Mark-to-market mutates open-position P&L without durable persistence |
| KAI-MARKET-017 | HIGH | Market-data failures become empty success-like results |
| KAI-MARKET-018 | MEDIUM | “Real-time” quote age uses local fetch time rather than market event time |
| KAI-MARKET-019 | MEDIUM | Symbol fan-out and query size are unbounded |
| KAI-MARKET-020 | MEDIUM | Synchronous external HTTP runs on calling threads and recreates clients |
| KAI-MARKET-021 | MEDIUM | Price cache is process-local, wall-clock based and concurrency-unsafe |
| KAI-MARKET-022 | MEDIUM | External response bytes and JSON structure are unbounded |
| KAI-MARKET-023 | MEDIUM | First singleton construction permanently captures unvalidated custom mapping/configuration |
| KAI-MARKET-024 | HIGH | Open-interest USD value assumes contracts × price for every contract type |
| KAI-MARKET-025 | HIGH | Alpha-signal fields accept non-finite and out-of-range upstream values |
| KAI-MARKET-026 | MEDIUM | Symbols and long/short periods are unvalidated and create unbounded cache keys |
| KAI-MARKET-027 | MEDIUM | Composite alpha retrieval performs repeated sequential HTTP requests |
| KAI-MARKET-028 | MEDIUM | Alpha freshness uses local receipt time rather than exchange timestamps |
| KAI-MARKET-029 | MEDIUM | Partial alpha failure is returned as a normal composite signal |
| KAI-MARKET-030 | MEDIUM | Funding annualisation and sentiment thresholds are hard-coded assumptions |
| KAI-MARKET-031 | MEDIUM | Alpha-signal feature gating is not enforced by the module |
| KAI-MARKET-032 | HIGH | Crude bag-of-words tone classification can manipulate market regime decisions |
| KAI-MARKET-033 | HIGH | Failed market/news sources are converted to and cached as neutral evidence |
| KAI-MARKET-034 | HIGH | Untrusted search abstracts are promoted into financial and macro evidence |
| KAI-MARKET-035 | MEDIUM | Macro context performs five synchronous sequential web searches |
| KAI-MARKET-036 | MEDIUM | Fear/Greed and global market values lack finite/range/schema validation |
| KAI-MARKET-037 | MEDIUM | Trending-symbol normalisation can remove `USD` from unintended positions |
| KAI-MARKET-038 | MEDIUM | Market-intelligence caches are volatile, unbounded and unsynchronised |
| KAI-MARKET-039 | MEDIUM | Market-intelligence feature gating is only documented |
| KAI-MARKET-040 | HIGH | One directional heuristic can produce financial conviction 10/10 |
| KAI-MARKET-041 | HIGH | Content and affiliate conviction claims are unsupported by demand, supply or commission data |
| KAI-MARKET-042 | HIGH | Opportunity searches hard-code the stale year 2025 |
| KAI-MARKET-043 | HIGH | Shared and correlated inputs are described as independent aligned signals |
| KAI-MARKET-044 | HIGH | Directive financial/content actions are generated without verification or calibration |
| KAI-MARKET-045 | MEDIUM | Caller-controlled subjects create unbounded opportunity cache entries |
| KAI-MARKET-046 | MEDIUM | Dependency failure still returns a recommendation and time horizon |
| KAI-MARKET-047 | MEDIUM | Opportunity-intelligence feature gating is not enforced by the module |

---

## Strategy engine: `agentic/strategy_engine.py`

### KAI-MARKET-001 — HIGH — Auto-trading governance fails open
**Issue:** `_check_trust` catches every unexpected import/runtime governance exception and continues. Only an explicit `PermissionError` blocks `auto_trade`.  
**Risk:** Financial-state mutation proceeds precisely when the trust authority is broken or unavailable. Although current positions are simulated, the resulting track record is intended to influence later real-capital autonomy.  
**Recommendation:** Fail closed for every state-changing financial action and expose governance readiness as mandatory.  
**Status:** OPEN

### KAI-MARKET-002 — HIGH — Trust evidence is self-asserted
**Issue:** The strategy engine always submits `conviction=7.0` to `gate_autonomous_action`, independent of signal confidence, evidence completeness, market freshness or verifier outcome.  
**Risk:** A fixed caller-selected number is treated as decision confidence at the governance boundary.  
**Recommendation:** Use a signed decision record whose calibrated conviction is independently computed and verified.  
**Status:** OPEN

### KAI-MARKET-003 — HIGH — Correlated strategies are counted as independent evidence
**Issue:** Momentum, moving-average cross and RSI all derive from the same caller-provided price series. Majority voting treats them as three separate voters with equal independence.  
**Risk:** One manipulated or erroneous series generates several agreeing signals and creates apparent consensus.  
**Recommendation:** Model dependency/correlation and require independent data/evidence sources.  
**Status:** OPEN

### KAI-MARKET-004 — HIGH — Vote share is labelled confidence
**Issue:** Consensus confidence is `number voting for winner / total strategies`. It does not measure predictive accuracy, calibration, sample size, data quality or signal magnitude.  
**Risk:** Two votes out of three become 0.667 “confidence” and satisfy the 0.5 auto-trade threshold despite no validated statistical confidence.  
**Recommendation:** Rename to vote share and gate actions on calibrated out-of-sample performance and uncertainty.  
**Status:** OPEN

### KAI-MARKET-005 — HIGH — Broken strategies disappear into HOLD
**Issue:** Strategy exceptions are logged at debug and converted to HOLD when prices exist. No degraded flag or failed-voter count accompanies consensus.  
**Risk:** A decision can look safely evaluated while one or more required strategies failed; HOLD votes also alter the winning fraction.  
**Recommendation:** Fail or explicitly degrade consensus when required voters error.  
**Status:** OPEN

### KAI-MARKET-006 — HIGH — Invalid price vectors enter decision logic
**Issue:** Price arrays and strategy parameters are ordinary floats/lists. NaN, infinity, negative prices, zeroes, extreme values and malformed periods are not rejected across strategy entry points.  
**Risk:** Comparisons and arithmetic produce invalid confidence/actions, non-standard JSON or exceptions hidden as HOLD.  
**Recommendation:** Validate finite positive prices, bounded history and legal parameter ranges.  
**Status:** OPEN

### KAI-MARKET-007 — HIGH — Flat markets become maximally overbought
**Issue:** RSI returns 100 whenever average loss is zero. For a completely flat series, average gain is also zero, but the function still returns 100 and emits SELL.  
**Risk:** No movement is misclassified as maximum RSI/overbought, creating false exit signals and automatic closes.  
**Recommendation:** Return neutral/undefined when both gain and loss are zero and validate against reference implementations.  
**Status:** OPEN

### KAI-MARKET-008 — HIGH — One SELL closes every matching long
**Issue:** `auto_trade` enumerates all open long positions for the symbol and closes each one. No position ID, maximum quantity, strategy ownership or operator scope is applied.  
**Risk:** A single consensus signal can liquidate unrelated positions opened by different strategies or sessions.  
**Recommendation:** Bind every decision to explicit position/quantity/risk scope.  
**Status:** OPEN

### KAI-MARKET-009 — HIGH — Bulk closes are partially irreversible
**Issue:** Positions are closed one-by-one. No transaction or compensating rollback exists if a later close fails.  
**Risk:** The returned error/state can conceal that some positions were already closed, producing inconsistent portfolios and retries.  
**Recommendation:** Use one atomic portfolio transaction or durable per-position operation plan/status.  
**Status:** OPEN

### KAI-MARKET-010 — MEDIUM — Arbitrary voter selects execution price
**Issue:** Consensus uses `signals[-1].price`, regardless of which strategies voted for the winner or whether responses disagree on price.  
**Risk:** Strategy ordering rather than a trusted quote authority determines the mutation price.  
**Recommendation:** Pass one independently validated market quote into all signals and execution.  
**Status:** OPEN

### KAI-MARKET-011 — MEDIUM — Engine state is process-local
**Issue:** The engine and strategy list are held in an unlocked singleton; reset/reconfiguration can race requests and workers instantiate different engines.  
**Risk:** Decisions vary across workers and concurrent resets/evaluations are unsafe.  
**Recommendation:** Use immutable versioned strategy configuration and one authoritative decision service.  
**Status:** OPEN

### KAI-MARKET-012 — MEDIUM — Feature enforcement is external only
**Issue:** The module docstring claims feature-flagging, but direct calls to `get_strategy_engine().auto_trade()` do not check a flag.  
**Risk:** Internal callers bypass route-level disablement.  
**Recommendation:** Enforce the capability policy at the mutation boundary.  
**Status:** OPEN

### KAI-MARKET-013 — MEDIUM — Errors look like business outcomes
**Issue:** Denial and internal failure are returned as ordinary dictionaries containing raw reason strings rather than typed failures.  
**Risk:** Downstream code can treat execution failure as a completed strategy result, while diagnostics leak.  
**Recommendation:** Use typed blocked/error states and protected traces.  
**Status:** OPEN

---

## Market data feed: `agentic/market_data.py`

### KAI-MARKET-014 — HIGH — One public feed directly changes financial state
**Issue:** CoinGecko’s unauthenticated `/simple/price` response is the sole quote source used by `mark_positions`; no corroboration, signed payload, exchange timestamp or outlier check is required. The proactive observer calls marking automatically when enabled.  
**Risk:** Compromised, stale or erroneous data directly changes displayed paper P&L and the track record used for future trust/autonomy.  
**Recommendation:** Use multiple independently authenticated venues, freshness checks and price-deviation controls.  
**Status:** OPEN

### KAI-MARKET-015 — HIGH — Invalid quote values are cached
**Issue:** Upstream `usd` is converted with `float` but not checked for finiteness or positivity. NaN, infinity, negative and zero values can enter cache/results.  
**Risk:** Position P&L and later strategy evidence become invalid, and NaN/Infinity can contaminate JSON/persistence.  
**Recommendation:** Reject non-finite/non-positive/outlier quotes.  
**Status:** OPEN

### KAI-MARKET-016 — HIGH — Marked P&L is not durable
**Issue:** `PaperTrader.mark_to_market` mutates each in-memory position’s unrealised P&L, but the market-data path does not save positions after marking.  
**Risk:** API-visible P&L disappears on restart and differs across workers, while callers receive a successful mark result.  
**Recommendation:** Persist one timestamped quote/valuation transaction or clearly treat values as ephemeral projections.  
**Status:** OPEN

### KAI-MARKET-017 — HIGH — Outage is indistinguishable from no data/no positions
**Issue:** Network, parsing, import and marking failures return `{}`.  
**Risk:** Automation cannot distinguish a healthy empty portfolio from a failed market feed, allowing stale/unmarked positions to look normal.  
**Recommendation:** Return typed unavailable/stale states and block decisions requiring current quotes.  
**Status:** OPEN

### KAI-MARKET-018 — MEDIUM — Quote age is local receipt age
**Issue:** `fetched_at` is local `time.time()` and labelled quote age. No provider market-event timestamp is recorded.  
**Risk:** Old upstream values received now appear fresh.  
**Recommendation:** preserve source event time and report source/receipt ages separately.  
**Status:** OPEN

### KAI-MARKET-019 — MEDIUM — Symbol fan-out is unbounded
**Issue:** Callers can supply arbitrary-length symbol lists and strings. Duplicates are retained and the batched provider query can grow without bounds.  
**Risk:** Requests consume parsing, URL, upstream quota and cache resources.  
**Recommendation:** enforce approved symbols and strict item/length limits.  
**Status:** OPEN

### KAI-MARKET-020 — MEDIUM — Blocking connection churn
**Issue:** Every fetch creates a synchronous `httpx.Client`. Direct async callers would block, and repeated marking creates new pools/TLS connections.  
**Risk:** Market operations degrade event-loop/worker throughput and upstream efficiency.  
**Recommendation:** use one lifecycle-managed asynchronous bounded client.  
**Status:** OPEN

### KAI-MARKET-021 — MEDIUM — Cache is volatile and race-prone
**Issue:** Cache dictionaries use wall-clock TTL and no locks/shared storage.  
**Risk:** Workers return different quotes, concurrent updates race and clock changes alter freshness.  
**Recommendation:** use immutable timestamped shared quote snapshots and monotonic freshness logic.  
**Status:** OPEN

### KAI-MARKET-022 — MEDIUM — Provider payloads are unbounded
**Issue:** Complete HTTP bodies and arbitrary JSON are materialised without byte/schema/depth constraints.  
**Risk:** Malformed or oversized provider responses consume memory and parsing time.  
**Recommendation:** enforce response limits and strict schemas.  
**Status:** OPEN

### KAI-MARKET-023 — MEDIUM — Singleton configuration is first-caller controlled
**Issue:** The first `get_market_data(ttl_s, timeout_s)` call fixes settings for the process; custom symbol maps are retained by reference rather than copied/validated.  
**Risk:** Internal call order or mutable external dictionaries silently change feed behaviour.  
**Recommendation:** initialise one immutable validated configuration during startup.  
**Status:** OPEN

---

## Alpha signals: `agentic/alpha_signals.py`

### KAI-MARKET-024 — HIGH — Open-interest notional formula is instrument-blind
**Issue:** Estimated `oi_usd` is always `openInterest * markPrice`. Contract size, inverse/linear specification, multiplier and unit semantics are ignored.  
**Risk:** Notional leverage can be materially wrong while presented as USD evidence for financial scoring.  
**Recommendation:** use exchange instrument metadata and validated contract-specific formulas.  
**Status:** OPEN

### KAI-MARKET-025 — HIGH — Upstream numerical fields are unvalidated
**Issue:** Funding, prices, ratios, percentages, interest and timestamps are converted directly without finite/range or cross-field checks.  
**Risk:** Invalid/provider-manipulated values generate extreme sentiment and conviction.  
**Recommendation:** apply strict endpoint schemas, plausible ranges and consistency checks.  
**Status:** OPEN

### KAI-MARKET-026 — MEDIUM — Inputs grow requests and caches without bound
**Issue:** Symbol strings and long/short `period` values are accepted directly. `_bnb_symbol` appends `USDT` to arbitrary text, and period participates in cache keys/upstream parameters.  
**Risk:** Caller-controlled values create unlimited cache entries and malformed provider requests.  
**Recommendation:** enforce an approved symbol/period enum.  
**Status:** OPEN

### KAI-MARKET-027 — MEDIUM — Composite retrieval duplicates sequential calls
**Issue:** Composite calls funding, open interest, long/short and premium sequentially. Open-interest calculation separately calls premium-index, and other methods call it again, each with a new client.  
**Risk:** One evaluation creates avoidable latency, quota load and inconsistent point-in-time evidence.  
**Recommendation:** retrieve one bounded timestamped market snapshot and derive signals from it.  
**Status:** OPEN

### KAI-MARKET-028 — MEDIUM — Source freshness is discarded
**Issue:** Most objects use local fetch time; exchange timestamps are ignored except next funding time.  
**Risk:** stale provider data appears current and signals from different instants are combined.  
**Recommendation:** preserve and validate source event timestamps and snapshot coherence.  
**Status:** OPEN

### KAI-MARKET-029 — MEDIUM — Incomplete composite appears normal
**Issue:** Missing signal calls are represented as `None` fields in a normal composite dictionary with a new current timestamp. No completeness/readiness score exists.  
**Risk:** Downstream scorers can act on partial evidence without recognising source failures.  
**Recommendation:** publish required/missing fields, source errors and a no-decision state.  
**Status:** OPEN

### KAI-MARKET-030 — MEDIUM — Financial interpretation is hard-coded
**Issue:** Funding is linearly annualised as three settlements every day, and sentiment thresholds are static constants without instrument/regime calibration.  
**Risk:** Annualised figures and crowd labels can be misleading across changing schedules/products/regimes.  
**Recommendation:** use exchange schedules and versioned empirically calibrated rules.  
**Status:** OPEN

### KAI-MARKET-031 — MEDIUM — Feature flag is not enforced internally
**Issue:** The module advertises `FF_ALPHA_SIGNALS`, but public methods never check it.  
**Risk:** direct internal callers operate when routes/UI claim the feature is disabled.  
**Recommendation:** enforce policy at the feed boundary.  
**Status:** OPEN

---

## Market intelligence: `agentic/market_intel.py`

### KAI-MARKET-032 — HIGH — Sentiment classifier is trivially manipulable
**Issue:** Tone is decided by set intersection of whitespace-split words against fixed bullish/bearish lists. Punctuation, negation, context, duplicated evidence and source reliability are ignored.  
**Risk:** Search text such as “not bullish”, injected keywords or punctuation changes can reverse/mute regime classification and financial scoring.  
**Recommendation:** use evidence-aware verified classification with source attribution and calibration.  
**Status:** OPEN

### KAI-MARKET-033 — HIGH — Failure becomes cached neutral evidence
**Issue:** Failed macro/news queries create neutral entries, and the resulting dictionary is cached for up to 30 minutes.  
**Risk:** An outage is treated as market neutrality and influences opportunity scoring as real evidence.  
**Recommendation:** represent unavailable sources explicitly and never cache failure as a valid regime.  
**Status:** OPEN

### KAI-MARKET-034 — HIGH — Search text is promoted to market evidence
**Issue:** DuckDuckGo abstracts/topics enter sentiment and macro context without provenance, source verification, recency checking or prompt-injection treatment.  
**Risk:** poisoned/irrelevant web text changes financial direction and recommended actions.  
**Recommendation:** use verified dated primary market sources and preserve evidence provenance.  
**Status:** OPEN

### KAI-MARKET-035 — MEDIUM — Macro collection is serial and blocking
**Issue:** Five web-scout searches run one after another in a synchronous method.  
**Risk:** One macro scan has cumulative latency and can block request workers.  
**Recommendation:** use bounded parallel asynchronous retrieval with one snapshot deadline.  
**Status:** OPEN

### KAI-MARKET-036 — MEDIUM — Provider values lack validation
**Issue:** Fear/Greed and CoinGecko market totals, percentages, counts and changes are converted directly with no finite/range/cross-field schema. Provider labels are accepted even if inconsistent with numeric values.  
**Risk:** malformed data yields impossible regimes and financial context.  
**Recommendation:** validate strict source-specific schemas and derive labels from validated values.  
**Status:** OPEN

### KAI-MARKET-037 — MEDIUM — Symbol normalisation is over-broad
**Issue:** `symbol.replace("USD", "")` removes every occurrence, not just a terminal quote suffix.  
**Risk:** unusual symbols can be transformed incorrectly and matched to unrelated trending entries.  
**Recommendation:** parse approved base/quote symbols structurally.  
**Status:** OPEN

### KAI-MARKET-038 — MEDIUM — Intelligence cache is ungoverned
**Issue:** Caches are process-local, unlocked, wall-clock based and can grow with caller-selected sentiment symbols.  
**Risk:** workers disagree, clock changes alter freshness and arbitrary symbols consume memory.  
**Recommendation:** bound keys and use shared immutable timestamped snapshots.  
**Status:** OPEN

### KAI-MARKET-039 — MEDIUM — Feature flag is not enforced internally
**Issue:** The documented `FF_MARKET_INTEL` is not checked by public methods.  
**Risk:** internal direct calls bypass disablement.  
**Recommendation:** enforce policy at module boundaries.  
**Status:** OPEN

---

## Opportunity intelligence: `agentic/opportunity_intel.py`

### KAI-MARKET-040 — HIGH — Single signal produces maximum conviction
**Issue:** Financial conviction is `round(abs(bull-bear) * 10 / total_points)`. If only one one-point signal exists, net equals total and conviction becomes 10/10.  
**Risk:** One weak heuristic is represented as maximum multi-signal certainty and assigned an immediate horizon/action.  
**Recommendation:** require minimum independent evidence and calibrate probability/uncertainty from historical outcomes.  
**Status:** OPEN

### KAI-MARKET-041 — HIGH — Unsupported opportunity claims
**Issue:** Content scoring grants points for topic tone, abstract length and finance keywords while claiming search interest/low supply. Affiliate scoring calls keyword categories “high-commission tier” without commission, conversion, product or programme data.  
**Risk:** Speculative heuristics are presented as monetisable opportunity evidence.  
**Recommendation:** measure the claimed demand, competition, economics and source reliability directly.  
**Status:** OPEN

### KAI-MARKET-042 — HIGH — Searches are temporally stale
**Issue:** Content and affiliate queries hard-code `2025` although the source review date is 27 July 2026.  
**Risk:** Search retrieval is biased toward stale material while outputs describe current/trending opportunities.  
**Recommendation:** use explicit current as-of dates and verify result publication/event dates.  
**Status:** OPEN

### KAI-MARKET-043 — HIGH — Correlation is described as independent alignment
**Issue:** Headlines count `signals` and state they are aligned, although funding, long/short, premium and macro signals share venues/data and macro tones share one crude search classifier.  
**Risk:** Evidence quantity/correlation is misrepresented as independent corroboration.  
**Recommendation:** track source lineage and discount shared causes.  
**Status:** OPEN

### KAI-MARKET-044 — HIGH — Recommendations bypass verification
**Issue:** The module emits actions such as “consider long/short”, publishing deadlines and affiliate-funnel launches directly from heuristics. No verifier, risk model, source freshness threshold or calibrated confidence gate is used.  
**Risk:** Downstream strategy/user decisions receive directive advice with authoritative conviction labels unsupported by evidence.  
**Recommendation:** classify output as unverified hypotheses and require evidence-backed verification/risk review before action.  
**Status:** OPEN

### KAI-MARKET-045 — MEDIUM — Cache keys are caller-controlled and unbounded
**Issue:** Arbitrary symbols, topics and categories create new cache entries; text length and cache cardinality are unrestricted.  
**Risk:** Repeated unique scans consume memory and retain potentially sensitive query/evidence data.  
**Recommendation:** constrain approved inputs and apply bounded LRU/TTL storage.  
**Status:** OPEN

### KAI-MARKET-046 — MEDIUM — Failure still yields an action
**Issue:** Exceptions leave empty evidence/neutral defaults, but every scan still constructs a recommendation and time horizon.  
**Risk:** Callers cannot distinguish an evaluated low-opportunity result from a failed scan.  
**Recommendation:** return an unavailable/incomplete status and no recommendation.  
**Status:** OPEN

### KAI-MARKET-047 — MEDIUM — Feature flag is not enforced internally
**Issue:** The module advertises `FF_OPPORTUNITY_INTEL`, but direct scanner methods never check it.  
**Risk:** internal callers bypass operational disablement.  
**Recommendation:** enforce feature and trust policy at the scanner boundary.  
**Status:** OPEN

---

## Batch totals

- Findings: **47**
- Critical: **0**
- High: **23**
- Medium: **24**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **904**
- Critical: **87**
- High: **339**
- Medium: **475**
- Low: **3**

## Files materially reviewed in this batch

`agentic/strategy_engine.py`, `agentic/market_data.py`, `agentic/alpha_signals.py`, `agentic/market_intel.py`, `agentic/opportunity_intel.py`, with integration confirmation against `agentic/app.py`. Existing paper-trader persistence/governance findings remain in `CODE_AUDIT_BATCH_AUTONOMOUS_STATE.md` and were not duplicated here.
