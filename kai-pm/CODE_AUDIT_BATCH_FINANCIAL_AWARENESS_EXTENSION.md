# Kai Code Audit — Financial Awareness Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_FINANCIAL_AWARENESS.md` or `CODE_AUDIT_BATCH_PERSISTENCE_1.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-FINANCEX-001 | HIGH | The tax-year start calculation is wrong on days 1–5 of every month from May through December |
| KAI-FINANCEX-002 | HIGH | Current-tax-year filters enforce no upper date bound and include future-dated records |
| KAI-FINANCEX-003 | HIGH | VAT rolling-income filters also include future-dated records |
| KAI-FINANCEX-004 | HIGH | Future CIS deductions can artificially reduce the reported current tax balance |
| KAI-FINANCEX-005 | HIGH | The rolling VAT window is modelled as 365 days rather than a versioned calendar/tax-rule period |
| KAI-FINANCEX-006 | HIGH | Monetary validators accept NaN and infinity because they reject only values below zero |
| KAI-FINANCEX-007 | HIGH | Python JSON persistence permits non-standard `NaN` and `Infinity` tokens in the financial ledger |
| KAI-FINANCEX-008 | HIGH | One non-finite record can poison tax, VAT, CIS and alert calculations |
| KAI-FINANCEX-009 | HIGH | A valid JSON object/non-list file is accepted by `_load_records` and causes downstream iteration/type failures |
| KAI-FINANCEX-010 | HIGH | Non-dictionary entries inside the record array can crash summaries rather than being quarantined |
| KAI-FINANCEX-011 | HIGH | Tampered string/object monetary fields can cause TypeErrors or unsafe coercion during summation |
| KAI-FINANCEX-012 | HIGH | The tax-balance floor at zero hides potential excess CIS credit/refund amounts |
| KAI-FINANCEX-013 | HIGH | Financial results omit an as-of date, tax-year end and dataset revision |
| KAI-FINANCEX-014 | HIGH | All records share one global financial identity with no authenticated taxpayer/business partition |
| KAI-FINANCEX-015 | HIGH | CIS deduction status is accepted as a caller assertion without evidence of contractor verification status |
| KAI-FINANCEX-016 | HIGH | Hard-coded CIS rates have no effective date, jurisdiction or rule-pack revision |
| KAI-FINANCEX-017 | HIGH | CIS records contain no source statement, payment evidence, contractor document or reconciliation identity |
| KAI-FINANCEX-018 | HIGH | Duplicate CIS submissions are not detected and are counted repeatedly |
| KAI-FINANCEX-019 | HIGH | CIS records have no update, void, correction, supersession or deletion lifecycle |
| KAI-FINANCEX-020 | HIGH | Time-derived CIS IDs have no storage uniqueness constraint across workers/processes |
| KAI-FINANCEX-021 | HIGH | The data model has no currency field and silently treats every amount as GBP |
| KAI-FINANCEX-022 | HIGH | The CIS calculation has no VAT amount/component and cannot distinguish VAT-inclusive payment data |
| KAI-FINANCEX-023 | HIGH | Invoice generation has no VAT amount/status/currency model |
| KAI-FINANCEX-024 | HIGH | Invoice output is not persisted, versioned or linked to its underlying CIS/payment record |
| KAI-FINANCEX-025 | HIGH | Invoice and record text fields permit newline/control-character injection into rendered invoices and logs |
| KAI-FINANCEX-026 | HIGH | Invoice/financial responses expose UTR and personal/business identifiers without `Cache-Control: no-store` |
| KAI-FINANCEX-027 | HIGH | `finance_summary` loads the financial file independently three times and can return a mixed-state snapshot |
| KAI-FINANCEX-028 | HIGH | A concurrent record write between CIS, VAT and tax subcalls can produce internally contradictory summary values |
| KAI-FINANCEX-029 | HIGH | Sensitive contractor names and exact amounts are written to ordinary application logs |
| KAI-FINANCEX-030 | HIGH | The image unnecessarily copies the repository security directory into the financial-data service |
| KAI-FINANCEX-031 | HIGH | Trusted-token and policy files become additional compromise loot beside persistent financial records |
| KAI-FINANCEX-032 | HIGH | Financial records and calculations have no tamper-evident source/access/mutation audit trail |
| KAI-FINANCEX-033 | MEDIUM | Tax-year boundary logic is duplicated in three functions, increasing inconsistent-fix and drift risk |
| KAI-FINANCEX-034 | MEDIUM | `date.today()` uses the container timezone rather than a configured taxpayer/accounting timezone |
| KAI-FINANCEX-035 | MEDIUM | Invalid dates remain stored but silently disappear from every tax/VAT calculation |
| KAI-FINANCEX-036 | MEDIUM | The service exposes no endpoint to list or reconcile the exact records used in a summary |
| KAI-FINANCEX-037 | MEDIUM | Record-count output covers only selected YTD records and does not expose excluded/invalid/future records |
| KAI-FINANCEX-038 | MEDIUM | Summary calculations do not report data completeness or invalid-record counts |
| KAI-FINANCEX-039 | MEDIUM | `MILEAGE_RATE` is configured but unused by every endpoint and calculation |
| KAI-FINANCEX-040 | MEDIUM | MTD alerts use manually recorded gross CIS receipts without qualifying-income/completeness evidence |
| KAI-FINANCEX-041 | MEDIUM | Invoice references and dates have no response-level validation or generated-item uniqueness proof |
| KAI-FINANCEX-042 | MEDIUM | Plain-text invoice rendering has no canonical escaping, line-length or layout contract |
| KAI-FINANCEX-043 | MEDIUM | Endpoints define no strict response models or financial API-schema version |
| KAI-FINANCEX-044 | MEDIUM | Results expose no calculation/rule-pack hash for later reproduction |
| KAI-FINANCEX-045 | MEDIUM | The full summary repeats complete file parsing and financial calculations rather than using one immutable snapshot |
| KAI-FINANCEX-046 | MEDIUM | No service-wide rate limit, caller quota or calculation workload admission exists |
| KAI-FINANCEX-047 | MEDIUM | The service exposes no metrics for invalid records, rule age, calculation failures or ledger size |
| KAI-FINANCEX-048 | MEDIUM | Startup creates the finance directory but does not verify owner, symlink, mode, free space or encryption properties |
| KAI-FINANCEX-049 | MEDIUM | pip, setuptools and wheel are upgraded without version pins during the image build |
| KAI-FINANCEX-050 | MEDIUM | FastAPI dependencies and the Python base image are not reproducibly digest-pinned |
| KAI-FINANCEX-051 | MEDIUM | No dedicated Financial Awareness test suite was found for date boundaries, non-finite values or mixed snapshots |
| KAI-FINANCEX-052 | MEDIUM | The service has no authoritative external tax-rule verification or rule-expiry readiness state |
| KAI-FINANCEX-053 | MEDIUM | The lifespan has no graceful write drain, snapshot lock or ledger reconciliation phase |
| KAI-FINANCEX-054 | MEDIUM | Financial outputs have no calculation ID linking exact inputs, rules and result values |

---

## High-severity findings

### KAI-FINANCEX-001 — HIGH — Incorrect tax-year boundary on recurring dates
**Issue:** The condition is `today.month >= 4 and today.day >= 6`. For 1–5 May, June, July, August, September, October, November and December it selects 6 April of the previous year.  
**Risk:** Those dates can include approximately an extra year of receipts/deductions in “current YTD” tax and CIS calculations.  
**Recommendation:** Compare `(month, day)` with `(4, 6)` and test every boundary date.  
**Status:** OPEN

### KAI-FINANCEX-002 — HIGH — Future records enter YTD
All tax-year helpers check only `record_date >= tax_year_start`; no `record_date <= today` or tax-year-end bound exists.

### KAI-FINANCEX-003 — HIGH — Future records enter VAT turnover
The rolling filter checks only `date >= lookback_start`.

### KAI-FINANCEX-004 — HIGH — Future deductions suppress current liability
Future-dated deduction amounts are subtracted from current estimated tax/NI.

### KAI-FINANCEX-005 — HIGH — Rolling-window rule is hard-coded to 365 days
The implementation has no effective-date/rule definition explaining the supported accounting period.

### KAI-FINANCEX-006 — HIGH — Non-finite amounts pass validation
For NaN, `v < 0` is false; positive infinity also passes.

### KAI-FINANCEX-007 — HIGH — Invalid JSON-number persistence
`json.dumps` defaults to `allow_nan=True`, writing tokens outside strict JSON financial interchange.

### KAI-FINANCEX-008 — HIGH — Non-finite arithmetic propagation
Sums, percentages, threshold distances and returned estimates can become NaN/infinity.

### KAI-FINANCEX-009 — HIGH — File root type is not validated
`_load_records` returns any successfully parsed JSON value.

### KAI-FINANCEX-010 — HIGH — Record item type is not validated
Summary comprehensions call `.get` on every entry.

### KAI-FINANCEX-011 — HIGH — Persisted field schema is not enforced on read
File tampering or older schema values bypass Pydantic ingestion validation.

### KAI-FINANCEX-012 — HIGH — Overpayment information is lost
`max(..., 0.0)` reports zero balance due instead of the negative credit magnitude.

### KAI-FINANCEX-013 — HIGH — Missing financial snapshot chronology
Consumers cannot reproduce which date/rule/data generation produced a result.

### KAI-FINANCEX-014 — HIGH — Global taxpayer namespace
No taxpayer, business, currency or jurisdiction owner appears in persisted records.

### KAI-FINANCEX-015 — HIGH — Unverified deduction status
The caller selects registered/unregistered/gross and thereby the rate applied.

### KAI-FINANCEX-016 — HIGH — CIS rule provenance absent
Rates are timeless constants.

### KAI-FINANCEX-017 — HIGH — No source-document reconciliation
A record is accepted without payment statement/reference/evidence digest.

### KAI-FINANCEX-018 — HIGH — Duplicate ledger poisoning
No idempotency or duplicate-payment key exists.

### KAI-FINANCEX-019 — HIGH — Incorrect records are permanent
Only append and summaries are exposed.

### KAI-FINANCEX-020 — HIGH — Weak distributed identity
A microsecond timestamp is not protected by a unique storage constraint.

### KAI-FINANCEX-021 — HIGH — Implicit GBP
Amounts cannot represent or reject another currency.

### KAI-FINANCEX-022 — HIGH — VAT-inclusive CIS ambiguity
The record has gross/materials/labour but no VAT field, so the calculation cannot know whether the supplied gross includes VAT.

### KAI-FINANCEX-023 — HIGH — Invoice tax model incomplete
The generated invoice cannot represent VAT status/amount or currency.

### KAI-FINANCEX-024 — HIGH — Generated invoices are unaudited transient output
No invoice ledger or link to source records exists.

### KAI-FINANCEX-025 — HIGH — Invoice/log formatting injection
Names, addresses, descriptions and references are interpolated into multiline text and logs without control-character policy.

### KAI-FINANCEX-026 — HIGH — Cacheable financial identifiers
Responses include UTRs, counterparties and exact amounts with ordinary cache semantics.

### KAI-FINANCEX-027 — HIGH — Non-atomic combined snapshot
Each sub-endpoint calls `_load_records` separately.

### KAI-FINANCEX-028 — HIGH — Contradictory combined results
The CIS, VAT and tax sections may describe different ledger versions.

### KAI-FINANCEX-029 — HIGH — Sensitive logging
The success log records contractor and exact gross/deduction/net values.

### KAI-FINANCEX-030 — HIGH — Unnecessary security bundle
The Dockerfile copies `security/` without using it.

### KAI-FINANCEX-031 — HIGH — Compound compromise impact
A vulnerability in the public financial service exposes both the finance volume and repository security material.

### KAI-FINANCEX-032 — HIGH — Missing financial audit chain
No immutable event records actor, source evidence, before/after ledger revision and calculation effects.

---

## Medium-severity findings

### KAI-FINANCEX-033 — MEDIUM — Repeated boundary implementation
The same faulty logic appears in income, deductions and CIS summary.

### KAI-FINANCEX-034 — MEDIUM — Accounting date uses host timezone
No configured locale/timezone controls the date cutover.

### KAI-FINANCEX-035 — MEDIUM — Silent invalid-date exclusion
Stored-but-unused records are not counted or surfaced as errors.

### KAI-FINANCEX-036 — MEDIUM — Summary evidence cannot be inspected
No record-list/detail endpoint supports reconciliation.

### KAI-FINANCEX-037 — MEDIUM — Misleading record count
Excluded data is invisible.

### KAI-FINANCEX-038 — MEDIUM — Completeness unknown
Outputs have no invalid/excluded/source coverage indicators.

### KAI-FINANCEX-039 — MEDIUM — Dead mileage configuration
The advertised financial parameter has no effect.

### KAI-FINANCEX-040 — MEDIUM — Weak MTD evidence
The alert uses one incomplete receipt stream and fixed distance.

### KAI-FINANCEX-041 — MEDIUM — Invoice identity not verified
Response generation does not prove unique numbering/date validity.

### KAI-FINANCEX-042 — MEDIUM — Unversioned text format
Consumers cannot rely on a stable invoice layout.

### KAI-FINANCEX-043 — MEDIUM — Free-form output schemas
No response model or schema version exists.

### KAI-FINANCEX-044 — MEDIUM — Calculation rules not reproducible
Only values, not the exact rule set, are returned.

### KAI-FINANCEX-045 — MEDIUM — Repeated full-ledger work
One summary invokes three complete file loads/calculation passes.

### KAI-FINANCEX-046 — MEDIUM — No workload governance
Public endpoints can be polled/mutated without capacity controls.

### KAI-FINANCEX-047 — MEDIUM — Missing operational metrics
No financial/data-quality telemetry exists.

### KAI-FINANCEX-048 — MEDIUM — Storage properties unverified
Existing paths/volumes are trusted without security checks.

### KAI-FINANCEX-049 — MEDIUM — Mutable build toolchain
Build-time package tools are unpinned.

### KAI-FINANCEX-050 — MEDIUM — Mutable application dependencies
Dependencies/base use ranges/tags.

### KAI-FINANCEX-051 — MEDIUM — Missing source tests
Repository search found no dedicated service test module.

### KAI-FINANCEX-052 — MEDIUM — No rule expiry/readiness
The service never declares its tax rules expired or unsupported.

### KAI-FINANCEX-053 — MEDIUM — Incomplete storage lifecycle
No mutation drain/snapshot reconciliation occurs at shutdown.

### KAI-FINANCEX-054 — MEDIUM — No calculation identity
A returned estimate cannot be linked to exact records and rule revision.

---

## Batch totals

- Findings: **54**
- Critical: **0**
- High: **32**
- Medium: **22**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **3,009**
- Critical: **195**
- High: **1,548**
- Medium: **1,263**
- Low: **3**

## Files materially reviewed

`financial-awareness/app.py`, `financial-awareness/Dockerfile`, `financial-awareness/requirements.txt`, persistent-volume/full-stack deployment, Dashboard consumption and the existing Financial Awareness/persistence audits.
