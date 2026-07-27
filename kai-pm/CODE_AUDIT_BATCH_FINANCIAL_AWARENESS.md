# Kai Code Audit — Financial Awareness Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-FINANCE-001 | CRITICAL | Unauthenticated callers can create persistent CIS financial records |
| KAI-FINANCE-002 | CRITICAL | CIS, tax and VAT positions are exposed without authentication |
| KAI-FINANCE-003 | HIGH | Income Tax and NI are calculated from gross receipts rather than taxable profit |
| KAI-FINANCE-004 | HIGH | Hard-coded 2024/25 thresholds and rates are used in a live financial service |
| KAI-FINANCE-005 | HIGH | Personal Allowance taper and other material tax rules are omitted |
| KAI-FINANCE-006 | HIGH | Directive VAT registration guidance is generated from incomplete records and static thresholds |
| KAI-FINANCE-007 | HIGH | Financial records are stored unencrypted in a plain JSON file |
| KAI-FINANCE-008 | HIGH | Read-modify-write persistence is non-atomic and race-prone |
| KAI-FINANCE-009 | MEDIUM | Corrupt record storage is silently treated as an empty ledger |
| KAI-FINANCE-010 | MEDIUM | Monetary calculations use binary floating point |
| KAI-FINANCE-011 | MEDIUM | Dates are not validated when records or invoices are created |
| KAI-FINANCE-012 | MEDIUM | Materials can exceed gross payment without rejection |
| KAI-FINANCE-013 | MEDIUM | Invoice references can collide and are caller-controlled |
| KAI-FINANCE-014 | MEDIUM | Contractor names, addresses, UTRs and work descriptions are unbounded |
| KAI-FINANCE-015 | MEDIUM | Persistence and file reads execute synchronously in async handlers |
| KAI-FINANCE-016 | MEDIUM | Persistence errors and finance-root paths are exposed |
| KAI-FINANCE-017 | MEDIUM | Health reports ok without validating storage integrity or tax-rule currency |
| KAI-FINANCE-018 | MEDIUM | Configuration thresholds and paths are not validated |

---

## Financial awareness: `financial-awareness/app.py`

### KAI-FINANCE-001 — CRITICAL — Unauthenticated persistent financial-record creation
**Issue:** `POST /finance/cis/record` requires no authentication or authorisation and appends caller-supplied contractor, UTR, payment and work information to persistent storage.  
**Risk:** Any reachable caller can poison the user’s financial ledger, inflate income, alter CIS deductions and change every subsequent tax/VAT summary.  
**Recommendation:** Require strong user authentication, immutable audit provenance and validation against source documents before persistence.  
**Status:** OPEN — immediate remediation required

### KAI-FINANCE-002 — CRITICAL — Private financial position is public
**Issue:** `/finance/cis/summary`, `/finance/vat`, `/finance/tax`, `/finance/summary` and `/health` require no authentication. They expose income, deductions, tax estimates, VAT position, record counts and storage location.  
**Risk:** Callers can infer turnover, tax liability, CIS credits and registration status, exposing highly sensitive personal/business financial information.  
**Recommendation:** Require owner-scoped access and minimise returned financial and filesystem metadata.  
**Status:** OPEN — immediate remediation required

### KAI-FINANCE-003 — HIGH — Tax is calculated from gross receipts, not profit
**Issue:** `_income_tax` and `_class4_ni` receive `gross_ytd` from CIS records. Allowable expenses are not deducted, despite the endpoint describing an estimated tax liability and the module docstring referring to self-employment profit.  
**Risk:** Estimates can materially overstate taxable income and tax/NI due, potentially driving incorrect financial decisions.  
**Recommendation:** Model taxable profit explicitly from income less allowable expenses and clearly separate incomplete estimates from filing calculations.  
**Status:** OPEN

### KAI-FINANCE-004 — HIGH — Static tax-year logic is stale by design
**Issue:** Comments and code hard-code 2024/25 Income Tax and Class 4 NI rates, while environment defaults hard-code tax, MTD and VAT thresholds. No tax-year/version selection or effective-date table exists.  
**Risk:** The live service continues producing calculations and warnings after rates and thresholds change, without detecting that its rule set is obsolete.  
**Recommendation:** Version rules by jurisdiction and tax year, require an explicit current rule pack and fail closed when no applicable rules exist.  
**Status:** OPEN

### KAI-FINANCE-005 — HIGH — Material tax rules are omitted
**Issue:** `_income_tax` subtracts the full Personal Allowance at all income levels and applies simple bands. It does not implement Personal Allowance taper above £100,000, Scottish rates, other income, payments on account or other material interactions.  
**Risk:** High-income and non-England/Wales/NI estimates can be substantially wrong while presented as calculated liability.  
**Recommendation:** Define supported scope precisely and implement complete jurisdiction/year rules or label output as a narrow rough illustration.  
**Status:** OPEN

### KAI-FINANCE-006 — HIGH — Directive VAT advice rests on incomplete evidence
**Issue:** `/finance/vat` calculates rolling income only from manually entered CIS records and emits statements such as “MANDATORY REGISTRATION” and “You must register ... within 30 days.” It does not account for all taxable turnover, exceptions, future-look-forward tests or record completeness.  
**Risk:** Users may rely on authoritative-sounding legal/tax guidance that is unsupported by the data model.  
**Recommendation:** Present bounded informational estimates, disclose completeness requirements and direct users to current HMRC/professional confirmation.  
**Status:** OPEN

### KAI-FINANCE-007 — HIGH — Sensitive records are plaintext
**Issue:** Contractor names, UTRs, payment details and work descriptions are stored in `cis_records.json` without encryption, permission hardening or retention controls.  
**Risk:** Any process or user with filesystem access can read complete financial records and identifiers.  
**Recommendation:** Use encrypted transactional storage with restrictive permissions and retention/deletion policy.  
**Status:** OPEN

### KAI-FINANCE-008 — HIGH — Persistence loses concurrent updates
**Issue:** Each record request loads the full JSON list, appends in memory and rewrites the whole file with no lock, temporary file or atomic rename.  
**Risk:** Concurrent requests can overwrite one another; interruption can truncate/corrupt the ledger.  
**Recommendation:** Use transactional database storage or locked atomic writes with fsync and version checks.  
**Status:** OPEN

### KAI-FINANCE-009 — MEDIUM — Corruption becomes an empty ledger
**Issue:** `_load_records` catches every exception and returns `[]`. No error is exposed or recovery/quarantine occurs.  
**Risk:** File corruption or invalid JSON makes all summaries appear as zero income rather than a storage failure, potentially producing false “no tax/VAT issue” conclusions.  
**Recommendation:** Fail visibly, preserve the damaged file and require recovery before calculations.  
**Status:** OPEN

### KAI-FINANCE-010 — MEDIUM — Money uses binary floats
**Issue:** Amounts, rates, thresholds and calculations use `float`, with repeated rounding.  
**Risk:** Binary rounding and cumulative summation can create penny-level discrepancies and unstable exact comparisons.  
**Recommendation:** Use `Decimal` with defined currency precision and rounding policy.  
**Status:** OPEN

### KAI-FINANCE-011 — MEDIUM — Record and invoice dates are not validated
**Issue:** `payment_date` and `invoice_date` are optional strings with no ISO-date validator. Invalid values are persisted/rendered; later summaries silently skip invalid record dates.  
**Risk:** Records can disappear from tax-year calculations while remaining stored, and invoices can contain invalid dates.  
**Recommendation:** Parse and validate dates at ingestion and reject impossible/out-of-scope values.  
**Status:** OPEN

### KAI-FINANCE-012 — MEDIUM — Invalid payment composition is accepted
**Issue:** `materials_amount` may exceed `gross_amount`. Labour is clamped to zero, but materials and gross are stored unchanged.  
**Risk:** Internally inconsistent records distort materials/labour summaries and CIS treatment without warning.  
**Recommendation:** Enforce `materials_amount <= gross_amount` and validate the full accounting identity.  
**Status:** OPEN

### KAI-FINANCE-013 — MEDIUM — Invoice identifiers are weak
**Issue:** Generated invoice references use minute-resolution timestamps; caller-supplied references are accepted without uniqueness, format or length validation.  
**Risk:** Multiple invoices can share identifiers, undermining auditability and document reconciliation.  
**Recommendation:** Use persistent unique sequences/UUIDs and enforce uniqueness.  
**Status:** OPEN

### KAI-FINANCE-014 — MEDIUM — Sensitive text fields are unbounded
**Issue:** Contractor/subcontractor names, addresses, UTRs, descriptions and invoice references have no maximum lengths or UTR format validation.  
**Risk:** Oversized or malformed values consume storage/response/log capacity and create misleading financial documents.  
**Recommendation:** Apply strict field lengths, formats and aggregate request limits.  
**Status:** OPEN

### KAI-FINANCE-015 — MEDIUM — Filesystem work blocks async handlers
**Issue:** Every endpoint synchronously reads/parses the complete JSON file; record creation synchronously serialises and rewrites it.  
**Risk:** Large ledgers block the event-loop worker and degrade every API request.  
**Recommendation:** Use asynchronous transactional storage or bounded worker execution.  
**Status:** OPEN

### KAI-FINANCE-016 — MEDIUM — Internal details are disclosed
**Issue:** Record persistence exceptions are returned directly in HTTP details, and `/health` returns the full finance-root path.  
**Risk:** Callers learn filesystem structure and storage diagnostics.  
**Recommendation:** Return stable error codes and restrict operational metadata.  
**Status:** OPEN

### KAI-FINANCE-017 — MEDIUM — Health does not verify trustworthy operation
**Issue:** `/health` always returns `status: ok`. A corrupt file is counted as zero records because `_load_records` suppresses errors; storage writability and current tax-rule applicability are not checked.  
**Risk:** Orchestration treats a corrupted or obsolete financial calculator as ready.  
**Recommendation:** Separate liveness, storage integrity and rule-pack currency/readiness.  
**Status:** OPEN

### KAI-FINANCE-018 — MEDIUM — Configuration lacks validation
**Issue:** Financial root, port, thresholds and rates are parsed directly. Negative, zero, reversed or nonsensical band values are not rejected.  
**Risk:** Misconfiguration silently produces invalid calculations or startup failure.  
**Recommendation:** Validate ordered monetary thresholds, rate ranges, paths and tax-year metadata at startup.  
**Status:** OPEN

---

## Batch totals

- Findings: **18**
- Critical: **2**
- High: **6**
- Medium: **10**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **596**
- Critical: **67**
- High: **210**
- Medium: **316**
- Low: **3**

## Files materially reviewed in this batch

`financial-awareness/app.py`.
