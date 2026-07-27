# Kai Code Audit — Business, Tax and RAMS Advisor Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/invoice.py`, `scripts/hse_rams.py`, `scripts/deduct_advisor.py` and `common/self_emp_advisor.py`.

External-rule verification used current official UK guidance available on the review date, including GOV.UK guidance titled **When to register for VAT**, **Invoices — what they must include**, **Charging VAT**, **Making Tax Digital for Income Tax**, **Penalties for Making Tax Digital for Income Tax**, and HMRC Tax Confident guidance for self-employed drivers.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-BIZSAFE-001 | HIGH | Invoice generation defaults to the obsolete £85,000 VAT-registration threshold |
| KAI-BIZSAFE-002 | HIGH | VAT status is inferred from one generic `income` value rather than taxable turnover over the statutory period |
| KAI-BIZSAFE-003 | HIGH | The code ignores the expected-next-30-days VAT-registration test |
| KAI-BIZSAFE-004 | HIGH | VAT is added or omitted without checking whether the business is actually VAT registered |
| KAI-BIZSAFE-005 | HIGH | A VAT-registered business below the supplied threshold is told not to add VAT |
| KAI-BIZSAFE-006 | HIGH | Voluntary VAT registration and other registration circumstances are ignored |
| KAI-BIZSAFE-007 | HIGH | The default `.pdf` output is plain UTF-8 text, not a PDF |
| KAI-BIZSAFE-008 | HIGH | The generated “invoice” omits mandatory invoice identity, supplier, customer, date, description and total fields |
| KAI-BIZSAFE-009 | HIGH | VAT invoices omit VAT number, VAT rate, net amount, VAT amount and gross amount |
| KAI-BIZSAFE-010 | HIGH | Currency calculations use binary floating-point values |
| KAI-BIZSAFE-011 | HIGH | NaN and infinity income/threshold values are accepted |
| KAI-BIZSAFE-012 | HIGH | Negative income and threshold values are accepted |
| KAI-BIZSAFE-013 | HIGH | A caller-controlled threshold can force or suppress VAT advice |
| KAI-BIZSAFE-014 | HIGH | Output may target any filesystem path accessible to the caller |
| KAI-BIZSAFE-015 | HIGH | Existing files and symlink targets are overwritten without confirmation |
| KAI-BIZSAFE-016 | MEDIUM | Invoice writes are non-atomic and lack file locking or backup |
| KAI-BIZSAFE-017 | MEDIUM | Output permissions and ownership are not explicitly restricted |
| KAI-BIZSAFE-018 | MEDIUM | No invoice number or duplicate-number authority exists |
| KAI-BIZSAFE-019 | MEDIUM | No invoice date or supply date is recorded |
| KAI-BIZSAFE-020 | MEDIUM | No customer, supplier or legal-address validation exists |
| KAI-BIZSAFE-021 | MEDIUM | No line item, quantity, unit price, rate or total calculation exists |
| KAI-BIZSAFE-022 | MEDIUM | No currency code or rounding policy is recorded |
| KAI-BIZSAFE-023 | MEDIUM | Rules and thresholds carry no as-of date or official source revision |
| KAI-BIZSAFE-024 | MEDIUM | Invalid environment floats fail with uncontrolled startup exceptions |
| KAI-BIZSAFE-025 | MEDIUM | Parent directories are created recursively without an approved output root |
| KAI-BIZSAFE-026 | MEDIUM | No checksum, immutable copy or accounting-system record is generated |
| KAI-BIZSAFE-027 | MEDIUM | The tool does not track VAT effective-registration date |
| KAI-BIZSAFE-028 | MEDIUM | Exempt, zero-rated and reduced-rate supplies are not represented |
| KAI-BIZSAFE-029 | MEDIUM | The success message claims an invoice was generated without validating its legal or file format |
| KAI-BIZSAFE-030 | MEDIUM | Generation leaves no actor, input or ruleset audit record |
| KAI-BIZSAFE-031 | HIGH | RAMS documents are automatically labelled `APPROVED FOR USE` |
| KAI-BIZSAFE-032 | HIGH | Approval status is applied without an authenticated approver or recorded approval event |
| KAI-BIZSAFE-033 | HIGH | Blank operative signature rows coexist with the approved status |
| KAI-BIZSAFE-034 | HIGH | Document issue is hard-coded to 1.0 for every generation |
| KAI-BIZSAFE-035 | HIGH | Date-only document references collide for multiple RAMS generated on the same day |
| KAI-BIZSAFE-036 | HIGH | No revision history, supersession, distribution or withdrawal control exists |
| KAI-BIZSAFE-037 | HIGH | Review date is automatically set to 90 days without task, change or legal review criteria |
| KAI-BIZSAFE-038 | HIGH | Severity and likelihood values are not constrained to the displayed 1–5 matrix |
| KAI-BIZSAFE-039 | HIGH | Negative or zero risk values are classified as Very Low |
| KAI-BIZSAFE-040 | HIGH | Very large values are silently collapsed into the same Very High label |
| KAI-BIZSAFE-041 | HIGH | Residual risk may exceed initial risk without a warning or rejection |
| KAI-BIZSAFE-042 | HIGH | Very High residual risks do not block approved-document generation |
| KAI-BIZSAFE-043 | HIGH | Empty hazard, activity, persons-at-risk, controls, PPE and responsible fields are accepted |
| KAI-BIZSAFE-044 | HIGH | Missing numeric CSV columns silently receive generic risk defaults |
| KAI-BIZSAFE-045 | HIGH | The generated document contains no enforceable residual-risk acceptance rule |
| KAI-BIZSAFE-046 | HIGH | The claimed method statement contains no work sequence, hold points or controlled method steps |
| KAI-BIZSAFE-047 | HIGH | Permits, isolations, temporary works, plant interfaces and emergency arrangements are not modelled |
| KAI-BIZSAFE-048 | HIGH | Competence, supervision, training and authorised-person requirements are not validated |
| KAI-BIZSAFE-049 | HIGH | COSHH, environmental, welfare and rescue controls are not structurally required |
| KAI-BIZSAFE-050 | HIGH | The scope paragraph is generic boilerplate rather than site/task-specific evidence |
| KAI-BIZSAFE-051 | HIGH | Project, site and preparer identity are free text with no authority or source record |
| KAI-BIZSAFE-052 | HIGH | CSV input size, row count and cell length are unbounded |
| KAI-BIZSAFE-053 | HIGH | Arbitrary input and output paths are accepted |
| KAI-BIZSAFE-054 | HIGH | Existing RAMS and symlink targets can be overwritten without revision checks |
| KAI-BIZSAFE-055 | HIGH | Document writes are non-atomic and can leave a corrupt approved file |
| KAI-BIZSAFE-056 | MEDIUM | Invalid integer fields fail the whole run without row/column diagnostics |
| KAI-BIZSAFE-057 | MEDIUM | CSV headers and schema are not validated before per-row defaults are applied |
| KAI-BIZSAFE-058 | MEDIUM | Embedded control characters can break DOCX XML generation |
| KAI-BIZSAFE-059 | MEDIUM | Generated documents contain no input-file checksum or source-data manifest |
| KAI-BIZSAFE-060 | MEDIUM | Generated documents contain no template/tool version metadata |
| KAI-BIZSAFE-061 | MEDIUM | File permissions and ownership are not explicitly set |
| KAI-BIZSAFE-062 | MEDIUM | Host-local date determines issue/review references |
| KAI-BIZSAFE-063 | MEDIUM | No independent validation confirms the resulting DOCX can be opened or rendered |
| KAI-BIZSAFE-064 | MEDIUM | No PDF, approval watermark or signed release artefact is produced |
| KAI-BIZSAFE-065 | MEDIUM | The 14-column table has no page-orientation, column-width or readability validation |
| KAI-BIZSAFE-066 | MEDIUM | The sign-off section is fixed to ten operatives and has no controlled continuation mechanism |
| KAI-BIZSAFE-067 | MEDIUM | The review-date function accepts arbitrary unvalidated text when called programmatically |
| KAI-BIZSAFE-068 | MEDIUM | The CLI does not expose review-date or revision controls despite the function accepting review date |
| KAI-BIZSAFE-069 | MEDIUM | No generated-document or approval audit trail is written |
| KAI-BIZSAFE-070 | MEDIUM | The source claim that output is suitable for HSE submission is not backed by a compliance validation routine |
| KAI-BIZSAFE-071 | MEDIUM | Duplicate activities/hazards are not detected or consolidated |
| KAI-BIZSAFE-072 | MEDIUM | Control effectiveness is not linked to the residual scores supplied by the same row |
| KAI-BIZSAFE-073 | MEDIUM | The risk matrix has no documented definitions for severity and likelihood values |
| KAI-BIZSAFE-074 | MEDIUM | Prepared-by status defaults to the generic `Site Manager` |
| KAI-BIZSAFE-075 | MEDIUM | The tool has no change-trigger review when methods, plant, people or conditions change |
| KAI-BIZSAFE-076 | HIGH | The deduction advisor executes immediately at import time |
| KAI-BIZSAFE-077 | HIGH | Accounting paths are hard-coded to one `/data/self-emp/Accounting` layout |
| KAI-BIZSAFE-078 | HIGH | Missing income and expense files are silently treated as zero and empty data |
| KAI-BIZSAFE-079 | HIGH | Missing data can produce “No major tax timing risks detected” |
| KAI-BIZSAFE-080 | HIGH | Invalid monetary values silently become £0 |
| KAI-BIZSAFE-081 | HIGH | NaN and infinity values are accepted by `_to_float` |
| KAI-BIZSAFE-082 | HIGH | Negative income/expense values are accepted without accounting semantics |
| KAI-BIZSAFE-083 | HIGH | All income rows are summed without tax-year or rolling-period filtering |
| KAI-BIZSAFE-084 | HIGH | Duplicate income rows are not detected |
| KAI-BIZSAFE-085 | HIGH | Currency amounts use binary floats rather than decimal accounting arithmetic |
| KAI-BIZSAFE-086 | HIGH | VAT advice uses the obsolete £85,000 threshold |
| KAI-BIZSAFE-087 | HIGH | VAT advice ignores the rolling 12-month and expected-next-30-days statutory tests |
| KAI-BIZSAFE-088 | HIGH | VAT registration status, effective date and voluntary registration are ignored |
| KAI-BIZSAFE-089 | HIGH | MTD eligibility is inferred from one undated aggregate rather than prior-tax-year qualifying income |
| KAI-BIZSAFE-090 | HIGH | Staged MTD thresholds and start years are not represented |
| KAI-BIZSAFE-091 | HIGH | The warning claims a generic £100 MTD penalty, contrary to the current points-based regime |
| KAI-BIZSAFE-092 | HIGH | The first MTD year’s quarterly-update penalty holiday is not represented |
| KAI-BIZSAFE-093 | HIGH | MTD exemptions and HMRC confirmation/sign-up conditions are ignored |
| KAI-BIZSAFE-094 | HIGH | Laptop-purchase timing advice is triggered by one keyword and a crude income band |
| KAI-BIZSAFE-095 | HIGH | Laptop advice ignores accounting basis, business-use proportion, allowances and purchase date |
| KAI-BIZSAFE-096 | HIGH | “Wait till April” is emitted without checking the current date or tax-year position |
| KAI-BIZSAFE-097 | HIGH | Any expense line containing `car` and `300` triggers a fixed £300 scenario |
| KAI-BIZSAFE-098 | HIGH | A vehicle service cost is incorrectly converted into a mileage-method deduction |
| KAI-BIZSAFE-099 | HIGH | The configured 45p mileage rate is stale after 6 April 2026 for the first 10,000 business miles |
| KAI-BIZSAFE-100 | HIGH | Business miles and the first-10,000-mile boundary are never collected |
| KAI-BIZSAFE-101 | HIGH | The mutual exclusivity of simplified mileage and actual running costs is ignored |
| KAI-BIZSAFE-102 | HIGH | Fabricated/stale market-cache data can directly trigger fuel-purchase advice |
| KAI-BIZSAFE-103 | HIGH | Petrol trend is classified from a plus sign or the substring `up` |
| KAI-BIZSAFE-104 | HIGH | “Fill up today” is recommended without location, vehicle, quantity, price or source freshness |
| KAI-BIZSAFE-105 | MEDIUM | Expense logs are read completely without a size or line-count limit |
| KAI-BIZSAFE-106 | MEDIUM | Sensitive accounting-derived advice is printed directly to terminal output |
| KAI-BIZSAFE-107 | MEDIUM | No adviser disclaimer, escalation or uncertainty classification accompanies high-stakes advice |
| KAI-BIZSAFE-108 | MEDIUM | No official source citations or as-of date appear in the generated advice |
| KAI-BIZSAFE-109 | MEDIUM | Environment thresholds accept zero, negative and non-finite values |
| KAI-BIZSAFE-110 | MEDIUM | Invalid environment numbers silently become zero rather than failing configuration |
| KAI-BIZSAFE-111 | MEDIUM | No business entity, residency, VAT scheme or accounting period is represented |
| KAI-BIZSAFE-112 | MEDIUM | Expense amounts and dates are not parsed; only text substrings are inspected |
| KAI-BIZSAFE-113 | MEDIUM | Missing/incorrect CSV headers silently produce zero income |
| KAI-BIZSAFE-114 | MEDIUM | Credits, parentheses, multiple currencies and locale formats are not handled |
| KAI-BIZSAFE-115 | MEDIUM | Advice has no structured output, operation ID or ruleset version |
| KAI-BIZSAFE-116 | MEDIUM | The CLI always exits successfully when misleading fallback advice is produced |
| KAI-BIZSAFE-117 | MEDIUM | No record shows who requested or received the financial advice |
| KAI-BIZSAFE-118 | MEDIUM | Advice is not linked to the exact input-file checksums or row set |
| KAI-BIZSAFE-119 | MEDIUM | Threshold changes require manual environment/code updates with no freshness check |
| KAI-BIZSAFE-120 | MEDIUM | Expense lines are lowercased and stripped, losing original evidential formatting without retaining provenance |

---

## Invoice helper — `scripts/invoice.py`

### KAI-BIZSAFE-001 — HIGH — Stale VAT threshold
Current official VAT registration guidance uses £90,000 taxable turnover; the default remains £85,000.

### KAI-BIZSAFE-002 — HIGH — Wrong measurement basis
One generic amount is compared directly with the threshold.

### KAI-BIZSAFE-003 — HIGH — Future-turnover test absent
The statutory expected-30-day route is not modelled.

### KAI-BIZSAFE-004 — HIGH — Registration state absent
Threshold crossing alone controls `vat_applied`.

### KAI-BIZSAFE-005 — HIGH — Registered-business false advice
A registered business below threshold would be told not to add VAT.

### KAI-BIZSAFE-006 — HIGH — Voluntary/special registration absent
Below-threshold registration is a valid state.

### KAI-BIZSAFE-007 — HIGH — Fake PDF
The file contains two plaintext lines.

### KAI-BIZSAFE-008 — HIGH — Invalid invoice content
Official invoice minimum fields are omitted.

### KAI-BIZSAFE-009 — HIGH — Invalid VAT invoice content
VAT-specific fields/calculation are absent.

### KAI-BIZSAFE-010 — HIGH — Float money
Accounting amounts use `float`.

### KAI-BIZSAFE-011 — HIGH — Non-finite money
NaN/infinity pass comparisons and formatting.

### KAI-BIZSAFE-012 — HIGH — Negative values
No nonnegative constraint.

### KAI-BIZSAFE-013 — HIGH — User-defined law
The threshold is freely supplied.

### KAI-BIZSAFE-014 — HIGH — Arbitrary write path
No approved invoice directory.

### KAI-BIZSAFE-015 — HIGH — Destructive overwrite
No exclusive-create or symlink protection.

### KAI-BIZSAFE-016 — MEDIUM — Unsafe durability
Direct write only.

### KAI-BIZSAFE-017 — MEDIUM — File protection absent
Umask controls access.

### KAI-BIZSAFE-018 — MEDIUM — No invoice sequence
Uniqueness cannot be established.

### KAI-BIZSAFE-019 — MEDIUM — Dates absent
Neither issue nor supply date.

### KAI-BIZSAFE-020 — MEDIUM — Legal identities absent
No parties/addresses.

### KAI-BIZSAFE-021 — MEDIUM — Commercial detail absent
No goods/services calculation.

### KAI-BIZSAFE-022 — MEDIUM — Currency semantics absent
No code/rounding.

### KAI-BIZSAFE-023 — MEDIUM — Ruleset freshness absent
No as-of/source.

### KAI-BIZSAFE-024 — MEDIUM — Configuration parse crash
Invalid floats raise before controlled output.

### KAI-BIZSAFE-025 — MEDIUM — Recursive arbitrary directories
Parent path is created.

### KAI-BIZSAFE-026 — MEDIUM — Accounting evidence absent
No immutable record/checksum.

### KAI-BIZSAFE-027 — MEDIUM — Effective date absent
Registration timing cannot be applied.

### KAI-BIZSAFE-028 — MEDIUM — Supply types absent
VAT treatment is Boolean only.

### KAI-BIZSAFE-029 — MEDIUM — Misleading success
Legal/file validity is not checked.

### KAI-BIZSAFE-030 — MEDIUM — Audit absent
No actor/input/output event.

---

## RAMS generator — `scripts/hse_rams.py`

### KAI-BIZSAFE-031 — HIGH — Self-approval
Every document says APPROVED FOR USE.

### KAI-BIZSAFE-032 — HIGH — Approval identity absent
Prepared-by text is not an approval.

### KAI-BIZSAFE-033 — HIGH — Sign-off contradiction
Blank signature table does not prevent release.

### KAI-BIZSAFE-034 — HIGH — Fixed issue
No revision increment.

### KAI-BIZSAFE-035 — HIGH — Document-ref collision
Date-only identity.

### KAI-BIZSAFE-036 — HIGH — Change control absent
No supersession/distribution history.

### KAI-BIZSAFE-037 — HIGH — Arbitrary review policy
Fixed 90-day interval.

### KAI-BIZSAFE-038 — HIGH — Risk range absent
Any integer accepted.

### KAI-BIZSAFE-039 — HIGH — Negative false-low risk
Score ordering maps negatives to Very Low.

### KAI-BIZSAFE-040 — HIGH — Extreme-risk collapse
All scores above 16 end as Very High without invalid-input indication.

### KAI-BIZSAFE-041 — HIGH — Residual regression accepted
No comparison.

### KAI-BIZSAFE-042 — HIGH — High residual release
No stop gate.

### KAI-BIZSAFE-043 — HIGH — Empty safety fields
Text presence is not required.

### KAI-BIZSAFE-044 — HIGH — Silent numeric defaults
Missing fields become 3/3 and 2/1.

### KAI-BIZSAFE-045 — HIGH — Acceptance absent
No risk owner/ALARP/authorisation control.

### KAI-BIZSAFE-046 — HIGH — No method statement
Only risk rows and generic scope exist.

### KAI-BIZSAFE-047 — HIGH — Critical interfaces omitted
No structured permit/plant/emergency controls.

### KAI-BIZSAFE-048 — HIGH — Competence omitted
No verified roles/qualifications.

### KAI-BIZSAFE-049 — HIGH — Other mandatory controls omitted
COSHH/environment/rescue are optional free text at best.

### KAI-BIZSAFE-050 — HIGH — Generic scope
Project name interpolation does not create site specificity.

### KAI-BIZSAFE-051 — HIGH — Unauthenticated authorship
All identity fields are CLI text.

### KAI-BIZSAFE-052 — HIGH — Unbounded document workload
Rows/cells are unlimited.

### KAI-BIZSAFE-053 — HIGH — Arbitrary paths
Input/output unrestricted.

### KAI-BIZSAFE-054 — HIGH — Existing release overwrite
No revision/lock.

### KAI-BIZSAFE-055 — HIGH — Partial approved file
Direct save can fail mid-operation.

### KAI-BIZSAFE-056 — MEDIUM — Weak row diagnostics
Integer conversion exceptions omit controlled row context.

### KAI-BIZSAFE-057 — MEDIUM — Schema validation absent
Headers are optional through `row.get()`.

### KAI-BIZSAFE-058 — MEDIUM — XML input safety
Control characters are not normalised.

### KAI-BIZSAFE-059 — MEDIUM — Source digest absent
Cannot prove source CSV.

### KAI-BIZSAFE-060 — MEDIUM — Tool/template revision absent
Cannot reproduce format/rules.

### KAI-BIZSAFE-061 — MEDIUM — File access not hardened
No explicit mode.

### KAI-BIZSAFE-062 — MEDIUM — Host date
No project timezone/authoritative clock.

### KAI-BIZSAFE-063 — MEDIUM — Output not reopened
No integrity/render smoke test.

### KAI-BIZSAFE-064 — MEDIUM — No controlled release format
Unsigned DOCX only.

### KAI-BIZSAFE-065 — MEDIUM — Readability untested
Dense table layout.

### KAI-BIZSAFE-066 — MEDIUM — Fixed signatory capacity
Ten lines only.

### KAI-BIZSAFE-067 — MEDIUM — Arbitrary programmatic review date
No date parsing.

### KAI-BIZSAFE-068 — MEDIUM — CLI change-control gap
No review/revision flags.

### KAI-BIZSAFE-069 — MEDIUM — Audit absent
No generation/approval event.

### KAI-BIZSAFE-070 — MEDIUM — Compliance claim untested
No compliance checklist/validation.

### KAI-BIZSAFE-071 — MEDIUM — Duplicate rows
No uniqueness check.

### KAI-BIZSAFE-072 — MEDIUM — Controls not causally linked
Residual values are caller assertions.

### KAI-BIZSAFE-073 — MEDIUM — Scale definitions absent
Users cannot calibrate ratings consistently.

### KAI-BIZSAFE-074 — MEDIUM — Generic author default
Prepared by Site Manager.

### KAI-BIZSAFE-075 — MEDIUM — Dynamic review triggers absent
Only a date is shown.

---

## Self-employment advisor — `scripts/deduct_advisor.py`, `common/self_emp_advisor.py`

### KAI-BIZSAFE-076 — HIGH — Import-time advice
Import reads files/cache and prints guidance.

### KAI-BIZSAFE-077 — HIGH — Fixed personal path
No user/business selection.

### KAI-BIZSAFE-078 — HIGH — Missing-data fail-open
Absence becomes zero/empty.

### KAI-BIZSAFE-079 — HIGH — False reassurance
No data can produce no-risk advice.

### KAI-BIZSAFE-080 — HIGH — Invalid amount laundering
Parse failures become zero.

### KAI-BIZSAFE-081 — HIGH — Non-finite arithmetic
Float accepts NaN/Infinity.

### KAI-BIZSAFE-082 — HIGH — Negative semantics absent
Refunds/credits/errors are not distinguished.

### KAI-BIZSAFE-083 — HIGH — No period selection
All rows are summed.

### KAI-BIZSAFE-084 — HIGH — Duplicate detection absent
Repeated rows inflate total.

### KAI-BIZSAFE-085 — HIGH — Float accounting
No Decimal/currency rounding.

### KAI-BIZSAFE-086 — HIGH — Stale VAT threshold
£85,000 instead of current £90,000.

### KAI-BIZSAFE-087 — HIGH — Wrong VAT window
No rolling/future-turnover logic.

### KAI-BIZSAFE-088 — HIGH — Registration state absent
Threshold-only advice.

### KAI-BIZSAFE-089 — HIGH — Wrong MTD evidence period
Current aggregate is not previous-tax-year qualifying income.

### KAI-BIZSAFE-090 — HIGH — Staged mandate absent
Only one threshold.

### KAI-BIZSAFE-091 — HIGH — Incorrect penalty amount/model
Current regime uses penalty points and £200 at threshold, not generic £100.

### KAI-BIZSAFE-092 — HIGH — First-year exception absent
No quarterly-update points in 2026–27.

### KAI-BIZSAFE-093 — HIGH — Exemption/signup context absent
Threshold alone is insufficient.

### KAI-BIZSAFE-094 — HIGH — Keyword tax planning
“laptop” triggers advice.

### KAI-BIZSAFE-095 — HIGH — Accounting context absent
No business use/basis/allowance.

### KAI-BIZSAFE-096 — HIGH — Date-blind April advice
Current date ignored.

### KAI-BIZSAFE-097 — HIGH — Substring vehicle trigger
Any car/300 text matches.

### KAI-BIZSAFE-098 — HIGH — Expense/mileage conflation
A service bill is multiplied by a per-mile rate.

### KAI-BIZSAFE-099 — HIGH — Stale mileage rate
From 6 April 2026 HMRC guidance uses 55p for the first 10,000 car/goods-vehicle miles; code uses 45p.

### KAI-BIZSAFE-100 — HIGH — Mileage inputs absent
No business-mile count/tier.

### KAI-BIZSAFE-101 — HIGH — Double-claim risk
Actual costs versus simplified expenses exclusivity is not enforced.

### KAI-BIZSAFE-102 — HIGH — Fabricated cache integration
Fallback market data is treated as current evidence.

### KAI-BIZSAFE-103 — HIGH — Weak trend parsing
Any plus or `up` substring.

### KAI-BIZSAFE-104 — HIGH — Unsupported purchase instruction
No local/current price evidence.

### KAI-BIZSAFE-105 — MEDIUM — Unbounded expense read
Complete file loaded.

### KAI-BIZSAFE-106 — MEDIUM — Terminal privacy
Advice reveals derived accounting state.

### KAI-BIZSAFE-107 — MEDIUM — No uncertainty/escalation
High-stakes recommendations are direct.

### KAI-BIZSAFE-108 — MEDIUM — No source/as-of display
Users cannot verify rules.

### KAI-BIZSAFE-109 — MEDIUM — Unsafe threshold numbers
No finite/positive validation.

### KAI-BIZSAFE-110 — MEDIUM — Invalid config becomes zero
`_to_float` hides mistakes.

### KAI-BIZSAFE-111 — MEDIUM — Business context absent
Entity/schemes/residency omitted.

### KAI-BIZSAFE-112 — MEDIUM — Expenses not structured
Text search only.

### KAI-BIZSAFE-113 — MEDIUM — Header failure hidden
Missing amount yields zero.

### KAI-BIZSAFE-114 — MEDIUM — Locale/accounting formats incomplete
Limited stripping only.

### KAI-BIZSAFE-115 — MEDIUM — No machine-readable decision record
Terminal strings only.

### KAI-BIZSAFE-116 — MEDIUM — Misleading success exit
Fallback guidance is not an error.

### KAI-BIZSAFE-117 — MEDIUM — Actor absent
No access/advice audit.

### KAI-BIZSAFE-118 — MEDIUM — Input provenance absent
No file hashes/row IDs.

### KAI-BIZSAFE-119 — MEDIUM — Freshness control absent
Rules never expire automatically.

### KAI-BIZSAFE-120 — MEDIUM — Original evidence discarded
Lowercased stripped strings lose formatting/provenance.

---

## Batch totals

- Findings: **120**
- Critical: **0**
- High: **78**
- Medium: **42**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,646**
- Critical: **181**
- High: **1,338**
- Medium: **1,124**
- Low: **3**

## Files materially reviewed

`scripts/invoice.py`, `scripts/hse_rams.py`, `scripts/deduct_advisor.py`, `common/self_emp_advisor.py`, with current UK rule verification against official GOV.UK/HMRC guidance available on 27 July 2026.
