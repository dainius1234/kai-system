# Kai Code Audit — Repository Quality, Commit and Documentation Automation Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch covers previously unlogged findings in `scripts/quality_gate.py`, `scripts/check_commit_msg.py`, `scripts/auto_changelog.py` and `scripts/sync_docs.py`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-REPODOC-001 | HIGH | The script quality gate scans only top-level `scripts/*.py` files |
| KAI-REPODOC-002 | HIGH | Every `test_*.py` script is excluded from stub, TODO and docstring checking |
| KAI-REPODOC-003 | HIGH | The advertised missing-docstring gate checks only module docstrings |
| KAI-REPODOC-004 | HIGH | Bare `pass`, ellipsis bodies and placeholder return values are not detected as stubs |
| KAI-REPODOC-005 | HIGH | Qualified or aliased `NotImplementedError` raises are not detected |
| KAI-REPODOC-006 | HIGH | TODO/FIXME markers are detected only when they begin a comment |
| KAI-REPODOC-007 | HIGH | Shell scripts and other executable maintenance files are outside the gate |
| KAI-REPODOC-008 | HIGH | A passing quality result proves no importability, execution, test success or security property |
| KAI-REPODOC-009 | MEDIUM | Unreachable/dead-code `raise NotImplementedError` is still treated as an active stub |
| KAI-REPODOC-010 | MEDIUM | Tokenisation errors are suppressed during marker detection |
| KAI-REPODOC-011 | MEDIUM | File read errors abort the gate rather than producing a complete structured failure list |
| KAI-REPODOC-012 | MEDIUM | Complete script files are read and parsed without byte or complexity limits |
| KAI-REPODOC-013 | MEDIUM | The gate executes at module import time rather than behind a `main()` boundary |
| KAI-REPODOC-014 | MEDIUM | The known-stub exemption has no owner, expiry, issue or approval metadata |
| KAI-REPODOC-015 | MEDIUM | Function and class documentation quality is not checked |
| KAI-REPODOC-016 | MEDIUM | Stub detection does not identify empty functions whose only statement is a docstring |
| KAI-REPODOC-017 | MEDIUM | The result contains no source commit, scanned-file manifest or configuration digest |
| KAI-REPODOC-018 | MEDIUM | Successful output overstates broad “script quality” from a narrow syntactic scan |
| KAI-REPODOC-019 | HIGH | Commit-message validation uses prefix matching rather than a complete-line match |
| KAI-REPODOC-020 | HIGH | Commit scopes accept arbitrary text and unbalanced or misleading parentheses |
| KAI-REPODOC-021 | HIGH | Commit-message files and stdin are read completely without a size limit |
| KAI-REPODOC-022 | MEDIUM | Only the first line is validated and the body is ignored |
| KAI-REPODOC-023 | MEDIUM | Merge, revert, squash and fixup commit forms are rejected despite being normal Git operations |
| KAI-REPODOC-024 | MEDIUM | Description length has no upper bound and control/terminal characters are not rejected |
| KAI-REPODOC-025 | MEDIUM | The failing first line is printed verbatim and can inject terminal or CI-log control sequences |
| KAI-REPODOC-026 | MEDIUM | Arbitrary caller-selected files can be read as commit messages |
| KAI-REPODOC-027 | MEDIUM | File encoding and read errors are not handled as controlled validation failures |
| KAI-REPODOC-028 | MEDIUM | Breaking-change, security-impact and issue/sign-off metadata are not validated |
| KAI-REPODOC-029 | MEDIUM | Case-insensitive type matching permits forms different from the documented canonical format |
| KAI-REPODOC-030 | MEDIUM | The validator emits no structured result or repository/source identity |
| KAI-REPODOC-031 | HIGH | Git tag and log subprocesses have no timeout |
| KAI-REPODOC-032 | HIGH | Git return codes and stderr are ignored, so command failure becomes “no commits” success |
| KAI-REPODOC-033 | HIGH | A changelog version with no matching Git tag creates an invalid ref and silently suppresses updates |
| KAI-REPODOC-034 | HIGH | The semver-tag regex accepts partial and malformed version names |
| KAI-REPODOC-035 | HIGH | Tag and changelog-version disagreement is not detected |
| KAI-REPODOC-036 | HIGH | Commit subjects are written into Markdown without escaping or provenance boundaries |
| KAI-REPODOC-037 | HIGH | Missing release references cause an unbounded full-history scan |
| KAI-REPODOC-038 | HIGH | Generated unreleased entries are newest-first rather than chronological |
| KAI-REPODOC-039 | HIGH | Existing manually curated `[Unreleased]` content is replaced rather than merged |
| KAI-REPODOC-040 | HIGH | Changelog replacement is a non-atomic complete-file rewrite |
| KAI-REPODOC-041 | HIGH | Concurrent changelog runs can overwrite one another |
| KAI-REPODOC-042 | HIGH | The changelog section parser is a fragile regular expression rather than a Markdown structure |
| KAI-REPODOC-043 | HIGH | Git failure and a genuinely empty release interval have identical successful outcomes |
| KAI-REPODOC-044 | MEDIUM | Merge commits are omitted from release history |
| KAI-REPODOC-045 | MEDIUM | Test commits are categorised as product features under `Added` |
| KAI-REPODOC-046 | MEDIUM | Unknown or malformed commit types are silently classified as `Changed` |
| KAI-REPODOC-047 | MEDIUM | Commit scopes are discarded, losing component ownership and impact context |
| KAI-REPODOC-048 | MEDIUM | Breaking changes and security fixes receive no dedicated classification |
| KAI-REPODOC-049 | MEDIUM | Duplicate or reverted commit subjects are not reconciled |
| KAI-REPODOC-050 | MEDIUM | Dry-run output prints complete commit subjects to stdout |
| KAI-REPODOC-051 | MEDIUM | Changelog reads and writes rely on platform-default encoding |
| KAI-REPODOC-052 | MEDIUM | No backup, fsync, file lock or compare-and-swap protects the update |
| KAI-REPODOC-053 | MEDIUM | The output contains no source SHA, tag digest or generated-entry provenance |
| KAI-REPODOC-054 | MEDIUM | Check mode compares generated text only and does not validate release-link definitions or history completeness |
| KAI-REPODOC-055 | HIGH | Test counting misses `async def test_*` functions |
| KAI-REPODOC-056 | HIGH | Test counting is limited to `scripts/` and `kai-advisor/` filename patterns |
| KAI-REPODOC-057 | HIGH | Counting `def test_` text treats skipped, dead, nested and uncollectable functions as successful test inventory |
| KAI-REPODOC-058 | HIGH | Test-file count does not verify that files parse or are collected by pytest |
| KAI-REPODOC-059 | HIGH | Test-target count parses only dependencies on one Makefile line |
| KAI-REPODOC-060 | HIGH | Documentation hard-codes `Failures | 0` without executing or reading test results |
| KAI-REPODOC-061 | HIGH | Service count is derived from regex-like text scanning rather than parsed Compose models |
| KAI-REPODOC-062 | HIGH | Service definitions across mutually exclusive Compose variants are presented as one Docker-container count |
| KAI-REPODOC-063 | HIGH | README milestone count is derived from the README’s own `DONE` labels |
| KAI-REPODOC-064 | HIGH | Missing Project Status table is treated as successful synchronisation |
| KAI-REPODOC-065 | HIGH | Missing PROJECT_BACKLOG is treated as successful synchronisation |
| KAI-REPODOC-066 | HIGH | README and backlog updates are non-atomic complete-file rewrites without locking |
| KAI-REPODOC-067 | HIGH | Backlog plus-sign patching reuses stale string indices after mutation and can corrupt the row |
| KAI-REPODOC-068 | HIGH | Documentation can claim current status without validating running services, tests or deployment readiness |
| KAI-REPODOC-069 | HIGH | File naming alone determines Compose-file and test-file metrics |
| KAI-REPODOC-070 | MEDIUM | Python LOC includes tests and other non-production Python while excluding files on read error |
| KAI-REPODOC-071 | MEDIUM | LOC and service scans have no manifest, exclusion policy version or reproducibility record |
| KAI-REPODOC-072 | MEDIUM | Raw Compose scanning can count comments, malformed YAML or inactive profile definitions incorrectly |
| KAI-REPODOC-073 | MEDIUM | The “latest commit” subprocess return code is ignored |
| KAI-REPODOC-074 | MEDIUM | The collected commit hash is never written into the synced documentation |
| KAI-REPODOC-075 | MEDIUM | Local host date and platform-specific `%-d` formatting govern README status dates |
| KAI-REPODOC-076 | MEDIUM | Daily date changes make documentation stale even when all measured repository content is unchanged |
| KAI-REPODOC-077 | MEDIUM | Check-mode diff uses `zip()` and omits added or removed trailing rows |
| KAI-REPODOC-078 | MEDIUM | Backlog check returns after the first stale metric and hides additional drift |
| KAI-REPODOC-079 | MEDIUM | README/backlog reads and writes mostly use platform-default encoding |
| KAI-REPODOC-080 | MEDIUM | README table matching is coupled to exact heading and separator formatting |
| KAI-REPODOC-081 | MEDIUM | Makefile absence or format changes raise rather than produce a complete diagnostic report |
| KAI-REPODOC-082 | MEDIUM | Source read failures are inconsistently ignored, raised or converted to zero counts |
| KAI-REPODOC-083 | MEDIUM | “Individual tests” reports source-function count rather than executed passing tests |
| KAI-REPODOC-084 | MEDIUM | Synced documentation has no generated-data signature, source digest or audit history |

---

## Script quality gate — `scripts/quality_gate.py`

### KAI-REPODOC-001 — HIGH — Top-level-only coverage
`SCRIPTS.glob("*.py")` excludes nested script/tool directories and every non-Python executable.

### KAI-REPODOC-002 — HIGH — Test scripts excluded
Files beginning `test_` are skipped entirely, allowing incomplete assurance code to remain invisible to this gate.

### KAI-REPODOC-003 — HIGH — Module-only docstring check
The script description says it scans missing docstrings, but only `ast.get_docstring(mod)` is checked.

### KAI-REPODOC-004 — HIGH — Common stub forms missed
Only a specially commented `pass` and selected NotImplementedError form are detected.

### KAI-REPODOC-005 — HIGH — Qualified exceptions missed
`raise builtins.NotImplementedError`, aliases and dynamically referenced equivalents are not recognised.

### KAI-REPODOC-006 — HIGH — Narrow marker syntax
Comments such as `# temporary — TODO ...` and other embedded markers pass.

### KAI-REPODOC-007 — HIGH — Non-Python assurance gap
Shell setup, backup and operational scripts are not scanned.

### KAI-REPODOC-008 — HIGH — Syntax gate presented as quality
No imports, subprocesses, tests, endpoint contracts or security controls are executed.

### KAI-REPODOC-009 — MEDIUM — Reachability ignored
AST walking flags a NotImplementedError even inside unreachable branches or illustrative nested code.

### KAI-REPODOC-010 — MEDIUM — Tokenisation failure hides markers
`TokenizeError` returns false from `_has_marker_comment()`.

### KAI-REPODOC-011 — MEDIUM — Incomplete error collection
A filesystem read error terminates the whole scan outside the failures list.

### KAI-REPODOC-012 — MEDIUM — Unbounded static analysis
Complete file contents and token streams are materialised.

### KAI-REPODOC-013 — MEDIUM — Import side effect
The scan and `sys.exit()` execute when the module is imported.

### KAI-REPODOC-014 — MEDIUM — Exemption governance absent
`KNOWN_STUBS` carries only free-text reasons and no expiry or evidence.

### KAI-REPODOC-015 — MEDIUM — Documentation completeness absent
Functions/classes and public interfaces can remain undocumented.

### KAI-REPODOC-016 — MEDIUM — Empty documented bodies pass
A function containing only a docstring is not classified as incomplete.

### KAI-REPODOC-017 — MEDIUM — No reproducible scan evidence
The result cannot establish which files/revision were scanned.

### KAI-REPODOC-018 — MEDIUM — Overstated success string
“All scripts pass quality gate” exceeds the actual scope and checks.

---

## Commit-message gate — `scripts/check_commit_msg.py`

### KAI-REPODOC-019 — HIGH — Prefix-only validation
`PATTERN.match()` is not anchored with `$` or `fullmatch()`.

### KAI-REPODOC-020 — HIGH — Unrestricted scope grammar
`(.+)` accepts arbitrary characters and nested/unbalanced formatting.

### KAI-REPODOC-021 — HIGH — Unbounded input read
Both file and stdin paths materialise the complete message.

### KAI-REPODOC-022 — MEDIUM — Body ignored
Only the first stripped line is examined.

### KAI-REPODOC-023 — MEDIUM — Normal Git-generated forms rejected
Merge/revert/fixup/squash operations lack an allowed policy path.

### KAI-REPODOC-024 — MEDIUM — No output-safety/length contract
Control characters and excessively long lines are accepted.

### KAI-REPODOC-025 — MEDIUM — Raw terminal output
The rejected line is printed directly.

### KAI-REPODOC-026 — MEDIUM — Arbitrary file read
The CLI accepts any readable path.

### KAI-REPODOC-027 — MEDIUM — Unhandled decoding/filesystem errors
Failures become tracebacks rather than a controlled result.

### KAI-REPODOC-028 — MEDIUM — Governance metadata absent
Security/breaking impact, issue references and sign-off are not checked.

### KAI-REPODOC-029 — MEDIUM — Canonical-format drift
Uppercase/mixed-case types pass despite lowercase documentation.

### KAI-REPODOC-030 — MEDIUM — No attested result
Only an exit code/terminal text is produced.

---

## Changelog automation — `scripts/auto_changelog.py`

### KAI-REPODOC-031 — HIGH — No Git deadline
Tag/log operations may hang indefinitely.

### KAI-REPODOC-032 — HIGH — Git failure is silent
Return code and stderr are ignored.

### KAI-REPODOC-033 — HIGH — Constructed nonexistent ref
A changelog version becomes `v<version>` even when that tag does not exist.

### KAI-REPODOC-034 — HIGH — Partial semver matching
The tag regex does not require a complete valid semantic version.

### KAI-REPODOC-035 — HIGH — Release-authority disagreement hidden
The script simply prefers a tag over changelog state.

### KAI-REPODOC-036 — HIGH — Markdown injection
Commit subjects become list entries verbatim.

### KAI-REPODOC-037 — HIGH — Full-history amplification
With no reference, all non-merge commits are loaded.

### KAI-REPODOC-038 — HIGH — Reverse release chronology
Git log output is not reversed.

### KAI-REPODOC-039 — HIGH — Manual release notes lost
The entire current Unreleased block is replaced.

### KAI-REPODOC-040 — HIGH — Non-atomic write
`write_text()` replaces the tracked file directly.

### KAI-REPODOC-041 — HIGH — Concurrent lost updates
No lock or expected-file digest exists.

### KAI-REPODOC-042 — HIGH — Fragile section boundaries
Markdown is parsed with one regex tied to exact heading syntax.

### KAI-REPODOC-043 — HIGH — False successful no-op
A failed log query returns an empty list and exits zero.

### KAI-REPODOC-044 — MEDIUM — Merge history omitted
Relevant release integrations may be absent.

### KAI-REPODOC-045 — MEDIUM — Test commits presented as Added features
The type map conflates test coverage with shipped functionality.

### KAI-REPODOC-046 — MEDIUM — Unknown type laundering
Malformed subjects silently enter Changed.

### KAI-REPODOC-047 — MEDIUM — Component scope discarded
Only the description is retained.

### KAI-REPODOC-048 — MEDIUM — Security/breaking semantics lost
No dedicated sections or markers exist.

### KAI-REPODOC-049 — MEDIUM — No history reconciliation
Reverts, duplicates and superseded entries remain.

### KAI-REPODOC-050 — MEDIUM — Dry-run disclosure
Subjects are printed in full.

### KAI-REPODOC-051 — MEDIUM — Platform encoding
Tracked Markdown reads/writes omit explicit encoding.

### KAI-REPODOC-052 — MEDIUM — Weak durability
No backup/fsync/atomic replacement is used.

### KAI-REPODOC-053 — MEDIUM — Missing provenance
Generated entries do not identify commit hashes or source range.

### KAI-REPODOC-054 — MEDIUM — Narrow staleness definition
Text equality alone does not prove release-history correctness.

---

## Documentation synchronisation — `scripts/sync_docs.py`

### KAI-REPODOC-055 — HIGH — Async tests omitted
The regex matches `def test_` but not `async def test_`.

### KAI-REPODOC-056 — HIGH — Partial test inventory
Only two filename roots/patterns are included.

### KAI-REPODOC-057 — HIGH — Collection is not measured
Source text, not pytest collection or execution, defines test count.

### KAI-REPODOC-058 — HIGH — Broken tests still count
Files need not import, parse or run.

### KAI-REPODOC-059 — HIGH — Make target parser is incomplete
Multiline/continued/generated Make dependencies are not handled.

### KAI-REPODOC-060 — HIGH — Fabricated zero failures
The README row is constant.

### KAI-REPODOC-061 — HIGH — Compose is not parsed
Indentation/string scanning substitutes for YAML/profile resolution.

### KAI-REPODOC-062 — HIGH — Variant union mislabeled deployment
Definitions across minimal/full/sovereign files are counted once and labelled Docker containers.

### KAI-REPODOC-063 — HIGH — Self-certifying milestones
README DONE text creates the milestone metric written back into README.

### KAI-REPODOC-064 — HIGH — Missing status table passes
The checker warns but returns true.

### KAI-REPODOC-065 — HIGH — Missing backlog passes
Absence returns true.

### KAI-REPODOC-066 — HIGH — Unsafe tracked-file mutation
Both documents are complete-file rewrites without locks/atomic replacement.

### KAI-REPODOC-067 — HIGH — Stale-index row mutation
The plus-sign correction references indices captured before modifying the text.

### KAI-REPODOC-068 — HIGH — Static counts become operational claims
No Docker/test/health execution backs the updated status.

### KAI-REPODOC-069 — HIGH — Naming is treated as evidence
Glob patterns define what counts as a test or Compose authority.

### KAI-REPODOC-070 — MEDIUM — LOC scope ambiguity
Production, test and utility code are mixed; unreadable files disappear.

### KAI-REPODOC-071 — MEDIUM — No reproducibility manifest
The included/excluded path set is not recorded.

### KAI-REPODOC-072 — MEDIUM — Text scanner miscounts Compose
Comments, inactive profiles and malformed blocks can affect counts.

### KAI-REPODOC-073 — MEDIUM — Git errors ignored
A failed rev-parse can yield an empty string.

### KAI-REPODOC-074 — MEDIUM — Commit metric is dead
The value is collected but omitted from the table/report.

### KAI-REPODOC-075 — MEDIUM — Host-dependent date
Timezone and platform formatting affect output.

### KAI-REPODOC-076 — MEDIUM — Daily churn
Date equality is part of currentness.

### KAI-REPODOC-077 — MEDIUM — Incomplete check diff
`zip()` ignores unequal tails.

### KAI-REPODOC-078 — MEDIUM — Partial backlog diagnosis
The first mismatch returns immediately.

### KAI-REPODOC-079 — MEDIUM — Inconsistent encoding
Most tracked-file operations omit UTF-8 explicitly.

### KAI-REPODOC-080 — MEDIUM — Format-coupled matching
Minor Markdown edits disable synchronisation.

### KAI-REPODOC-081 — MEDIUM — Scanner exceptions are not consolidated
Makefile/path errors can abort the script.

### KAI-REPODOC-082 — MEDIUM — Inconsistent error semantics
Some read failures vanish, others fail the run.

### KAI-REPODOC-083 — MEDIUM — Misleading test label
Function count is reported as individual tests.

### KAI-REPODOC-084 — MEDIUM — Unattested generated documentation
No signature, source digest or historical generated-data record exists.

---

## Batch totals

- Findings: **84**
- Critical: **0**
- High: **39**
- Medium: **45**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,268**
- Critical: **181**
- High: **1,120**
- Medium: **964**
- Low: **3**

## Files materially reviewed

`scripts/quality_gate.py`, `scripts/check_commit_msg.py`, `scripts/auto_changelog.py`, `scripts/sync_docs.py`, with repository layout, Makefile, Compose and generated-document semantics checked through source references.
