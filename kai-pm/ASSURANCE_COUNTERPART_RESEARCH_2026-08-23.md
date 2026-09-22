# Assurance counterpart research — 2026-08-23

> **STATUS: SNAPSHOT-BOUND RESEARCH EVIDENCE / KAI→ORION BRIEF — NOT CANONICAL PROGRAMME STATE.**
>
> This file preserves external research and Kai's review so it cannot disappear with a chat thread. It does **not** allocate KAI-GATE IDs, close findings, authorize experiments, or supersede `DECISIONS.md`. Orion must deduplicate/consume it into the repaired canonical machinery at the appropriate milestone.

## 0. Repository state used for this brief

- Last D-numbered programme entry reviewed: **D332**, commit `b14e6f9ce7879c71d288ba1af79cf7c97dea7d03`.
- D332 frozen qualification subject remains `9d15bcd207ad7a33e1087667b245970f989e366f`, tree `627104d61b4f91e110a36cf65a44fed2cfbad078`.
- Kai continuity protocol was then added in governance-only commit `b35639e187daf8fc5ed94c29f5528fad5cddd64f`.
- This research brief is a reporting/governance artefact, not a new qualification subject.

D332 standing accepted: R1B not started; instrument not frozen; Phase B stopped; `ITEM8_GO` absent; Stage 2 and six subject builds unauthorised; P1 stands and P0 remains permanent.

## 1. External counterparts researched

The conclusion is **not** to replace KAI with an external project. The mature pattern is to split concerns and reuse standards below the KAI-specific reasoning layer.

### Provenance / attestations

- **W3C PROV** — interoperable provenance model based on Entity / Activity / Agent, usage, generation, derivation, responsibility and plans. https://www.w3.org/TR/prov-primer/
- **SLSA provenance** — binds produced artifacts to build definition, builder/run details, inputs and source. https://slsa.dev/spec/
- **in-toto / Witness** — pipeline/SDLC attestations and verification; Witness can create in-toto attestations and verify them against policy. https://witness.dev/docs/

**Decision direction:** A-4 should retain KAI's exact evidence semantics but map its underlying provenance graph onto W3C PROV concepts and use in-toto/SLSA-compatible envelopes where useful. Do not retrofit this into frozen Item 8.

### Policy

- **Open Policy Agent / Rego** — declarative policy decisions over structured data, separable from fact collection. https://www.openpolicyagent.org/docs/

**Decision direction:** Python/detectors establish qualified facts; policy logic may later use OPA where it materially simplifies promotion/release/authority rules. Do not move observation/measurement into Rego.

### Findings / remediation

- **NIST OSCAL POA&M** — stable machine-readable representation for risks, deviations, disposition and milestones. https://pages.nist.gov/OSCAL/

**Decision direction:** do not adopt full OSCAL complexity blindly. Use an OSCAL-inspired machine-readable canonical finding schema: stable ID, state, severity, evidence identity, root-cause state, owner, milestone, repair hypothesis, absence-detecting test and closure evidence. Human Markdown becomes a derived view.

### Generic repository assurance

- **OpenSSF Scorecard** — independent checks including Branch Protection, Code Review, Dangerous Workflow, Pinned Dependencies, Token Permissions and Security Policy. https://www.scorecard.dev/
- **GitHub rulesets / protected branches** — can require PRs, status checks, signed commits, etc. https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-rulesets/about-rulesets

**Decision direction:** use external Scorecard later as an independent control, and enforce repository governance through GitHub rulesets only after required CI is truthful/green enough not to lock the project behind known-broken checks.

## 2. New / strengthened repository findings from the counterpart review

These are **research candidates / instances**. Do not allocate new IDs until the repaired register can deduplicate them.

### R-EXT-1 — default branch governance is declared but not enforced — HIGH

GitHub currently reports `main` as unprotected with no required status checks. `.github/CODEOWNERS` exists and says core/safety/infrastructure changes require scrutiny, but repository configuration does not enforce that declaration.

Interpretation: another MISBINDING — declared governance attached to no enforcing mechanism.

Milestone: mandatory before professionalisation/final release. Do **not** enable required red checks prematurely; restore CI truth first, then apply a ruleset/branch protection deliberately.

### R-EXT-2 — mutable GitHub Action identities — HIGH/MEDIUM supply-chain assurance

Representative active workflow references include `actions/checkout@v4`, `actions/setup-python@v5` and especially `aquasecurity/trivy-action@master`.

Interpretation: workflow behaviour can change upstream without this repository tree changing. That conflicts with exact-evidence identity.

Milestone: professionalisation. Mechanically derive all external `uses:` references before repair; pin third-party actions to reviewed immutable commit SHAs and record update policy. Do not fix instances one-by-one without the denominator.

### R-EXT-3 — CI execution environments are not hermetic — HIGH, strengthens existing CI/local-divergence class

Several workflows construct one broad interpreter by installing generic packages plus many service `requirements.txt` files; some installs are best-effort (`|| true`). Per-service constraints can therefore conflict or drift from runtime images.

Interpretation: likely contributor to CI/local divergence and dependency-version failures. It is not a newly proven root cause for every red workflow.

Milestone: CI Truth Restoration / professionalisation. Preferred direction is **per-service locked runtime environments plus a separately specified integration/test environment**, not one magical interpreter containing every service dependency.

### R-EXT-4 — Weekly Report Card can emit global-looking `All green` from a narrow subset — dedupe candidate

`.github/workflows/weekly-report-card.yml` decides `✅ All green` from `make go_no_go` plus a hand-picked Python subset. It does not consume Core Tests, Unified Hunter, Python app or the full applicable CI population.

Interpretation: direct instance of **execution scope smaller than claim scope / subset-green presented as global-green**. Prefer dedupe under the existing CI/global-claim finding rather than a new root ID.

Milestone: CI Truth Restoration.

### R-EXT-5 — PM automation points contributors at a known stale status authority — dedupe candidate

`.github/workflows/pm-status.yml` tells PR authors to update `kai-pm/STATUS.md` when scope changes, while House-in-Order has already shown that STATUS represents an old programme state.

Interpretation: automation reinforces stale authority. Dedupe under document-authority/currentness findings.

Milestone: House-in-Order document reconciliation.

### R-EXT-6 — suspicious tracked root artifact `=0.2.0` — provenance required

The active tree contains an empty tracked root file named `=0.2.0`; `agentic/requirements.txt` contains `langgraph>=0.2.0`.

A plausible shell failure is an unquoted command such as `pip install langgraph>=0.2.0`, where `>` becomes redirection, but **this cause is not proven**.

Milestone: professionalisation/repository hygiene. Trace introducing commit/provenance before deleting or assigning cause.

### R-EXT-7 — no repository security policy found — professionalisation gap

No `SECURITY.md` was found at the repository root or `.github/SECURITY.md` during the review. OpenSSF Scorecard treats Security Policy as a standard repository assurance surface.

Milestone: professionalisation. Draft only after ownership/reporting route is decided; do not add ceremonial boilerplate with no real handling path.

## 3. D332 rulings — close instrument qualification without turning it into an endless parser project

### 3.1 F2 — retro-qualify D328–D331 mutation claims: YES, bounded

Before instrument freeze, re-run the prior mutation tests with the new qualification contract. For every mutation record:

1. **APPLIED** — intended mutation applied exactly as designed / expected cardinality.
2. **REACHABLE** — the mutated semantic branch/property is exercised by the fixture.
3. **DISCRIMINATING** — at least one predeclared assertion distinguishes clean from mutant because of that intended property.
4. **TARGETED** — failure is attributable to the property being tested rather than syntax damage or unrelated global breakage.

Observed failures in D328–D331 are useful evidence, but the new standard should be applied explicitly before freezing the instrument. If a prior mutation cannot satisfy the contract, downgrade its wording to `MUTANT CAUSED FAILURE; PROPERTY QUALIFICATION UNRESOLVED` rather than preserving `mutation-proven` by narrative.

### 3.2 P14's four witnesses are not an R5 denominator if the general proof obligation is explicit

Do not claim the four witness implementations are the complete universe of constructive exclusion.

General admissibility rule:

> A constructive exclusion witness is valid only when it positively proves that the operation's possible repository target set is disjoint from target T.

The current four are implementations of that proof obligation, not a hand-written population of all possible witnesses. Each witness type needs a stable ID, stated assumptions, positive/negative fixtures and default fallback to `COULD_REACH_T` when proof cannot be made.

### 3.3 `/dev/null` reasoning must be corrected

If 99 exclusions are actually absolute `/dev/null` redirects, describing them as repository directory `dev` being disjoint from `data` is a **MISBINDING of the rationale** even if the exclusion verdict is correct.

Correct the path-domain model before instrument freeze. Distinguish at minimum:

- repository-relative path;
- absolute filesystem path / system sink;
- URI/remote target;
- unresolved/dynamic target.

`/dev/null` should be excluded because it is a positively identified absolute/system sink outside the repository target domain, not because a fictional repository directory `dev` differs from `data`.

A correct verdict with a false explanation is not qualified evidence.

### 3.4 Dict-key path fragments must be repaired now

`os.environ["P"]` must not turn the dictionary key `P` into a fixed path fragment. Add the regression fixture and repair this before freeze.

### 3.5 YAML/Make parser gap may remain open at instrument freeze — with a milestone

Because the static writer analyser has been deliberately demoted to corroborative/lower-bound evidence, do **not** turn House-in-Order into a Bash/Make interpreter project.

The bounded YAML `run:` / Make recipe extraction design is sound, but implementation can remain open **provided**:

- the current analyser declares the blind spot;
- no negative/authority verdict depends on its silence;
- the gap is given a named milestone **before static generation corroboration becomes enforcing**.

## 4. House-in-Order exit path — stop the instrument work at a defined boundary

The programme should not remain in census-tool development indefinitely. Use these exit stages.

### H0 — instrument qualification and freeze (CURRENT)

Complete only the load-bearing corrections above: dict-key misbinding, path-domain/witness semantics, retro mutation qualification, and any fixture needed to prove them. Freeze/hash the instrument/specification. Do not add general parser sophistication merely to improve percentages.

### H1 — final census on a new exact subject

After instrument freeze, choose the then-current exact development HEAD/tree as **FINAL HOUSE CENSUS SUBJECT**. Run once, capture complete output, and keep reporting tree separate from measured tree.

Mechanically derive the tracked-document denominator on that exact tree. Do not reuse the old `268` as a timeless number.

### H2 — multi-axis document/region classification

For every tracked Markdown subject, classify independent axes rather than one `role`:

- lifecycle;
- function;
- authority state;
- generation state;
- validity binding;
- scope/region.

`UNKNOWN` is first-class on every axis. Mixed files use semantic region overrides rather than byte/line offsets.

Inferred textual reference pairs are discovery/impact evidence, not authority relationships.

### H3 — active claim qualification

Exhaustively qualify material current-state claims from active/derived candidates against exact tree, machine registers, CI runs, experiment artefacts and explicit human authority.

Historical documents are bound to their historical snapshot rather than forced to match today's tree.

Only after this may a candidate become `AUTHORITATIVE` or `VERIFIED_DERIVED`.

### H4 — repair the control sources

Repair the KAI-GATE denominator/register first, then reconcile active documentation.

Preferred canonical finding architecture: a simple machine-readable schema inspired by OSCAL POA&M, with Markdown tables/reports derived from it. Do not widen one fragile Markdown table into the database.

Repair false/current claims by append/supersession where historical truth must survive. Do not rewrite old experiment evidence to agree with today.

### H5 — install document authority / drift enforcement

Create a machine-derived document authority manifest/check. A tracked controlled document must have explicit classification. Derived regions must declare generator/source/scope/verification. Historical material must bind to snapshot. Superseded material must point to successor. UNKNOWN cannot be used as programme authority.

The check must fail on an unclassified new controlled document and on declared derived regions whose generator/scope contract is broken.

### H6 — bank House-in-Order baseline

Re-run the census and negative fixtures against the repaired current tree. Record exact HEAD/tree and prove the drift checks can detect their own absence. At this point the management/control layer is trustworthy enough to resume the pre-existing programme.

## 5. Programme order after House-in-Order

Do not let the external research redraw frozen experiments or reorder the long-term programme casually.

1. **House-in-Order H0–H6 now.** This is the minimum governance substrate needed because current trackers/registers are known misleading.
2. **Return to KAI-GATE-048 / Item 8.** Resume only under existing authority sequence: Phase B resolution/authorization, spent preflight-sentinel retirement, exact-tree review, separate `ITEM8_GO`, six subject builds only after explicit Dainius authority, then formal 048 closure. Do not retrofit Witness/SLSA/OPA into the frozen experiment.
3. **A-4 provenance repair/review/freeze/hash.** Before freezing A-4's final schema, map KAI provenance onto W3C PROV and in-toto/SLSA concepts where they fit. Preserve KAI statuses and applicability semantics.
4. **Assurance-integration mapping.** Map the repaired evidence/provenance layer against the real system.
5. **Professionalisation / CI Truth Restoration.** Mandatory closure of red CI, CI/local divergence, environment locking, workflow action pinning, global-green misclaims, branch/ruleset enforcement, security policy, suspicious root artefacts and all other zero-loose-ends obligations before final release.
6. **Evidence Plane / Kingsman implementation.** Build KAI's custom qualification/reasoning layer on top of standard provenance/attestation primitives rather than reimplementing all plumbing. OPA/Witness are candidates, not pre-authorised dependencies; evaluate them before adoption.

## 6. Continuity requirement for Orion's next D entry

Read `kai-pm/KAI_ORION_CONTINUITY.md`.

The next D-numbered governance entry should:

- acknowledge Kai governance commits after D332;
- record/deduplicate this research delta without allocating unsupported IDs;
- state the House-in-Order H0–H6 path and milestones;
- preserve the external findings so none disappear before the register is repaired;
- end with the `THREAD RECOVERY BLOCK` required by the continuity protocol.

No production/workflow/register/doc repair is authorised merely by this research brief. Any mutation beyond current instrument scratchpad qualification still needs the relevant Dainius authority and programme-stage decision.
