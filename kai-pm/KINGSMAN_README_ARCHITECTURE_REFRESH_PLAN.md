# Kingsman README & Architecture Truth Refresh Plan

> **STATUS: PLANNING / FUTURE EXECUTION OBLIGATION — NO README REWRITE AUTHORISED BY THIS FILE.**
>
> Purpose: preserve the strategy for rebuilding the public/front-page presentation of Kai after House-in-Order and the current assurance work have established which claims are actually current, verified, provisional, historical or future-facing.
>
> The README is not programme authority. The repository/evidence/decision chain remains authoritative. This plan exists so the eventual README is derived from verified truth rather than becoming another stale tracker.

## 1. Operator intent

Dainius wants the front page to become a professional, coherent representation of the final Kingsman-tier vision rather than an accumulation of historical feature additions.

Required outcome:

- every current factual claim checked against repository/runtime evidence;
- stale and contradictory counts removed;
- architecture represented accurately;
- capabilities grouped into coherent systems instead of duplicate names;
- House-in-Order / Evidence Plane / future A4 self-diagnosis integrated into the final vision;
- engineering truths/rules discovered during audit, assurance and future work preserved as part of Kai's engineering/diagnostic doctrine;
- diagrams and visual hierarchy that let a new reader understand Kai quickly;
- honest separation of LIVE / PRESENT-BUT-NOT-CUT-OVER / STUB / PLANNED / HISTORICAL / UNKNOWN;
- professional polish suitable for serious technical review without turning the README into marketing fiction.

## 2. Why the current README cannot simply be patched

The active-branch README currently mixes several different time periods and authority levels.

Examples already observed during planning:

- badge row says 60 services, 2,888 tests, ~67,500 Python LOC and 52 milestones;
- Project Status below says 61 containers, 4,751 tests, ~168,056 LOC and 45 milestones;
- Quick Reference then says 90 test targets / ~2,888 tests while Project Status says 91 / 4,751;
- architecture prose says Unified Hunter is built but not cut over, while other sections use broad present-tense capability language;
- README declares that every capability listed has code/tests while the same page also contains explicit GPU/graph stubs;
- diagnostic/recovery concepts are split across House Doctor, Doctor teammate, Supervisor auto-heal, System FSM DEGRADED/RECOVERING, Self-Capability Map, anomaly detection, Agent-Evolver/dream learning and older self-healing terminology;
- `docs/architecture.md` still describes a 26-service, 10-way-context architecture with old ports and therefore cannot be reused as current architecture without qualification;
- current repository visibility/licensing/status presentation must be re-verified at rewrite time rather than inherited from badges or old prose.

Conclusion: eventual work should be a **truth-led reconstruction**, not sentence-by-sentence cosmetic editing.

## 3. Kingsman-tier information architecture for the future README

Recommended front-page order:

### A. Hero / identity

Short, disciplined identity statement.

Explain in one screen:

- what Kai is;
- what makes it structurally different;
- local/sovereign operating intent;
- evidence-bound decision philosophy;
- operator approval / earned autonomy principle.

Avoid unqualified superlatives.

### B. Verified status snapshot

Machine-derived, exact-subject metrics only.

Possible fields:

- exact measured commit/tree;
- services by stack/profile;
- test targets / assertions / suites;
- LOC only if useful and derived reproducibly;
- currently supported hardware/runtime profile;
- current major programme stage;
- last verified known-good run / waypoint where appropriate.

No duplicated manually maintained counts in badges, prose and tables. One source produces all displayed metrics.

### C. Architecture at a glance

One primary professional diagram showing the Kingsman control loop:

`PERCEPTION`
→ `EVIDENCE / PROVENANCE`
→ `QUALIFIED WORLD STATE`
→ `COGNITION / SWARM`
→ `DELIBERATION / ADVERSARY`
→ `POLICY`
→ `DAINIUS APPROVAL`
→ `CAPABILITY / ACTUATOR`
→ `EXECUTION`
→ `INDEPENDENT OUTCOME VERIFICATION`
→ `LEARNING / SELF-DIAGNOSIS`

This becomes the umbrella architecture rather than presenting a list of unrelated services.

### D. What Kai can actually do now

Use explicit status vocabulary, preferably mechanically backed:

- **VERIFIED LIVE**
- **PRESENT / NOT CUT OVER**
- **QUALIFIED BUT SUBJECT-RESTRICTED**
- **STUB / FOUNDATION ONLY**
- **PLANNED / RESEARCH**
- **UNKNOWN / NOT YET QUALIFIED**

Never collapse PRESENT into LIVE.

### E. Major capability systems

Group by coherent architectural role rather than by the chronological D-number in which the feature appeared.

Candidate groups:

1. Perception & world awareness
2. Memory & identity
3. Reasoning / swarm / adversarial review
4. Evidence, provenance & epistemic control
5. Policy, approval & controlled execution
6. Self-diagnosis, recovery & learning
7. Skills / growth / sandboxed capability expansion
8. Operator relationship & interfaces
9. Hardware/runtime architecture

D-number history can live in technical docs rather than dominate the front page.

### F. Self-Diagnosis & Recovery — one umbrella

Do **not** create another competing "doctor" feature.

Unify existing/future components as one architecture:

`SEE`
→ `UNDERSTAND STRUCTURE`
→ `DIAGNOSE`
→ `EXPLAIN`
→ `PROPOSE`
→ `APPROVE`
→ `HEAL`
→ `VERIFY`
→ `LEARN`

Map existing/future mechanisms beneath it:

- watchers / proactive observer / anomaly detection = SEE;
- A4 + Census + Evidence Plane + component/file/dependency map = UNDERSTAND;
- House Doctor + causal reasoning = DIAGNOSE;
- Doctor teammate / Kai interface = EXPLAIN;
- Sage/adversary/counterfactual reasoning = PROPOSE / challenge;
- Tool Gate + operator governance = APPROVE;
- Supervisor / controlled actuators = HEAL;
- Evidence Plane / independent checks = VERIFY;
- dream / Agent-Evolver / failure-pattern memory = LEARN.

See `kai-pm/A4_SELF_DIAGNOSIS_EVOLUTION.md`.

### G. Engineering truths / diagnostic doctrine

The README should explain the existence of this discipline without dumping every rule onto the front page.

Create a concise "Engineering Doctrine" summary and link to the canonical doctrine/evidence documents.

Rules/lessons that must remain visible in the architecture include:

- truth > progress;
- present != executed != enforced;
- UNKNOWN is not negative evidence;
- non-detection != absence;
- context must not become semantics;
- inference discovers, declarations govern;
- evidence identity immutable, applicability separate;
- exact subject/environment binding;
- before measuring a property, prove the environment can observe it;
- calibration must reach the state it claims to test;
- declared alphabet and implemented/emitted alphabet must agree;
- subject-population applicability travels with evidence;
- binding is not enforcement until consumers refuse misuse;
- negative claims require a closed search space;
- repair must preserve boundaries and independently verify outcomes;
- disagreement should be settled with the cheapest discriminating measurement/test;
- future rules discovered during build, assurance or operation must be banked and considered for the self-diagnostic doctrine.

The eventual A4/Kai Doctor should use these not merely as documentation but as diagnostic anti-patterns and test-generation seeds.

### H. Hardware / deployment profile

Replace stale historical hardware assumptions with current verified target architecture.

At rewrite time derive the current target from canonical architecture decisions, including the ASUS ROG Flow Z13 / Strix Halo direction if it remains current and accepted.

Separate:

- current development/runtime environment;
- target laptop deployment;
- CPU/GPU/NPU role;
- optional/experimental hardware;
- what is actually tested today versus future design.

### I. Honest limitations

Keep this prominent.

Examples:

- model-dependent reasoning quality;
- local-model constraints;
- dynamic code/static-analysis limitations;
- components that are present but not cut over;
- incomplete authority/claim qualification;
- features awaiting GPU, graph scale, data history or explicit programme authorisation.

A serious README gains credibility by stating boundaries precisely.

### J. Quick start / operator usage

Only after architecture/status, give concise current commands.

Commands must be machine-verified at the exact README subject. No stale test counts in comments.

### K. Documentation map

Front page should route readers to canonical deep documents instead of duplicating them.

Potential destinations:

- architecture;
- Evidence Plane / provenance;
- Unified Hunter / Kingsman execution path;
- engineering doctrine;
- security posture;
- House-in-Order / assurance;
- A4 self-diagnosis evolution;
- deployment/runbooks;
- roadmap / current programme status.

Each linked document must first be qualified as current or explicitly marked historical.

## 4. Visual / graphics strategy

The final front page should be visually professional but not decorative noise.

Target **2-4 load-bearing visuals**, each with a defined job:

1. **Kingsman Architecture Overview** — the main end-to-end control/evidence loop.
2. **Self-Diagnosis & Recovery Loop** — shows how House Doctor, A4/Evidence Plane, Supervisor and learning fit into one system.
3. **Runtime / Deployment Topology** — laptop hardware + CPU/GPU/NPU + core services/stacks, if still useful after simplification.
4. Optional **Evidence/Authority Flow** — only if it materially improves understanding and is not redundant with the main architecture.

Do not show 60+ services individually on the hero diagram. Detailed service/network maps belong in architecture docs.

Before graphics are produced:

- freeze the semantic architecture;
- identify current service groups and counts mechanically;
- decide which names survive professionalisation;
- verify hardware assumptions;
- verify every arrow in the diagram.

Graphics should be generated from a documented design/source where possible so they can be updated without hand-redrawing contradictory pictures.

## 5. Naming / duplicate-control strategy

Before rewriting, build a capability synonym/ownership register.

Example family:

- House Doctor = deterministic/system diagnostic engine;
- Doctor teammate = conversational specialist/explainer;
- Supervisor = watchdog/recovery executor;
- System FSM = operational state controller;
- Self-Capability Map = introspection/inventory;
- A4/Evidence Plane = structural/evidence self-understanding;
- Self-Healing = umbrella recovery capability, not a competing module name.

For every overlapping term determine:

`public name | technical component | role | authority | current status | retain/rename/merge/deprecate`

No public capability should appear twice under different names unless the distinction is intentional and explained.

## 6. Claim-quality contract for the future README

Every substantive current-state claim must carry an evidence route during authoring.

Internal authoring ledger should record:

`README claim`
→ `exact subject`
→ `evidence source`
→ `status (verified / qualified / planned / historical / unknown)`
→ `expiry/currentness rule`

Examples:

- service count -> compose-derived machine count;
- tests -> actual test inventory/run evidence;
- "live" capability -> executable code + wiring + qualifying test/runtime evidence;
- "all" / "none" / "zero" claims -> closed denominator required;
- security claims -> exact enforcement path, not presence of configuration alone;
- hardware claims -> accepted architecture decision + tested capability where applicable.

The README may summarize evidence, but it must not invent authority.

## 7. Proposed execution phases

### Phase R0 — House-in-Order dependency

Do not perform the final rewrite while the current document authority/currentness work is unresolved.

Use House-in-Order/H2/H3 outputs to determine which existing docs/claims can be trusted.

### Phase R1 — Front-page inventory

Mechanically parse README headings, tables, badges, links, commands, numerical claims, capability names, status language and architecture references.

Produce a claim inventory with evidence status.

### Phase R2 — Duplicate / naming map

Build capability synonym map and decide the canonical public architecture vocabulary.

No implementation changes required.

### Phase R3 — Canonical architecture reconstruction

Rebuild architecture from repository/runtime facts and current Kingsman design, not from old README/architecture prose.

Produce diagrams only after arrows/components are verified.

### Phase R4 — Content skeleton

Freeze section order and what belongs on front page versus linked technical docs.

### Phase R5 — External adversarial review

Use DeepSeek / specialist review before prose publication.

Review questions should include:

- Are capability boundaries coherent or duplicated?
- Does the architecture mix evidence, cognition, policy and execution layers incorrectly?
- Are status labels misleading?
- Are any claims stronger than their evidence?
- Is self-diagnosis/recovery represented as one architecture rather than duplicate features?
- Does the page communicate the Kingsman vision clearly to a technical outsider?

External review remains advisory and repo-dependent claims must be independently verified.

### Phase R6 — README rewrite

Rewrite from the frozen content skeleton and claim ledger.

Do not copy old paragraphs merely because they sound good.

### Phase R7 — Architecture/document synchronization

Update or supersede stale architecture/reference docs so README does not link to contradictions.

`docs/architecture.md` is already known to contain historical 26-service / old-port / 10-way-context material and must not be silently treated as current.

### Phase R8 — Visual production / polish

Produce final architecture graphics, hierarchy, badges and tables from verified data.

### Phase R9 — Mechanical drift controls

Where practical, derive volatile metrics automatically rather than maintaining duplicate numbers.

Add checks so:

- badge metrics and status table cannot disagree;
- command comments do not carry stale test counts;
- linked canonical docs exist;
- prohibited stale phrases/statuses can be detected;
- README claim sources/expiry are reviewed when architecture changes.

### Phase R10 — Independent final review

Kai + Orion + DeepSeek/adversarial reviewer compare the rendered README against repo truth and final Kingsman architecture.

Only then present it as the professional front page.

## 8. What must NOT happen

- no cosmetic README rewrite before claim qualification;
- no treating old README text as truth because it is on the front page;
- no new marketing superlatives without evidence;
- no duplication of House Doctor / Self-Healing / A4 as separate competing architectures;
- no hand-maintained copies of volatile counts in multiple places;
- no architecture graphic with unverified arrows;
- no hiding stubs or not-cut-over components behind generic "live" language;
- no deleting useful historical ideas merely because names change;
- no turning A4/self-diagnosis into uncontrolled self-repair authority;
- no forgetting new engineering truths discovered after this plan: they must be considered for doctrine + future diagnostic rules.

## 9. Deliverables when this workstream is eventually authorised

1. README claim inventory and contradiction report.
2. Capability duplicate/naming register.
3. Current architecture fact sheet.
4. Kingsman-tier architecture diagram specification.
5. Self-Diagnosis & Recovery diagram specification.
6. README information architecture / content skeleton.
7. Machine-derived status/metrics contract.
8. Stale document/link remediation list.
9. DeepSeek adversarial review packet + reconciled findings.
10. Rewritten README.
11. Updated/superseded architecture docs.
12. Drift controls/tests.
13. Final rendered front-page review.

## 10. Relationship to other durable notes

This plan should be recovered together with:

- `kai-pm/A4_SELF_DIAGNOSIS_EVOLUTION.md`
- `kai-pm/KAI_ORION_CONTINUITY.md`
- `kai-pm/ENGINEERING_DOCTRINE.md`
- latest D-numbered House-in-Order / Evidence Plane decisions
- current Unified Hunter/Kingsman architecture evidence
- `kai-pm/ASSURANCE_COUNTERPART_RESEARCH_2026-08-23.md` where still applicable.

## 11. Plain-language target

The final README should let a serious reader understand, within a few minutes:

> What is Kai? What actually works today? What is still being built? How does information become evidence, decisions and actions? Why can it be trusted more than an ordinary pile of agents? How does it diagnose itself and recover? What remains uncertain? What hardware is it intended to run on? Where can I inspect the proof?

If the page cannot answer those questions clearly, it is not finished.
