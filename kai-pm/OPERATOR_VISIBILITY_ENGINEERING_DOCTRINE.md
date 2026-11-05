# Operator Visibility Engineering Doctrine

> **STATUS: STANDING ENGINEERING DOCTRINE EXTENSION.**
>
> This file is part of the engineering doctrine for Kai/Orion work. It exists separately from `ENGINEERING_DOCTRINE.md` until the Phase-2 doctrine consolidation can merge/check all doctrine sources mechanically without creating another hand-maintained copy.
>
> Operator ruling: Dainius's ability to understand the current system is part of the governance/control loop. Machine evidence that only Kai/Orion can practically interpret is insufficient for operator sovereignty.

## OV-1 — The operator cannot govern what the system does not make legible

**A consequential programme state is not adequately governed merely because the evidence exists somewhere in the repository. The operator must have a current, comprehensive and intelligible view of the state needed to make the next decision.**

This means operator-facing information is not cosmetic documentation. It is part of the control surface.

Kai/Orion must maintain or explicitly qualify the operator view of, at minimum where material:

- what has changed;
- what is currently proven;
- what remains UNKNOWN / unresolved;
- what is authorised and not authorised;
- what workstream is active;
- what comes next and why;
- material defects/risks;
- current architecture/capability ownership;
- whether a feature is LIVE / PRESENT-NOT-CUT-OVER / STUB / PLANNED / UNKNOWN;
- current counts/metrics where those numbers are used for decisions;
- the evidence/subject to which a status statement applies.

The exact public README does not need to be rewritten on every experimental commit. But **some operator-facing current-state surface must remain truthful and usable**, and the final README/front page must eventually become a reliable high-level view derived from qualified truth.

## OV-2 — Hidden correctness is not sufficient governance

A system can be technically correct while operationally ungovernable if the operator cannot see what the engineers/agents see.

Therefore:

> **REPO TRUTH WITHOUT OPERATOR LEGIBILITY IS AN INCOMPLETE CONTROL LOOP.**

Examples:

- 300 correct evidence files do not substitute for a clear statement of the current conclusion;
- a green CI gate does not help the operator if the front page still presents contradictory counts;
- a correct architectural change is not complete if the operator-facing architecture still describes the superseded system;
- a new rule is not safely adopted if only the agents know it exists;
- an unresolved risk hidden in a technical register but absent from the handover is not adequately surfaced.

## OV-3 — Currentness must be subject-bound, not assumed

Operator-facing claims must distinguish:

- **CURRENT FOR SUBJECT X**;
- **HISTORICAL**;
- **PLANNED / TARGET**;
- **UNKNOWN / NOT YET QUALIFIED**.

Where practical, volatile operator-facing state should carry exact commit/tree/run/generator identity.

A date is useful context but is not sufficient currentness proof.

## OV-4 — Do not make the human reconcile duplicate truth manually

If the same volatile fact appears in more than one operator-facing place, it should be generated from one qualified source or mechanically reconciled.

The operator must not be expected to notice that:

- a badge says 60 services;
- a table says 61;
- a quick-reference block says 59;
- an old diagram shows 26.

That is an engineering-control failure, not a reading-comprehension task for the operator.

## OV-5 — Handoff completeness includes operator visibility

Before Kai/Orion calls a material work package, thread handoff or phase checkpoint complete, verify:

1. the repository evidence is banked;
2. the latest D-numbered/continuity state is current;
3. material changes/decisions are explained in plain language;
4. operator-facing current-state notes are not knowingly contradicting the new state;
5. any operator-facing surface that remains stale is explicitly marked **STALE / NOT AUTHORITY** and tracked for repair;
6. the operator can tell what is true now, what is uncertain, what comes next and what requires their decision.

If these conditions are not met, report:

`OPERATOR VISIBILITY INCOMPLETE`

Do not call the handoff complete merely because code/tests/evidence are green.

## OV-6 — Plain-language explanation is a deliverable

For any material technical conclusion, freeze, architecture decision, experiment result or new risk, Kai/Orion must be able to state the result in plain language without destroying the technical distinctions.

A technically precise explanation may sit underneath it, but the operator should not need to decode specialist vocabulary to exercise authority.

The obligation is not to oversimplify. It is to translate.

## OV-7 — The future front page is an operator instrument

During House-in-Order Phase 2, the README/front page, architecture graphics, current-state summary and related notes should be redesigned as a deliberate **operator instrument** as well as a repository cover page.

It should answer, quickly and truthfully:

- What is Kai now?
- What actually works?
- What is still being built?
- What changed recently?
- What are the current major risks/unknowns?
- What is the current programme stage?
- What is the final Kingsman destination?
- What needs Dainius's decision?
- Where is the detailed evidence?

This operator view should be derived from the same qualified truth used by engineering controls where practical.

## OV-7A — Visual mission-control view is mandatory

Dainius primarily consumes complex programme information visually. The operator surface must therefore provide an **overview-first, drill-down-second** experience rather than requiring him to reconstruct programme state from long prose files.

The Phase-2 operator/front-page design should include, where practical:

### 1. Full architecture overview

One current high-level architecture view showing the major Kai systems and how they connect, including the Kingsman control/evidence loop.

The visual must distinguish at least:

- current/live architecture;
- target/future architecture;
- transitional/not-cut-over components where material;
- authority/evidence boundaries;
- major dependency/order relationships such as Item 8 before A4.

### 2. Programme roadmap / phase strip

A visual progression showing major stages and current position, for example:

`House-in-Order Phase 1 → Item 8 → A4 / Evidence Plane → Phase 2 Professionalisation → Final Kingsman Review`

The exact sequence must be derived from the current D-numbered authority, not this illustrative line.

### 3. Outstanding-work board

Provide a concise status view of material obligations with visible states such as:

- `✅ COMPLETE / VERIFIED`
- `🟡 IN PROGRESS`
- `⏸ BLOCKED / WAITING AUTHORITY`
- `⚠ OPEN DEFECT / RISK`
- `⬜ NOT STARTED`
- `❓ UNKNOWN / NEEDS QUALIFICATION`

The state must be evidence-backed; a tick is a claim and must not be used merely because work feels finished.

### 4. Operator decisions / approvals

Clearly surface what requires Dainius's decision separately from technical work the team can continue without him.

### 5. Risks / unknowns overview

Show the small number of material unresolved risks/unknowns that could affect current direction, with links to full evidence.

### 6. Recent-change summary

Show the most important recent architecture/programme changes in plain language so the operator can re-orient quickly after time away.

### 7. Drill-down navigation

Every overview element should point to the deeper canonical evidence/specification/decision rather than duplicating full technical detail on the front page.

The desired operator experience is:

> **See the whole machine → see where we are → see what is done → see what is outstanding → see what needs my decision → drill into detail only when I choose.**

## OV-7B — Visuals are evidence-bearing views, not decoration

Architecture diagrams, progress boards and status graphics must obey the same truth rules as prose.

A diagram arrow is a claim. A green box is a claim. A tick is a claim.

Therefore:

- visuals must carry or inherit exact subject/currentness identity;
- a superseded diagram must be visibly marked historical/stale;
- generated status colours/icons must come from qualified status data where practical;
- no visual should imply a component is live merely because it exists in the target architecture;
- Phase 2 should prefer diagrams/status views that can be regenerated from structured sources to reduce visual drift.

## OV-8 — Mechanical enforcement target

The doctrine must not stop at prose.

Phase 2 must implement/check, where practical:

- one machine source for volatile README/status metrics;
- regeneration/diff checks for generated regions;
- exact-subject stamps for current state;
- contradiction detection across duplicated operator-facing metrics;
- currentness/authority markers for linked architecture/reference documents;
- a closed denominator showing which operator-facing surfaces/claims the gate actually checks;
- known-positive mutation demonstrating that a stale/contradictory operator fact makes the control fail;
- handoff checks that refuse a complete status when material operator state is missing or knowingly stale;
- reconciliation between structured programme status and the visual roadmap/checklist;
- checks that a displayed completion tick cannot survive withdrawal of its underlying closure evidence where practical.

Until those controls exist, Kai/Orion must compensate procedurally by explicitly checking operator-facing state at handoff.

## OV-9 — This feeds A4 / Kai Doctor

Future A4 self-diagnosis should treat disagreement between Kai's measured implementation and Kai's operator-facing representation as a diagnosable fault class.

Examples:

- implementation changed but architecture diagram did not;
- service is no longer live but README says LIVE;
- current metrics differ across the front page;
- a capability was superseded but old operator guidance still routes to it;
- a governing rule changed but the handover still states the old rule;
- roadmap says COMPLETE while its evidence/closure condition no longer holds.

Kai Doctor should be able to report:

> "My operator-facing description no longer matches my measured state. Here are the conflicting claims, their subjects, and the proposed correction."

## Why this rule was earned

During House-in-Order planning on 23 August 2026, the active README was found to contain contradictory service/test/LOC/milestone figures and overlapping old/new architecture language even though a docs-sync/check mechanism existed.

Inspection showed the control only covered a narrow generated population: one README status table and two backlog metrics. Other badges, Quick Reference counts, architecture/status prose and graphics could drift while the docs gate remained green.

The operator stated the practical consequence directly: without a current and comprehensive front page/notes/handover, they cannot keep up with the technical team well enough to lead it. The operator further clarified that a visual overview of architecture, planning, outstanding work and completed/checkable items is the primary way they intake complex programme state. That converts visual status design from presentation preference into part of the governance interface.

## Plain-language rule

> **Do not make Dainius reverse-engineer the project from our code and evidence. We are responsible for keeping the truth visible enough for him to lead it — preferably as a clear visual overview first, with technical depth underneath.**
