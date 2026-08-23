# KAI ↔ ORION continuity protocol

> **STATUS: RECOVERY PROTOCOL — NOT A SOURCE OF PROGRAMME TRUTH.**
>
> This file exists only so a fresh Kai/ChatGPT thread can reconstruct the current state from the repository without depending on chat memory. If this file conflicts with the repository, the repository wins.

## 1. Authority order

Use this order when recovering state:

1. **Git branch/commit/tree and machine evidence** — exact repository state.
2. **Latest D-numbered entry in `kai-pm/DECISIONS.md` on the active work branch** — latest programme/governance record.
3. **`CLAUDE.md`** — binding operating rules and stop signals.
4. **`kai-pm/ENGINEERING_DOCTRINE.md` and standing doctrine extensions** — engineering doctrine/fingerprint where applicable.
5. **Canonical machine registers / experiment artefacts / workflow run evidence** for the specific claim.
6. Other plans, trackers, READMEs and status documents only after their authority/currentness has been qualified.

Never promote README, STATUS, backlog, continuation notes, this continuity file, external-model advice, or chat memory above the sources above.

## 2. Fresh-thread recovery algorithm

Before giving consequential programme advice, a fresh Kai thread must:

1. Open repository `dainius1234/kai-system`.
2. Identify the active development branch by finding the branch containing the latest D-numbered governance commit; do not assume `main` is current.
3. Record exact **HEAD SHA + tree SHA**.
4. Read `CLAUDE.md` first.
5. Read the **latest D entry** in `kai-pm/DECISIONS.md` and enough immediately preceding D entries to understand corrections/supersession.
6. Extract the latest explicit standing state, including at minimum:
   - current frozen/qualification subject if one exists;
   - current programme gate/workstream;
   - what is authorised;
   - what is explicitly unauthorised;
   - whether any instrument/evidence is provisional, frozen, or quarantined;
   - next permitted action;
   - open conflicts/corrections.
7. Verify any material state claim against the tree/machine source before relying on it. `Recorded`, `banked`, `closed`, `green`, `absent`, `present`, `authorised` and `current` are verifiable claims.
8. Recover `kai-pm/OPERATOR_VISIBILITY_ENGINEERING_DOCTRINE.md` before a material handoff, architecture decision, phase checkpoint, or operator-facing status claim. Operator visibility is part of the governance/control loop, not optional documentation polish.
9. Only after steps 1–8 may Kai continue the programme.

If the latest D entry cannot be recovered, stop and report **STATE RECOVERY INCOMPLETE** rather than reconstructing from memory.

## 3. Orion handoff obligation

Every future D-numbered governance entry should finish with a compact **THREAD RECOVERY BLOCK** containing:

- `REPORTING_COMMIT` (once known, in the commit message or next append if necessary)
- `MEASURED/FROZEN SUBJECT` and tree, if different from reporting tree
- `CURRENT WORKSTREAM`
- `LAST PROVEN STATE`
- `AUTHORISED NEXT ACTION`
- `EXPLICITLY NOT AUTHORISED`
- `OPEN / UNRESOLVED ITEMS`
- `CORRECTIONS TO PRIOR RECORDS`
- `OPERATOR VISIBILITY` — `COMPLETE` or `INCOMPLETE`, with any known stale/missing operator-facing surface named explicitly

This block is a **navigation aid only**. It must point to evidence rather than replace it.

If material operator-facing state is knowingly stale, contradictory or missing, the handoff must report:

`OPERATOR VISIBILITY INCOMPLETE`

and may not be called fully complete merely because code/tests/evidence are green.

## 4. Anti-staleness rule

Do **not** maintain a duplicated prose copy of the current programme state in this file.

This protocol should remain stable while programme state changes beneath it. Current state is recovered from the latest D entry and exact repository evidence. That is deliberate: a recovery file that needs manual status updates becomes another stale tracker.

Operator visibility does **not** mean duplicating all evidence here. It means ensuring the operator has a current, intelligible surface that points to the evidence and states what is true, uncertain, authorised, blocked and next.

If the recovery procedure itself changes, update this file and record the reason in `DECISIONS.md`.

## 5. Thread-end rule for Kai

When a Kai thread is approaching a handoff or context limit:

1. Verify that all material new decisions/findings are present in the repository rather than only in chat.
2. Ask Orion to bank a D-numbered governance entry if material state exists only in conversation.
3. Confirm the latest D entry contains the THREAD RECOVERY BLOCK.
4. Check operator-facing current-state material for material contradiction/staleness introduced by the work. Where immediate synchronization is outside the authorised workstream, mark the stale surface explicitly and bank the obligation.
5. Give Dainius a plain-language explanation of what changed, what is proven, what remains unresolved, what comes next and what decision/authority is required from him.
6. Do not claim the handoff is complete until the repository artefact is independently visible **and operator visibility is complete or explicitly reported incomplete**.

The repository is the workshop. Chat is the working conversation, not the durable record. The operator-facing summary is the control panel; it must not knowingly lie about the workshop.

## 6. External Technical Review Protocol — DeepSeek / specialist second opinion

Dainius has authorised use of an outside technical/coding adviser when a fresh independent view or specialist expertise can materially improve a decision.

### Role separation

- **Orion** — repository-side evidence collection, controlled execution and implementation.
- **Kai** — programme architecture, historical context, repository review, reconciliation and challenge.
- **DeepSeek / external specialist** — adversarial technical/coding review, alternative implementation ideas, algorithmic critique and specialist second opinion.
- **Dainius** — final operator authority for consequential programme decisions.

An external adviser does **not** have repository authority merely because its technical analysis is strong. Advice that depends on repository reality remains `HYPOTHESIS / REVIEW INPUT` until verified against the exact repo/tree/run/evidence by Kai or Orion.

### When Kai should seek an external technical review

Use judgement rather than a fixed quota, but positively consider review when any of these apply:

1. **Novel or difficult implementation** — complex algorithm, parser, concurrency, security, provenance, build/release, model-runtime or infrastructure design.
2. **Repeated failure class** — two or more fixes expose defects in the same conceptual family, or a repair creates a neighbouring defect.
3. **Disagreement or unresolved interpretation** — Kai and Orion reach materially different technical conclusions, or available evidence supports competing explanations.
4. **High-consequence boundary** — before freezing a major instrument/specification, adopting a new architecture, approving a consequential control, or making a hard-to-reverse implementation choice.
5. **Fresh-eye checkpoint** — a long run of internally consistent work risks shared assumptions or tunnel vision.
6. **Specialist gap** — a question is likely to benefit from deeper coding/algorithmic expertise than the current review has exercised.

Routine, already-proven work does not require external review merely for ceremony.

### Before asking the external adviser

Kai should formulate the smallest technically complete question and provide the minimum evidence/context needed to avoid solving the wrong problem. Where relevant include:

- exact claim/question;
- applicable frozen rules/constraints;
- relevant code or pseudocode;
- observed results/failures;
- what is known versus inferred;
- what the adviser cannot independently verify because it lacks repository access.

Do not ask an external adviser to infer current repository state from incomplete prose if Kai/Orion can measure it directly.

### How external advice is admitted

For every material external recommendation:

1. classify it as `SUPPORTED / PARTIALLY_SUPPORTED / CONFLICTS_WITH_REPO / UNVERIFIED` after repo review;
2. verify every repository-dependent factual premise independently;
3. surface disagreements rather than averaging them away;
4. where conclusions differ, identify the cheapest discriminating measurement/test;
5. bank any adopted material design change or new obligation in the repository decision/evidence trail.

External advice can strengthen, challenge or replace a proposed technical approach after verification. It cannot by itself close findings, alter frozen evidence, authorize scope, or establish repository facts.

### Orion signal

When Orion reaches a question that would materially benefit from outside specialist review, it should explicitly flag:

`EXTERNAL TECHNICAL REVIEW CANDIDATE`

and state:

- the precise technical question;
- why current internal evidence is insufficient or why fresh review is valuable;
- the minimum code/evidence/context to send;
- what decision is blocked or could be improved by the review.

Kai may also invoke external review independently when the conditions above are met.

### Principle

**Use multiple minds, but one evidence standard.** Different reviewers are valuable because they fail differently. Repository truth remains evidence-bound, and consequential authority remains with Dainius.

## 7. Persistent future-design reminder — A4 self-diagnosis evolution

At thread recovery and especially before future **A4 / Evidence Plane / self-diagnosis architecture work**, check:

`kai-pm/A4_SELF_DIAGNOSIS_EVOLUTION.md`

That file preserves Dainius's operator intent that the House-in-Order/Census machinery and lessons should not be discarded as temporary build tooling. The proven concepts — component/file mapping, reader/writer relationships, drift detection, applicability, UNKNOWN/UNRESOLVED semantics, calibrated diagnostics and evidence-bound repair proposals — are candidates to evolve into Kai's later self-diagnosis capability.

This is a **reminder to recover and reconsider the design obligation, not authority to implement it**. A future Kai must reconcile it against the then-current architecture, Evidence Plane, D-numbered decisions and Dainius's authority before taking action.

When closing a thread that materially changes Census/H2/Evidence-Plane machinery or discovers a reusable diagnostic principle, check whether `A4_SELF_DIAGNOSIS_EVOLUTION.md` needs a durable update so those lessons are not lost before A4 planning begins.

## 8. Persistent future-design reminder — Kingsman README / architecture truth refresh

Before any substantial rewrite of the repository front page, architecture graphics, public capability description or professionalisation pass, recover:

`kai-pm/KINGSMAN_README_ARCHITECTURE_REFRESH_PLAN.md`

The existing README and legacy architecture documents are **inputs to audit, not sources of truth**. The refresh must be rebuilt from qualified repository/runtime evidence, House-in-Order results and the then-current Kingsman architecture.

The future work must specifically preserve:

- checked, non-duplicated metrics and status claims;
- a coherent Kingsman end-to-end architecture rather than chronological feature accumulation;
- one unified Self-Diagnosis & Recovery architecture rather than duplicate Doctor/Self-Healing concepts;
- engineering truths/rules already discovered and new rules discovered later;
- explicit LIVE / NOT-CUT-OVER / STUB / PLANNED / UNKNOWN distinctions;
- professional visual/diagram strategy;
- current hardware/runtime truth;
- honest limitations and evidence routes;
- DeepSeek / external adversarial review before publication where it materially improves the result.

When closing a thread that materially changes final architecture, capability naming, engineering doctrine, Evidence Plane, A4/self-diagnosis, runtime topology or verified counts, check whether the Kingsman README refresh plan needs an update so the eventual front page is not rebuilt from stale assumptions.

## 9. Persistent programme reminder — House-in-Order Phase 2 professionalisation

Dainius has defined the broader professionalisation work as **Phase 2 of House-in-Order**. Recover:

`kai-pm/HOUSE_IN_ORDER_PHASE2_PROFESSIONALISATION.md`

Interpretation:

- Phase 1 establishes repository truth, authority, evidence, applicability and currentness.
- Phase 2 takes that qualified truth and professionalises the repository one subsystem/file/capability family at a time to the final Kingsman standard.

A stale, messy or poor-quality file is not automatically a poor idea. Before reworking, merging, superseding, archiving or deleting an artefact, recover the original capability intent and decide where that idea belongs in the final architecture.

The Phase-2 target is not cosmetic tidiness. It is production-grade coherence across implementation, architecture, engineering doctrine, tests, runtime/deployment, documentation, self-diagnosis/recovery, graphics, README/front page and branch/merge hygiene.

DeepSeek remains available under §6 for adversarial technical review at difficult or high-consequence points. Dainius remains final programme authority.

## 10. Persistent programme reminder — Kingsman final master canon and Item 8 correction

Before Phase-2 architecture/professionalisation is materially executed, recover:

`kai-pm/KINGSMAN_FINAL_VISION_MASTER_CANON_PLAN.md`

This is the P2.0 plan for reconciling the final Kingsman vision into one controlled blueprint.

**Correction that must survive thread handoff:** the earlier draft incorrectly said `Team 8`. The operator meant **Item 8**. No Team-8 concept is implied.

**Programme ordering that must survive handoff:**

> **ITEM 8 BEFORE A4.**

Item 8 is a separately governed assurance/build workstream with frozen-design and execution-authority controls. It is not a cognitive team and the master canon does not authorise its execution.

A4 remains the later structural/evidence/self-diagnosis evolution. The latest D-numbered programme sequence remains authoritative where it is more specific.

The master canon is the design baseline for where Kai is going; repository/test/runtime evidence remains the authority for where Kai actually is.

## 11. Standing operator-visibility doctrine

Recover and apply:

`kai-pm/OPERATOR_VISIBILITY_ENGINEERING_DOCTRINE.md`

Core rule:

> **The operator cannot govern what the system does not make legible.**

Dainius is not expected to reverse-engineer programme truth from source code, CI logs, hundreds of evidence files or contradictory status pages. Kai/Orion are responsible for keeping material truth visible enough for him to understand the current state and exercise final authority.

Until Phase 2 implements calibrated mechanical checks, this is enforced procedurally at every material handoff. The Phase-2 automated target is defined in:

`kai-pm/PHASE2_DOCUMENT_SYNC_AND_DRIFT_CONTROL.md`

A green narrow docs gate does not prove operator-facing state is current outside its declared population.
