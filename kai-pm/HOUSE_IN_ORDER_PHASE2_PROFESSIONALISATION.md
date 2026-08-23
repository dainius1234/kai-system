# House-in-Order Phase 2 — Kingsman Professionalisation

> **STATUS: FUTURE PROGRAMME PHASE / PLANNING AUTHORITY ONLY — EXECUTION NOT STARTED BY THIS FILE.**
>
> Operator ruling: the Kingsman README/architecture refresh and the broader repository clean-up/professionalisation are to be treated as **Phase 2 of House-in-Order**.
>
> Phase 1 establishes what is true, current, authoritative, generated, applicable and reproducible. Phase 2 uses that truth base to turn the repository into a coherent, production-grade Kingsman system without losing valuable ideas that are currently buried in stale, duplicated, badly named or poorly implemented files.

## 1. Mission

**Phase 1: KNOW WHAT IS TRUE.**

**Phase 2: MAKE THE WHOLE SYSTEM MATCH THAT TRUTH, PROFESSIONALLY.**

Phase 2 is not a cosmetic tidy-up.

It is a controlled professionalisation programme covering:

- architecture;
- source layout;
- stale/duplicate/reference files;
- capability naming and ownership;
- service boundaries;
- engineering doctrine;
- self-diagnosis/recovery architecture;
- README/front page;
- diagrams and architecture graphics;
- deployment/runtime documentation;
- tests and drift controls;
- historical artefact handling;
- branch/merge hygiene;
- production-grade polish.

The objective is a repository where the implementation, architecture, documentation, evidence and public presentation tell the same story.

## 2. Preserve ideas; challenge implementations

A poor file is not automatically a poor idea.

Many Kai concepts were created incrementally and may currently exist as:

- stale files;
- duplicated components;
- old names;
- partial implementations;
- stubs;
- prototypes;
- disconnected services;
- outdated architecture descriptions;
- good concepts implemented at the wrong layer;
- multiple mechanisms solving part of the same problem.

Phase 2 must recover the **intent** before deciding the artefact's fate.

For each material artefact/capability ask:

1. What problem was this trying to solve?
2. Is that problem still part of the Kingsman vision?
3. What does the current repository actually implement?
4. Is the implementation live, wired, tested and used?
5. Does another component now own the same responsibility?
6. Is the concept worth preserving even if this implementation is not?
7. Where should the capability live in the final architecture?
8. What evidence proves the replacement/evolution is better?

Permitted dispositions after evidence review:

`RETAIN`
`REWORK`
`MERGE`
`RENAME`
`REHOME`
`SUPERSEDE`
`ARCHIVE/HISTORICAL`
`DELETE`
`UNKNOWN — MORE EVIDENCE REQUIRED`

Deletion is never the default merely because a file looks old or poor.

## 3. One-by-one professionalisation loop

For each subsystem/file/family:

`DISCOVER INTENT`
→ `VERIFY CURRENT REALITY`
→ `MAP DEPENDENCIES / OWNERSHIP`
→ `COMPARE WITH FINAL KINGSMAN ARCHITECTURE`
→ `IDENTIFY DUPLICATION / DRIFT / DEFECTS`
→ `PRESERVE VALUABLE CONCEPTS`
→ `DESIGN PRODUCTION-GRADE FORM`
→ `ADVERSARIAL REVIEW`
→ `IMPLEMENT WITH BOUNDED SCOPE`
→ `TEST / MUTATE / VERIFY`
→ `UPDATE EVIDENCE + ARCHITECTURE + DOCS TOGETHER`
→ `BANK KNOWN-GOOD STATE`

Do not professionalise the repository through large uncontrolled rewrites.

## 4. Production-grade bar

Phase 2 should raise each retained capability toward the same engineering standard established during House-in-Order:

- clear responsibility and ownership;
- explicit interfaces/contracts;
- fail-closed where safety/security requires it;
- deterministic schemas rather than free-text coupling where practical;
- evidence/provenance for important claims/actions;
- explicit uncertainty and applicability;
- no silent fallback that changes authority;
- calibrated tests including known-positive/known-negative/boundary cases;
- mutation/discriminating tests for load-bearing controls;
- portable reproduction;
- observable runtime behaviour;
- operational recovery/rollback path;
- consumer enforcement, not merely metadata binding;
- documentation derived from qualified truth;
- no duplicate manually maintained truth where a machine source can derive it.

## 5. Engineering truths are cumulative

All engineering rules learned in Phase 1, earlier audit/remediation, and future Phase 2 work form a growing engineering/diagnostic doctrine.

New rules are not temporary chat lessons.

When a reusable failure shape is discovered:

1. bank the evidence;
2. determine whether it is instance-specific or systemic;
3. add the general rule/check to the appropriate engineering doctrine/control;
4. consider whether it belongs in future A4 self-diagnosis as a diagnostic anti-pattern/test seed;
5. ensure the rule is mechanically enforced where practical rather than relying only on memory/discipline.

See:

- `kai-pm/ENGINEERING_DOCTRINE.md`
- `kai-pm/A4_SELF_DIAGNOSIS_EVOLUTION.md`

## 6. Phase 2 architecture objective

The final system should read as one coherent Kingsman architecture, not a historical pile of features.

Top-level conceptual loop:

`PERCEPTION`
→ `EVIDENCE / PROVENANCE`
→ `QUALIFIED WORLD STATE`
→ `COGNITION / SWARM`
→ `ADVERSARIAL DELIBERATION`
→ `POLICY / AUTHORITY`
→ `DAINIUS APPROVAL`
→ `CAPABILITY / ACTUATION`
→ `EXECUTION`
→ `INDEPENDENT VERIFICATION`
→ `LEARNING / SELF-DIAGNOSIS`

Every major retained component should have a clear home in this model or be justified as supporting infrastructure.

## 7. Self-Diagnosis & Recovery consolidation

Phase 2 must prevent diagnostic capability fragmentation.

Existing/future concepts such as:

- House Doctor;
- Doctor teammate;
- Supervisor/auto-heal;
- System FSM DEGRADED/RECOVERING;
- Self-Capability Map;
- anomaly detection;
- proactive observer;
- Census / A4 structural understanding;
- Evidence Plane;
- causal reasoning;
- dream / Agent-Evolver learning;

should be evaluated as organs of **one Self-Diagnosis & Recovery architecture**, not marketed or implemented as competing independent "doctor" systems.

Target loop:

`SEE → UNDERSTAND → DIAGNOSE → EXPLAIN → PROPOSE → APPROVE → HEAL → VERIFY → LEARN`

See `kai-pm/A4_SELF_DIAGNOSIS_EVOLUTION.md`.

## 8. README and public architecture are Phase 2 deliverables

The front page is the final presentation layer of the professionalisation work, not its source.

Use:

`kai-pm/KINGSMAN_README_ARCHITECTURE_REFRESH_PLAN.md`

The README rewrite should occur only when enough underlying truth has been qualified to prevent immediately recreating drift.

It should include verified metrics, coherent architecture, professional diagrams, clear status labels, honest limitations and evidence routes.

## 9. DeepSeek / external adversarial review

DeepSeek remains available as a standing specialist/adversarial reviewer under the existing external-review protocol.

Use it particularly for:

- architecture consolidation;
- repeated defect classes;
- difficult implementation choices;
- parser/static-analysis/evidence logic;
- security/trust boundaries;
- naming/abstraction conflicts;
- freeze/release checkpoints;
- fresh-eye review after long internally consistent work.

Workflow remains:

`ORION REPO EVIDENCE`
→ `KAI INDEPENDENT REVIEW / PROGRAMME CONTEXT`
→ `DEEPSEEK ADVERSARIAL TECHNICAL REVIEW`
→ `KAI RECONCILIATION AGAINST EVIDENCE`
→ `ORION EXECUTION OF APPROVED CONCLUSION`
→ `DAINIUS FINAL AUTHORITY`

Different minds are useful because they fail differently. One evidence standard remains.

## 10. Suggested Phase 2 workstreams

Final sequencing must be decided against the then-current programme state, but expected workstreams include:

### P2.1 — Repository truth-to-structure reconciliation

Use House-in-Order outputs to identify stale, duplicated, historical, generated and authority-bearing material.

### P2.2 — Capability/name consolidation

Build the public/technical synonym map and remove accidental duplicate architectures.

### P2.3 — Core architecture professionalisation

Review major subsystems one by one against the final Kingsman control/evidence model.

### P2.4 — Engineering rule/control consolidation

Ensure discovered truths are in doctrine and mechanically enforced where practical.

### P2.5 — Self-Diagnosis & Recovery consolidation

Integrate existing diagnostic/healing/introspection concepts with future A4/Evidence Plane direction.

### P2.6 — Reference/document rationalisation

Rework, merge, supersede or archive stale technical documents with explicit lineage rather than silent deletion.

### P2.7 — Runtime/deployment/hardware truth

Align compose profiles, service maps, hardware assumptions and deployment documentation with the accepted target architecture.

### P2.8 — README / architecture graphics professionalisation

Execute the dedicated Kingsman README refresh plan.

### P2.9 — Drift prevention

Make volatile facts machine-derived and prevent the cleaned repository from becoming stale again.

### P2.10 — Final Kingsman professional review

Independent architecture/code/docs/evidence review before declaring the repository production-grade.

## 11. Definition of done

Phase 2 is not complete because the repository looks tidy.

It is complete only when, to the agreed scope:

- retained capabilities have clear architectural ownership;
- duplicate concepts are intentionally merged or distinguished;
- stale files are reworked/superseded/archived with traceable decisions;
- important concepts from historical material have not been accidentally erased;
- engineering doctrine reflects accumulated lessons;
- implementation and docs agree;
- diagrams agree with implementation;
- status/metrics are reproducible;
- live/stub/planned/unknown distinctions are honest;
- critical controls are tested and discriminating;
- repository structure is professional and maintainable;
- future Kai/Orion can recover the system without relying on tribal knowledge;
- README/front page accurately represents the final Kingsman vision;
- no known material defect has been labelled "optional" merely because it was unrelated to the current local task.

## 12. Plain-language statement

Phase 1 sorts out the workshop and tells us what every box really contains.

Phase 2 takes each useful part, decides where it belongs in the final machine, rebuilds weak parts properly, removes duplicates, keeps the good ideas, tests everything, labels it correctly, draws the real architecture, and leaves the whole Kai repository looking and behaving like one professionally engineered system rather than years of development history piled together.
