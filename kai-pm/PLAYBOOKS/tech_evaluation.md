# Tech Evaluation Playbook

## Goal
Evaluate a proposed tool/model/framework consistently before adoption.

## Evaluation criteria
1. **Strategic fit:** Does it move a locked sequence step or remove a real blocker?
2. **Offline/sovereign fit:** Can it run in Kai's local-first constraints?
3. **Operational cost:** Setup, maintenance, and failure blast radius.
4. **Security/privacy impact:** New attack surface or data movement risk.
5. **Performance impact:** Latency, memory, and hardware requirements.
6. **Migration effort:** Refactor scope and rollback path.

## Decision authority
- Final call: project owner (`@dainius1234`) with PM recommendation.
- If architecture/security impact is high, decision must be logged in `DECISIONS.md`.

## Process
1. Open `tech_adoption` issue template.
2. Fill criteria with evidence and trade-offs.
3. Assign provisional radar state: Adopt / Trial / Assess / Hold.
4. If approved, update `kai-pm/TECH_WATCH.md` with assessed date and re-evaluate date.
5. If rejected/deferred, record rationale and next review date.
