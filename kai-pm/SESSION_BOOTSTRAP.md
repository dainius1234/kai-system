# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

## 1) Project one-liner (what is Kai)
Kai is a self-sovereign, local-first personal AI system built as cooperating services, designed to grow memory, reasoning, and operational reliability without external platform lock-in.

## 2) Current phase + current jewel
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** Coverage uplift + CI green + PM v2 cleanup

## 3) In-flight PRs (link)
- PM cleanup and CI fix: https://github.com/dainius1234/kai-system/pull/49
- Consolidation baseline: https://github.com/dainius1234/kai-system/pull/46

## 4) Blocked items + unlock conditions
- **Phase 1/2/4/5 blocked** until RTX 5080 hardware is procured, provisioned, and validated.
- Unlock criteria: GPU host online, Docker + Ollama healthy, and target local model baseline validated.

## 5) Next 3 actions in priority order
1. Merge PR #49 (PM v2 corrections + CI fix + cleanup).
2. Confirm CI is green on `main` post-merge.
3. Continue Phase 0 coverage/documentation hardening work.

## 6) Pointer to STATUS.md, SEQUENCE.md, DECISIONS.md
- Status dashboard: [STATUS.md](STATUS.md)
- Phase sequence: [SEQUENCE.md](SEQUENCE.md)
- Append-only decisions: [DECISIONS.md](DECISIONS.md)
- Canonical roadmap: [STRATEGIC_PLAN.md](STRATEGIC_PLAN.md)
- Prior session backlog (if present): [SESSION_BACKLOG.md](SESSION_BACKLOG.md)

## 7) PM operating rules (commitments)
- No drift between docs, status, and delivered code.
- Decision log is append-only; supersede with new entries.
- Before claiming any item is in flight, run the **diff-vs-README ritual**: verify against `README.md`, `CHANGELOG.md`, and current open PRs.
- After each merge, refresh `STATUS.md`, `RISKS.md`, and changelog entries as needed.
