# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

## 1) Project one-liner (what is Kai)
Kai is a self-sovereign, local-first personal AI system built as cooperating services, designed to grow memory, reasoning, and operational reliability without external platform lock-in.

## 2) Current phase + current jewel
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current focus:** CI green-again sweep complete; resume H2 backlog + close/land PR #46

## 3) In-flight PRs (link)
- GPU Phase 0 consolidation: https://github.com/dainius1234/kai-system/pull/46 (draft, awaiting review)

## 4) Blocked items + unlock conditions
- **Phase 1/2/4/5 blocked** until RTX 5080 hardware is procured, provisioned, and validated.
- Unlock criteria: GPU host online, Docker + Ollama healthy, and target local model baseline validated.

## 5) Next 3 actions in priority order
1. Land or close PR #46 (GPU Phase 0 consolidation).
2. Resume H2 backlog: H2.3 (proactive-scan gather), H2.4 (embedding executor).
3. Automate coverage gate to reduce R3 risk.

## 6) Pointer to STATUS.md, SEQUENCE.md, DECISIONS.md
- Status dashboard: [STATUS.md](STATUS.md)
- Phase sequence: [SEQUENCE.md](SEQUENCE.md)
- Append-only decisions: [DECISIONS.md](DECISIONS.md)
- Canonical roadmap: [STRATEGIC_PLAN.md](STRATEGIC_PLAN.md)
- Latest reality check: [REALITY_CHECK_2026-05-10.md](REALITY_CHECK_2026-05-10.md)
- Prior session backlog (if present): [SESSION_BACKLOG.md](SESSION_BACKLOG.md)

## 7) PM operating rules (commitments)
- No drift between docs, status, and delivered code.
- Decision log is append-only; supersede with new entries.
- Before claiming any item is in flight, run the **diff-vs-README ritual**: verify against `README.md`, `CHANGELOG.md`, and current open PRs.
- After each merge, refresh `STATUS.md`, `RISKS.md`, and changelog entries as needed.
