# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

## 1) Project one-liner (what is Kai)
Kai is a self-sovereign, local-first personal AI system built as cooperating services, designed to grow memory, reasoning, and operational reliability without external platform lock-in.

## 2) Current phase + current jewel
- **Current phase:** Phase 3 software delivery/consolidation (hardware-independent track)
- **Current jewel context:** J2 wake-word/intent work is implemented and sits in merge sequencing with consolidation and PM-system updates

## 3) In-flight PRs (link)
- Consolidation baseline: https://github.com/dainius1234/kai-system/pull/46
- PM System v2 (this operating system): https://github.com/dainius1234/kai-system/pull/48
- J2 reference PR (merged): https://github.com/dainius1234/kai-system/pull/47

## 4) Blocked items + unlock conditions
- **Phase 1/2/4/5 blocked** until GPU hardware is procured, provisioned, and validated.
- Unlock criteria: GPU host online, Docker+Ollama healthy, 7B model baseline passing smoke checks.

## 5) Next 3 actions in priority order
1. Merge PR #46 to unify `main` baseline.
2. Merge PR #48 and switch PM operations to `kai-pm/` + `.github` automation.
3. Dispatch J6 + MCP refactor in parallel per D13.

## 6) Pointer to STATUS.md, SEQUENCE.md, DECISIONS.md
- Status dashboard: [STATUS.md](STATUS.md)
- Locked roadmap: [SEQUENCE.md](SEQUENCE.md)
- Append-only decisions: [DECISIONS.md](DECISIONS.md)

## 7) PM operating rules (commitments)
- No drift between docs, status, and delivered code.
- Use parallel dispatch only for independent scopes (logical + file-scope independence).
- Decision log is append-only; never rewrite history.
- Every PR declares sequence step and jewel scope.
- After each merge, run post-merge checklist and refresh status/metrics.
