# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

> 📌 **If you are a new chat / new agent / future-Dainius reading this:**
> The single source of truth is `kai-pm/CLEANUP_TODO.md`. Read it. Then read this file.
> Then check open PRs. You will be productive in under 2 minutes.

---

## 1) Project one-liner (what is Kai)
Kai is a self-sovereign, local-first personal AI system built as cooperating services, designed to grow memory, reasoning, and operational reliability without external platform lock-in.

## 2) Current phase + current focus
- **Current phase:** Phase 0 — Pre-GPU Hardening
- **Current sprint (2026-06-01):** **Cleanup Sprint Week 1** — see `kai-pm/CLEANUP_TODO.md`
- **Why we paused features:** Audit on 2026-06-01 revealed 3 hot spots — `langgraph/` shadows the PyPI package, `langgraph/app.py` is 80 KB of mixed concerns, and ~100 Makefile targets with no honesty gate. Foundation must stabilise before we resume J-series / P-series work.

## 3) In-flight cloud agent sessions (dispatched 2026-06-01)
Track these PRs — they are the active work right now:

| # | Task | Type | Status |
|---|---|---|---|
| 1 | **KEYSTONE: rename `langgraph/` → `agentic/`** | code change | dispatched |
| 2 | Map `agentic/app.py` responsibilities → `kai-pm/AGENTIC_APP_MAP.md` | docs only | dispatched |
| 3 | Diff compose files → `kai-pm/COMPOSE_DRIFT.md` | docs only | dispatched |
| 4 | Audit ~100 Makefile targets → `kai-pm/MAKEFILE_AUDIT.md` | docs only | dispatched |

Plus the original Week 1 housekeeping agent (close #59 + #60, label PRs, delete tempo tests).

**Where to check:** https://github.com/dainius1234/kai-system/pulls

## 4) The cleanup roadmap (4 weeks)
Live tracker: [`CLEANUP_TODO.md`](CLEANUP_TODO.md). Summary:

- **Week 1 — Stop the Bleeding** *(in progress)*: PR housekeeping, delete orphan tempo tests, **rename `langgraph/` → `agentic/`** ⭐.
- **Week 2 — Untangle the Giant**: split `agentic/app.py` (one PR per slice: routes / state / flows / providers / prompts), reconcile docker-compose, prune Makefile.
- **Week 3 — Honest Verification**: repo-wide coverage gate, honest `merge-gate`.
- **Week 4 — Resume Features** (only if 1–3 done): multi-backend LLM router, skills templates, journal templates, CIS P29.

## 5) Blocked items + unlock conditions
- **Phases 1 / 2 / 4 / 5 (GPU work)** — blocked until RTX 5080 is procured, provisioned, validated.
- **Feature PRs** — paused until cleanup Weeks 1–3 are done.
- **PR #46** (GPU Phase 0 consolidation, draft since 2026-04-21) — needs land-or-close decision.

## 6) Next 3 actions in priority order (as of 2026-06-01)
1. **Land the keystone rename PR** (`langgraph/` → `agentic/`). This unblocks all of Week 2.
2. **Review the 3 prep docs** (AGENTIC_APP_MAP, COMPOSE_DRIFT, MAKEFILE_AUDIT) when they land.
3. **Dispatch first Week 2.1 split PR** — smallest leaf split from AGENTIC_APP_MAP (probably `prompts/` extraction — pure data, no behavior).

## 7) PM operating rules (commitments)
- **Document everything in the repo.** Chat sessions are ephemeral; the repo is forever.
- **One PR = one concern.** No big-bang refactors during cleanup.
- No drift between docs, status, and delivered code.
- Decision log is append-only; supersede with new entries.
- Before claiming any item is in flight, run the **diff-vs-README ritual**: verify against `README.md`, `CHANGELOG.md`, and current open PRs.
- After each merge, refresh `STATUS.md`, `CLEANUP_TODO.md`, and `SESSION_BACKLOG.md`.

## 8) How to resume after a context loss
If the chat resets or you (Dainius) come back tomorrow:
1. Open this file.
2. Open `kai-pm/CLEANUP_TODO.md` — see what's checked off.
3. Open `kai-pm/STATUS.md` — see current phase + next 3 actions.
4. Open https://github.com/dainius1234/kai-system/pulls — see what's in flight.
5. Say to your assistant: *"Resume brother — read SESSION_BOOTSTRAP and tell me the next move."*

## 9) Pointer index
- Live cleanup tracker: [CLEANUP_TODO.md](CLEANUP_TODO.md) ← **most important file right now**
- Status dashboard: [STATUS.md](STATUS.md)
- Phase sequence: [SEQUENCE.md](SEQUENCE.md)
- Append-only decisions: [DECISIONS.md](DECISIONS.md)
- Canonical roadmap: [STRATEGIC_PLAN.md](STRATEGIC_PLAN.md)
- Latest reality check: [REALITY_CHECK_2026-05-10.md](REALITY_CHECK_2026-05-10.md)
- Session backlog (running log): [../SESSION_BACKLOG.md](../SESSION_BACKLOG.md)
- Repo health audit: [REPO_HEALTH_AUDIT_2026-05-10.md](REPO_HEALTH_AUDIT_2026-05-10.md)
