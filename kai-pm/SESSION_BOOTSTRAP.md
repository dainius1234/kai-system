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
- **Current sprint (last refreshed 2026-06-02):** **Cleanup Sprint — Week 1 closing, Week 2 opening** — see `kai-pm/CLEANUP_TODO.md`
- **Why we paused features:** Audit on 2026-06-01 revealed 3 hot spots — `langgraph/` shadowed the PyPI package (now renamed ✅), `agentic/app.py` is 80 KB of mixed concerns, and ~100 Makefile targets with no honesty gate. Foundation must stabilise before we resume J-series / P-series work.

## 3) Sprint state (refreshed 2026-06-02)

**✅ Merged to main:**
- Week 1.4 KEYSTONE — `langgraph/` → `agentic/` rename
- Week 2.3 prep doc — `kai-pm/MAKEFILE_AUDIT.md`

**🟡 In flight / pending:**

| # | Task | Type | Status |
|---|---|---|---|
| 1 | Week 1 housekeeping (close #59 + #60, label PRs, delete tempo tests) | code | dispatched |
| 2 | Week 2.1 prep — `AGENTIC_APP_MAP.md` | docs | dispatched, not landed |
| 3 | Week 2.2 prep — `COMPOSE_DRIFT.md` | docs | dispatched, not landed |

**Where to check:** https://github.com/dainius1234/kai-system/pulls

## 4) The cleanup roadmap (4 weeks)
Live tracker: [`CLEANUP_TODO.md`](CLEANUP_TODO.md). Summary:

- **Week 1 — Stop the Bleeding** *(closing)*: PR housekeeping in flight; tempo tests being deleted; **keystone rename ✅ merged**.
- **Week 2 — Untangle the Giant** *(opening)*: split `agentic/app.py` (one PR per slice: routes / state / flows / providers / prompts), reconcile docker-compose, prune Makefile per `MAKEFILE_AUDIT.md`.
- **Week 3 — Honest Verification**: repo-wide coverage gate (currently 78% on `common/` only), honest `merge-gate`.
- **Week 4 — Resume Features** (only if 1–3 done): multi-backend LLM router, skills templates, journal templates, CIS P29.

## 5) Blocked items + unlock conditions
- **Phases 1 / 2 / 4 / 5 (GPU work)** — blocked until RTX 5080 is procured, provisioned, validated.
- **Feature PRs** — paused until cleanup Weeks 1–3 are done.
- **PR #46** (GPU Phase 0 consolidation, draft since 2026-04-21) — needs land-or-close decision.

## 6) Next 3 actions in priority order (as of 2026-06-02)
1. **Review + merge `AGENTIC_APP_MAP.md` and `COMPOSE_DRIFT.md`** when their PRs land. These are the design specs for Week 2.
2. **Dispatch first Week 2.1 split PR** — smallest leaf from AGENTIC_APP_MAP (probably `prompts/` extraction — pure data, no behavior). Sets the pattern for routes / state / flows / providers.
3. **Land-or-close PR #46** — stale draft from 2026-04-21 must be resolved.

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
- Makefile audit (Week 2.3 input): [MAKEFILE_AUDIT.md](MAKEFILE_AUDIT.md)
- Latest reality check: [REALITY_CHECK_2026-05-10.md](REALITY_CHECK_2026-05-10.md)
- Session backlog (running log): [../SESSION_BACKLOG.md](../SESSION_BACKLOG.md)
- Repo health audit: [REPO_HEALTH_AUDIT_2026-05-10.md](REPO_HEALTH_AUDIT_2026-05-10.md)
