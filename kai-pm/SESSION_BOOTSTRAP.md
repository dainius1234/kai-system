# Session Bootstrap

**Read this first. In 60 seconds you will know everything.**

> 📌 **If you are a new chat / new agent / future-Dainius reading this:**
> The single source of truth for "what's actually true right now" is
> [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md) — read it before
> trusting anything else in this directory, including the rest of this file.
> Then check open PRs. You will be productive in under 2 minutes.

---

## 1) Project one-liner (what is Kai)
Kai is a self-sovereign, local-first personal AI system built as cooperating services, designed to grow memory, reasoning, and operational reliability without external platform lock-in.

## 2) Current phase + current focus
- **Current phase:** Phase 0 — Pre-GPU Hardening.
- **Active branch with real, tested work:** `claude/project-rework-plan-pgvp35` —
  carries the trust-loop unification (conviction/gate/PUB mode), the minimal-stack
  fix (it now actually has `agentic`+`ollama`), and the `agentic` hot/cold split
  (Phase A: hot-path fix; Phase B: `agentic-introspect` service). **None of this is
  merged to `main` yet.**
- **The 2026-06-02 "Cleanup Sprint Week 1/2" plan referenced below did not execute
  as written.** The planned routes/state/flows/providers/prompts split of
  `agentic/app.py` stalled in two open draft PRs (#67, #69) that never merged.
  A different, better approach (process-level failure-domain split, not a file
  reorg) shipped instead, on the branch above. See
  [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md) for the full
  three-way divergence and the merge-order decision this creates.

## 3) Sprint state — superseded, see reality check

Everything below this line in the old "Sprint state (refreshed 2026-06-02)"
section was aspirational and never landed on `main`. Treat it as historical
record of intent, not current state:

- Week 1 housekeeping, keystone rename: actually merged, still true.
- Week 2.1/2.2 prep docs (`AGENTIC_APP_MAP.md`, `COMPOSE_DRIFT.md`): **never
  landed** — files do not exist in the repo today.
- Week 2.1 first split (`prompts/`): attempted in PR #67, then exceeded by PR #69's
  branch (which already has `routes_identity.py`, `routes_observability.py`,
  `routes_ops.py`, `routes_skills.py`). Both PRs are still open drafts, untouched
  since 2026-06-02.

**Where to check:** https://github.com/dainius1234/kai-system/pulls

## 4) What to actually do next (supersedes the old "4-week cleanup roadmap")

1. **Decide the merge order** for `claude/project-rework-plan-pgvp35` vs PR #67/#69
   — they restructure `agentic/app.py` in incompatible ways. Recommended: land this
   branch first (it's tested, documents its own tradeoffs, and closes real bugs),
   then decide whether #67/#69 still add value against the new structure or should
   be closed.
2. **Live-verify** what's only been config/unit-tested so far: boot
   `docker-compose.minimal.yml` and confirm `agentic-introspect` can die without
   taking `/chat`/`/run` down — no Docker daemon was available in the sessions that
   built this, so it's unverified end-to-end.
3. **Next monolith candidate**: `memu-core` (~6,100 lines), once the above lands —
   audit it for the same hot/cold coupling `agentic` had, don't assume it's clean.

## 5) Blocked items + unlock conditions
- **GPU-dependent phases** — blocked until RTX 5080 is procured, provisioned, validated.
- **PR #54** (chassis polish C2/C5/C9, open since 2026-04-25) — stale, needs a
  land-or-close decision, independent of the agentic-split decision above (it
  doesn't touch `agentic/app.py`).
- (PR #46, GPU Phase 0 consolidation, is **already merged** to `main` — corrected
  2026-06-18, was wrongly listed here as open in an earlier pass.)

## 6) PM operating rules (commitments)
- **Document everything in the repo.** Chat sessions are ephemeral; the repo is forever.
- **Reality checks are append-by-new-file, not edit-in-place** — write a new
  `REALITY_CHECK_<date>.md` when state has drifted, don't silently rewrite history.
- No drift between docs, status, and delivered code — if you find drift (like this
  one), fix the PM docs in the same session you found it, not "later."
- Decision log (`DECISIONS.md`) is append-only; supersede with new entries.

## 7) How to resume after a context loss
1. Open this file.
2. Open [`REALITY_CHECK_2026-06-18.md`](REALITY_CHECK_2026-06-18.md) — the actual
   current state, verified against branches/PRs, not assumed.
3. Open https://github.com/dainius1234/kai-system/pulls — see what's in flight.
4. Say to your assistant: *"Resume brother — read SESSION_BOOTSTRAP and the latest
   reality check and tell me the next move."*

## 8) Pointer index
- Latest reality check: [REALITY_CHECK_2026-06-18.md](REALITY_CHECK_2026-06-18.md) ← **most important file right now**
- Previous reality check (historical): [REALITY_CHECK_2026-05-10.md](REALITY_CHECK_2026-05-10.md)
- Status dashboard: [STATUS.md](STATUS.md)
- Live cleanup tracker (historical, superseded — see §3 above): [CLEANUP_TODO.md](CLEANUP_TODO.md)
- Phase sequence: [SEQUENCE.md](SEQUENCE.md)
- Append-only decisions: [DECISIONS.md](DECISIONS.md)
- Canonical roadmap: [STRATEGIC_PLAN.md](STRATEGIC_PLAN.md)
- Session backlog (running log): [../SESSION_BACKLOG.md](../SESSION_BACKLOG.md)
