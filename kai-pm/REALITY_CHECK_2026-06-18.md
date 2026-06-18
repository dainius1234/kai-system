# Reality Check — 2026-06-18 UTC

Purpose: reconcile PM brain (`SESSION_BOOTSTRAP.md`, `STATUS.md`, `CLEANUP_TODO.md`)
with actual repo/branch/PR state. Previous reality check (2026-05-10) is now over
five weeks stale and describes a sprint plan that did not execute as written.

## Headline finding: three divergent, unmerged attempts at the same problem

The "Week 2 — Untangle the Giant: split `agentic/app.py`" plan from the 2026-05-10/
2026-06-02 PM docs never landed on `main`. Instead, three separate efforts now exist,
none merged, none aware of each other until this audit:

| # | Where | Approach | State |
|---|---|---|---|
| 1 | `main` | — | Untouched. `agentic/app.py` is still the original monolith (no `prompts.py`, no `routes_*.py`, no `introspect_app.py`). |
| 2 | PR #67 `copilot/cleanup-sprint-week-2-1-prompts` | File-level split: pulled prompt strings into `agentic/prompts.py` | Open, **draft**, untouched since 2026-06-02. Never merged. |
| 3 | PR #69 `copilot/fix-core-tests-dockerfile` | File-level split, further than #67: `agentic/prompts.py`, `routes_identity.py`, `routes_observability.py`, `routes_ops.py`, `routes_skills.py` already exist on this branch | Open, **draft**. Opened to fix a stale `agentic/config.py` Dockerfile reference, but its branch already carries the bulk of a same-process route-module split. Never merged. |
| 4 | `claude/project-rework-plan-pgvp35` (this branch) | **Process-level** failure-domain split: `agentic` (hot: chat/run/checkpoints/skills) vs new `agentic-introspect` service (cold: dream/evolve/security-audit) | Committed and pushed here. Includes Phase 0 (trust-loop unification), Phase 0.5 (minimal stack gets a real `agentic`+`ollama`), Phase A (hot-path fix), Phase B (the split itself), and a sovereign-profile parity fix. **Not merged to `main`.** |

**Why this matters:** rows 3 and 4 both rewrite large parts of `agentic/app.py`,
in incompatible ways (file-reorg-in-one-process vs. process-split-into-two-services).
If both eventually merge to `main`, expect a substantial, non-mechanical conflict in
`agentic/app.py`. This needs an explicit decision (see Decisions Needed below), not a
default "whichever lands first" outcome.

`main` itself has none of this — including none of the Phase 0 trust-loop work
(unified conviction scale, PUB-mode real enforcement, irreversible-action floor),
which only exists on `claude/project-rework-plan-pgvp35` today.

## What is actually true right now (verified this session)

| Item | Old PM claim (2026-06-02 docs) | Actual (2026-06-18, verified) |
|---|---|---|
| `agentic/app.py` split | "Week 2.1 in progress, first split = `prompts/`" | Not on `main` at all. Two different unmerged drafts exist (#67, #69); a third, different approach is done on a feature branch (this one) |
| Current phase framing | "Phase 0 — Pre-GPU Hardening, Cleanup Sprint Week 1→2" | Cleanup Sprint as scoped (routes/state/flows/providers split, one-PR-per-slice) stalled; superseded in practice by the trust-loop + failure-domain work below |
| Trust loop (conviction/gate/PUB mode) | Not mentioned in 2026-06-02 docs | Unified to one 0–10 scale end-to-end, PUB mode is a real threshold not a hardcoded block, irreversible actions require conviction ≥9.0 + cosign (`claude/project-rework-plan-pgvp35` only — see DECISIONS D7) |
| Minimal stack (`docker-compose.minimal.yml`) | Not flagged as broken | Was missing `agentic`/`ollama` entirely — Chat was non-functional despite README claiming otherwise. Fixed on this branch (D8) |
| `agentic`/`ollama` inter-service HMAC auth in `full.yml` | Assumed working | Was silently broken (missing `INTERSERVICE_HMAC_SECRET` wiring) — fixed alongside the minimal-stack fix (D8) |
| `agentic` hot/cold coupling | Not identified | P13 snapshot capture was inline on the `/run` hot path; dream/evolve/security-audit lived in the same process as chat. Both fixed via Phase A/B on this branch |
| `docker-compose.sovereign.yml` | Assumed to intentionally omit `agentic` | Actually already runs `agentic` (just not `ollama`, per its own TODO) — the "intentionally omitted" framing in D8 was wrong; corrected this session, `agentic-introspect` added there too |
| Open PRs | 3 (#46, #54, #58) per 2026-05-10 reality check | 3 open now: #54 (chassis polish, draft since 2026-04-25, stale ~8 weeks), #67, #69 (both above). #46/#58 status not reconfirmed this pass — check live. |
| Stale remote branches | 6 (2026-05-10 count) | 33 branches on remote right now (`git ls-remote --heads origin`), most `copilot/*`, several clearly abandoned. Not cleaned up this session — flagged as a growing risk, not actioned. |

## Decisions needed (not made by this audit — flagging for Dainius)

1. **PR #67 and #69**: close, or extract anything salvageable before closing? Both are
   stale drafts (no activity since 2026-06-02) and both conflict in shape with the
   work already shipped on `claude/project-rework-plan-pgvp35`.
2. **Merge order for `claude/project-rework-plan-pgvp35` → `main`**: this branch is
   the only one with real, tested work (Phase 0/0.5/A/B). Recommend deciding its
   merge path *before* touching #67/#69, since merging this branch first makes the
   conflict in #67/#69 moot (they'd need a full rebase against a restructured
   `agentic/app.py` either way).
3. **Branch cleanup**: 33 remote branches is a lot of surface area to keep "situational
   awareness" over. Not blocking, but worth a dedicated pass.

## Validation run log

- `make check-docs`: PASSED (README/PROJECT_BACKLOG metrics current as of `fa18739`).
- `git diff origin/main..claude/project-rework-plan-pgvp35 --stat`: 28 files changed,
  1,136 insertions / 229 deletions — confirms this branch carries substantial,
  un-merged work beyond what `main` has.
- Verified via `git ls-tree -r origin/<branch> -- agentic/` for `main`, PR #69's head
  branch, and this branch — the three-way divergence above is a direct file-listing
  comparison, not inference from PR descriptions.

## Outstanding flags (still honest TBD)

- No live Docker daemon available in this environment — Phase 0.5 and Phase B are
  config/unit-test-validated only, not booted for real (carried over from prior
  reality, still true).
- Test coverage percentage still only measured for `common/` (78%, 2026-06-01) —
  repo-wide gate remains Week 3 scope, itself unstarted.
