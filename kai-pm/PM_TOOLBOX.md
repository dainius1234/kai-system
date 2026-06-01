# PM Toolbox

This file lists PM-facing tools and automations available in this repo.

## Read-only tools

- GitHub PR/issue/workflow views for status, drift checks, and CI triage.
- Repo docs for sequencing and context (`kai-pm/*.md`, `docs/CODEX_BRIEFS/*`).
- Workflow logs and run summaries for failure diagnosis.

## Write tools

- PR/issue templates under `.github/` for structured intake.
- Labels managed by `.github/labels.yml` + `labels-sync.yml`.
- `kai-pm/CLEANUP_TODO.md` as cleanup execution tracker.
- `kai-pm/PM_HANDOFF.md` for cross-session continuity.

## Agent dispatch and delegation

- Copilot coding agents can be dispatched for scoped cleanup tasks.
- PM can split work into reversible PRs and track status by labels.
- Hand-off docs preserve instructions and constraints between sessions.

## Automations

- `.github/workflows/pm-dashboard.yml`
  - Runs daily at 08:00 UTC and on manual dispatch.
  - Updates pinned issue: `📊 PM Dashboard — auto-updated daily`.
  - Reports open PR label counts, CI health, merge cadence, and TODO progress proxies.
- `.github/workflows/labels-sync.yml`
  - Syncs canonical PM labels from `.github/labels.yml`.

## CI gates and quality checks

- Existing CI workflows under `.github/workflows/` (core tests, python app, PM status).
- Pre-commit checks in `.pre-commit-config.yaml`.
- `scripts/check_pypi_shadow.sh` drift tripwire for root-folder package shadowing.

## Drift tripwires

- CODEOWNERS protections for high-risk paths.
- PM dashboard blocker rule (`pm-blocked`) when merges stall >5 days.
- Label taxonomy (`triage:*`, `cleanup-week-*`, `drift-watch`, `keystone`, `salvage-later`).
- PyPI-shadow guard with temporary cleanup-only allow override (`KAI_SHADOW_ALLOW`).
