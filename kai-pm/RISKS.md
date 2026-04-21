# Kai Risk Register

| ID | Description | Severity | Likelihood | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| R-001 | GPU procurement delay blocks Phases 1/2/4/5 and model-tier upgrades | H | H | Keep software-only sequence moving; predefine GPU bring-up checklist; review procurement weekly | @dainius1234 | Active |
| R-002 | Model deprecation or API behavior drift breaks planned model upgrades | H | M | Track model registry updates monthly in TECH_WATCH; keep fallback model map maintained | PM + Runtime maintainer | Active |
| R-003 | Scope creep across jewels causes roadmap slippage | M | H | Enforce locked sequence and jewel-scoped PRs; require explicit decision log entry for scope expansions | PM lead | Active |
| R-004 | Single-maintainer bus-factor creates continuity and response risk | H | M | Keep SESSION_BOOTSTRAP current; preserve append-only decisions; automate PM checks in GitHub | @dainius1234 | Active |
| R-005 | Dependency drift (Python/packages/actions) introduces silent breakage | M | M | Run regular dependency and workflow hygiene checks; keep changelog + tech watch updated | PM + Engineering | Active |
