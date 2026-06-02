# Makefile Audit

Prep work for Cleanup Sprint Week 2.3. This is a read-only audit of the current `/tmp/workspace/dainius1234/kai-system/Makefile`; no Makefile changes are proposed here, only recommendations.

## Scope and method

- Audited **123 targets** in the current Makefile.
- Cross-reference scan used the requested scope: `.github/workflows/`, `scripts/`, `docs/`, `kai-pm/`, and `README.md`, searching for `make <target>`.
- Baseline validation before writing this doc: strict flake8 pass ✅, `make go_no_go` ✅, broad repo pytest collection ❌ in this sandbox because service dependencies such as `fastapi` were not installed. That failure affects environment confidence, not the reference scan.
- **39 targets have zero external references** in the requested scan scope; those are strong archive/delete candidates.

## Inventory table

| Target | Lines | Description (from comment or inferred) | Last referenced (CI / docs / scripts / nowhere) | Recommendation (keep / archive / delete) | Reason |
|---|---|---|---|---|---|
| `self-audit` | `2-5` | Run repository self-audit script. | nowhere | **archive** | ARCHIVE — ad hoc audit helper with zero external refs; useful to keep in an archive, not in the primary Make UX. |
| `go_no_go` | `6-9` | Syntax-compile key service entry points and run go/no-go checks. | docs / scripts | **keep** | KEEP — documented in README/known issues, already used as a lightweight syntax gate, and explicitly references `langgraph/app.py`, so it needs the keystone rename PR before final cleanup. |
| `hardening_smoke` | `10-13` | Run hardening smoke checks. | nowhere | **archive** | ARCHIVE — useful one-off hardening check, but zero external refs and currently only pulled through a dishonest `merge-gate`. |
| `build-kai-control` | `14-17` | Install kai-control build deps and package the CLI with PyInstaller. | nowhere | **archive** | ARCHIVE — packaging helper with zero external refs; preserve in an archive if the standalone CLI still matters. |
| `kai-control-selftest` | `18-21` | Run kai-control self-test in test mode. | nowhere | **archive** | ARCHIVE — self-test for the archived kai-control packaging flow; not part of the core live loop. |
| `test-conviction` | `22-25` | Run conviction scoring tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `kai-drill` | `26-29` | Run the Kai drill shell workflow. | nowhere | **archive** | ARCHIVE — one-off drill workflow with zero external refs; preserve historically, but remove from the live surface area. |
| `kai-drill-test` | `30-33` | Run the Kai drill shell workflow in test mode. | nowhere | **archive** | ARCHIVE — one-off drill test mode with zero external refs; preserve historically, but remove from the live surface area. |
| `test-self-emp` | `34-37` | Run kai-advisor/self-employment tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `game-day-scorecard` | `38-41` | Generate the game-day scorecard. | docs | **archive** | ARCHIVE — command may still be useful historically, but it is not part of the curated day-to-day live target set. |
| `hmac-rotation-drill` | `42-45` | Run the HMAC rotation drill. | docs / scripts | **keep** | KEEP — active ops/security drill with dedicated runbook references; keep one canonical target. |
| `hmac-auto-rotate` | `46-49` | Run automatic HMAC key rotation helper. | nowhere | **archive** | ARCHIVE — command may still be useful historically, but it is not part of the curated day-to-day live target set. |
| `chaos-ci` | `50-53` | Run chaos/fault-injection CI helper. | nowhere | **archive** | ARCHIVE — fault-injection helper documented as on-demand in README, but not part of the slim live target set. |
| `hmac-migration-advice` | `54-57` | Print HMAC migration advice. | docs | **keep** | KEEP — active HMAC migration helper referenced in docs and still useful during rotation work. |
| `test-auth-hmac` | `58-61` | Run auth/HMAC hardening tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-phase-b-memu` | `62-64` | Run memu-core Phase B tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-memu-pg` | `65-69` | Run pgvector-backed memu tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-audio` | `70-72` | Run audio-service tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-camera` | `73-75` | Run camera-service tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-executor` | `76-78` | Run executor service tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-langgraph` | `79-81` | Run langgraph service tests. | docs | **archive** | ARCHIVE — useful service-specific test, but the target name is tied to the `langgraph/` directory and should be revisited only after the `langgraph/` → `agentic/` rename lands. |
| `test-kai-advisor` | `82-84` | Run kai-advisor tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-tts` | `85-87` | Run text-to-speech service tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-avatar` | `88-90` | Run avatar service tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-tool-gate` | `91-93` | Run tool-gate API tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-telegram` | `94-96` | Run Telegram bot tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-agentic` | `97-100` | Run agentic integration tests. | docs | **keep** | KEEP — this is the focused integration check for the agentic service; it survives the rename better than `test-langgraph` and is worth keeping visible. |
| `test-heartbeat` | `101-103` | Run heartbeat service tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-episode-saver` | `104-106` | Run episode saver tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-episode-spool` | `107-109` | Run episode spool tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-tool-gate-security` | `110-112` | Run tool-gate security tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-error-budget` | `113-115` | Run error-budget breaker tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-invoice` | `116-118` | Run invoice tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-dashboard` | `119-121` | Run dashboard tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-thinking-pathways` | `122-124` | Run thinking-pathways tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-memu-retrieval` | `125-127` | Run memu retrieval tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-router` | `128-130` | Run router tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-planner` | `131-133` | Run planner tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-adversary` | `134-136` | Run adversary tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-failure-taxonomy` | `137-139` | Run failure taxonomy tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-selaur` | `140-142` | Run SELAUR tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-contradiction` | `143-145` | Run contradiction tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-gem` | `146-148` | Run GEM preference tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-planner-prefs` | `149-151` | Run planner preference tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-silence` | `152-154` | Run silence-signal tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-self-deception` | `155-157` | Run self-deception tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-temporal-self` | `158-160` | Run temporal-self tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-predictive` | `161-163` | Run predictive tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-tempo` | `164-166` | Run tempo endpoint tests. | nowhere | **delete** | DELETE — PM docs already mark the tempo tests as orphaned/dead (`kai-pm/CLEANUP_TODO.md`, `kai-pm/REPO_HEALTH_AUDIT_2026-05-10.md`). |
| `test-improvement-gate` | `167-169` | Run improvement-gate tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-dream-state` | `170-172` | Run dream-state tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-security-audit` | `173-175` | Run security-audit tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-gaps-sprint` | `176-178` | Run gaps-sprint tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-tree-search` | `179-181` | Run tree-search tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-priority-queue` | `182-184` | Run priority-queue tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-model-selector` | `185-187` | Run model-selector tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-prod-hardening` | `188-190` | Run production hardening tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-hmac-rotation-drill` | `191-193` | Run the HMAC rotation drill test alias. | docs | **delete** | DELETE — exact duplicate alias of `hmac-rotation-drill`; keep one ops target, not both. |
| `test-p3-organic` | `194-196` | Run P3 organic memory tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p4-personality` | `197-199` | Run P4 personality tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p16-operational` | `200-202` | Run P16 operational tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p17-emotional-intelligence` | `203-205` | Run P17 emotional-intelligence tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p18-narrative-identity` | `206-208` | Run P18 narrative-identity tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p19-imagination-engine` | `209-211` | Run P19 imagination-engine tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p20-conscience-values` | `212-214` | Run P20 conscience/values tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p21-proactive-agent` | `215-217` | Run P21 proactive-agent tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-p22-operator-model` | `218-220` | Run P22 operator-model tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-h1-hardening` | `221-223` | Run H1 hardening tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-h2-self-healing` | `224-226` | Run H2 self-healing tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-mars-consolidation` | `227-229` | Run MARS consolidation tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-focus-compress` | `230-232` | Run focus-compress tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-context-budget` | `233-235` | Run context-budget tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-sage-critique` | `236-238` | Run SAGE critique tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-agent-evolver` | `239-241` | Run agent-evolver tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-checkpoint` | `242-244` | Run checkpoint tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-error-codes` | `245-247` | Run structured error-code tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-feature-flags` | `248-250` | Run feature-flag tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-predictive-failure` | `251-253` | Run predictive-failure tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-multi-modal` | `254-256` | Run multi-modal tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-world-anchor` | `257-259` | Run world-anchor tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-self-healing-phases` | `260-262` | Run self-healing phase tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-j-series` | `263-265` | Run the full J-series test module. | nowhere | **archive** | ARCHIVE — useful as historical/focused test coverage, but it has zero external refs and is not part of the proposed slim live target set. |
| `test-j2-wake-word` | `266-268` | Run only the J2 wake-word slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `test-j3-pii-redact` | `269-271` | Run only the J3 PII-redaction slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `test-j4-proactive-voice` | `272-274` | Run only the J4 proactive-voice slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `test-j5-memory-diary` | `275-277` | Run only the J5 memory-diary slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `test-j6-soul-agents` | `278-280` | Run only the J6 soul-agents slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `test-j7-skills-hub` | `281-283` | Run only the J7 skills-hub slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `test-wake` | `284-286` | Run wake-intent tests. | nowhere | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-behavioral` | `287-289` | Run behavioral tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-docker-e2e` | `290-292` | Run Docker end-to-end tests. | docs | **archive** | ARCHIVE — test still exists, but it is only doc-listed (or unreferenced) and does not need a first-class top-level Make target once the live list is trimmed. |
| `test-chassis` | `293-295` | Run chassis tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-j1-live-canvas` | `296-298` | Run only the J1 live-canvas slice of J-series tests. | nowhere | **delete** | DELETE — filtered alias of `test-j-series`; redundant noise in the live Makefile. |
| `dep-audit` | `299-301` | Run pip-audit against installed dependencies. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `coverage` | `302-305` | Run pytest with common/ coverage reporting. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `test-v7-verifier` | `306-308` | Run v7 verifier tests. | nowhere | **archive** | ARCHIVE — leaf of the `test-v7` aggregate; keep the aggregate live and archive the sub-targets. |
| `test-v7-quarantine` | `309-311` | Run v7 quarantine tests. | nowhere | **archive** | ARCHIVE — leaf of the `test-v7` aggregate; keep the aggregate live and archive the sub-targets. |
| `test-v7-policy` | `312-314` | Run v7 policy and rate-limit tests. | nowhere | **archive** | ARCHIVE — leaf of the `test-v7` aggregate; keep the aggregate live and archive the sub-targets. |
| `test-v7-idempotency` | `315-317` | Run v7 idempotency tests. | nowhere | **archive** | ARCHIVE — leaf of the `test-v7` aggregate; keep the aggregate live and archive the sub-targets. |
| `test-integration-chain` | `318-320` | Run v7 integration-chain tests. | nowhere | **archive** | ARCHIVE — leaf of the `test-v7` aggregate; keep the aggregate live and archive the sub-targets. |
| `test-v7` | `321-323` | Aggregate the v7 test family. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-core` | `324-325` | Aggregate the current test-core dependency list. | CI / docs / scripts | **keep** | KEEP — CI directly invokes it in `.github/workflows/core-tests.yml`; this is the one aggregate test target that must remain live. |
| `test-dashboard-ui` | `326-328` | Run dashboard UI tests. | docs | **keep** | KEEP — focused high-value regression target that is worth invoking directly even after the Makefile is slimmed down. |
| `test-integration` | `329-332` | Run the core integration smoke script. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `full-up` | `333-335` | Bring up the full docker-compose stack. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `full-down` | `336-339` | Tear down the full docker-compose stack. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `health-sweep` | `340-343` | Run /health checks against running services. | docs | **keep** | KEEP — lightweight operational smoke target documented in README and scripts/self_audit.py. |
| `contract-smoke` | `344-349` | Run contract smoke tests. | nowhere | **archive** | ARCHIVE — useful smoke helper, but zero external refs and not part of the proposed core live list. |
| `merge-gate` | `350-369` | Run the current pre-merge validation chain. | docs / scripts | **keep** | KEEP — it is still the public pre-merge entry point, but its composition must be rewritten to call only honest validation targets. |
| `phase1-closure` | `370-373` | Run the Phase 1 closure check. | nowhere | **archive** | ARCHIVE — milestone-specific closure script with zero current refs; keep for history if needed, but move out of the live Makefile. |
| `core-up` | `374-377` | Bring up the minimal/core docker-compose stack. | docs / scripts | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `core-down` | `378-381` | Tear down the minimal/core docker-compose stack. | docs / scripts | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `core-smoke` | `382-385` | Run quick health/smoke checks for core services. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `init-memu-db` | `386-390` | Initialize the memu-core database schema. | nowhere | **keep** | KEEP — practical local/dev database bootstrap target even though it lacks docs references. |
| `paper-backup` | `391-393` | Run the monthly paper-backup script. | nowhere | **archive** | ARCHIVE — operational script exists, but there are zero external refs in the requested scan scope and it should not sit in the day-to-day live Makefile. |
| `weekly-key-rotate` | `394-397` | Run the weekly key-rotation script. | nowhere | **archive** | ARCHIVE — operational script exists, but there are zero external refs in the requested scan scope and it should not sit in the day-to-day live Makefile. |
| `weekly-ed25519-rotate` | `398-400` | Run the weekly Ed25519 rotation script. | nowhere | **archive** | ARCHIVE — operational script exists, but there are zero external refs in the requested scan scope and it should not sit in the day-to-day live Makefile. |
| `setup` | `401-403` | Run repository setup/bootstrap script. | scripts | **keep** | KEEP — bootstrap command that is still referenced from `scripts/setup.sh` and belongs in the live developer entry points. |
| `sync-docs` | `404-406` | Synchronize README/backlog metrics from the repo state. | docs / scripts | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `check-docs` | `407-409` | Check whether docs/metrics are stale. | docs | **keep** | KEEP — part of the documented core developer loop or required as a top-level aggregate/entry point. |
| `auto-changelog` | `410-412` | Auto-generate changelog updates. | nowhere | **archive** | ARCHIVE — zero external refs and not part of the core dev loop; keep for historical automation if desired. |
| `auto-session-log` | `413-416` | Auto-generate session-log updates. | nowhere | **archive** | ARCHIVE — zero external refs and not part of the core dev loop; keep for historical automation if desired. |
| `cache-test-core` | `417-418` | Cache and compare test-core results across runs. | nowhere | **delete** | DELETE — stale helper: hard-coded list covers only 50 of 74 `test-core` targets and duplicates `test-mars-consolidation`, so it is actively misleading. |

## Categorization

### Core dev loop — KEEP

These are the commands worth keeping front-and-center for everyday development and review:

- `setup`
- `core-up`
- `core-down`
- `core-smoke`
- `full-up`
- `full-down`
- `go_no_go`
- `sync-docs`
- `check-docs`
- `dep-audit`
- `coverage`
- `test-core`
- `test-integration`
- `test-phase-b-memu`
- `test-memu-pg`
- `test-dashboard`
- `test-dashboard-ui`
- `test-tool-gate`
- `test-tool-gate-security`
- `test-agentic`
- `test-v7`
- `test-chassis`
- `merge-gate`

### CI / merge-gate — KEEP

These are explicitly part of current CI or should remain the top-level gating entry points:

- `test-core`
- `go_no_go`
- `check-docs`
- `merge-gate`

### Operations — KEEP if active

Keep these only if they still correspond to an active runbook / drill; otherwise they can move out later:

- `health-sweep`
- `hmac-rotation-drill`
- `hmac-migration-advice`

### One-shot historical — ARCHIVE

The large majority of single-purpose tests and sprint-era helpers fit better in `Makefile.archive` than in the day-to-day live Makefile. They remain callable, but they stop crowding the top-level UX.

`self-audit`, `hardening_smoke`, `build-kai-control`, `kai-control-selftest`, `kai-drill`, `kai-drill-test`, `game-day-scorecard`, `hmac-auto-rotate`, `chaos-ci`, `test-conviction`, `test-self-emp`, `test-auth-hmac`, `test-audio`, `test-camera`, `test-executor`, `test-langgraph`, `test-kai-advisor`, `test-tts`, `test-avatar`, `test-telegram`, `test-heartbeat`, `test-episode-saver`, `test-episode-spool`, `test-error-budget`, `test-invoice`, `test-thinking-pathways`, `test-memu-retrieval`, `test-router`, `test-planner`, `test-adversary`, `test-failure-taxonomy`, `test-selaur`, `test-contradiction`, `test-gem`, `test-planner-prefs`, `test-silence`, `test-self-deception`, `test-temporal-self`, `test-predictive`, `test-improvement-gate`, `test-dream-state`, `test-security-audit`, `test-gaps-sprint`, `test-tree-search`, `test-priority-queue`, `test-model-selector`, `test-prod-hardening`, `test-p3-organic`, `test-p4-personality`, `test-p16-operational`, `test-p17-emotional-intelligence`, `test-p18-narrative-identity`, `test-p19-imagination-engine`, `test-p20-conscience-values`, `test-p21-proactive-agent`, `test-p22-operator-model`, `test-h1-hardening`, `test-h2-self-healing`, `test-mars-consolidation`, `test-focus-compress`, `test-context-budget`, `test-sage-critique`, `test-agent-evolver`, `test-checkpoint`, `test-error-codes`, `test-feature-flags`, `test-predictive-failure`, `test-multi-modal`, `test-world-anchor`, `test-self-healing-phases`, `test-j-series`, `test-wake`, `test-behavioral`, `test-docker-e2e`, `contract-smoke`, `phase1-closure`, `paper-backup`, `weekly-key-rotate`, `weekly-ed25519-rotate`, `auto-changelog`, `auto-session-log`, `test-v7-verifier`, `test-v7-quarantine`, `test-v7-policy`, `test-v7-idempotency`, `test-integration-chain`

### Dead — DELETE

- `test-tempo`
- `cache-test-core`

### Duplicates / aliases — consolidate

- `test-hmac-rotation-drill`
- `test-j1-live-canvas`
- `test-j2-wake-word`
- `test-j3-pii-redact`
- `test-j4-proactive-voice`
- `test-j5-memory-diary`
- `test-j6-soul-agents`
- `test-j7-skills-hub`

## Cross-reference scan

Every target below was scanned for `make <target>` references in `.github/workflows/`, `scripts/`, `docs/`, `kai-pm/`, and `README.md`.

| Target | Referenced in |
|---|---|
| `self-audit` | _none found_ |
| `go_no_go` | `README.md`, `docs/PROJECT_BACKLOG.md`, `docs/known_issues.md`, `scripts/setup.sh` |
| `hardening_smoke` | _none found_ |
| `build-kai-control` | _none found_ |
| `kai-control-selftest` | _none found_ |
| `test-conviction` | `README.md` |
| `kai-drill` | _none found_ |
| `kai-drill-test` | _none found_ |
| `test-self-emp` | `README.md` |
| `game-day-scorecard` | `docs/hmac_rotation_runbook.md` |
| `hmac-rotation-drill` | `docs/hmac_rotation_runbook.md`, `docs/next_level_roadmap.md`, `scripts/hmac_rotation_drill.py` |
| `hmac-auto-rotate` | _none found_ |
| `chaos-ci` | _none found_ |
| `hmac-migration-advice` | `docs/hmac_rotation_runbook.md` |
| `test-auth-hmac` | `README.md` |
| `test-phase-b-memu` | `README.md` |
| `test-memu-pg` | `README.md` |
| `test-audio` | `README.md` |
| `test-camera` | `README.md` |
| `test-executor` | `README.md` |
| `test-langgraph` | `README.md` |
| `test-kai-advisor` | `README.md` |
| `test-tts` | `README.md` |
| `test-avatar` | `README.md` |
| `test-tool-gate` | `README.md` |
| `test-telegram` | `README.md` |
| `test-agentic` | `README.md` |
| `test-heartbeat` | `README.md` |
| `test-episode-saver` | `README.md` |
| `test-episode-spool` | `README.md` |
| `test-tool-gate-security` | `README.md` |
| `test-error-budget` | `README.md` |
| `test-invoice` | `README.md` |
| `test-dashboard` | `README.md` |
| `test-thinking-pathways` | `README.md` |
| `test-memu-retrieval` | `README.md` |
| `test-router` | `README.md` |
| `test-planner` | `README.md` |
| `test-adversary` | `README.md` |
| `test-failure-taxonomy` | `README.md` |
| `test-selaur` | `README.md` |
| `test-contradiction` | `README.md` |
| `test-gem` | _none found_ |
| `test-planner-prefs` | _none found_ |
| `test-silence` | _none found_ |
| `test-self-deception` | _none found_ |
| `test-temporal-self` | _none found_ |
| `test-predictive` | `README.md` |
| `test-tempo` | _none found_ |
| `test-improvement-gate` | _none found_ |
| `test-dream-state` | `README.md` |
| `test-security-audit` | `README.md` |
| `test-gaps-sprint` | _none found_ |
| `test-tree-search` | `README.md` |
| `test-priority-queue` | `README.md` |
| `test-model-selector` | `README.md` |
| `test-prod-hardening` | `README.md` |
| `test-hmac-rotation-drill` | `README.md` |
| `test-p3-organic` | `README.md` |
| `test-p4-personality` | `README.md` |
| `test-p16-operational` | `README.md` |
| `test-p17-emotional-intelligence` | `README.md` |
| `test-p18-narrative-identity` | `README.md` |
| `test-p19-imagination-engine` | `README.md` |
| `test-p20-conscience-values` | `README.md` |
| `test-p21-proactive-agent` | `README.md` |
| `test-p22-operator-model` | `README.md` |
| `test-h1-hardening` | `README.md` |
| `test-h2-self-healing` | `README.md` |
| `test-mars-consolidation` | `README.md` |
| `test-focus-compress` | `README.md` |
| `test-context-budget` | `README.md` |
| `test-sage-critique` | `README.md` |
| `test-agent-evolver` | `README.md` |
| `test-checkpoint` | `README.md` |
| `test-error-codes` | `README.md` |
| `test-feature-flags` | `README.md` |
| `test-predictive-failure` | `README.md` |
| `test-multi-modal` | `README.md` |
| `test-world-anchor` | `README.md` |
| `test-self-healing-phases` | `README.md` |
| `test-j-series` | _none found_ |
| `test-j2-wake-word` | _none found_ |
| `test-j3-pii-redact` | _none found_ |
| `test-j4-proactive-voice` | _none found_ |
| `test-j5-memory-diary` | _none found_ |
| `test-j6-soul-agents` | _none found_ |
| `test-j7-skills-hub` | _none found_ |
| `test-wake` | _none found_ |
| `test-behavioral` | `README.md` |
| `test-docker-e2e` | `README.md` |
| `test-chassis` | `README.md` |
| `test-j1-live-canvas` | _none found_ |
| `dep-audit` | `README.md` |
| `coverage` | `README.md`, `kai-pm/STATUS.md` |
| `test-v7-verifier` | _none found_ |
| `test-v7-quarantine` | _none found_ |
| `test-v7-policy` | _none found_ |
| `test-v7-idempotency` | _none found_ |
| `test-integration-chain` | _none found_ |
| `test-v7` | `README.md` |
| `test-core` | `.github/workflows/core-tests.yml`, `README.md`, `docs/PROJECT_BACKLOG.md`, `docs/agentic_patterns_spec.md`, `docs/unfair_advantages.md`, `scripts/setup.sh`, `scripts/sync_docs.py` |
| `test-dashboard-ui` | `README.md` |
| `test-integration` | `README.md` |
| `full-up` | `README.md` |
| `full-down` | `README.md` |
| `health-sweep` | `README.md` |
| `contract-smoke` | _none found_ |
| `merge-gate` | `README.md`, `docs/gpu_integration_phase0.md`, `scripts/setup.sh` |
| `phase1-closure` | _none found_ |
| `core-up` | `README.md`, `docs/PROJECT_BACKLOG.md`, `docs/gpu_integration_phase0.md`, `scripts/setup.sh` |
| `core-down` | `README.md`, `docs/PROJECT_BACKLOG.md`, `docs/gpu_integration_phase0.md`, `scripts/setup.sh` |
| `core-smoke` | `README.md` |
| `init-memu-db` | _none found_ |
| `paper-backup` | _none found_ |
| `weekly-key-rotate` | _none found_ |
| `weekly-ed25519-rotate` | _none found_ |
| `setup` | `scripts/setup.sh` |
| `sync-docs` | `README.md`, `docs/known_issues.md`, `kai-pm/METRICS.md`, `kai-pm/NAVIGATION.md`, `kai-pm/REALITY_CHECK_2026-05-10.md`, `scripts/sync_docs.py` |
| `check-docs` | `README.md`, `kai-pm/NAVIGATION.md`, `kai-pm/REALITY_CHECK_2026-05-10.md` |
| `auto-changelog` | _none found_ |
| `auto-session-log` | _none found_ |
| `cache-test-core` | _none found_ |

## Proposed live target list

Proposed live Makefile surface area after pruning: **27 targets** (close enough to the target “~25”, while still leaving a usable dev loop + a few active ops hooks).

- `setup` — Bootstrap a fresh clone.
- `init-memu-db` — Initialize memu-core schema for PostgreSQL/pgvector dev runs.
- `core-up` — Start the minimal local stack.
- `core-down` — Stop the minimal local stack.
- `core-smoke` — Run quick health checks against the minimal stack.
- `full-up` — Start the full stack.
- `full-down` — Stop the full stack.
- `go_no_go` — Fast syntax/entry-point gate before deeper testing.
- `sync-docs` — Refresh README/backlog metrics from source-of-truth code.
- `check-docs` — Read-only doc freshness gate.
- `dep-audit` — Check Python dependencies for known CVEs.
- `coverage` — Produce the current pytest coverage report.
- `test-core` — Run the curated aggregate test suite used by CI.
- `test-integration` — Run the core integration smoke script.
- `test-phase-b-memu` — Focused memu-core regression suite.
- `test-memu-pg` — Focused pgvector persistence regression suite.
- `test-dashboard` — Focused dashboard backend regression suite.
- `test-dashboard-ui` — Focused dashboard UI regression suite.
- `test-tool-gate` — Focused tool-gate API regression suite.
- `test-tool-gate-security` — Focused tool-gate security regression suite.
- `test-agentic` — Focused agentic integration suite.
- `test-v7` — Focused v7 aggregate suite.
- `test-chassis` — Higher-level chassis/LLM-layer regression suite.
- `hmac-rotation-drill` — Canonical HMAC rotation drill target.
- `hmac-migration-advice` — Operator helper for HMAC migrations.
- `health-sweep` — Operational health sweep for running services.
- `merge-gate` — Single honest pre-merge entry point after cleanup.

## Proposed `merge-gate` composition

Current `merge-gate` is dishonest because it mixes validation with side-effectful operational tasks (paper backup, key rotation) and still does not cover the true live validation surface. After cleanup, `make merge-gate` should be composed only of honest validation targets:

1. `make go_no_go`
2. `make check-docs`
3. `make dep-audit`
4. `make test-core`
5. `make test-integration`
6. `make coverage` **only after** Week 3 establishes a repo-wide honest coverage gate (today it covers only `common/`)

Notes:

- `health-sweep` is useful, but it should stay an on-demand ops target rather than a merge gate requirement.
- `hmac-rotation-drill`, `paper-backup`, `weekly-key-rotate`, and `weekly-ed25519-rotate` should not run inside `merge-gate`; they have side effects or represent operator drills, not code validation.
- If docker-based smoke checks are required, prefer a separate CI job (`core-up` → `core-smoke` → `core-down`) rather than hiding them inside `merge-gate`.

## Rename note: `langgraph/` → `agentic/` keystone dependency

Do not block this audit on the rename, but note these cleanup dependencies:

- `go_no_go` explicitly compiles `langgraph/app.py`; its command must be updated only after the keystone rename PR lands.
- `test-langgraph` is named after the old service directory and is a good archive/rename candidate once `langgraph/` becomes `agentic/`.
- `test-core` currently includes `test-langgraph`; that dependency list should be revisited as part of the same post-rename Makefile cleanup PR.

## Suggested PR sequence

1. **PR 1 — Delete obviously dead targets**
   - Delete `test-tempo` and `cache-test-core` first.
   - Delete alias-only noise: `test-hmac-rotation-drill`, `test-j1-live-canvas`, `test-j2-wake-word`, `test-j3-pii-redact`, `test-j4-proactive-voice`, `test-j5-memory-diary`, `test-j6-soul-agents`, `test-j7-skills-hub`.
2. **PR 2 — Move historical targets to `Makefile.archive`**
   - Move the one-shot sprint helpers, phase/jewel targets, and niche subsystem tests that are no longer part of the curated live loop.
   - Keep the underlying scripts/tests in place; only slim the top-level Makefile surface area.
3. **PR 3 — Consolidate duplicates / aliases**
   - Collapse leaf aggregates under `test-v7` and decide whether any archived J-series wrapper should survive at all.
   - Rename/archive `test-langgraph` after the keystone `langgraph/` → `agentic/` PR lands.
4. **PR 4 — Make `merge-gate` honest**
   - Recompose `merge-gate` around validation-only targets (`go_no_go`, `check-docs`, `dep-audit`, `test-core`, `test-integration`, and later an honest coverage gate).
   - Remove operational side effects from the gate.

