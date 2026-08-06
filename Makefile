# Offline/CI default: use hash-based fake embeddings so tests that don't care
# about embedding quality can run without sentence-transformers installed.
# Override with MEMU_ALLOW_FAKE_EMBEDDINGS=false for real-embedding tests.
export MEMU_ALLOW_FAKE_EMBEDDINGS ?= true

# Self-audit and feedback
self-audit:
	python3 scripts/self_audit.py
.PHONY: lint-blocking test-d109-ohana-core test-d92-socratic test-d93-hypothesis test-d94-forecaster test-d95-d100-foundations test-d101-causal-world-model test-d102-global-workspace test-d89-cognitive-depth go_no_go hardening_smoke pypi-shadow-check test-letta test-financial test-agentic-service test-agentic-introspect build-kai-control kai-control-selftest test-conviction kai-drill kai-drill-test test-self-emp game-day-scorecard hmac-rotation-drill hmac-auto-rotate hmac-migration-advice test-auth-hmac test-phase-b-memu chaos-ci health-sweep contract-smoke merge-gate phase1-closure paper-backup weekly-key-rotate weekly-ed25519-rotate core-up core-down core-smoke test-v7-verifier test-v7-quarantine test-v7-policy test-v7-idempotency test-integration-chain test-v7 test-heartbeat test-episode-saver test-episode-spool test-tool-gate-security test-error-budget test-invoice test-dashboard test-memu-retrieval test-memu-routes test-agentic-routes test-context-enrichment-ab test-ab-log test-screen-capture test-shell-sandbox test-hse-rams test-agentic test-router test-planner test-adversary test-failure-taxonomy test-selaur test-contradiction test-gem test-planner-prefs test-silence test-self-deception test-temporal-self test-predictive test-improvement-gate test-thinking-pathways test-dream-state test-security-audit test-gaps-sprint test-tree-search test-priority-queue test-model-selector test-prod-hardening test-p3-organic test-p4-personality test-p16-operational test-p17-emotional-intelligence test-p18-narrative-identity test-p19-imagination-engine test-p20-conscience-values test-p21-proactive-agent test-p22-operator-model test-error-codes test-feature-flags dep-audit coverage coverage-floors test-restart-persistence test-upload-fuzz test-audio-transcribe test-browser-agent test-vision-service test-clipboard-service test-files-service test-notify-service test-document-parser test-monitor-service test-broker-bridge test-sysmetrics test-screen-watcher test-email-reader test-news-feed test-weather-service test-docker-watcher test-airquality test-calendar-service test-git-watcher test-kai-intelligence test-cognitive-mechanisms sync-docs check-docs auto-changelog auto-session-log test-focus-compress test-context-budget test-predictive-failure test-multi-modal test-world-anchor test-self-healing-phases test-j-series test-behavioral test-docker-e2e test-chassis test-wake test-broker-bridge-yfinance test-perception-spine test-world-state test-proposal-workspace test-policy-bridge test-vertical-slice test-actuator-registry test-autonomy test-payload-bounds test-assessment test-invariant-guards test-concurrency-clock test-service-auth test-erasure test-legacy-bridge test-migration test-full-migration test-flags-enabled test-preflight test-architecture-rules test-dashboard-findings test-dashboard-auth test-dashboard-ui-auth test-degraded test-hygiene-gate test-assertion-floors test-gate-registry test-compose-drift test-secret-gates test-compose-gates test-ci-tolerations test-test-wiring test-test-isolation test-suite-floor test-conftest-guards suite-run suite-floor suite-floor-record test-isolation test-isolation-report test-isolation-baseline test-trust-ladder test-market test-cognition test-j-features test-perception-misc test-hmac-advisor test-untargeted test-smoke-core test-graph-live dashboard-findings hygiene-survey hygiene-gate hygiene-baseline verify-live-endpoints verify-live-mutating setup-service-token preflight test-uh assertion-floors assertion-floors-update gate-registry gate-registry-gate

# `lint-blocking` runs FIRST and byte-identically to python-app.yml's flake8
# step. That step gates CI *ahead* of the tests, so a file that fails it
# stops the suite from running at all — which is how a broken edit of mine
# reached `main` while `make policy-check`, `make test-uh` and 4,208 local
# tests were all green. A local gate that does not include what CI checks
# first is a gate with a hole in exactly the shape of the thing that bit.
lint-blocking:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics \
	  --exclude=.venv,_archive,__pycache__

policy-check: lint-blocking
	python3 scripts/security/check_port_bindings.py
	python3 scripts/security/check_default_profiles.py
	python3 scripts/security/check_secret_fallbacks.py
	python3 scripts/security/check_service_tokens.py
	python3 scripts/security/check_network_zones.py
	python3 scripts/security/check_turbovec_writers.py
	python3 scripts/security/check_restart_recovery.py
	python3 scripts/security/check_image_tags.py
	python3 scripts/security/check_compose_drift.py
	python3 scripts/security/check_architecture_rules.py
	python3 scripts/security/hygiene_survey.py --gate
	python3 scripts/security/check_ci_tolerations.py
	python3 scripts/security/check_workflow_filters.py
	python3 scripts/security/check_workflow_outputs.py
	python3 scripts/security/check_dockerfile_flags.py
	python3 scripts/security/check_dockerfile_coverage.py
	python3 scripts/security/check_dockerfile_context.py
	python3 scripts/security/check_compose_env.py
	python3 scripts/security/check_compose_interpolation.py
	python3 scripts/security/check_unreachable_bindings.py
	python3 scripts/security/check_implicit_deps.py
	python3 scripts/security/check_image_modules.py
	python3 scripts/security/check_test_identity.py
	python3 scripts/security/check_test_wiring.py
	python3 scripts/security/check_gate_registry.py --gate
	@# Doc-drift last: it fails on every commit that adds a test, so it
	@# must be enforced HERE rather than remembered. core-tests.yml
	@# failed 30 consecutive runs on this check and skipped its other 50
	@# steps — every service test and the whole Docker stack — because
	@# nothing made it visible before the push.
	$(MAKE) --no-print-directory check-docs

test-contracts:
	PYTHONPATH=. python scripts/test_contracts.py

test-perception-spine:
	PYTHONPATH=. python scripts/test_perception_spine.py

test-world-state:
	PYTHONPATH=. python scripts/test_world_state.py

test-proposal-workspace:
	PYTHONPATH=. python scripts/test_proposal_workspace.py

test-policy-bridge:
	PYTHONPATH=. python scripts/test_policy_bridge.py

test-vertical-slice:
	PYTHONPATH=. python scripts/test_vertical_slice.py

test-actuator-registry:
	PYTHONPATH=. python scripts/test_actuator_registry.py

test-autonomy:
	PYTHONPATH=. python scripts/test_autonomy.py

test-payload-bounds:
	PYTHONPATH=. python scripts/test_payload_bounds.py

test-assessment:
	PYTHONPATH=. python scripts/test_assessment.py

test-invariant-guards:
	PYTHONPATH=. python scripts/test_invariant_guards.py

test-concurrency-clock:
	PYTHONPATH=. python scripts/test_concurrency_clock.py

test-service-auth:
	PYTHONPATH=. python scripts/test_service_auth.py

test-erasure:
	PYTHONPATH=. python scripts/test_erasure.py

test-legacy-bridge:
	PYTHONPATH=. python scripts/test_legacy_bridge.py

test-migration:
	PYTHONPATH=. python scripts/test_migration.py

test-full-migration:
	PYTHONPATH=. python scripts/test_full_migration.py

test-flags-enabled:
	PYTHONPATH=. python scripts/test_flags_enabled.py

test-preflight:
	PYTHONPATH=. python scripts/test_preflight.py

test-architecture-rules:
	PYTHONPATH=. python scripts/test_architecture_rules.py

test-dashboard-findings:
	PYTHONPATH=. python scripts/test_dashboard_findings.py

test-dashboard-auth:
	PYTHONPATH=. python scripts/test_dashboard_auth.py

# Browser-side credential shim (KAI-DASH-D01). Needs node.
test-dashboard-ui-auth:
	node scripts/test_dashboard_ui_auth.js

# Track D: an outage must not be mistaken for an answer.
test-degraded:
	PYTHONPATH=. python scripts/test_degraded.py

# H-5: the hygiene ratchet must be able to fail.
test-hygiene-gate:
	PYTHONPATH=. python scripts/test_hygiene_gate.py

# A-02: the assertion ratchet must be able to fail, on synthetic inputs.
test-assertion-floors:
	PYTHONPATH=. python scripts/test_assertion_floors.py

# A-04: the instrumentation meta-check must be able to fail, on synthetic
# registries. This is the terminus of the depth-one recursion.
test-gate-registry:
	PYTHONPATH=. python scripts/test_gate_registry.py

# A-04b: the drift ratchet must fire on a weaker profile — and must NOT
# fire on a stricter one, or it would push toward weakening sovereign.
test-compose-drift:
	PYTHONPATH=. python scripts/test_compose_drift.py

# A-04b: the secret rule and the restart allowlist, both of which were
# previously unable to fail. KAI-GATE-007 and 008.
test-secret-gates:
	PYTHONPATH=. python scripts/test_secret_gates.py

# A-04b: ports, network zones and image tags. KAI-GATE-010/011/012.
test-compose-gates:
	PYTHONPATH=. python scripts/test_compose_gates.py

# A-04c: CI suppressions must be declared, and workflows must parse.
test-ci-tolerations:
	PYTHONPATH=. python scripts/test_ci_tolerations.py

# A-04d: a test that is never called is not a test.
test-test-wiring:
	PYTHONPATH=. python scripts/test_test_wiring.py

# A-05: no test file may change the interpreter for the files after it.
test-test-isolation:
	PYTHONPATH=. python scripts/test_test_isolation.py

# KAI-GATE-020: the repo-wide result may not regress.
test-suite-floor:
	PYTHONPATH=. python scripts/test_suite_floor.py

# The global test guards in conftest.py (ML stack blocked).
test-conftest-guards:
	PYTHONPATH=. python -m pytest scripts/test_conftest_guards.py -q

# The one command that defines the floor's population.
#
# The floor compares a number against a recorded number, so both have to
# come from the same command or the comparison is meaningless. On
# 2026-08-05 they did not: CI --ignores the two dashboard files (they get
# their own step, run from dashboard/), a local `pytest scripts/` does
# not, and the gate read a 24-test difference in *invocation* as 24
# deleted tests. Same denominator mismatch as everything else this week,
# in the gate built to catch regressions.
#
# So the invocation lives here, once, and both CI and a developer call
# this target. Changing what the suite covers is now an edit to this
# line, in a diff, rather than a drift between two files.
suite-run:
	PYTHONPATH=. python3 -m pytest \
	  -p scripts.security.isolation_plugin \
	  --ignore=_archive --ignore=.venv \
	  --ignore=scripts/test_dashboard.py \
	  --ignore=scripts/test_dashboard_ui.py \
	  $(PYTEST_EXTRA) \
	  -q > .pytest-run.log 2>&1; rc=$$?; cat .pytest-run.log; exit $$rc

# Check a captured run against the floor.
suite-floor:
	python3 scripts/security/check_suite_floor.py --from-log .pytest-run.log

# Ratchet it. Refuses to loosen.
suite-floor-record:
	python3 scripts/security/check_suite_floor.py --from-log .pytest-run.log --record

# The gate itself. Runs the whole pytest suite with the isolation plugin
# attached, because the question "does this file affect the next one?" is
# only answerable in a real session. Minutes, not seconds.
test-isolation:
	python3 scripts/security/check_test_isolation.py

# Consume a report the test run already produced, rather than running the
# suite twice. This is what CI uses.
test-isolation-report:
	python3 scripts/security/check_test_isolation.py --from-report .isolation-report.json

# Record current leakage. Refuses to record a replaced module.
test-isolation-baseline:
	python3 scripts/security/check_test_isolation.py --write-baseline

# ── A-05: the 32 scripts that no target named ────────────────────────
# They were collected by the repo-wide pytest, which is how they ran at
# all — but that run was aborting, so in practice they ran nowhere, and
# none of them could be run on its own to find out. Grouped by subject so
# the list stays legible; `test-untargeted` runs every one.
test-trust-ladder:
	PYTHONPATH=. python -m pytest scripts/test_trust_core.py scripts/test_trust_ledger.py \
	  scripts/test_trust_integration.py scripts/test_trust_promotion.py \
	  scripts/test_trust_auditor.py scripts/test_memu_trust_tier_ranking.py -q

test-market:
	PYTHONPATH=. python -m pytest scripts/test_alpha_signals.py scripts/test_market_data.py \
	  scripts/test_market_intel.py scripts/test_opportunity_intel.py \
	  scripts/test_paper_trader.py scripts/test_strategy_engine.py -q

test-cognition:
	PYTHONPATH=. python -m pytest scripts/test_cortex.py scripts/test_model_council.py \
	  scripts/test_moral_imagination.py scripts/test_soul_identity.py \
	  scripts/test_wisdom_graph.py scripts/test_wisdom_ingestion.py \
	  scripts/test_cross_session_context.py -q

test-j-features:
	PYTHONPATH=. python -m pytest scripts/test_j1_live_canvas.py scripts/test_j3_pii_redaction.py \
	  scripts/test_j5_diary.py scripts/test_h3_coverage_gate.py -q

test-perception-misc:
	PYTHONPATH=. python -m pytest scripts/test_camera_proactive_gate.py \
	  scripts/test_supervisor_signal_proactive.py scripts/test_service_watchdog.py \
	  scripts/test_web_scout.py scripts/test_tool_gate_taxonomy.py \
	  scripts/test_p1_p4_enhancements.py -q

# Pure, needs nothing running. It was the only one of the 32 that ran in
# no way at all: no target, and pytest collects nothing from it because
# everything is behind `if __name__ == "__main__"`.
test-hmac-advisor:
	PYTHONPATH=. python scripts/test_hmac_migration_advisor.py

test-untargeted: test-trust-ladder test-market test-cognition test-j-features \
	test-perception-misc test-hmac-advisor test-smoke-core
	@echo "All previously untargeted suites ran."

# scripts/test_smoke_core.py needs no stack: it asserts that the probe
# reports failure when nothing answers, which is the property that
# matters and the one its old comment described without asserting.
test-smoke-core:
	PYTHONPATH=. python -m pytest scripts/test_smoke_core.py -q

test-graph-live:
	PYTHONPATH=. python scripts/test_graph_live.py

# Wave 1 status report: revalidates all 96 KAI-DASH findings (not a gate).
dashboard-findings:
	python3 scripts/security/check_dashboard_findings.py

# Repo-wide HTTP/time hygiene scale. See W1_GLOBAL_HYGIENE_SUBPLAN.md
hygiene-survey:
	python3 scripts/security/hygiene_survey.py

# Ratchet: fails if any hygiene count has risen above the recorded baseline.
hygiene-gate:
	python3 scripts/security/hygiene_survey.py --gate

# Lock in an improvement. Refuses to raise the ceiling.
hygiene-baseline:
	python3 scripts/security/hygiene_survey.py --update-baseline

# Requires a running stack. Verifies handlers against live services (G-10/E-01).
verify-live-endpoints:
	PYTHONPATH=. python scripts/verify_live_endpoints.py

# Requires a running stack + KAI_SERVICE_TOKEN. Mutating actuators (E-02).
verify-live-mutating:
	PYTHONPATH=. python scripts/verify_live_mutating.py

# Generate KAI_SERVICE_TOKEN into the gitignored .env (E-03).
setup-service-token:
	./scripts/setup_service_token.sh $(FORCE)

# Check deployment readiness before shipping (E-03).
preflight:
	PYTHONPATH=. python scripts/preflight_deploy.py

# Aggregate: every Unified Hunter work package + adversarial guards.
test-uh: test-contracts test-perception-spine test-world-state \
	test-proposal-workspace test-policy-bridge test-vertical-slice \
	test-actuator-registry test-autonomy test-payload-bounds \
	test-assessment test-invariant-guards test-concurrency-clock \
	test-service-auth test-erasure test-legacy-bridge test-migration \
	test-full-migration test-flags-enabled test-preflight \
	test-architecture-rules test-dashboard-findings test-dashboard-auth \
	test-dashboard-ui-auth test-degraded test-hygiene-gate \
	test-assertion-floors test-gate-registry test-compose-drift \
	test-secret-gates test-compose-gates test-ci-tolerations \
	test-test-wiring test-test-isolation test-live-smoke \
	test-compose-probe test-workflow-filters test-workflow-outputs \
	test-dockerfile-flags test-dockerfile-coverage \
	test-dockerfile-context test-service-tokens test-compose-env \
	test-compose-interpolation test-unreachable-bindings \
	test-implicit-deps test-image-modules test-test-identity \
	test-execution-coverage test-bringup-guards test-ci-scripts \
	test-suite-floor
	@echo "All Unified Hunter suites passed."

# A-02 ratchet: runs `test-uh` and fails if any suite exercises less than
# its recorded floor. Not part of `test-uh` itself — it consumes that
# run's output, so nesting it would recurse.
assertion-floors:
	python3 scripts/security/check_assertion_floors.py --determinism

# Lock in added coverage. Refuses to lower a floor.
assertion-floors-update:
	python3 scripts/security/check_assertion_floors.py --update-floors

# A-04a: the four instrumentation invariants, in reporting mode. Exits 0
# by design — it makes the debt visible without blocking, the same way
# H-5 landed. A-04e flips it to `--gate`.
gate-registry:
	python3 scripts/security/check_gate_registry.py

# A-04e will move this into policy-check. Runnable now to see it bite.
gate-registry-gate:
	python3 scripts/security/check_ci_tolerations.py
	python3 scripts/security/check_test_wiring.py
	python3 scripts/security/check_gate_registry.py --gate

go_no_go:
	python -m py_compile dashboard/app.py tool-gate/app.py memu-core/app.py agentic/app.py executor/app.py heartbeat/app.py supervisor/app.py verifier/app.py fusion-engine/app.py common/llm.py common/errors.py common/feature_flags.py memory-compressor/app.py ledger-worker/app.py metrics-gateway/app.py telegram-bot/app.py
	# --allow-absent: this stage compiles modules, it does not run a
	# dashboard. Declared here rather than assumed inside the script,
	# where it made every invocation everywhere pass on absence.
	python scripts/go_no_go_check.py --allow-absent

hardening_smoke:
	python scripts/hardening_smoke.py

pypi-shadow-check:
	bash scripts/check_pypi_shadow.sh


build-kai-control:
	python -m pip install -r scripts/requirements-kai-control.txt
	pyinstaller --onefile --name kai-control scripts/kai_control.py

kai-control-selftest:
	KAI_CONTROL_TEST_MODE=true python scripts/kai_control_selftest.py


test-conviction:
	python scripts/test_conviction.py


kai-drill:
	sh scripts/kai-drill.sh


kai-drill-test:
	KAI_DRILL_TEST_MODE=true KAI_CONTROL_TEST_MODE=true sh scripts/kai-drill.sh


test-self-emp:
	python scripts/test_self_emp_advisor.py


game-day-scorecard:
	PYTHONPATH=. python scripts/gameday_scorecard.py


hmac-rotation-drill:
	PYTHONPATH=. python scripts/hmac_rotation_drill.py


hmac-auto-rotate:
	PYTHONPATH=. python scripts/auto_rotate_hmac.py


chaos-ci:
	PYTHONPATH=. python scripts/chaos_ci.py


hmac-migration-advice:
	PYTHONPATH=. python scripts/hmac_migration_advisor.py


test-auth-hmac:
	PYTHONPATH=. python scripts/test_auth_hmac_hardening.py


test-phase-b-memu:
	PYTHONPATH=. MEMU_ALLOW_FAKE_EMBEDDINGS=true python scripts/test_phase_b_memu_core.py

test-memu-pg:
	PYTHONPATH=. MEMU_ALLOW_FAKE_EMBEDDINGS=true python scripts/test_memu_pgvector.py

test-memu-turbovec:
	PYTHONPATH=. MEMU_ALLOW_FAKE_EMBEDDINGS=true python scripts/test_memu_turbovec.py

# audio & camera smoke

test-audio:
	PYTHONPATH=. python scripts/test_audio_service.py

test-camera:
	PYTHONPATH=. python -m pytest scripts/test_camera_service.py -v

test-executor:
	KAI_ALLOW_UNAUTHENTICATED=true PYTHONPATH=. python scripts/test_executor_service.py

test-agentic-service:
	PYTHONPATH=. python scripts/test_langgraph_service.py

test-kai-advisor:
	PYTHONPATH=. python kai-advisor/test_kai_advisor.py

test-tts:
	PYTHONPATH=. python scripts/test_tts_service.py

test-avatar:
	PYTHONPATH=. python scripts/test_avatar_service.py

test-tool-gate:
	PYTHONPATH=. python scripts/test_tool_gate_api.py

test-telegram:
	PYTHONPATH=. python scripts/test_telegram_bot.py

test-agentic:
	python3 scripts/agentic_integration_test.py

test-agentic-introspect:
	PYTHONPATH=. python scripts/test_agentic_introspect.py

# previously orphan tests — now wired in
test-heartbeat:
	PYTHONPATH=. python scripts/test_heartbeat.py

test-episode-saver:
	PYTHONPATH=. python scripts/test_episode_saver.py

test-episode-spool:
	PYTHONPATH=. python scripts/test_episode_spool.py

test-tool-gate-security:
	PYTHONPATH=. python scripts/test_tool_gate_security.py

test-error-budget:
	PYTHONPATH=. python scripts/test_error_budget_breaker.py

test-invoice:
	PYTHONPATH=. python -m pytest scripts/test_invoice.py -q

test-dashboard:
	PYTHONPATH=. python scripts/test_dashboard.py

test-thinking-pathways:
	PYTHONPATH=. python -m pytest scripts/test_thinking_pathways.py -v

test-memu-retrieval:
	PYTHONPATH=. python scripts/test_memu_retrieval.py

test-router:
	PYTHONPATH=. python scripts/test_router.py

test-memu-routes:
	PYTHONPATH=. python -m pytest scripts/test_memu_routes.py -v

test-agentic-routes:
	KAI_ALLOW_UNAUTHENTICATED=true PYTHONPATH=. python -m pytest scripts/test_agentic_routes.py -v

test-context-enrichment-ab:
	PYTHONPATH=. python scripts/test_context_enrichment_ab.py

test-ab-log:
	PYTHONPATH=. python scripts/test_ab_log.py

test-screen-capture:
	PYTHONPATH=. python -m pytest scripts/test_screen_capture.py -v

test-shell-sandbox:
	PYTHONPATH=. python -m pytest scripts/test_shell_sandbox.py -v

test-restart-persistence:
	python3 scripts/test_restart_persistence.py

test-hse-rams:
	PYTHONPATH=. python -m pytest scripts/test_hse_rams.py -v

test-upload-fuzz:
	PYTHONPATH=. python -m pytest scripts/security_fuzz_upload.py -v

test-audio-transcribe:
	PYTHONPATH=. python -m pytest scripts/test_audio_transcribe.py -v

test-browser-agent:
	KAI_ALLOW_UNAUTHENTICATED=true PYTHONPATH=. python -m pytest scripts/test_browser_agent.py -v

test-vision-service:
	PYTHONPATH=. python -m pytest scripts/test_vision_service.py -v

test-clipboard-service:
	PYTHONPATH=. python -m pytest scripts/test_clipboard_service.py -v

test-files-service:
	PYTHONPATH=. python -m pytest scripts/test_files_service.py -v

test-notify-service:
	KAI_ALLOW_UNAUTHENTICATED=true PYTHONPATH=. python -m pytest scripts/test_notify_service.py -v

test-document-parser:
	PYTHONPATH=. python -m pytest scripts/test_document_parser.py -v

test-monitor-service:
	KAI_ALLOW_UNAUTHENTICATED=true PYTHONPATH=. python -m pytest scripts/test_monitor_service.py -v

test-broker-bridge:
	PYTHONPATH=. python -m pytest scripts/test_broker_bridge.py -v

test-sysmetrics:
	PYTHONPATH=. python -m pytest scripts/test_sysmetrics.py -v

test-screen-watcher:
	PYTHONPATH=. python -m pytest scripts/test_screen_watcher.py -v

test-email-reader:
	PYTHONPATH=. python -m pytest scripts/test_email_reader.py -v

test-news-feed:
	PYTHONPATH=. python -m pytest scripts/test_news_feed.py -v

test-weather-service:
	PYTHONPATH=. python -m pytest scripts/test_weather_service.py -v

test-docker-watcher:
	PYTHONPATH=. python -m pytest scripts/test_docker_watcher.py -v

test-airquality:
	PYTHONPATH=. python -m pytest scripts/test_airquality.py -v

test-calendar-service:
	PYTHONPATH=. python -m pytest scripts/test_calendar_service.py -v

test-broker-bridge-yfinance:
	python -m pytest scripts/test_broker_bridge_yfinance.py -v

test-git-watcher:
	python -m pytest scripts/test_git_watcher.py -v

test-kai-intelligence:
	python -m pytest scripts/test_kai_intelligence.py -v

test-cognitive-mechanisms:
	python -m pytest scripts/test_cognitive_mechanisms.py -v

test-d89-cognitive-depth:
	python -m pytest scripts/test_d89_cognitive_depth.py -v

test-d90-swarm:
	python -m pytest scripts/test_d90_swarm.py -v

test-d91-vault-sync:
	KAI_ALLOW_UNAUTHENTICATED=true python -m pytest scripts/test_d91_vault_sync.py -v

test-d92-socratic:
	python -m pytest scripts/test_d92_socratic.py -v

test-d93-hypothesis:
	python -m pytest scripts/test_d93_hypothesis.py -v

test-d94-forecaster:
	python -m pytest scripts/test_d94_forecaster.py -v

test-d95-d100-foundations:
	python -m pytest scripts/test_d95_d100_foundations.py -v

test-d101-causal-world-model:
	python -m pytest scripts/test_d101_causal_world_model.py -v

test-d102-global-workspace:
	python -m pytest scripts/test_d102_global_workspace.py -v

test-d109-ohana-core:
	python -m pytest scripts/test_d109_ohana_core.py -v

test-planner:
	PYTHONPATH=. python scripts/test_planner.py

test-adversary:
	PYTHONPATH=. python scripts/test_adversary.py

test-failure-taxonomy:
	PYTHONPATH=. python scripts/test_failure_taxonomy.py

test-selaur:
	PYTHONPATH=. python scripts/test_selaur.py

test-contradiction:
	PYTHONPATH=. python scripts/test_contradiction.py

test-gem:
	PYTHONPATH=. python scripts/test_gem_preferences.py

test-planner-prefs:
	PYTHONPATH=. python scripts/test_planner_preferences.py

test-silence:
	PYTHONPATH=. python scripts/test_silence_signal.py

test-self-deception:
	PYTHONPATH=. python scripts/test_self_deception.py

test-temporal-self:
	PYTHONPATH=. python scripts/test_temporal_self.py

test-predictive:
	PYTHONPATH=. python scripts/test_predictive.py

test-improvement-gate:
	PYTHONPATH=. python scripts/test_improvement_gate.py

test-dream-state:
	PYTHONPATH=. python scripts/test_dream_state.py

test-security-audit:
	PYTHONPATH=. python scripts/test_security_audit.py

test-gaps-sprint:
	PYTHONPATH=. python scripts/test_gaps_sprint.py

test-github-models:
	PYTHONPATH=. python scripts/test_github_models_eval.py

test-tree-search:
	PYTHONPATH=. python scripts/test_tree_search.py

test-priority-queue:
	PYTHONPATH=. python scripts/test_priority_queue.py

test-model-selector:
	PYTHONPATH=. python scripts/test_model_selector.py

test-prod-hardening:
	PYTHONPATH=. python scripts/test_prod_hardening.py

test-p3-organic:
	PYTHONPATH=. python scripts/test_p3_organic_memory.py

test-p4-personality:
	PYTHONPATH=. python -m pytest scripts/test_p4_personality.py -v

test-p16-operational:
	PYTHONPATH=. python -m pytest scripts/test_p16_operational.py -v

test-p17-emotional-intelligence:
	PYTHONPATH=. python -m pytest scripts/test_p17_emotional_intelligence.py -v

test-p18-narrative-identity:
	PYTHONPATH=. python -m pytest scripts/test_p18_narrative_identity.py -v

test-p19-imagination-engine:
	PYTHONPATH=. python -m pytest scripts/test_p19_imagination_engine.py -v

test-p20-conscience-values:
	PYTHONPATH=. python -m pytest scripts/test_p20_conscience_values.py -v

test-p21-proactive-agent:
	PYTHONPATH=. python -m pytest scripts/test_p21_proactive_agent.py -v

test-p22-operator-model:
	PYTHONPATH=. python -m pytest scripts/test_p22_operator_model.py -v

test-h1-hardening:
	PYTHONPATH=. python -m pytest scripts/test_h1_hardening.py -v

test-h2-self-healing:
	PYTHONPATH=. python -m pytest scripts/test_h2_self_healing.py -v

test-mars-consolidation:
	PYTHONPATH=. python -m pytest scripts/test_mars_consolidation.py -v

test-focus-compress:
	PYTHONPATH=. python -m pytest scripts/test_focus_compress.py -v

test-context-budget:
	PYTHONPATH=. python -m pytest scripts/test_context_budget.py -v

test-sage-critique:
	PYTHONPATH=. python -m pytest scripts/test_sage_critique.py -v

test-agent-evolver:
	PYTHONPATH=. python -m pytest scripts/test_agent_evolver.py -v

test-checkpoint:
	PYTHONPATH=. python -m pytest scripts/test_checkpoint.py -v

test-error-codes:
	PYTHONPATH=. python -m pytest scripts/test_error_codes.py -v

test-feature-flags:
	PYTHONPATH=. python -m pytest scripts/test_feature_flags.py -v

test-predictive-failure:
	PYTHONPATH=. python -m pytest scripts/test_predictive_failure.py -v

test-multi-modal:
	PYTHONPATH=. python -m pytest scripts/test_multi_modal.py -v

test-world-anchor:
	PYTHONPATH=. python -m pytest scripts/test_world_anchor.py -v

test-self-healing-phases:
	PYTHONPATH=. python -m pytest scripts/test_self_healing_phases.py -v

test-j-series:
	PYTHONPATH=. python -m pytest scripts/test_j_series.py -v

test-letta:
	PYTHONPATH=. python -m pytest scripts/test_letta_agent.py -v

test-financial:
	PYTHONPATH=. python -m pytest scripts/test_financial_awareness.py -v

test-wake:
	PYTHONPATH=. python -m pytest scripts/test_wake_intent.py -v

test-behavioral:
	PYTHONPATH=. python -m pytest scripts/test_behavioral.py -v

test-docker-e2e:
	PYTHONPATH=. python -m pytest scripts/test_docker_e2e.py -v

test-chassis:
	PYTHONPATH=. python -m pytest scripts/test_chassis.py -v

test-chassis-runtime:
	PYTHONPATH=. python -m pytest scripts/test_chassis_runtime.py -v

dep-audit:
	pip-audit --strict --desc 2>/dev/null || echo "WARNING: pip-audit found issues (non-fatal — mirrors CI behaviour)"

coverage:
	# Phase 1: run the two isolated test files that inject sys.modules stubs.
	# They must run alone — their module-level stubs (tree_search, priority_queue,
	# model_selector etc.) would contaminate other tests if collected together.
	PYTHONPATH=. MEMU_ALLOW_FAKE_EMBEDDINGS=true python -m pytest \
	  scripts/test_agentic_routes.py scripts/test_memu_routes.py \
	  --cov=common --cov=agentic --cov=memu-core --cov=letta-agent --cov=financial-awareness \
	  --cov-report= -q || true
	# Phase 2: all other scripts, appending to the .coverage file from phase 1.
	PYTHONPATH=. MEMU_ALLOW_FAKE_EMBEDDINGS=true python -m pytest scripts/ \
	  --ignore=scripts/test_agentic_routes.py \
	  --ignore=scripts/test_memu_routes.py \
	  --cov=common --cov=agentic --cov=memu-core --cov=letta-agent --cov=financial-awareness \
	  --cov-append \
	  --cov-report=term-missing --cov-report=html:output/coverage_html \
	  --cov-fail-under=60 -q

coverage-floors: coverage
	@echo "--- Per-module coverage floors ---"
	PYTHONPATH=. python -m coverage report --include="agentic/*" --fail-under=45 \
	  || (echo "FAIL: agentic coverage below 45%" && exit 1)
	PYTHONPATH=. python -m coverage report --include="memu-core/*" --fail-under=60 \
	  || (echo "FAIL: memu-core coverage below 60%" && exit 1)
	@echo "All per-module coverage floors met"

# v7 feature tests
test-v7-verifier:
	PYTHONPATH=. python scripts/test_v7_verifier.py

test-v7-quarantine:
	PYTHONPATH=. python scripts/test_v7_quarantine.py

test-v7-policy:
	PYTHONPATH=. python scripts/test_v7_policy_and_ratelimit.py

test-v7-idempotency:
	PYTHONPATH=. python scripts/test_v7_idempotency.py

test-integration-chain:
	PYTHONPATH=. python scripts/test_integration_chain.py

test-v7: test-v7-verifier test-v7-quarantine test-v7-policy test-v7-idempotency test-integration-chain

# wrapper to run all core unit/smoke tests
test-core: test-phase-b-memu test-memu-pg test-memu-turbovec test-letta test-financial test-dashboard-ui test-dashboard test-thinking-pathways test-tool-gate test-tool-gate-security test-telegram test-conviction test-audio test-camera test-executor test-agentic-service test-agentic-introspect test-kai-advisor test-tts test-avatar test-heartbeat test-episode-saver test-episode-spool test-error-budget test-invoice test-memu-retrieval test-router test-planner test-adversary test-failure-taxonomy test-selaur test-self-emp test-auth-hmac test-agentic test-v7 test-contradiction test-gem test-planner-prefs test-silence test-self-deception test-temporal-self test-predictive test-improvement-gate test-dream-state test-security-audit test-gaps-sprint test-tree-search test-priority-queue test-model-selector test-prod-hardening test-p3-organic test-p4-personality test-p16-operational test-p17-emotional-intelligence test-p18-narrative-identity test-p19-imagination-engine test-p20-conscience-values test-p21-proactive-agent test-p22-operator-model test-h1-hardening test-h2-self-healing test-mars-consolidation test-sage-critique test-agent-evolver test-checkpoint test-error-codes test-feature-flags test-predictive-failure test-multi-modal test-world-anchor test-self-healing-phases test-j-series test-wake test-behavioral test-docker-e2e test-chassis test-chassis-runtime test-git-watcher test-broker-bridge-yfinance test-kai-intelligence test-cognitive-mechanisms test-d89-cognitive-depth test-d90-swarm test-d91-vault-sync test-d92-socratic test-d93-hypothesis test-d94-forecaster test-d95-d100-foundations test-d101-causal-world-model test-d102-global-workspace test-d109-ohana-core

test-dashboard-ui:
	PYTHONPATH=. python scripts/test_dashboard_ui.py

# Live end-to-end smoke against a running stack. Needs a stack: it is
# run by core-tests.yml after the bring-up, not by `test-uh`.
#
# The old target ran `scripts/test_core_integration.py`, which ended in a
# bare `return 0` after catching every exception it could raise — with
# the whole stack down it printed eleven failures and exited 0. Its
# replacement fails. `test-live-smoke` (in `test-uh`) tests *this*
# script's logic with synthetic input and needs no stack at all.
test-integration:
	python3 scripts/ci/live_smoke.py --compose-file docker-compose.minimal.yml

test-live-smoke:
	python3 scripts/test_live_smoke.py

test-compose-probe:
	python3 scripts/test_compose_probe.py

test-workflow-filters:
	python3 scripts/test_workflow_filters.py

test-workflow-outputs:
	python3 scripts/test_workflow_outputs.py

test-dockerfile-flags:
	python3 scripts/test_dockerfile_flags.py

test-dockerfile-coverage:
	python3 scripts/test_dockerfile_coverage.py

test-dockerfile-context:
	python3 scripts/test_dockerfile_context.py

test-service-tokens:
	python3 scripts/test_service_tokens.py

test-compose-env:
	python3 scripts/test_compose_env.py

test-compose-interpolation:
	python3 scripts/test_compose_interpolation.py

test-unreachable-bindings:
	python3 scripts/test_unreachable_bindings.py

test-implicit-deps:
	python3 scripts/test_implicit_deps.py

test-image-modules:
	python3 scripts/test_image_modules.py

test-test-identity:
	python3 scripts/test_test_identity.py

test-execution-coverage:
	python3 scripts/test_execution_coverage.py

test-bringup-guards:
	python3 scripts/test_bringup_guards.py

test-ci-scripts:
	python3 scripts/test_ci_scripts.py

execution-coverage:
	python3 scripts/security/report_execution_coverage.py

# bring up full-stack composition
full-up:
	docker compose -f docker-compose.full.yml up -d --build

full-down:
	docker compose -f docker-compose.full.yml down


health-sweep:
	bash scripts/health_sweep.sh


contract-smoke:
	bash scripts/contract_smoke.sh




merge-gate:
	$(MAKE) policy-check
	$(MAKE) go_no_go
	$(MAKE) pypi-shadow-check
	$(MAKE) check-docs
	python3 scripts/quality_gate.py
	$(MAKE) dep-audit
	$(MAKE) test-core
	$(MAKE) test-integration
	$(MAKE) coverage
	$(MAKE) assertion-floors


phase1-closure:
	PYTHONPATH=. python scripts/phase1_closure_check.py

# bring up the minimal sovereign AI core stack for development
core-up:
	docker compose -f docker-compose.minimal.yml up -d --build

# tear down the minimal core stack
core-down:
	docker compose -f docker-compose.minimal.yml down

# run quick health checks against the core services
core-smoke:
	python3 scripts/smoke_core.py

# create database schema for memu-core when using postgres
init-memu-db:
	PG_URI=$${PG_URI:-postgresql://keeper:$${DB_PASSWORD:-localdev}@postgres:5432/sovereign} \
	python3 scripts/init_memu_db.py


paper-backup:
	bash scripts/monthly_paper_backup.sh

weekly-key-rotate:
	bash scripts/weekly_key_rotation.sh


weekly-ed25519-rotate:
	bash scripts/weekly_ed25519_rotation.sh

capture-baseline:
	PYTHONPATH=. python scripts/capture_baseline_responses.py

setup:
	bash scripts/setup.sh

sync-docs:
	python3 scripts/sync_docs.py

check-docs:
	python3 scripts/sync_docs.py --check

auto-changelog:
	python3 scripts/auto_changelog.py

auto-session-log:
	python3 scripts/auto_session_log.py

