# Offline/CI default: use hash-based fake embeddings so tests that don't care
# about embedding quality can run without sentence-transformers installed.
# Override with MEMU_ALLOW_FAKE_EMBEDDINGS=false for real-embedding tests.
export MEMU_ALLOW_FAKE_EMBEDDINGS ?= true

# Self-audit and feedback
self-audit:
	python3 scripts/self_audit.py
.PHONY: go_no_go hardening_smoke pypi-shadow-check test-letta test-financial test-agentic-service test-agentic-introspect build-kai-control kai-control-selftest test-conviction kai-drill kai-drill-test test-self-emp game-day-scorecard hmac-rotation-drill hmac-auto-rotate hmac-migration-advice test-auth-hmac test-phase-b-memu chaos-ci health-sweep contract-smoke merge-gate phase1-closure paper-backup weekly-key-rotate weekly-ed25519-rotate core-up core-down core-smoke test-v7-verifier test-v7-quarantine test-v7-policy test-v7-idempotency test-integration-chain test-v7 test-heartbeat test-episode-saver test-episode-spool test-tool-gate-security test-error-budget test-invoice test-dashboard test-memu-retrieval test-memu-routes test-agentic-routes test-context-enrichment-ab test-ab-log test-screen-capture test-agentic test-router test-planner test-adversary test-failure-taxonomy test-selaur test-contradiction test-gem test-planner-prefs test-silence test-self-deception test-temporal-self test-predictive test-improvement-gate test-thinking-pathways test-dream-state test-security-audit test-gaps-sprint test-tree-search test-priority-queue test-model-selector test-prod-hardening test-p3-organic test-p4-personality test-p16-operational test-p17-emotional-intelligence test-p18-narrative-identity test-p19-imagination-engine test-p20-conscience-values test-p21-proactive-agent test-p22-operator-model test-error-codes test-feature-flags dep-audit coverage sync-docs check-docs auto-changelog auto-session-log test-focus-compress test-context-budget test-predictive-failure test-multi-modal test-world-anchor test-self-healing-phases test-j-series test-behavioral test-docker-e2e test-chassis test-wake

go_no_go:
	python -m py_compile dashboard/app.py tool-gate/app.py memu-core/app.py agentic/app.py executor/app.py heartbeat/app.py supervisor/app.py verifier/app.py fusion-engine/app.py common/llm.py common/errors.py common/feature_flags.py memory-compressor/app.py ledger-worker/app.py metrics-gateway/app.py telegram-bot/app.py
	python scripts/go_no_go_check.py

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
	PYTHONPATH=. python scripts/test_executor_service.py

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
	PYTHONPATH=. python scripts/test_invoice.py

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
	PYTHONPATH=. python -m pytest scripts/test_agentic_routes.py -v

test-context-enrichment-ab:
	PYTHONPATH=. python scripts/test_context_enrichment_ab.py

test-ab-log:
	PYTHONPATH=. python scripts/test_ab_log.py

test-screen-capture:
	PYTHONPATH=. python -m pytest scripts/test_screen_capture.py -v

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
test-core: test-phase-b-memu test-memu-pg test-memu-turbovec test-letta test-financial test-dashboard-ui test-dashboard test-thinking-pathways test-tool-gate test-tool-gate-security test-telegram test-conviction test-audio test-camera test-executor test-agentic-service test-agentic-introspect test-kai-advisor test-tts test-avatar test-heartbeat test-episode-saver test-episode-spool test-error-budget test-invoice test-memu-retrieval test-router test-planner test-adversary test-failure-taxonomy test-selaur test-self-emp test-auth-hmac test-agentic test-v7 test-contradiction test-gem test-planner-prefs test-silence test-self-deception test-temporal-self test-predictive test-improvement-gate test-dream-state test-security-audit test-gaps-sprint test-tree-search test-priority-queue test-model-selector test-prod-hardening test-p3-organic test-p4-personality test-p16-operational test-p17-emotional-intelligence test-p18-narrative-identity test-p19-imagination-engine test-p20-conscience-values test-p21-proactive-agent test-p22-operator-model test-h1-hardening test-h2-self-healing test-mars-consolidation test-sage-critique test-agent-evolver test-checkpoint test-error-codes test-feature-flags test-predictive-failure test-multi-modal test-world-anchor test-self-healing-phases test-j-series test-wake test-behavioral test-docker-e2e test-chassis test-chassis-runtime

test-dashboard-ui:
	PYTHONPATH=. python scripts/test_dashboard_ui.py

test-integration:
	python3 scripts/test_core_integration.py

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
	$(MAKE) go_no_go
	$(MAKE) pypi-shadow-check
	$(MAKE) check-docs
	python3 scripts/quality_gate.py
	$(MAKE) dep-audit
	$(MAKE) test-core
	$(MAKE) test-integration
	$(MAKE) coverage


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

