# Self-audit and feedback
self-audit:
	python3 scripts/self_audit.py
.PHONY: go_no_go hardening_smoke build-kai-control kai-control-selftest test-conviction kai-drill kai-drill-test test-self-emp game-day-scorecard hmac-rotation-drill hmac-auto-rotate hmac-migration-advice test-auth-hmac test-phase-b-memu chaos-ci health-sweep contract-smoke merge-gate phase1-closure paper-backup weekly-key-rotate weekly-ed25519-rotate core-up core-down core-smoke test-v7-verifier test-v7-quarantine test-v7-policy test-v7-idempotency test-integration-chain test-v7 test-heartbeat test-episode-saver test-episode-spool test-tool-gate-security test-error-budget test-invoice test-dashboard test-memu-retrieval test-agentic test-router test-planner test-adversary test-failure-taxonomy test-selaur test-contradiction test-gem test-planner-prefs test-silence test-self-deception test-temporal-self test-predictive test-tempo test-improvement-gate test-thinking-pathways test-dream-state test-security-audit test-gaps-sprint test-tree-search test-priority-queue test-model-selector test-prod-hardening test-hmac-rotation-drill test-p3-organic test-p4-personality test-p16-operational test-p17-emotional-intelligence

go_no_go:
	python -m py_compile dashboard/app.py tool-gate/app.py memu-core/app.py langgraph/app.py executor/app.py heartbeat/app.py supervisor/app.py verifier/app.py fusion-engine/app.py common/llm.py memory-compressor/app.py ledger-worker/app.py metrics-gateway/app.py telegram-bot/app.py
	python scripts/go_no_go_check.py

hardening_smoke:
	python scripts/hardening_smoke.py


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
	PYTHONPATH=. python scripts/test_phase_b_memu_core.py

test-memu-pg:
	PYTHONPATH=. python scripts/test_memu_pgvector.py

# audio & camera smoke

test-audio:
	PYTHONPATH=. python scripts/test_audio_service.py

test-camera:
	PYTHONPATH=. python scripts/test_camera_service.py

test-executor:
	PYTHONPATH=. python scripts/test_executor_service.py

test-langgraph:
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

test-tempo:
	PYTHONPATH=. python scripts/test_tempo.py

test-improvement-gate:
	PYTHONPATH=. python scripts/test_improvement_gate.py

test-dream-state:
	PYTHONPATH=. python scripts/test_dream_state.py

test-security-audit:
	PYTHONPATH=. python scripts/test_security_audit.py

test-gaps-sprint:
	PYTHONPATH=. python scripts/test_gaps_sprint.py

test-tree-search:
	PYTHONPATH=. python scripts/test_tree_search.py

test-priority-queue:
	PYTHONPATH=. python scripts/test_priority_queue.py

test-model-selector:
	PYTHONPATH=. python scripts/test_model_selector.py

test-prod-hardening:
	PYTHONPATH=. python scripts/test_prod_hardening.py

test-hmac-rotation-drill:
	PYTHONPATH=. python scripts/hmac_rotation_drill.py

test-p3-organic:
	PYTHONPATH=. python scripts/test_p3_organic_memory.py

test-p4-personality:
	PYTHONPATH=. python -m pytest scripts/test_p4_personality.py -v

test-p16-operational:
	PYTHONPATH=. python -m pytest scripts/test_p16_operational.py -v

test-p17-emotional-intelligence:
	PYTHONPATH=. python -m pytest scripts/test_p17_emotional_intelligence.py -v

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
test-core: test-phase-b-memu test-memu-pg test-dashboard-ui test-dashboard test-thinking-pathways test-tool-gate test-tool-gate-security test-telegram test-conviction test-audio test-camera test-executor test-langgraph test-kai-advisor test-tts test-avatar test-heartbeat test-episode-saver test-episode-spool test-error-budget test-invoice test-memu-retrieval test-router test-planner test-adversary test-failure-taxonomy test-selaur test-self-emp test-auth-hmac test-agentic test-v7 test-contradiction test-gem test-planner-prefs test-silence test-self-deception test-temporal-self test-predictive test-tempo test-improvement-gate test-dream-state test-security-audit test-gaps-sprint test-tree-search test-priority-queue test-model-selector test-prod-hardening test-hmac-rotation-drill test-p3-organic test-p4-personality test-p16-operational test-p17-emotional-intelligence

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
	python3 scripts/quality_gate.py
	$(MAKE) test-conviction
	$(MAKE) test-tool-gate
	$(MAKE) test-self-emp
	$(MAKE) kai-control-selftest
	$(MAKE) hardening_smoke
	$(MAKE) kai-drill-test
	$(MAKE) test-auth-hmac
	$(MAKE) test-phase-b-memu
	$(MAKE) hmac-migration-advice
	$(MAKE) health-sweep
	$(MAKE) contract-smoke
	$(MAKE) paper-backup
	$(MAKE) weekly-key-rotate
	$(MAKE) weekly-ed25519-rotate


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
	PG_URI=$${PG_URI:-postgresql://keeper:localdev@postgres:5432/sovereign} \
	python3 scripts/init_memu_db.py


paper-backup:
	bash scripts/monthly_paper_backup.sh

weekly-key-rotate:
	bash scripts/weekly_key_rotation.sh


weekly-ed25519-rotate:
	bash scripts/weekly_ed25519_rotation.sh

setup:
	bash scripts/setup.sh
