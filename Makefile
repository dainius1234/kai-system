.PHONY: go_no_go hardening_smoke build-kai-control kai-control-selftest test-conviction kai-drill kai-drill-test test-self-emp game-day-scorecard hmac-rotation-drill hmac-auto-rotate hmac-migration-advice test-auth-hmac test-phase-b-memu chaos-ci health-sweep contract-smoke merge-gate phase1-closure paper-backup weekly-key-rotate

go_no_go:
	python -m py_compile dashboard/app.py tool-gate/app.py memu-core/app.py langgraph/app.py executor/app.py
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


health-sweep:
	bash scripts/health_sweep.sh


contract-smoke:
	bash scripts/contract_smoke.sh


merge-gate:
	$(MAKE) go_no_go
	$(MAKE) test-conviction
	$(MAKE) test-self-emp
	$(MAKE) kai-control-selftest
	$(MAKE) hardening_smoke
	$(MAKE) kai-drill-test
	$(MAKE) test-auth-hmac
	$(MAKE) test-phase-b-memu
	$(MAKE) hmac-migration-advice


phase1-closure:
	PYTHONPATH=. python scripts/phase1_closure_check.py


paper-backup:
	bash scripts/monthly_paper_backup.sh

weekly-key-rotate:
	bash scripts/weekly_key_rotation.sh
