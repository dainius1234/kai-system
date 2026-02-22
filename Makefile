.PHONY: go_no_go hardening_smoke build-kai-control kai-control-selftest test-conviction kai-drill kai-drill-test test-self-emp game-day-scorecard hmac-rotation-drill

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
