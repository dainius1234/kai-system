.PHONY: go_no_go hardening_smoke build-kai-control kai-control-selftest test-conviction

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
