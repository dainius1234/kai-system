.PHONY: go_no_go hardening_smoke script_consistency

go_no_go:
	python -m py_compile dashboard/app.py tool-gate/app.py memu-core/app.py langgraph/app.py executor/app.py
	python scripts/go_no_go_check.py

hardening_smoke:
	python scripts/hardening_smoke.py

script_consistency:
	python scripts/scan_script_consistency.py
