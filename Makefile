.PHONY: go_no_go

go_no_go:
	python -m py_compile dashboard/app.py tool-gate/app.py memu-core/app.py langgraph/app.py executor/app.py
	python scripts/go_no_go_check.py
