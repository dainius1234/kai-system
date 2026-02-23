from __future__ import annotations

from common.runtime import ErrorBudgetCircuitBreaker

b = ErrorBudgetCircuitBreaker(warn_ratio=0.05, open_ratio=0.10, window_seconds=300, recovery_seconds=1)

for _ in range(20):
    b.record(200)
assert b.snapshot()["state"] == "closed"

for _ in range(3):
    b.record(500)
assert b.snapshot()["state"] in {"half_open", "open"}

for _ in range(3):
    b.record(500)
assert b.snapshot()["state"] == "open"

print("error-budget breaker tests passed")
