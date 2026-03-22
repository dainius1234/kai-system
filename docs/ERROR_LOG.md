# Kai-System Error & Warning Log

> Every issue tracked until eliminated. No matter how small.

**Updated:** 2026-03-23 | **Test suite:** 1,513 passed, 5 skipped, 0 failed

---

## Active Issues

### Deprecation Warnings — NONE

All `datetime.utcnow()` and `get_event_loop` deprecations resolved.

### Flake8 Lint — Medium/High Severity CLEAR

| # | Category | Count | Severity | Notes |
|---|----------|-------|----------|-------|
| L1 | E501 line too long (>127) | ~101 | Style | Low — cosmetic, non-blocking |
| L3 | C901 too complex (>10) | ~45 | Medium | Refactor candidates — tracked |
| L4 | E402 import not at top | ~47 | Low | Intentional lazy/conditional imports |
| L5 | E302/E301 blank line issues | ~31 | Style | Low — cosmetic |
| L6 | F541 f-string no placeholders | ~16 | Low | Convert to plain strings |
| L9 | E231/E225 whitespace | ~21 | Style | Low — cosmetic |
| L10 | E702 multiple statements | ~2 | Style | Low |
| L11 | E401 multiple imports | ~3 | Style | Low |

**Critical (E9/F63/F7/F82): 0** | **F401: 0** | **F811: 0** | **F841: 0**

### Skipped Tests (5)

| # | Test | Reason | Action |
|---|------|--------|--------|
| S1 | agentic_integration_test::test_langgraph | langgraph.graph not installed | Env-specific — OK |
| S2 | agentic_integration_test::test_autogen | autogen not installed | Env-specific — OK |
| S3 | agentic_integration_test::test_crewai | crewai not installed | Env-specific — OK |
| S4 | agentic_integration_test::test_openagents | openagents not installed | Env-specific — OK |
| S5 | test_cross_session_context | No vector store configured | Env-specific — OK |

---

## Resolved

| Date | Issue | Resolution |
|------|-------|------------|
| 2026-03-23 | W1-W4: 63× `datetime.utcnow()` deprecation warnings | Replaced with `datetime.now(datetime.UTC)` |
| 2026-03-23 | W5: `no current event loop` in test_gaps_sprint | Fixed to `asyncio.new_event_loop()` |
| 2026-03-23 | L2: F401 unused imports (98 instances) | autoflake bulk removal + manual cleanup |
| 2026-03-23 | L7: F811 redefinition of unused var (16 instances) | autoflake + `# noqa: F811` for intentional try/except re-imports |
| 2026-03-23 | L8: F841 assigned but unused (10 instances) | autoflake + manual removal of dead assignments |
| 2026-03-22 | kai_supervisor.py duplicate function defs + missing main() | Removed dupes, added main() |
| 2026-03-22 | langgraph/app.py missing `import asyncio` (6 F821) | Added import |
| 2026-03-22 | memu-core/app.py undefined `proactive_full_scan` | Fixed to `full_proactive_scan` |
| 2026-03-22 | agentic_integration_test 4 hard fails on missing deps | Added pytest.skip() |
| 2026-03-22 | test_agent_evolver cross-test state contamination | Isolated temp paths |
| 2026-03-22 | test_cross_session_context fails without vector store | Added skip |
| 2026-03-22 | hse_rams.py empty stub passes silently | NotImplementedError |
| 2026-03-22 | common/runtime.py PII regex order (phone ate credit cards) | Reordered patterns |
| 2026-03-22 | kai_supervisor.py E999 IndentationError | Fixed duplicate defs |

---

## Tracking Rules

1. Every warning, error, or lint issue gets a row
2. Fixed items move to "Resolved" with date
3. Log reviewed before every commit
4. Goal: **zero warnings, zero lint errors, zero skips**
