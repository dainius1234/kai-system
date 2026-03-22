# Kai-System Error & Warning Log

> Every issue tracked until eliminated. No matter how small.

**Generated:** 2026-03-22 | **Test suite:** 1,488 passed, 5 skipped, 0 failed

---

## Active Issues

### Deprecation Warnings (110 total — 3 source patterns)

| # | File | Line(s) | Warning | Fix |
|---|------|---------|---------|-----|
| W1 | memu-core/app.py | 1814, 1889, 2670, 2726, 2813, 2955 | `datetime.utcnow()` deprecated | Replace with `datetime.now(datetime.UTC)` |
| W2 | scripts/test_p3_organic_memory.py | 76, 197, 216, 229, 405, 418, 434, 482, 510 | `datetime.utcnow()` deprecated | Replace with `datetime.now(datetime.UTC)` |
| W3 | scripts/test_silence_signal.py | 28, 57, 62, 100 | `datetime.utcnow()` deprecated | Replace with `datetime.now(datetime.UTC)` |
| W4 | scripts/test_tempo.py | 18 (x48 calls) | `datetime.utcnow()` deprecated | Replace with `datetime.now(datetime.UTC)` |
| W5 | scripts/test_gaps_sprint.py | 127 | `no current event loop` | Use `asyncio.new_event_loop()` |

### Flake8 Lint (435 total — by category)

| # | Category | Count | Severity | Fix Priority |
|---|----------|-------|----------|-------------|
| L1 | E501 line too long (>127) | 101 | Style | Low — cosmetic |
| L2 | F401 unused imports | 98 | Medium | Batch cleanup |
| L3 | C901 too complex (>10) | 45 | Medium | Refactor hot paths |
| L4 | E402 import not at top | 47 | Low | Intentional (lazy imports) |
| L5 | E302/E301 blank line issues | 31 | Style | Low — cosmetic |
| L6 | F541 f-string no placeholders | 16 | Low | Convert to plain strings |
| L7 | F811 redefinition of unused var | 16 | Medium | Dead code in memu-core |
| L8 | F841 assigned but unused | 10 | Medium | Remove dead assignments |
| L9 | E231/E225 whitespace | 21 | Style | Low — cosmetic |
| L10 | E702 multiple statements | 2 | Style | Low |
| L11 | E401 multiple imports | 3 | Style | Low |

### Skipped Tests (5)

| # | Test | Reason | Action |
|---|------|--------|--------|
| S1 | agentic_integration_test::test_langgraph | langgraph.graph not installed | Install or keep skip |
| S2 | agentic_integration_test::test_autogen | autogen not installed | Install or keep skip |
| S3 | agentic_integration_test::test_crewai | crewai not installed | Install or keep skip |
| S4 | agentic_integration_test::test_openagents | openagents not installed | Install or keep skip |
| S5 | test_cross_session_context | No vector store configured | Env-specific — OK |

---

## Resolved (This Sprint)

| Date | Issue | Resolution |
|------|-------|------------|
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
