# Kai System — Known Issues & Gotchas

> Things that trip you up. Read this before debugging something weird.

## Environment Quirks

### `test-agentic` fails in Codespace
The `test-agentic` target requires `langgraph`, `autogen`, and `crewai` pip
packages which are only installed inside Docker.  
**Workaround:** Use `make -k test-core` to skip failures and run all other targets.

### Ollama not available in Codespace
Ollama requires GPU or significant CPU resources. In Codespace, the LLM calls
stub out. Real LLM testing requires `docker compose -f docker-compose.full.yml up`.  
**Impact:** `/chat` and `/run` endpoints return stub responses in Codespace tests.

### pgvector tests need PG_URI
`test-memu-pg` requires a running PostgreSQL instance with pgvector extension.  
**Setup:** `export PG_URI=postgresql://keeper:localdev@localhost:5432/sovereign`  
**Skip:** It's fine to skip this target in quick local dev; CI covers it.

### Pre-commit hooks need install
The `.pre-commit-config.yaml` exists but hooks aren't active until you run:
```bash
pip install pre-commit && pre-commit install
```
This only needs to be done once per clone.

## Common Mistakes

### Editing memu-core/app.py
This file is ~6,100 lines. Before editing, always check which function you're
modifying — there are similarly named functions for different subsystems
(e.g., `memorize` vs `auto_memorize`, `get_memories` vs `retrieve`).  
**Rule:** Always run `make go_no_go` after editing.

### Adding new common/ modules
When you add a new file to `common/`, update these places:
1. `common/__init__.py` — may need import
2. `Makefile` go_no_go target — add to py_compile list
3. `.pre-commit-config.yaml` — mypy covers `^common/` automatically
4. `.coveragerc` — already includes `common/` in source

### Adding new test files
When you add `scripts/test_xyz.py`:
1. Add `test-xyz:` target to `Makefile`
2. Add `test-xyz` to the `test-core:` dependency list
3. Add `test-xyz` to `.PHONY:` line
4. Update `.github/copilot-instructions.md` test list
5. Run `make sync-docs` to update README/backlog counts

### HMAC key rotation
HMAC keys live in environment variables. When rotating:
1. Set `TOOL_GATE_DUAL_SIGN=true` (overlap period)
2. Deploy new key alongside old key
3. After all services pick up new key, set `INTERSERVICE_HMAC_STRICT_KEY_ID=true`
4. Full runbook: `docs/hmac_rotation_runbook.md`

### Memory record schema changes
When adding new fields to MemoryRecord, update ALL of these:
1. MemoryRecord model class
2. PG schema CREATE TABLE
3. `_init_schema` migration list
4. `_SELECT_COLS`
5. `_row_to_record`
6. INSERT VALUES
7. `update_record` allowed set

## CI Gotchas

### Flake8 in CI vs local
CI runs two flake8 passes:
- Strict: `E9,F63,F7,F82` (syntax errors, undefined names) — **fails build**
- Soft: full lint with `--exit-zero` — **warnings only**

Local pre-commit hook only runs the strict pass.

### Trivy scanning
Container scanning in `core-tests.yml` runs with `exit-code: 0` (non-blocking).
It reports CRITICAL+HIGH vulns but won't fail the build yet. This is intentional
while we baseline existing images.

### pip-audit
Both CI workflows run `pip-audit --strict --desc`. If a dependency has a known
CVE, the build fails. Fix by pinning a patched version in `requirements.txt`.

## Performance Notes

### 10-way parallel context fetch
The `/chat` endpoint fires 10 parallel requests to memu-core. If memu-core is
slow (cold start, large DB), this can timeout. Circuit breaker will trip after
3 failures.  
**Debug:** Check `memu-core /health` for degraded status.

### Dream cycle duration
`/dream` runs 6 phases sequentially. On a large memory store, MARS consolidation
(phase 3) can take 30+ seconds.  
**Impact:** Checkpoint is taken before AND after dream, so interrupted dreams
can be recovered.

## File Size Watch List

| File | Lines | Notes |
|---|---|---|
| memu-core/app.py | ~6,100 | Memory engine — largest single file |
| langgraph/kai_config.py | ~1,500 | Brain config, checkpoint engine, agent-evolver |
| langgraph/app.py | ~1,500 | All LangGraph endpoints |
| dashboard/app.py | ~1,200 | Dashboard proxy + static serving |
| supervisor/app.py | ~800 | Watchdog + recovery |

These files are the most likely to cause merge conflicts or editing mistakes.
