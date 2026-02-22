# Phase-1 Patch Set (Concrete, Repo-Aligned)

## Goal
Harden the existing core implementation with explicit API schemas, policy checks, and operator-visible dashboard fields while preserving current service layout.

## Patch Set A — Tool Gate Policy & Contracts

### Files
- `tool-gate/app.py`

### Changes
1. **Strict policy checks**
   - enforce `actor_did` non-empty and allowlist `tool` names (`shell`, `qgis`, `n8n`, `noop`).
   - add explicit reason codes: `PUB_MODE`, `LOW_CONFIDENCE`, `ALLOWLIST_BLOCK`, `APPROVED`.
2. **Request schema extension**
   - include `request_source`, `trace_id`, `device`.
3. **Decision schema extension**
   - include `reason_code`, `evaluated_at`, `policy_version`.
4. **Ledger consistency endpoint**
   - `GET /ledger/verify` to verify hash-chain integrity.

### Acceptance criteria
- invalid tool returns `400` with reason code.
- `/ledger/verify` returns `{status:"ok", valid:true}` on clean chain.

---

## Patch Set B — memu-core State & Routing

### Files
- `memu-core/app.py`

### Changes
1. **Routing contract**
   - ensure `/route` always returns:
     - `specialist`
     - `context_payload.query`
     - `context_payload.metadata.session_id`
     - `context_payload.device`
2. **State mutation rules**
   - reject duplicate keys in `state_delta` (already present) and add max key/value size limits.
3. **Memory retention controls**
   - add optional env var `MAX_MEMORY_RECORDS` and cap in-memory list.
4. **Diagnostic endpoint**
   - `GET /memory/diagnostics` with counts by `event_type`.

### Acceptance criteria
- oversized state key/value returns `400`.
- capped memory retains most recent N entries.

---

## Patch Set C — Executor Integration Contract

### Files
- `executor/app.py`

### Changes
1. **Input contract**
   - require `device` in execution payload (`cpu`/`cuda`).
2. **Structured execution output**
   - add `exit_code`, `stderr`, `policy_context` fields.
3. **Timeout and payload controls**
   - explicit timeout + max output truncation with flags (`truncated: true/false`).

### Acceptance criteria
- payload missing `device` rejected.
- timeout returns structured 408 response body.

---

## Patch Set D — Dashboard Operational Fields

### Files
- `dashboard/app.py`

### Changes
1. Display fields at `/`:
   - `core_ready` (bool)
   - `alive_nodes` (list)
   - `ledger_size`
   - `memory_count`
   - `policy_mode`
   - `device_summary` (always `running (CPU)` now unless enabled)
2. Add `GET /readiness` endpoint requiring:
   - tool-gate up
   - memu-core up
   - executor up
   - non-negative counts

### Acceptance criteria
- `/readiness` fails if any core dependency down.
- `/` response has all required fields.

---

## Patch Set E — Compose Runtime Guarantees

### Files
- `docker-compose.sovereign.yml`

### Changes
1. Keep startup ordering:
   - tool-gate → postgres
   - memu-core → postgres + tool-gate
   - executor → tool-gate + memu-core
2. Ensure least privilege defaults are inherited by core services.
3. Keep GPU-heavy services commented with TODO markers.
4. Add healthchecks for core services (`/health`) and postgres (`pg_isready`).

### Acceptance criteria
- `docker compose config` succeeds.
- all core containers become healthy.

---

## Patch Set F — Test Artifacts

### Files
- `scripts/health_sweep.sh` (new)
- `scripts/contract_smoke.sh` (new)

### Changes
1. `health_sweep.sh`
   - checks each core endpoint, writes `/var/log/sovereign/health_sweep.log`.
2. `contract_smoke.sh`
   - exercises tool-gate, memu-core, dashboard fields and validates required keys.

### Acceptance criteria
- both scripts return `0` on green path.
- any failed endpoint returns non-zero and prints failing URL.
