# Contingency Library (Method Statements + Check Sheets)

This library defines **what to do in the event of** critical incidents.
It is designed for offline sovereign operations with keeper visibility.

## Principles
- Local-first recovery steps.
- Deterministic checklist execution.
- Always log actions.
- Notify keeper on failures.

## Event: `intrusion_detected`
**Method statement**
Contain suspicious execution and preserve state for keeper review.

**Check sheet**
1. Notify keeper.
2. Restart executor process.
3. Trigger memory compression checkpoint.
4. Write audit trail entry.

## Event: `executor_stale`
**Method statement**
Recover executor liveness while preserving logs and current memory state.

**Check sheet**
1. Notify keeper.
2. Restart executor.
3. Run go/no-go check.
4. Record result in audit.

## Event: `low_conviction_trend`
**Method statement**
Stabilize answer quality and reduce stale context noise.

**Check sheet**
1. Notify keeper.
2. Trigger memory compression.
3. Record intervention.

## Runtime integration
- Library source: `heartbeat/contingencies.json`.
- API read: `GET /contingency/library`.
- API execute: `POST /contingency/run?event=<name>`.
- Auto-trigger: `/status` executes `intrusion_detected` when intrusion hits are detected.
