#!/usr/bin/env bash
#
# KAI-GATE-048 C — Q1/Q2/Q6 capture. OBSERVATION ONLY.
#
# Changes nothing: no instructor mode, no model, no timeout, no retry
# count, no schema, no cognee validation, no network topology, no
# memu-graph production behaviour. The probe wraps one client method,
# calls through with the arguments it was given, and returns the original
# object.
#
# It drives cognee IN-PROCESS inside the same image rather than proxying
# the delegate or patching the live service -- a proxy would change the
# topology this is meant to observe, and would make the measurement about
# the proxy.
#
set -uo pipefail

COMPOSE="docker-compose.full.yml"
SERVICE="memu-graph"
EVIDENCE="llm-contract.evidence"
LOGDIR="llm-contract-logs"
# Run 11 measured ~391s for a full cognify with three attempts. This is
# generous enough that a slower runner still completes, and it bounds
# nothing the system does -- the probe has no client budget of its own.
BUDGET=1200

TREE_SHA="$(git rev-parse 'HEAD^{tree}' 2>/dev/null || echo UNKNOWN)"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
DIRTY="$(git status --porcelain 2>/dev/null | grep -c . || true)"

mkdir -p "$LOGDIR"
: > "$EVIDENCE"
record() { printf '%s\n' "$*" | tee -a "$EVIDENCE"; }

record "KAI-GATE-048 C — structured-output request/response capture"
record "commit ${COMMIT}  tree ${TREE_SHA}  dirty ${DIRTY}"
record "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record ""
record "OBSERVATION ONLY. No mode, model, timeout, retry, schema,"
record "validation or topology change. Stops after Q1/Q2/Q6."
record ""

docker compose -f "$COMPOSE" up -d ollama ollama-pull "$SERVICE" \
    > "$LOGDIR/up.log" 2>&1
if [ $? -ne 0 ]; then
  record "MEASUREMENT ABORTED: PREREQUISITE STARTUP FAILURE."
  record "  Q1/Q2/Q6 were NOT measured."
  tail -n 40 "$LOGDIR/up.log" | sed 's/^/  | /' | tee -a "$EVIDENCE"
  exit 2
fi

python3 scripts/ci/compose_probe.py --compose-file "$COMPOSE" \
    --services ollama "$SERVICE" --timeout 300 > "$LOGDIR/health.log" 2>&1
health_rc=$?
tail -n 4 "$LOGDIR/health.log" | sed 's/^/  | /' | tee -a "$EVIDENCE"
if [ "$health_rc" -ne 0 ]; then
  # R11: a capture against a subject that never became ready measures the
  # readiness failure and would still produce a full-looking table.
  record "MEASUREMENT ABORTED: PREREQUISITE READINESS FAILURE (exit ${health_rc})."
  record "  Q1/Q2/Q6 were NOT measured."
  exit 2
fi

record "== CAPTURE — in-process, same image, same environment =="
timeout "$BUDGET" docker compose -f "$COMPOSE" exec -T "$SERVICE" \
    python - < scripts/security/probe_llm_contract.py \
    > "$LOGDIR/capture.jsonl" 2>"$LOGDIR/capture.err"
probe_rc=$?
record "  probe exit: ${probe_rc}"
case "$probe_rc" in
  0)   record "    0 = the capture ran to completion" ;;
  2)   record "    2 = the probe could not install its capture point; nothing measured" ;;
  124) record "    124 = the ${BUDGET}s collection bound elapsed. This is the"
       record "          COLLECTOR's bound, not a system property, and any"
       record "          attempts already written below are still valid." ;;
  *)   record "    ${probe_rc} = unassigned exit code; treat as unmeasured" ;;
esac
bytes=$(wc -c < "$LOGDIR/capture.jsonl" | tr -d ' ')
lines=$(grep -c . "$LOGDIR/capture.jsonl" || true)
record "  capture: ${lines} record(s), ${bytes} bytes, full-log=${LOGDIR}/capture.jsonl"
record "  --- EXCERPT: last 6 lines of ${LOGDIR}/capture.err ---"
tail -n 6 "$LOGDIR/capture.err" 2>/dev/null | sed 's/^/  | /' | tee -a "$EVIDENCE"
record "  --- end excerpt ---"
record ""

docker compose -f "$COMPOSE" logs "$SERVICE" --no-color --timestamps \
    > "$LOGDIR/service.log" 2>&1
docker compose -f "$COMPOSE" logs ollama --no-color --timestamps \
    > "$LOGDIR/ollama.log" 2>&1
record "  service log: $(wc -c < "$LOGDIR/service.log" | tr -d ' ') bytes"
record "  ollama log:  $(wc -c < "$LOGDIR/ollama.log" | tr -d ' ') bytes"

docker compose -f "$COMPOSE" down -v > "$LOGDIR/down.log" 2>&1 || true

record ""
record "collection finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "the classification is the NEXT step, not this one."
exit 0
