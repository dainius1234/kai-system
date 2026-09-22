#!/usr/bin/env bash
#
# KAI-GATE-050 — does /graph/ingest report success over a failed pipeline?
#
# MEASUREMENT ONLY. Nothing is remediated: no timeout raised, no model
# changed, memu-graph/app.py untouched, cognee untouched,
# scripts/test_graph_live.py untouched.
#
# WHAT RUN 9 ESTABLISHED, AND WHAT IT DID NOT
# ===========================================
#
# Established from source:
#   * cognee raises PipelineRunFailedError when a data item fails
#     (run_tasks.py:147), then catches it and deliberately does NOT
#     re-raise it -- the one exception type meaning "the pipeline failed"
#     is the one excluded from re-raising (run_tasks.py:185-187, with the
#     intent in a comment). The failure travels as a RETURN VALUE.
#   * memu-graph/app.py:96 discards cognify's return value entirely, and
#     its only failure predicate is `except Exception` (:97). That
#     predicate cannot fire for this failure mode.
#
# Observed once, at runtime (run 31733359906):
#   * 422 PipelineRunFailedError inside, HTTP 200 {"status":"ingested"}
#     out, data_id null, after 396.3s.
#
# NOT established, and why this collector exists:
#   * whether the relationship REPRODUCES on a clean stack;
#   * what cognee's own TERMINAL marker for the failing pipeline is --
#     run 9 sampled the cognee log file every 20s and stopped when the
#     request returned, and the pipeline failed AFTER the last sample, so
#     the terminal line was never captured;
#   * whether memu-graph's 502 branch is reachable at all. R8: the
#     defects live in code that never runs, and if cognee never raises
#     for pipeline failure then app.py:97-99 is dead for the failure mode
#     that matters most.
#
# TWO CLEAN STACKS, NOT TWO REQUESTS
# ==================================
#
# A second request against the SAME stack shares cognee's databases, its
# dataset, and any cached pipeline state, so it is a weaker replication
# than it looks. Each observation here gets `down -v` and a fresh `up`.
#
set -uo pipefail

COMPOSE="docker-compose.full.yml"
SERVICE="memu-graph"
EVIDENCE="ingest-contract.evidence"
LOGDIR="contract-stage-logs"

# Run 9 measured one return at 396.3s. This watches well past that so a
# slower runner still yields a RETURN rather than a window expiry -- the
# correlation needs the response, and a timeout would supply none.
BUDGET=900
OBSERVATIONS=2

# R1 corollary: read git state BEFORE creating any file, or the
# collector's own evidence file counts as an uncommitted modification.
TREE_SHA="$(git rev-parse 'HEAD^{tree}' 2>/dev/null || echo UNKNOWN)"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
DIRTY="$(git status --porcelain 2>/dev/null | grep -c . || true)"

mkdir -p "$LOGDIR"
: > "$EVIDENCE"
record() { printf '%s\n' "$*" | tee -a "$EVIDENCE"; }

stage() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  "$@" > "$log" 2>&1
  local rc=$?
  local bytes; bytes=$(wc -c < "$log" | tr -d ' ')
  record "  stage ${name}: exit=${rc} bytes=${bytes} full-log=${log}"
  return $rc
}

excerpt() {
  local log="$1" lines="${2:-20}"
  local bytes; bytes=$(wc -c < "$log" | tr -d ' ')
  record "  --- EXCERPT: last ${lines} lines of ${log} (${bytes} bytes total) ---"
  tail -n "$lines" "$log" | sed 's/^/  | /' | tee -a "$EVIDENCE"
  record "  --- end excerpt ---"
}

record "KAI-GATE-050 — /graph/ingest contract vs internal pipeline outcome"
record "commit ${COMMIT}  tree ${TREE_SHA}  dirty ${DIRTY}"
record "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record ""
record "THE SUCCESS PREDICATE IS COGNEE'S TERMINAL PIPELINE STATUS, not the"
record "HTTP code. The HTTP code is the thing under suspicion."
record ""
record "${OBSERVATIONS} observation(s), each on its OWN clean stack, ${BUDGET}s budget."
record ""

for n in $(seq 1 "$OBSERVATIONS"); do
  record "=============================================================="
  record "== OBSERVATION ${n} — clean stack =="
  record "=============================================================="

  # A fresh stack every time, volumes included. `|| true` on the teardown
  # only: a failure to tear down a stack that does not yet exist is not a
  # measurement failure.
  docker compose -f "$COMPOSE" down -v > "$LOGDIR/down-${n}.log" 2>&1 || true

  stage "up-${n}" docker compose -f "$COMPOSE" up -d ollama ollama-pull "$SERVICE"
  if [ $? -ne 0 ]; then
    record ""
    record "MEASUREMENT ABORTED at observation ${n}: PREREQUISITE STARTUP FAILURE."
    record "  Neither the internal outcome nor the external one was measured."
    excerpt "$LOGDIR/up-${n}.log" 40
    exit 2
  fi

  stage "health-${n}" python3 scripts/ci/compose_probe.py \
      --compose-file "$COMPOSE" --services ollama "$SERVICE" --timeout 300
  health_rc=$?
  excerpt "$LOGDIR/health-${n}.log" 4
  if [ "$health_rc" -ne 0 ]; then
    # R11. A request fired at a subject that never became ready measures
    # the readiness failure, and would still produce a full-looking row.
    record ""
    record "MEASUREMENT ABORTED at observation ${n}: PREREQUISITE READINESS FAILURE."
    record "  compose_probe.py exit ${health_rc}. Nothing below was measured."
    exit 2
  fi

  record "  -- the request --"
  docker compose -f "$COMPOSE" exec -T "$SERVICE" \
      python - ingest "$BUDGET" "kai-gate-050-obs-${n}" \
      < scripts/security/probe_ingest_contract.py \
      > "$LOGDIR/ingest-${n}.log" 2>&1
  ingest_rc=$?
  record "  ingest probe exit: ${ingest_rc}"
  case "$ingest_rc" in
    0) record "    0 = the request RETURNED (any status; a 5xx is a return too)" ;;
    1) record "    1 = did NOT return inside ${BUDGET}s" ;;
    2) record "    2 = THE PROBE REJECTED ITS OWN COMMAND LINE. No request was"
       record "        sent. Instrument invocation failure, not a measurement." ;;
    *) record "    ${ingest_rc} = unassigned exit code; treat as unmeasured" ;;
  esac
  # The transcript is 3 lines. Printing it whole avoids an excerpt that
  # would hide the body -- the body IS the claim under test.
  record "  --- FULL ingest-${n}.log ---"
  sed 's/^/  | /' "$LOGDIR/ingest-${n}.log" | tee -a "$EVIDENCE"
  record "  --- end ---"

  # AFTER the request returns, so the terminal pipeline marker exists.
  # This is precisely what run 9 could not capture.
  record "  -- cognee's own terminal state, read AFTER the return --"
  docker compose -f "$COMPOSE" exec -T "$SERVICE" \
      python - cognee-log < scripts/security/probe_ingest_contract.py \
      > "$LOGDIR/cognee-log-${n}.log" 2>&1
  record "  cognee-log probe exit: $?"
  excerpt "$LOGDIR/cognee-log-${n}.log" 30

  stage "service-${n}" docker compose -f "$COMPOSE" logs "$SERVICE" \
      --no-color --timestamps
  record ""
done

docker compose -f "$COMPOSE" down -v > "$LOGDIR/down-final.log" 2>&1 || true

record "== RUN STATE, for the verdict step =="
{
  printf 'OBSERVATIONS=%s\n' "$OBSERVATIONS"
  printf 'BUDGET=%s\n' "$BUDGET"
  printf 'TREE_SHA=%s\n' "$TREE_SHA"
  printf 'COMMIT=%s\n' "$COMMIT"
  printf 'DIRTY=%s\n' "$DIRTY"
} > "$LOGDIR/rc.env"
sed 's/^/  /' "$LOGDIR/rc.env" | tee -a "$EVIDENCE"

record ""
record "collection finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "the correlation is the NEXT step, not this one."
exit 0
