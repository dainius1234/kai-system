#!/usr/bin/env bash
#
# KAI-GATE-049 — what exactly owns the silence after chunking completes?
#
# DIAGNOSTIC ONLY. Nothing is remediated. No timeout is raised in any
# shipped path, no model is changed, memu-graph is not modified, and
# `scripts/test_graph_live.py` is untouched.
#
# THE FIRST THING THIS UNIT HAD TO CORRECT WAS ITS OWN PREMISE
# ============================================================
#
# Runs 4 and 6 both reported ~291-292s of silence in
# `extract_graph_and_summarize`, from two different runners. That looked
# like a reproducible system boundary. The arithmetic says otherwise:
#
#     cognee log file created            16:55:42
#     + 300s  (test_graph_live.py ingest budget)
#     =                                  17:00:42   == when C3 gave up
#     extract_graph_and_summarize began  16:55:51   -> 291s of "silence"
#
# The two runs agree because both used the same 300s client budget, not
# because the system did the same thing twice. **We have never observed
# what the operation does; we have observed when our own client stops
# watching.** Measuring again with that client would keep reproducing
# our own number — R9's watcher, and I-8's rule that evidence and claim
# must come from different places.
#
# Nothing in cognee's LLM stack contains a 290-300s constant either. The
# ollama adapter carries `max_retries=2` and no explicit timeout.
#
# So this collector watches PAST that budget, and the window below is
# labelled an OBSERVATION WINDOW rather than a property of the system.
#
# THE PIPELINE IS DERIVED FROM COGNEE'S SOURCE, NOT INVENTED
# ==========================================================
#
# cognee/api/v1/cognify/cognify.py:315-341 --
#
#   classify_documents
#   extract_chunks_from_documents          <- the tokenizer step
#   extract_graph_and_summarize            <- LAST marker seen
#       = asyncio.gather( extract_graph_from_data, summarize_text )
#   add_data_points                        <- persistence + embeddings
#   extract_dlt_fk_edges
#
# `extract_graph_and_summarize` fans out into TWO concurrent LLM paths.
# Whichever is slow, cognee emits one "started" marker for the pair, so
# its own logging cannot tell them apart. That is why the samples below
# read CPU and sockets rather than waiting for a nicer log line.
#
# WHAT EACH SAMPLE DISTINGUISHES
# ==============================
#
#   CPU growing, socket to :11434 open   -> genuinely slow LLM work
#   CPU flat,    socket to :11434 open   -> waiting on the delegate
#   CPU flat,    no socket               -> stuck somewhere else
#   CPU growing, no socket               -> local compute, not the LLM
#
# A timeout can express none of those, which is why raising one would
# answer nothing.
#
set -uo pipefail

COMPOSE="docker-compose.full.yml"
SERVICE="memu-graph"
EVIDENCE="graph-stall.evidence"
LOGDIR="stall-stage-logs"

# THE OBSERVATION WINDOW IS DERIVED AND LABELLED AS SUCH.
#
# The only configured bound anywhere in this path is the live cycle's
# 300s ingest budget. This watches 3x that, so a completion anywhere
# near the existing budget is observable and one well past it is too.
# If the operation has still not returned at the end, that is the WINDOW
# ending -- not a proven hang, and the report says so.
LIVE_CYCLE_BUDGET=300
WINDOW=$(( LIVE_CYCLE_BUDGET * 3 ))
SAMPLE_EVERY=20

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

record "KAI-GATE-049 — stage decomposition of the post-chunking silence"
record "commit ${COMMIT}  tree ${TREE_SHA}  dirty ${DIRTY}"
record "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record ""
record "OBSERVATION WINDOW ${WINDOW}s = 3 x the live cycle's ${LIVE_CYCLE_BUDGET}s"
record "ingest budget, which is the ONLY configured bound in this path and"
record "is the number runs 4 and 6 actually reproduced. The window is a"
record "property of this measurement, not of the system."
record ""

# ── bring-up: the repo-defined route, unchanged ──────────────────────
record "== BRING-UP — core-tests.yml's command, unchanged =="
stage "up" docker compose -f "$COMPOSE" up -d ollama ollama-pull "$SERVICE"
if [ $? -ne 0 ]; then
  record ""
  record "MEASUREMENT ABORTED: PREREQUISITE STARTUP FAILURE."
  record "  Nothing below was measured."
  excerpt "$LOGDIR/up.log" 40
  exit 2
fi
stage "probe-health" python3 scripts/ci/compose_probe.py \
    --compose-file "$COMPOSE" --services ollama "$SERVICE" --timeout 300
excerpt "$LOGDIR/probe-health.log" 6
record ""

# ── the delegate's state BEFORE the request ──────────────────────────
record "== DELEGATE BASELINE — what ollama has been asked so far =="
stage "ollama-baseline" docker compose -f "$COMPOSE" logs ollama --no-color --timestamps
excerpt "$LOGDIR/ollama-baseline.log" 10
record ""

# ── fire the ingest, in the background, watching past our own budget ─
record "== INGEST — fired with a ${WINDOW}s budget, NOT the 300s one =="
docker compose -f "$COMPOSE" exec -T "$SERVICE" \
    python - "$WINDOW" < scripts/security/probe_graph_stall.py \
    > "$LOGDIR/ingest.log" 2>&1 &
INGEST_PID=$!
record "  ingest probe pid (on the runner, not in the container): ${INGEST_PID}"
record "  sampling every ${SAMPLE_EVERY}s until it returns or the window ends"
record ""

# The watcher waits on a PID CAPTURED BEFORE THE LOOP, never on a name.
# R9: `pgrep -f` on a pattern the waiting shell's own command line
# contains finds itself and waits forever. Eight of those accumulated in
# one stint and the gates they guarded never ran.
: > "$LOGDIR/samples.log"
elapsed=0
while kill -0 "$INGEST_PID" 2>/dev/null && [ "$elapsed" -lt "$WINDOW" ]; do
  {
    printf '=== sample at +%ss ===\n' "$elapsed"
    docker compose -f "$COMPOSE" exec -T "$SERVICE" \
        python - sample < scripts/security/probe_graph_stall.py 2>&1
  } >> "$LOGDIR/samples.log"
  sleep "$SAMPLE_EVERY"
  elapsed=$(( elapsed + SAMPLE_EVERY ))
done

wait "$INGEST_PID"
ingest_rc=$?
record "  ingest probe exit: ${ingest_rc}  (1 = did not return inside the window)"
record "  observation ended at +${elapsed}s"
stage "ingest-result" cat "$LOGDIR/ingest.log"
excerpt "$LOGDIR/ingest.log" 15
record ""

record "== SAMPLES — CPU, sockets and cognee's own log file =="
bytes=$(wc -c < "$LOGDIR/samples.log" | tr -d ' ')
record "  stage samples: bytes=${bytes} full-log=${LOGDIR}/samples.log"
excerpt "$LOGDIR/samples.log" 40
record ""

# ── after the window: what did each side actually record? ────────────
record "== AFTER — the two logs that were never read together =="
stage "service-logs" docker compose -f "$COMPOSE" logs "$SERVICE" --no-color --timestamps
excerpt "$LOGDIR/service-logs.log" 25
stage "ollama-after" docker compose -f "$COMPOSE" logs ollama --no-color --timestamps
excerpt "$LOGDIR/ollama-after.log" 25
record ""

record "== EXIT CODES, for the verdict step =="
{
  printf 'INGEST_RC=%s\n' "$ingest_rc"
  printf 'WINDOW=%s\n' "$WINDOW"
  printf 'ELAPSED=%s\n' "$elapsed"
  printf 'LIVE_CYCLE_BUDGET=%s\n' "$LIVE_CYCLE_BUDGET"
  printf 'TREE_SHA=%s\n' "$TREE_SHA"
  printf 'COMMIT=%s\n' "$COMMIT"
  printf 'DIRTY=%s\n' "$DIRTY"
} > "$LOGDIR/rc.env"
cat "$LOGDIR/rc.env" | sed 's/^/  /' | tee -a "$EVIDENCE"

record ""
record "collection finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "the analysis is the NEXT step, not this one."
exit 0
