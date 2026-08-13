#!/usr/bin/env bash
#
# KAI-GATE-048: does `memu-graph` acquire a model BEFORE its declared
# readiness boundary, and what local/offline machinery does that path
# have?
#
# THE BRING-UP IS NOT INVENTED HERE
# ================================
#
# The command below is copied from the repo-defined path that already
# starts this service — `.github/workflows/core-tests.yml`, the step
# "Bring up memu-graph (Cognee/Kuzu live verification)". The brief was
# explicit: do not create a synthetic startup route. So the bring-up, the
# readiness wait and the request-time exercise are all the existing ones;
# what is new is only the OBSERVATION around them.
#
# WHY THE HF CACHE IS THE INSTRUMENT
# ==================================
#
# `memu-graph` never calls a model loader in repo source — cognee's
# chunker does, via `transformers`. There is no line to instrument. But a
# HuggingFace tokenizer load has an unavoidable side effect: it
# materialises files under $HF_HOME. So the cache is snapshotted three
# times,
#
#   1. in the IMAGE, in a throwaway container with `--network none`, so
#      it cannot fetch anything while being looked at;
#   2. at READINESS, before any request has been made;
#   3. AFTER the repo-defined request-time exercise.
#
# and the differences between them are the chronology. Snapshot 2 minus
# snapshot 1 is what startup did; snapshot 3 minus snapshot 2 is what the
# first request did. Neither is inferred from an import graph.
#
# `/proc/1/maps` is read as a second, independent signal: `tokenizers` is
# a compiled extension, so its presence in the serving process's mapped
# files says the tokenizer machinery is loaded THERE — not in some
# `docker exec` python that would have proved nothing about the server.
# Two signals from different mechanisms, per I-8.
#
# EGRESS IS PROVEN, NOT INFERRED
# ==============================
#
# `internal: true` in a compose file is a declaration. The probe below
# opens a socket from inside the container and reports what happened.
# The brief said not to infer "no egress" from naming; this does not
# infer it from configuration either.
#
# R10 / R11, both earned this week
# ================================
#
#   * every stage keeps its FULL stdout+stderr under stage-logs/, records
#     the byte count, and any inline excerpt says it is one;
#   * if the bring-up fails, this ABORTS at the prerequisite boundary
#     instead of collecting a table of correct-looking rows that all say
#     the same thing. Run 1 of the degradation proof recorded fifty
#     probes against a stack that did not exist.
#
# The summary prints LAST, deliberately: the Actions log API serves a
# fixed byte window from the END of the job, measured at ~15.8KB, and
# anything outside it may as well not exist.
#
set -uo pipefail

COMPOSE="docker-compose.full.yml"
EVIDENCE="memu-graph-startup.evidence"
LOGDIR="stage-logs"
SERVICE="memu-graph"
mkdir -p "$LOGDIR"
: > "$EVIDENCE"

record() { printf '%s\n' "$*" | tee -a "$EVIDENCE"; }

# Run a stage, keep everything, report the size. Never truncates without
# saying so.
stage() {
  local name="$1"; shift
  local log="$LOGDIR/${name}.log"
  "$@" > "$log" 2>&1
  local rc=$?
  local bytes
  bytes=$(wc -c < "$log" | tr -d ' ')
  record "  stage ${name}: exit=${rc} bytes=${bytes} full-log=${log}"
  return $rc
}

# A tail excerpt that announces itself as one.
excerpt() {
  local log="$1" lines="${2:-25}"
  local bytes; bytes=$(wc -c < "$log" | tr -d ' ')
  record "  --- EXCERPT: last ${lines} lines of ${log} (${bytes} bytes total) ---"
  tail -n "$lines" "$log" | sed 's/^/  | /' | tee -a "$EVIDENCE"
  record "  --- end excerpt ---"
}

cid() { docker compose -f "$COMPOSE" ps -q "$SERVICE" 2>/dev/null; }

record "KAI-GATE-048 — memu-graph model-startup chronology"
record "compose file: $COMPOSE"
record "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record ""

# ── 0. IMAGE-level: what the built image already carries ─────────────
record "== 0. IMAGE — the cache as built, looked at with no network =="
IMAGE=$(docker compose -f "$COMPOSE" config --format json 2>/dev/null \
        | python3 -c 'import json,sys; print((json.load(sys.stdin)["services"]["memu-graph"].get("image") or ""))')
if [ -z "$IMAGE" ]; then
  IMAGE=$(docker compose -f "$COMPOSE" images -q "$SERVICE" 2>/dev/null | head -1)
fi
record "  image ref: ${IMAGE:-<unresolved>}"
if [ -n "$IMAGE" ]; then
  stage "image-cache" docker run --rm --network none --entrypoint sh "$IMAGE" \
      -c 'echo "HF_HOME=${HF_HOME:-<unset>}"; echo "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"; echo "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-<unset>}"; echo "--- find ${HF_HOME:-/data/hf_cache} ---"; find "${HF_HOME:-/data/hf_cache}" -type f 2>/dev/null | head -200; echo "--- file count ---"; find "${HF_HOME:-/data/hf_cache}" -type f 2>/dev/null | wc -l'
  excerpt "$LOGDIR/image-cache.log" 20
else
  record "  IMAGE SNAPSHOT NOT TAKEN — image reference unresolved."
fi
record ""

# ── 1. the repo-defined bring-up, unmodified ─────────────────────────
record "== 1. BRING-UP — the command from core-tests.yml, unchanged =="
record "  docker compose -f $COMPOSE up -d ollama ollama-pull $SERVICE"
stage "up" docker compose -f "$COMPOSE" up -d ollama ollama-pull "$SERVICE"
up_rc=$?
if [ $up_rc -ne 0 ]; then
  record ""
  record "MEASUREMENT ABORTED: PREREQUISITE STARTUP FAILURE (up exit $up_rc)."
  record "  The subject of this measurement does not exist, so nothing"
  record "  below it was measured. This is the instrument working, not"
  record "  failing: the unmet prerequisite is named and the dependent"
  record "  observations are refused."
  record ""
  record "  NOT MEASURED as a result:"
  record "    - readiness boundary and health chronology"
  record "    - loader chronology (cache snapshots 2 and 3)"
  record "    - egress probe"
  record "    - request-time exercise"
  excerpt "$LOGDIR/up.log" 40
  stage "abort-ps" docker compose -f "$COMPOSE" ps -a
  excerpt "$LOGDIR/abort-ps.log" 20
  record ""
  record "VERDICT: UNKNOWN (prerequisite startup failure)"
  exit 2
fi

record "  readiness wait: the repo-defined probe, same services, same timeout"
stage "probe" python3 scripts/ci/compose_probe.py \
    --compose-file "$COMPOSE" --services ollama "$SERVICE" --timeout 300
probe_rc=$?
excerpt "$LOGDIR/probe.log" 15
record ""

# ── 2. readiness boundary, from the daemon ───────────────────────────
record "== 2. READINESS BOUNDARY — the daemon's record, not the app's =="
CID=$(cid)
record "  container id: ${CID:-<none>}"
if [ -z "$CID" ]; then
  record "  ABORTED: no container id. Nothing below was measured."
  record "VERDICT: UNKNOWN (container not found)"
  exit 2
fi
stage "inspect-health" docker inspect "$CID" \
    --format '{{json .State}}'
stage "inspect-config" docker inspect "$CID" \
    --format '{{json .Config.Healthcheck}}{{"\n"}}{{json .NetworkSettings.Networks}}'
# Elapsed is COMPUTED from StartedAt, never judged by eye -- D183/D184.
stage "chronology" python3 scripts/ci/health_chronology.py "$CID"
excerpt "$LOGDIR/chronology.log" 30
record ""

# ── 3. egress, proven from inside ────────────────────────────────────
record "== 3. EGRESS — a socket, not a compose declaration =="
stage "egress-probe" docker compose -f "$COMPOSE" exec -T "$SERVICE" python - <<'PY'
import socket, sys
for host, port in (("huggingface.co", 443), ("1.1.1.1", 443)):
    try:
        socket.setdefaulttimeout(5)
        s = socket.create_connection((host, port), timeout=5)
        s.close()
        print(f"{host}:{port} CONNECTED  -> egress AVAILABLE")
    except Exception as exc:
        print(f"{host}:{port} FAILED ({type(exc).__name__}: {exc}) -> no egress on this path")
PY
excerpt "$LOGDIR/egress-probe.log" 10
record ""

# ── 4. snapshot 2 — the cache AT READINESS, before any request ───────
record "== 4. CACHE AT READINESS — before a single request is made =="
stage "cache-ready" docker compose -f "$COMPOSE" exec -T "$SERVICE" sh -c \
    'echo "HF_HOME=${HF_HOME:-<unset>}"; find "${HF_HOME:-/data/hf_cache}" -type f 2>/dev/null | head -200; echo "--- file count ---"; find "${HF_HOME:-/data/hf_cache}" -type f 2>/dev/null | wc -l'
excerpt "$LOGDIR/cache-ready.log" 20
record "  second, independent signal — compiled extensions mapped into"
record "  the SERVING process (pid 1), which a fresh exec could not show:"
stage "maps-ready" docker compose -f "$COMPOSE" exec -T "$SERVICE" sh -c \
    'for m in tokenizers transformers torch safetensors cognee; do printf "%s: " "$m"; grep -c "$m" /proc/1/maps 2>/dev/null || echo 0; done'
excerpt "$LOGDIR/maps-ready.log" 10
record ""

# ── 5. the repo-defined request-time exercise ────────────────────────
record "== 5. REQUEST-TIME — the exercise core-tests.yml already runs =="
record "  docker compose exec -T $SERVICE python - < scripts/test_graph_live.py"
stage "live-cycle" sh -c \
    "docker compose -f '$COMPOSE' exec -T '$SERVICE' python - < scripts/test_graph_live.py"
live_rc=$?
record "  live-cycle exit: $live_rc (best-effort in CI; recorded either way)"
excerpt "$LOGDIR/live-cycle.log" 40
record ""

# ── 6. snapshot 3 — the cache AFTER the request ──────────────────────
record "== 6. CACHE AFTER THE REQUEST — snapshot 3 =="
stage "cache-after" docker compose -f "$COMPOSE" exec -T "$SERVICE" sh -c \
    'find "${HF_HOME:-/data/hf_cache}" -type f 2>/dev/null | head -200; echo "--- file count ---"; find "${HF_HOME:-/data/hf_cache}" -type f 2>/dev/null | wc -l'
excerpt "$LOGDIR/cache-after.log" 20
stage "maps-after" docker compose -f "$COMPOSE" exec -T "$SERVICE" sh -c \
    'for m in tokenizers transformers torch safetensors cognee; do printf "%s: " "$m"; grep -c "$m" /proc/1/maps 2>/dev/null || echo 0; done'
excerpt "$LOGDIR/maps-after.log" 10
record ""

record "== 7. APPLICATION LOG — full, kept, excerpted with its size =="
stage "service-logs" docker compose -f "$COMPOSE" logs "$SERVICE" --no-color --timestamps
excerpt "$LOGDIR/service-logs.log" 60
record ""

# ── 8. classify, from the observations only ──────────────────────────
record "== 8. CLASSIFICATION — from the observations above, by a"
record "      function that takes no service name =="
python3 scripts/security/summarise_memu_graph_startup.py \
    --stage-logs "$LOGDIR" --probe-rc "$probe_rc" --live-rc "$live_rc" \
    2>&1 | tee -a "$EVIDENCE"

record ""
record "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "full evidence: $EVIDENCE   stage logs: $LOGDIR/"
exit 0
