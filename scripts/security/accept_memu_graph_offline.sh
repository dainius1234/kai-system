#!/usr/bin/env bash
#
# KAI-GATE-048 Phase 1 acceptance: is the remediation runtime-proven?
#
# D191's closure condition, which a green build does NOT satisfy:
#
#   memu-graph remains correctly lazy and readiness-independent, AND its
#   first model-dependent request succeeds without external
#   model-registry egress because the required tokenizer asset is
#   locally satisfiable.
#
# TWO ACCEPTANCE LEVELS, AND THEY USE DIFFERENT TOPOLOGIES
# ========================================================
#
# D190 originally proposed `--network none` as the service-level
# criterion. That was wrong and D191 corrected it. `memu-graph`
# deliberately delegates LLM and embedding work to `ollama`
# (LLM_ENDPOINT / EMBEDDING_ENDPOINT -> http://ollama:11434). Its
# contract is *no external registry egress*, NOT *no networking*.
# Requiring a real /graph/ingest to work under `--network none` would
# demand the service survive without the internal peer it is explicitly
# designed to use — a stricter and DIFFERENT topology than production.
#
#   A, D  asset level      --network none        the baked asset is
#                                                locally complete
#   B, C  capability level the repo-defined      no external egress
#                          intended topology     while the real
#                                                capability works
#
# Same trap as `ollama-pull`, one level up: a rule of "no egress + a
# model = broken" reports a correct design as a defect. Here it would
# have been my own acceptance criterion doing it.
#
# EVERY STAGE KEEPS ITS FULL OUTPUT (R10) AND ABORTS AT THE PREREQUISITE
# BOUNDARY (R11). The summary prints LAST, because the Actions log API
# serves a fixed byte window from the end of a job.
#
set -uo pipefail

COMPOSE="docker-compose.full.yml"
SERVICE="memu-graph"
EVIDENCE="memu-graph-acceptance.evidence"
LOGDIR="acceptance-stage-logs"
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

record "KAI-GATE-048 Phase 1 ACCEPTANCE — memu-graph offline asset contract"
record "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record ""

# ── E. artefact identity, first, so everything below is bound to it ──
record "== E. ARTEFACT IDENTITY =="
TREE_SHA="$(git rev-parse 'HEAD^{tree}' 2>/dev/null || echo UNKNOWN)"
COMMIT="$(git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
DIRTY="$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
record "  commit          ${COMMIT}"
record "  TESTED TREE SHA ${TREE_SHA}"
record "  working tree modifications: ${DIRTY}"
if [ "$DIRTY" != "0" ]; then
  record "  WARNING: the tree under test is NOT the committed tree. Any"
  record "  evidence below describes something that was never committed."
fi

IMAGE=$(docker compose -f "$COMPOSE" images -q "$SERVICE" 2>/dev/null | head -1)
if [ -z "$IMAGE" ]; then
  IMAGE=$(docker images --format '{{.ID}}' --filter 'reference=*memu-graph*' | head -1)
fi
record "  image id        ${IMAGE:-<unresolved>}"
if [ -z "$IMAGE" ]; then
  record ""
  record "ABORTED: image unresolved. Nothing below was measured."
  record "VERDICT: UNKNOWN (prerequisite: no image)"
  exit 2
fi
record ""

# ── A. asset proof, from the FINAL image, with no network at all ─────
record "== A. FINAL-IMAGE ASSET PROOF — --network none =="
record "  the same verify path the build ran, but against the shipped"
record "  image in a container that has no network interface at all."
stage "A-final-image-offline" docker run --rm --network none "$IMAGE" \
    python /tmp/bake_tokenizer.py verify
a_rc=$?
excerpt "$LOGDIR/A-final-image-offline.log" 12
record ""

# ── D. can-fail calibration, DEMONSTRATED ────────────────────────────
record "== D. CAN-FAIL CALIBRATION — the same check, asset withheld =="
record "  HF_HOME redirected to an empty path in the same image. If this"
record "  passes, stage A proves nothing: the check would be reporting"
record "  something other than the asset."
stage "D-canfail-no-asset" docker run --rm --network none \
    -e HF_HOME=/tmp/deliberately-empty "$IMAGE" \
    python /tmp/bake_tokenizer.py verify
d_rc=$?
excerpt "$LOGDIR/D-canfail-no-asset.log" 12
record "  (a NON-ZERO exit here is the PASS. Zero would mean the check"
record "   cannot distinguish a present asset from an absent one.)"
record ""

# ── the intended topology, from the repo-defined bring-up ────────────
record "== INTENDED TOPOLOGY — core-tests.yml's bring-up, unchanged =="
record "  docker compose -f $COMPOSE up -d ollama ollama-pull $SERVICE"
stage "up" docker compose -f "$COMPOSE" up -d ollama ollama-pull "$SERVICE"
if [ $? -ne 0 ]; then
  record ""
  record "MEASUREMENT ABORTED: PREREQUISITE STARTUP FAILURE."
  record "  B and C were NOT measured. A and D above stand on their own,"
  record "  because they never needed the stack."
  excerpt "$LOGDIR/up.log" 40
  record "VERDICT: INCOMPLETE (asset level only)"
  exit 2
fi
stage "probe" python3 scripts/ci/compose_probe.py \
    --compose-file "$COMPOSE" --services ollama "$SERVICE" --timeout 300
probe_rc=$?
excerpt "$LOGDIR/probe.log" 8
CID=$(docker compose -f "$COMPOSE" ps -q "$SERVICE" 2>/dev/null)
record "  container id    ${CID:-<none>}"
record ""

# ── B. readiness preserved, model NOT loaded prematurely ─────────────
record "== B. READINESS PRESERVED — lazy design must be intact =="
stage "B-chronology" python3 scripts/ci/health_chronology.py "$CID"
excerpt "$LOGDIR/B-chronology.log" 12
record "  and the serving process must NOT have the tokenizer machinery"
record "  mapped before any request — the baked cache is present now, so"
record "  file counts can no longer distinguish this. /proc/1/maps can."
stage "B-maps-ready" docker compose -f "$COMPOSE" exec -T "$SERVICE" sh -c \
    'for m in tokenizers torch safetensors; do printf "%s: " "$m"; grep -c "$m" /proc/1/maps 2>/dev/null || echo 0; done'
excerpt "$LOGDIR/B-maps-ready.log" 8
record ""

# ── C. capability under the intended topology ────────────────────────
record "== C. CAPABILITY — intended topology, internal peers intact =="
record "  C1: external registry egress must remain UNAVAILABLE"
stage "C1-external-egress" docker compose -f "$COMPOSE" exec -T "$SERVICE" python - <<'PY'
import socket
for host, port in (("huggingface.co", 443), ("1.1.1.1", 443)):
    try:
        s = socket.create_connection((host, port), timeout=5); s.close()
        print(f"{host}:{port} CONNECTED  -> EXTERNAL EGRESS AVAILABLE")
    except Exception as exc:
        print(f"{host}:{port} FAILED ({type(exc).__name__}) -> no external egress")
PY
excerpt "$LOGDIR/C1-external-egress.log" 6

record "  C2: the internal delegate must remain REACHABLE — this is the"
record "      half a --network none test would have destroyed"
stage "C2-internal-reachability" docker compose -f "$COMPOSE" exec -T "$SERVICE" python - <<'PY'
import socket, urllib.request
try:
    s = socket.create_connection(("ollama", 11434), timeout=5); s.close()
    print("ollama:11434 CONNECTED -> internal delegate REACHABLE")
except Exception as exc:
    print(f"ollama:11434 FAILED ({type(exc).__name__}: {exc}) -> delegate UNREACHABLE")
try:
    with urllib.request.urlopen("http://ollama:11434/api/tags", timeout=10) as r:
        print(f"ollama /api/tags HTTP {r.status} -> delegate SERVING")
except Exception as exc:
    print(f"ollama /api/tags FAILED ({type(exc).__name__}: {exc})")
PY
excerpt "$LOGDIR/C2-internal-reachability.log" 6

record "  C3: the real model-dependent operation — the same live cycle"
record "      core-tests.yml already runs, not a new route"
stage "C3-live-cycle" sh -c \
    "docker compose -f '$COMPOSE' exec -T '$SERVICE' python - < scripts/test_graph_live.py"
c3_rc=$?
record "  live-cycle exit: $c3_rc"
excerpt "$LOGDIR/C3-live-cycle.log" 30

record "  C4: and the service log must show NO Hugging Face retry sequence"
stage "C4-service-logs" docker compose -f "$COMPOSE" logs "$SERVICE" --no-color --timestamps
excerpt "$LOGDIR/C4-service-logs.log" 25
record ""

# The VERDICT is deliberately NOT computed here. The collector gathers;
# a separate, workflow-visible step decides. That split matters twice
# over: the deciding half stays calibratable without a daemon, and the
# gate becomes something `check_gate_registry`'s workflow parse can
# actually see rather than a claim about a shell script's exit code.
record "== EXIT CODES, for the verdict step =="
{
  printf 'PROBE_RC=%s\n' "$probe_rc"
  printf 'A_RC=%s\n' "$a_rc"
  printf 'D_RC=%s\n' "$d_rc"
  printf 'C3_RC=%s\n' "$c3_rc"
  printf 'TREE_SHA=%s\n' "$TREE_SHA"
  printf 'COMMIT=%s\n' "$COMMIT"
  printf 'IMAGE=%s\n' "$IMAGE"
  printf 'DIRTY=%s\n' "$DIRTY"
} > "$LOGDIR/rc.env"
cat "$LOGDIR/rc.env" | sed 's/^/  /' | tee -a "$EVIDENCE"

record ""
record "collection finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "full evidence: $EVIDENCE   stage logs: $LOGDIR/"
record "the verdict is the NEXT step, not this one."
exit 0
