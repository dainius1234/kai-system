#!/usr/bin/env bash
#
# #41-B DEPLOYED: what the contained core actually does, in containers,
# when profile-gated dependencies are absent.
#
# NO PROFILE IS ENABLED. That is the point: profiles-off is the intended
# posture, so this measures the system as it is meant to run.
#
# THE INVARIANT THIS SCRIPT IS BUILT AROUND
# =========================================
#
#   A component under test is not a verification authority.
#
# The dashboard is the SUBJECT of this measurement. Its own status
# payload therefore cannot establish whether a dependency is absent, and
# nothing here lets it. Ground truth comes from two places it cannot
# influence:
#
#   1. a DNS + connect probe run INSIDE a container, which reports the
#      transport boundary directly;
#   2. `docker inspect` health, which is the daemon's opinion of the
#      container, not the application's opinion of itself.
#
# Only after both are recorded is the caller's own behaviour observed,
# and the comparison is the finding. Dashboard output never creates a
# PASS.
#
# THE OBSERVATION WINDOW IS DERIVED, NOT CHOSEN
# =============================================
#
# The arithmetic below establishes that 150s CONSERVATIVELY EXCEEDS the
# configured health / retry / circuit-breaker envelope. It is NOT a claim
# that Docker cannot produce an unhealthy verdict before exactly 100s --
# that would be an assertion about the daemon's implementation, which
# this has not measured. The OBSERVED health-transition timestamps,
# recorded per container below, are authoritative for any given run.
#
#   dashboard healthcheck start_period          10s
#   interval x retries                          30s x 3 = 90s
#   -> earliest a health verdict can exist     100s
#   one further interval, to see it settle      30s
#   slowest single call chain                  ~20.9s  (_proxy_get:
#                                              10s timeout x 2 attempts
#                                              + 0.3 + 0.6 backoff)
#   circuit-breaker recovery                    30s
#
# A shorter window measures the start-up grace period and calls it
# health. Every term is read from the compose file and the source; none
# is a round number someone liked.
#
set -uo pipefail

COMPOSE="${1:-docker-compose.minimal.yml}"
WINDOW="${DEGRADATION_WINDOW_SECONDS:-150}"
EVIDENCE="degradation-deployed.evidence"
: > "$EVIDENCE"

record() { printf '%s\n' "$*" | tee -a "$EVIDENCE"; }

# RUN 1 DESTROYED ITS OWN EVIDENCE HERE. `compose up` failed, and this
# function recorded `cut -c1-1500` of a build log tens of kilobytes long
# -- so the record held the FIRST 1500 characters (dockerfile parsing)
# and the failure at the END was cut off. Every downstream probe then
# correctly reported "service dashboard is not running", and the run
# could not say why.
#
# This is the run-2-of-#47 defect wearing different clothes: there, a
# resolver sent stdout to /dev/null; here, a collector kept the wrong
# 1500 characters. Diagnostics fail at the same place every time --
# where the output is large and the interesting part is at the end.
#
# So: full output is kept as a FILE and uploaded, and the recorded
# excerpt is the TAIL, because that is where errors live.
STAGE_LOGS="stage-logs"
mkdir -p "$STAGE_LOGS"

stage() {
  local name="$1"; shift
  local out err rc
  out="$STAGE_LOGS/${name}.out"; err="$STAGE_LOGS/${name}.err"
  if "$@" > "$out" 2> "$err"; then rc=0; else rc=$?; fi
  record "stage=$name"
  record "  cmd=$*"
  record "  exit=$rc"
  record "  stdout_bytes=$(wc -c < "$out") stderr_bytes=$(wc -c < "$err")"
  record "  stdout_TAIL=$( [ -s "$out" ] && tail -c 1200 "$out" | tr '\n' '|' || echo '(empty)' )"
  record "  stderr_TAIL=$( [ -s "$err" ] && tail -c 1200 "$err" | tr '\n' '|' || echo '(empty)' )"
  STAGE_OUT="$(cat "$out")"
  return $rc
}

record "compose_file=$COMPOSE"
record "profiles_enabled=NONE (intended contained-core posture)"
record "observation_window_seconds=$WINDOW"

# ── 0. the flag must not leak in, as in #47 ─────────────────────────────
if [ -n "${COMPOSE_PROFILES:-}" ]; then
  record "REFUSING: COMPOSE_PROFILES is set to '${COMPOSE_PROFILES}'."
  record "This measurement is only meaningful with every profile OFF."
  exit 2
fi

# ── 1. which dependencies are supposed to be absent ─────────────────────
#
# Derived from the registered topology report, never listed here. A list
# beside the thing is the defect this programme keeps finding.
GATED="$(python3 - <<'PY'
import sys
sys.path.insert(0, ".")
from scripts.security import report_degradation_tolerance as dt
print(" ".join(sorted({e["dependency"] for e in dt.edges()})))
PY
)"
record "gated_dependencies=$GATED"
record "gated_count=$(printf '%s\n' $GATED | wc -w)"

# ── 2. bring up the contained core ──────────────────────────────────────
stage compose_up docker compose -f "$COMPOSE" up -d --build
up_rc=$?
record "compose_up_exit=$up_rc"

# WHICH services actually exist, and in what state. Run 1 continued past
# a failed bring-up into 50 probes that could only say "not running",
# producing a full evidence table about nothing. The state of the stack
# is established BEFORE anything is asked of it.
stage compose_ps docker compose -f "$COMPOSE" ps -a
if [ "$up_rc" -ne 0 ]; then
  record ""
  record "MEASUREMENT ABORTED: the contained core did not come up (exit $up_rc)."
  record "This is an INSTRUMENT/ENVIRONMENT failure, NOT a finding about"
  record "default-core degradation. Nothing below would measure the system;"
  record "it would measure the absence of the system. Classification order:"
  record "  instrument/environment failure  <-- THIS RUN"
  record "  actual default-core defect      -- not established"
  record "  #53 deployed manifestation      -- not established"
  record "  successful bounded degradation  -- not established"
  record "The failing stage's full output is in $STAGE_LOGS/compose_up.{out,err}."
  exit 2
fi

# ── 3. wait out the DERIVED window ──────────────────────────────────────
record "waiting ${WINDOW}s — see the header for how this number is derived"
sleep "$WINDOW"

# ── 4. GROUND TRUTH, from a source the subject cannot influence ─────────
#
# Run inside the dashboard's own container so DNS is the real container
# DNS, but performed by a socket probe rather than by the application.
# `python -c` because no image here installs curl or wget.
record ""
record "=== GROUND TRUTH: transport boundary, probed from inside the container ==="
for dep in $GATED; do
  stage "groundtruth_${dep}" docker compose -f "$COMPOSE" exec -T dashboard \
    python -c "
import socket, sys
host = '$dep'
try:
    addrs = socket.getaddrinfo(host, None)
except Exception as exc:
    print('DNS_FAILURE', type(exc).__name__, exc); sys.exit(0)
print('DNS_RESOLVED', addrs[0][4][0])
s = socket.socket(); s.settimeout(5)
try:
    s.connect((host, 80))
    print('CONNECTED — the dependency is NOT absent')
except socket.timeout:
    print('TIMEOUT')
except ConnectionRefusedError:
    print('REFUSED')
except Exception as exc:
    print('OTHER', type(exc).__name__, exc)
finally:
    s.close()
"
  record "ground_truth_${dep}=$(printf '%s' "$STAGE_OUT" | tr '\n' ' ')"
done

# ── 5. CALLER HEALTH, from the daemon rather than the application ───────
record ""
record "=== CALLER HEALTH: the daemon's opinion, not the app's ==="
for svc in dashboard agentic memu-core; do
  cid="$(docker compose -f "$COMPOSE" ps -q "$svc" 2>/dev/null)"
  if [ -z "$cid" ]; then
    record "health_${svc}=NO CONTAINER — cannot be asked"
    continue
  fi
  stage "health_${svc}" docker inspect \
    --format '{{.State.Status}} health={{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} restarts={{.RestartCount}}' \
    "$cid"
  # THE OBSERVED TRANSITIONS ARE AUTHORITATIVE FOR THIS RUN, not the
  # window arithmetic in the header. That arithmetic says only that 150s
  # conservatively EXCEEDS the configured health/retry/breaker envelope;
  # it is not a claim that Docker cannot reach a verdict sooner. What
  # Docker actually did is recorded here and wins.
  stage "health_log_${svc}" docker inspect \
    --format '{{if .State.Health}}{{range .State.Health.Log}}{{.Start}} exit={{.ExitCode}} {{end}}{{else}}no healthcheck{{end}}' \
    "$cid"
done

# ── 6. THE CALLER'S OWN BEHAVIOUR, in the deployed topology ─────────────
#
# The dashboard's real proxy helper, executed inside the dashboard
# container against a genuinely absent dependency, so container DNS and
# the deployed network are in the path. This is the strongest EXISTING
# observable surface: no new endpoint is created to make the measurement
# possible, which would change the thing being measured.
record ""
record "=== CALLER BEHAVIOUR: the deployed proxy helper against an absent dep ==="
for dep in $GATED; do
  stage "caller_${dep}" docker compose -f "$COMPOSE" exec -T dashboard \
    python -c "
import asyncio, json, sys, time
sys.path.insert(0, '/app')
from app import _proxy_get
started = time.monotonic()
try:
    result = asyncio.run(_proxy_get('http://$dep:8080/status', fallback={'entries': [], 'count': 0}))
    print(json.dumps({'elapsed': round(time.monotonic()-started, 2), 'result': result}))
except Exception as exc:
    print(json.dumps({'elapsed': round(time.monotonic()-started, 2),
                      'raised': type(exc).__name__, 'detail': str(exc)[:200]}))
"
  record "caller_behaviour_${dep}=$(printf '%s' "$STAGE_OUT" | tr '\n' ' ')"
done

# ── 7. corroboration, explicitly labelled as corroboration ──────────────
record ""
record "=== CORROBORATION ONLY — the subject's own report ==="
record "The dashboard's status payload is recorded LAST and is not"
record "permitted to establish anything. If it disagrees with the ground"
record "truth above, the ground truth wins and the disagreement is the"
record "finding."
stage dashboard_self_report docker compose -f "$COMPOSE" exec -T dashboard \
  python -c "
import json, urllib.request
try:
    with urllib.request.urlopen('http://localhost:8000/health', timeout=5) as r:
        print(json.dumps({'health': json.loads(r.read().decode())}))
except Exception as exc:
    print(json.dumps({'error': type(exc).__name__, 'detail': str(exc)[:200]}))
"

record ""
record "=== LOGS: is anything storming? ==="
stage dashboard_log_volume docker compose -f "$COMPOSE" logs --no-color --tail 4000 dashboard
record "dashboard_log_lines=$(printf '%s\n' "$STAGE_OUT" | wc -l)"

docker compose -f "$COMPOSE" down -v >/dev/null 2>&1 || true

record ""
record "=== WHAT THIS DOES AND DOES NOT ESTABLISH ==="
record "Established: real container DNS and transport boundary; the"
record "  daemon's health verdict; the deployed caller's behaviour and"
record "  whether it substitutes; log volume."
record "NOT established: what the dashboard UI renders to a human, and"
record "  whether any downstream automated consumer treats a false-empty"
record "  result as fact. Those remain UNKNOWN and must not be inferred"
record "  from anything above."
exit 0
