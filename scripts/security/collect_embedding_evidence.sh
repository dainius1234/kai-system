#!/usr/bin/env bash
#
# Collect Claim-A evidence for ONE service, with every stage observable.
#
# Usage:  collect_embedding_evidence.sh <compose-file> <service> <probe-path>
#
# TWO INDEPENDENT AXES, and conflating them is the defect this replaces:
#
#   measurement = COMPLETE | INCOMPLETE | INSTRUMENT_ERROR
#                 did the collector perform and record the measurement?
#   claim       = REAL | FAKE | WRONG_DIMENSION | NO_OBSERVATION |
#                 TIMEOUT_UNKNOWN | UNKNOWN
#                 what did the capability turn out to be?
#
# A probe that proves the semantic backend absent is a SUCCESSFUL
# MEASUREMENT of a FAILED capability. So this exits 0 for every defined
# probe verdict, and non-zero ONLY when the instrument itself
# malfunctioned. Run #2 failed a collector step because a claim was
# incomplete, and GitHub then skipped Claim B entirely — one measurement
# suppressing an independent one.
#
#   A measurement verdict may affect its own claim.
#   It may never suppress an independent measurement.
#
# WHY EVERY STAGE IS RECORDED
#
# The previous resolver sent stdout, stderr and the exit status of every
# attempt to /dev/null and returned one string. When it came back empty,
# the whole chain collapsed to "UNRESOLVED" and the failing stage was
# unknowable. Run #2 could not say whether agentic failed at container
# creation, at the id lookup or at inspect — evidence destroyed by the
# instrument built to gather it.
#
# So each stage records its command, stdout, stderr and exit status, and
# an empty result becomes a classified stage outcome rather than silence.
#
set -uo pipefail

COMPOSE_FILE="${1:?compose file required}"
SERVICE="${2:?service required}"
PROBE="${3:?probe path required}"

EVIDENCE="claim-a-${SERVICE}.evidence"
: > "$EVIDENCE"

measurement="INCOMPLETE"
claim="UNKNOWN"
producer="resolver"
probe_exit=""
image=""

record() { printf '%s\n' "$*" >> "$EVIDENCE"; }

# stage <name> <command...> — runs it, records everything, returns its status
stage() {
  local name="$1"; shift
  local out err rc
  out="$(mktemp)"; err="$(mktemp)"
  if "$@" > "$out" 2> "$err"; then rc=0; else rc=$?; fi
  record "stage=$name"
  record "  cmd=$*"
  record "  exit=$rc"
  record "  stdout=$( [ -s "$out" ] && tr '\n' '|' < "$out" || echo '(empty)' )"
  record "  stderr=$( [ -s "$err" ] && tr '\n' '|' < "$err" | cut -c1-400 || echo '(empty)' )"
  STAGE_OUT="$(cat "$out")"
  rm -f "$out" "$err"
  return $rc
}

record "service=$SERVICE"
record "compose_file=$COMPOSE_FILE"

# ── 1. does Compose know this service at all? ───────────────────────────
stage config_services docker compose -f "$COMPOSE_FILE" config --services
services="$STAGE_OUT"
if printf '%s\n' "$services" | grep -qx "$SERVICE"; then
  record "service_known=yes"
else
  record "service_known=no"
fi

# ── 2. profiles, measured rather than assumed ───────────────────────────
#
# fusion-engine built successfully yet `config --images` exposed no image
# name for it. Profiles are ONE hypothesis; this records what Compose
# actually reports so the cause is measured, not named in advance.
stage config_profiles docker compose -f "$COMPOSE_FILE" config --profiles
record "profiles_declared=$(printf '%s' "$STAGE_OUT" | tr '\n' ',')"

stage config_services_all_profiles env COMPOSE_PROFILES='*' \
  docker compose -f "$COMPOSE_FILE" config --services
if printf '%s\n' "$STAGE_OUT" | grep -qx "$SERVICE"; then
  record "service_known_with_all_profiles=yes"
else
  record "service_known_with_all_profiles=no"
fi

# ── 3. the image NAME compose declares for it ───────────────────────────
#
# `config --images <service>` RETURNS THE SERVICE'S WHOLE DEPENDENCY
# GRAPH, in an order that is not the service's own. Run 3 measured it:
#
#   config --images memu-core      -> redis:7-alpine | kai-system-memu-core | pgvector…
#   config --images agentic        -> ollama/ollama:0.6.8 | kai-system-agentic | …
#   config --images fusion-engine  -> kai-system-memu-core | … | kai-system-fusion-engine
#
# Taking `head -1` therefore named a DEPENDENCY'S image as the service's
# own. For memu-core and agentic the first entry was not built locally,
# `docker image inspect` failed, and the row honestly read UNKNOWN. For
# fusion-engine the first entry WAS built — so the probe ran against
# **memu-core's image** and the collector recorded `fusion-engine
# claim-A: REAL`. A confident verdict about the wrong artefact, which is
# worse than the UNKNOWN it replaced, and it survived every
# verdict-integrity control because those protect a verdict's TRANSPORT,
# not its SUBJECT.
#
# So the name is read from the service's own resolved definition, which
# is single-valued and cannot include a neighbour.
stage config_json env COMPOSE_PROFILES='*' \
  docker compose -f "$COMPOSE_FILE" config --format json
config_json="$STAGE_OUT"
image_name="$(printf '%s' "$config_json" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(0)
svc = (doc.get("services") or {}).get(sys.argv[1]) or {}
print(svc.get("image", ""))
' "$SERVICE" 2>/dev/null)"
record "image_name=${image_name:-(none)}"

# CORROBORATION, and a guard against the defect above. The dependency
# listing is recorded as a second, independent channel: the name chosen
# above must appear in it, and it must not be a name the listing gives to
# some OTHER service. A resolver that cannot show its answer belongs to
# the service it was asked about has not resolved anything.
stage config_image_list env COMPOSE_PROFILES='*' \
  docker compose -f "$COMPOSE_FILE" config --images "$SERVICE"
if [ -n "$image_name" ] && printf '%s\n' "$STAGE_OUT" | grep -qxF "$image_name"; then
  record "image_name_corroborated=yes"
else
  record "image_name_corroborated=no"
fi

# THE BINDING CHECK. Does this image name belong to a DIFFERENT service?
# Known-positive from run 3: fusion-engine resolved to kai-system-memu-core,
# which is memu-core's image, and nothing said so.
other="$(printf '%s' "$config_json" | python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(0)
me, name = sys.argv[1], sys.argv[2]
if not name:
    sys.exit(0)
for other, spec in (doc.get("services") or {}).items():
    if other != me and (spec or {}).get("image") == name:
        print(other)
        break
' "$SERVICE" "$image_name" 2>/dev/null)"
record "image_name_also_used_by=${other:-(none)}"
binding_failure=""
if [ -n "$other" ]; then
  binding_failure="'$image_name' is also $other's image"
  record "BINDING FAILURE: $binding_failure"
fi

# ── 4. create the container Compose would run ───────────────────────────
#
# NOT `--no-deps`: `docker compose create` has no such flag and run 3
# recorded `unknown flag: --no-deps` for all three services, which killed
# the container-scoped resolution path that had worked in run 2 and
# forced every row onto the name-based fallback above.
stage compose_create env COMPOSE_PROFILES='*' \
  docker compose -f "$COMPOSE_FILE" create "$SERVICE"

# ── 5. its container id ─────────────────────────────────────────────────
stage container_id env COMPOSE_PROFILES='*' \
  docker compose -f "$COMPOSE_FILE" ps -aq "$SERVICE"
cid="$(printf '%s' "$STAGE_OUT" | head -1)"
record "container_id=${cid:-(none)}"

# ── 6. the immutable image id, from the container if we have one ────────
if [ -n "$cid" ]; then
  stage inspect_container_image docker inspect --format '{{.Image}}' "$cid"
  image="$(printf '%s' "$STAGE_OUT" | head -1)"
fi
record "image_source=$( [ -n "$image" ] && echo container || echo '(container path failed)' )"

# BUILD-SCOPED FALLBACK, and the ONLY path the binding check governs.
# The container path resolves from the container Compose created, which
# cannot be another service's; the name path can be, and was. So the
# check applies exactly where the risk is, and a mis-bound name is an
# INSTRUMENT malfunction rather than a claim about the service.
if [ -z "$image" ] && [ -n "$image_name" ]; then
  if [ -n "$binding_failure" ]; then
    record "probe_refused=the fallback name is not uniquely this service's: $binding_failure"
    measurement="INSTRUMENT_ERROR"
  else
    # Still an IMMUTABLE id -- the name is only used to look it up,
    # never recorded as the evidence itself.
    stage inspect_image_by_name docker image inspect -f '{{.Id}}' "$image_name"
    image="$(printf '%s' "$STAGE_OUT" | head -1)"
    [ -n "$image" ] && record "image_source=config-name"
  fi
fi
record "image_id=${image:-(unresolved)}"

# ── 7. the probe ────────────────────────────────────────────────────────
if [ -z "$image" ]; then
  record "probe_ran=no"
  record "reason=no immutable image id could be resolved; the failing stage is above"
  # An INSTRUMENT_ERROR already set upstream must SURVIVE. "The resolver
  # would have probed another service's image" and "nothing resolved" are
  # different facts, and overwriting the first with the second is the
  # same silent substitution R-VERDICT-INTEGRITY forbids one layer down.
  [ "$measurement" = "INSTRUMENT_ERROR" ] || measurement="INCOMPLETE"
  claim="UNKNOWN"
else
  producer=probe
  if timeout 600 docker run --rm --network none \
      -v "$PROBE:/probe.py:ro" -e MEMU_ALLOW_FAKE_EMBEDDINGS= \
      "$image" python /probe.py "$SERVICE" > "claim-a-${SERVICE}.log" 2>&1
  then probe_exit=0; else probe_exit=$?; fi
  record "probe_ran=yes"
  record "probe_exit=$probe_exit"
  [ "$probe_exit" -eq 124 ] && producer=timeout-wrapper

  if [ "$producer" != "probe" ]; then
    measurement="INCOMPLETE"; claim="TIMEOUT_UNKNOWN"
  else
    measurement="COMPLETE"
    case "$probe_exit" in
      0) claim=REAL ;;
      3) claim=FAKE ;;
      4) claim=WRONG_DIMENSION ;;
      5) claim=NO_OBSERVATION ;;
      *) measurement="INSTRUMENT_ERROR"
         claim="UNKNOWN" ;;   # an undefined status is not a verdict
    esac
  fi
fi

record "measurement=$measurement"
record "producer=$producer"
record "claim_verdict=$claim"

printf '%s claim-A: measurement=%s claim_verdict=%s producer=%s probe_exit=%s image=%s\n' \
  "$SERVICE" "$measurement" "$claim" "$producer" "${probe_exit:-n/a}" \
  "${image:-unresolved}" | tee -a claim-a-summary.txt

echo "::group::$SERVICE — stage-by-stage evidence"
cat "$EVIDENCE"
[ -f "claim-a-${SERVICE}.log" ] && { echo "--- probe output ---"; cat "claim-a-${SERVICE}.log"; }
echo "::endgroup::"

# A NEGATIVE CLAIM IS NOT A COLLECTOR FAILURE. Only a malfunctioning
# instrument fails this step, so an independent measurement is never
# suppressed by this one's result.
[ "$measurement" = "INSTRUMENT_ERROR" ] && exit 2
exit 0
