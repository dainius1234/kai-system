#!/usr/bin/env bash
#
# Answer the one remaining UNKNOWN: is Ed25519 usable in the real Kai
# service images, and does the /observe_turn deployment path work end to
# end between two containers?
#
# Everything proven so far was proven on a developer host. That is not
# the same place. This script exists because "works in this Python
# environment" must never be allowed to read as "works in the Kai
# runtime" — the distance between those two claims is where this
# programme keeps finding defects.
#
# REQUIRES A DOCKER DAEMON. It has never been run: the environment where
# it was written has the CLI and no daemon. Until it passes somewhere,
# the status stays:
#
#     Ed25519 real-image feasibility = UNKNOWN
#
# Usage:  bash scripts/security/verify_identity_in_containers.sh
#
set -uo pipefail

COMPOSE_FILE="docker-compose.minimal.yml"
PROFILE="introspection"
KEYS="secrets/service-identity"
PASS=0
FAIL=0

check() {                       # check <label> <expected> <actual>
  if [ "$2" = "$3" ]; then
    PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"
  else
    FAIL=$((FAIL + 1)); printf '  FAIL  %s (expected %s, got %s)\n' "$1" "$2" "$3"
  fi
}

echo "Ed25519 in real service images — container proof"
echo "=================================================================="

if ! docker info >/dev/null 2>&1; then
  echo "  REFUSING TO REPORT: no Docker daemon reachable."
  echo "  This script proves nothing without one, and a skipped proof"
  echo "  reads exactly like a passed one. Status remains UNKNOWN."
  exit 2
fi

# ── 0. key material must exist before anything is built ──
if [ ! -f "$KEYS/keymap.json" ]; then
  echo "  key material missing — generating it first"
  python3 scripts/security/generate_service_keys.py \
      --service agentic --service cortex \
      --grant cortex_observe_turn=agentic || exit 1
fi

echo
echo "── 1. the images build with cryptography==43.0.1 ──"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" build cortex agentic
check "cortex and agentic images build" 0 $?

echo
echo "── 2. cryptography imports INSIDE the image, and signs ──"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" run --rm --no-deps \
  -T cortex python -c "
import sys
sys.path.insert(0, '/app')
from common.service_identity import import_status, generate_keypair, sign, verify, KeyEntry, ALG_ED25519
ok, detail = import_status()
assert ok, detail
print('   backend:', detail)
priv, pub = generate_keypair()
sig = sign(b'container proof', ALG_ED25519, priv)
entry = KeyEntry(key_id='k', identity='i', algorithm=ALG_ED25519, public_key=pub)
assert verify(b'container proof', sig, entry), 'verify failed in container'
assert not verify(b'tampered', sig, entry), 'tampering NOT detected in container'
print('   sign/verify/tamper-reject all OK in-image')
"
check "ed25519 works inside the real image" 0 $?

echo
echo "── 3. mounted key material has the right permissions ──"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" run --rm --no-deps -T agentic \
  python -c "
import os, stat, sys
path = os.environ['KAI_SERVICE_PRIVATE_KEY']
mode = stat.S_IMODE(os.stat(path).st_mode)
assert not (mode & 0o077), f'private key is group/other readable: {mode:o}'
open(path).read()
print(f'   private key {path} mode {mode:o}')
"
check "agentic's private key is mounted and not world-readable" 0 $?

docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" run --rm --no-deps -T cortex \
  python -c "
import os, sys
sys.path.insert(0, '/app')
from common.service_identity import KeyMap
km = KeyMap.load()
print('   key map:', len(km), 'key(s), identities', km.identities())
assert 'agentic' in km.identities()
assert km.granted('cortex_observe_turn', 'agentic')
assert not km.granted('cortex_observe_turn', 'cortex')
try:
    open(os.environ['KAI_SERVICE_KEYMAP'], 'a'); raise SystemExit('key map is WRITABLE')
except PermissionError:
    print('   key map is read-only, as mounted')
"
check "cortex's key map loads, grants agentic, and is read-only" 0 $?

echo
echo "── 4. the service starts and serves ──"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" up -d cortex
for _ in $(seq 1 30); do
  state=$(docker inspect -f '{{.State.Health.Status}}' kai-cortex 2>/dev/null || echo starting)
  [ "$state" = "healthy" ] && break
  sleep 2
done
check "cortex reaches healthy with identity wiring present" "healthy" "$state"

echo
echo "── 5. the governed endpoint, from a real caller container ──"
# Signed by agentic's key, through the same helper the service uses.
run_case() {                    # run_case <label> <python-snippet> <expected>
  # Prints "<status> <turn_source>" so a case can assert WHO the
  # receiver decided the caller was, not merely that it answered.
  # stderr is kept: a crash inside the container would otherwise arrive
  # as an empty string and read like a wrong status code.
  raw_out=$(docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" run --rm -T agentic \
    python -c "
import sys, json, urllib.request, urllib.error
sys.path.insert(0, '/app')
from common.service_identity import signed_json_request, encode_json_body
$2
req = urllib.request.Request('http://cortex:8048/observe_turn', data=raw,
                             headers=headers, method='POST')
try:
    with urllib.request.urlopen(req, timeout=5) as r:
        body = json.loads(r.read() or b'{}')
        print(r.status, body.get('turn_source'))
except urllib.error.HTTPError as e:
    print(e.code)
" 2>&1)
  actual=$(printf '%s\n' "$raw_out" | tail -1)
  if [ "$actual" != "$3" ]; then
    printf '  FAIL  %s (expected %s, got %s)\n' "$1" "$3" "$actual"
    printf '        container output:\n%s\n' "$raw_out" | sed 's/^/        /'
    FAIL=$((FAIL + 1))
  else
    PASS=$((PASS + 1)); printf '  ok    %s\n' "$1"
  fi
}

PAYLOAD='{"session_id": "proof", "user_message": "container proof"}'

run_case "a correctly signed, granted request is ACCEPTED, as agentic" "
raw, headers = signed_json_request(destination='cortex', method='POST',
                                   path='/observe_turn', payload=$PAYLOAD)
" "200 agentic"

run_case "an unsigned request with only the shared token is REFUSED" "
raw = encode_json_body($PAYLOAD)
headers = {'content-type': 'application/json',
           'Authorization': 'Bearer ' + __import__('os').environ.get('KAI_SERVICE_TOKEN','')}
" 401

# The load-bearing one. A VALID signature carrying forged identity
# headers must still be identified by the KEY. Deleting the signature
# first would only prove an unsigned request fails, which is a weaker
# claim and is already covered above.
run_case "forged identity headers on a VALID signature change nothing" "
raw, headers = signed_json_request(destination='cortex', method='POST',
                                   path='/observe_turn', payload=$PAYLOAD)
headers['X-Kai-Identity'] = 'cortex'
headers['X-Actor-Did'] = 'cortex'
headers['X-Service-Name'] = 'cortex'
" "200 agentic"

run_case "and forged headers without a signature are still refused" "
raw = encode_json_body($PAYLOAD)
headers = {'content-type': 'application/json',
           'X-Kai-Identity': 'agentic', 'X-Actor-Did': 'agentic'}
" 401

run_case "an altered body is REFUSED" "
raw, headers = signed_json_request(destination='cortex', method='POST',
                                   path='/observe_turn', payload=$PAYLOAD)
raw = encode_json_body({'session_id': 'proof', 'user_message': 'SOMETHING ELSE'})
" 401

run_case "a signature bound to another path is REFUSED" "
raw, headers = signed_json_request(destination='cortex', method='POST',
                                   path='/state', payload=$PAYLOAD)
" 401

run_case "a signature bound to another method is REFUSED" "
raw, headers = signed_json_request(destination='cortex', method='DELETE',
                                   path='/observe_turn', payload=$PAYLOAD)
" 401

run_case "a signature bound to another service is REFUSED" "
raw, headers = signed_json_request(destination='executor', method='POST',
                                   path='/observe_turn', payload=$PAYLOAD)
" 401

echo
echo "── 6. replay, across two separate requests ──"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" run --rm -T agentic python -c "
import sys, urllib.request, urllib.error
sys.path.insert(0, '/app')
from common.service_identity import signed_json_request
raw, headers = signed_json_request(destination='cortex', method='POST',
                                   path='/observe_turn', payload=$PAYLOAD)
codes = []
for _ in range(2):
    req = urllib.request.Request('http://cortex:8048/observe_turn', data=raw,
                                 headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=5) as r: codes.append(r.status)
    except urllib.error.HTTPError as e: codes.append(e.code)
assert codes == [200, 401], f'expected [200, 401], got {codes}'
print('   first', codes[0], 'replay', codes[1])
"
check "the same signed request succeeds ONCE, then is refused as a replay" 0 $?

echo
echo "── 7. the replay cache survives restart ──"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" restart cortex
for _ in $(seq 1 30); do
  state=$(docker inspect -f '{{.State.Health.Status}}' kai-cortex 2>/dev/null || echo starting)
  [ "$state" = "healthy" ] && break
  sleep 2
done
check "cortex is healthy again after restart" "healthy" "$state"
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" run --rm -T agentic python -c "
import sys, urllib.request, urllib.error
sys.path.insert(0, '/app')
from common.service_identity import signed_json_request
raw, headers = signed_json_request(destination='cortex', method='POST',
                                   path='/observe_turn', payload=$PAYLOAD)
req = urllib.request.Request('http://cortex:8048/observe_turn', data=raw,
                             headers=headers, method='POST')
try:
    with urllib.request.urlopen(req, timeout=5) as r: code = r.status
except urllib.error.HTTPError as e: code = e.code
assert code == 200, f'a fresh signed request after restart got {code}'
print('   fresh request after restart:', code)
"
check "a FRESH signed request is still accepted after restart" 0 $?

echo
echo "=================================================================="
echo "Container proof: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
  echo "EXIT GATE: PASS — Ed25519 real-image feasibility is now PROVEN."
  echo "Update kai-pm/SERVICE_IDENTITY_MEASUREMENT.md 12 accordingly."
else
  echo "EXIT GATE: FAIL — status remains UNKNOWN. Do not upgrade the claim."
fi
docker compose -f "$COMPOSE_FILE" --profile "$PROFILE" stop cortex >/dev/null 2>&1
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
