#!/usr/bin/env bash
#
# KAI-GATE-048 implementation definition: what EXACTLY does memu-graph's
# model path need locally, and what offline switch does its stack honour?
#
# D189 authorised this as a short read-only definition unit, ahead of a
# remediation plan and before any code change. It answers five questions
# and nothing else:
#
#   1. what model(s) does the failing path request?
#   2. where does the stack expect them locally?
#   3. can the revision be pinned?
#   4. what offline switch/API does this stack actually honour?
#   5. one asset, or multiple transitive assets?
#
# WHY IT IS MEASURED IN A THROWAWAY CONTAINER
# ===========================================
#
# Every stage below runs `docker run --rm` from the ALREADY-BUILT
# memu-graph image. Nothing in the deployed topology is touched: no
# compose service is started, no profile is activated, no network
# declaration is altered. Stage A deliberately runs on Docker's default
# bridge — which has egress — because the question "what does this path
# fetch" cannot be answered by a container that cannot fetch. That is an
# INSTRUMENT with network access, not a relaxation of `agent-net`, and
# stages B and C put the network back to `none` to prove the point.
#
# THE THREE STAGES ARE A KNOWN-POSITIVE AND TWO KNOWN-NEGATIVES
# =============================================================
#
#   A  network available, empty cache   -> enumerate what gets fetched
#   B  --network none, cache from A     -> MUST SUCCEED. Proves the asset
#                                          set from A is SUFFICIENT and
#                                          that the offline switch is
#                                          honoured. Without this, A is a
#                                          list of files nobody proved
#                                          was complete.
#   C  --network none, EMPTY cache      -> MUST FAIL, and fail FAST. The
#                                          measured contrast with the
#                                          ~47s retry storm is the whole
#                                          argument for obligation 2.
#
# B is the load-bearing one. A on its own would be a plausible list; only
# B turns it into a proven contract.
#
set -uo pipefail

EVIDENCE="memu-graph-asset-contract.evidence"
LOGDIR="asset-stage-logs"
CACHE="$(pwd)/asset-probe-cache"
mkdir -p "$LOGDIR" "$CACHE"
chmod 777 "$CACHE"
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
  local log="$1" lines="${2:-30}"
  local bytes; bytes=$(wc -c < "$log" | tr -d ' ')
  record "  --- EXCERPT: last ${lines} lines of ${log} (${bytes} bytes total) ---"
  tail -n "$lines" "$log" | sed 's/^/  | /' | tee -a "$EVIDENCE"
  record "  --- end excerpt ---"
}

record "KAI-GATE-048 — memu-graph model ASSET CONTRACT definition"
record "started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record ""

# The image is the one compose builds. Resolved from compose rather than
# from a guessed tag, and the run aborts if it is not there -- an
# unresolved image would make every stage below measure nothing.
IMAGE=$(docker compose -f docker-compose.full.yml images -q memu-graph 2>/dev/null | head -1)
if [ -z "$IMAGE" ]; then
  IMAGE=$(docker images --format '{{.Repository}}:{{.Tag}}' \
          | grep -E 'memu-graph' | head -1)
fi
record "image under test: ${IMAGE:-<unresolved>}"
if [ -z "$IMAGE" ]; then
  record ""
  record "ABORTED: the memu-graph image could not be resolved."
  record "  Nothing below was measured. This is a prerequisite failure,"
  record "  not a finding about the asset contract."
  record "VERDICT: UNKNOWN (image unresolved)"
  exit 2
fi

# The tokenizer name is read from the compose file, not written here --
# a second copy of that value beside the thing is the defect this
# repository keeps finding.
TOKENIZER=$(python3 -c "
import yaml
d = yaml.safe_load(open('docker-compose.full.yml'))
print((d['services']['memu-graph'].get('environment') or {}).get('HUGGINGFACE_TOKENIZER', ''))
")
record "HUGGINGFACE_TOKENIZER (from docker-compose.full.yml): ${TOKENIZER:-<unset>}"
record ""

# The exact call cognee makes, quoted from its source so the probe cannot
# drift from the thing it is describing:
#   cognee/infrastructure/llm/tokenizer/HuggingFace/adapter.py:32
#       self.tokenizer = AutoTokenizer.from_pretrained(model)
#   reached from OllamaEmbeddingEngine.__init__ -> get_tokenizer()
PROBE='
import os, sys, time, json
os.environ.setdefault("HF_HOME", "/data/hf_cache")
import huggingface_hub
from huggingface_hub import constants as C
print("huggingface_hub", huggingface_hub.__version__)
import transformers
print("transformers", transformers.__version__)
print("HF_HOME              ", os.environ.get("HF_HOME"))
print("HF_HUB_CACHE         ", C.HF_HUB_CACHE)
print("HF_HUB_OFFLINE env   ", os.environ.get("HF_HUB_OFFLINE"))
print("TRANSFORMERS_OFFLINE ", os.environ.get("TRANSFORMERS_OFFLINE"))
print("huggingface_hub.constants.HF_HUB_OFFLINE ", C.HF_HUB_OFFLINE)
name = os.environ["HUGGINGFACE_TOKENIZER"]
print("model requested      ", name)
t0 = time.monotonic()
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    print("RESULT: LOADED in %.2fs  class=%s" % (time.monotonic() - t0, type(tok).__name__))
except Exception as exc:
    print("RESULT: FAILED after %.2fs  %s: %s"
          % (time.monotonic() - t0, type(exc).__name__, str(exc)[:600]))
root = os.environ.get("HF_HOME", "/data/hf_cache")
total = 0
files = []
for dirpath, _dirs, names in os.walk(root):
    for n in names:
        p = os.path.join(dirpath, n)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        total += sz
        files.append((os.path.relpath(p, root), sz))
print("--- cache tree under %s ---" % root)
for rel, sz in sorted(files):
    print("  %10d  %s" % (sz, rel))
print("--- file count ---")
print(len(files))
print("--- total bytes ---")
print(total)
'

# ── A. what does the path actually fetch? ────────────────────────────
record "== A. WITH NETWORK, EMPTY CACHE — what does the path request? =="
record "  (a throwaway container on the default bridge; the deployed"
record "   service stays on agent-net and is not started at all)"
stage "A-fetch" docker run --rm \
    -e "HUGGINGFACE_TOKENIZER=$TOKENIZER" \
    -e "HF_HOME=/probe-cache" \
    -v "$CACHE:/probe-cache" \
    --entrypoint python "$IMAGE" -c "$PROBE"
a_rc=$?
excerpt "$LOGDIR/A-fetch.log" 45
record ""

if [ $a_rc -ne 0 ]; then
  record "A FAILED — stages B and C are DEPENDENT on it and were NOT RUN."
  record "  No subject, no observation: without a fetched asset set there"
  record "  is nothing to prove sufficient and nothing to withhold."
  record "VERDICT: UNKNOWN (fetch stage failed)"
  exit 2
fi

# ── B. is that asset set SUFFICIENT, with no network at all? ─────────
record "== B. --network none, CACHE FROM A — the asset set must SUFFICE =="
record "  known-positive. If this fails, A's list is incomplete and any"
record "  bake built from it would ship a still-broken image."
stage "B-offline-with-asset" docker run --rm --network none \
    -e "HUGGINGFACE_TOKENIZER=$TOKENIZER" \
    -e "HF_HOME=/probe-cache" \
    -e "HF_HUB_OFFLINE=1" -e "TRANSFORMERS_OFFLINE=1" \
    -v "$CACHE:/probe-cache:ro" \
    --entrypoint python "$IMAGE" -c "$PROBE"
excerpt "$LOGDIR/B-offline-with-asset.log" 30
record ""

# ── C. and does it FAIL FAST without the asset? ──────────────────────
record "== C. --network none, EMPTY CACHE — must fail, and fail FAST =="
record "  known-negative, and the measurement behind obligation 2: the"
record "  deployed run spent ~47s retrying an unreachable host before"
record "  returning the same answer this stage should give immediately."
stage "C-offline-no-asset" docker run --rm --network none \
    -e "HUGGINGFACE_TOKENIZER=$TOKENIZER" \
    -e "HF_HOME=/empty-cache" \
    -e "HF_HUB_OFFLINE=1" -e "TRANSFORMERS_OFFLINE=1" \
    --entrypoint python "$IMAGE" -c "$PROBE"
excerpt "$LOGDIR/C-offline-no-asset.log" 30
record ""

# ── D. and WITHOUT the offline switch, no network — the status quo ───
record "== D. --network none, EMPTY CACHE, NO offline switch — today =="
record "  the control: this is what the deployed service does now."
stage "D-noflag-no-asset" timeout 180 docker run --rm --network none \
    -e "HUGGINGFACE_TOKENIZER=$TOKENIZER" \
    -e "HF_HOME=/empty-cache" \
    --entrypoint python "$IMAGE" -c "$PROBE"
excerpt "$LOGDIR/D-noflag-no-asset.log" 25
record ""

record "== SUMMARY =="
python3 scripts/security/summarise_asset_contract.py --stage-logs "$LOGDIR" \
    2>&1 | tee -a "$EVIDENCE"

record ""
record "finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
record "full evidence: $EVIDENCE   stage logs: $LOGDIR/"
exit 0
