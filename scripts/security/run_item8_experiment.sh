#!/usr/bin/env bash
# Item 8's six frozen builds. The branch criteria live HERE, in one
# reviewable place, rather than spread across YAML.
#
# Frozen design: D285's canonical region, 0055ead8...8796, verified by
# check_item8_design.py before this script is reached.
#
#   B1  genuine fetch                       --no-cache, 0 mutations
#   B2  injected first-attempt failure      1 mutation
#   B3  persistent denial on the HF RUN     1 mutation
#
# ── THREE FIELDS, NOT ONE ────────────────────────────────────────────
#
# Frozen R2: *"A collector fault leaves Axis 1's result standing and
# leaves item 10's provenance unmoved; a clean binding cannot turn a
# failed contingency into a success."*
#
# The first implementation of this script had ONE `verdict` field, and a
# failed `.Image` binding rewrote it to UNMEASURED -- so an image-
# provenance fault silently became a contingency measurement. Caught in
# adversarial review before any build. Every row now carries:
#
#   axis1_verdict          the contingency: PASS / WRONG_FAILURE / UNMEASURED
#   axis2_provenance       the image identity: RECORDED / MISMATCH /
#                          UNRECORDED / IMAGE_NOT_PRODUCED_BY_DESIGN /
#                          NOT_APPLICABLE_BY_DESIGN
#   qualified_for_closure  true only when BOTH are sound
#
# Axis 2 may block closure. It may never rewrite Axis 1.
#
# ── EXIT STATUS IS ABOUT THE INSTRUMENT, NOT THE SUBJECT ─────────────
#
# Doctrine rule 5. A branch that is WRONG_FAILURE or UNMEASURED is a
# RESULT and is banked as one, and the script still exits 0. It exits
# NON-ZERO only when it genuinely cannot measure -- a missing instrument,
# an unwritable results file. The first implementation said this and then
# ended `exit 0` unconditionally, which made the sentence decorative.
#
# NO RE-DRAWS. Six builds, every outcome recorded, nothing repeated
# "until it works" -- D247 §5, and an external prerequisite is not an
# exemption from it (D289).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO" || { echo "INSTRUMENT FAILURE: cannot enter $REPO"; exit 2; }

DOCKER="${DOCKER:-docker}"
DERIVED="${ITEM8_DERIVED:-item8-derived}"
IDENT="${ITEM8_IDENT:-item8-identity}"
RESULTS="${ITEM8_RESULTS:-item8-results.jsonl}"

EXPLICIT="scripts/security/collect_explicit_image_identity.py"
BINDER="scripts/security/collect_image_identity.py"

# INSTRUMENT PREREQUISITES. Missing here means we cannot measure at all,
# which is a different thing from a branch having no subject.
for f in "$EXPLICIT" "$BINDER"; do
  [ -f "$f" ] || { echo "INSTRUMENT FAILURE: $f is missing"; exit 2; }
done
mkdir -p "$DERIVED" "$IDENT" || { echo "INSTRUMENT FAILURE: cannot create output dirs"; exit 2; }
: > "$RESULTS" || { echo "INSTRUMENT FAILURE: cannot write $RESULTS"; exit 2; }

RUN_ID="${GITHUB_RUN_ID:-local}"
TREE_SHA="$(git rev-parse 'HEAD^{tree}' 2>/dev/null || echo UNKNOWN)"

# An EMPTY payload must never reach the results file. A blank line is
# counted by `wc -l` and skipped by the summariser, so the two disagree
# about the denominator -- which is how a lost row hides.
emit() {
  [ -n "$1" ] || { echo "INSTRUMENT FAILURE: empty result row"; exit 2; }
  echo "$1" >> "$RESULTS" || { echo "INSTRUMENT FAILURE: lost a result row"; exit 2; }
}

# GENUINE retry lines, derived from what the SHIPPED Dockerfiles print:
#   memu-core   "model download attempt /5 failed; retrying in ...s"
#   memu-graph  "tokenizer bake attempt failed; retrying in 10s"
# Both contain "retrying in". The B2 injection marker deliberately does
# NOT -- the first implementation counted `attempt.*failed`, which its
# own injected line satisfied, so B2's retry detector was measuring the
# injection instead of the recovery. The detector must be independent of
# the treatment.
RETRY_MARK='retrying in'
INJECT_MARK='ITEM8-B2: first attempt failed by injection'

for IMAGE in memu-core memu-graph; do
  for BRANCH in B1 B2 B3; do
    LOWER="$(echo "$BRANCH" | tr 'A-Z' 'a-z')"
    TAG="kai-item8:${LOWER}-${IMAGE}"
    DF="${DERIVED}/Dockerfile.${IMAGE}.${BRANCH}"
    IID="${DERIVED}/${IMAGE}.${BRANCH}.iid"
    LOG="${DERIVED}/${IMAGE}.${BRANCH}.build.log"
    LABEL="item8-${LOWER}-${IMAGE}"

    echo "::group::${IMAGE} ${BRANCH}"

    A1="UNMEASURED"; A2="UNRECORDED"; QUAL="false"; NOTE=""
    RETRIES=0; ELAPSED=0; BUILD_RC=-1; PRE_CLEAN="n/a"; IIDCORR="n/a"

    # ── the derived file must already exist (R11) ─────────────────────
    if [ ! -s "$DF" ]; then
      emit "$(BR="$BRANCH" IM="$IMAGE" RI="$RUN_ID" TS="$TREE_SHA" python3 -c '
import json,os
print(json.dumps({"image":os.environ["IM"],"branch":os.environ["BR"],
 "axis1_verdict":"UNMEASURED","axis2_provenance":"UNRECORDED",
 "qualified_for_closure":False,"run_id":os.environ["RI"],
 "tree_sha":os.environ["TS"],
 "note":"derivation refused; no experimental Dockerfile exists, so the "
        "branch has no subject"}))')"
      echo "::endgroup::"; continue
    fi
    DF_SHA="$(python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$DF")"

    # ── B3 pre-assertion: tag and iidfile must NOT already exist ──────
    if [ "$BRANCH" = "B3" ]; then
      PRE_CLEAN="clean"
      "$DOCKER" image inspect "$TAG" >/dev/null 2>&1 && PRE_CLEAN="tag-already-exists"
      [ -e "$IID" ] && PRE_CLEAN="${PRE_CLEAN},iidfile-already-exists"
      if [ "$PRE_CLEAN" != "clean" ]; then
        emit "$(BR="$BRANCH" IM="$IMAGE" RI="$RUN_ID" TS="$TREE_SHA" DS="$DF_SHA" PC="$PRE_CLEAN" python3 -c '
import json,os
print(json.dumps({"image":os.environ["IM"],"branch":os.environ["BR"],
 "axis1_verdict":"UNMEASURED","axis2_provenance":"UNRECORDED",
 "qualified_for_closure":False,"run_id":os.environ["RI"],
 "tree_sha":os.environ["TS"],"dockerfile_sha256":os.environ["DS"],
 "pre_build_state":os.environ["PC"],
 "note":"pre-build state not clean; a stale tag or leftover iidfile would "
        "contaminate IMAGE_NOT_PRODUCED_BY_DESIGN"}))')"
        echo "::endgroup::"; continue
      fi
    fi

    # ── build ─────────────────────────────────────────────────────────
    NOCACHE=""; [ "$BRANCH" = "B1" ] && NOCACHE="--no-cache"
    START="$(date +%s)"
    # shellcheck disable=SC2086
    DOCKER_BUILDKIT=1 "$DOCKER" build $NOCACHE \
      -f "$DF" -t "$TAG" --iidfile "$IID" . > "$LOG" 2>&1
    BUILD_RC=$?
    ELAPSED=$(( $(date +%s) - START ))

    # `grep -c` PRINTS the count and EXITS 1 when the count is zero, so
    # `|| echo 0` appended a second zero and produced "0\n0" -- which
    # crashed the row builder and emitted a blank line that `wc -l` then
    # counted as a result. Found by this script's own calibration.
    RETRIES="$(grep -c "$RETRY_MARK" "$LOG" 2>/dev/null || true)"
    RETRIES="${RETRIES:-0}"
    INJECTED="no"; grep -q "$INJECT_MARK" "$LOG" 2>/dev/null && INJECTED="yes"
    TARGET_REFUSAL="no"; grep -q 'REFUSING TO BUILD' "$LOG" 2>/dev/null && TARGET_REFUSAL="yes"

    # ── AXIS 1: the contingency, decided WITHOUT any identity input ───
    if [ "$BRANCH" = "B3" ]; then
      A2="NOT_APPLICABLE_BY_DESIGN"
      POST_TAG="absent"; POST_IID="absent"
      "$DOCKER" image inspect "$TAG" >/dev/null 2>&1 && POST_TAG="present"
      [ -s "$IID" ] && POST_IID="present"
      if [ "$BUILD_RC" -eq 0 ]; then
        A1="UNMEASURED"; NOTE="the build SUCCEEDED under network denial; the intended refusal did not occur"
      elif [ "$TARGET_REFUSAL" != "yes" ]; then
        A1="WRONG_FAILURE"; NOTE="the build failed without the target step's refusal marker; the failure was elsewhere"
      elif [ "$RETRIES" -ne 5 ]; then
        # Frozen R2 requires FIVE attempts observed. The first
        # implementation asserted "five attempts" in its note while never
        # checking the count -- a claim in the place of a measurement.
        A1="UNMEASURED"; NOTE="the refusal occurred but ${RETRIES} genuine retry line(s) were observed, not the five the design requires"
      elif [ "$POST_TAG" != "absent" ] || [ "$POST_IID" != "absent" ]; then
        A1="UNMEASURED"; NOTE="post-build non-existence not established (tag=${POST_TAG}, iidfile=${POST_IID})"
      else
        A1="PASS"; A2="IMAGE_NOT_PRODUCED_BY_DESIGN"
        NOTE="${RETRIES} genuine retries, target-step refusal, no image at either end"
      fi
    elif [ "$BUILD_RC" -ne 0 ]; then
      if [ "$TARGET_REFUSAL" = "yes" ]; then
        A1="UNMEASURED"; NOTE="genuine-fetch prerequisite not established; cause unresolved"
      else
        A1="WRONG_FAILURE"; NOTE="the build failed at a step other than the intended target"
      fi
    else
      # the offline load, in a NAMED disposable container
      CNAME="item8-offline-${LABEL}"
      "$DOCKER" rm -f "$CNAME" >/dev/null 2>&1
      if [ "$IMAGE" = "memu-core" ]; then
        PROBE='import os,sentence_transformers as st; st.SentenceTransformer(os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2")); print("OFFLINE LOAD OK")'
      else
        PROBE='import subprocess,sys; sys.exit(subprocess.call([sys.executable,"/tmp/bake_tokenizer.py","verify"]))'
      fi
      "$DOCKER" run --name "$CNAME" --network none "$TAG" \
        python -c "$PROBE" > "${IDENT}/${LABEL}.offline.log" 2>&1
      OFFLINE_RC=$?

      if [ "$OFFLINE_RC" -ne 0 ]; then
        A1="UNMEASURED"; NOTE="the image built but the offline asset load failed; the branch's criterion is not established"
      elif [ "$BRANCH" = "B2" ] && [ "$INJECTED" != "yes" ]; then
        A1="UNMEASURED"; NOTE="the injected first-attempt failure was not observed; B2 measured nothing about recovery"
      elif [ "$BRANCH" = "B2" ] && [ "$RETRIES" -lt 1 ]; then
        A1="UNMEASURED"; NOTE="no GENUINE retry line observed; recovery cannot be claimed from the injection marker alone"
      elif [ "$BRANCH" = "B2" ] && ! awk "/${INJECT_MARK}/{i=NR} /${RETRY_MARK}/{if(i&&NR>i){f=1}} END{exit !f}" "$LOG"; then
        A1="UNMEASURED"; NOTE="no genuine retry line appears AFTER the injected failure; ordering does not establish recovery"
      else
        A1="PASS"
        NOTE="built, loaded offline with the network denied"
      fi

      # ── AXIS 2: provenance, computed SEPARATELY and never fed back ──
      python3 "$EXPLICIT" --image-ref "$TAG" --label "$LABEL" \
        --run-id "$RUN_ID" --docker "$DOCKER" \
        --out "${IDENT}/${LABEL}.jsonl" \
        > "${IDENT}/${LABEL}.log" 2>&1
      A2="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['identity_state'])" \
            "${IDENT}/${LABEL}.jsonl" 2>/dev/null || echo UNRECORDED)"
      COLLECTED_ID="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('docker_image_id') or '')" \
            "${IDENT}/${LABEL}.jsonl" 2>/dev/null || echo '')"

      # iidfile corroboration -- required by R2 and previously written
      # then ignored. Disagreement is an AXIS 2 fault, never an Axis 1 one.
      IIDCORR="UNRECORDED"
      if [ -s "$IID" ]; then
        IIDVAL="$(tr -d ' \n' < "$IID")"
        if [ -n "$COLLECTED_ID" ] && [ "$IIDVAL" = "$COLLECTED_ID" ]; then
          IIDCORR="CORROBORATED"
        else
          IIDCORR="MISMATCH"; A2="MISMATCH"
        fi
      fi

      if [ "$A2" = "RECORDED" ]; then
        if python3 "$BINDER" --verify-executed "$CNAME" --service "$LABEL" \
             --docker "$DOCKER" --against "${IDENT}/${LABEL}.jsonl" \
             --out "${IDENT}/${LABEL}.executed.jsonl" \
             > "${IDENT}/${LABEL}.bind.log" 2>&1; then
          A2="BOUND"
        else
          A2="MISMATCH"
        fi
      fi
      "$DOCKER" rm -f "$CNAME" >/dev/null 2>&1
    fi

    case "$A1:$A2" in
      PASS:BOUND|PASS:IMAGE_NOT_PRODUCED_BY_DESIGN) QUAL="true" ;;
      *) QUAL="false" ;;
    esac

    emit "$(IM="$IMAGE" BR="$BRANCH" A1="$A1" A2="$A2" QU="$QUAL" NT="$NOTE" \
            RI="$RUN_ID" TS="$TREE_SHA" DS="$DF_SHA" TG="$TAG" RT="$RETRIES" \
            EL="$ELAPSED" RC="$BUILD_RC" PC="$PRE_CLEAN" IC="$IIDCORR" \
            IJ="${INJECTED:-n/a}" python3 -c '
import json,os
e=os.environ
row={"image":e["IM"],"branch":e["BR"],"axis1_verdict":e["A1"],
     "axis2_provenance":e["A2"],"qualified_for_closure":e["QU"]=="true",
     "note":e["NT"],"run_id":e["RI"],"tree_sha":e["TS"],
     "dockerfile_sha256":e["DS"],"image_ref":e["TG"],
     "genuine_retries_observed":int(e["RT"]),"elapsed_seconds":int(e["EL"]),
     "build_exit":int(e["RC"]),"pre_build_state":e["PC"],
     "iidfile_corroboration":e["IC"],"injection_observed":e["IJ"]}
if e["BR"]=="B3":
    row["failure_mode"]="persistent network denial on the HF RUN only"
print(json.dumps(row))')"
    echo "::endgroup::"
  done
done

ROWS="$(grep -c . "$RESULTS" || true)"; ROWS="${ROWS:-0}"
echo "six branch(es) attempted; ${ROWS} result row(s) written"
[ "$ROWS" -eq 6 ] || { echo "INSTRUMENT FAILURE: ${ROWS} rows written, 6 required"; exit 2; }
exit 0
