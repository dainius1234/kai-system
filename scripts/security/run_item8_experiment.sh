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
# EXIT STATUS IS ABOUT THE INSTRUMENT, NOT THE SUBJECT (doctrine rule 5).
# A branch that is WRONG_FAILURE or UNMEASURED is a RESULT and is banked
# as one; this script still exits 0, because a failing contingency must
# not be reported as a broken workflow. It exits non-zero only when it
# cannot measure at all.
#
# NO RE-DRAWS. Six builds, every outcome recorded, nothing repeated
# "until it works" -- D247 §5, and an external prerequisite is not an
# exemption from it (D289).
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

DERIVED="item8-derived"
IDENT="item8-identity"
RESULTS="item8-results.jsonl"
mkdir -p "$DERIVED" "$IDENT"
: > "$RESULTS"

RUN_ID="${GITHUB_RUN_ID:-local}"
TREE_SHA="$(git rev-parse 'HEAD^{tree}')"

# One row per branch. Written even when everything went wrong, because a
# missing row is indistinguishable from a branch nobody ran.
emit() {  # emit <json>
  echo "$1" >> "$RESULTS"
}

json_escape() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

for IMAGE in memu-core memu-graph; do
  for BRANCH in B1 B2 B3; do
    TAG="kai-item8:$(echo "$BRANCH" | tr 'A-Z' 'a-z')-${IMAGE}"
    DF="${DERIVED}/Dockerfile.${IMAGE}.${BRANCH}"
    IID="${DERIVED}/${IMAGE}.${BRANCH}.iid"
    LOG="${DERIVED}/${IMAGE}.${BRANCH}.build.log"
    LABEL="item8-$(echo "$BRANCH" | tr 'A-Z' 'a-z')-${IMAGE}"

    echo "::group::${IMAGE} ${BRANCH}"

    # ── the derived file must already exist ───────────────────────────
    # Derivation happens in the workflow BEFORE any build, so a refusal
    # costs zero builds. R11: if it is missing, this branch has no
    # subject and no observation is made about it.
    if [ ! -s "$DF" ]; then
      echo "no derived Dockerfile at $DF"
      emit "$(python3 - "$IMAGE" "$BRANCH" "$RUN_ID" "$TREE_SHA" <<'PY'
import json,sys
i,b,r,t = sys.argv[1:5]
print(json.dumps({"image":i,"branch":b,"verdict":"UNMEASURED","run_id":r,
  "tree_sha":t,"unmet_prerequisite":"derivation refused; no experimental "
  "Dockerfile exists, so the branch has no subject"}))
PY
)"
      echo "::endgroup::"; continue
    fi
    DF_SHA="$(python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$DF")"

    # ── B3 pre-assertion: the tag and iidfile must NOT already exist ───
    PRE_CLEAN="n/a"
    if [ "$BRANCH" = "B3" ]; then
      PRE_CLEAN="clean"
      if docker image inspect "$TAG" >/dev/null 2>&1; then PRE_CLEAN="tag-already-exists"; fi
      if [ -e "$IID" ]; then PRE_CLEAN="${PRE_CLEAN},iidfile-already-exists"; fi
      if [ "$PRE_CLEAN" != "clean" ]; then
        emit "$(python3 - "$IMAGE" "$BRANCH" "$RUN_ID" "$TREE_SHA" "$DF_SHA" "$PRE_CLEAN" <<'PY'
import json,sys
i,b,r,t,s,p = sys.argv[1:7]
print(json.dumps({"image":i,"branch":b,"verdict":"UNMEASURED","run_id":r,
  "tree_sha":t,"dockerfile_sha256":s,
  "unmet_prerequisite":f"pre-build state not clean ({p}); a stale tag or "
  "leftover iidfile would contaminate IMAGE_NOT_PRODUCED_BY_DESIGN"}))
PY
)"
        echo "::endgroup::"; continue
      fi
    fi

    # ── build ─────────────────────────────────────────────────────────
    NOCACHE=""
    [ "$BRANCH" = "B1" ] && NOCACHE="--no-cache"
    START="$(date +%s)"
    # shellcheck disable=SC2086
    DOCKER_BUILDKIT=1 docker build $NOCACHE \
      -f "$DF" -t "$TAG" --iidfile "$IID" . > "$LOG" 2>&1
    BUILD_RC=$?
    ELAPSED=$(( $(date +%s) - START ))

    ATTEMPTS="$(grep -c 'attempt.*failed' "$LOG" || true)"
    TARGET_REFUSAL="no"
    grep -q 'REFUSING TO BUILD' "$LOG" && TARGET_REFUSAL="yes"

    # ── classify ──────────────────────────────────────────────────────
    VERDICT="UNMEASURED"
    NOTE=""
    IMAGE_STATE="UNRECORDED"

    if [ "$BRANCH" = "B3" ]; then
      POST_TAG="absent"; POST_IID="absent"
      docker image inspect "$TAG" >/dev/null 2>&1 && POST_TAG="present"
      [ -s "$IID" ] && POST_IID="present"
      if [ "$BUILD_RC" -eq 0 ]; then
        VERDICT="UNMEASURED"
        NOTE="the build SUCCEEDED under network denial; the intended refusal did not occur"
      elif [ "$TARGET_REFUSAL" != "yes" ]; then
        VERDICT="WRONG_FAILURE"
        NOTE="the build failed without the target step's refusal marker; the failure was elsewhere"
      elif [ "$POST_TAG" != "absent" ] || [ "$POST_IID" != "absent" ]; then
        VERDICT="UNMEASURED"
        NOTE="post-build non-existence not established (tag=${POST_TAG}, iidfile=${POST_IID})"
      else
        VERDICT="PASS"
        IMAGE_STATE="IMAGE_NOT_PRODUCED_BY_DESIGN"
        NOTE="five attempts, target-step refusal, no image at either end"
      fi
    else
      if [ "$BUILD_RC" -ne 0 ]; then
        if [ "$TARGET_REFUSAL" = "yes" ]; then
          VERDICT="UNMEASURED"
          NOTE="genuine-fetch prerequisite not established; cause unresolved"
        else
          VERDICT="WRONG_FAILURE"
          NOTE="the build failed at a step other than the intended target"
        fi
      else
        # identity of the image that was just built
        python3 scripts/security/collect_explicit_image_identity.py \
          --image-ref "$TAG" --label "$LABEL" --run-id "$RUN_ID" \
          --out "${IDENT}/${LABEL}.jsonl" > "${IDENT}/${LABEL}.log" 2>&1
        # the offline load, in a NAMED disposable container kept until
        # its .Image has been read
        CNAME="item8-offline-${LABEL}"
        docker rm -f "$CNAME" >/dev/null 2>&1 || true
        if [ "$IMAGE" = "memu-core" ]; then
          PROBE='import os,sentence_transformers as st; st.SentenceTransformer(os.getenv("EMBEDDING_MODEL","all-MiniLM-L6-v2")); print("OFFLINE LOAD OK")'
        else
          PROBE='import subprocess,sys; sys.exit(subprocess.call([sys.executable,"/tmp/bake_tokenizer.py","verify"]))'
        fi
        docker run --name "$CNAME" --network none "$TAG" \
          python -c "$PROBE" > "${IDENT}/${LABEL}.offline.log" 2>&1
        OFFLINE_RC=$?
        BIND_RC=1
        if python3 scripts/security/collect_image_identity.py \
             --verify-executed "$CNAME" --service "$LABEL" \
             --against "${IDENT}/${LABEL}.jsonl" \
             --out "${IDENT}/${LABEL}.executed.jsonl" \
             > "${IDENT}/${LABEL}.bind.log" 2>&1; then BIND_RC=0; fi
        docker rm -f "$CNAME" >/dev/null 2>&1 || true

        IMAGE_STATE="$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['identity_state'])" \
          "${IDENT}/${LABEL}.jsonl" 2>/dev/null || echo UNRECORDED)"

        INJECTED="n/a"
        if [ "$BRANCH" = "B2" ]; then
          INJECTED="no"
          grep -q 'ITEM8-B2: first attempt failed by injection' "$LOG" && INJECTED="yes"
        fi

        if [ "$OFFLINE_RC" -ne 0 ]; then
          VERDICT="UNMEASURED"
          NOTE="the image built but the offline asset load failed; the branch's PASS criterion is not established"
        elif [ "$BRANCH" = "B2" ] && [ "$INJECTED" != "yes" ]; then
          VERDICT="UNMEASURED"
          NOTE="the injected first-attempt failure was not observed; B2 measured nothing about recovery"
        elif [ "$BRANCH" = "B2" ] && [ "${ATTEMPTS:-0}" -lt 1 ]; then
          VERDICT="UNMEASURED"
          NOTE="no retry line observed; recovery cannot be claimed"
        elif [ "$BIND_RC" -ne 0 ]; then
          VERDICT="UNMEASURED"
          NOTE="the executed-container binding did not MATCH; see the bind log"
        else
          VERDICT="PASS"
          NOTE="built, loaded offline, and the container ran the recorded image"
        fi
      fi
    fi

    emit "$(python3 - "$IMAGE" "$BRANCH" "$RUN_ID" "$TREE_SHA" "$DF_SHA" \
              "$VERDICT" "$NOTE" "$IMAGE_STATE" "$TAG" \
              "${ATTEMPTS:-0}" "$ELAPSED" "$BUILD_RC" "$PRE_CLEAN" <<'PY'
import json,sys
(i,b,r,t,s,v,n,st,tag,att,el,rc,pre) = sys.argv[1:14]
row = {"image":i,"branch":b,"verdict":v,"note":n,"run_id":r,"tree_sha":t,
       "dockerfile_sha256":s,"image_state":st,"image_ref":tag,
       "attempts_observed":int(att),"elapsed_seconds":int(el),
       "build_exit":int(rc),"pre_build_state":pre}
if b == "B3":
    row["executed_binding"] = "NOT_APPLICABLE_BY_DESIGN"
    row["failure_mode"] = "persistent network denial on the HF RUN only"
print(json.dumps(row))
PY
)"
    echo "::endgroup::"
  done
done

echo "six branch(es) attempted; $(wc -l < "$RESULTS") result row(s) written"
exit 0
