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
#
# Axis 2 may block closure. It may never rewrite Axis 1.
#
# The COMPOSITE claim -- whether a branch qualifies for closure -- is
# deliberately NOT a field here. This script produces observations; the
# summariser derives the conclusion from them. A producer that also
# certifies the conclusion drawn from it is a second authority for the
# same statement (rule 26), so the field is absent rather than trusted.
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
PARSER="scripts/security/parse_buildkit_events.py"
TCCHECK="scripts/security/check_item8_toolchain.py"
TOOLCHAIN="${ITEM8_TOOLCHAIN:-item8-toolchain.txt}"

# INSTRUMENT PREREQUISITES. Missing here means we cannot measure at all,
# which is a different thing from a branch having no subject.
for f in "$EXPLICIT" "$BINDER" "$PARSER" "$TCCHECK"; do
  [ -f "$f" ] || { echo "INSTRUMENT FAILURE: $f is missing"; exit 2; }
done
mkdir -p "$DERIVED" "$IDENT" || { echo "INSTRUMENT FAILURE: cannot create output dirs"; exit 2; }
: > "$RESULTS" || { echo "INSTRUMENT FAILURE: cannot write $RESULTS"; exit 2; }

# EVERY ROW BINDS TO THE TOOLCHAIN RECORD, AND THE RECORD IS VALIDATED
# BEFORE BUILD 1. Frozen R2 requires the frontend, docker/buildx
# versions, base-image digest, runner OS, tree and run identity recorded
# with every branch. A digest in each row binds a row to a FILE; it says
# nothing about whether that file contains those identities.
#
# The first implementation hashed whatever was there and looked for the
# literal word UNRESOLVED. The calibration proved the gap without meaning
# to: its fixture toolchain had TWO fields and all six branches qualified
# against it. And because the CI generator runs under `set -uo pipefail`
# rather than `-e`, a failed lookup leaves `key=` -- a present key with
# nothing behind it, which the UNRESOLVED search cannot see.
#
# So the record is validated by a separate instrument, HERE, before any
# irreversible build. A defect found now costs zero builds; found after
# build 6 it costs the whole no-redraw denominator. (D294)
# RUN_ID IS DEFINED BEFORE THE CHECK THAT USES IT, not after.
#
# The workflow passed `--expect-run-id`; this script called the same
# validator WITHOUT it, and defined RUN_ID afterwards -- so the shipped
# entry point could not fail closed on a stale record, and only the
# surrounding YAML was catching it. A prerequisite this script names as
# its own must hold when this script is what runs. (D296)
RUN_ID="${GITHUB_RUN_ID:-local}"
TREE_SHA="$(git rev-parse 'HEAD^{tree}' 2>/dev/null || echo UNKNOWN)"

[ -f "$TOOLCHAIN" ] || {
  echo "INSTRUMENT FAILURE: $TOOLCHAIN does not exist."
  echo "R2 requires the toolchain identities recorded WITH EVERY BRANCH."
  echo "There is no record, so no branch can be bound to one, and the"
  echo "cheapest moment to find that out is before build 1."
  exit 2
}
if ! python3 "$TCCHECK" --toolchain "$TOOLCHAIN" --expect-run-id "$RUN_ID"; then
  echo "INSTRUMENT FAILURE: the toolchain record did not validate."
  echo "No build has started, and none will. A SHA-256 of an incomplete"
  echo "record is a perfect hash of bad evidence."
  exit 2
fi
TOOLCHAIN_SHA="$(python3 -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$TOOLCHAIN")"

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
#
# The injection marker carries NO interpolated value. The derivation used
# to emit `ITEM8-B2-INJECTED-ATTEMPT=\$attempt`, and inside a double-
# quoted string the backslash SUPPRESSES expansion -- so the container
# printed the literal `$attempt` while the calibration's fake docker
# manufactured `=1`. The fixture was semantically better than the shipped
# command path. The shim's own control flow already guarantees the
# injected iteration is the first, so the number is not needed and the
# criterion is "exactly one occurrence". (D294)
RETRY_MARK='retrying in'
REFUSE_MARK='REFUSING TO BUILD'
INJECT_MARK='ITEM8-B2-INJECTED-FIRST-ATTEMPT'
BAKED_MARK='BAKED '

for IMAGE in memu-core memu-graph; do
  for BRANCH in B1 B2 B3; do
    LOWER="$(echo "$BRANCH" | tr 'A-Z' 'a-z')"
    TAG="kai-item8:${LOWER}-${IMAGE}"
    DF="${DERIVED}/Dockerfile.${IMAGE}.${BRANCH}"
    IID="${DERIVED}/${IMAGE}.${BRANCH}.iid"
    # No `.build.log`: there is no rendered log any more. The two event
    # captures and the target vertex's runtime-only log are the record,
    # and a variable naming a file nothing writes is an invitation to
    # read one that does not exist.
    LABEL="item8-${LOWER}-${IMAGE}"

    echo "::group::${IMAGE} ${BRANCH}"

    A1="UNMEASURED"; A2="UNRECORDED"; NOTE=""
    RETRIES=0; ELAPSED=0; BUILD_RC=-1; PRE_CLEAN="n/a"; IIDCORR="n/a"

    # WHAT THE MUTABLE BASE TAG RESOLVES TO, AT THIS BRANCH'S BUILD.
    #
    # `python:3.11-slim` can move under a six-build experiment, and two
    # arms on two base images are not two arms of one experiment. Pinning
    # it would change the subject; observing it does not. The summariser
    # requires all six to agree with each other and with the record taken
    # before build 1. Unresolvable is UNRESOLVED -- a fact, not a blank.
    # (D295)
    BASEDIG="$("$DOCKER" buildx imagetools inspect python:3.11-slim \
                 --format '{{.Manifest.Digest}}' 2>/dev/null | tr -d ' \n')"
    [ -n "$BASEDIG" ] || BASEDIG="UNRESOLVED"

    # ── the derived file must already exist (R11) ─────────────────────
    if [ ! -s "$DF" ]; then
      emit "$(BR="$BRANCH" IM="$IMAGE" RI="$RUN_ID" TS="$TREE_SHA" TC="$TOOLCHAIN_SHA" BD="$BASEDIG" python3 -c '
import json,os
print(json.dumps({"image":os.environ["IM"],"branch":os.environ["BR"],
 "axis1_verdict":"UNMEASURED","axis2_provenance":"UNRECORDED",
 "run_id":os.environ["RI"],
 "tree_sha":os.environ["TS"],"toolchain_sha256":os.environ.get("TC","ABSENT"),
 "base_image_digest":os.environ.get("BD","UNRESOLVED"),
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
        emit "$(BR="$BRANCH" IM="$IMAGE" RI="$RUN_ID" TS="$TREE_SHA" DS="$DF_SHA" PC="$PRE_CLEAN" TC="$TOOLCHAIN_SHA" BD="$BASEDIG" python3 -c '
import json,os
print(json.dumps({"image":os.environ["IM"],"branch":os.environ["BR"],
 "axis1_verdict":"UNMEASURED","axis2_provenance":"UNRECORDED",
 "run_id":os.environ["RI"],
 "tree_sha":os.environ["TS"],"dockerfile_sha256":os.environ["DS"],
 "toolchain_sha256":os.environ.get("TC","ABSENT"),"pre_build_state":os.environ["PC"],
 "base_image_digest":os.environ.get("BD","UNRESOLVED"),
 "note":"pre-build state not clean; a stale tag or leftover iidfile would "
        "contaminate IMAGE_NOT_PRODUCED_BY_DESIGN"}))')"
        echo "::endgroup::"; continue
      fi
    fi

    # ── build ─────────────────────────────────────────────────────────
    # STRUCTURED EVENTS, not a rendered log. `--progress=rawjson` emits
    # BuildKit SolveStatus objects in which a step's INSTRUCTION TEXT and
    # its RUNTIME OUTPUT are different fields of different objects, so no
    # amount of the log echoing an instruction can be read as execution.
    # Three reviews found that same confusion wearing three disguises;
    # this ends it structurally rather than with a stronger pattern.
    EVENTS="${DERIVED}/${IMAGE}.${BRANCH}.events-stderr.jsonl"
    EVENTS_OUT="${DERIVED}/${IMAGE}.${BRANCH}.events-stdout.jsonl"
    RTLOG="${DERIVED}/${IMAGE}.${BRANCH}.runtime.log"
    NOCACHE=""; [ "$BRANCH" = "B1" ] && NOCACHE="--no-cache"
    START="$(date +%s)"
    # shellcheck disable=SC2086
    DOCKER_BUILDKIT=1 "$DOCKER" build $NOCACHE --progress=rawjson \
      -f "$DF" -t "$TAG" --iidfile "$IID" . > "$EVENTS_OUT" 2> "$EVENTS"
    BUILD_RC=$?
    ELAPSED=$(( $(date +%s) - START ))

    # The target vertex is the HF retry loop, identified by its
    # instruction text -- which is the one legitimate use of the name:
    # asking WHICH STEP is the subject, never what happened in it.
    # BOTH descriptors. buildx writes its progress printer -- rawjson
    # included -- to STDERR, and CLI diagnostics land there too. Capturing
    # only stdout would have produced an empty stream AFTER all six
    # irreversible builds, and the calibration modelled the wrong
    # transport, so the fake was better than the real command path. (D294)
    FACTS="$(python3 "$PARSER" --events "$EVENTS" --events "$EVENTS_OUT" \
               --target-substring 'for attempt in 1 2 3 4 5' \
               --count "$RETRY_MARK" \
               --count "$REFUSE_MARK" \
               --count "$INJECT_MARK" \
               --emit-log "$RTLOG" --json 2>"${EVENTS}.parse.err")"
    PARSE_RC=$?

    if [ "$PARSE_RC" -ne 0 ] || [ -z "$FACTS" ]; then
      emit "$(BR="$BRANCH" IM="$IMAGE" RI="$RUN_ID" TS="$TREE_SHA" DS="$DF_SHA" \
              TC="$TOOLCHAIN_SHA" BD="$BASEDIG" PE="$(head -c 300 "${EVENTS}.parse.err" 2>/dev/null)" python3 -c '
import json,os
e=os.environ
print(json.dumps({"image":e["IM"],"branch":e["BR"],
 "axis1_verdict":"UNMEASURED","axis2_provenance":"UNRECORDED",
 "run_id":e["RI"],"tree_sha":e["TS"],"dockerfile_sha256":e["DS"],
 "toolchain_sha256":e["TC"],"base_image_digest":e["BD"],
 "note":"the build event stream could not be parsed, so nothing about "
        "this branch was observed: " + e["PE"]}))')"
      echo "::endgroup::"; continue
    fi

    EXECUTED="$(printf '%s' "$FACTS" | python3 -c 'import json,sys;print(json.load(sys.stdin)["executed"])')"
    CACHED="$(printf '%s' "$FACTS" | python3 -c 'import json,sys;print(json.load(sys.stdin)["cached"])')"
    VERR="$(printf '%s' "$FACTS" | python3 -c 'import json,sys;print(json.load(sys.stdin)["error"][:200])')"
    RETRIES="$(printf '%s' "$FACTS" | K="$RETRY_MARK" python3 -c 'import json,os,sys;print(json.load(sys.stdin)["counts"][os.environ["K"]])')"
    TARGET_REFUSALS="$(printf '%s' "$FACTS" | K="$REFUSE_MARK" python3 -c 'import json,os,sys;print(json.load(sys.stdin)["counts"][os.environ["K"]])')"
    INJECTIONS="$(printf '%s' "$FACTS" | K="$INJECT_MARK" python3 -c 'import json,os,sys;print(json.load(sys.stdin)["counts"][os.environ["K"]])')"
    TARGET_REFUSAL="no"; [ "${TARGET_REFUSALS:-0}" -ge 1 ] && TARGET_REFUSAL="yes"

    # ── AXIS 1: the contingency, decided WITHOUT any identity input ───
    if [ "$BRANCH" = "B3" ]; then
      # A2 stays UNRECORDED until non-existence is ESTABLISHED. Asserting
      # NOT_APPLICABLE_BY_DESIGN up front would name the expected answer
      # before measuring it.
      POST_TAG="absent"; POST_IID="absent"
      "$DOCKER" image inspect "$TAG" >/dev/null 2>&1 && POST_TAG="present"
      # EXISTENCE, not size. `-s` treats a zero-byte iidfile as absent,
      # and a zero-byte file is a file. The pre-check already used -e;
      # the two ends must ask the same question.
      [ -e "$IID" ] && POST_IID="present"
      # ARCHIVED ABSENCE. R2's B3 contract is that no image exists at
      # either end, and "axis2_provenance says so" is not evidence of
      # it. The summariser reads this AND checks the iidfile's absence
      # for itself, since that file is in the artefact package. (D296)
      PRE="$PRE_CLEAN" PT="$POST_TAG" PI="$POST_IID" python3 -c '
import json,os
e=os.environ
print(json.dumps({"pre_build_state":e["PRE"],"post_build_tag":e["PT"],
 "post_build_iidfile":e["PI"]}))' > "${IDENT}/${LABEL}.absence.json"
      if [ "$BUILD_RC" -eq 0 ]; then
        A1="UNMEASURED"; NOTE="the build SUCCEEDED under network denial; the intended refusal did not occur"
      elif [ "$TARGET_REFUSAL" != "yes" ]; then
        A1="WRONG_FAILURE"; NOTE="the build failed without the target step's refusal marker; the failure was elsewhere"
      elif [ -z "$VERR" ]; then
        # R2: the non-zero exit must ARISE FROM the target step. A whole
        # build failing while the target carries no error of its own is
        # a failure somewhere else that happens to have printed our text.
        # The parser gives us the vertex's own error; the first version
        # computed it and threw it away.
        A1="WRONG_FAILURE"; NOTE="the build failed but the target vertex carries no error of its own; the failure is not attributable to it"
      elif [ "${RETRIES:-0}" -ne 5 ]; then
        # Frozen R2 requires FIVE attempts observed. The first
        # implementation asserted "five attempts" in its note while never
        # checking the count -- a claim in the place of a measurement.
        A1="UNMEASURED"; NOTE="the refusal occurred but ${RETRIES} runtime retry line(s) were attributed to the target vertex, not the five the design requires"
      elif [ "$POST_TAG" != "absent" ] || [ "$POST_IID" != "absent" ]; then
        A1="UNMEASURED"; NOTE="post-build non-existence not established (tag=${POST_TAG}, iidfile=${POST_IID})"
      else
        A1="PASS"; A2="IMAGE_NOT_PRODUCED_BY_DESIGN"
        NOTE="${RETRIES} runtime retries and the refusal, both attributed to the target vertex; no image at either end"
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
      # ARCHIVED, because the claim engine may not take this from a
      # row field this same script wrote. The container's own exit
      # status is the observation; the row is a summary of it. (D296)
      echo "$OFFLINE_RC" > "${IDENT}/${LABEL}.offline.rc"

      if [ "$EXECUTED" != "True" ]; then
        # R2's B1 requires the HF instruction to PROVABLY execute rather
        # than be served from cache. Requesting --no-cache is a request;
        # the vertex's own `started` field is the observation.
        A1="UNMEASURED"; NOTE="the target vertex did not execute in this build"
      elif [ "$CACHED" = "True" ]; then
        A1="UNMEASURED"; NOTE="the target vertex was served FROM CACHE; the genuine fetch path did not run"
      elif [ "$OFFLINE_RC" -ne 0 ]; then
        A1="UNMEASURED"; NOTE="the image built but the offline asset load failed; the branch's criterion is not established"
      elif [ "$BRANCH" = "B2" ] && [ "${INJECTIONS:-0}" -ne 1 ]; then
        # EXACTLY ONE. The shim consumes a sentinel file on its first
        # iteration and can never fire again, so two occurrences mean the
        # subject is not the thing the derivation produced. The count is
        # the whole criterion now: the marker carries no attempt number,
        # because the version that tried to interpolate one printed the
        # literal `$attempt` in the container while the fake docker
        # manufactured `=1`. (D294)
        A1="UNMEASURED"; NOTE="${INJECTIONS} injection marker(s) in the target vertex runtime output; exactly one is required"
      elif [ "$BRANCH" = "B2" ] && [ "${RETRIES:-0}" -lt 1 ]; then
        # R2 requires a retry line OBSERVED. Build success plus a later
        # attempt is persuasive but is a DIFFERENT criterion, and one
        # criterion may not be substituted for another.
        A1="UNMEASURED"; NOTE="no runtime retry line attributed to the target vertex; recovery is not established"
      elif [ "$BRANCH" = "B2" ] && ! awk -v inj="$INJECT_MARK" \
             -v ret="$RETRY_MARK" -v bak="$BAKED_MARK" '
             index($0, inj) {i=NR}
             index($0, ret) {if (i && NR>i) r=NR}
             index($0, bak) {if (r && NR>r) b=1}
             END {exit !b}' "$RTLOG"; then
        # The frozen criterion is a later genuine fetch SUCCEEDING, and
        # both subjects print "BAKED ..." on that path. Now that runtime
        # output is attributable per vertex, observe it in order rather
        # than inferring it from the build's overall exit status.
        A1="UNMEASURED"; NOTE="the target vertex runtime log does not show injection then retry then a BAKED success, in that order"
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
      # ABSENT is its own state. The first implementation left it
      # UNRECORDED and let the branch qualify anyway, so a positive
      # branch could close with the corroboration R2 requires simply
      # missing. ABSENT/NULL/VALUE, rule 20, applied to a file.
      IIDCORR="ABSENT"
      if [ -e "$IID" ]; then
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

    emit "$(IM="$IMAGE" BR="$BRANCH" A1="$A1" A2="$A2" NT="$NOTE" \
            RI="$RUN_ID" TS="$TREE_SHA" DS="$DF_SHA" TG="$TAG" RT="$RETRIES" \
            EL="$ELAPSED" RC="$BUILD_RC" PC="$PRE_CLEAN" IC="$IIDCORR" \
            EX="${EXECUTED:-unknown}" CA="${CACHED:-unknown}" \
            TC="${TOOLCHAIN_SHA:-ABSENT}" IJ2="${INJECTIONS:-0}" BD="${BASEDIG:-UNRESOLVED}" \
            VE="${VERR:-}" python3 -c '
import json,os
e=os.environ
row={"image":e["IM"],"branch":e["BR"],"axis1_verdict":e["A1"],
     "axis2_provenance":e["A2"],"note":e["NT"],
     "target_vertex_executed":e["EX"],"target_vertex_cached":e["CA"],
     "toolchain_sha256":e["TC"],"base_image_digest":e["BD"],
     "injection_markers":int(e["IJ2"]),
     "target_vertex_error":e["VE"],
     "run_id":e["RI"],"tree_sha":e["TS"],
     "dockerfile_sha256":e["DS"],"image_ref":e["TG"],
     "runtime_retries_observed":int(e["RT"]),"elapsed_seconds":int(e["EL"]),
     "build_exit":int(e["RC"]),"pre_build_state":e["PC"],
     "iidfile_corroboration":e["IC"]}
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
