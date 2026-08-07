FROM python:3.11-slim

WORKDIR /app

COPY memu-core/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

# The TurboVec index and its writer lock live here, and this container
# runs as `app`. Docker seeds a fresh named volume from the image's
# directory — contents *and* ownership — so the mountpoint has to exist
# and be owned by `app` before the volume is created, or `/data/turbovec`
# arrives owned by root and the service cannot write to it:
#
#   PermissionError: [Errno 13] Permission denied: '/data/turbovec/.writer.lock'
#
# Nobody had hit this because nobody had got here: the writer branch
# died on an unbound `socket` before it ever tried to write. Two layers
# of never-executed code, one behind the other.
RUN mkdir -p /data/turbovec && chown -R app:app /data

COPY memu-core/app.py ./
COPY memu-core/lakefs_client.py ./
COPY common/ ./common/
COPY security/ ./security/

# ── The embedding model is part of the image, not something fetched at
# boot ────────────────────────────────────────────────────────────────
#
# This service is attached only to networks declared `internal: true` in
# all three compose profiles, so it has **no egress, ever**. It shipped
# `sentence-transformers` without a model, which meant:
#
#   * MEMU_ALLOW_FAKE_EMBEDDINGS=false — the documented production
#     default, "real model or crash" — could never start. It raised at
#     import in every profile. Nothing had seen it because every CI
#     bring-up overrides the flag to true.
#   * MEMU_ALLOW_FAKE_EMBEDDINGS=true still spent 70-100s in DNS backoff
#     against a healthcheck that gives up at ~100s. CI run 708 lost that
#     race twice; the six runs before it had won it.
#
# `memu-graph/Dockerfile` already solved this exact problem and is green
# in CI. This is that pattern, applied where it was missing — including
# the two traps it documents:
#
#   * `adduser --system` sets HOME=/nonexistent, so HF's cache resolution
#     has nowhere to write. HOME is pointed at a real directory.
#   * a cache written as root is unreadable by `app`. The directories are
#     created and chowned first, and USER is switched *before* the
#     download, so the files land owned correctly. This repository has
#     already been bitten by exactly that with /data/turbovec.
# NOT under /data, and that is the whole point. `docker-compose.sovereign.yml`
# mounts `memu_data:/data` — the entire tree — and Docker seeds a named
# volume from the image **only when the volume is new**. Put the model at
# /data/hf_cache and an existing deployment's volume shadows it on
# upgrade: CI stays green on its fresh volume while a real deployment
# comes back up with no model and refuses to start. /opt is mounted by
# nothing in any of the three profiles, so image content stays image
# content. (memu-graph can safely use /data/hf_cache — it declares no
# volumes at all. Same rule, different answer, because the mounts differ.)
RUN mkdir -p /opt/hf_cache /opt/home && chown -R app:app /opt/hf_cache /opt/home
ENV HF_HOME=/opt/hf_cache
ENV HOME=/opt/home

# Offline by default: the image carries the model, so the running
# container has no reason to reach the network for one. Overridable from
# compose for anyone deliberately using a different EMBEDDING_MODEL.
ENV HF_HUB_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

USER app

# NOT best-effort, unlike memu-graph's extension pre-install. That one is
# recoverable on first run; this is not — an image without the model is
# an image that cannot do its job, and the whole point here is that the
# runtime cannot fetch it later. I-1: fail closed on a missing input.
# The cost is a build that depends on huggingface.co being reachable.
#
# The revision is NOT pinned yet, deliberately. Pinning needs a real
# commit sha and huggingface.co is unreachable from the authoring
# environment; a guessed sha fails every build. So the build *prints* the
# sha it resolved, and pinning follows from that measured value rather
# than an invented one.
#
# Retried, because "fail closed" means *never ship an image without the
# model* — not *die on the first dropped packet*. A single transient blip
# at huggingface.co should not fail a build; five consecutive failures
# over ~100s is a real outage and should. Without this the hard failure
# above is brittle rather than strict, which is a different property
# wearing the same word.
RUN for attempt in 1 2 3 4 5; do \
      HF_HUB_OFFLINE=0 python -c "\
import sentence_transformers, huggingface_hub, os; \
name = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2'); \
sentence_transformers.SentenceTransformer(name); \
info = huggingface_hub.model_info(f'sentence-transformers/{name}'); \
print(f'BAKED {name} revision={info.sha}')" && exit 0; \
      echo "model download attempt ${attempt}/5 failed; retrying in $((attempt * 10))s"; \
      sleep $((attempt * 10)); \
    done; \
    echo "REFUSING TO BUILD: could not fetch the embedding model in 5 attempts."; \
    echo "This image would start only with MEMU_ALLOW_FAKE_EMBEDDINGS=true and"; \
    echo "could never serve real embeddings, because the running container has"; \
    echo "no egress by design. Rollback point: kai-pm/WAYPOINTS.md."; \
    exit 1

EXPOSE 8001

# start-period was 10s, which is not enough time for a service that loads
# an ML model before it binds a port. Combined with the download stall it
# is what failed run 708. 60s is a deliberate over-allowance, not a
# measurement: app.py now logs the actual load time as
# "embedding backend ready in N.Ns", and this should be tuned down to a
# real number once CI has reported one.
#
# The cost of raising it is real and worth stating: every dependent now
# waits on `service_healthy`, so a genuinely dead memu-core takes
# 60+3*30 = 150s to be declared dead instead of 100s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["python", "app.py"]
