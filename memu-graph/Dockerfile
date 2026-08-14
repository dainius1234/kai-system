FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY memu-graph/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY common/ ./common/
COPY security/ ./security/
COPY memu-graph/app.py ./app.py
# KAI-GATE-050's success predicate. app.py imports it at module
# scope, so an image without it fails at import, not at request time.
COPY memu-graph/cognify_result.py ./cognify_result.py

RUN mkdir -p /data/cognee /data/hf_cache /data/home && chown -R app:app /data/cognee /data/hf_cache /data/home
ENV SYSTEM_ROOT_DIRECTORY=/data/cognee/system
ENV DATA_ROOT_DIRECTORY=/data/cognee/data
# adduser --system gives the "app" user HOME=/nonexistent, which breaks
# HuggingFace's tokenizer download (it caches under $HF_HOME/$HOME by
# default and can't write there). Point it at a writable directory instead.
ENV HF_HOME=/data/hf_cache
# Cognee's Ladybug/Kuzu adapter caches its downloaded JSON extension under
# $HOME — also broken by HOME=/nonexistent. Give it a real, writable HOME
# (same path at build time and runtime, see RUN below) instead of leaving
# the install-extension warm-up to fail silently at startup.
ENV HOME=/data/home

USER app

# Pre-install Ladybug/Kuzu's JSON extension at build time, when network
# access is available — avoids "Extension: json ... has not been installed"
# Binder errors at query time in network-restricted runtime containers
# (e.g. CI). Same install path/HOME the running adapter uses (see
# cognee/infrastructure/databases/graph/ladybug/adapter.py's
# _initialize_connection), so the cached extension lands exactly where the
# server looks for it. Best-effort: a failed download must not break the build.
RUN python -c "from cognee_db_workers._kuzu_helpers import install_json_extension_local; install_json_extension_local(buffer_pool_size=268435456)" \
    || echo "WARNING: JSON extension pre-install skipped (no network at build time); it will be installed on first run if the container has network access."

# ── The tokenizer is part of the image, not something fetched on the
#    first request ─────────────────────────────────────────────────────
#
# KAI-GATE-048, measured in runs 31681569455 / 31682452927 / 31684208364.
# This service is attached only to `agent-net`, declared `internal: true`,
# and a socket probe from inside the container confirmed it: DNS fails
# for huggingface.co and 1.1.1.1:443 is unreachable. Yet cognee's chunker
# builds a HuggingFace tokenizer in-process on the first `/graph/*`
# request:
#
#   OllamaEmbeddingEngine.__init__ -> get_tokenizer()
#     -> cognee/infrastructure/llm/tokenizer/HuggingFace/adapter.py:32
#        AutoTokenizer.from_pretrained(model)
#
# Deployed, that produced two 5-attempt retry sequences against
# huggingface.co, ~47s, and HTTP 502. The service itself was fine — ready
# at +5.1s, /health still 200 afterwards. What was missing was the ASSET.
#
# WHY THE LOADER RUNS THE BAKE INSTEAD OF A COPY
# ==============================================
#
# The measured cache is not four files. Alongside config.json,
# tokenizer.json, tokenizer_config.json and vocab.txt it carries
# `refs/main`, `CACHEDIR.TAG`, and three `.no_exist` NEGATIVE-cache
# markers (added_tokens.json, chat_template.jinja,
# special_tokens_map.json). Those markers are how the offline loader
# knows not to go looking for files that do not exist upstream. A bake
# that hand-copied the visible four could ship an image that still
# reaches for the network. So the loader itself writes the tree.
#
# WHY THE REVISION IS ASSERTED, NOT REQUESTED
# ===========================================
#
# cognee calls `from_pretrained(model)` with NO `revision=` — the whole
# package contains none — so at runtime it resolves `refs/main`. Fetching
# here with an explicit revision would populate `snapshots/<sha>` without
# necessarily writing the `refs/main` the runtime then needs, and the
# offline load would fail on an image whose build looked correct. So the
# fetch is by `main`, exactly as the runtime resolves it, and the sha is
# then CHECKED against the pinned value. Upstream moving fails the build
# instead of silently changing what ships.
ARG MEMU_GRAPH_TOKENIZER=bert-base-uncased
# Measured 2026-08-13, run 31684208364. scripts/test_asset_contract.py
# asserts this stays equal to docker-compose.full.yml's
# HUGGINGFACE_TOKENIZER — a value in two places is a value that drifts.
ARG MEMU_GRAPH_TOKENIZER_REVISION=86b5e0934494bd15c9632b12f734a8a67f723594
# Promoted to ENV BEFORE the RUNs that read them, and left in the image
# so the acceptance probes load exactly the name the bake used rather
# than one written down a second time.
ENV MEMU_GRAPH_TOKENIZER=${MEMU_GRAPH_TOKENIZER}
ENV MEMU_GRAPH_TOKENIZER_REVISION=${MEMU_GRAPH_TOKENIZER_REVISION}

COPY memu-graph/bake_tokenizer.py /tmp/bake_tokenizer.py

# NOT best-effort, unlike the Kuzu extension above. That one is
# recoverable on first run; this is not — the running container has no
# egress by design, so an image without the tokenizer is an image whose
# graph capability cannot work. Retried, because "fail closed" means
# never ship an image without the asset, not die on the first dropped
# packet.
#
# The loop body is a separate script rather than an inline `python -c`.
# memu-core's equivalent inlines it and its retry message prints
# "attempt /5", because Docker substitutes ${attempt} — which is not a
# build arg — to the empty string before the shell ever sees it. A file
# has no quoting or substitution hazard at all.
RUN for attempt in 1 2 3 4 5; do \
      python /tmp/bake_tokenizer.py fetch && exit 0; \
      echo "tokenizer bake attempt failed; retrying in 10s"; \
      sleep 10; \
    done; \
    echo "REFUSING TO BUILD: could not obtain the pinned tokenizer in 5 attempts."; \
    echo "This image would start, answer /health, and then fail every"; \
    echo "/graph/* request with a 502 after ~47s of retries against a"; \
    echo "host it cannot reach — measured, KAI-GATE-048. Rollback point:"; \
    echo "kai-pm/WAYPOINTS.md."; \
    exit 1

# A green build is not the contract. This disables external access the
# same way the runtime will and loads the tokenizer again from the cache
# just written — proving "downloaded files -> loader-readable complete
# asset" rather than "files appear to exist". The `--network none` proof
# against the FINAL image is stronger and is where the acceptance
# evidence comes from; this is the build refusing to ship something that
# cannot load.
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /tmp/bake_tokenizer.py verify

# Offline by default: the image carries the tokenizer, so the running
# container has no reason to reach a registry for one. Measured honoured
# by this stack — huggingface_hub.constants.HF_HUB_OFFLINE reads back
# True — and measured to turn a 46.4s retry storm into a 0.3s failure
# when an asset IS missing. That is obligation 2 of the D189 invariant;
# the bake above is obligation 1. Neither is sufficient alone.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_DISABLE_TELEMETRY=1

EXPOSE 8061

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8061/health')" || exit 1

CMD ["python", "app.py"]
