FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY memu-graph/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY common/ ./common/
COPY security/ ./security/
COPY memu-graph/app.py ./app.py

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

EXPOSE 8061

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8061/health')" || exit 1

CMD ["python", "app.py"]
