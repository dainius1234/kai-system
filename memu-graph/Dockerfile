FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY memu-graph/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY common/ ./common/
COPY security/ ./security/
COPY memu-graph/app.py ./app.py

RUN mkdir -p /data/cognee /data/hf_cache && chown -R app:app /data/cognee /data/hf_cache
ENV SYSTEM_ROOT_DIRECTORY=/data/cognee/system
ENV DATA_ROOT_DIRECTORY=/data/cognee/data
# adduser --system gives the "app" user HOME=/nonexistent, which breaks
# HuggingFace's tokenizer download (it caches under $HF_HOME/$HOME by
# default and can't write there). Point it at a writable directory instead.
ENV HF_HOME=/data/hf_cache

EXPOSE 8061

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8061/health')" || exit 1

USER app

CMD ["python", "app.py"]
