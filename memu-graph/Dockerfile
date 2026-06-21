FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY memu-graph/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY common/ ./common/
COPY security/ ./security/
COPY memu-graph/app.py ./app.py

RUN mkdir -p /data/cognee && chown -R app:app /data/cognee
ENV SYSTEM_ROOT_DIRECTORY=/data/cognee/system
ENV DATA_ROOT_DIRECTORY=/data/cognee/data

EXPOSE 8061

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8061/health')" || exit 1

USER app

CMD ["python", "app.py"]
