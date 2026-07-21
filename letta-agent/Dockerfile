FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY letta-agent/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system app && adduser --system --ingroup app app
COPY common/ ./common/
COPY security/ ./security/
COPY letta-agent/app.py ./app.py
RUN mkdir -p /data/letta && chown -R app:app /data/letta
# Letta writes its SQLite store and agent configs to LETTA_BASE_PATH.
# HOME must exist for any library that defaults to ~/.letta.
ENV LETTA_BASE_PATH=/data/letta
ENV HOME=/data/home
RUN mkdir -p /data/home && chown -R app:app /data/home
USER app
EXPOSE 8062
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8062/health')" || exit 1
CMD ["python", "app.py"]
