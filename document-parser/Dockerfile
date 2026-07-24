FROM python:3.11-slim

# libredwg-tools provides dwg2dxf for DWG→DXF conversion (optional — falls back to 503)
RUN apt-get update && \
    (apt-get install -y --no-install-recommends libredwg-tools 2>/dev/null || true) && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY document-parser/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY document-parser/app.py ./

RUN addgroup --system app && adduser --system --ingroup app app

EXPOSE 8032

HEALTHCHECK --interval=30s --timeout=5s --start_period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8032/health')"

USER app

CMD ["python", "app.py"]
