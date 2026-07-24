FROM python:3.11-slim

# libgl1 + libglib2.0 required by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY perception/vision/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY perception/vision/app.py ./

RUN addgroup --system app && adduser --system --ingroup app app

EXPOSE 8023

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8023/health')"

USER app

CMD ["python", "app.py"]
