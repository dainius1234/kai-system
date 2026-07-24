FROM python:3.11-slim

# Playwright installs chromium + system deps here (runs as root)
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY browser-agent/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Install Chromium and its OS-level dependencies into /ms-playwright
RUN mkdir -p /ms-playwright && chmod 777 /ms-playwright \
    && playwright install --with-deps chromium

COPY common/ ./common/
COPY browser-agent/app.py ./

RUN addgroup --system app && adduser --system --ingroup app app

EXPOSE 8040

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8040/health')"

USER app

CMD ["python", "app.py"]
