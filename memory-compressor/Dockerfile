FROM python:3.11-slim

WORKDIR /app

COPY memory-compressor/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY common/ ./common/
COPY security/ ./security/
COPY memory-compressor/app.py ./app.py

EXPOSE 8057

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8057/health')" || exit 1

USER app

CMD ["python", "app.py"]
