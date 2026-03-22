FROM python:3.11-slim

WORKDIR /app

COPY memu-core/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY memu-core/app.py ./
COPY memu-core/lakefs_client.py ./
COPY common/ ./common/
COPY security/ ./security/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

USER app

CMD ["python", "app.py"]
