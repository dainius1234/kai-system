FROM python:3.11-slim

WORKDIR /app

COPY monitor-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY monitor-service/app.py ./

RUN addgroup --system app && adduser --system --ingroup app app

EXPOSE 8033

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8033/health')"

USER app

CMD ["python", "app.py"]
