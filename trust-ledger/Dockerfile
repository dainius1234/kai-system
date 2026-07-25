FROM python:3.11-slim

WORKDIR /app

COPY trust-ledger/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY trust-ledger/app.py ./
COPY trust-ledger/ledger.py ./
COPY trust-ledger/score.py ./
COPY common/ ./common/

RUN mkdir -p /data/trust-ledger && chown -R app:app /data

EXPOSE 8047

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8047/health')" || exit 1

USER app

CMD ["python", "app.py"]
