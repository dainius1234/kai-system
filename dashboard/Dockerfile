FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY dashboard/app.py ./
COPY dashboard/static/ ./static/
COPY common/ ./common/
COPY security/ ./security/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

USER app

CMD ["python", "app.py"]
