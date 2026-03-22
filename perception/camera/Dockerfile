FROM python:3.11-slim

WORKDIR /app

COPY perception/camera/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY perception/camera/app.py ./
COPY common/ ./common/

EXPOSE 8020

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8020/health')" || exit 1

USER app

CMD ["python", "app.py"]
