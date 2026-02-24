FROM python:3.11-slim

WORKDIR /app

COPY heartbeat/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY heartbeat/app.py ./
COPY common/ ./common/

EXPOSE 8010

CMD ["python", "app.py"]
