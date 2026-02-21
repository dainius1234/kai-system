FROM python:3.11-slim

WORKDIR /app

COPY heartbeat/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY heartbeat/app.py ./app.py

EXPOSE 8010

CMD ["python", "app.py"]
