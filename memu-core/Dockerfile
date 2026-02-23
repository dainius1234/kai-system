FROM python:3.11-slim

WORKDIR /app

COPY memu-core/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY memu-core/app.py ./
COPY memu-core/lakefs_client.py ./
COPY common/ ./common/

EXPOSE 8001

CMD ["python", "app.py"]
