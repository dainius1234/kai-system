FROM python:3.11-slim

WORKDIR /app

COPY backup-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backup-service/app.py ./app.py

EXPOSE 8054

CMD ["python", "app.py"]
