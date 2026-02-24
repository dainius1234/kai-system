FROM python:3.11-slim

WORKDIR /app

COPY perception/audio/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY perception/audio/app.py ./
COPY common/ ./common/

EXPOSE 8021

CMD ["python", "app.py"]
