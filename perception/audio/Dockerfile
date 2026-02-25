FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY perception/audio/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY perception/audio/app.py ./
COPY common/ ./common/

EXPOSE 8021

CMD ["python", "app.py"]
