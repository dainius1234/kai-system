FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" app && \
    apt-get update && apt-get install -y --no-install-recommends libnotify-bin && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY ../../common /app/common

USER app
ENV PORT=8031
EXPOSE 8031
CMD ["python", "app.py"]
