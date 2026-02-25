FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY screen-capture/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY screen-capture/app.py ./app.py

EXPOSE 8059

CMD ["python", "app.py"]
