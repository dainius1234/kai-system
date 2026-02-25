FROM python:3.11-slim

WORKDIR /app

COPY memory-compressor/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY memory-compressor/app.py ./app.py

EXPOSE 8057

CMD ["python", "app.py"]
