FROM python:3.11-slim

WORKDIR /app

COPY ledger-worker/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY ledger-worker/app.py ./app.py

EXPOSE 8056

CMD ["python", "app.py"]
