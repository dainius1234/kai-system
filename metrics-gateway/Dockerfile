FROM python:3.11-slim

WORKDIR /app

COPY metrics-gateway/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY metrics-gateway/app.py ./app.py

EXPOSE 8058

CMD ["python", "app.py"]
