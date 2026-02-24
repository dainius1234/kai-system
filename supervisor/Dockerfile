FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY supervisor/app.py ./app.py

EXPOSE 8051

CMD ["python", "app.py"]
