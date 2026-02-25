FROM python:3.11-slim

WORKDIR /app

COPY telegram-bot/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY telegram-bot/app.py ./app.py
COPY common/ ./common/

EXPOSE 8025

CMD ["python", "app.py"]
