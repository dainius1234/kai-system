FROM python:3.11-slim

WORKDIR /app

COPY perception/audio/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY perception/audio/app.py ./app.py

EXPOSE 8021

CMD ["python", "app.py"]
