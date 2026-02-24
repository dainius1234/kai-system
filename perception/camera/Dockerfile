FROM python:3.11-slim

WORKDIR /app

COPY perception/camera/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY perception/camera/app.py ./
COPY common/ ./common/

EXPOSE 8020

CMD ["python", "app.py"]
