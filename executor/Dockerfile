FROM python:3.11-slim

WORKDIR /app

COPY executor/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY executor/app.py ./
COPY common/ ./common/

EXPOSE 8002

CMD ["python", "app.py"]
