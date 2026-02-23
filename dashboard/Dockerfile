FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard/app.py ./
COPY dashboard/static/ ./static/
COPY common/ ./common/

EXPOSE 8080

CMD ["python", "app.py"]
