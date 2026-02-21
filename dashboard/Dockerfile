FROM python:3.11-slim

WORKDIR /app

COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY dashboard/app.py ./app.py

EXPOSE 8080

CMD ["python", "app.py"]
