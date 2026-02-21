FROM python:3.11-slim

WORKDIR /app

COPY memu-core/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY memu-core/app.py ./app.py

EXPOSE 8001

CMD ["python", "app.py"]
