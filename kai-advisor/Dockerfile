FROM python:3.11-slim

WORKDIR /app

COPY kai-advisor/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY kai-advisor/app.py ./app.py
COPY common/ ./common/
COPY data/ ./data/

EXPOSE 8090

CMD ["python", "app.py"]
