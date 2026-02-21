FROM python:3.11-slim

WORKDIR /app

COPY langgraph/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY langgraph/app.py ./app.py

EXPOSE 8007

CMD ["python", "app.py"]
