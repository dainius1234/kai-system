FROM python:3.11-slim

WORKDIR /app

COPY langgraph/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY langgraph/app.py ./
COPY langgraph/config.py ./
COPY langgraph/conviction.py ./
COPY common/ ./common/

EXPOSE 8007

CMD ["python", "app.py"]
