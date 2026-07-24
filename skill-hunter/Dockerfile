FROM python:3.11-slim
WORKDIR /app
COPY skill-hunter/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY skill-hunter/app.py .
COPY common/ common/
EXPOSE 8045
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8045"]
