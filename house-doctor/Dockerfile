FROM python:3.11-slim
WORKDIR /app
COPY house-doctor/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY house-doctor/app.py .
COPY common/ common/
EXPOSE 8046
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8046"]
