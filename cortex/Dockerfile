FROM python:3.11-slim
WORKDIR /app
COPY cortex/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY cortex/app.py .
EXPOSE 8048
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8048"]
