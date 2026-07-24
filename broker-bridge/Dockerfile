FROM python:3.11-slim
WORKDIR /app
COPY broker-bridge/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY broker-bridge/app.py .
RUN useradd -r -u 1001 -s /sbin/nologin app && chown -R app:app /app
USER app
EXPOSE 8034
ENV PYTHONUNBUFFERED=1
CMD ["python", "app.py"]
