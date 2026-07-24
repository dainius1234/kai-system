FROM python:3.11-slim

WORKDIR /app
RUN groupadd -r app && useradd -r -g app -u 1001 app

COPY docker-watcher/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY docker-watcher/app.py ./

USER app
EXPOSE 8041
CMD ["python", "app.py"]
