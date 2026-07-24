FROM python:3.11-slim

WORKDIR /app
RUN groupadd -r app && useradd -r -g app -u 1001 app

COPY git-watcher/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY git-watcher/app.py ./

USER app
EXPOSE 8044
CMD ["python", "app.py"]
