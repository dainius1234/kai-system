FROM python:3.11-slim

WORKDIR /app
RUN groupadd -r app && useradd -r -g app -u 1001 app

COPY email-reader/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY email-reader/app.py ./

USER app
EXPOSE 8037
CMD ["python", "app.py"]
