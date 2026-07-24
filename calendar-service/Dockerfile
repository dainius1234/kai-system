FROM python:3.11-slim

WORKDIR /app
RUN groupadd -r app && useradd -r -g app -u 1001 app

COPY calendar-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY calendar-service/app.py ./

USER app
EXPOSE 8043
CMD ["python", "app.py"]
