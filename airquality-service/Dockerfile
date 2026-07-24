FROM python:3.11-slim

WORKDIR /app
RUN groupadd -r app && useradd -r -g app -u 1001 app

COPY airquality-service/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY airquality-service/app.py ./

USER app
EXPOSE 8042
CMD ["python", "app.py"]
