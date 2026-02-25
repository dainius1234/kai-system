FROM python:3.11-slim

WORKDIR /app

COPY output/avatar/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY output/avatar/app.py ./app.py

EXPOSE 8081

CMD ["python", "app.py"]
