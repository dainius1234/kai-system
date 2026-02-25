FROM python:3.11-slim

WORKDIR /app

COPY fusion-engine/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY fusion-engine/app.py ./app.py

EXPOSE 8053

CMD ["python", "app.py"]
