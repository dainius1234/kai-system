FROM python:3.11-slim

WORKDIR /app

COPY tool-gate/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY tool-gate/app.py ./
COPY common/ ./common/
COPY security/ ./security/

EXPOSE 8000

CMD ["python", "app.py"]
