FROM python:3.11-slim

WORKDIR /app

COPY verifier/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY verifier/app.py ./app.py

EXPOSE 8052

CMD ["python", "app.py"]
