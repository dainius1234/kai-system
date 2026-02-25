FROM python:3.11-slim

WORKDIR /app

COPY output/tts/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY common/ ./common/
COPY security/ ./security/
COPY output/tts/app.py ./app.py

EXPOSE 8030

CMD ["python", "app.py"]
