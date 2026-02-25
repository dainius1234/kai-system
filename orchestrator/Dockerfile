FROM python:3.11-slim

WORKDIR /app

COPY orchestrator/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator/app.py ./app.py

EXPOSE 8050

CMD ["python", "app.py"]
