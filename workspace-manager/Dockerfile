FROM python:3.11-slim

WORKDIR /app

COPY workspace-manager/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY workspace-manager/app.py ./app.py

EXPOSE 8060

CMD ["python", "app.py"]
