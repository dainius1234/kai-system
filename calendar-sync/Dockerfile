FROM python:3.11-slim

WORKDIR /app

COPY calendar-sync/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY calendar-sync/app.py ./app.py

EXPOSE 8055

CMD ["python", "app.py"]
