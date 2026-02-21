FROM python:3.11-slim

WORKDIR /app

COPY tool-gate/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY common ./common
COPY tool-gate/app.py ./app.py

EXPOSE 8000

CMD ["python", "app.py"]
