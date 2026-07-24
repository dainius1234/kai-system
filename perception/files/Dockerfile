FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./
COPY ../../common /app/common

USER app
ENV PORT=8025
EXPOSE 8025
CMD ["python", "app.py"]
