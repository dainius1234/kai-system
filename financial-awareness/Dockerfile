FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY financial-awareness/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN addgroup --system app && adduser --system --ingroup app app
COPY common/ ./common/
COPY security/ ./security/
COPY financial-awareness/app.py ./app.py
RUN mkdir -p /data/finance && chown -R app:app /data/finance
ENV FINANCE_ROOT=/data/finance
USER app
EXPOSE 8063
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8063/health')" || exit 1
CMD ["python", "app.py"]
