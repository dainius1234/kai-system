FROM python:3.11-slim

WORKDIR /app

COPY agentic/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

COPY agentic/adversary.py ./
COPY agentic/app.py ./
COPY agentic/conviction.py ./
COPY agentic/kai_config.py ./
COPY agentic/model_selector.py ./
COPY agentic/planner.py ./
COPY agentic/priority_queue.py ./
COPY agentic/router.py ./
COPY agentic/security_audit.py ./
COPY agentic/tree_search.py ./
COPY common/ ./common/

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8007/health')" || exit 1

USER app

CMD ["python", "app.py"]
