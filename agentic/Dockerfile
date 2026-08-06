FROM python:3.11-slim

WORKDIR /app

COPY agentic/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

# The whole directory, not a list of ten files.
#
# This was ten hand-written COPY lines against a directory holding
# thirty-seven modules, and `app.py` imports twenty-seven of the ones
# the list omitted. The container died at import on every boot:
#
#   File "/app/app.py", line 27, in <module>
#       from system_fsm import KaiEvent as SysEvent, ...
#   ModuleNotFoundError: No module named 'system_fsm'
#
# It never blocked anything until today, because nothing had ever got
# far enough to start it. The list-beside-the-thing pattern, fourteenth
# venue, and the same remedy every time: state the denominator, and
# derive it from the tree rather than from a list somebody has to
# remember to update.
#
# `.dockerignore` keeps `__pycache__` and the caches out.
COPY agentic/ ./
COPY common/ ./common/
COPY data/ ./data/

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8007/health')" || exit 1

USER app

CMD ["python", "app.py"]
