FROM python:3.11-slim

WORKDIR /app

COPY memu-core/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

RUN addgroup --system app && adduser --system --ingroup app app

# The TurboVec index and its writer lock live here, and this container
# runs as `app`. Docker seeds a fresh named volume from the image's
# directory — contents *and* ownership — so the mountpoint has to exist
# and be owned by `app` before the volume is created, or `/data/turbovec`
# arrives owned by root and the service cannot write to it:
#
#   PermissionError: [Errno 13] Permission denied: '/data/turbovec/.writer.lock'
#
# Nobody had hit this because nobody had got here: the writer branch
# died on an unbound `socket` before it ever tried to write. Two layers
# of never-executed code, one behind the other.
RUN mkdir -p /data/turbovec && chown -R app:app /data

COPY memu-core/app.py ./
COPY memu-core/lakefs_client.py ./
COPY common/ ./common/
COPY security/ ./security/

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

USER app

CMD ["python", "app.py"]
