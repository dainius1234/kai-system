FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" app && \
    apt-get update && apt-get install -y --no-install-recommends libnotify-bin && \
    rm -rf /var/lib/apt/lists/*

# COPY paths are relative to the build CONTEXT, and every profile
# builds this with `context: .` (the repository root). The bare
# `COPY requirements.txt` and the `COPY ../../common` below were
# written for a context of this directory — and `../..` escapes
# the context, which Docker rejects under ANY context. So this
# image could not build in any configuration. Nothing noticed
# because no CI step built this profile until 2026-08-05.
COPY output/notify/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY output/notify/app.py ./
COPY common/ ./common/

USER app
ENV PORT=8031
EXPOSE 8031
CMD ["python", "app.py"]
