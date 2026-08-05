FROM python:3.11-slim
WORKDIR /app
RUN adduser --disabled-password --gecos "" app

# COPY paths are relative to the build CONTEXT, and every profile
# builds this with `context: .` (the repository root). The bare
# `COPY requirements.txt` and the `COPY ../../common` below were
# written for a context of this directory — and `../..` escapes
# the context, which Docker rejects under ANY context. So this
# image could not build in any configuration. Nothing noticed
# because no CI step built this profile until 2026-08-05.
COPY perception/files/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY perception/files/app.py ./
COPY common/ ./common/

USER app
ENV PORT=8025
EXPOSE 8025
CMD ["python", "app.py"]
