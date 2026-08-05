FROM python:3.11-slim

WORKDIR /app

# Root-relative, because every profile builds this with `context: .`.
# `COPY . .` under that context would copy the entire repository into
# the image — every test, every kai-pm document, the .git directory —
# which is a supply-chain and image-size problem as much as a build one.
COPY vault-sync/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY vault-sync/ ./
COPY common/ ./common/

EXPOSE 8047

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8047"]
