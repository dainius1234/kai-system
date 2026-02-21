from __future__ import annotations

import json
import os
import shutil
import urllib.request
from pathlib import Path

URL = "http://localhost:8080/go-no-go"
KEEPER_TPM_KEY_CTX = Path(os.getenv("KAI_TPM_SIGNING_KEY_CTX", "/etc/kai/keeper_signing.ctx"))

if shutil.which("tpm2_sign") is None or shutil.which("tpm2_verifysignature") is None:
    print("TPM not ready: tpm2-tools missing")
    raise SystemExit(1)

if not KEEPER_TPM_KEY_CTX.exists():
    print(f"TPM not ready: keeper key not sealed ({KEEPER_TPM_KEY_CTX})")
    raise SystemExit(1)

try:
    with urllib.request.urlopen(URL, timeout=3) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception:
    print("go_no_go failed: dashboard not running")
    raise SystemExit(1)

if payload.get("decision") != "GO":
    print(f"go_no_go failed: {payload}")
    raise SystemExit(1)

print("go_no_go passed")
