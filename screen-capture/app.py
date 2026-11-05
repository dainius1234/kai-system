"""Screen Capture service — captures screenshots and extracts text via OCR.

Designed to work without GPU: uses mss for screen capture and
pytesseract (Tesseract OCR) for text extraction. In container mode,
captures are simulated from a configurable watch directory.

Endpoints:
  /health            - service health check
  /capture           - take a screenshot and return OCR text
  /capture/file      - OCR a specific image file from disk
  /metrics           - error budget snapshot
"""
from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from common.degraded import record_degradation
from common.http_hygiene import pooled_client
from common.runtime import ErrorBudget, detect_device, sanitize_string, setup_json_logger

logger = setup_json_logger("screen-capture", os.getenv("LOG_PATH", "/tmp/screen-capture.json.log"))
DEVICE = detect_device()
logger.info("Running on %s.", DEVICE)

app = FastAPI(title="Screen Capture", version="0.3.0")
budget = ErrorBudget(window_seconds=300)

MEMU_URL = os.getenv("MEMU_URL", "http://memu-core:8001")
WATCH_DIR = Path(os.getenv("SCREEN_WATCH_DIR", "/tmp/screen-captures"))
WATCH_DIR.mkdir(parents=True, exist_ok=True)
OCR_ENABLED = os.getenv("OCR_ENABLED", "true").lower() == "true"

# try to load optional dependencies
_mss_available = False
_tesseract_available = False
_pil_available = False

try:
    import mss  # noqa: F401
    _mss_available = True
except ImportError:
    logger.info("mss not available — screen capture will use file-based mode")

try:
    import pytesseract  # noqa: F401
    pytesseract.get_tesseract_version()  # verify the binary is present, not just the Python package
    _tesseract_available = True
except Exception:
    logger.info("pytesseract/tesseract not available — OCR will be simulated")

try:
    from PIL import Image  # noqa: F401
    _pil_available = True
except ImportError:
    logger.info("Pillow not available — image processing limited")


class CaptureResult(BaseModel):
    status: str
    text: str
    source: str
    timestamp: float
    ocr_available: bool
    image_b64: Optional[str] = None  # base64-encoded thumbnail if available


def _ocr_image_bytes(img_bytes: bytes) -> str:
    """Run OCR on raw image bytes. Falls back to stub if dependencies missing."""
    if _tesseract_available and _pil_available:
        try:
            from PIL import Image  # noqa: F811
            import pytesseract  # noqa: F811
            img = Image.open(io.BytesIO(img_bytes))
            return pytesseract.image_to_string(img).strip()
        except Exception as e:
            logger.warning("OCR failed: %s", e)
            return "[OCR error]"
    return "[OCR unavailable — install pytesseract + Pillow + tesseract-ocr]"


def _capture_screen() -> tuple[bytes, str]:
    """Capture screen and return (image_bytes, source_description)."""
    if _mss_available:
        import mss  # noqa: F811
        with mss.mss() as sct:
            monitor = sct.monitors[0]  # all monitors combined
            screenshot = sct.grab(monitor)
            # convert to PNG bytes
            if _pil_available:
                from PIL import Image  # noqa: F811
                img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                return buf.getvalue(), f"screen-{monitor['width']}x{monitor['height']}"
            return screenshot.raw, f"screen-{monitor['width']}x{monitor['height']}"

    # fallback: check watch directory for latest image
    images = sorted(WATCH_DIR.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    if images:
        return images[0].read_bytes(), f"file:{images[0].name}"

    raise HTTPException(status_code=503, detail="No screen capture backend available and no images in watch dir")


@app.get("/health")
async def health() -> Dict[str, str]:
    return {
        "status": "ok",
        "service": "screen-capture",
        "device": DEVICE,
        "mss_available": str(_mss_available),
        "tesseract_available": str(_tesseract_available),
        "ocr_enabled": str(OCR_ENABLED),
        "watch_dir": str(WATCH_DIR),
    }


@app.get("/metrics")
async def metrics() -> Dict[str, Any]:
    return budget.snapshot()


@app.post("/capture", response_model=CaptureResult)
async def capture() -> CaptureResult:
    """Capture the current screen, OCR it, and return the text."""
    try:
        img_bytes, source = _capture_screen()
    except HTTPException:
        raise
    except Exception as e:
        logger.error("screen capture failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Capture failed: {e}")

    text = _ocr_image_bytes(img_bytes) if OCR_ENABLED else "[OCR disabled]"

    # optionally auto-memorize to memu-core
    if os.getenv("AUTO_MEMORIZE", "false").lower() == "true" and text.strip():
        try:
            import httpx
            async with pooled_client(timeout=5.0) as client:
                await client.post(f"{MEMU_URL}/memory/memorize", json={
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "event_type": "screen_capture",
                    "result_raw": sanitize_string(text[:2000]),
                    "user_id": "screen-capture",
                })
        except Exception as e:
        # A failed WRITE is not a failed read. A read that fails degrades
        # and can be retried; this loses the observation permanently, and
        # the only trace was a warning line nobody reads.
        #
        # Verified 2026-08-07: this call has NEVER SUCCEEDED in any
        # profile. This service is on `sensor-net`, which is a sealed
        # island — not one of its members is attached to any other
        # network — and memu-core is on agent-net/data-net. Docker's DNS
        # only resolves names on a joined network, so every one of these
        # POSTs has failed at name resolution for the life of the
        # project, been swallowed here, and the service reported healthy.
        #
        # The topology is almost certainly RIGHT: perception is the least
        # trusted surface in the system (microphone, camera, screen,
        # clipboard, files) and a direct route from it to the memory
        # store would be the most valuable pivot available. The fix is a
        # mediated, write-only path, not a new network edge — see
        # kai-pm/DEEPSEEK_BRIEF_2026-08-07_TOPOLOGY.md.
        #
        # Until that exists this at least stops the loss being silent:
        # record_degradation surfaces it in the degradation report rather
        # than only in a log line.
            record_degradation("screen-capture", "auto_memorize", e, logger=logger)

    # save capture to disk
    ts = int(time.time())
    out_path = WATCH_DIR / f"capture_{ts}.png"
    try:
        out_path.write_bytes(img_bytes)
    except Exception as e:
        logger.warning("failed to save capture: %s", e)

    return CaptureResult(
        status="ok",
        text=sanitize_string(text[:5000]),
        source=source,
        timestamp=time.time(),
        ocr_available=_tesseract_available,
    )


@app.post("/capture/file")
async def capture_file(file: UploadFile = File(...)):
    """OCR an uploaded image file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    img_bytes = await file.read()
    if len(img_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(status_code=413, detail="File too large (max 10MB)")

    text = _ocr_image_bytes(img_bytes) if OCR_ENABLED else "[OCR disabled]"

    return JSONResponse({
        "status": "ok",
        "text": sanitize_string(text[:5000]),
        "source": f"upload:{file.filename}",
        "timestamp": time.time(),
        "ocr_available": _tesseract_available,
    })


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        budget.record(response.status_code)
        return response
    except Exception:
        budget.record(500)
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8059")))
