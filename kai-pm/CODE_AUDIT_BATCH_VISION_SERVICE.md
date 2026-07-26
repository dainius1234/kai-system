# Kai Code Audit — Vision Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-VISION-001 | HIGH | Face-location, presence and emotion analysis are exposed without authentication |
| KAI-VISION-002 | HIGH | Image uploads are fully read into memory without a size limit |
| KAI-VISION-003 | HIGH | CPU-intensive image and emotion analysis runs directly in async request handlers |
| KAI-VISION-004 | HIGH | Decoded image dimensions and decompression cost are not bounded |
| KAI-VISION-005 | MEDIUM | Emotion analysis uses `enforce_detection=False` and can classify non-face/background content |
| KAI-VISION-006 | MEDIUM | Multi-face emotion output retains only the first DeepFace result |
| KAI-VISION-007 | MEDIUM | Presence confidence is fabricated from face count rather than detector confidence |
| KAI-VISION-008 | MEDIUM | Health reports `ok` while the service is operating in stub mode |
| KAI-VISION-009 | MEDIUM | No rate limit or bounded inference concurrency is implemented |
| KAI-VISION-010 | MEDIUM | Error-budget recording passes a Boolean and omits raised exceptions |
| KAI-VISION-011 | MEDIUM | Face-size and port configuration are not validated |

---

## Vision service: `perception/vision/app.py`

### KAI-VISION-001 — HIGH — Unauthenticated biometric and emotion processing
**Issue:** `POST /analyze/frame` and `POST /analyze/presence` require no authentication or authorisation. They return face bounding boxes, face count, presence and inferred emotion scores for caller-supplied images.  
**Risk:** Any reachable caller can use the service as a biometric/emotion-analysis endpoint, consuming private images and producing sensitive behavioural inferences without consent, provenance or retention policy.  
**Recommendation:** Require authenticated, consent-bound sessions and restrict biometric processing to approved local workflows.  
**Status:** OPEN

### KAI-VISION-002 — HIGH — Uploads are unbounded before allocation
**Issue:** Both endpoints call `await file.read()` and impose no byte-size limit.  
**Risk:** Oversized or concurrent uploads can exhaust process memory before decoding begins.  
**Recommendation:** Enforce server-side request limits and stream with a strict byte counter.  
**Status:** OPEN

### KAI-VISION-003 — HIGH — Blocking inference runs on the event loop
**Issue:** OpenCV decoding, face detection and DeepFace analysis are synchronous CPU-intensive calls executed directly inside async handlers.  
**Risk:** One expensive frame blocks the event-loop worker; concurrent submissions can make health, metrics and all analysis endpoints unavailable.  
**Recommendation:** Use a bounded worker pool or inference queue with cancellation and timeouts.  
**Status:** OPEN

### KAI-VISION-004 — HIGH — Decoded image cost is not bounded
**Issue:** The service validates neither pixel dimensions nor decoded memory footprint before `cv2.imdecode`, grayscale conversion and DeepFace inference.  
**Risk:** Highly compressed large-dimension images can cause decompression-memory amplification and excessive CPU use despite modest upload size.  
**Recommendation:** Inspect headers safely, enforce maximum dimensions/pixels and reject pathological encodings before full decode.  
**Status:** OPEN

### KAI-VISION-005 — MEDIUM — Emotion inference can proceed without a detected face
**Issue:** `DeepFace.analyze` is called with `enforce_detection=False`. The service only gates the call on Haar-cascade detection somewhere in the frame, then passes the complete image rather than a verified face crop.  
**Risk:** Background, partial faces or false-positive cascade detections can produce confident-looking emotional classifications unrelated to the person.  
**Recommendation:** Analyse validated face crops and represent low-confidence/failed detection explicitly.  
**Status:** OPEN

### KAI-VISION-006 — MEDIUM — Multi-face emotion results are discarded
**Issue:** When DeepFace returns a list, only `result[0]` is retained, while all Haar-detected face boxes are returned.  
**Risk:** The response presents one dominant emotion beside multiple faces without identifying which face produced it, creating misleading attribution.  
**Recommendation:** Bind each emotion result to a specific validated face or omit ambiguous attribution.  
**Status:** OPEN

### KAI-VISION-007 — MEDIUM — Presence confidence is not detector confidence
**Issue:** `/analyze/presence` computes confidence as `min(1.0, len(faces) * 0.85)`. It does not use any confidence score from the detector. Two faces automatically produce 1.0.  
**Risk:** Consumers can treat an arbitrary face-count formula as a calibrated probability of human presence.  
**Recommendation:** Label it as a heuristic score or use a calibrated detector confidence with validation data.  
**Status:** OPEN

### KAI-VISION-008 — MEDIUM — Stub mode reports healthy
**Issue:** `/health` always returns `status: ok`, including when OpenCV is unavailable and both analysis endpoints return stub results asserting no presence.  
**Risk:** Orchestration treats a non-functional perception service as ready and may interpret stub negatives as real observations.  
**Recommendation:** Separate liveness from face/emotion capability readiness and mark stub output as unavailable rather than negative evidence.  
**Status:** OPEN

### KAI-VISION-009 — MEDIUM — No inference abuse controls
**Issue:** The service has no caller identity, rate limit, concurrent-job bound, timeout or queue depth around OpenCV and DeepFace operations.  
**Risk:** Repeated image submissions can saturate CPU and memory.  
**Recommendation:** Apply authenticated quotas and bounded inference concurrency.  
**Status:** OPEN

### KAI-VISION-010 — MEDIUM — Error-budget telemetry is incomplete
**Issue:** Middleware records `response.status_code >= 500`, passing a Boolean rather than an HTTP status code, and does not record exceptions raised before a response.  
**Risk:** Reliability metrics can misclassify or omit failures.  
**Recommendation:** Record actual status codes and exception outcomes consistently.  
**Status:** OPEN

### KAI-VISION-011 — MEDIUM — Configuration is not validated
**Issue:** `MIN_FACE_SIZE` and port are parsed directly without bounds. Zero/negative/extreme values can alter detector cost and behaviour or fail startup.  
**Risk:** Misconfiguration can disable useful detection, increase false positives or create excessive computation.  
**Recommendation:** Validate typed startup configuration with explicit safe ranges.  
**Status:** OPEN

---

## Batch totals

- Findings: **11**
- Critical: **0**
- High: **4**
- Medium: **7**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **387**
- Critical: **40**
- High: **152**
- Medium: **192**
- Low: **3**

## Files materially reviewed in this batch

`perception/vision/app.py`.
