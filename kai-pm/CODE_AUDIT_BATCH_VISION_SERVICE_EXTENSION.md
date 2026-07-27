# Kai Code Audit — Vision Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_VISION_SERVICE.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-VISIONX-001 | HIGH | OpenCV readiness is set true without verifying that the face-cascade classifier loaded successfully |
| KAI-VISIONX-002 | HIGH | DeepFace readiness is based only on import success and does not warm or verify the emotion model |
| KAI-VISIONX-003 | HIGH | First emotion request can trigger unpinned runtime model-weight acquisition and initialisation |
| KAI-VISIONX-004 | HIGH | Concurrent first emotion requests can race heavyweight model initialisation |
| KAI-VISIONX-005 | HIGH | Per-request DeepFace failure is silently converted to no emotion while the backend still claims `+deepface` |
| KAI-VISIONX-006 | HIGH | Emotion labels and scores are accepted without range, finiteness or allowed-label validation |
| KAI-VISIONX-007 | HIGH | Face-detection results have no detector score or minimum-confidence evidence |
| KAI-VISIONX-008 | HIGH | Image content is not restricted to JPEG or PNG despite the public error contract |
| KAI-VISIONX-009 | HIGH | EXIF orientation and image-coordinate transformations are not represented in returned face boxes |
| KAI-VISIONX-010 | HIGH | Face results have no image digest, frame ID, capture time, source device or model revision |
| KAI-VISIONX-011 | HIGH | The service has no consent, retention or processing-purpose evidence for biometric/emotion inference |
| KAI-VISIONX-012 | HIGH | Multiple workers independently load large vision models and return inconsistent instantaneous results |
| KAI-VISIONX-013 | HIGH | Face-result cardinality and response size are not capped after detection |
| KAI-VISIONX-014 | MEDIUM | Public health reveals biometric and emotion capability availability |
| KAI-VISIONX-015 | MEDIUM | `sys.path` is mutated at import using a deployment-dependent parent path |
| KAI-VISIONX-016 | MEDIUM | DeepFace import catches every exception and discards the actual readiness failure reason |
| KAI-VISIONX-017 | MEDIUM | OpenCV initialisation catches only `ImportError`, not cascade-file or native-library initialisation failures |
| KAI-VISIONX-018 | MEDIUM | Face boxes omit source image dimensions and cannot be interpreted safely after resizing/cropping |
| KAI-VISIONX-019 | MEDIUM | Presence and frame endpoints duplicate full decode/detection work instead of sharing one bounded analysis result |
| KAI-VISIONX-020 | MEDIUM | Responses contain no inference latency, queue delay or degraded-path indicator |
| KAI-VISIONX-021 | MEDIUM | Dependency ranges for OpenCV, NumPy, DeepFace, TensorFlow/Keras and FastAPI are not reproducibly pinned |
| KAI-VISIONX-022 | MEDIUM | Missing shared-runtime imports silently replace structured telemetry with no-op fallbacks |
| KAI-VISIONX-023 | MEDIUM | No immutable audit links caller, image digest, model versions, detected faces and returned emotion |
| KAI-VISIONX-024 | MEDIUM | The service has no lifespan-owned model warm-up, bounded inference pool or graceful in-flight shutdown |

---

## High-severity findings

### KAI-VISIONX-001 — HIGH — Cascade readiness is assumed
**Issue:** `_OPENCV_OK=True` is set after constructing `cv2.CascadeClassifier`, but the code never checks `_FACE_CASCADE.empty()`.  
**Risk:** A missing/corrupt XML cascade is reported ready until `detectMultiScale()` fails during a user request.  
**Recommendation:** validate the classifier at startup with a controlled test and fail face-detection readiness when empty/unusable.  
**Status:** OPEN

### KAI-VISIONX-002 — HIGH — DeepFace import is not model readiness
Import success sets `_DEEPFACE_OK=True`; no model construction, weights availability or test inference occurs before health reports readiness.

### KAI-VISIONX-003 — HIGH — Runtime model acquisition
Broad lower-bound DeepFace/TensorFlow dependencies and lazy first analysis can resolve/download model weights during a request without an immutable artefact digest.

### KAI-VISIONX-004 — HIGH — First-use model race
No application lock/single-flight mechanism protects DeepFace’s internal lazy model build across concurrent requests.

### KAI-VISIONX-005 — HIGH — Emotion failure is hidden
`_detect_emotion()` catches all exceptions and returns `None`; the endpoint still reports backend `opencv+deepface`, making unavailable/failed inference look like an ordinary no-result.

### KAI-VISIONX-006 — HIGH — Unvalidated emotion output
Dominant labels are arbitrary backend strings. Score values are converted/rounded but not checked for NaN, infinity, negative values, percentages versus fractions or a known label set.

### KAI-VISIONX-007 — HIGH — Face evidence lacks confidence
Haar detections are returned as facts with no detector weight, false-positive threshold or calibrated confidence.

### KAI-VISIONX-008 — HIGH — Decoder accepts broader formats
`cv2.imdecode()` accepts many encoded image formats; the service performs no media-type or magic-byte allowlist even though errors instruct callers to send JPEG/PNG.

### KAI-VISIONX-009 — HIGH — Orientation ambiguity
The service does not preserve or report EXIF orientation/rotation handling, so returned coordinates may not match the operator-visible image orientation.

### KAI-VISIONX-010 — HIGH — Missing frame provenance
Responses cannot be tied to exact bytes, capture/source time, camera/browser session, device identity or model/cascade revision.

### KAI-VISIONX-011 — HIGH — Missing biometric governance
No authenticated consent revision, processing purpose, retention/deletion policy or affected-person identity accompanies face/emotion analysis.

### KAI-VISIONX-012 — HIGH — Replica model and inference divergence
Each worker separately imports/initialises models and processes frames; results, readiness and model caches may differ while callers receive one undifferentiated service identity.

### KAI-VISIONX-013 — HIGH — Unbounded result cardinality
Every Haar box is serialised. A pathological image/parameter configuration can generate a large list and expensive DeepFace processing without a face-count cap.

---

## Medium-severity findings

### KAI-VISIONX-014 — MEDIUM — Public capability disclosure
Health publicly reveals whether OpenCV and DeepFace are installed/available.

### KAI-VISIONX-015 — MEDIUM — Import-path mutation
The source inserts a parent path at index zero. In the flattened Docker image that resolved parent differs from the source tree, and global module resolution is modified unnecessarily.

### KAI-VISIONX-016 — MEDIUM — DeepFace failure provenance lost
The broad import exception is discarded and only a generic disabled log message remains.

### KAI-VISIONX-017 — MEDIUM — Incomplete OpenCV startup handling
Native-library/cascade construction errors beyond ImportError can crash startup or surface later without a typed readiness state.

### KAI-VISIONX-018 — MEDIUM — Coordinate context missing
Face boxes omit frame width/height, scaling and coordinate-space version.

### KAI-VISIONX-019 — MEDIUM — Duplicate analysis paths
Presence and full-frame endpoints each read, decode, grayscale and run the detector independently for the same submitted frame.

### KAI-VISIONX-020 — MEDIUM — Missing performance provenance
Responses do not identify inference time, backend/model fallback, queue depth or whether emotion failed.

### KAI-VISIONX-021 — MEDIUM — Non-reproducible ML/runtime dependencies
Several critical packages use `>=` ranges and no lockfile hashes.

### KAI-VISIONX-022 — MEDIUM — Silent telemetry downgrade
If shared runtime imports fail, basic logging/no-op metrics are used while health remains normal.

### KAI-VISIONX-023 — MEDIUM — Missing inference audit
No tamper-evident event binds principal/consent, input digest, model/cascade versions and output.

### KAI-VISIONX-024 — MEDIUM — Missing model lifecycle ownership
No lifespan performs warm-up/readiness validation, owns a bounded executor, drains inference or closes model resources.

---

## Batch totals

- Findings: **24**
- Critical: **0**
- High: **13**
- Medium: **11**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,460**
- Critical: **189**
- High: **1,233**
- Medium: **1,035**
- Low: **3**

## Files materially reviewed

`perception/vision/app.py`, `perception/vision/Dockerfile`, `perception/vision/requirements.txt` and the existing Vision Service audit.
