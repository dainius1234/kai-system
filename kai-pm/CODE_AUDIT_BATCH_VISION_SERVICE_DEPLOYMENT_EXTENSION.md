# Kai Code Audit — Vision Service Deployment and Policy Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_VISION_SERVICE.md` or `CODE_AUDIT_BATCH_VISION_SERVICE_EXTENSION.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-VISIOND-001 | HIGH | The configured `VISION_BACKEND` value is never read or enforced |
| KAI-VISIOND-002 | HIGH | Installing/importing DeepFace automatically activates emotion inference even when deployment selects `opencv` |
| KAI-VISIOND-003 | HIGH | Vision has unrestricted network egress for implicit model/weight acquisition |
| KAI-VISIOND-004 | HIGH | No immutable offline model bundle or persistent governed model-cache volume exists |
| KAI-VISIOND-005 | HIGH | Face/emotion responses define no strict response models or versioned schema |
| KAI-VISIOND-006 | HIGH | Sensitive biometric responses lack `Cache-Control: no-store` and equivalent privacy headers |
| KAI-VISIOND-007 | HIGH | Emotion inference cannot be independently disabled while retaining face-presence detection |
| KAI-VISIOND-008 | HIGH | Replayed identical frames always consume full decode/detection/emotion work |
| KAI-VISIOND-009 | HIGH | Precise face boxes and complete emotion distributions are returned without purpose minimisation |
| KAI-VISIOND-010 | HIGH | No model-performance or calibration metadata accompanies results consumed as presence/emotion evidence |
| KAI-VISIOND-011 | MEDIUM | Public metrics expose service reliability data without administrative authentication |
| KAI-VISIOND-012 | MEDIUM | TensorFlow/DeepFace CPU threads, memory use and parallelism are not bounded by service policy |
| KAI-VISIOND-013 | MEDIUM | The Python base image and APT packages are not pinned by immutable digest |
| KAI-VISIOND-014 | MEDIUM | Vision is deployed in the minimal topology but omitted from the full topology |
| KAI-VISIOND-015 | MEDIUM | Host port 8023 identifies Vision in the minimal topology and Agentic Introspection in the full topology |
| KAI-VISIOND-016 | MEDIUM | Dashboard depends on Vision only reaching `service_started`, not verified readiness |
| KAI-VISIOND-017 | MEDIUM | No service-side frame replay/frequency policy limits Dashboard’s automatic five-second camera feed |
| KAI-VISIOND-018 | MEDIUM | No explicit CPU/GPU device or deterministic-inference policy is configured or reported |

---

## High-severity findings

### KAI-VISIOND-001 — HIGH — Backend selection is cosmetic
**Issue:** `docker-compose.minimal.yml` sets `VISION_BACKEND`, but `perception/vision/app.py` never reads the variable.  
**Risk:** Operators cannot reliably select, disable or audit the active biometric/emotion backend through the deployment contract.  
**Recommendation:** Validate a strict backend enum at startup and make readiness depend on the selected implementation.  
**Status:** OPEN

### KAI-VISIOND-002 — HIGH — Emotion inference activates implicitly
Because DeepFace is installed in requirements, successful import sets `_DEEPFACE_OK=True`; every face-bearing `/analyze/frame` call then attempts emotion inference even when the declared backend is `opencv`.

### KAI-VISIOND-003 — HIGH — Ungoverned model egress
The container has ordinary network access and no egress allowlist. Lazy DeepFace model resolution can contact external artefact sources from a biometric-processing service.

### KAI-VISIOND-004 — HIGH — No governed model bundle/cache
The image does not contain a reviewed immutable weight bundle, and Compose mounts no dedicated signed/persistent model cache. Runtime artefacts may vary across rebuilds, restarts and replicas.

### KAI-VISIOND-005 — HIGH — Unversioned output contract
Health, presence and frame endpoints return ordinary dictionaries without FastAPI response models, strict emotion enums or an API-schema revision.

### KAI-VISIOND-006 — HIGH — Cacheable biometric output
Responses containing precise face locations, face count and inferred emotional scores do not set privacy-oriented cache headers.

### KAI-VISIOND-007 — HIGH — No independent emotion kill switch
The unused backend setting and lack of a feature/policy gate mean presence detection cannot remain available while sensitive emotion inference is reliably disabled.

### KAI-VISIOND-008 — HIGH — Replay-amplified inference
The service computes a full analysis for every request and has no frame digest, deduplication window, result cache or caller idempotency key.

### KAI-VISIOND-009 — HIGH — Excess biometric disclosure
The frame endpoint always returns all detector coordinates and the complete model emotion map, even when a consumer requires only a Boolean presence result.

### KAI-VISIOND-010 — HIGH — No calibration contract
Downstream consumers receive `present`, face boxes and emotional scores without detector/model validation revision, measured error rates, operating conditions or known limitations.

---

## Medium-severity findings

### KAI-VISIOND-011 — MEDIUM — Public metrics
`GET /metrics` requires no administrative identity and exposes internal request/error behaviour.

### KAI-VISIOND-012 — MEDIUM — Unbounded ML runtime resources
TensorFlow/DeepFace thread pools, CPU usage and memory growth are not configured, isolated or reported by service policy.

### KAI-VISIOND-013 — MEDIUM — Mutable build foundation
`python:3.11-slim` and APT packages are not digest-locked, so the effective native/ML runtime can change across builds.

### KAI-VISIOND-014 — MEDIUM — Topology-specific service disappearance
`vision-service` exists in `docker-compose.minimal.yml` but is absent from `docker-compose.full.yml`, while common UI documentation and code still describe the capability.

### KAI-VISIOND-015 — MEDIUM — Port identity collision across topologies
Host port 8023 maps to Vision in minimal Compose and Agentic Introspection in full Compose, creating conflicting firewall, monitoring and operator assumptions.

### KAI-VISIOND-016 — MEDIUM — Startup is not readiness
Dashboard waits only for `service_started`, so browser camera frames can arrive before model/cascade inference is verified usable.

### KAI-VISIOND-017 — MEDIUM — No frame-frequency contract
The Dashboard client can upload a frame every five seconds. Vision has no minimum interval, duplicate-frame suppression or per-source frame budget beyond the already-logged general lack of rate limiting.

### KAI-VISIOND-018 — MEDIUM — Implicit execution device
No CPU/GPU selection, deterministic mode, model-device provenance or accelerator-memory policy is configured or returned.

---

## Batch totals

- Findings: **18**
- Critical: **0**
- High: **10**
- Medium: **8**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,518**
- Critical: **191**
- High: **1,261**
- Medium: **1,063**
- Low: **3**

## Files materially reviewed

`perception/vision/app.py`, `perception/vision/Dockerfile`, `perception/vision/requirements.txt`, Vision and Dashboard definitions in `docker-compose.minimal.yml`, the full-stack port/service topology, Dashboard Vision proxy/client behaviour, and both prior Vision audit batches.
