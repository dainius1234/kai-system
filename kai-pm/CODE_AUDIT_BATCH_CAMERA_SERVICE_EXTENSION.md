# Kai Code Audit — Camera Service Extension Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

This batch records only findings not already present in `CODE_AUDIT_BATCH_CAMERA_SERVICE.md`.

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CAMX-001 | HIGH | The deployed Camera image installs none of OpenCV, NumPy or MSS, so every real capture/analysis path is unavailable |
| KAI-CAMX-002 | HIGH | Compose mounts no physical camera device into the Camera container |
| KAI-CAMX-003 | HIGH | Compose provides no X11, Wayland or equivalent display/socket for screen capture |
| KAI-CAMX-004 | HIGH | The Camera healthcheck remains green in the exact deployment where all capture dependencies are absent |
| KAI-CAMX-005 | HIGH | Supervisor’s configured `CAMERA_URL` is ignored because Camera is absent from the live service registry |
| KAI-CAMX-006 | HIGH | Supervisor declares a startup dependency on Camera while never monitoring or recovering it |
| KAI-CAMX-007 | HIGH | The default audio-service hostname is wrong (`perception-audio` instead of `audio-service`) |
| KAI-CAMX-008 | HIGH | The default TTS destination uses the wrong service name and port (`tts:8022` instead of `tts-service:8030`) |
| KAI-CAMX-009 | HIGH | The default Agentic URL uses port 8000 while Agentic listens on port 8007 |
| KAI-CAMX-010 | HIGH | Camera sends `message` to Agentic `/run`, but the endpoint requires `user_input` and `session_id` |
| KAI-CAMX-011 | HIGH | Camera reads a nonexistent `response` field from Agentic’s `GraphResponse` schema |
| KAI-CAMX-012 | HIGH | The claimed LLM-enhanced sensor-fusion path is therefore non-functional in the deployed stack |
| KAI-CAMX-013 | HIGH | Agentic transport, validation and schema failures silently become ordinary heuristic results |
| KAI-CAMX-014 | HIGH | A misconfigured service returning HTTP 200 can be labelled `fusion_mode=llm` with an empty interpretation |
| KAI-CAMX-015 | HIGH | Proactive request dictionaries have no aggregate byte, nesting or field-count limits |
| KAI-CAMX-016 | HIGH | Sensor numeric fields accept strings, booleans, NaN, infinity and extreme values without a typed schema |
| KAI-CAMX-017 | HIGH | Proactive decisions have no authenticated sensor, device, operator, session or source-event provenance |
| KAI-CAMX-018 | HIGH | Camera consumes the globally latest audio transcript without a maximum age or freshness requirement |
| KAI-CAMX-019 | HIGH | Camera audio selection is not partitioned by user, microphone or session |
| KAI-CAMX-020 | HIGH | Camera ignores the transcript’s injection-detected or blocked state when consuming emotion metadata |
| KAI-CAMX-021 | HIGH | A latest transcript without emotion hides an earlier usable audio observation without exposing missing evidence |
| KAI-CAMX-022 | HIGH | Audio-service failure is converted into a normal neutral observation rather than sensor unavailability |
| KAI-CAMX-023 | HIGH | Screen-capture failure in `/proactive/auto` is converted into a normal brightness/motion observation |
| KAI-CAMX-024 | HIGH | Complete audio and video source failure can still return `status=ok` and a normal heuristic decision |
| KAI-CAMX-025 | HIGH | Tool Gate approval covers only generic `speak`, not the exact text, voice or TTS destination later used |
| KAI-CAMX-026 | HIGH | Camera uses a predictable Gate session token committed in the repository and enabled by the default Compose value |
| KAI-CAMX-027 | HIGH | Anonymous `/proactive/auto` callers can cause Camera to use its server-held HMAC identity as `perception-camera` |
| KAI-CAMX-028 | HIGH | Gate decisions are reduced to a Boolean and discard decision ID, ledger hash, policy version and rationale |
| KAI-CAMX-029 | HIGH | `bool(approved)` treats non-empty strings such as `"false"` as approval if the Gate response schema drifts |
| KAI-CAMX-030 | HIGH | Speech requests have no idempotency key, so retries or concurrent triggers can duplicate output |
| KAI-CAMX-031 | MEDIUM | `VideoCapture.open/read` has no operation deadline or cancellation boundary |
| KAI-CAMX-032 | MEDIUM | Camera resources are not released through a `finally` block if frame reading raises |
| KAI-CAMX-033 | MEDIUM | Concurrent requests can open the same physical camera without a capture mutex |
| KAI-CAMX-034 | MEDIUM | Captured frame dimensions, dtype and memory footprint are not bounded before NumPy/OpenCV processing |
| KAI-CAMX-035 | MEDIUM | Screen capture always selects monitor index 1 and has no governed monitor identity |
| KAI-CAMX-036 | MEDIUM | Camera and legacy history entries omit a source identifier while screen entries include one |
| KAI-CAMX-037 | MEDIUM | Analysis timestamps use processing wall-clock time, not a capture-device timestamp or monotonic sequence |
| KAI-CAMX-038 | MEDIUM | Camera-frame brightness can generate a message claiming the “screen” is dark |
| KAI-CAMX-039 | MEDIUM | Edge density is labelled `screen_activity` even when analysing a physical-camera frame |
| KAI-CAMX-040 | MEDIUM | Detailed audio score dictionaries are accepted but ignored by a no-op `get()` call |
| KAI-CAMX-041 | MEDIUM | `VIRTUAL_DEVICE`, `CAPTURE_DIR` and `MEMU_URL` are exposed/configured but unused by the implementation |
| KAI-CAMX-042 | MEDIUM | Health encodes capability booleans as strings, allowing `"False"` to be treated as truthy by weak consumers |
| KAI-CAMX-043 | MEDIUM | Health omits MSS/display, Tool Gate, audio, Agentic and TTS readiness |
| KAI-CAMX-044 | MEDIUM | Every audio, Gate, Agentic and TTS operation creates a new HTTP client and connection pool |
| KAI-CAMX-045 | MEDIUM | Camera has no circuit breakers, shared deadlines or dependency-freshness state for downstream calls |
| KAI-CAMX-046 | MEDIUM | The service has no structured capture/decision/approval/delivery audit trail or correlation ID |
| KAI-CAMX-047 | MEDIUM | Dependency versions use broad lower bounds and the Python base image is not pinned by immutable digest |
| KAI-CAMX-048 | MEDIUM | The test suite explicitly accepts a 503 capture path and tests neither deployed capture capability nor proactive integrations |

---

## High-severity findings

### KAI-CAMX-001 — HIGH — Capture dependencies absent from the image
**Issue:** `perception/camera/requirements.txt` installs only FastAPI, Starlette, Uvicorn and HTTPX. The application’s OpenCV, NumPy and MSS imports therefore fail in the normal image.  
**Risk:** Camera capture returns 503; screen capture returns 503; analysis cannot operate, while health and orchestration still treat the service as healthy.  
**Recommendation:** Build a reproducible image containing the required capture libraries and make successful capability probes mandatory for readiness.  
**Status:** OPEN

### KAI-CAMX-002 — HIGH — Physical device not mounted
Compose provides no `/dev/video*` device mapping. Even after dependencies are installed, the configured `/dev/video0` is unavailable in the container.

### KAI-CAMX-003 — HIGH — Display capture not connected
Compose provides no display environment, X11/Wayland socket or portal. MSS cannot access the operator’s screen.

### KAI-CAMX-004 — HIGH — Deployment-specific false health
The Docker healthcheck calls `/health`, which always reports `ok` and merely stringifies optional import flags. This exact broken image remains Docker-healthy.

### KAI-CAMX-005 — HIGH — Supervisor ignores Camera configuration
Compose supplies `CAMERA_URL`, but `supervisor/app.py` never reads it into `SERVICES`; Camera is absent unless separately added through `SUPERVISOR_EXTRA_SERVICES`.

### KAI-CAMX-006 — HIGH — Startup coupling without operational coverage
Supervisor `depends_on` Camera, but does not include it in sweeps, breakers, fleet history or recovery actions.

### KAI-CAMX-007 — HIGH — Wrong audio DNS name
Camera defaults to `http://perception-audio:8021`; the Compose service is `audio-service`. Camera Compose does not override `AUDIO_URL`.

### KAI-CAMX-008 — HIGH — Wrong TTS service and port
Camera defaults to `http://tts:8022`; the deployed service is `tts-service` on 8030. Camera Compose does not override `TTS_URL`.

### KAI-CAMX-009 — HIGH — Wrong Agentic port
Camera defaults to `http://agentic:8000`, while the deployed Agentic service listens on 8007.

### KAI-CAMX-010 — HIGH — Agentic request schema mismatch
Camera posts `{"message": ...}`. Agentic `/run` requires `user_input` and `session_id`, so a correctly reached request is rejected with 422.

### KAI-CAMX-011 — HIGH — Agentic response schema mismatch
Agentic `GraphResponse` contains `specialist`, `plan` and optional `gate_decision`; Camera reads `data.get("response", "")`.

### KAI-CAMX-012 — HIGH — Non-functional fusion feature
The combined URL, request and response mismatches mean the LLM-fusion feature cannot produce its claimed interpretation in the deployed stack.

### KAI-CAMX-013 — HIGH — Hidden downgrade
Every Agentic exception/status/schema failure is swallowed and returned as `status=ok`, `fusion_mode=heuristic`.

### KAI-CAMX-014 — HIGH — False LLM-success state
Any endpoint at the configured URL returning HTTP 200 JSON without `response` produces `fusion_mode=llm` and an empty `llm_interpretation`.

### KAI-CAMX-015 — HIGH — Unbounded nested sensor bodies
The proactive endpoints use raw `Dict[str, Any]` body fields with no total body, depth, list or field-count limit.

### KAI-CAMX-016 — HIGH — Unsafe sensor numerics
Comparisons and arithmetic operate directly on unvalidated values. Strings raise 500; booleans behave as numbers; non-finite/extreme values distort urgency or JSON.

### KAI-CAMX-017 — HIGH — Missing sensor provenance
The same raw body format represents physical sensors, test data, Dashboard calls and arbitrary network clients. No signed device/session/source event accompanies a decision.

### KAI-CAMX-018 — HIGH — Stale audio accepted
`/proactive/auto` takes the last transcript regardless of its age.

### KAI-CAMX-019 — HIGH — Cross-user audio selection
Audio’s global buffer is not filtered by principal, session or microphone before Camera uses its emotion.

### KAI-CAMX-020 — HIGH — Injection state ignored
The audio service may append a transcript even when injection is detected. Camera consumes only its `emotion` field and never checks the record status.

### KAI-CAMX-021 — HIGH — Latest-record masking
If the newest transcript lacks `emotion`, Camera keeps neutral defaults rather than searching for a fresh valid observation or declaring evidence unavailable.

### KAI-CAMX-022 — HIGH — Audio outage becomes neutral evidence
DNS, status, parsing and schema failures are silently replaced with a normal neutral audio state.

### KAI-CAMX-023 — HIGH — Video outage becomes normal evidence
`/proactive/auto` replaces every screen-capture failure with brightness 128, no motion and no-motion-detected false/normal fields.

### KAI-CAMX-024 — HIGH — Complete sensor outage remains successful
When both sources fail, the endpoint still returns an ordinary successful heuristic result with no degraded flags.

### KAI-CAMX-025 — HIGH — Approval is not bound to spoken content
The Gate request includes only tool identity and conviction. The message and voice sent to TTS are not in signed parameters or the approval record.

### KAI-CAMX-026 — HIGH — Predictable Gate token
`camera-gate-token-1` is committed in `security/trusted_tokens.txt` and is the Compose default unless explicitly replaced.

### KAI-CAMX-027 — HIGH — Anonymous-to-trusted deputy transition
An unauthenticated `/proactive/auto` request can cause Camera to sign a Gate request using its mounted HMAC secret and trusted actor identity.

### KAI-CAMX-028 — HIGH — Gate evidence discarded
Only the `approved` field survives. The result cannot prove which decision, policy revision or ledger entry authorised later TTS work.

### KAI-CAMX-029 — HIGH — Weak approval type handling
`bool(value)` is applied without a strict response schema. A non-empty string value is truthy even when its text is `false`.

### KAI-CAMX-030 — HIGH — Duplicate speech has no operation identity
Neither Gate nor TTS calls include an idempotency key linked to the sensor event.

---

## Medium-severity findings

### KAI-CAMX-031 — MEDIUM — No capture deadline
OpenCV device open/read is synchronous and has no explicit timeout or cancellation.

### KAI-CAMX-032 — MEDIUM — Incomplete resource cleanup
`cap.release()` is not guaranteed if `cap.read()` raises.

### KAI-CAMX-033 — MEDIUM — No capture serialisation
Concurrent requests can contend for one device and alter the shared motion baseline.

### KAI-CAMX-034 — MEDIUM — Unbounded frame workload
No maximum width, height, channels or bytes is enforced before full-frame copies, Canny and difference calculations.

### KAI-CAMX-035 — MEDIUM — Fixed monitor identity
MSS always uses `monitors[1]`; multi-monitor and headless states are not modelled or returned.

### KAI-CAMX-036 — MEDIUM — Inconsistent history provenance
Only `/capture/screen` adds `source=screen`; camera and legacy entries are indistinguishable.

### KAI-CAMX-037 — MEDIUM — Weak event chronology
The timestamp is generated during processing, not at device capture, and has no sequence/source clock.

### KAI-CAMX-038 — MEDIUM — Wrong-source message
The darkness nudge says the screen is dark even for a webcam frame.

### KAI-CAMX-039 — MEDIUM — Camera edges called screen activity
The same edge-density label is used regardless of source.

### KAI-CAMX-040 — MEDIUM — Detailed scores ignored
`audio_signals.get("scores", {})` has no assignment or effect; only selected top-level labels are used.

### KAI-CAMX-041 — MEDIUM — Dead configuration/interfaces
`VIRTUAL_DEVICE`, `CAPTURE_DIR` and `MEMU_URL` imply capabilities that are not used.

### KAI-CAMX-042 — MEDIUM — Boolean strings in health
`"False"` is a non-empty string and may be treated as true by weak clients.

### KAI-CAMX-043 — MEDIUM — Incomplete readiness model
Health omits display capture, actual device open, dependency connectivity, token/secret load and current model/action readiness.

### KAI-CAMX-044 — MEDIUM — HTTP connection churn
Every downstream operation builds and destroys its own HTTPX client.

### KAI-CAMX-045 — MEDIUM — No dependency resilience contract
There are no shared breakers, bounded retries, source freshness states or post-delivery checks.

### KAI-CAMX-046 — MEDIUM — No decision audit
The service retains only a volatile result history and does not record caller, source event, capture identity, Gate decision and TTS delivery as one operation.

### KAI-CAMX-047 — MEDIUM — Non-reproducible build inputs
Dependencies use broad lower bounds and `python:3.11-slim` is not pinned by digest.

### KAI-CAMX-048 — MEDIUM — Tests accept non-operation
The capture test skips on 503 and health asserts only `status=ok`; no test requires actual image capability, correct service URLs, Gate binding or TTS delivery.

---

## Batch totals

- Findings: **48**
- Critical: **0**
- High: **30**
- Medium: **18**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,437**
- Critical: **189**
- High: **1,223**
- Medium: **1,022**
- Low: **3**

## Files materially reviewed

`perception/camera/app.py`, `perception/camera/Dockerfile`, `perception/camera/requirements.txt`, `scripts/test_camera_service.py`, Camera deployment in `docker-compose.full.yml`, `security/trusted_tokens.txt`, and direct integrations with Audio, Agentic, Tool Gate, TTS and Supervisor.
