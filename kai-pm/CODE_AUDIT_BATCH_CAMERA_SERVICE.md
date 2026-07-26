# Kai Code Audit — Camera Service Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending consolidation into `CODE_AUDIT_MASTER.md`  
Reviewed: 26 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-CAMERA-001 | CRITICAL | Unauthenticated callers can activate the physical camera |
| KAI-CAMERA-002 | CRITICAL | Unauthenticated callers can capture the current screen |
| KAI-CAMERA-003 | HIGH | Unauthenticated fabricated sensor inputs can manipulate proactive-speech state |
| KAI-CAMERA-004 | HIGH | Caller-controlled sensor summaries are sent into the agentic LLM path |
| KAI-CAMERA-005 | HIGH | Camera and screen capture plus image analysis run synchronously in async handlers |
| KAI-CAMERA-006 | HIGH | Failed screen capture is converted into a normal black dummy frame |
| KAI-CAMERA-007 | MEDIUM | Analysis history and device paths are exposed without authentication |
| KAI-CAMERA-008 | MEDIUM | Proactive cooldown is consumed before the tool gate approves speech |
| KAI-CAMERA-009 | MEDIUM | TTS delivery status is ignored |
| KAI-CAMERA-010 | MEDIUM | Sensor fusion uses mutable dictionary defaults |
| KAI-CAMERA-011 | MEDIUM | Shared frame and history state are unsynchronised and worker-local |
| KAI-CAMERA-012 | MEDIUM | Motion analysis compares unrelated camera and screen sources |
| KAI-CAMERA-013 | MEDIUM | Heuristic emotional and wellbeing conclusions are presented without calibrated confidence |
| KAI-CAMERA-014 | MEDIUM | Health reports ok without validating capture capability |
| KAI-CAMERA-015 | MEDIUM | Configuration values and device paths are not validated |

---

## Camera service: `perception/camera/app.py`

### KAI-CAMERA-001 — CRITICAL — Unauthenticated physical-camera activation
**Issue:** `POST /capture/camera` requires no authentication, authorisation or user-presence confirmation. It opens `CAMERA_DEVICE`, captures one frame and analyses it.  
**Risk:** Any network-reachable caller can remotely activate the attached camera and infer environmental brightness, movement and frame dimensions. Repeated calls permit ongoing surveillance even though raw frames are not returned.  
**Recommendation:** Require authenticated local-device control, explicit user consent and visible capture indicators.  
**Status:** OPEN — immediate remediation required

### KAI-CAMERA-002 — CRITICAL — Unauthenticated screen capture
**Issue:** `POST /capture/screen`, legacy `POST /process` and `POST /proactive/auto` call `_capture_screen()` without authentication. The function captures the primary monitor through `mss`.  
**Risk:** Reachable callers can repeatedly trigger capture and analysis of the operator’s current display, exposing activity patterns and creating a direct privacy boundary violation.  
**Recommendation:** Restrict screen capture to authenticated local sessions with explicit consent and capability-scoped access.  
**Status:** OPEN — immediate remediation required

### KAI-CAMERA-003 — HIGH — Fabricated signals manipulate proactive state
**Issue:** `/proactive/evaluate` and `/proactive/interpret` accept arbitrary unauthenticated dictionaries. `_speak_or_not` updates the global `_last_proactive_ts` whenever the fabricated urgency reaches the threshold.  
**Risk:** A caller can force the cooldown active and suppress later legitimate proactive alerts, or generate high-urgency decisions from invented stress, fatigue, shouting, darkness or motion signals.  
**Recommendation:** Accept only authenticated, signed sensor events and update state only after an authorised action is committed.  
**Status:** OPEN

### KAI-CAMERA-004 — HIGH — Untrusted sensor text enters the LLM
**Issue:** `interpret_multi` interpolates caller-controlled audio and video values into a prompt sent to `LANGGRAPH_URL/run`. No provenance, schema bounds or untrusted-data delimiter is used.  
**Risk:** Malicious values can inject instructions into the agentic reasoning path and produce trusted-looking multimodal interpretations.  
**Recommendation:** Use strict typed schemas, provenance and structured non-instructional tool data rather than prompt interpolation.  
**Status:** OPEN

### KAI-CAMERA-005 — HIGH — Blocking capture and analysis execute on the event loop
**Issue:** `cv2.VideoCapture`, `cap.read`, `mss.grab`, OpenCV conversion/Canny/difference and NumPy operations run directly in async endpoint handlers.  
**Risk:** Slow hardware or expensive frames block the event-loop worker and make all service endpoints unavailable. Repeated unauthenticated calls create trivial denial of service.  
**Recommendation:** Use a bounded capture/inference worker with strict timeouts and concurrency limits.  
**Status:** OPEN

### KAI-CAMERA-006 — HIGH — Capture failure becomes false sensor evidence
**Issue:** `_capture_screen` catches every import/runtime exception and returns a zero-filled 480×640 frame when NumPy is available. `/proactive/auto` also converts capture errors into default no-motion values.  
**Risk:** Missing permissions, display failures and backend faults are represented as a valid very-dark/static environment, which can trigger wellbeing nudges and corrupt motion history rather than surfacing sensor unavailability.  
**Recommendation:** Fail closed with explicit unavailable/error state; never convert capture failure into a real observation.  
**Status:** OPEN

### KAI-CAMERA-007 — MEDIUM — Operational surveillance metadata is publicly readable
**Issue:** `/analysis/history` and `/health` disclose timestamps, brightness, motion, dimensions, screen-activity results and configured physical/virtual device paths without authentication.  
**Risk:** Callers can infer occupancy/activity patterns and enumerate device configuration.  
**Recommendation:** Require scoped operational access and minimise exposed sensor history.  
**Status:** OPEN

### KAI-CAMERA-008 — MEDIUM — Cooldown starts before authorisation
**Issue:** `_speak_or_not` updates `_last_proactive_ts` as soon as its heuristic says `should_speak`. `/proactive/auto` asks the tool gate only afterwards. A rejected or unreachable gate still consumes the cooldown.  
**Risk:** Legitimate later alerts are suppressed even though no speech occurred or was approved.  
**Recommendation:** Commit cooldown only after successful gate approval and confirmed delivery.  
**Status:** OPEN

### KAI-CAMERA-009 — MEDIUM — Speech delivery is not acknowledged
**Issue:** After gate approval, `/proactive/auto` posts to TTS but does not inspect the HTTP status or body. Exceptions are swallowed.  
**Risk:** The endpoint can imply an authorised proactive action while synthesis failed; state and cooldown still indicate completion.  
**Recommendation:** Validate delivery acknowledgement and return distinct decision, approval and delivery states.  
**Status:** OPEN

### KAI-CAMERA-010 — MEDIUM — Mutable default request objects
**Issue:** `/proactive/evaluate` and `/proactive/interpret` define `audio_signals: Dict = {}` and `video_signals: Dict = {}`.  
**Risk:** Shared mutable defaults are unsafe API state and can retain mutations if future code modifies them, producing cross-request contamination.  
**Recommendation:** Use validated request models with factory-created dictionaries.  
**Status:** OPEN

### KAI-CAMERA-011 — MEDIUM — Sensor state is unsynchronised and process-local
**Issue:** `_last_frame`, `_analysis_history` and `_last_proactive_ts` are shared globals with no locks.  
**Risk:** Concurrent camera/screen calls race, compare inconsistent frames and reorder history. Multiple workers maintain different cooldowns and motion baselines; restart erases them.  
**Recommendation:** Serialise capture state or use a dedicated sensor worker and shared timestamped store.  
**Status:** OPEN

### KAI-CAMERA-012 — MEDIUM — Motion baseline mixes capture sources
**Issue:** The single `_last_frame` is used for camera frames, screen frames and legacy/process captures. Source identity is not tracked. Frames with matching dimensions can be compared even when they come from different sensors.  
**Risk:** Motion scores can represent a camera-to-screen source switch rather than movement, producing false proactive triggers.  
**Recommendation:** Maintain independent, timestamped baselines per sensor source.  
**Status:** OPEN

### KAI-CAMERA-013 — MEDIUM — Heuristics imply wellbeing conclusions without calibration
**Issue:** Fixed brightness, RMS and motion thresholds generate messages such as “You sound stressed,” “Energy seems low,” “Things sound heated” and “everything okay?” No confidence calibration, temporal duration or false-positive handling is present.  
**Risk:** Weak sensor heuristics are presented as personal/emotional assessments and may create intrusive or misleading interventions.  
**Recommendation:** Treat outputs as low-confidence signals, require temporal corroboration and preserve uncertainty explicitly.  
**Status:** OPEN

### KAI-CAMERA-014 — MEDIUM — Health is readiness-blind
**Issue:** `/health` always returns `status: ok` and stringifies dependency booleans. It does not test camera access, screen permission, monitor availability, observer state or a successful capture.  
**Risk:** Orchestration treats the service as ready while all real capture paths may fail or return dummy evidence.  
**Recommendation:** Separate liveness from camera, screen and analysis readiness.  
**Status:** OPEN

### KAI-CAMERA-015 — MEDIUM — Configuration is not validated
**Issue:** Device paths, capture directory, cooldown, service URLs, session ID and port are accepted directly. Negative cooldown values disable effective throttling; invalid paths or URLs fail only during use.  
**Risk:** Misconfiguration causes unsafe capture behaviour, unbounded proactive triggering or silent routing failure.  
**Recommendation:** Validate typed startup configuration with explicit ranges and approved path/URL schemes.  
**Status:** OPEN

---

## Batch totals

- Findings: **15**
- Critical: **2**
- High: **4**
- Medium: **9**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **426**
- Critical: **44**
- High: **162**
- Medium: **217**
- Low: **3**

## Files materially reviewed in this batch

`perception/camera/app.py`.
