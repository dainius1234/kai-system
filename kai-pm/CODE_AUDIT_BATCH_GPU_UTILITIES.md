# Kai Code Audit — GPU Detection and Model Recommendation Batch

Repository: `dainius1234/kai-system`  
Status: CONFIRMED — pending final consolidation  
Reviewed: 27 July 2026

## Consolidated batch index

| ID | Severity | Finding |
|---|---|---|
| KAI-GPU-001 | HIGH | `nvidia-smi` process success is treated as proof that CUDA model inference is usable |
| KAI-GPU-002 | HIGH | PyTorch CUDA availability is not corroborated when `nvidia-smi` succeeds |
| KAI-GPU-003 | HIGH | A PATH-substituted executable named `nvidia-smi` controls hardware detection |
| KAI-GPU-004 | HIGH | `has_cuda()` propagates several process/decoding/OS failures instead of returning a controlled unavailable result |
| KAI-GPU-005 | HIGH | `get_gpu_info()` returns `available=True` when the detail command fails |
| KAI-GPU-006 | HIGH | `get_gpu_info()` returns `available=True` when output is empty |
| KAI-GPU-007 | HIGH | Every parsing exception returns `available=True` with blank hardware fields |
| KAI-GPU-008 | HIGH | Detection and detailed inspection run separate commands and can observe different hardware states |
| KAI-GPU-009 | HIGH | Hardware checks are repeated on every call with no cache, freshness or concurrency control |
| KAI-GPU-010 | HIGH | Model recommendations use total VRAM rather than currently free/allocatable memory |
| KAI-GPU-011 | HIGH | Recommendations ignore quantisation, context window, batch size, concurrent workloads and runtime overhead |
| KAI-GPU-012 | HIGH | Recommended model identifiers are not verified to be installed or reachable |
| KAI-GPU-013 | HIGH | Static model recommendations are another independent model registry that can conflict with active routing |
| KAI-GPU-014 | HIGH | Speculative decoding is enabled from non-empty environment strings rather than verified compatible models |
| KAI-GPU-015 | HIGH | Draft and verifier models are not required to be distinct |
| KAI-GPU-016 | HIGH | Draft and verifier tokenizer, vocabulary, architecture and backend compatibility are not checked |
| KAI-GPU-017 | HIGH | Invalid `ENABLE_SPECULATIVE_DECODING` values enable the feature by default unless they equal exact `false` |
| KAI-GPU-018 | HIGH | `SPECULATIVE_DRAFT_TOKENS` accepts zero, negative and arbitrarily large values |
| KAI-GPU-019 | HIGH | Invalid draft-token text raises at configuration-read time rather than producing a controlled disabled state |
| KAI-GPU-020 | HIGH | Hardware/model recommendations have no deployment, actor or policy authority boundary |
| KAI-GPU-021 | HIGH | GPU details and recommendations lack source, timestamp, command digest and confidence provenance |
| KAI-GPU-022 | MEDIUM | `FORCE_CPU` recognises only exact case-insensitive `true`, not other normal boolean forms |
| KAI-GPU-023 | MEDIUM | Environment-controlled `FORCE_CPU` changes global routing behaviour without audit or validation |
| KAI-GPU-024 | MEDIUM | Subprocesses inherit the complete service environment |
| KAI-GPU-025 | MEDIUM | Complete subprocess output is buffered without a byte limit |
| KAI-GPU-026 | MEDIUM | Two-second process timeouts are hard-coded and not calibrated or configurable |
| KAI-GPU-027 | MEDIUM | Only the first GPU supplies device name, memory, driver and CUDA information |
| KAI-GPU-028 | MEDIUM | Heterogeneous multi-GPU systems are reduced to first-device characteristics plus a line count |
| KAI-GPU-029 | MEDIUM | Comma-space output parsing is fragile to formatting, localisation and device-name content |
| KAI-GPU-030 | MEDIUM | Memory, device count and version values have no finite/range/schema validation |
| KAI-GPU-031 | MEDIUM | NaN, infinity and negative memory values can enter `GPUInfo` |
| KAI-GPU-032 | MEDIUM | `GPUInfo` is a mutable dataclass and can be changed to contradictory values |
| KAI-GPU-033 | MEDIUM | `available=True` can coexist with zero devices and empty model-relevant information |
| KAI-GPU-034 | MEDIUM | No CUDA compute capability, driver/runtime compatibility or kernel execution test is performed |
| KAI-GPU-035 | MEDIUM | No device-selection, MIG partition or container GPU visibility information is represented |
| KAI-GPU-036 | MEDIUM | No utilisation, reserved memory, thermal or health state is considered |
| KAI-GPU-037 | MEDIUM | `get_recommended_model()` invokes CUDA detection repeatedly through nested calls |
| KAI-GPU-038 | MEDIUM | Unknown/blank GPU details silently select a small GPU model rather than an explicit uncertain state |
| KAI-GPU-039 | MEDIUM | Speculative configuration is returned as an ordinary mutable dictionary without a versioned schema |
| KAI-GPU-040 | MEDIUM | Speculative configuration omits backend endpoint, timeout, context and resource budget |
| KAI-GPU-041 | MEDIUM | No metrics distinguish command detection, PyTorch detection, forced CPU or degraded parsing |
| KAI-GPU-042 | MEDIUM | No lifecycle warm-up or graceful hardware-state refresh contract exists |

---

## Findings

### KAI-GPU-001 — HIGH — Driver CLI is treated as inference readiness
Any successful `nvidia-smi --query-gpu=name` output makes `has_cuda()` true.

### KAI-GPU-002 — HIGH — CUDA runtime not corroborated
The PyTorch check runs only after the CLI check fails.

### KAI-GPU-003 — HIGH — Executable identity absent
The command is resolved through PATH.

### KAI-GPU-004 — HIGH — Incomplete exception handling
Only FileNotFoundError and TimeoutExpired are caught around the first subprocess.

### KAI-GPU-005 — HIGH — Detail-command failure fails open
A nonzero result returns `GPUInfo(available=True)`.

### KAI-GPU-006 — HIGH — Empty output fails open
No lines still returns available true.

### KAI-GPU-007 — HIGH — Parse failure fails open
The broad exception handler returns available true.

### KAI-GPU-008 — HIGH — Time-of-check inconsistency
`has_cuda()` and detail collection are separate subprocesses.

### KAI-GPU-009 — HIGH — Unbounded repeated probing
No shared cache/lock exists.

### KAI-GPU-010 — HIGH — Total memory is not capacity
Current free/allocatable memory is ignored.

### KAI-GPU-011 — HIGH — Fit model is incomplete
Static thresholds omit runtime/model workload characteristics.

### KAI-GPU-012 — HIGH — Installation not verified
The returned identifier may not exist in Ollama or any backend.

### KAI-GPU-013 — HIGH — Model-authority drift
The function conflicts with other static/dynamic model selectors.

### KAI-GPU-014 — HIGH — Environment strings enable speculation
No live models are probed.

### KAI-GPU-015 — HIGH — Same model accepted twice
Distinctness is not checked.

### KAI-GPU-016 — HIGH — Compatibility absent
No tokenizer/model/back-end contract is validated.

### KAI-GPU-017 — HIGH — Invalid enablement is permissive
Only exact `false` disables.

### KAI-GPU-018 — HIGH — Unsafe draft-token range
Direct integer value is returned.

### KAI-GPU-019 — HIGH — Bad configuration crashes
`int()` is unprotected.

### KAI-GPU-020 — HIGH — Policy context absent
Environment and local hardware decide routing with no governance revision.

### KAI-GPU-021 — HIGH — Hardware evidence provenance absent
Results contain no source/freshness/confidence.

### KAI-GPU-022 — MEDIUM — Narrow boolean parsing
`1`, `yes` and `on` do not force CPU.

### KAI-GPU-023 — MEDIUM — Global environment control
One variable changes every caller.

### KAI-GPU-024 — MEDIUM — Environment inheritance
The CLI receives all process variables.

### KAI-GPU-025 — MEDIUM — Buffered output
Complete stdout/stderr are captured.

### KAI-GPU-026 — MEDIUM — Fixed timeout
No deployment calibration.

### KAI-GPU-027 — MEDIUM — First-device-only details
Later GPUs are not characterised.

### KAI-GPU-028 — MEDIUM — Heterogeneous fleet collapse
Line count is the only multi-device information.

### KAI-GPU-029 — MEDIUM — Fragile CSV parsing
The format is split manually.

### KAI-GPU-030 — MEDIUM — Field validation absent
Parsed values are trusted.

### KAI-GPU-031 — MEDIUM — Non-finite memory accepted
`float()` permits these values.

### KAI-GPU-032 — MEDIUM — Mutable contradictory state
Dataclass fields are not frozen/validated.

### KAI-GPU-033 — MEDIUM — Contradictory availability object
Blank fields do not clear availability.

### KAI-GPU-034 — MEDIUM — No functional CUDA test
No allocation/kernel operation occurs.

### KAI-GPU-035 — MEDIUM — Partition/visibility absent
Container and MIG details are omitted.

### KAI-GPU-036 — MEDIUM — Operational GPU health absent
Utilisation/temperature/errors are ignored.

### KAI-GPU-037 — MEDIUM — Nested repeated process work
Recommendation calls detection and then detailed detection, which calls detection again.

### KAI-GPU-038 — MEDIUM — Uncertainty hidden
Blank details select `qwen2:1.5b`.

### KAI-GPU-039 — MEDIUM — Mutable configuration response
A plain dictionary is returned.

### KAI-GPU-040 — MEDIUM — Incomplete speculation contract
Endpoint/context/deadlines are absent.

### KAI-GPU-041 — MEDIUM — Detection-mode observability absent
Callers cannot distinguish evidence sources.

### KAI-GPU-042 — MEDIUM — Lifecycle absent
No startup warm-up or controlled refresh exists.

---

## Batch totals

- Findings: **42**
- Critical: **0**
- High: **21**
- Medium: **21**
- Low: **0**

## Provisional repository totals after all logged batches

- Findings: **2,966**
- Critical: **182**
- High: **1,515**
- Medium: **1,266**
- Low: **3**

## Files materially reviewed

`common/gpu_utils.py`, with repository model-routing context used to assess recommendation and speculative-decoding effects.
