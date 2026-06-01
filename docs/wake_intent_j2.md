# J2 — Wake-word + Intent Judge

## Overview

J2 adds a dedicated wake/intent front-door service at `perception/wake/app.py`.

It provides:
- Wake-word detection from text (`WAKE_WORDS`)
- Optional audio transcription from `audio_b64` (faster-whisper when available)
- Debounce cooldown (`WAKE_COOLDOWN_SECONDS`)
- Tiny-model intent classification with strict JSON validation
- Regex/keyword heuristic fallback when the model is unavailable or returns invalid output

## Endpoints

### `POST /wake/detect`
Input:
```json
{"text":"Kai, what's next?"}
```
or
```json
{"audio_b64":"<base64 wav payload>"}
```

Output:
```json
{"detected":true,"confidence":0.95,"wake_word":"kai","latency_ms":6}
```

### `POST /wake/intent`
Input:
```json
{"text":"set a reminder for 8am"}
```

Output:
```json
{"intent":"task","confidence":0.82,"reasoning":"Detected actionable request"}
```

Intent taxonomy:
- `chat`
- `task`
- `question`
- `command`
- `emotional`
- `unknown`

### `POST /wake/process`
Runs detect + classify:
```json
{"wake": {...}, "intent": {...}}
```

### `GET /health`
Dependency-aware health:
- `status: ok|degraded`
- wake config
- whisper/model availability flags

## Configuration

```env
WAKE_WORDS=kai,hey kai,ok kai
WAKE_COOLDOWN_SECONDS=2
WAKE_INTENT_MODEL=qwen2:0.5b
WAKE_CONFIDENCE_THRESHOLD=0.6
```

## Integrations

- **Dashboard proxy**
  - `POST /api/wake/detect`
  - `POST /api/wake/intent`
  - `POST /api/wake/process`

- **Langgraph pre-routing**
  - `FF_WAKE_INTENT_ROUTING` (default: off)
  - when enabled, `/chat` asks wake service for intent before route decision

## Extending

- Add new intent labels by updating:
  1. prompt template (`common/prompt_templates.py`)
  2. allowed intent set in wake service
  3. fallback heuristic map
  4. tests in `scripts/test_wake_intent.py`
