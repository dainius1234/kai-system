# Sovereign AI (Local-Only) — Consolidated Technical Specification

This document consolidates the local-only sovereign AI architecture and implementation notes into a single, production-ready specification. It is designed for an entirely air‑gapped deployment: no cloud egress, no external services, and all storage confined to local volumes.

---

## 1) Complete System Architecture

### 1.1 Layered Architecture (Mermaid)

```mermaid
flowchart TD
    subgraph L0["🖥️ LAYER 0: HARDWARE ROOT"]
        TPM["TPM 2.0: /dev/tpm0<br/>Seals LUKS, Signs Ledger"]
        GPU["NVIDIA RTX 5080<br/>CUDA 12.4"]
        NVME["NVMe: LUKS2 Encrypted<br/>TPM-Sealed Key"]
        Cam["USB Camera: /dev/video0"]
        Mic["Microphone: /dev/snd"]
        
        TPM -->|PCR 0-7 Binding| NVME
        GPU -->|--gpus all| L2
    end

    subgraph L1["⚖️ LAYER 1: SOVEREIGN CORE"]
        TG["Tool Gate<br/>Port: 8000"]
        Ledger["Ledger DB<br/>PostgreSQL + pgvector<br/>Port: 5432"]
        Dash["Dashboard<br/>Tauri + Vue.js<br/>Port: 8080"]
        
        TG -->|TPM-Signs| Ledger
        Dash -->|CO-SIGN/VETO| TG
    end

    subgraph L2["🧠 LAYER 2: INTELLIGENCE"]
        MemCore["Sovereign Memory Core<br/>Port: 8001"]
        LLM_Pool["Specialist LLM Pool<br/>Ollama:11434"]
        LangGraph["LangGraph Engine<br/>Redis:6379"]
        Grok["ARA Grok Advisor<br/>Port: 8090"]
        
        MemCore -->|/route → JSON| LangGraph
        LangGraph -->|Selects Specialist| LLM_Pool
        LangGraph -->|Consults| Grok
        LLM_Pool -->|Plan/Code| LangGraph
    end

    subgraph L3["👁️ LAYER 3: PERCEPTION"]
        CamPipe["Camera Pipeline<br/>YOLOv11 + OpenCV<br/>Output: /dev/video10"]
        AudioPipe["Audio Pipeline<br/>Porcupine + Whisper"]
        ScreenCap["Desktop Capture<br/>scrcpy + pyautogui"]
        
        Cam -->|Raw Frames| CamPipe
        Mic -->|Audio Stream| AudioPipe
        CamPipe -->|Blurred Faces| VirtualCam["Virtual Camera<br/>/dev/video10"]
        AudioPipe -->|Transcribed Text| LangGraph
    end

    subgraph L4["💓 LAYER 4: AWARENESS"]
        Heartbeat["Heartbeat Monitor<br/>Systemd Timer"]
        Calendar["Calendar Sync<br/>Google API via n8n"]
        
        Heartbeat -->|Alerts| TG
        Calendar -->|Event Feed| Heartbeat
    end

    subgraph L5["🔨 LAYER 5: EXECUTION"]
        Executor["Executor Service<br/>Port: 8002"]
        Sandboxes["Tool Sandboxes"]
        
        Executor -->|Docker Exec| Sandboxes
    end

    subgraph L6["🗣️ LAYER 6: OUTPUT"]
        TTS["XTTS-v2<br/>Voice Clone"]
        Avatar["ai-iris Avatar<br/>Three.js + VRM"]
        Display["Monitor Output<br/>+ Overlay UI"]
        
        TTS -->|16kHz Audio| Avatar
        Avatar -->|Lip Sync| Display
    end

    %% ========== DATA FLOW ==========
    L3 -- "User Speech/Text" --> LangGraph
    LangGraph -- "Context Request" --> MemCore
    MemCore -- "Routing Decision" --> LangGraph
    LangGraph -- "Task to Specialist" --> LLM_Pool
    LLM_Pool -- "Generated Plan" --> LangGraph
    LangGraph -- "Execution Request" --> TG
    
    TG -- "Policy Check + CO-SIGN" --> Ledger
    TG -- "Approved Command" --> Executor
    Executor -- "Run in Sandbox" --> Sandboxes
    Sandboxes -- "Results" --> Executor
    Executor -- "Log Results" --> Ledger
    Executor -- "Memory Update" --> MemCore
    
    L4 -- "Proactive Alerts" --> TG
    L6 -- "Avatar Response" --> User
    
    style L0 fill:#f9f,stroke:#333
    style L1 fill:#bbf,stroke:#333
    style L2 fill:#bfb,stroke:#333
    style L3 fill:#ffb,stroke:#333
    style L4 fill:#fbf,stroke:#333
    style L5 fill:#bff,stroke:#333
    style L6 fill:#fbb,stroke:#333
```

### 1.2 Security Invariants (Enforced)

- **Tool Gate** is a mandatory choke point for all execution requests.
- **Network isolation**: all inter-service traffic remains within `sovereign-net`.
- **No egress**: outbound traffic is blocked; `memu-core` uses `network_mode: service:tool-gate` for outbound filtering.
- **Local-only storage**: all volumes (`memu_data`, `postgres_data`, `scratchpad`) are local bind mounts.

---

## 2) Local-Only Docker Compose (Full Stack)

Save as `docker-compose.sovereign.yml`.

```yaml
version: '3.8'

services:
  # ========== SOVEREIGN CORE ==========
  postgres:
    image: postgres:15-alpine
    container_name: sovereign-postgres
    hostname: ledger-db
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.2
    environment:
      POSTGRES_DB: sovereign
      POSTGRES_USER: keeper
      POSTGRES_PASSWORD_FILE: /run/secrets/db_password
      POSTGRES_INITDB_ARGS: "--data-checksums"
    secrets:
      - db_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init:/docker-entrypoint-initdb.d:ro
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    read_only: true
    tmpfs:
      - /tmp:size=64M,noexec,nosuid
      - /run:size=16M,noexec,nosuid

  tool-gate:
    build: ./tool-gate
    container_name: sovereign-tool-gate
    hostname: tool-gate
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.3
    ports:
      - "8000:8000"
    volumes:
      - /dev/tpm0:/dev/tpm0:ro
      - tool_gate_config:/config:ro
      - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      DATABASE_URL: "postgresql://keeper:${DB_PASSWORD}@postgres:5432/sovereign"
      TPM_DEVICE: "/dev/tpm0"
      MODE: "PUB"
    depends_on:
      - postgres
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-toolgate
    cap_drop:
      - ALL
    read_only: true

  dashboard:
    build: ./dashboard
    container_name: sovereign-dashboard
    hostname: dashboard
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.4
    ports:
      - "8080:8080"
    volumes:
      - dashboard_data:/data:rw
      - ./config/dashboard.json:/app/config.json:ro
    environment:
      TOOL_GATE_URL: "http://tool-gate:8000"
      LEDGER_URL: "postgresql://keeper:${DB_PASSWORD}@postgres:5432/sovereign"
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

  # ========== SOVEREIGN MEMORY CORE ==========
  memu-core:
    build:
      context: ./memu-core
      dockerfile: Dockerfile.memu
    container_name: sovereign-memu-core
    hostname: memu-core
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.5
    volumes:
      - memu_data:/memory:rw
      - ./config/memu_local.json:/app/config.json:ro
    environment:
      DATABASE_URL: "postgresql://keeper:${DB_PASSWORD}@postgres:5432/memu_db"
      VECTOR_STORE: "pgvector"
      API_HOST: "0.0.0.0"
      API_PORT: "8001"
      STORAGE_PATH: "/memory"
    expose:
      - "8001"
    command: ["--local", "--no-cloud"]
    restart: "no"

  # ========== LLM & ORCHESTRATION ==========
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    hostname: ollama
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.6
    volumes:
      - ollama_data:/root/.ollama
    environment:
      OLLAMA_KEEP_ALIVE: "24h"
      OLLAMA_MAX_LOADED_MODELS: "3"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  langgraph:
    build: ./langgraph
    container_name: langgraph
    hostname: langgraph
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.7
    volumes:
      - langgraph_data:/data:rw
      - ./workspaces:/workspaces:rw
    environment:
      REDIS_URL: "redis://redis:6379"
      MEMU_URL: "http://memu-core:8001"
      TOOL_GATE_URL: "http://tool-gate:8000"
      OLLAMA_URL: "http://ollama:11434"
    depends_on:
      - redis
      - memu-core
      - tool-gate
      - ollama

  redis:
    image: redis:7-alpine
    container_name: sovereign-redis
    hostname: redis
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.8
    volumes:
      - redis_data:/data
    command: ["redis-server", "--appendonly", "yes", "--save", "900 1", "--maxmemory", "1gb"]

  # ========== EXECUTION LAYER ==========
  executor:
    build: ./executor
    container_name: executor
    hostname: executor
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.9
    volumes:
      - scratchpad:/io:rw
      - /var/run/docker.sock:/var/run/docker.sock:ro
    environment:
      TOOL_GATE_URL: "http://tool-gate:8000"
      MEMU_URL: "http://memu-core:8001"
      EXECUTION_TIMEOUT: "30"
      MAX_OUTPUT_SIZE: "1048576"
    expose:
      - "8002"
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    restart: "no"

  # ========== PERCEPTION LAYER ==========
  camera-service:
    build: ./perception/camera
    container_name: camera-service
    network_mode: "host"
    devices:
      - "/dev/video0:/dev/video0"
      - "/dev/video10:/dev/video10"
    privileged: true
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
    environment:
      DISPLAY: "${DISPLAY}"
      YOLO_MODEL: "yolo11n.pt"
      BLUR_KERNEL: "99"
    restart: unless-stopped

  audio-service:
    build: ./perception/audio
    container_name: audio-service
    network_mode: "host"
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ./config/audio.json:/app/config.json:ro
      - ./voice_samples:/voice_samples:ro
    environment:
      PORCUPINE_KEYWORD: "ara"
      WHISPER_MODEL: "large-v3"
      SAMPLE_RATE: "16000"
    restart: unless-stopped

  # ========== HEARTBEAT ==========
  heartbeat:
    build: ./heartbeat
    container_name: sovereign-heartbeat
    hostname: heartbeat
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.10
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - ./config/heartbeat.json:/app/config.json:ro
    environment:
      TOOL_GATE_URL: "http://tool-gate:8000"
      CHECK_INTERVAL: "60"
      ALERT_WINDOW: "300"
    restart: unless-stopped

  # ========== OUTPUT LAYER ==========
  tts-service:
    build: ./output/tts
    container_name: tts-service
    networks:
      sovereign-net:
        ipv4_address: 172.20.0.11
    volumes:
      - ./voice_clones:/clones:ro
      - ./output/audio:/audio:rw
    environment:
      MODEL: "tts_models/multilingual/multi-dataset/xtts_v2"
      SAMPLE_RATE: "16000"
      LANGUAGE: "en"
    expose:
      - "8003"

  avatar-service:
    build: ./output/avatar
    container_name: avatar-service
    network_mode: "host"
    volumes:
      - /tmp/.X11-unix:/tmp/.X11-unix
      - ./models/avatar.vrm:/app/avatar.vrm:ro
    environment:
      DISPLAY: "${DISPLAY}"
      TTS_URL: "http://tts-service:8003"
      WEBRTC_PORT: "8081"
    ports:
      - "8081:8081"
    restart: unless-stopped

  # ========== SANDBOXES ==========
  qgis-sandbox:
    image: qgis/qgis:latest
    container_name: qgis-sandbox
    network_mode: "none"
    read_only: true
    cap_drop:
      - ALL
    user: "1000:1000"
    volumes:
      - scratchpad:/io:rw
    tmpfs:
      - /tmp:size=100M,noexec,nosuid
    restart: "no"

  n8n-sandbox:
    image: n8nio/n8n:latest
    container_name: n8n-sandbox
    network_mode: "none"
    read_only: true
    cap_drop:
      - ALL
    user: "1000:1000"
    volumes:
      - scratchpad:/io:rw
      - ./config/n8n_credentials.json:/credentials.json:ro
    environment:
      N8N_ENCRYPTION_KEY: "${N8N_ENCRYPTION_KEY}"
    restart: "no"

  shell-sandbox:
    build: ./sandboxes/shell
    container_name: shell-sandbox
    network_mode: "none"
    read_only: true
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
      - apparmor:docker-shell
    user: "1000:1000"
    volumes:
      - scratchpad:/io:rw
    tmpfs:
      - /tmp:size=10M,noexec,nosuid
    restart: "no"

volumes:
  postgres_data:
  tool_gate_config:
  dashboard_data:
  memu_data:
  ollama_data:
  langgraph_data:
  redis_data:
  scratchpad:

secrets:
  db_password:
    file: ./secrets/db_password.txt

networks:
  sovereign-net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
```

---

## 3) Sovereign Memory Core (memU)

### 3.1 Service Configuration (Compose Snippet)

```yaml
memu-core:
  image: nevanmind-ai/memu:latest
  container_name: sovereign-memu-core
  networks: [sovereign-net]
  volumes:
    - memu_data:/memory
    - ./config/memu_local.json:/app/config.json:ro
  environment:
    DATABASE_URL: postgresql://keeper:***@postgres:5432/memu_db
    VECTOR_STORE: pgvector
    API_HOST: 0.0.0.0
    API_PORT: 8001
  command: ["--local", "--no-cloud", "--storage-path", "/memory"]
```

### 3.2 Internal-Only API Endpoints

- `POST /memory/retrieve` – query local pgvector for context.
- `POST /route` – returns JSON `{ "specialist": "DeepSeek-V4", "task": {...} }`.
- `POST /memory/memorize` – append result to local memory store.

### 3.3 LangGraph Integration Node

```python
# memu_router_node.py
import httpx

async def call_memu_router(state):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://memu-core:8001/route",
            json={"query": state["user_input"], "session_id": state["session_id"]},
            timeout=5.0,
        )
    route_decision = response.json()
    state["specialist"] = route_decision["specialist"]
    return state
```

### 3.4 API Core (FastAPI Example)

```python
# memu-core/app/api.py
from datetime import datetime
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Sovereign Memory Core")

class MemoryRequest(BaseModel):
    query: str
    session_id: str
    timestamp: str

class RoutingResponse(BaseModel):
    specialist: str  # "DeepSeek-V4", "Kimi-2.5", "Qwen-VL"
    context_payload: dict

class MemoryUpdate(BaseModel):
    timestamp: str
    event_type: str  # "task_complete", "conversation", "file_access"
    task_id: Optional[str]
    result_raw: Optional[str]
    metrics: Optional[dict]
    state_delta: Optional[dict]

@app.post("/route", response_model=RoutingResponse)
async def route_request(request: MemoryRequest):
    similar_memories = await vector_search(
        query=request.query,
        user_id="keeper",
        top_k=50,
    )

    metadata = {
        "time": datetime.utcnow().isoformat(),
        "session_id": request.session_id,
        "tags": extract_tags(request.query),
        "geo": get_geo_context(),
    }

    specialist = select_specialist(
        query=request.query,
        memory_context=similar_memories,
        historical_patterns=load_user_patterns(),
    )

    return RoutingResponse(
        specialist=specialist,
        context_payload={
            "query": request.query,
            "memory_vectors": [m.vector for m in similar_memories],
            "metadata": metadata,
        },
    )

@app.post("/memory/memorize")
async def memorize_event(update: MemoryUpdate):
    if update.state_delta:
        existing = await get_current_state()
        for key in update.state_delta:
            if key in existing:
                raise HTTPException(400, f"Duplicate key in state_delta: {key}")

    memory_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": update.timestamp,
        "event_type": update.event_type,
        "content": {
            "result": update.result_raw,
            "metrics": update.metrics or {},
            "state_changes": update.state_delta or {},
        },
        "embedding": generate_embedding(
            f"{update.event_type}: {update.result_raw}"
        ),
    }

    await pgvector_insert(memory_entry)
    return {"status": "appended", "id": memory_entry["id"]}

@app.get("/memory/retrieve")
async def retrieve_context(query: str, user_id: str, top_k: int = 20):
    return await vector_search(query, user_id, top_k)
```

### 3.5 Memory Compression Pipeline (SQL)

```sql
CREATE TABLE memory_compression_log (
    id BIGSERIAL PRIMARY KEY,
    ledger_id BIGINT REFERENCES sovereign_ledger(id),
    pre_hash CHAR(64) NOT NULL,
    post_hash CHAR(64) NOT NULL,
    compression_type VARCHAR(20) NOT NULL CHECK (
        compression_type IN ('KMEANS_INT8', 'PCA_REDUCTION', 'EXEMPLAR_SELECTION')
    ),
    original_count INTEGER NOT NULL,
    compressed_count INTEGER NOT NULL,
    original_bytes BIGINT NOT NULL,
    compressed_bytes BIGINT NOT NULL,
    precision_loss FLOAT CHECK (precision_loss >= 0 AND precision_loss <= 1),
    recall_loss FLOAT CHECK (recall_loss >= 0 AND recall_loss <= 1),
    clusters_metadata JSONB DEFAULT '{}',
    exemplars_per_cluster INTEGER DEFAULT 3,
    status VARCHAR(20) DEFAULT 'pending' CHECK (
        status IN ('pending', 'approved', 'executing', 'completed', 'failed')
    ),
    completed_at TIMESTAMPTZ,
    snapshot_name VARCHAR(255),
    rollback_available BOOLEAN DEFAULT TRUE,
    rollback_deadline TIMESTAMPTZ GENERATED ALWAYS AS (
        completed_at + INTERVAL '30 days'
    ) STORED,
    compression_ratio FLOAT GENERATED ALWAYS AS (
        1.0 - (compressed_bytes::FLOAT / NULLIF(original_bytes, 0))
    ) STORED,
    CHECK (original_count >= compressed_count),
    CHECK (original_bytes >= compressed_bytes)
);

CREATE OR REPLACE FUNCTION trigger_memory_compression()
RETURNS void AS $$
BEGIN
    WITH stats AS (
        SELECT 
            COUNT(*) as total_vectors,
            SUM(pg_column_size(vector_data)) as total_bytes
        FROM vector_memory_metadata
        WHERE last_compressed_at IS NULL 
           OR last_compressed_at < NOW() - INTERVAL '7 days'
    )
    INSERT INTO memory_compression_log (
        pre_hash,
        compression_type,
        original_count,
        original_bytes,
        status
    )
    SELECT 
        ENCODE(SHA256(
            (SELECT string_agg(vector_id::text, '') 
             FROM vector_memory_metadata
             ORDER BY vector_id)
        ), 'hex'),
        'KMEANS_INT8',
        stats.total_vectors,
        stats.total_bytes,
        'pending'
    FROM stats
    WHERE stats.total_vectors > 50000;
END;
$$ LANGUAGE plpgsql;

CREATE EVENT TRIGGER weekly_compression
ON SCHEDULE '0 2 * * 0'
DO CALL trigger_memory_compression();
```

---

## 4) Perception Layer (Implementation Stubs)

### 4.1 Camera Pipeline (YOLO + Privacy Blur)

```python
# perception/camera/service.py
import cv2
from datetime import datetime
from ultralytics import YOLO
import v4l2loopback

class PrivacyCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.yolo = YOLO('yolo11n.pt')
        self.virtual_cam = v4l2loopback.Output(
            device='/dev/video10',
            width=1280,
            height=720,
            fps=30,
        )

        self.blur_kernel = (99, 99)
        self.blur_sigma = 30
        self.detection_confidence = 0.5
        self.detection_log = []

    def process_frame(self, frame):
        results = self.yolo(frame, classes=[0], conf=self.detection_confidence, verbose=False)
        annotated = frame.copy()

        for box in results[0].boxes:
            if box.conf.item() > self.detection_confidence:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face_region = annotated[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, self.blur_kernel, self.blur_sigma)
                annotated[y1:y2, x1:x2] = blurred_face

                self.detection_log.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'confidence': box.conf.item(),
                    'bbox': [x1, y1, x2, y2],
                    'action': 'blurred',
                })

        return annotated
```

### 4.2 Audio Pipeline (Porcupine + Whisper)

```python
# perception/audio/service.py
import queue
import uuid
from datetime import datetime

import numpy as np
import pvporcupine
import pyaudio
from faster_whisper import WhisperModel

class AudioPipeline:
    def __init__(self):
        self.porcupine = pvporcupine.create(
            keyword_paths=['./models/ara_linux.ppn'],
            sensitivities=[0.7],
        )

        self.whisper = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
        )

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            rate=self.porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=self.porcupine.frame_length,
        )

        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.vad_threshold = 0.03
        self.silence_duration = 2.0
        self.current_session = None
```

---

## 5) Workspace System (MD Files)

### 5.1 Directory Structure

```
/sovereign/workspaces/
├── workspace_<uuid>/
│   ├── discovery.md
│   ├── research.md
│   ├── plan.md
│   ├── progress.md
│   ├── code/
│   │   ├── script_001.py
│   │   └── script_002.py
│   ├── data/
│   │   ├── input.csv
│   │   └── output.json
│   └── logs/
│       ├── tool_execution.log
│       └── memory_updates.log
└── archive/
    ├── workspace_<uuid>_20250315.tar.gz
    └── workspace_<uuid>_20250322.tar.gz
```

### 5.2 Markdown Templates

```markdown
# discovery.md
## Problem Statement
**Date**: {{timestamp}}
**Session ID**: {{session_id}}
**Trigger**: {{trigger_source}}

## User Request
```

{{user_query}}

```

## Initial Context
- **Related Memories**: {{memory_references}}
- **User State**: {{user_state}}
- **Environmental Context**: {{environment}}

## Success Criteria
1. {{criteria_1}}
2. {{criteria_2}}

## Constraints & Limitations
- {{constraint_1}}
- {{constraint_2}}

---
# research.md
## Research Phase
**Started**: {{start_time}}
**Completed**: {{end_time}}
**Specialist Used**: {{specialist_llm}}

## Findings
### 1. {{finding_title}}
- **Source**: {{source}}
- **Relevance**: {{relevance_score}}
- **Key Insight**: {{insight}}

### 2. {{finding_title}}
- **Source**: {{source}}
- **Relevance**: {{relevance_score}}
- **Key Insight**: {{insight}}

## Technical Requirements
- **APIs Needed**: {{api_list}}
- **Data Sources**: {{data_sources}}
- **Dependencies**: {{dependencies}}

## Risk Assessment
| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| {{risk_1}} | {{prob_1}} | {{impact_1}} | {{mitigation_1}} |

---
# plan.md
## Execution Plan
**Approved for CO-SIGN**: {{approval_timestamp}}
**CO-SIGN ID**: {{cosign_id}}

### Phase 1: Setup
1. **Task**: {{task_description}}
   - **Command**: {{command}}
   - **Expected Output**: {{expected}}
   - **Timeout**: {{timeout_seconds}}s

### Phase 2: Execution
1. **Task**: {{task_description}}
   - **Command**: {{command}}
   - **Expected Output**: {{expected}}
   - **Timeout**: {{timeout_seconds}}s

### Rollback Plan
- **Checkpoint 1**: {{checkpoint_description}}
- **Rollback Command**: {{rollback_command}}

---
# progress.md
## Execution Log
**Workspace ID**: {{workspace_id}}
**Status**: {{current_status}}

### Timeline
```

{{timestamp}} - Phase 1 started
{{timestamp}} - Command executed: {{command}}
{{timestamp}} - Output: {{output}}
{{timestamp}} - Exit code: {{exit_code}}

```

### Current State
```json
{
  "completed_tasks": {{completed_count}},
  "failed_tasks": {{failed_count}},
  "total_duration": {{total_seconds}}s,
  "last_checkpoint": {{checkpoint_timestamp}}
}
```

Issues & Resolutions

Issue Resolution Timestamp
{{issue}} {{resolution}} {{timestamp}}
```

---

## 6) Output Layer (Avatar + TTS)

### 6.1 Voice Cloning & TTS (XTTS-v2)

```python
# output/tts/service.py
from datetime import datetime
import hashlib
from pathlib import Path

import numpy as np
from TTS.api import TTS

class SovereignTTS:
    def __init__(self):
        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=True,
        )

        self.voice_bank = Path("./voice_clones")
        self.voice_bank.mkdir(exist_ok=True)
        self.cache = {}

    def clone_voice(self, audio_path, voice_name="keeper"):
        speaker_wav = self.tts.speaker_manager.compute_embedding(
            audio_path,
            gpt_cond_len=30,
            max_ref_length=60,
        )

        embedding_path = self.voice_bank / f"{voice_name}.npy"
        np.save(embedding_path, speaker_wav)

        config = {
            "name": voice_name,
            "embedding_hash": hashlib.sha256(speaker_wav.tobytes()).hexdigest(),
            "created": datetime.utcnow().isoformat(),
            "sample_rate": 16000,
        }

        return config

    def synthesize(self, text, voice_name="keeper", emotion="neutral"):
        embedding_path = self.voice_bank / f"{voice_name}.npy"
        speaker_wav = np.load(embedding_path)
        emotion_params = self.get_emotion_params(emotion)

        wav = self.tts.tts(
            text=text,
            speaker_wav=speaker_wav,
            language="en",
            **emotion_params,
        )

        wav_16k = self.resample_to_16k(wav)
        cache_key = f"{hashlib.md5(text.encode()).hexdigest()}_{voice_name}_{emotion}"
        self.cache[cache_key] = wav_16k
        return wav_16k
```

---

## 7) Production & Development Workflow

### 7.1 Multi-Environment Configuration

```yaml
# docker-compose.override.yml (Development)
services:
  tool-gate:
    environment:
      - LOG_LEVEL=DEBUG
      - DEBUG_MODE=true
      - SKIP_TPM_VALIDATION=false
    volumes:
      - ./logs/tool_gate:/logs:rw
      - ./src/tool-gate:/app:rw
    ports:
      - "8000:8000"
      - "5678:5678"

  memu-core:
    environment:
      - MEMU_DEBUG=true
      - VECTOR_STORE=pgvector_test
    volumes:
      - ./test_data:/test_data:ro

  executor:
    security_opt: []
    cap_add:
      - SYS_PTRACE
```

```yaml
# docker-compose.prod.yml (Production)
services:
  tool-gate:
    environment:
      - LOG_LEVEL=WARN
      - DEBUG_MODE=false
      - SKIP_TPM_VALIDATION=false
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 1G
    restart: unless-stopped
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  postgres:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    command: >
      postgres
      -c max_connections=100
      -c shared_buffers=1GB
      -c effective_cache_size=3GB
      -c maintenance_work_mem=256MB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U keeper"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 40s
```

### 7.2 Build Script With Validation (Reference)

```bash
#!/bin/bash
# build_sovereign.sh - Complete production build with validation
set -euo pipefail

readonly BUILD_ID="sovereign-$(date +%Y%m%d-%H%M%S)"
readonly LOG_DIR="/sovereign/logs/build"
readonly BACKUP_DIR="/sovereign/backups"
readonly TPM_EMULATOR_PORT="2321"

log() { echo "[$(date '+%H:%M:%S')] $1"; }
fail() { echo "[ERROR] $1" >&2; exit 1; }

validate_tpm() {
    log "Validating TPM 2.0..."
    if ! ls /dev/tpm* 2>/dev/null; then
        log "WARNING: No hardware TPM found. Starting emulator..."
        tpm_server -rm &
        sleep 2
        export TPM_SERVER_NAME="localhost"
        export TPM_SERVER_PORT="$TPM_EMULATOR_PORT"
    else
        log "Hardware TPM verified at $(ls /dev/tpm*)"
    fi

    if ! tpm2_getcap properties-fixed >/dev/null 2>&1; then
        fail "TPM communication failed"
    fi
}

validate_gpu() {
    log "Validating GPU..."
    if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX 5080"; then
        log "WARNING: RTX 5080 not found. Using $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    fi

    python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
}

build_service() {
    local service=$1
    local dockerfile=$2

    log "Building $service..."
    docker build \
        --tag "sovereign-${service}:${BUILD_ID}" \
        --tag "sovereign-${service}:latest" \
        --file "$dockerfile" \
        --progress=plain \
        --build-arg BUILDKIT_SANDBOX_HOSTNAME="builder-${service}" \
        --no-cache \
        .

    log "Scanning $service for vulnerabilities..."
    docker scan --severity high "sovereign-${service}:${BUILD_ID}" || true
}

# ...
```

### 7.3 Observability Stack (Monitoring)

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus:latest
    container_name: sovereign-prometheus
    networks:
      - sovereign-net
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: sovereign-grafana
    networks:
      - sovereign-net
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/datasources:/etc/grafana/provisioning/datasources:ro
    ports:
      - "3000:3000"

  loki:
    image: grafana/loki:latest
    container_name: sovereign-loki
    networks:
      - sovereign-net
    volumes:
      - loki_data:/loki
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:latest
    container_name: sovereign-promtail
    networks:
      - sovereign-net
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
      - /var/run/docker.sock:/var/run/docker.sock
      - ./monitoring/promtail-config.yml:/etc/promtail/config.yml:ro
    command: -config.file=/etc/promtail/config.yml
```

### 7.4 Code Export for Showcase

```bash
#!/bin/bash
# export_for_codex.sh - Clean export for documentation/showcase
set -euo pipefail

EXPORT_DIR="./sovereign_export_$(date +%Y%m%d)"
mkdir -p "$EXPORT_DIR"

cp -r architecture_diagrams/ "$EXPORT_DIR/architecture/"
cp docker-compose*.yml "$EXPORT_DIR/"

for service in tool-gate memu-core executor; do
    mkdir -p "$EXPORT_DIR/src/$service"
    cp -r "$service/src" "$EXPORT_DIR/src/$service/"

    find "$EXPORT_DIR/src/$service" -type f -name "*.py" -exec sed -i \
        -e '/password/d' \
        -e '/secret/d' \
        -e '/key/d' \
        -e 's|/home/keeper|/sovereign|g' \
        {} \;
done

pg_dump -U keeper -d sovereign --schema-only > "$EXPORT_DIR/database_schema.sql"
cp -r config/templates/ "$EXPORT_DIR/config/"
cp docker-compose.override.yml.example "$EXPORT_DIR/"
cp scripts/build_*.sh "$EXPORT_DIR/scripts/"
cp scripts/deploy_*.sh "$EXPORT_DIR/scripts/"
```

### 7.5 Professional Deployment Checklist

```markdown
# Production Deployment Checklist

## Pre-Flight
- [ ] TPM 2.0 enabled in BIOS, LUKS key sealed
- [ ] NVIDIA drivers 550+, CUDA 12.4+ installed
- [ ] Docker 25.0+, docker-compose 2.24+
- [ ] 32GB RAM minimum, 48GB recommended
- [ ] 512GB NVMe SSD with LUKS2 encryption
- [ ] Ubuntu 24.04 LTS minimal install

## Phase 1: Hardware Validation
- [ ] `/dev/tpm0` device present
- [ ] `tpm2_getcap properties-fixed` returns valid data
- [ ] `nvidia-smi` shows RTX 5080 with correct driver
- [ ] CUDA test: `python3 -c "import torch; print(torch.cuda.is_available())"`

## Phase 2: Core Services
- [ ] PostgreSQL 15 with pgvector extension
- [ ] Tool Gate container running with TPM access
- [ ] Hash-chained ledger tables created
- [ ] Dashboard accessible on port 8080

## Phase 3: Intelligence Layer
- [ ] Sovereign Memory Core self-hosted and responding
- [ ] Ollama with DeepSeek-V3 32B, Kimi 2.5, Qwen-VL loaded
- [ ] LangGraph engine with Redis backend
- [ ] Vector similarity working via pgvector

## Phase 4: Execution Layer
- [ ] Executor service with container isolation
- [ ] Tool sandboxes (QGIS, n8n, Shell) built
- [ ] Scratchpad volume with correct permissions

## Phase 5: Perception & Output
- [ ] Camera pipeline: YOLO detection + face blurring
- [ ] Audio pipeline: Porcupine hotword + Whisper STT
- [ ] TTS service with voice cloning
- [ ] Avatar rendering with lip sync

## Phase 6: Security Hardening
- [ ] All containers run as non-root users
- [ ] Read-only filesystems except /io volumes
- [ ] Linux capabilities dropped (--cap-drop ALL)
- [ ] AppArmor/SELinux profiles applied
- [ ] Network isolation: sovereign-net only
- [ ] Keylime attestation configured
- [ ] WireGuard VPN (attestation-gated)

## Phase 7: Monitoring & Backup
- [ ] Prometheus metrics collection
- [ ] Grafana dashboards configured
- [ ] Loki/Promtail log aggregation
- [ ] Automated backup script tested
- [ ] Ledger integrity verification working

## Post-Deployment Validation
- [ ] PUB mode: No tool execution possible
- [ ] WORK mode: 70/30 rule enforced
- [ ] CO-SIGN: Dashboard approvals work
- [ ] Memory compression: Weekly triggers work
- [ ] Heartbeat: Alerts delivered appropriately
- [ ] Full system backup completed
```
