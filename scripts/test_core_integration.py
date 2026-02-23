from __future__ import annotations

import time
import requests


def safe_get(url: str):
    try:
        r = requests.get(url, timeout=2)
        return r
    except Exception as e:
        print(f"GET {url} failed: {e}")
        return None


def safe_post(url: str, json=None):
    try:
        r = requests.post(url, json=json, timeout=2)
        return r
    except Exception as e:
        print(f"POST {url} failed: {e}")
        return None


def main():
    services = {
        "tool-gate": "http://localhost:8000/health",
        "memu-core": "http://localhost:8001/health",
        "heartbeat": "http://localhost:8010/health",
        "dashboard": "http://localhost:8080/health",
        "executor": "http://localhost:8002/health",
        "langgraph": "http://localhost:8007/health",
        "audio": "http://localhost:8021/health",
        "camera": "http://localhost:8040/health",
        "grok": "http://localhost:8090/health",
        "tts": "http://localhost:8030/health",
        "avatar": "http://localhost:8081/health",
    }
    for name, url in services.items():
        r = safe_get(url)
        if not r or r.status_code != 200:
            print(f"service {name} not reachable")
            continue
        print(f"{name} healthy")

    # tick heartbeat
    safe_post("http://localhost:8010/tick")

    # memorize an event
    memu_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "event_type": "test",
        "result_raw": "integration",
        "user_id": "keeper",
    }
    r = safe_post("http://localhost:8001/memory/memorize", json=memu_payload)
    print("memu response", r.status_code if r else None, r.text if r else "")

    # make a tool-gate request with low confidence
    tg_payload = {
        "tool": "noop",
        "actor_did": "tester",
        "session_id": "bootstrap-token",
        "confidence": 0.1,
    }
    r = safe_post("http://localhost:8000/gate/request", json=tg_payload)
    print("tool-gate response", r.status_code if r else None, r.text if r else "")

    # check dashboard go-no-go
    r = safe_get("http://localhost:8080/go-no-go")
    print("dashboard go-no-go", r.status_code if r else None, r.text if r else "")
    # fetch UI stub
    r2 = safe_get("http://localhost:8080/ui")
    print("dashboard UI", r2.status_code if r2 else None)
    # try a simple executor run if available
    r3 = safe_post("http://localhost:8002/execute", json={"tool":"noop","params":{},"task_id":"t1","device":"cpu"})
    if r3:
        print("executor execute", r3.status_code, r3.text)

    # test perception endpoints if reachable
    r4 = safe_post("http://localhost:8021/listen", json={"text":"hello","session_id":"int"})
    if r4:
        print("audio listen", r4.status_code, r4.text)
    r5 = safe_post("http://localhost:8040/process")
    if r5:
        print("camera process", r5.status_code, r5.text)
    # optionally ask grok
    r6 = safe_post("http://localhost:8090/ask", json={"question": "hello"})
    if r6:
        print("grok ask", r6.status_code, r6.text)
    # output services
    r7 = safe_post("http://localhost:8030/synthesize", json={"text": "hello"})
    if r7:
        print("tts synthesize", r7.status_code, r7.text)
    r8 = safe_post("http://localhost:8081/speak", json={"text": "hello"})
    if r8:
        print("avatar speak", r8.status_code, r8.text)

    return 0


if __name__ == "__main__":
    exit(main())
