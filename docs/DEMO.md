# Kai 5-Minute Demo

> Show a friend what Sovereign AI looks like. No GPU required — runs on CPU.

---

## Prerequisites (2 min)

1. Docker Desktop running
2. Repo cloned: `git clone https://github.com/dainius1234/kai-system && cd kai-system`
3. Copy env: `cp .env.example .env`

---

## Start the stack (1 min)

```bash
docker compose -f docker-compose.minimal.yml up -d --build
```

Wait for healthy status:
```bash
docker compose -f docker-compose.minimal.yml ps
```

All services should show `Up` or `(healthy)`.

---

## The demo (2 min)

### 1. Open the dashboard
Navigate to **http://localhost:8080** in your browser.

You'll see the Sovereign AI control panel — real-time service health, memory stats, and system status.

### 2. Have a conversation
Click **Chat** (or navigate to http://localhost:8080/chat).

Try:
- *"What can you help me with?"*
- *"Remember that I prefer morning meetings"* — watch it store to memory
- *"What do you know about me?"* — retrieves from memory

### 3. Check CIS finance (if you're a UK subcontractor)
```bash
# Log a CIS payment
curl -s -X POST http://localhost:8063/finance/cis/record \
  -H 'Content-Type: application/json' \
  -d '{"contractor_name":"Acme Build Ltd","gross_amount":2000,"deduction_status":"registered"}' \
  | python3 -m json.tool

# Get your YTD summary
curl -s http://localhost:8063/finance/summary | python3 -m json.tool
```

### 4. Install on your phone (PWA)
- Open **http://YOUR-LOCAL-IP:8080** on your phone
- Tap the browser menu → **Add to Home Screen**
- Kai installs as a native-feeling app

---

## What's running

| Service | Port | Purpose |
|---------|------|---------|
| Sovereign Dashboard | 8080 | Web UI + chat |
| Kai (agentic) | 8000 | LLM + conviction loop |
| Memory (memu-core) | 8001 | Vector memory store |
| Finance | 8063 | CIS / VAT / tax tracker |
| Tool Gate | 8007 | Action security gate |

---

## Tear down

```bash
docker compose -f docker-compose.minimal.yml down
```
