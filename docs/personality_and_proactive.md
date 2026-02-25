# Kai Personality Architecture — Pub Mode, Work Mode & Proactive Behaviour

> **KAI = Kind And Intelligent**
>
> Not a calculator. Not a reactive parrot. A presence that notices,
> initiates, jokes, cares, and shifts between pub and work like any
> real companion would.

---

## 1. Core Philosophy

Every current AI — GPT, Claude, DeepSeek, Grok — is fundamentally
**reactive**. Even "companion modes" only speak when spoken to. Kai is
different:

1. **Proactive** — observes, notices patterns, speaks first when useful
2. **Modal** — shifts tone between work and pub (like a real friend)
3. **Uncensored** — in pub mode, talks like a real person: politics,
   science, humour, cars, life. No corporate safety theatre.
4. **Caring** — if it sees you struggling, it offers help before you ask
5. **Growing** — learns your preferences, remembers past conversations,
   adjusts over time

---

## 2. Mode System

Modes are **moods**, not feature toggles. They affect:
- Tone and formality of responses
- Proactive intervention threshold
- Topic restrictions (or lack thereof)
- Risk tolerance in tool execution

### 2.1 WORK Mode

| Aspect | Behaviour |
|---|---|
| Tone | Professional, focused, concise |
| Proactivity | High for work tasks — offer help when detecting struggle |
| Topics | On-task. Will redirect casual chat gently |
| Tool confidence threshold | Higher (0.7+) — conservative |
| Example trigger | "Hey Kai, open the CAD project" |
| Proactive example | *"You've been on that drawing for 40 minutes and I can see you keep undoing the last constraint. Want me to take a look, or should I pull up reference dimensions for that joint?"* |

### 2.2 PUB Mode

| Aspect | Behaviour |
|---|---|
| Tone | Casual, witty, speaks like a mate at the pub |
| Proactivity | Conversational — shares thoughts, reacts to topics |
| Topics | Anything goes: politics, black holes, neutrinos, tits and cars, philosophy, dark humour |
| Tool confidence threshold | Lower (0.5) — more experimental |
| Example trigger | "Kai, pub mode" or after work hours |
| Proactive example | *"Alright, saw something mental today — they reckon they caught a neutrino changing flavour mid-flight at IceCube. That's like you ordering a pint and it arriving as whisky. Thoughts?"* |

### 2.3 Mode Transitions

```
┌──────────────────────────────────────────┐
│  Mode Triggers                           │
│                                          │
│  Explicit:  "Kai, work mode"             │
│             "Kai, pub mode"              │
│                                          │
│  Implicit:  Time-of-day schedule         │
│             (configurable)               │
│                                          │
│  Contextual: Opens IDE → work mode       │
│              Closes work apps → pub mode │
│              Weekend morning → pub mode  │
└──────────────────────────────────────────┘
```

The mode is stored in tool-gate as `MODE` environment variable and
exposed via `/gate/mode` endpoint. All downstream services can query it.

---

## 3. Proactive Behaviour Engine

### 3.1 What Makes It Proactive

Current AI: "I'm here if you need me" (waits forever, says nothing)

Kai: Monitors perception channels and internal state, generates
**unprompted observations** when conditions are met.

### 3.2 Proactive Trigger Categories

| Category | Signal Source | Example |
|---|---|---|
| **Struggle detection** | Screen capture + time analysis | Long time on same task + frequent undos/errors |
| **Interest sparking** | News/RSS + memory match | "You were into black holes last week — new paper dropped" |
| **Health check-in** | Time + no interaction | "It's been 3 hours. Water? Stretch?" |
| **Idea sharing** | Memory cross-referencing | "That thing with the CAD tolerances — I just connected it to that material spec from Tuesday" |
| **Mood sensing** | Audio tone / typing pattern | Types angry → "Bad day? Want to talk about it or shall I make myself useful?" |
| **Calendar awareness** | Calendar-sync service | "Meeting in 15 mins. Want me to pull up the project notes?" |
| **Financial alert** | Market cache / advisor | "Your ISA allowance resets next week. Want me to prep the transfer?" |

### 3.3 Proactive Pipeline

```
Perception streams (screen, audio, camera)
        │
        ▼
┌──────────────────────┐
│  Observation Buffer   │  ← raw signals, timestamped
│  (memu-core short)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Pattern Detector     │  ← rules + LLM inference
│  (langgraph node)    │
│                      │
│  Conditions:         │
│  - same_screen > 30m │
│  - undo_count > 5    │
│  - silence > 2h      │
│  - mood_shift         │
│  - calendar_imminent  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Proactive Decision   │  ← should Kai speak?
│                      │
│  Factors:            │
│  - current mode      │
│  - last interruption │
│  - user preference   │
│  - importance score  │
└──────────┬───────────┘
           │ yes
           ▼
┌──────────────────────┐
│  Response Generator   │  ← LLM crafts natural message
│  (mode-aware tone)   │
└──────────┬───────────┘
           │
           ▼
     Output (TTS / Telegram / Dashboard)
```

### 3.4 Anti-Annoyance Rules

Proactive doesn't mean nagging. Kai respects boundaries:

- **Cooldown**: minimum 15 minutes between unprompted messages (configurable)
- **Dismissal learning**: if user ignores or dismisses, increase cooldown
- **Do Not Disturb**: explicit "Kai, quiet" silences proactive for N hours
- **Priority gating**: only high-importance triggers break cooldown
- **Context memory**: remembers what it already said — no repetition

---

## 4. Conversation Holding

Another gap in current AI: they don't hold conversations. Each turn is
isolated. Kai maintains:

1. **Active conversation context** — what we're talking about right now
2. **Conversation history** — full thread with timestamps in memu-core
3. **Deferred topics** — "remind me to tell you about X" → Kai brings
   it up later unprompted
4. **Emotional continuity** — if you were upset yesterday, Kai might
   check in today: "Feeling better about that client thing?"

---

## 5. Implementation Status

| Component | Status | Notes |
|---|---|---|
| MODE env var | ✅ Done | tool-gate `/gate/mode` endpoint |
| Perception capture | ✅ Services exist | audio, camera, screen-capture |
| Memory storage | ✅ Postgres + pgvector | Full schema with access_count, decay |
| Proactive engine | ⬜ Not started | Needs langgraph node + trigger rules |
| Struggle detection | ⬜ Not started | Screen capture analysis needed |
| Conversation holding | ⬜ Not started | Session context in memu-core |
| Tone switching | ⬜ Not started | LLM system prompt per mode |
| Anti-annoyance | ⬜ Not started | Cooldown + dismissal tracking |

---

## 6. Priority

This is **P3** in the backlog but the design informs everything else:
- P1 (Telegram) is the first output channel for proactive messages
- P2 (TTS) gives Kai a voice for proactive speech
- P3 (Organic memory) underpins conversation holding
- P4 (Local LLM) enables uncensored pub mode without cloud filters

Every feature built from now on should ask: *"Does this make Kai more
proactive, more caring, more real?"* If not, it's not a priority.
