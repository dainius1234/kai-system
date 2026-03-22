# Kai's Agents Registry

> Defines Kai's capabilities — what Kai can do and how.
> Edit to add new skills or retire old ones.
> Loaded on startup by langgraph/app.py.

## Active Agents

| Agent | Domain | Trigger | Endpoint | Status |
|-------|--------|---------|----------|--------|
| Memory Recall | Retrieve past conversations | "remember", "what did we" | memu-core /memory/retrieve | ✅ Active |
| Tax Advisory | UK self-employment tax | "tax", "HMRC", "expense" | langgraph /chat (specialist route) | ✅ Active |
| Goal Tracker | Ohana goals & reminders | "goal", "remind me", "track" | memu-core /memory/goals | ✅ Active |
| Sentiment Analysis | Mood & emotion tracking | automatic on every message | memu-core /memory/emotion | ✅ Active |
| Fact Checker | Verify claims before memory | automatic on memorize | verifier /verify | ✅ Active |
| Risk Assessor | Evaluate action safety | before execution | orchestrator /assess | ✅ Active |
| Proactive Nudge | Time-based & context nudges | background timer | memu-core /memory/proactive | ✅ Active |
| Voice Emotion | Audio tone analysis | on mic capture | perception-audio /analyse/emotion | ✅ Active |
| Frame Analysis | Visual environment check | on camera capture | perception-camera /capture/screen | ✅ Active |
| Wake-word Judge | "Kai" detection + intent | on transcript | perception-audio /wake-word/detect | ✅ Active |
| PII Redactor | Strip personal data | before memorize | verifier /redact | ✅ Active |
| Dream Cycle | Overnight memory consolidation | scheduled | langgraph /dream | ✅ Active |
| Invoice Generator | PDF invoices & CSV export | "invoice", "receipt" | memu-core /memory/invoice | ✅ Active |

## Retired Agents

_None yet._

## Agent Design Rules

1. Every agent must have a clear trigger pattern.
2. Every agent must route through an existing service endpoint.
3. New agents should be tested with qwen2:0.5b first.
4. Agents should degrade gracefully — stub mode if backend is offline.
5. Register new agents in the router: `langgraph/router.py`.
