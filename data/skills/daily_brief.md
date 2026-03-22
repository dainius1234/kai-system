# Skill: Daily Brief

## Trigger patterns
- "morning brief"
- "daily brief"
- "start the day"
- "what's on today"

## Action
Compile a morning briefing from: goals due today, recent nudges, overnight memories, and calendar events.
Endpoint: GET /memory/proactive + GET /memory/goals

## Response template
**Good morning! Here's your daily brief:**

🎯 **Goals due today:** {goals}
📌 **Nudges:** {nudges}
🧠 **Overnight insights:** {insights}
