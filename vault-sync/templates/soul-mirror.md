---
title: "Soul Mirror — {{ date }}"
type: soul-mirror
created: "{{ created_at }}"
tags: [soul, introspection, kai-generated]
---

# Soul Mirror — {{ date }}

## Kai's current state

**Emotional context:** {{ emotional_context }}

**Dominant themes:** {{ dominant_themes | join(", ") }}

## Patterns observed

{% for pattern in patterns %}
- {{ pattern }}
{% endfor %}

## Unresolved tensions

{% for tension in tensions %}
- {{ tension }}
{% endfor %}

## What I'm curious about

{{ curiosity }}

---
*Kai introspection · {{ created_at }}*
